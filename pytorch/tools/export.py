import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.ops import voxels, dense, nms

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--pruned_model', type=str, default=None, help='checkpoint for pruned model')
    parser.add_argument('--pretrained_model', type=str, default=None, help='checkpoint for origin model')
    parser.add_argument('--eval_onnx_model', action='store_true', help='eval the onnx model with kitti')
    parser.add_argument('--pcs_for_export', type=str, default='000000.bin', help='point cloud data for export onnx model.')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

class OnnxModelPart1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pfn_layers = model.vfe.pfn_layers
        setattr(self.pfn_layers[0], 'act', nn.ReLU())
    def forward(self, input):
        features = input.unsqueeze(0).permute(0,3,1,2)  # (1, 10, n_voxels, n_points)
        for pfn in self.pfn_layers:
            features = pfn(features)

        features = features.permute(0, 2, 1, 3).squeeze(3).squeeze(0)

        return features


class AnchorHeadSingle(nn.Module):
    def __init__(self, model, cls_mask):
        super().__init__()
        self.conv_cls = model.conv_cls
        self.conv_box = model.conv_box
        self.conv_dir_cls = model.conv_dir_cls
        self.anchors = model.anchors
        self.box_coder = model.box_coder

        self.cls_mask = cls_mask # torch.Tensor(cls_mask).long().to(self.conv_cls.weight.device)
        self.conv_cls.weight.data = self.slice_weight(cls_mask, self.conv_cls.weight.data, cls_head=True)
        self.conv_box.weight.data = self.slice_weight(cls_mask, self.conv_box.weight.data)
        self.conv_dir_cls.weight.data = self.slice_weight(cls_mask, self.conv_dir_cls.weight.data, dir_head=True)

        self.conv_cls.bias.data = self.slice_bias(cls_mask, self.conv_cls.bias.data, cls_head=True)
        self.conv_box.bias.data = self.slice_bias(cls_mask, self.conv_box.bias.data)
        self.conv_dir_cls.bias.data = self.slice_bias(cls_mask, self.conv_dir_cls.bias.data, dir_head=True)

    def slice_bias(self, cls_index, data, cls_head=False, dir_head=False):
        num_rot = 2
        num_cls = 3
        num_cols = 2 if dir_head else 7
        if cls_head:
            data = data.view(num_cls, num_rot, num_cls)
            data = [data[idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=0)

            data = [data[:, :, idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=2).view(-1, )

        else:
            data = data.view(num_cls, num_rot, num_cols).contiguous()
            data = [data[idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=0).view(-1, )
        return data


    def slice_weight(self, cls_index, data, cls_head=False, dir_head=False):
        num_cls = 3
        num_rot = 2
        num_cols = 2 if dir_head else 7
        out_c, in_c, k1, k2 = data.size()
        if cls_head:
            data = data.view(num_cls, num_rot, num_cls, in_c, k1, k2).contiguous()
            data = [data[idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=0)

            data = [data[:, :, idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=2).view(-1, in_c, k1, k2)

        else:
            data = data.view(num_cls, num_rot, num_cols, in_c, k1, k2).contiguous()
            data = [data[idx:idx + 1] for idx in cls_index]
            data = torch.cat(data, dim=0).view(-1, in_c, k1, k2)
        return data

    def slice_tensor(self, data, cls_index):
        # size: torch.Size([1, 248, 216, 3, 2, 7])
        num_info = data.size(-1)
        num_rot = 2
        num_cls = 3
        data = [data[:,:,:,idx:idx+1] for idx in cls_index]
        data = torch.cat(data, dim=3)
        return data

    def forward(self, spatial_features_2d):
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)

        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        cls_preds, box_preds = self.generate_predicted_boxes(1, cls_preds, box_preds, dir_cls_preds)

        return cls_preds, box_preds

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds):
        assert batch_size==1
        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = self.slice_tensor(anchors, self.cls_mask)
        num_anchors = int(anchors.view(-1, anchors.shape[-1]).shape[0])
        batch_anchors = anchors.view(1, -1, int(anchors.shape[-1])).to(box_preds.device)

        num_anchors = int(box_preds.numel()/batch_size/7)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, int(cls_preds.numel()/batch_size/num_anchors))
        batch_box_preds = box_preds.view(batch_size, num_anchors, int(box_preds.numel()/batch_size/num_anchors))

        xg, yg, zg, dxg, dyg, dzg, rg = self.decode_torch(batch_box_preds, batch_anchors)

        dir_offset = 0.78539
        dir_limit_offset = 0
        NUM_DIR_BINS = 2
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, int(dir_cls_preds.numel()/batch_size/num_anchors))
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        period = (2 * np.pi / NUM_DIR_BINS)
        dir_rot = self.limit_period(rg - dir_offset, dir_limit_offset, period)

        dir_rot = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype).unsqueeze(2)

        batch_box_preds = torch.cat([xg, yg, zg, dxg, dyg, dzg, dir_rot], 2)
        return batch_cls_preds, batch_box_preds

    def limit_period(self, val, offset=0.5, period=np.pi):
        ans = val - torch.floor(val / period + offset) * period
        return ans

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        rg = rt + ra

        return xg, yg, zg, dxg, dyg, dzg, rg


class OnnxModelPart2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone_2d = model.backbone_2d
        self.dense_head_1x = AnchorHeadSingle(model.dense_head.single_head_1, [0])
        self.dense_head_2x = AnchorHeadSingle(model.dense_head.single_head_2, [1,2])

        self.nms3d = nms.NMSDeploy(nms_post_maxsize=100, nms_pre_maxsize=4000, nms_thresh=0.01, score_thresh=0.1, use_bev=1)

    def forward(self, input):
        data_dict = {}
        data_dict['batch_size'] = int(input.size(0))
        data_dict['spatial_features'] = input
        output_dict = self.backbone_2d(data_dict)
        batch_cls_preds_1x, batch_box_preds_1x = self.dense_head_1x(output_dict['spatial_features_2d'][0])
        batch_cls_preds_2x, batch_box_preds_2x = self.dense_head_2x(output_dict['spatial_features_2d'][1])
        batch_cls_preds_1x = batch_cls_preds_1x.sigmoid()
        batch_cls_preds_2x = batch_cls_preds_2x.sigmoid()
        batch_box_preds_1x = batch_box_preds_1x.permute(0, 2, 1).contiguous()
        batch_cls_preds_1x = batch_cls_preds_1x.permute(0, 2, 1).contiguous()
        batch_box_preds_2x = batch_box_preds_2x.permute(0, 2, 1).contiguous()
        batch_cls_preds_2x = batch_cls_preds_2x.permute(0, 2, 1).contiguous()

        outputs_1x = self.nms3d(batch_box_preds_1x, batch_cls_preds_1x)
        outputs_2x = self.nms3d(batch_box_preds_2x, batch_cls_preds_2x)
        outputs_1x = outputs_1x.permute(0, 2, 1).contiguous()
        outputs_2x = outputs_2x.permute(0, 2, 1).contiguous()
        return outputs_1x, outputs_2x

class OnnxModelPointPillars(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.part1 = OnnxModelPart1(model)
        self.part2 = OnnxModelPart2(model)

        self.VoxelGeneratorV1 = voxels.VoxelGenerator(voxel_size=[0.16, 0.16, 4],
                                                      point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                                                      max_num_points=64, max_voxels=20000,
                                                      type='raw', center_offset=1, cluster_offset=1,
                                                      supplement=1)
        self.h = 496
        self.w = 432
        self.batch_size = 1
    def forward(self, points, valid=None):
        if valid is None:
            valid = torch.zeros((self.batch_size, )).int().cuda() - 10
        features, coord, valid, valid_voxel = self.VoxelGeneratorV1(points, valid, self.batch_size)
        features = self.part1(features)
        features = dense.Dense(features, coord, 1, [1, self.h, self.w])
        features = features.view(1, -1, self.h, self.w)
        outputs_1x, outputs_2x = self.part2(features)

        return outputs_1x, outputs_2x

def eval_single_ckpt_onnx(model, test_loader, args, eval_output_dir, logger, tag, dist_test=False):
    eval_utils.eval_one_epoch_onnx(
        cfg, model, test_loader, tag, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file)

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    assert not dist_test and args.batch_size == 1
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = Path('deploy/eval')
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    if args.pruned_model is not None:
        tag = 'pruned_model'
        model = torch.load(args.pruned_model, map_location=torch.device('cpu'))
    elif args.pretrained_model is not None:
        tag = 'large_model'
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.pretrained_model, logger=logger, to_cpu=dist_test)

    else:
        raise RuntimeError('error: please input weights.')

    model = model.cuda()
    model.eval()

    ExportModel = OnnxModelPointPillars(model)
    ExportModel.eval()
    ExportModel = ExportModel.cuda()

    points = np.fromfile(args.pcs_for_export, dtype=np.float32).reshape(-1, 4)
    points = np.concatenate([np.zeros((len(points), 1), dtype=np.float32), points], 1)
    points = torch.from_numpy(points).float().cuda()
    points = torch.autograd.Variable(points.contiguous())
    valid = torch.Tensor([len(points)]).int().cuda()
    dummy_input = torch.zeros((25000, 5)).float().cuda() - 100
    dummy_input[:len(points)] = points

    torch.onnx.export(ExportModel, dummy_input, "pointpillars_%s.onnx" % tag, verbose=True, training=False,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,opset_version=10,
                      input_names=['points', 'valid'], output_names=['pointpillars_output1', 'pointpillars_output2'])

    if args.eval_onnx_model:
        with torch.no_grad():
            eval_single_ckpt_onnx(ExportModel, test_loader, args, eval_output_dir, logger, tag, dist_test=dist_test)

if __name__ == '__main__':
    main()
