import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt, pruning_eval_single_ckpt
import numpy as np
from easydict import EasyDict
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from eval_utils import eval_utils

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import sys
sys.path.append('/home/qiuzhongyuan/deeplearning')

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--pruning_dir', default='./pruning_dir', help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


args, cfg = parse_config()
# def val_ckpt(model, args, test_loader, epoch_id, dist_train, logger, output_dir):
def val_ckpt(model, dataloader, device):
    model = model.cuda()
    model.eval()
    res_dict, res_str = eval_utils.pruning_eval_one_epoch(cfg, model, dataloader, os.path.join(args.pruning_dir, 'val_dir'))

    results = res_str.split('\n')
    Car_bev_mod = float(results[2].split(", ")[1])
    Ped_bev_mod = float(results[22].split(", ")[1])
    Cyc_bev_mod = float(results[42].split(", ")[1])
    mAP_bev = (Car_bev_mod + Ped_bev_mod + Cyc_bev_mod)/300.0

    Car_3d_mod = float(results[3].split(", ")[1])
    Ped_3d_mod = float(results[23].split(", ")[1])
    Cyc_3d_mod = float(results[43].split(", ")[1])
    mAP_3d = (Car_3d_mod + Ped_3d_mod + Cyc_3d_mod)/300.0

    Car_aos_mod = float(results[3].split(", ")[1])
    Ped_aos_mod = float(results[23].split(", ")[1])
    Cyc_aos_mod = float(results[43].split(", ")[1])
    mAP_aos = (Car_aos_mod + Ped_aos_mod + Cyc_aos_mod)/300.0

    print("mAP: ",mAP_bev, "Car_bev_mod AP: ",Car_bev_mod, "Ped_bev_mod AP: ",Ped_bev_mod, "Cyc_bev_mod AP: ",Cyc_bev_mod)
    print("mAP: ",mAP_3d, "Car_3d_mod AP: ",Car_3d_mod, "Ped_3d_mod AP: ",Ped_3d_mod, "Cyc_3d_mod AP: ",Cyc_3d_mod)
    print("mAP: ",mAP_aos, "Car_aos_mod AP: ",Car_aos_mod, "Ped_aos_mod AP: ",Ped_aos_mod, "Cyc_aos_mod AP: ",Cyc_aos_mod)
    return mAP_bev


class PFNLayer(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.linear = model.linear
        self.norm = model.norm

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        x = nn.functional.relu(x)
        return x

class OnnxModelPart1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pfn_layers = nn.ModuleList()
        self.pfn_layers.append(PFNLayer(model.vfe.pfn_layers[0]))


    def forward(self, input):
        features = input
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features

class AnchorHeadSingle(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv_cls = model.conv_cls
        self.conv_box = model.conv_box
        self.conv_dir_cls = model.conv_dir_cls

    def forward(self, spatial_features_2d):

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)

        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1)
        cls_preds = cls_preds.permute(0, 2, 3, 1)  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1)  # [N, H, W, C]
        return cls_preds, box_preds, dir_cls_preds

class DenseHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.single_head_1 = AnchorHeadSingle(model.dense_head.single_head_1)
        self.single_head_2 = AnchorHeadSingle(model.dense_head.single_head_2)

    def forward(self, feat_1x, feat_2x):
        batch_cls_preds_1x, batch_box_preds_1x, dir_cls_preds_1x = self.single_head_1(feat_1x)
        batch_cls_preds_2x, batch_box_preds_2x, dir_cls_preds_2x = self.single_head_2(feat_2x)
        return batch_cls_preds_1x, batch_box_preds_1x, dir_cls_preds_1x, batch_cls_preds_2x, batch_box_preds_2x, dir_cls_preds_2x


class PointPillars(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vfe = OnnxModelPart1(model)
        self.backbone_2d = model.backbone_2d
        self.dense_head = DenseHead(model)

    def forward(self, features):
        assert features.dim()==4
        features = self.vfe(features)
        out = self.backbone_2d(features)
        batch_cls_preds_1x, batch_box_preds_1x, dir_cls_preds_1x, batch_cls_preds_2x, batch_box_preds_2x, dir_cls_preds_2x = self.dense_head(out[0], out[1])

        return batch_cls_preds_1x, batch_box_preds_1x, dir_cls_preds_1x, batch_cls_preds_2x, batch_box_preds_2x, dir_cls_preds_2x


def main():
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    # train_set, train_loader, train_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers,
    #     logger=logger,
    #     training=True,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    #     total_epochs=args.epochs
    # )

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )

    assert args.pretrained_model is not None
    args.pruning_dir = Path(args.pruning_dir)
    loc_type = torch.device('cuda:0')
    weight = torch.load(args.pretrained_model, map_location=loc_type)
    if not isinstance(weight, nn.Module):
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        # model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)
        model.load_state_dict(weight)
        print('use pretrained model.')
    else:
        model = weight
    # torch.save(model.cpu().state_dict(), 'check.pth', _use_new_zipfile_serialization=False)

    outdir = os.path.join(args.pruning_dir, 'sens_analysis')
    epoch_id = 10
    csv_file_path = os.path.join(args.pruning_dir, 'sens.csv')
    if os.path.exists(csv_file_path):
        from torch_sdk.utils.sensitivity_analysis import load_csv
        sensitivity = load_csv(csv_file_path)
    else:
        from torch_sdk.utils.sensitivity_analysis import SensitivityAnalysis
        s_analyzer = SensitivityAnalysis(model=model, val_func=val_ckpt, val_loader=test_loader, prune_type='l2')
        for k, v in s_analyzer.target_layer.items():
            print(k)

        sensitivity = s_analyzer.analysis()
        os.makedirs(outdir, exist_ok=True)
        s_analyzer.export(csv_file_path)

    # STEP.2.2 Compress
    from torch_sdk.pruning.one_shot import L1FilterPruner, L2FilterPruner
    from torch_sdk.pruning.apply_compression import compress
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    ### Not pruning layers
    config_list = [{
        'sparsity': 0.0,
        'op_types': ['Conv2d'],
        # 'op_names': []
        'op_names': ['dense_head.single_head_1.conv_cls', 'dense_head.single_head_1.conv_box', 'dense_head.single_head_1.conv_dir_cls',
                        'dense_head.single_head_2.conv_cls', 'dense_head.single_head_2.conv_box', 'dense_head.single_head_2.conv_dir_cls']
    }]
    checkpoints_dir = os.path.join(args.pruning_dir, 'pruned_ckpt')
    os.makedirs(checkpoints_dir, exist_ok=True)
    pruner_name = 'bev_1'
    precision = 0.7375
    pre_loss = 0.350

    dummy_input = [torch.randn([1, 10, 496, 432]).to("cpu")]
    PruningModel = PointPillars(model)
    PruningModel = PruningModel.to("cpu")
    """
        Change bev_backbone forward input/output type from dict to tensor
    """

    trace_model = PruningModel#.backbone_2d
    # deblocks = trace_model.deblocks
    # blocks = trace_model.blocks
    # del trace_model.deblocks
    # del trace_model.blocks
    # setattr(trace_model, 'vfe_pfn_layers_0', nn.Sequential(PruningModel.vfe.pfn_layers[0].linear,
    #                                                         PruningModel.vfe.pfn_layers[0].norm,
    #                                                         nn.ReLU()))
    #
    # setattr(trace_model, 'blocks', blocks)
    # setattr(trace_model, 'deblocks', deblocks)
    #
    # setattr(trace_model, 'dense_head_single_head_1_conv_cls', PruningModel.dense_head.single_head_1.conv_cls)
    # setattr(trace_model, 'dense_head_single_head_1_conv_box', PruningModel.dense_head.single_head_1.conv_box)
    # setattr(trace_model, 'dense_head_single_head_1_conv_dir_cls', PruningModel.dense_head.single_head_1.conv_dir_cls)
    # setattr(trace_model, 'dense_head_single_head_2_conv_cls', PruningModel.dense_head.single_head_2.conv_cls)
    # setattr(trace_model, 'dense_head_single_head_2_conv_box', PruningModel.dense_head.single_head_2.conv_box)
    # setattr(trace_model, 'dense_head_single_head_2_conv_dir_cls', PruningModel.dense_head.single_head_2.conv_dir_cls)

    print(trace_model)
    with torch.onnx.set_training(trace_model, False):
    # with torch.onnx.select_model_mode_for_export(trace_model, False):
        torch._C._jit_set_inline_everything_mode(True)
        trace = torch.jit.trace(trace_model, dummy_input)
        torch._C._jit_pass_inline(trace.graph)
    pointpillars_pruned, fixed_mask = compress(trace_model, dummy_input, L2FilterPruner,
                                 config_list, ori_metric=precision, metric_thres=pre_loss, sensitivity=sensitivity, trace=trace)   # BEV
                                #  config_list, ori_metric=0.6641, metric_thres=0.15, sensitivity=sensitivity, trace=trace)     # 3D
    pruned_model_path = os.path.join(checkpoints_dir,
                                     'pruned_{}_{}_{}.pth'.format('pointpillar', 'kitti', pruner_name))
    mask_path = os.path.join(checkpoints_dir,
                             'mask_{}_{}_{}.pth'.format('pointpillar', 'kitti', pruner_name))
    

    model.backbone_2d = pointpillars_pruned.backbone_2d
    model.module_list[-2] = model.backbone_2d
    model.dense_head.single_head_1.conv_cls = pointpillars_pruned.dense_head.single_head_1.conv_cls
    model.dense_head.single_head_1.conv_box = pointpillars_pruned.dense_head.single_head_1.conv_box
    model.dense_head.single_head_1.conv_dir_cls = pointpillars_pruned.dense_head.single_head_1.conv_dir_cls
    model.dense_head.single_head_2.conv_cls = pointpillars_pruned.dense_head.single_head_2.conv_cls
    model.dense_head.single_head_2.conv_box = pointpillars_pruned.dense_head.single_head_2.conv_box
    model.dense_head.single_head_2.conv_dir_cls = pointpillars_pruned.dense_head.single_head_2.conv_dir_cls
    model.module_list[-1] = model.dense_head
    model.vfe.pfn_layers[0].linear = pointpillars_pruned.vfe.pfn_layers[0].linear
    model.vfe.pfn_layers[0].norm = pointpillars_pruned.vfe.pfn_layers[0].norm
    model.module_list[0] = model.vfe

    from thop import profile
    macs, params = profile(pointpillars_pruned, inputs=dummy_input, verbose=False)
    print("MACs and Params after compression: MACs: {} G, Params: {} M".format(macs / 1000000000, params / 1000000))

    torch.save(fixed_mask, mask_path)
    torch.save(model, pruned_model_path)
    print('precision_loss is {}'.format(pre_loss))
    print('save {} ok.'.format(pruned_model_path))   

    



if __name__ == '__main__':
    main()
