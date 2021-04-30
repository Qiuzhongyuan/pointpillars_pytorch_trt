from torch.autograd import Function
import torch.nn as nn
import torch
from .nms_cuda import nms

class NMSFunction(Function):
    @staticmethod
    def forward(ctx, batch_box_preds, batch_cls_preds, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        dim = batch_cls_preds.dim()
        assert dim == 3, 'batch_cls_preds dim should be 3.'
        num_box_info = batch_box_preds.size(1)
        if use_bev==0:
            assert num_box_info==4
        else:
            assert num_box_info==7
        outputs = nms(batch_box_preds, batch_cls_preds, nms_pre_maxsize, nms_post_maxsize, nms_thresh, score_thresh, use_bev)

        return outputs

    @staticmethod
    def symbolic(g, batch_box_preds, batch_cls_preds, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        return g.op('NMS', batch_box_preds, batch_cls_preds, nms_post_maxsize_i=nms_post_maxsize,
                    nms_pre_maxsize_i=nms_pre_maxsize, nms_thresh_f=nms_thresh, score_thresh_f=score_thresh, use_bev_i=use_bev)

nms_func = NMSFunction.apply

class NMSDeploy(nn.Module):
    def __init__(self, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        super().__init__()
        self.nms_post_maxsize = nms_post_maxsize
        self.nms_pre_maxsize = nms_pre_maxsize
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.use_bev = use_bev

    def forward(self, batch_box_preds, batch_cls_preds):
        """
        batch_cls_preds: (B, num_classes | 1, num_boxes)
        batch_box_preds: (B, 7+C, num_boxes) or (B, 4+C, num_boxes)
        B=1
        """
        cat_output = nms_func(batch_box_preds, batch_cls_preds, self.nms_post_maxsize, self.nms_pre_maxsize, self.nms_thresh, self.score_thresh, self.use_bev)
        return cat_output




if __name__ == "__main__":
    nms3d = NMSDeploy(nms_post_maxsize=100, nms_pre_maxsize=4000, nms_thresh=0.01, score_thresh=0.1, use_bev=1)

    import numpy as np

    box_preds = torch.from_numpy(np.load('batch_box_preds.npy')).float().cuda()
    cls_preds = torch.from_numpy(np.load('batch_cls_preds.npy')).float().cuda()

    batch_size = 5
    batch_box_preds = []
    batch_cls_preds = []
    for _ in range(batch_size):
        batch_box_preds.append(box_preds)
        batch_cls_preds.append(cls_preds)
    batch_box_preds = torch.cat(batch_box_preds, 0)
    batch_cls_preds = torch.cat(batch_cls_preds, 0)
    out = nms3d(batch_box_preds, batch_cls_preds)

    for idx, result in enumerate(out):
        print('*' * 50)
        print('batch index:', idx)
        print(result[:10])
        print('*' * 50)