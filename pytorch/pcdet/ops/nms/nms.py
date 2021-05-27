from torch.autograd import Function
import torch.nn as nn
import torch
from .nms_cuda import nms

class NMSFunction(Function):
    @staticmethod
    def forward(ctx, batch_box_preds, batch_cls_preds, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        dim = batch_cls_preds.dim()
        assert dim == 3, 'batch_cls_preds dim should be 3.'
        num_box_info = batch_box_preds.size(2)
        if use_bev==0:
            assert num_box_info == 4
        else:
            assert num_box_info == 7
        outputs, valid = nms(batch_box_preds, batch_cls_preds, nms_pre_maxsize, nms_post_maxsize, nms_thresh, score_thresh, use_bev)

        return outputs, valid

    @staticmethod
    def symbolic(g, batch_box_preds, batch_cls_preds, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        return g.op('NMS', batch_box_preds, batch_cls_preds, nms_post_maxsize_i=nms_post_maxsize,
                    nms_pre_maxsize_i=nms_pre_maxsize, nms_thresh_f=nms_thresh, score_thresh_f=score_thresh, use_bev_i=use_bev, outputs=2)

nms_func = NMSFunction.apply

class NMS(nn.Module):
    def __init__(self, nms_post_maxsize, nms_pre_maxsize, nms_thresh, score_thresh, use_bev):
        super().__init__()
        '''
        nms_config = EasyDict()
        nms_config.nms_post_maxsize = 10   #处理后最多保留的框的个数
        nms_config.nms_pre_maxsize = 100   #处理前最多考虑的框的个数，按照置信度从大到小
        nms_config.nms_thresh = 0.85    
        nms_config.score_thresh = 0.1    # 根据置信度滤除检测结果
        nms_config.use_bev = 1  # 0/1，指定计算IOU的方式，是否采用旋转的检测框  
        '''
        self.nms_post_maxsize = nms_post_maxsize
        self.nms_pre_maxsize = nms_pre_maxsize
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.use_bev = use_bev

    def forward(self, batch_box_preds, batch_cls_preds):
        """
        batch_box_preds: (B, num_boxes, 7+C)  前七位：[x, y, z, dx, dy, dz, heading] or (B, num_boxes, 4+C) for use_bev==0
        batch_cls_preds: (B, num_boxes, num_classes | 1) #各类别的置信度
        """
        cat_output = nms_func(batch_box_preds, batch_cls_preds, self.nms_post_maxsize, self.nms_pre_maxsize, self.nms_thresh, self.score_thresh, self.use_bev)
        return cat_output
