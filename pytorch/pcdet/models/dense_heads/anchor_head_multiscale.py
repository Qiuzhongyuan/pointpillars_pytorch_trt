import numpy as np
import torch.nn as nn
from .anchor_head_template import AnchorHeadTemplate
from .anchor_head_single import AnchorHeadSingle
import copy
import torch

class AnchorHeadMultiScale(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training)

        del self.anchors

        anchor_cfg_2x = copy.deepcopy(model_cfg['ANCHOR_GENERATOR_CONFIG_2X'])
        del model_cfg['ANCHOR_GENERATOR_CONFIG_2X']
        self.single_head_1 = AnchorHeadSingle(model_cfg, input_channels[0], num_class, class_names, grid_size,
                                              point_cloud_range, predict_boxes_when_training, **kwargs)
        model_cfg['ANCHOR_GENERATOR_CONFIG'] = anchor_cfg_2x
        self.single_head_2 = AnchorHeadSingle(model_cfg, input_channels[1], num_class, class_names, grid_size,
                                              point_cloud_range, predict_boxes_when_training, **kwargs)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        assert len(spatial_features_2d) == 2
        feat_2d = spatial_features_2d[0]
        feat_2d_2x = spatial_features_2d[1]
        data_dict['spatial_features_2d'] = feat_2d
        out_dict = self.single_head_1(data_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds_1 = out_dict['batch_cls_preds']
            batch_box_preds_1 = out_dict['batch_box_preds']

        data_dict['spatial_features_2d'] = feat_2d_2x
        out_dict2 = self.single_head_2(data_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds_2 = out_dict2['batch_cls_preds']
            batch_box_preds_2 = out_dict2['batch_box_preds']
            batch_cls_preds_1[:, :, 1:3] -= 100
            batch_cls_preds_2[:, :, 0:1] -= 100
            out_dict['batch_cls_preds'] = torch.cat([batch_cls_preds_1, batch_cls_preds_2], 1)
            out_dict['batch_box_preds'] = torch.cat([batch_box_preds_1, batch_box_preds_2], 1)

        return out_dict

    def get_loss(self):
        rpn_loss, tb_dict = self.single_head_1.get_loss()
        rpn_loss1, tb_dict1 = self.single_head_2.get_loss()

        rpn_loss = rpn_loss + rpn_loss1
        for k, v in tb_dict.items():
            tb_dict[k] = v + tb_dict1[k]

        return rpn_loss, tb_dict