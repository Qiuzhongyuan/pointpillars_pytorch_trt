from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = padding
        self.s = stride

    def forward(self, x):
        if self.pad == 1 and self.s == 2:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        x = super().forward(x)

        return x

class Upsample(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, input):
        if self.size:
            new_size = self.size
        elif self.scale_factor:
            old_h = input.size(2)
            old_w = input.size(3)
            new_size = (int(old_h * self.scale_factor), int(old_w * self.scale_factor))
        else:
            raise RuntimeError('error args.')
        output = F.interpolate(input, size=new_size, mode=self.mode)
        return output

def conv_block_stride_1(in_c, out_c, k, s, p, b=False):
    m = [nn.Conv2d(in_c, out_c, kernel_size=k,stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01),
        nn.ReLU6()]
    return m

def conv_block_stride_2(in_c, out_c):
    m = [Conv2d(in_c, out_c, kernel_size=3,stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01),
        nn.ReLU6()]
    return m

class PointPillarPruned(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        old_module_list = self.build_networks()

        self.module_list = []
        for i, module in enumerate(old_module_list):
            name = module._get_name()
            if name == 'PillarVFE':
                setattr(self, 'vfe', module)
                self.module_list.append(getattr(self, 'vfe'))
            elif name == 'BaseBEVBackbone':
                blocks = nn.ModuleList()
                blocks.append(nn.Sequential(nn.Conv2d(32, 39, kernel_size=3,stride=1, padding=1, bias=False),
                                                    nn.BatchNorm2d(39, eps=1e-3, momentum=0.01),
                                                    nn.ReLU6()))


                cur_layers = conv_block_stride_2(39, 39)
                cur_layers += conv_block_stride_1(39, 39, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(39, 52, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(52, 32, 3, 1, 1, b=False)
                blocks.append(nn.Sequential(*cur_layers))

                cur_layers = conv_block_stride_2(32, 26)
                cur_layers += conv_block_stride_1(26, 13, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(13, 13, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(13, 39, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(39, 52, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(52, 39, 3, 1, 1, b=False)
                blocks.append(nn.Sequential(*cur_layers))

                cur_layers = conv_block_stride_2(39, 26)
                cur_layers += conv_block_stride_1(26, 52, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(52, 26, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(26, 26, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(26, 26, 3, 1, 1, b=False)
                cur_layers += conv_block_stride_1(26, 26, 3, 1, 1, b=False)
                blocks.append(nn.Sequential(*cur_layers))

                setattr(module, 'blocks', blocks)

                deblocks = nn.ModuleList()
                cur_layers = conv_block_stride_1(39, 7, 3, 1, 1, b=False)
                deblocks.append(nn.Sequential(*cur_layers))

                cur_layers = conv_block_stride_1(32, 20, 3, 1, 1, b=False)
                deblocks.append(nn.Sequential(*cur_layers))

                cur_layers = [Upsample(scale_factor=2)] + conv_block_stride_1(39, 10, 3, 1, 1, b=False)
                deblocks.append(nn.Sequential(*cur_layers))

                cur_layers = [Upsample(scale_factor=4)] + conv_block_stride_1(26, 10, 3, 1, 1, b=False)
                deblocks.append(nn.Sequential(*cur_layers))

                cur_layers = [Upsample(scale_factor=2)] + conv_block_stride_1(40, 116, 3, 1, 1, b=False)
                deblocks.append(nn.Sequential(*cur_layers))

                setattr(module, 'deblocks', deblocks)

                setattr(self, 'backbone_2d', module)
                self.module_list.append(getattr(self, 'backbone_2d'))
            elif name == 'AnchorHeadMultiScale':
                setattr(self, 'dense_head', module)
                self.module_list.append(getattr(self, 'dense_head'))
            elif name == 'PointPillarScatter':
                self.module_list.append(module)

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict