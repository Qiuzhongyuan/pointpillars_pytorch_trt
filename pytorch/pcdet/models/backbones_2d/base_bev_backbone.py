import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class PlaceHolder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input

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
            new_size = (int(old_h*self.scale_factor), int(old_w*self.scale_factor))
        else:
            raise RuntimeError('error args.')
        output = F.interpolate(input, size=new_size, mode=self.mode)
        return output

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.DPU = self.model_cfg.get('DPU', False)
        if self.DPU:
            conv = Conv2d
            relu = nn.ReLU6
        else:
            conv = nn.Conv2d
            relu = nn.ReLU
            
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                PlaceHolder(),
                conv(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=1, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                relu()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    relu()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride == 1:
                    self.deblocks.append(nn.Sequential(
                        PlaceHolder(),
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        relu()
                    ))
                elif stride > 1:
                    self.deblocks.append(nn.Sequential(
                        Upsample(scale_factor=stride, mode='nearest'),
                        nn.Conv2d(
                                num_filters[idx], num_upsample_filters[idx],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                                #stride=upsample_strides[idx], bias=False
                            ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        relu()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        relu()
                    ))

        self.multi_scale = False
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.multi_scale = True
            c_in -= num_upsample_filters[0] + num_upsample_filters[-1]
            self.upscale_outchannel = num_upsample_filters[0] + num_upsample_filters[0]*2

            self.deblocks.append(nn.Sequential(
                Upsample(scale_factor=upsample_strides[-1], mode='nearest'),
                nn.Conv2d(c_in, num_upsample_filters[0]*2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_upsample_filters[0]*2, eps=1e-3, momentum=0.01),
                relu(),
            ))

        if self.multi_scale:
            self.num_bev_features = [c_in, self.upscale_outchannel]
        else:
            self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features'] if isinstance(data_dict, dict) else data_dict
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            if self.multi_scale:
                x = torch.cat(ups[1:], dim=1)
            else:
                x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x_up = self.deblocks[-1](x)
            x_up = torch.cat([x_up, ups[0]], 1)

        if isinstance(data_dict, dict):
            data_dict['spatial_features_2d'] = [x, x_up] if self.multi_scale else x
            return data_dict
        else:
            return [x, x_up] if self.multi_scale else x
