import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfe_template import VFETemplate
import numpy as np
from pcdet.ops import voxels
import math
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False, act=None):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.act = act()
    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        x = self.act(x)
        h, w = x.shape[2:]
        x_max = F.max_pool2d(x, (1, int(w)))

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, batch_size):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM

        num_point_features = 4
        self.center_offset = self.model_cfg.get('CENTER_OFFSET', 0)
        self.cluster_offset = self.model_cfg.get('CLUSTER_OFFSET', 0)
        self.max_num_points = self.model_cfg.get('MAX_POINTS_PER_VOXEL', 32)
        self.max_voxels = self.model_cfg.get('MAX_NUMBER_OF_VOXELS', 16000)
        self.supplement = self.model_cfg.get('SUPPLEMENT', 1)

        self.act = nn.ReLU

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        num_point_features += 3 if self.center_offset != 0 else 0
        num_point_features += 3 if self.cluster_offset != 0 else 0

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0

        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2), act=self.act)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.VoxelGeneratorV1 = voxels.VoxelGenerator(voxel_size=[float(item) for item in self.voxel_size],
                                                      point_cloud_range=[float(item) for item in self.point_cloud_range],
                                                      max_num_points=self.max_num_points, max_voxels=self.max_voxels,
                                                      type='raw', center_offset=self.center_offset,
                                                      cluster_offset=self.cluster_offset, supplement=self.supplement)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        valid_num = batch_dict['valid_num'].int()

        features, coords, ValidOutput, num_points_per_voxel = self.VoxelGeneratorV1(points, valid_num)
        batch_dict['voxel_coords'] = coords # coord: (N, 3)
        batch_dict['voxel_valid'] = ValidOutput  # coord: (N,)
        features = features.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # (12000, 32, 10) -> (1, 12000, 32, 10) -> (1, 10, 12000, 32)

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze().t().contiguous()

        batch_dict['pillar_features'] = features
        return batch_dict
