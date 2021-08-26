import torch
import torch.nn as nn
from pcdet.ops import dense
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        voxel_valid = batch_dict['voxel_valid']

        features = dense.dense(pillar_features, coords, voxel_valid, [self.nz, self.ny, self.nx])
        batch_spatial_features = features.view(batch_size, -1, self.ny, self.nx)

        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

