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
        batch_dict['training_grid'] = (self.nx, self.ny)

        features = dense.Dense(pillar_features, coords, batch_size, [self.nz, self.ny, self.nx])
        batch_spatial_features = features.view(batch_size, -1, self.ny, self.nx)

        # channel = pillar_features.size(1)
        # batch_spatial_features = torch.zeros(batch_size, self.ny, self.nx, channel).cuda()
        # pillar_features = pillar_features.view(batch_size, -1, channel)
        # coords = coords.view(batch_size, -1, 4)
        # for i in range(batch_size):
        #     fea = pillar_features[i]
        #     coord = coords[i]
        #     valid = coord[:, 0] >=0
        #     fea = fea[valid]
        #     coord = coord[valid].long()
        #     coord_h = coord[:, 2]
        #     coord_w = coord[:, 3]
        #     batch_spatial_features[i, coord_h, coord_w] = fea

        # batch_spatial_features = batch_spatial_features.permute(0,3,1,2).contiguous()
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

