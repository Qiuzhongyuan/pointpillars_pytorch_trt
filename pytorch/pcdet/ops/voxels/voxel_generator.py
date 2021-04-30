import torch
from .voxels_cuda import VoxelGeneratorV1
from torch.autograd import Function

class VoxelGeneratorV1Function(Function):
    @staticmethod
    def forward(ctx,points,ValidInput,voxel_size,point_cloud_range,
            max_num_points,max_voxels, batch_size,center_offset, cluster_offset, supplement):
        assert points.dim()==2 and points.size(1)>=4
        voxels, coords, ValidOutput, num_points_per_voxel = \
                                VoxelGeneratorV1(points, ValidInput, voxel_size, point_cloud_range,\
                                max_num_points, max_voxels, batch_size, center_offset, cluster_offset, supplement)

        return voxels, coords, ValidOutput, num_points_per_voxel

    @staticmethod
    def symbolic(g, points, ValidInput, voxel_size, point_cloud_range, max_num_points, max_voxels, batch_size, center_offset, cluster_offset, supplement):
        return g.op('VoxelGeneratorV1', points, ValidInput, voxel_size_f=voxel_size, \
                    point_cloud_range_f=point_cloud_range, max_num_points_i=max_num_points, \
                    max_voxels_i=max_voxels, batch_size_i=batch_size, center_offset_i=center_offset,\
                    cluster_offset_i=cluster_offset, supplement_i=supplement, outputs=4)

        
class VoxelGenerator(torch.nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, type='raw', **kwargs):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

        self.generator_type = type.lower()
        if self.generator_type == 'mean':
            # self.voxel_generator = VoxelGeneratorV2Function.apply
            assert False, 'error type: ' + self.generator_type
        elif self.generator_type == 'raw':
            self.voxel_generator = VoxelGeneratorV1Function.apply
            self.center_offset = kwargs.get('center_offset', 0)
            self.cluster_offset = kwargs.get('cluster_offset', 0)
            self.supplement = kwargs.get('supplement', 0)

        else:
            raise RuntimeError('error: please check VoxelGenerator type: %s.' % type)
    
    def forward(self, points, ValidInput, batch_size):
        if self.generator_type == 'mean':
            voxel_output = self.voxel_generator(points, ValidInput, self.voxel_size, self.point_cloud_range, self.max_num_points, self.max_voxels, batch_size)
        elif self.generator_type == 'raw':
            voxel_output = self.voxel_generator(points, ValidInput, self.voxel_size, self.point_cloud_range, self.max_num_points,
                                                self.max_voxels, batch_size, self.center_offset, self.cluster_offset, self.supplement)
        return voxel_output
