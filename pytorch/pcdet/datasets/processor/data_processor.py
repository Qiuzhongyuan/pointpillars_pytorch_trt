from functools import partial
import numba
import numpy as np

from ...utils import box_utils, common_utils

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxels, coors = voxel_generator.generate(points)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coors
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


class VoxelGenerator(object):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)


    def generate(self, points):
        voxels, coors, num_points_per_voxel = points_to_voxel(points,
                                                            self.point_cloud_range,
                                                            self.grid_size,
                                                            max_points=self.max_num_points,
                                                            max_voxels=self.max_voxels)


        process(voxels, num_points_per_voxel, self.point_cloud_range)
        valid_num = len(voxels)
        coors = np.stack([np.zeros((valid_num,), dtype=np.int32), coors[:,0], coors[:,1]]).T
        return voxels, coors

@numba.jit(nopython=True)
def process(voxels, num_points, point_cloud_range):
    # voxels : (N, num_per_voxel, C+6)
    # coordinates: (N, 3) zyx
    # num_points: (N,)

    N, num_per_voxel, C = voxels.shape
    for i in range(N):
        num_point = num_points[i]
        voxels_i = voxels[i]

        for j in range(num_point, num_per_voxel):
            for k in range(C):
                voxels[i, j, k] = voxels_i[0, k]

# @numba.jit(nopython=True)
# def process(voxels, coordinates, num_points, point_cloud_range, voxel_size):
#     # voxels : (N, num_per_voxel, C+6)
#     # coordinates: (N, 3) zyx
#     # num_points: (N,)
#
#     xmin, ymin, zmin = point_cloud_range[:3]
#     N, num_per_voxel, C = voxels.shape
#     for i in range(N):
#         num_point = num_points[i]
#         voxels_i = voxels[i]
#
#         cluster_center = voxels_i[0, 0:3].copy()
#         for j in range(1, num_point):
#             cluster_center += voxels_i[j, 0:3]
#         cluster_center = cluster_center / num_point
#
#         center_x = xmin + (coordinates[i, 2] + 0.5) * voxel_size[0]
#         center_y = ymin + (coordinates[i, 1] + 0.5) * voxel_size[1]
#         center_z = zmin + (coordinates[i, 0] + 0.5) * voxel_size[2]
#         geom_center = np.array([center_x, center_y, center_z], dtype=np.float32)
#
#         for j in range(num_point):
#             voxels[i, j, 4] = (voxels_i[j, 0] - cluster_center[0]) / voxel_size[0]
#             voxels[i, j, 5] = (voxels_i[j, 1] - cluster_center[1]) / voxel_size[1]
#             voxels[i, j, 6] = (voxels_i[j, 2] - cluster_center[2]) / voxel_size[2]
#
#             voxels[i, j, 7] = (voxels_i[j, 0] - geom_center[0]) / voxel_size[0]
#             voxels[i, j, 8] = (voxels_i[j, 1] - geom_center[1]) / voxel_size[1]
#             voxels[i, j, 9] = (voxels_i[j, 2] - geom_center[2]) / voxel_size[2]
#
#         for j in range(num_point, num_per_voxel):
#             voxels[i, j] = voxels_i[0]



@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            voxelmap_shape,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=100,
                            max_voxels=20000):
    N = points.shape[0]
    ndim = 3
    grid_size = voxelmap_shape

    coor = np.zeros(shape=(2,), dtype=np.int32)
    # voxel_num = np.zeros(shape=(2,), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        cx = np.floor((points[i, 0] - coors_range[0]) / voxel_size[0])
        cy = np.floor((points[i, 1] - coors_range[1]) / voxel_size[1])
        if not (0 <= cx < grid_size[0] and 0 <= cy < grid_size[1] and coors_range[2] <= points[i, 2] <= coors_range[5]):
            continue
        coor = np.array([cy, cx], dtype=np.int32)
        voxelidx = coor_to_voxelidx[coor[1], coor[0]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num = voxel_num + 1
            coor_to_voxelidx[coor[1], coor[0]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num, 0] = (points[i, 0] - coors_range[0] - (cx + 0.5) * voxel_size[0]) / voxel_size[0]
            voxels[voxelidx, num, 1] = (points[i, 1] - coors_range[1] - (cy + 0.5) * voxel_size[1]) / voxel_size[1]
            voxels[voxelidx, num, 2] = points[i, 2] / 2.7
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def points_to_voxel(points,
                    coors_range,
                    voxelmap_shape,  # (x,y)方向格数
                    max_points,
                    max_voxels):

    if not isinstance(voxelmap_shape, np.ndarray):
        voxelmap_shape = np.array(voxelmap_shape, dtype=np.int32)
    voxelmap_shape = voxelmap_shape[:2]
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxel_size = (coors_range[3:5] - coors_range[:2]) / voxelmap_shape[:2]

    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)  # 确定每个voxel的实际点数
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, 3), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 2), dtype=np.int32)  # 确定voxel的位置

    voxel_num = _points_to_voxel_kernel(
        points, voxel_size, voxelmap_shape, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel
