#ifndef CUDA_VOXEL_GENERATOR_H
#define CUDA_VOXEL_GENERATOR_H
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "gpu_hash_map.h"
#include <string>
#include <kernel.h>
#define NAMESPACE VoxelGeneratorV1Space

namespace NAMESPACE
{


void cuda_points_to_voxel(const float* inPoints, const int* devNumValidPointsInput,
                         int* outCoords, int* devValidOutputPoints, float* outVoxels, int* dCounter,
                         int* map_tensor_rw, int* addr_tensor_rw, HashEntry* list,
                         std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size,
                         int batchSize, int inPointsNum, int inCols, int outCols,
                         int cluster_offset, int center_offset, int supplement,
                         int maxValidOutput, int max_points, const int value_map_z);

void cuda_points_to_voxel_fp16(const __half* inPoints, const int* devNumValidPointsInput,
                         int* outCoords, int* devValidOutputPoints, __half* outVoxels, int* dCounter,
                         int* map_tensor_rw, int* addr_tensor_rw, HashEntry* list,
                         std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size,
                         int batchSize, int inPointsNum, int inCols, int outCols,
                         int cluster_offset, int center_offset, int supplement,
                         int maxValidOutput, int max_points, const int value_map_z);

}//namespace

#endif
