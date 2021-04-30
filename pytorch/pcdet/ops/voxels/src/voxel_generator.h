#ifndef VOXEL_GENERATOR_H
#define VOXEL_GENERATOR_H
#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

std::vector<torch::Tensor>
VoxelGeneratorV1(torch::Tensor points, torch::Tensor ValidInput, std::vector<float> voxel_size,
	std::vector<float> point_cloud_range, int max_num_points, int max_voxels, int batch_size,
	int center_offset, int cluster_offset, int supplement);

#endif