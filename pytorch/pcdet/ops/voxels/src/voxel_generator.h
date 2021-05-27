#ifndef VOXEL_GENERATOR_H
#define VOXEL_GENERATOR_H
#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_voxel_generator.h"

#define CHECK_CUDA(x) do { \
  if (!x.device().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor>
VoxelGeneratorV1(torch::Tensor points, torch::Tensor ValidInput, std::vector<float> voxel_size,
	std::vector<float> point_cloud_range, int max_num_points, int max_voxels, int batch_size,
	int center_offset, int cluster_offset, int supplement);

#endif