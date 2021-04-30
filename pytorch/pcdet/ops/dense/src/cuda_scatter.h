#ifndef CUDA_SCATTER_H
#define CUDA_SCATTER_H
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

#define NAMESPACE DenseSpace

namespace NAMESPACE
{
static inline int DivUp(int a, int b) { return (a + b - 1) / b; }

inline
cudaError_t checkCudaErrors(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


void cuda_scatter(const float *features_rw, const int *indices_rw, 
                float *output_rw, std::vector<int> spatialShape_rw,
                int num_voxels, int num_features);

void cuda_scatter_fp16(const __half *features_rw, const int *indices_rw, __half *output_rw, std::vector<int> spatialShape,
                        int num_voxels, int num_features);

void cuda_scatter_backward(const float *features_rw, const int *indices_rw,  float *output_rw, std::vector<int> spatialShape,
                int num_voxels, int num_features);

}// namespace
#endif