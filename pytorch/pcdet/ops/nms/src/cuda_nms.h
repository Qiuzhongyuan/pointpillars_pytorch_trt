#ifndef CUDA_NMS_H
#define CUDA_NMS_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define NAMESPACE NMSSpace
#define DEBUG

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


void cuda_nms(const float *batch_box, const float *batch_cls_rw, float *score, int *cls_index, int *range_index_rw, int* pos_rw, int *cls_temp, float *box_temp,
    float *ious_rw, float *dst, int num_box, int num_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, int batch_size,
    float score_thresh, int use_bev);

}//namespace
#endif
