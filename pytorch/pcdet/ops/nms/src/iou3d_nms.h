#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

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


#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }


std::vector<torch::Tensor>
nms(torch::Tensor batch_box, torch::Tensor batch_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, float score_thresh, int use_bev);
#endif
