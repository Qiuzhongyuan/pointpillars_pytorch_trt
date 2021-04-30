#ifndef TRT_NMS_H
#define TRT_NMS_H
#include "NvInfer.h"
#include "plugin.h"
#include <sstream>
#include <cuda_fp16.h>

using namespace nvinfer1;


void copy_from_const1(const float *src, float *dst, int total_num);
void set_mask(const int *index, int* mask, int num, int shape0, int shape1, int shape2);
void half2float_nms(const __half *in, float *out, int num);
void float2half_nms(float *in, __half *out, int num);

cudaError_t cuda_nms(cudaStream_t stream, float *batch_box, float *batch_cls_rw, float *score, int *cls_index, int *range_index_rw, float *ious_rw, float *dst,
    float *score_temp, int *cls_temp, float *box_temp, int num_box, int num_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, int batch_size, float score_thresh, int use_bev);

#endif // TRT_NMS_H