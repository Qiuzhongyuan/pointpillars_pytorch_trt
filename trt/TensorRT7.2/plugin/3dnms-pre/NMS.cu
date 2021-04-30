#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>

#include "NMS_nova.h"
#include "plugin.h"
#include <NvInfer.h>
#include <assert.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <fstream>
#define THREADS_PER_BLOCK 16
#define MAX_THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
using namespace std;

// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;

__global__ static void float2half_kernel(float *in, __half *out, int num){
    int box_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(box_index>=num)
        return;
    out[box_index] = __float2half(in[box_index]);
}

void float2half_nms(float *in, __half *out, int num){
    int block_x_mask = (num > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : num;
    int grid_x_mask = (num - 1) / block_x_mask + 1;
    float2half_kernel<<<grid_x_mask, block_x_mask>>>(in, out, num);
}


__global__ static void half2float_kernel(const __half *in, float *out, int num){
    int box_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(box_index>=num)
        return;
    out[box_index] = __half2float(in[box_index]);
}

void half2float_nms(const __half *in, float *out, int num){
    int block_x_mask = (num > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : num;
    int grid_x_mask = (num - 1) / block_x_mask + 1;
    half2float_kernel<<<grid_x_mask, block_x_mask>>>(in, out, num);
}


struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline static float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline static float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ static int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline static int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline static int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

__device__ inline static void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline static int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline static float box_overlap(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline static float iou_bev(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}


__global__ static void nms_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}


__device__ inline static float iou_normal(float const * const a, float const * const b) {
    //params: a: [x, y, dx, dy]
    //params: b: [x, y, dx, dy]

    float left = fmaxf(a[0] - a[2] / 2, b[0] - b[2] / 2), right = fminf(a[0] + a[2] / 2, b[0] + b[2] / 2);
    float top = fmaxf(a[1] - a[3] / 2, b[1] - b[3] / 2), bottom = fminf(a[1] + a[3] / 2, b[1] + b[3] / 2);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = a[3] * a[2];
    float Sb = b[3] * b[2];
    return interS / fmaxf(Sa + Sb - interS, EPS);
}


__global__ void nms_normal_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_normal(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}





__global__ static void init_zeros_kernel(float *input, float value, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=size)
        return;
    input[i] = value;
}

void nms_init_zeros(float *input, float value, int size) {
    int BLOCK_X =512;
    int block_x1 = (size > BLOCK_X) ? BLOCK_X : size;
    int grid_x1 = (size - 1) / block_x1 + 1;
    init_zeros_kernel<<<grid_x1, block_x1>>>(input, value, size);
}














__global__ void squeeze_for_score_kernel(const float *cls_rw, float *score, int *cls_index, int *range_index_rw, int num_cls, int num_box, float score_thresh){
    int box_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(box_index>=num_box)
        return;

    const float *cls_rw_row = cls_rw + box_index * num_cls;

    int cls = 0;
    float score_item = cls_rw_row[0]; // init score

    for (int i = 1; i < num_cls; ++i){
        if(score_item<cls_rw_row[i]){
            cls = i;
            score_item = cls_rw_row[i];
        }
    }
    // printf(" %d^^^^%f ", cls, score_item);

    if(score_item<score_thresh){
        cls_index[box_index] = -1;
        range_index_rw[box_index] = -1;
    }
    else{
        cls_index[box_index] = cls;
        range_index_rw[box_index] = box_index;
        score[box_index] = -score_item; //negative num for sort
    }

}


__global__ void init_int(int *inputs, int value, int num){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=num)
        return;
    inputs[index] = value;
}

__global__ void init_float(float *inputs, float value, int num){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=num)
        return;
    inputs[index] = value;
}

__global__ void iou_self_bev_kernel(int nms_pre_maxsize, const float *boxes, int *index, float *ans_iou){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nms_pre_maxsize*nms_pre_maxsize){
        return;
    }
    int row = __float2int_rd(i / nms_pre_maxsize); //rows
    int col = i - row*nms_pre_maxsize;   //cols

    int a_idx = index[row];
    int b_idx = index[col];

    if(a_idx!=b_idx){
        const float * cur_box_a = boxes + a_idx * 7;
        const float * cur_box_b = boxes + b_idx * 7;
        float cur_iou_bev = iou_bev(cur_box_a, cur_box_b);

        ans_iou[row * nms_pre_maxsize + col] = cur_iou_bev;
        ans_iou[col * nms_pre_maxsize + row] = cur_iou_bev;
    }
    else{
        ans_iou[row * nms_pre_maxsize + col] = 1;
    }
}

__global__ void iou_self_kernel(int nms_pre_maxsize, const float *boxes, int *index, float *ans_iou){
    // params boxes: (N, 4) [x, y, dx, dy]

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nms_pre_maxsize*nms_pre_maxsize){
        return;
    }
    int row = __float2int_rd(i / nms_pre_maxsize); //rows
    int col = i - row*nms_pre_maxsize;   //cols

    int a_idx = index[row];
    int b_idx = index[col];

    if(a_idx!=b_idx){
        const float * cur_box_a = boxes + a_idx * 4;
        const float * cur_box_b = boxes + b_idx * 4;
        float cur_iou_bev = iou_normal(cur_box_a, cur_box_b);

        ans_iou[row * nms_pre_maxsize + col] = cur_iou_bev;
        ans_iou[col * nms_pre_maxsize + row] = cur_iou_bev;
    }
    else{
        ans_iou[row * nms_pre_maxsize + col] = 1;
    }
}

__global__ void nms_(int nms_pre_maxsize, int *index, float *ans_iou, float nms_thresh){
    for (int i = 0; i < nms_pre_maxsize; ++i){
        if(index[i]<0)
            continue;
        for (int j = i+1; j < nms_pre_maxsize; ++j){
            int offset = i * nms_pre_maxsize + j;
            float iou_s = ans_iou[offset];
            if(iou_s>nms_thresh)
                index[j] = -1;
        }
    }
}


__global__ void concat_outputs_kernel(float *box_temp, float *score_temp, int *cls_temp, int *range_index_rw,
             int num_box_info, int total_box, int nms_post_maxsize, float *dst_s_rw, int num_concat_info){
    int copy_num = 0;
    int dst_width = num_box_info + 2; //box info size + score + cls
    for (int i = 0; i < total_box; ++i){
        if(copy_num>=nms_post_maxsize)
            return;
        int offset = range_index_rw[i];
        if(offset<0)
            continue;

        #pragma unroll
        for (int j = 0; j < num_box_info; ++j){
            dst_s_rw[copy_num*dst_width+j] = box_temp[offset*num_box_info+j];
        }
        dst_s_rw[copy_num*dst_width+num_box_info] = -score_temp[i];//前面排序时存为负值,score_temp与range_index_rw是同续的，因此都用i索引
        dst_s_rw[copy_num*dst_width+num_box_info+1] = (float)cls_temp[offset];
        copy_num += 1;
    }
}

struct is_neg
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x < 0;
  }
};

__global__ void copy_to_temp_kernel(int *range_index_rw, float *score, int *cls_index, const float *box_s_rw, float *score_temp, int *cls_temp, float *box_temp,
                                    int num, int num_box_info){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= num)
		return;

	int index = range_index_rw[i];
	score_temp[i] = score[index];
	cls_temp[i] = cls_index[index];

	#pragma unroll
	for (int j = 0; j < num_box_info; ++j){
        box_temp[i*num_box_info+j] = box_s_rw[index*num_box_info+j];
    }

    range_index_rw[i] = i;

}

cudaError_t cuda_nms(cudaStream_t stream, float *batch_box, float *batch_cls_rw, float *score, int *cls_index, int *range_index_rw, float *ious_rw, float *dst,
    float *score_temp, int *cls_temp, float *box_temp,
    int num_box, int num_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, int batch_size, float score_thresh, int use_bev){
    int num_box_info;
    if(use_bev!=0){
        num_box_info=7;
    }
    else{
        num_box_info=4;
    }

    for (int i = 0; i < batch_size; ++i){
        const float *cls_s_rw = batch_cls_rw + i*num_box*num_cls;
        const float *box_s_rw = batch_box + i*num_box*num_box_info;
        float *dst_s_rw = dst + i*nms_post_maxsize*(num_box_info+2);

        int block_x = (num_box > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : num_box;
        int grid_x = (num_box - 1) / block_x + 1;

        // std::ofstream  outf("score.bin", std::ios::app | std::ios::binary);

        // float* points_h = (float*)malloc(sizeof(int)*num_box*num_cls);
        // cudaMemcpy(points_h, cls_s_rw, sizeof(int)*num_box*num_cls, cudaMemcpyDeviceToHost);
        // // for (int i =0; i<num_box*num_cls;i++){
        // //     // std::cout << "  " << points_h[i] << "  ";
        // //     outf << 
        // // }
        // // std::cout << " \n" << std::endl;
        // outf.write(reinterpret_cast<const char*>(points_h), sizeof(int)*num_box*num_cls);
        // outf.close();

        // printf("888888888888: %d  %d  %f %d  %d\n", num_cls, num_box, score_thresh, grid_x, block_x);

        //挑选每个box中score最大的类，记录类别索引和score；记录box索引，不满足阈值的box，索引记为-1
        squeeze_for_score_kernel<<<grid_x, block_x>>>(cls_s_rw, score, cls_index, range_index_rw, num_cls, num_box, score_thresh); //提取cls信息和score信息

        // int shou_size = num_box;
        // // int show_b = (shou_size > BLOCK_X) ? BLOCK_X : shou_size;
        // // int show_g = (shou_size - 1) / show_b + 1;
        // // std::ofstream  outf("voxel.bin", std::ios::app | std::ios::binary);
        // int* points_h = (int*)malloc(sizeof(int)*shou_size);
        // cudaMemcpy(points_h, range_index_rw, sizeof(int)*shou_size, cudaMemcpyDeviceToHost);
        // for (int i =0; i<shou_size;i++){
        //     std::cout << "  " << points_h[i] << "  ";
        // }
        // std::cout << " \n" << std::endl;
        // // outf.write(reinterpret_cast<const char*>(points_h), sizeof(int)*shou_size);
        // // outf.close();
        

        //聚合索引大于-1的box，为有效box
        int *new_end = thrust::remove_if(thrust::device, range_index_rw, range_index_rw + num_box, is_neg());
	    int num = new_end - range_index_rw;
        // printf("99999999999999999999 num is %d\n", num);
        if(num<1){
            return cudaGetLastError();
	    }
	    if(num>nms_pre_maxsize){
	        num = nms_pre_maxsize;
	    }

	    int block_x_mask = (num > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : num;
        int grid_x_mask = (num - 1) / block_x_mask + 1;
        //提取有效的box信息到临时内存
        copy_to_temp_kernel<<<grid_x_mask, block_x_mask>>>(range_index_rw, score, cls_index, box_s_rw, score_temp, cls_temp, box_temp, num, num_box_info);

	    //将有效的box按照置信度排序,注意，此时score_temp，range_index_rw都是排序的结果
	    thrust::stable_sort_by_key(thrust::device, score_temp, score_temp + num, range_index_rw);

	    block_x = (num*num > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : num*num;
        grid_x = (num*num - 1) / block_x + 1;

        if(use_bev!=0){
            iou_self_bev_kernel<<<grid_x, block_x>>>(num, box_temp, range_index_rw, ious_rw);
        }
        else{
            iou_self_kernel<<<grid_x, block_x>>>(num, box_temp, range_index_rw, ious_rw);
        }
        nms_<<<1,1>>>(num, range_index_rw, ious_rw, nms_thresh);
        concat_outputs_kernel<<<1,1>>>(box_temp, score_temp, cls_temp, range_index_rw, num_box_info, num, nms_post_maxsize, dst_s_rw, num_box_info+2);

    }

    return cudaGetLastError();
}