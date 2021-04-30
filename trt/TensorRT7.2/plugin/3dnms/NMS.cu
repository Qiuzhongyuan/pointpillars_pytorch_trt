#include <thrust/sort.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "NMS_nova.h"
#define MIN_THREADS_PER_BLOCK 128

using namespace std;

namespace NAMESPACE
{

inline int GetMaxOccupacy(int SMs, int processNum)
{
    int threshold = processNum / SMs;
    int thread = MIN_THREADS_PER_BLOCK;

    while(thread < threshold)
        thread = thread << 1;

    thread = thread >> 1;

    thread = thread > 512 ? 512 : thread;

    return thread;

}

// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;

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

__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
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

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){
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

__device__ inline float iou_bev(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__device__ inline float iou_normal(float const * const a, float const * const b) {
    //params: a: [x, y, dx, dy]
    //params: b: [x, y, dx, dy]

    float left = fmaxf(a[0] - a[2] / 2, b[0] - b[2] / 2), right = fminf(a[0] + a[2] / 2, b[0] + b[2] / 2);
    float top  = fmaxf(a[1] - a[3] / 2, b[1] - b[3] / 2), bottom = fminf(a[1] + a[3] / 2, b[1] + b[3] / 2);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = a[3] * a[2];
    float Sb = b[3] * b[2];
    return interS / fmaxf(Sa + Sb - interS, EPS);
}


extern "C"
__global__
void squeeze_for_score_kernel(const float *cls_rw, float *score, int *cls_index, int *range_index_rw, int* counter, int num_cls, int num_box, float score_thresh)
{

    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ int dCounter;

    for(int i = idx; i < num_box; i += stride)
    {
        if(threadIdx.x == 0)    dCounter = 0;
        __syncthreads();

        int cls = 0;
        float scoreTmp = cls_rw[i];

        for(int j = 1; j < num_cls; ++j)
        {
            float  scoreNow  = cls_rw[j * num_box + i];
            bool isSmaller = (scoreTmp < scoreNow);

            scoreTmp = isSmaller ? scoreNow : scoreTmp;
            cls      = isSmaller ? j        : cls;
        }

        bool isValid = (score_thresh < scoreTmp);
        int pos;

        if(isValid) pos = atomicAdd(&dCounter, 1);
        __syncthreads();
        if(threadIdx.x == 0) dCounter = atomicAdd(counter, dCounter);
        __syncthreads();

        if(isValid)
        {
            pos += dCounter;
            cls_index     [i] = cls;
            range_index_rw[pos] = i;
            score         [pos] = -scoreTmp;
        }

    }
}

__global__ void copy_to_temp_kernel_bev(int *range_index_rw, float *score, int *cls_index, const float *box_s_rw, int *cls_temp, float *box_temp,
                                        int num, int original_num, int num_box_info)
{

    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num; i += stride)
    {
        int index   = range_index_rw[i];
        cls_temp[i] = cls_index[index];
        #pragma unroll 7
        for (int j = 0; j < 7; ++j)
            box_temp[j * num + i] = box_s_rw[j * original_num + index];
    }

}

__global__ void copy_to_temp_kernel_nor(int *range_index_rw, float *score, int *cls_index, const float *box_s_rw, int *cls_temp, float *box_temp,
                                    int num, int original_num, int num_box_info)
{

    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num; i += stride)
    {
        int index   = range_index_rw[i];
        cls_temp[i] = cls_index[index];

        #pragma unroll 4
        for (int j = 0; j < 4; ++j)
            box_temp[j * num + i] = box_s_rw[j * original_num + index];

    }

}



__global__ void iou_self_bev_kernel(int nms_pre_maxsize, const float *boxes, int *index, float *ans_iou)
{
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]

    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < nms_pre_maxsize * nms_pre_maxsize; i += stride)
    {
        int row = i / nms_pre_maxsize; //rows
        int col = i - (row * nms_pre_maxsize);   //cols

        int a_idx = row;
        int b_idx = col;

        if(a_idx!=b_idx){
            float box_a[7];
            float box_b[7];
            #pragma unroll 7
            for(int j = 0; j < 7; ++j)
            {
                box_a[j] = boxes[j * nms_pre_maxsize + a_idx];
                box_b[j] = boxes[j * nms_pre_maxsize + b_idx];
            }
            float cur_iou_bev = iou_bev(reinterpret_cast<const float*>(&box_a[0]), reinterpret_cast<const float*>(&box_b[0]));

            ans_iou[row * nms_pre_maxsize + col] = cur_iou_bev;
            ans_iou[col * nms_pre_maxsize + row] = cur_iou_bev;
        }else{
            ans_iou[row * nms_pre_maxsize + col] = 1;
        }
    }
}

__global__ void iou_self_kernel(int nms_pre_maxsize, const float *boxes, int *index, float *ans_iou){
    // params boxes: (N, 4) [x, y, dx, dy]

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < nms_pre_maxsize * nms_pre_maxsize; i += stride)
    {
        int row = i / nms_pre_maxsize; //rows
        int col = i - (row * nms_pre_maxsize);   //cols

        int a_idx = index[row];
        int b_idx = index[col];

        if(a_idx!=b_idx){
            float box_a[4];
            float box_b[4];
            #pragma unroll 4
            for(int j = 0; j < 4; ++j)
            {
                box_a[j] = boxes[j * nms_pre_maxsize + a_idx];
                box_b[j] = boxes[j * nms_pre_maxsize + b_idx];
            }
            float cur_iou = iou_normal(reinterpret_cast<const float*>(&box_a[0]), reinterpret_cast<const float*>(&box_b[0]));

            ans_iou[row * nms_pre_maxsize + col] = cur_iou;
            ans_iou[col * nms_pre_maxsize + row] = cur_iou;
        }
        else{
            ans_iou[row * nms_pre_maxsize + col] = 1;
        }
    }

}


__global__
void nms_kernel(int nms_pre_maxsize, int *range_index, float *ans_iou, float nms_thresh, int box_idx)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < nms_pre_maxsize; i += stride)
    {
        if(range_index[box_idx]<0)  continue;
        if(i<=box_idx)  continue;
        if(ans_iou[box_idx * nms_pre_maxsize + i] > nms_thresh) range_index[i] = -1;

    }
}

void nms_func(int nms_pre_maxsize, int *range_index, float *ans_iou, float nms_thresh)
{

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nms_kernel, 0, nms_pre_maxsize));
    minGridSize = std::min(minGridSize, DivUp(nms_pre_maxsize, blockSize));
    for (int i = 0; i < nms_pre_maxsize; ++i)
        nms_kernel<<<minGridSize, blockSize>>>(nms_pre_maxsize, range_index, ans_iou, nms_thresh, i);
    //cudaDeviceSynchronize();

}

__global__
void concat_outputs_kernel_bev(float *box_temp, float *score_temp, int *cls_temp, int *range_index_rw,
                                 int total_box, int nms_post_maxsize, float *dst_s_rw, int orign_num)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_box; i += stride)
    {
        int index = range_index_rw[i];
        #pragma unroll 7
        for (int j = 0; j < 7; ++j)
            dst_s_rw[j * nms_post_maxsize + i] = box_temp[j * orign_num + index];

        dst_s_rw[7 * nms_post_maxsize + i] = -score_temp[index];//前面排序时存为负值
        dst_s_rw[8 * nms_post_maxsize + i] = (float)cls_temp[index];

    }

}

__global__
void concat_outputs_kernel_nor(float *box_temp, float *score_temp, int *cls_temp, int *range_index_rw,
                                 int total_box, int nms_post_maxsize, float *dst_s_rw, int orign_num)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_box; i += stride)
    {
        int index = range_index_rw[i];
        #pragma unroll 4
        for (int j = 0; j < 4; ++j)
            dst_s_rw[j * nms_post_maxsize + i] = box_temp[j * orign_num + index];

        dst_s_rw[4 * nms_post_maxsize + i] = -score_temp[index];//前面排序时存为负值
        dst_s_rw[5 * nms_post_maxsize + i] = (float)cls_temp[index];

    }

}

__global__
void range_kernel(int *index, int num)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num; i += stride)
    {
        index[i] = i;
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

void cuda_nms(const float *batch_box, const float *batch_cls_rw, float *score, int *cls_index, int *range_index_rw, int* pos_rw, int *cls_temp, float *box_temp,
    float *ious_rw, float *dst, int num_box, int num_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, int batch_size,
    float score_thresh, int use_bev)
{

    int SMs = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, 0));
    int num_box_info = use_bev == 0 ? 4 : 7;

    for (int i = 0; i < batch_size; ++i){
        const float *cls_s_rw = batch_cls_rw + i * num_box * num_cls;
        const float *box_s_rw = batch_box    + i * num_box * num_box_info;

        float *dst_s_rw = dst + i * nms_post_maxsize * (num_box_info + 2);
        int   *pos_s_rw = pos_rw + i;

        //挑选每个box中score最大的类，记录类别索引和score；记录box索引，不满足阈值的box，索引记为-1
        squeeze_for_score_kernel<<<SMs, GetMaxOccupacy(SMs, num_box), sizeof(int)>>>(cls_s_rw, score, cls_index, range_index_rw, pos_s_rw, num_cls, num_box, score_thresh); //提取cls信息和score信息

        int num;
        checkCudaErrors(cudaMemcpy(&num, pos_s_rw, sizeof(int), cudaMemcpyDeviceToHost));
        if(num < 1)   continue;

        //将有效的box按照置信度排序,注意，此时score，range_index_rw都是排序的结果
	    thrust::stable_sort_by_key(thrust::device, score, score + num, range_index_rw);

	    if(num > nms_pre_maxsize) num = nms_pre_maxsize;

        if(use_bev != 0)  copy_to_temp_kernel_bev<<<SMs, GetMaxOccupacy(SMs, num)>>>(range_index_rw, score, cls_index, box_s_rw, cls_temp, box_temp, num, num_box, num_box_info);
        else              copy_to_temp_kernel_nor<<<SMs, GetMaxOccupacy(SMs, num)>>>(range_index_rw, score, cls_index, box_s_rw, cls_temp, box_temp, num, num_box, num_box_info);

        if(use_bev != 0)  iou_self_bev_kernel<<<SMs, GetMaxOccupacy(SMs, num * num)>>>(num, box_temp, range_index_rw, ious_rw);
        else              iou_self_kernel    <<<SMs, GetMaxOccupacy(SMs, num * num)>>>(num, box_temp, range_index_rw, ious_rw);

        range_kernel<<<SMs, GetMaxOccupacy(SMs, num)>>>(range_index_rw, num);// mark temp index
        nms_func(num, range_index_rw, ious_rw, nms_thresh);

        //聚合索引大于-1的box，为有效box
        int *new_end = thrust::remove_if(thrust::device, range_index_rw, range_index_rw + num, is_neg());
        int valid_num = new_end - range_index_rw;

        if(valid_num < 1)   continue;
        valid_num = valid_num > nms_post_maxsize ? nms_post_maxsize : valid_num;

        if(use_bev != 0)  concat_outputs_kernel_bev<<<SMs, GetMaxOccupacy(SMs, valid_num)>>>(box_temp, score, cls_temp, range_index_rw, valid_num, nms_post_maxsize, dst_s_rw, num);
        else              concat_outputs_kernel_nor<<<SMs, GetMaxOccupacy(SMs, valid_num)>>>(box_temp, score, cls_temp, range_index_rw, valid_num, nms_post_maxsize, dst_s_rw, num);


    }

    checkCudaErrors(cudaDeviceSynchronize());

}

}//namespace