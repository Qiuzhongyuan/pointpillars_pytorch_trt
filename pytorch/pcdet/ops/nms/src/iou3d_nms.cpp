#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_nms.h"
#include "cuda_nms.h"
#include <iostream>
using namespace std;

std::vector<torch::Tensor>
nms(torch::Tensor batch_box, torch::Tensor batch_cls, int nms_pre_maxsize, int nms_post_maxsize, float nms_thresh, float score_thresh, int use_bev = 1){
    CHECK_INPUT(batch_box);
    CHECK_INPUT(batch_cls);

    auto inputType = batch_cls.scalar_type();
    int batch_size = batch_box.size(0);
    int num_box = batch_box.size(1);
    int num_cls = batch_cls.size(2);

    int num_out_info = use_bev == 0 ? 6 : 9;

    torch::Tensor index    = torch::zeros({num_box,}, torch::dtype(torch::kInt32).device(batch_box.device()));
    torch::Tensor score    = torch::zeros({num_box,}, torch::dtype(torch::kFloat32).device(batch_box.device())); // always float32
    torch::Tensor cls_type = torch::zeros({num_box,}, torch::dtype(torch::kInt32).device(batch_box.device()));

    torch::Tensor cls_temp = torch::zeros({nms_pre_maxsize,}, torch::dtype(torch::kInt32).device(batch_box.device()));
    torch::Tensor box_temp = torch::zeros({nms_pre_maxsize, num_out_info-2}, torch::dtype(torch::kFloat32).device(batch_box.device())); // always float32
    torch::Tensor pos      = torch::zeros({batch_size, 2}, torch::dtype(torch::kInt32).device(batch_box.device()));
    torch::Tensor ious     = torch::zeros({nms_pre_maxsize, nms_pre_maxsize}, torch::dtype(torch::kFloat32).device(batch_box.device())); // always float32

    torch::Tensor outputs  = torch::zeros({batch_size, nms_post_maxsize, num_out_info}, torch::dtype(inputType).device(batch_box.device())) - 1;
    torch::Tensor validboxes    = torch::zeros({batch_size,}, torch::dtype(torch::kInt32).device(batch_box.device()));

    int   *index_rw = index.data_ptr<int>();
    int   *cls_type_rw = cls_type.data_ptr<int>();
    int   *cls_temp_index_rw = cls_temp.data_ptr<int>();
    int   *pos_rw  = pos.data_ptr<int>();

    int   *validboxes_rw  = validboxes.data_ptr<int>();

    float *ious_rw = ious.data_ptr<float>();
    float *score_rw = score.data_ptr<float>();
    float *box_temp_index_rw = box_temp.data_ptr<float>();

    if(inputType == torch::kFloat32)
    {
        const float *batch_box_rw = batch_box.data_ptr<float>();
        const float *batch_cls_rw = batch_cls.data_ptr<float>();
        float       *outputs_rw   = outputs.data_ptr<float>();
        NMSSpace::cuda_nms(batch_box_rw, batch_cls_rw, score_rw, cls_type_rw, index_rw, pos_rw, cls_temp_index_rw, box_temp_index_rw,
                            ious_rw, outputs_rw, num_box, num_cls, nms_pre_maxsize, nms_post_maxsize, nms_thresh, batch_size, score_thresh, use_bev, validboxes_rw);
    }
    else if(inputType == torch::kHalf)
    {
        const __half *batch_box_rw = reinterpret_cast<__half*>(batch_box.data_ptr<at::Half>());
        const __half *batch_cls_rw = reinterpret_cast<__half*>(batch_cls.data_ptr<at::Half>());
        __half       *outputs_rw   = reinterpret_cast<__half*>(outputs.data_ptr<at::Half>());
        NMSSpace::cuda_nms_fp16(batch_box_rw, batch_cls_rw, score_rw, cls_type_rw, index_rw, pos_rw, cls_temp_index_rw, box_temp_index_rw,
                            ious_rw, outputs_rw, num_box, num_cls, nms_pre_maxsize, nms_post_maxsize, nms_thresh, batch_size, score_thresh, use_bev, validboxes_rw);
    }
    else
    {
        cout<< "error inputs type in nms: " << inputType << endl;
    }

    return {outputs.contiguous(), validboxes};
}
