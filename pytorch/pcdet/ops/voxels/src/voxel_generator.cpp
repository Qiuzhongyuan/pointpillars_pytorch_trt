#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda_voxel_generator.h"
using namespace std;

std::vector<torch::Tensor>
VoxelGeneratorV1(torch::Tensor points, torch::Tensor ValidInput, std::vector<float> voxel_size,
	std::vector<float> point_cloud_range, int max_num_points, int max_voxels, int batch_size,
	int center_offset, int cluster_offset, int supplement)
{
	CHECK_INPUT(points);
	CHECK_INPUT(ValidInput);
	auto inputType = points.scalar_type();

	int inCols = points.size(1);
	int num_features = inCols - 1;
	int outCols = num_features;
	if(cluster_offset !=0) outCols += 3;
	if(center_offset !=0) outCols += 3;
	int N = points.size(0);

	const int NDim = 3;
	int grid_size[NDim];
	  for (int i = 0; i < NDim; ++i) {
		grid_size[i] =
			round((point_cloud_range[NDim + i] - point_cloud_range[i]) / voxel_size[i]);
	  }
    int cuda_idx = points.device().index();
    auto options = torch::TensorOptions({at::kCUDA, cuda_idx}).dtype(torch::kInt32);

	torch::Tensor voxels = torch::zeros({max_voxels * batch_size, max_num_points, outCols}, torch::dtype(inputType).device(points.device())); //init 0
	torch::Tensor coors = torch::zeros({max_voxels * batch_size, 4}, options) - 1;  //init -1
    torch::Tensor num_points_per_voxel = torch::zeros({max_voxels * batch_size, }, options); //init 0
    torch::Tensor ValidOutput = torch::zeros({batch_size, }, options); //init 0
    torch::Tensor prob_voxel_index = torch::zeros({N, }, torch::dtype(torch::kInt32).device(points.device())) - 1; //init -1

    //for CSR
    const int batchRowOffset = grid_size[0]*grid_size[1] + 1;
    torch::Tensor TensorRowOffset = torch::zeros({batch_size, batchRowOffset}, options);
    torch::Tensor TensorColunms = torch::zeros({batch_size, max_voxels}, options);

	const int *ValidInput_ptr = ValidInput.data_ptr<int>();
	int *coors_ptr = coors.data_ptr<int>();
	int *num_points_per_voxel_ptr = num_points_per_voxel.data_ptr<int>();
	int *ValidOutput_ptr = ValidOutput.data_ptr<int>();
    int *prob_voxel_index_ptr = prob_voxel_index.data_ptr<int>();

    int* TensorRowOffsetPtr = TensorRowOffset.data_ptr<int>();
    int* TensorColumnsPtr = TensorColunms.data_ptr<int>();

    if(inputType == torch::kFloat32)
    {
        const float *points_ptr = points.data_ptr<float>();
        float *voxels_ptr = voxels.data_ptr<float>();

        VoxelGeneratorV1Space::cuda_points_to_voxel(points_ptr, ValidInput_ptr, prob_voxel_index_ptr,
                                                   coors_ptr, voxels_ptr, num_points_per_voxel_ptr, ValidOutput_ptr,
                                                   grid_size[0], grid_size[1], grid_size[2],
                                                   point_cloud_range[0], point_cloud_range[1], point_cloud_range[2],
                                                   voxel_size[0], voxel_size[1], voxel_size[2],
                                                   batch_size, N, inCols, max_voxels, max_num_points,
                                                   TensorRowOffsetPtr, TensorColumnsPtr,
                                                   cluster_offset, center_offset, supplement, cuda_idx);
    }
    else if(inputType == torch::kHalf)
    {
        const __half *points_ptr = reinterpret_cast<__half*>(points.data_ptr<at::Half>());
        __half *voxels_ptr   = reinterpret_cast<__half*>(voxels.data_ptr<at::Half>());

        VoxelGeneratorV1Space::cuda_points_to_voxel_fp16(points_ptr, ValidInput_ptr, prob_voxel_index_ptr,
                                                       coors_ptr, voxels_ptr, num_points_per_voxel_ptr, ValidOutput_ptr,
                                                       grid_size[0], grid_size[1], grid_size[2],
                                                       point_cloud_range[0], point_cloud_range[1], point_cloud_range[2],
                                                       voxel_size[0], voxel_size[1], voxel_size[2],
                                                       batch_size, N, inCols, max_voxels, max_num_points,
                                                       TensorRowOffsetPtr, TensorColumnsPtr,
                                                       cluster_offset, center_offset, supplement, cuda_idx);
    }
    else
    {
        cout<< "error inputs type in VoxelGeneratorV1: " << inputType << endl;
    }

	return {voxels.contiguous(), coors.contiguous(), ValidOutput.contiguous(), num_points_per_voxel};
}

