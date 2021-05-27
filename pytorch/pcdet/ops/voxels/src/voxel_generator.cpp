#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "voxel_generator.h"

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
	std::vector<int> grid_size(3);
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

    float bev_map = (float)(grid_size[0] * grid_size[1] * sizeof(float)) / 1024 / 1024;
    int value_map_z = 40.0 / bev_map;//限制映射图的尺寸，超过40M只用一层

    value_map_z = std::max(value_map_z, 1);
    value_map_z = std::min(value_map_z, grid_size[2]);
    int mapsize = grid_size[0] * grid_size[1] * value_map_z;

    torch::Tensor map_tensor  = torch::zeros({batch_size, mapsize}, options) - 1;
    int* map_tensor_rw = map_tensor.data_ptr<int>();

    torch::Tensor addr_tensor  = torch::zeros({batch_size*max_voxels, }, options) - 1;
    int* addr_tensor_rw = addr_tensor.data_ptr<int>();

    const int listBytes = N * sizeof(VoxelGeneratorSpace::HashEntry);
    torch::Tensor list_tensor = torch::zeros({listBytes / 4, }, options);
    VoxelGeneratorSpace::HashEntry* list_tensor_rw = reinterpret_cast<VoxelGeneratorSpace::HashEntry*>(list_tensor.data_ptr<int>());

	const int *ValidInput_ptr = ValidInput.data_ptr<int>();
	int *coors_ptr = coors.data_ptr<int>();
	int *num_points_per_voxel_ptr = num_points_per_voxel.data_ptr<int>();
	int *ValidOutput_ptr = ValidOutput.data_ptr<int>();


    if(inputType == torch::kFloat32)
    {
        const float *points_ptr = points.data_ptr<float>();
        float *voxels_ptr = voxels.data_ptr<float>();

        VoxelGeneratorSpace::cuda_points_to_voxel(points_ptr, ValidInput_ptr,
                                                   coors_ptr, num_points_per_voxel_ptr, voxels_ptr, ValidOutput_ptr,
                                                   map_tensor_rw, addr_tensor_rw, list_tensor_rw,
                                                   point_cloud_range, voxel_size, grid_size,
                                                   batch_size, N, inCols, outCols,
                                                   cluster_offset, center_offset, supplement, max_voxels, max_num_points, value_map_z);
    }
    else if(inputType == torch::kHalf)
    {
        const __half *points_ptr = reinterpret_cast<__half*>(points.data_ptr<at::Half>());
        __half *voxels_ptr   = reinterpret_cast<__half*>(voxels.data_ptr<at::Half>());

        VoxelGeneratorSpace::cuda_points_to_voxel_fp16(points_ptr, ValidInput_ptr,
                                                   coors_ptr, num_points_per_voxel_ptr, voxels_ptr, ValidOutput_ptr,
                                                   map_tensor_rw, addr_tensor_rw, list_tensor_rw,
                                                   point_cloud_range, voxel_size, grid_size,
                                                   batch_size, N, inCols, outCols,
                                                   cluster_offset, center_offset, supplement, max_voxels, max_num_points, value_map_z);
    }
    else
    {
        cout<< "error inputs type in VoxelGeneratorV1: " << inputType << endl;
    }

	return {voxels.contiguous(), coors.contiguous(), ValidOutput.contiguous(), num_points_per_voxel};
}


