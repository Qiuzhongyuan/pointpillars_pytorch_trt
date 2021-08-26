#ifndef DENSE_H
#define DENSE_H
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "cuda_scatter.h"
#include <iostream>
using namespace std;

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

#define CHECK_DIM(x, num) do { \
  if (x.dim() != num) { \
    fprintf(stderr, "%s should have %d dims but find %d at %s:%d\n", #x, num, x.dim(), __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


torch::Tensor
dense(torch::Tensor features, torch::Tensor indices, torch::Tensor valid, int batch_size, std::vector<int> spatialShape){
	CHECK_INPUT(features);
	CHECK_INPUT(indices);
	CHECK_INPUT(valid);
	CHECK_DIM(features, 2);
	CHECK_DIM(indices, 2);
    auto inputType = features.scalar_type();

	int num_voxels = features.size(0);
	int max_voxels = num_voxels / batch_size;//每个batch内部的最大个数
	int num_features = features.size(1);
	torch::Tensor output = torch::zeros({batch_size, num_features, spatialShape[0], spatialShape[1], spatialShape[2]}, torch::dtype(inputType).device(features.device()));
	const int *indices_rw = indices.data_ptr<int>();
	const int *valid_rw = valid.data_ptr<int>();

	if(inputType == torch::kFloat32)
    {
        const float *features_rw = features.data_ptr<float>();
        float *output_rw = output.data_ptr<float>();
        DenseSpace::cuda_scatter(features_rw, indices_rw, valid_rw, output_rw, spatialShape, max_voxels, batch_size, num_features);
    }
    else if(inputType == torch::kHalf)
    {
        const __half* features_rw = reinterpret_cast<__half*>(features.data_ptr<at::Half>());
        __half* output_rw   = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
	    DenseSpace::cuda_scatter_fp16(features_rw, indices_rw, valid_rw, output_rw, spatialShape, max_voxels, batch_size, num_features);
    }
    else
    {
        cout<< "error inputs type in Dense: " << inputType << endl;
    }
	return output.contiguous();

}

torch::Tensor
dense_backward(torch::Tensor grad_output, torch::Tensor indices, torch::Tensor valid, std::vector<int> spatialShape){
	CHECK_INPUT(grad_output);
	CHECK_INPUT(indices);
	CHECK_INPUT(valid);
    CHECK_DIM(grad_output, 5);
	CHECK_DIM(indices, 2);

	int num_voxels = indices.size(0);
	int batch_size = grad_output.size(0);
	int max_voxels = num_voxels / batch_size;//每个batch内部的最大个数
	int num_features = grad_output.size(1);
	torch::Tensor output = torch::zeros({num_voxels, num_features}, torch::dtype(torch::kFloat32).device(grad_output.device()));
	const float *features_rw = grad_output.data_ptr<float>();
	const int *indices_rw = indices.data_ptr<int>();
	const int *valid_rw = valid.data_ptr<int>();
	float *output_rw = output.data_ptr<float>();

    DenseSpace::cuda_scatter_backward(features_rw, indices_rw, valid_rw, output_rw, spatialShape, max_voxels, batch_size, num_features);

	return output.contiguous();
}

#endif










