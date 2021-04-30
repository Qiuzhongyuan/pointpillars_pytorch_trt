#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dense.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dense", &dense, "dense");
	m.def("dense_backward", &dense_backward, "dense_backward");
}
