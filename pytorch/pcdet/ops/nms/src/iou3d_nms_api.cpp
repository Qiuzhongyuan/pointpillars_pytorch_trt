#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "iou3d_nms.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("nms", &nms, "nms func");
}
