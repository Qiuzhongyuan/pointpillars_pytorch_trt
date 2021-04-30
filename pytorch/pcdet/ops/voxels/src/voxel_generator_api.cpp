#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "voxel_generator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("VoxelGeneratorV1", &VoxelGeneratorV1, "VoxelGeneratorV1-3D-model");
}
