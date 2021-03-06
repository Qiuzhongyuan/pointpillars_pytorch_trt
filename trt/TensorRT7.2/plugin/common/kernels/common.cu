/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cublas_v2.h"
#include <cub/cub.cuh>
#include <stdint.h>
#include "kernel.h"
#include "bboxUtils.h"

#define CUDA_MEM_ALIGN 256

__global__ static void float2half_kernel(float *in, __half *out, int size){
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += stride)
    {
        out[idx] = __float2half(in[idx]);
    }
}

void float2half_custom(float *in, __half *out, int size){
    int blockSize;   // The launch configurator returned block size
    int minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, float2half_kernel, 0, size);
    minGridSize = std::min(minGridSize, DivUp(size, blockSize));
    float2half_kernel<<<minGridSize, blockSize>>>(in, out, size);
}


__global__ static void half2float_kernel(const __half *in, float *out, int size){
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += stride)
    {
        out[idx] = __half2float(in[idx]);
    }
}

void half2float_custom(const __half *in, float *out, int size){
    int blockSize;   // The launch configurator returned block size
    int minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, half2float_kernel, 0, size);
    minGridSize = std::min(minGridSize, DivUp(size, blockSize));
    half2float_kernel<<<minGridSize, blockSize>>>(in, out, size);
}

__global__ void init_float_half_kernel(__half *input, __half value, int size){
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += stride)
    {
        input[i] = value;
    }
}

void init_float_half(__half *input, __half value, int size){
    int blockSize;   // The launch configurator returned block size
    int minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_float_half_kernel, 0, size);
    minGridSize = std::min(minGridSize, DivUp(size, blockSize));
    init_float_half_kernel<<<minGridSize, blockSize>>>(input, value, size);
}

__global__ void init_int_kernel(int *input, int value, int size){
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += stride)
    {
        input[i] = value;
    }
}

void init_int_(int *input, int value, int size) {
    int blockSize;   // The launch configurator returned block size
    int minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_int_kernel, 0, size);
    minGridSize = std::min(minGridSize, DivUp(size, blockSize));
    init_int_kernel<<<minGridSize, blockSize>>>(input, value, size);
}

__global__ void init_float_kernel(float *input, float value, int size){
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < size; i += stride)
    {
        input[i] = value;
    }
}

void init_float(float *input, float value, int size){
    int blockSize;   // The launch configurator returned block size
    int minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_float_kernel, 0, size);
    minGridSize = std::min(minGridSize, DivUp(size, blockSize));
    init_float_kernel<<<minGridSize, blockSize>>>(input, value, size);
}


// HASH 
unsigned int hash(const void* array_, size_t size)
{
    // Apply hashing only when debugging RPN codes.
    if (DEBUG_ENABLE)
    {
        const char* array_const;
        char* array;
        cudaMallocHost((void**) &array, size);
        cudaMemcpy(array, array_, size, cudaMemcpyDeviceToHost);
        array_const = array;
        unsigned int hash = 45599;
        for (size_t i = 0; i < size; i++)
        {
            unsigned int value = array_const[i];
            hash = hash * 1487 + value;
            hash = hash * 317;
            hash = hash % 105359;
        }
        return hash;
    }
    else
    {
        return 0;
    }
}

// ALIGNPTR 
int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

// NEXTWORKSPACEPTR 
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE 
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN)
        {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

using nvinfer1::DataType;

// DATA TYPE SIZE 
size_t dataTypeSize(const DataType dtype)
{
    switch (dtype)
    {
    case DataType::kINT8: return sizeof(char);
    case DataType::kHALF: return sizeof(short);
    case DataType::kFLOAT: return sizeof(float);
    default: return 0;
    }
}

// CUB 
/*
size_t cubSortFloatIntPairsWorkspaceSize(int num_items, int num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
    (int *)NULL, temp_storage_bytes,
    (const float *)NULL, (float *)NULL,
    (const int *)NULL, (int *)NULL, 
    num_items,     // # items
    num_segments,  // # segments 
    (const int *)NULL, (const int *)NULL);
    return temp_storage_bytes;
}

size_t cubSortFloatBboxInfoPairsWorkspaceSize(int num_items, int num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
    (int *)NULL, temp_storage_bytes,
    (const float *)NULL, (float *)NULL,
    (const BboxInfo<float> *)NULL, (BboxInfo<float> *)NULL, 
    num_items,     // # items
    num_segments,  // # segments 
    (const int *)NULL, (const int *)NULL);
    return temp_storage_bytes;
}
*/

template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void setUniformOffsets_kernel(
        const int num_segments,
        const int offset,
        int* d_offsets)
{
    const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    if (idx <= num_segments)
        d_offsets[idx] = idx * offset;
}

void setUniformOffsets(
    cudaStream_t stream,
    const int num_segments,
    const int offset,
    int* d_offsets)
{
    const int BS = 32;
    const int GS = (num_segments + 1 + BS - 1) / BS;
    setUniformOffsets_kernel<BS><<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}


const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}
