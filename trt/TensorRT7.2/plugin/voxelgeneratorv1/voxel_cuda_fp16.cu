// #include <torch/serialize/tensor.h>
#include "voxelgeneratev1_nova.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cub/cub.cuh"
#include <cuda_fp16.h>


#define DELTA 0.01
namespace NAMESPACE
{

__global__
void ConvertFloat2Half(float* in, __half* out, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num; i += stride)
    {
        out[i] = __float2half_rn(in[i]);
    }
}


void convertFP32ToFP16(float* in, __half* out, int num)
{

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch

    checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertFloat2Half));
    ConvertFloat2Half<<<minGridSize, blockSize>>>(in, out, num);
    checkCudaErrors_voxelv1(cudaDeviceSynchronize());
}


__global__
void ConvertHalf2Float(__half* in, float* out, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num; i += stride)
    {
        out[i] = __half2float(in[i]);
    }
}



void convertFP16ToFP32(__half* in, float* out, int num)
{

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch

    checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertHalf2Float));
    ConvertHalf2Float<<<minGridSize, blockSize>>>(in, out, num);
    checkCudaErrors_voxelv1(cudaDeviceSynchronize());
}

__global__
void TraverseInputPointsPerBatchFP16(const __half* inPoints, int numPointsPerBatchInput,
                                    int* outPosIndexBuff, int* dCounter,
                                    int GridX, int GridY, int GridZ,
                                    __half RangeMinX, __half RangeMinY, __half RangeMinZ,
                                    __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                                    int inCols, int numValidPointsInput)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidPointsInput; i += stride)
    {
        const  __half* point = inPoints + i * inCols;
        //printf(" %.9f ", __half2float(point[1]));
        int bs = __half2int_rd(__hadd(point[0], __float2half(0.01f)));
        int x  = __half2int_rd(__hdiv(__hsub(point[1], RangeMinX), VoxelSizeX));
        int y  = __half2int_rd(__hdiv(__hsub(point[2], RangeMinY), VoxelSizeY));
        int z  = __half2int_rd(__hdiv(__hsub(point[3], RangeMinZ), VoxelSizeZ));
        if(bs<0 || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        int pos = atomicAdd(dCounter, 1);
        outPosIndexBuff[pos] = x * GridZ * GridY + y * GridZ + z; //XYZ
    }

}



__global__
void GetVoxelPerBatchFP16(const __half* inPoints, __half* outVoxels, int* numPerVoxel,
                         int* RO, int* CI,  int GridX, int GridY, int GridZ,
                         __half RangeMinX,  __half RangeMinY,  __half RangeMinZ,
                         __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                         int inCols, int outCols, int numValidPointsInput, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidPointsInput; i += stride)
    {
        const __half *point = inPoints + i * inCols;

        int bs = __half2int_rd(__hadd(point[0], __float2half(0.01f)));
        int x  = __half2int_rd(__hdiv(__hsub(point[1], RangeMinX), VoxelSizeX));
        int y  = __half2int_rd(__hdiv(__hsub(point[2], RangeMinY), VoxelSizeY));
        int z  = __half2int_rd(__hdiv(__hsub(point[3], RangeMinZ), VoxelSizeZ));

        if(bs<0 || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        const int row = x * GridY + y;
        int offset = Access(RO, CI, row, z);
        if(offset<0) continue;

        __half* voxel = outVoxels + offset * outCols * maxPointsPerVoxel;
        int old_num = atomicAdd(&numPerVoxel[offset], 1);

        if(old_num<maxPointsPerVoxel)
        {
            __half* voxel_temp = voxel + old_num * outCols;
            #pragma unroll
            for(int j = 0; j < inCols-1; ++j)
                voxel_temp[j] = point[j + 1];
        }
        else{
            atomicSub(&numPerVoxel[offset], 1);
        }

    }
}


__global__
void ExtendVoxelClusterOffsetPerBatchFP16(__half* Voxels, int* numPerVoxel, int numValidVoxelsOutput,
                                          __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                                            int cols, int write_offset, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        if(numpoints < 2) continue;
        __half* voxel = Voxels + i * cols * maxPointsPerVoxel;
        __half resolution[] = {VoxelSizeX, VoxelSizeY, VoxelSizeZ};
        for(int j = 0; j < 3; ++j){
            __half sum = __float2half(0.0f);
            for(int k = 0; k < numpoints; ++k)
                sum = __hadd(voxel[k * cols + j], sum);
            // sum += voxel[k*cols + j];

            __half mean_v = __hdiv(sum, __int2half_rn(numpoints));
            #pragma unroll
            for(int k = 0; k < numpoints; ++k)
                // voxel[k * cols + write_offset + j] = (voxel[k * cols + j] - mean_v) / resolution[j];
                voxel[k * cols + write_offset + j] = __hdiv(__hsub(voxel[k * cols + j], mean_v), resolution[j]);
        }
    }
}


__global__
void ExtendVoxelCenterOffsetPerBatchFP16(__half* Voxels, int* Coords, int* numPerVoxel, int numValidVoxelsOutput,
                                        __half RangeMinX,  __half RangeMinY,  __half RangeMinZ,
                                        __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                                        int cols, int write_offset, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        __half* voxel = Voxels + i * cols * maxPointsPerVoxel;
        // int4 coorTmp  = *(reinterpret_cast<int4*>(Coords + i * 4));
        // int* coor     =   reinterpret_cast<int*>(&coorTmp);
        int* coor  = Coords + i * 4;

        __half resolution[] = {VoxelSizeX, VoxelSizeY, VoxelSizeZ};
        __half rangemin[]   = {RangeMinX,  RangeMinY,  RangeMinZ};

        for(int j = 0; j < numpoints; ++j)
        {
            #pragma unroll
            for(int k = 0; k < 3; ++k)
            {
                __half center = __hadd(__int2half_rn(coor[3 - k]), __float2half(0.5f));
                center = __hfma(center, resolution[k], rangemin[k]);
                voxel[j * cols + write_offset + k] = __hdiv(__hsub(voxel[j * cols + k], center), resolution[k]);
            }

        }
    }
}

__global__
void GetVoxelSupplementPerBatchFP16(__half* Voxels, int* numPerVoxel, int numValidVoxelsOutput, int cols, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        if(numpoints < maxPointsPerVoxel){
            __half* voxel = Voxels + i * cols * maxPointsPerVoxel;
            #pragma unroll
            for(int j = numpoints; j < maxPointsPerVoxel; ++j)
            {
                __half* voxel_temp = voxel + cols * j;
                #pragma unroll
                for(int k = 0; k < cols; ++k)
                    voxel_temp[k] = voxel[k];
            }
        }
    }
}

void cuda_points_to_voxel_fp16(const __half* inPoints, const int* devNumValidPointsInput, int* outPosIndexBuff,
                                 int* outCoords, __half* outVoxels, int* outVoxelNum, int* dCounter,
                                 int GridX, int GridY, int GridZ,
                                 __half RangeMinX,  __half RangeMinY,  __half RangeMinZ,
                                 __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                                 int batchSize, int inPointsNum, int inCols, int maxValidOutput,
                                 int maxPointsPerVoxel, int* TensorRowOffsetPtr, int* TensorColumnsPtr,
                                 int cluster_offset, int center_offset, int supplement, int cuda_idx){

    std::vector<int> numValidPointsInput(batchSize);
    const int numPointsPerBatchInput = inPointsNum / batchSize;
    checkCudaErrors_voxelv1(cudaMemcpy(numValidPointsInput.data(), devNumValidPointsInput, sizeof(int) * batchSize, cudaMemcpyDeviceToHost));
    for(int i = 0; i < batchSize; ++i)
        numValidPointsInput[i] = numValidPointsInput[i] < 0 ? numPointsPerBatchInput:numValidPointsInput[i];

    int outCols = inCols-1, write_offset_cluster = inCols-1, write_offset_center = inCols-1;
    if(cluster_offset!=0){
        outCols += 3;
        write_offset_center += 3;
    }
    if(center_offset!=0) outCols += 3;

    const int batchRowOffset = GridX*GridY + 1;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch

    std::vector<int> numvalidPosOut(batchSize, 0);
    for(int i = 0; i < batchSize; ++i)
    {
        const __half* inPointsPerBatch = inPoints + i * numPointsPerBatchInput * inCols;
        int* outPosIndexBuffPerBatch = outPosIndexBuff + i * numPointsPerBatchInput;
        int* outCoordsPerBatch = outCoords + i * maxValidOutput * 4;
        __half* outVoxelsPerBatch = outVoxels + i * maxValidOutput * maxPointsPerVoxel * outCols;
        int* outVoxelNumPerBatch = outVoxelNum + i * maxValidOutput;
        int* TensorRowOffsetPtrPerBatch = TensorRowOffsetPtr + i * batchRowOffset;
        int* TensorColumnsPtrPerBatch = TensorColumnsPtr + i * maxValidOutput;
        int* dCounterPerBatch = dCounter + i;
        int* hCounterPerBatch = numvalidPosOut.data() + i;
        const int* devNumValidPointsInputPerbatch = devNumValidPointsInput + i;
        int numValidPointsInputPerBatch = numValidPointsInput[i];

        checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, TraverseInputPointsPerBatchFP16, 0, numValidPointsInputPerBatch));
        minGridSize = std::min(minGridSize, DivUp(numValidPointsInputPerBatch, blockSize));

        // std::cout<<"blockSize  " << blockSize << "  minGridSize  "<< minGridSize << "  inPointsNum: " << numValidPointsInputPerBatch << std::endl;
        TraverseInputPointsPerBatchFP16<<<minGridSize, blockSize>>>(inPointsPerBatch, numPointsPerBatchInput,
                                                                                 outPosIndexBuffPerBatch, dCounterPerBatch,
                                                                                 GridX, GridY, GridZ,
                                                                                 RangeMinX, RangeMinY, RangeMinZ,
                                                                                 VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                                 inCols, numValidPointsInputPerBatch);

        checkCudaErrors_voxelv1(cudaMemcpy(hCounterPerBatch, dCounterPerBatch, sizeof(int), cudaMemcpyDeviceToHost));
        int* outBuffStartNew = NULL;


#ifdef USE_TORCH
        auto options = torch::TensorOptions({at::kCUDA, cuda_idx}).dtype(torch::kInt32);
        if(*hCounterPerBatch<1) continue;
        std::tuple<torch::Tensor, torch::Tensor> out_results;
        bool sorted = true;
        bool return_inverse = false;
        auto outPosIndexBuffTensor = torch::from_blob(outPosIndexBuffPerBatch, {*hCounterPerBatch, }, options);
        out_results = at::_unique(outPosIndexBuffTensor, sorted, return_inverse);
        torch::Tensor UniqueTensor = std::get<0>(out_results);
        int outvalidbatch = UniqueTensor.size(0);
        outBuffStartNew = UniqueTensor.data_ptr<int>();
        *hCounterPerBatch = outvalidbatch;
#else
        thrust::sort(thrust::device, outPosIndexBuffPerBatch, outPosIndexBuffPerBatch + *hCounterPerBatch);
        void*  pickingSpace = NULL;
        size_t pickingSize = 0;
        checkCudaErrors_voxelv1(cudaMemset(dCounterPerBatch, 0, sizeof(int)));

        cub::DeviceSelect::Unique(pickingSpace, pickingSize, outPosIndexBuffPerBatch, outPosIndexBuffPerBatch, dCounterPerBatch, *hCounterPerBatch);
        checkCudaErrors_voxelv1(cudaMalloc(&pickingSpace, pickingSize));
        cub::DeviceSelect::Unique(pickingSpace, pickingSize, outPosIndexBuffPerBatch, outPosIndexBuffPerBatch, dCounterPerBatch, *hCounterPerBatch);
        checkCudaErrors_voxelv1(cudaFree(pickingSpace));
        outBuffStartNew = outPosIndexBuffPerBatch;
        checkCudaErrors_voxelv1(cudaMemcpy(hCounterPerBatch, dCounterPerBatch, sizeof(int), cudaMemcpyDeviceToHost));
#endif

        int valid_out_num = *hCounterPerBatch;
        if(valid_out_num<1) continue;

        float stride = 1.0;
        if(valid_out_num>maxValidOutput)
        {
            stride = (float)valid_out_num / (float)maxValidOutput;
            *hCounterPerBatch = maxValidOutput;
        }
        //compute OutpuCoords
        checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, OutputCoords, 0, *hCounterPerBatch));
        minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
        OutputCoords<<<minGridSize, blockSize>>>(outCoordsPerBatch, outBuffStartNew, GridX, GridY, GridZ, maxValidOutput, valid_out_num, *hCounterPerBatch, stride, i); // order: ZYX
        // valid_out_num = *hCounterPerBatch;


        //CSR: row
        checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeRowOffsetOA, 0, *hCounterPerBatch));
        minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
        computeRowOffsetOA<<<minGridSize, blockSize>>>(outCoordsPerBatch, TensorRowOffsetPtrPerBatch, GridY, *hCounterPerBatch);

        //CSR: cumsum
        int* TensorRowOffsetPtr_new = NULL;
#ifdef USE_TORCH
        torch::Tensor cumsumTensor;
        auto TensorRowOffsetPtrTensor = torch::from_blob(TensorRowOffsetPtrPerBatch, {batchRowOffset, }, options);
        cumsumTensor = at::_cumsum(TensorRowOffsetPtrTensor, 0);
        TensorRowOffsetPtr_new = cumsumTensor.data_ptr<int>();
#else
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, TensorRowOffsetPtrPerBatch,
                                      TensorRowOffsetPtrPerBatch, batchRowOffset, 0, false);
        checkCudaErrors_voxelv1(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, TensorRowOffsetPtrPerBatch,
                                      TensorRowOffsetPtrPerBatch, batchRowOffset, 0, false);
        checkCudaErrors_voxelv1(cudaFree(d_temp_storage));
        TensorRowOffsetPtr_new = TensorRowOffsetPtrPerBatch;
#endif

        //CSR: cols
        checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeColumnIdx, 0, *hCounterPerBatch));
        minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
        computeColumnIdx<<<minGridSize, blockSize>>>(TensorColumnsPtrPerBatch, TensorRowOffsetPtr_new, outCoordsPerBatch, GridY, GridX, *hCounterPerBatch);


        //Voxel
        checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GetVoxelPerBatchFP16, 0, numValidPointsInputPerBatch));
        minGridSize = std::min(minGridSize, DivUp(numValidPointsInputPerBatch, blockSize));

        GetVoxelPerBatchFP16<<<minGridSize, blockSize>>>(inPointsPerBatch, outVoxelsPerBatch, outVoxelNumPerBatch,
                                                         TensorRowOffsetPtr_new, TensorColumnsPtrPerBatch,
                                                         GridX, GridY, GridZ,
                                                         RangeMinX, RangeMinY, RangeMinZ,
                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                         inCols, outCols, numValidPointsInputPerBatch, maxPointsPerVoxel);
        if(cluster_offset!=0)
        {
            checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtendVoxelClusterOffsetPerBatchFP16, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            ExtendVoxelClusterOffsetPerBatchFP16<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outVoxelNumPerBatch, *hCounterPerBatch,
                                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                         outCols, write_offset_cluster, maxPointsPerVoxel);
        }
        if(center_offset!=0)
        {
            checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtendVoxelCenterOffsetPerBatchFP16, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            ExtendVoxelCenterOffsetPerBatchFP16<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outCoordsPerBatch,
                                                                         outVoxelNumPerBatch, *hCounterPerBatch,
                                                                         RangeMinX, RangeMinY, RangeMinZ,
                                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                         outCols, write_offset_center, maxPointsPerVoxel);
        }

        if(supplement!=0){
            //Voxel supplement
            checkCudaErrors_voxelv1(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GetVoxelSupplementPerBatchFP16, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            GetVoxelSupplementPerBatchFP16<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outVoxelNumPerBatch, *hCounterPerBatch, outCols, maxPointsPerVoxel);
        }

    }
    //for op output validPosNum
    checkCudaErrors_voxelv1(cudaMemcpy(dCounter, numvalidPosOut.data(), sizeof(int) * batchSize, cudaMemcpyHostToDevice));

}
} // namespace