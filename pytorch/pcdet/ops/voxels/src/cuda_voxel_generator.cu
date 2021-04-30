#include "cuda_voxel_generator.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "cub-1.8.0/cub/cub.cuh"

namespace NAMESPACE
{

__global__
void OutputCoords(int* outCoor, int* Index,
                      int oX, int oY, int oZ,
                      int maxValidOutput, int realNum, int N,
                      float sample_stride, int batch_idx)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += stride)
    {
        int idxS;
        if(realNum<=maxValidOutput)
            idxS = i;
        else{
            float idFLT = i * sample_stride;
            idxS = __float2int_rd(idFLT);
        }
        int id = Index[idxS];
        int4* outBuff = reinterpret_cast<int4*>(outCoor);

        int4 coor;
        coor.x = batch_idx;
        coor.y = (id % (oY * oZ)) % oZ;
        coor.z = (id % (oY * oZ)) / oZ;
        coor.w =  id / (oY * oZ);
        outBuff[i] = coor;

    }
}

__global__
void computeRowOffsetOA(int* outCoords, int* RO, int oY, int N)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += stride)
    {
        int4 coor = reinterpret_cast<int4*>(outCoords)[i]; // bs,ZYX
        int row = coor.w * oY + coor.z;
#ifdef USE_TORCH
        atomicAdd(RO + row + 1, 1);
#else
        atomicAdd(RO + row, 1);
#endif
    }
}

__global__
void computeColumnIdx(int* columnIdx, int*RO, int* outCoords,
                      int oY, int oX, int N)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += stride)
    {
        int4 coor  = reinterpret_cast<int4*>(outCoords)[i];
        columnIdx[i] = coor.y;
    }
}

__global__
void TraverseInputPointsPerBatch(const float* inPoints, int numPointsPerBatchInput,
                                 int* outPosIndexBuff, int* dCounter,
                                 int GridX, int GridY, int GridZ,
                                 float RangeMinX, float RangeMinY, float RangeMinZ,
                                 float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                                 int inCols, const int numValidPointsInput)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidPointsInput; i += stride)
    {
        const float *point = inPoints + i * inCols;
        int bs = __float2int_rd(point[0] + DELTA);
        int x  = __float2int_rd((point[1] - RangeMinX) / VoxelSizeX);
        int y  = __float2int_rd((point[2] - RangeMinY) / VoxelSizeY);
        int z  = __float2int_rd((point[3] - RangeMinZ) / VoxelSizeZ);
        if(bs<0 || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        int pos = atomicAdd(dCounter, 1);
        outPosIndexBuff[pos] = x * GridZ * GridY + y * GridZ + z; //XYZ
    }

}

__global__
void GetVoxelPerBatch(const float* inPoints, float* outVoxels, int* numPerVoxel,
                         int* RO, int* CI,
                         int GridX, int GridY, int GridZ,
                         float RangeMinX, float RangeMinY, float RangeMinZ,
                         float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                         int inCols, int outCols, int numValidPointsInput, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidPointsInput; i += stride)
    {
        const float *point = inPoints + i * inCols;
        int bs = __float2int_rd(point[0] + DELTA);
        int x  = __float2int_rd((point[1] - RangeMinX) / VoxelSizeX);
        int y  = __float2int_rd((point[2] - RangeMinY) / VoxelSizeY);
        int z  = __float2int_rd((point[3] - RangeMinZ) / VoxelSizeZ);

        if(bs<0 || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        const int row = x * GridY + y;
        int offset = Access(RO, CI, row, z);
        if(offset<0) continue;

        float* voxel = outVoxels + offset * outCols * maxPointsPerVoxel;
        int old_num = atomicAdd(&numPerVoxel[offset], 1);

        if(old_num<maxPointsPerVoxel)
        {
            float* voxel_temp = voxel + old_num * outCols;
            #pragma unroll
            for(int j = 0; j < inCols-1; ++j) voxel_temp[j] = point[j+1];
        }
        else{
            atomicSub(&numPerVoxel[offset], 1);
        }

    }
}

__global__
void GetVoxelSupplementPerBatch(float* Voxels, int* numPerVoxel, int numValidVoxelsOutput, int cols, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        if(numpoints<maxPointsPerVoxel){
            float* voxel = Voxels + i * cols * maxPointsPerVoxel;
            #pragma unroll
            for(int j = numpoints; j < maxPointsPerVoxel; ++j)
            {
                float* voxel_temp = voxel + cols * j;
                #pragma unroll
                for(int k = 0; k < cols; ++k) voxel_temp[k] = voxel[k];
            }
        }
    }
}

__global__
void ExtendVoxelClusterOffsetPerBatch(float* Voxels, int* numPerVoxel, int numValidVoxelsOutput,
                                      float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                                      int cols, int write_offset, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        if(numpoints<2) continue;
        float* voxel = Voxels + i * cols * maxPointsPerVoxel;
        float resolution[] = {VoxelSizeX, VoxelSizeY, VoxelSizeZ};
        for(int j = 0; j < 3; ++j){
            float sum=0.0;
            for(int k = 0; k < numpoints; ++k) sum += voxel[k*cols + j];

            float mean_v = sum / numpoints;
            for(int k = 0; k < numpoints; ++k)
                voxel[k*cols + write_offset + j] = (voxel[k*cols + j] - mean_v) / resolution[j];
        }
    }
}

__global__
void ExtendVoxelCenterOffsetPerBatch(float* Voxels, int* Coords, int* numPerVoxel, int numValidVoxelsOutput,
                                     float RangeMinX, float RangeMinY, float RangeMinZ,
                                     float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                                     int cols, int write_offset, int maxPointsPerVoxel)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numValidVoxelsOutput; i += stride)
    {
        int numpoints = numPerVoxel[i];
        float* voxel = Voxels + i * cols * maxPointsPerVoxel;
        int* coor  = Coords + i*4; //bs,zyx

        float resolution[] = {VoxelSizeX, VoxelSizeY, VoxelSizeZ};
        float rangemin[] = {RangeMinX, RangeMinY, RangeMinZ};
        for(int j = 0; j < numpoints; ++j){
            for(int k = 0; k < 3; ++k){
                float center = ((float)coor[3-k] + 0.5) * resolution[k] + rangemin[k];
                voxel[j*cols + write_offset + k] = (voxel[j*cols + k] - center) / resolution[k];
            }
        }
    }
}

void cuda_points_to_voxel(const float* inPoints, const int* devNumValidPointsInput, int* outPosIndexBuff,
                         int* outCoords, float* outVoxels, int* outVoxelNum, int* dCounter,
                         int GridX, int GridY, int GridZ,
                         float RangeMinX, float RangeMinY, float RangeMinZ,
                         float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                         int batchSize, int inPointsNum, int inCols, int maxValidOutput,
                         int maxPointsPerVoxel, int* TensorRowOffsetPtr, int* TensorColumnsPtr,
                         int cluster_offset, int center_offset, int supplement, int cuda_idx){

    std::vector<int> numValidPointsInput(batchSize);
    const int numPointsPerBatchInput = inPointsNum / batchSize;
    checkCudaErrors(cudaMemcpy(numValidPointsInput.data(), devNumValidPointsInput, sizeof(int) * batchSize, cudaMemcpyDeviceToHost));
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
        const float* inPointsPerBatch = inPoints + i * numPointsPerBatchInput * inCols;
        int* outPosIndexBuffPerBatch = outPosIndexBuff + i * numPointsPerBatchInput;
        int* outCoordsPerBatch = outCoords + i * maxValidOutput * 4;
        float* outVoxelsPerBatch = outVoxels + i * maxValidOutput * maxPointsPerVoxel * outCols;
        int* outVoxelNumPerBatch = outVoxelNum + i * maxValidOutput;
        int* TensorRowOffsetPtrPerBatch = TensorRowOffsetPtr + i * batchRowOffset;
        int* TensorColumnsPtrPerBatch = TensorColumnsPtr + i * maxValidOutput;
        int* dCounterPerBatch = dCounter + i;
        int* hCounterPerBatch = numvalidPosOut.data() + i;
        int numValidPointsInputPerBatch = numValidPointsInput[i];

        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, TraverseInputPointsPerBatch, 0, numValidPointsInputPerBatch));
        minGridSize = std::min(minGridSize, DivUp(numValidPointsInputPerBatch, blockSize));

        // std::cout<<"blockSize  " << blockSize << "  minGridSize  "<< minGridSize << "  inPointsNum: " << numValidPointsInputPerBatch << std::endl;
        TraverseInputPointsPerBatch<<<minGridSize, blockSize>>>(inPointsPerBatch, numPointsPerBatchInput,
                                                                 outPosIndexBuffPerBatch, dCounterPerBatch,
                                                                 GridX, GridY, GridZ,
                                                                 RangeMinX, RangeMinY, RangeMinZ,
                                                                 VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                 inCols, numValidPointsInputPerBatch);

        checkCudaErrors(cudaMemcpy(hCounterPerBatch, dCounterPerBatch, sizeof(int), cudaMemcpyDeviceToHost));
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
        checkCudaErrors(cudaMemset(dCounterPerBatch, 0, sizeof(int)));

        cub::DeviceSelect::Unique(pickingSpace, pickingSize, outPosIndexBuffPerBatch, outPosIndexBuffPerBatch, dCounterPerBatch, *hCounterPerBatch);
        checkCudaErrors(cudaMalloc(&pickingSpace, pickingSize));
        cub::DeviceSelect::Unique(pickingSpace, pickingSize, outPosIndexBuffPerBatch, outPosIndexBuffPerBatch, dCounterPerBatch, *hCounterPerBatch);
        checkCudaErrors(cudaFree(pickingSpace));
        outBuffStartNew = outPosIndexBuffPerBatch;
        checkCudaErrors(cudaMemcpy(hCounterPerBatch, dCounterPerBatch, sizeof(int), cudaMemcpyDeviceToHost));
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
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, OutputCoords, 0, *hCounterPerBatch));
        minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
        OutputCoords<<<minGridSize, blockSize>>>(outCoordsPerBatch, outBuffStartNew, GridX, GridY, GridZ, maxValidOutput, valid_out_num, *hCounterPerBatch, stride, i); // order: ZYX
        // valid_out_num = *hCounterPerBatch;

        //CSR: row
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeRowOffsetOA, 0, *hCounterPerBatch));
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
        checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, TensorRowOffsetPtrPerBatch,
                                      TensorRowOffsetPtrPerBatch, batchRowOffset, 0, false);
        checkCudaErrors(cudaFree(d_temp_storage));
        TensorRowOffsetPtr_new = TensorRowOffsetPtrPerBatch;
#endif
        //CSR: cols
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeColumnIdx, 0, *hCounterPerBatch));
        minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
        computeColumnIdx<<<minGridSize, blockSize>>>(TensorColumnsPtrPerBatch, TensorRowOffsetPtr_new, outCoordsPerBatch, GridY, GridX, *hCounterPerBatch);

        //Voxel
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GetVoxelPerBatch, 0, numValidPointsInputPerBatch));
        minGridSize = std::min(minGridSize, DivUp(numValidPointsInputPerBatch, blockSize));
        GetVoxelPerBatch<<<minGridSize, blockSize>>>(inPointsPerBatch, outVoxelsPerBatch, outVoxelNumPerBatch,
                                                         TensorRowOffsetPtr_new, TensorColumnsPtrPerBatch,
                                                         GridX, GridY, GridZ,
                                                         RangeMinX, RangeMinY, RangeMinZ,
                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                         inCols, outCols, numValidPointsInputPerBatch, maxPointsPerVoxel);
        if(cluster_offset!=0)
        {
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtendVoxelClusterOffsetPerBatch, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            ExtendVoxelClusterOffsetPerBatch<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outVoxelNumPerBatch, *hCounterPerBatch,
                                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                         outCols, write_offset_cluster, maxPointsPerVoxel);
        }

        if(center_offset!=0)
        {
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtendVoxelCenterOffsetPerBatch, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            ExtendVoxelCenterOffsetPerBatch<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outCoordsPerBatch,
                                                                         outVoxelNumPerBatch, *hCounterPerBatch,
                                                                         RangeMinX, RangeMinY, RangeMinZ,
                                                                         VoxelSizeX, VoxelSizeY, VoxelSizeZ,
                                                                         outCols, write_offset_center, maxPointsPerVoxel);
        }
        if(supplement!=0){
            //Voxel supplement
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GetVoxelSupplementPerBatch, 0, *hCounterPerBatch));
            minGridSize = std::min(minGridSize, DivUp(*hCounterPerBatch, blockSize));
            GetVoxelSupplementPerBatch<<<minGridSize, blockSize>>>(outVoxelsPerBatch, outVoxelNumPerBatch, *hCounterPerBatch, outCols, maxPointsPerVoxel);
        }
    }
    //for op output validPosNum
    checkCudaErrors(cudaMemcpy(dCounter, numvalidPosOut.data(), sizeof(int) * batchSize, cudaMemcpyHostToDevice));
}
} // namespace
