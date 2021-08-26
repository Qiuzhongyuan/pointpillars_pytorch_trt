#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include "cuda_voxel_generator.h"


namespace NAMESPACE
{
__global__
void SupplementVoxelsKernel(float* outVoxels, int* devValidOutputPoints, int* dCounter,
                            int max_points, int maxValidOutput, int batchSize, int outCols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx ; i < batchSize * maxValidOutput * max_points; i += stride)
    {
        int cur_batch = i / (maxValidOutput * max_points);
        int cur_voxel = (i % (maxValidOutput * max_points)) / max_points;
        int cur_point = (i % (maxValidOutput * max_points)) % max_points;
        if(cur_voxel >= dCounter[cur_batch] || cur_point < devValidOutputPoints[cur_batch * maxValidOutput + cur_voxel]) continue;

        float* cur_outVoxel = outVoxels + (cur_batch * maxValidOutput + cur_voxel) * max_points * outCols;
        for(int j = 0 ; j < outCols; ++j)
        {
            cur_outVoxel[cur_point * outCols + j] = cur_outVoxel[j];
        }
    }
}


__global__
void ExtractValidOutVoxelKernel(const float* inPoints, int* map_addr, int* map, HashEntry* list, int* devValidOutputCoors, float* outVoxels,
                                int* outCoors, int* devValidOutputPoints, int batchSize, int inCols, int outCols, int maxInputPoints, int maxValidOutput,
                                int cluster_offset, int center_offset, int supplement, int max_points,
                                float vsizeX, float vsizeY, float vsizeZ,
                                float rangeMinX, float rangeMinY, float rangeMinZ)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx ; i < batchSize * maxValidOutput; i += stride)
    {
        const int curBatch = i / maxValidOutput;

        int hash_addr = map_addr[i];
        if(hash_addr<0) continue;

        int3* outCoorsBatch = ((int3*)outCoors) + curBatch * maxValidOutput;
        float* outVoxelsBatch = outVoxels + curBatch * maxValidOutput * max_points * outCols;
        int* devValidOutputPointsBatch = devValidOutputPoints + curBatch * maxValidOutput;

        int entry_idx = map[hash_addr];
        while(entry_idx >= 0) {
            HashEntry* entry = list + entry_idx;
            if((entry->intCoor).y >= 0)
            {
                int old_num = atomicAdd(devValidOutputCoors + curBatch, 1);
                if(old_num >= maxValidOutput)
                {
                    devValidOutputCoors[curBatch] = maxValidOutput;
                    break;
                }

                int3* outCoors_cur = outCoorsBatch + old_num;
                float* outVoxels_cur = outVoxelsBatch + old_num * outCols * max_points;

                outCoors_cur -> x = (entry->intCoor).x;
                outCoors_cur -> y = (entry->intCoor).y;
                outCoors_cur -> z = (entry->intCoor).z;

                int num = 1;
                for(int j = 0; j < inCols; ++j)
                    outVoxels_cur[j] = inPoints[entry_idx * inCols + j];

                int next_idx = entry->nextId;
                while(next_idx >= 0)
                {
                    HashEntry* entry_next = list + next_idx;
                    if((entry_next->intCoor).y == (entry->intCoor).y)
                    {
                        (entry_next->intCoor).y = -1;
                        if(num < max_points)
                        {
                            for(int j = 0; j < inCols; ++j)
                                outVoxels_cur[num * outCols + j] = inPoints[next_idx * inCols + j];
                            num += 1;
                        }
                    }
                    next_idx = entry_next -> nextId;
                }

                int offset = 0;
                if(cluster_offset != 0)
                {
                    offset = 3;

                    float sum = 0.0;
                    for(int k = 0; k < num; ++k)
                        sum += outVoxels_cur[k * outCols];
                    float mean = sum / (float)num;
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols   ] = (outVoxels_cur[k * outCols] - mean) / vsizeX;

                    sum = 0.0;
                    for(int k = 0; k < num; ++k)
                        sum += outVoxels_cur[k * outCols + 1];
                    mean = sum / (float)num;
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols + 1] = (outVoxels_cur[k * outCols + 1] - mean) / vsizeY;

                    sum = 0.0;
                    for(int k = 0; k < num; ++k)
                        sum += outVoxels_cur[k * outCols + 2];
                    mean = sum / (float)num;
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols + 2] = (outVoxels_cur[k * outCols + 2] - mean) / vsizeZ;

                }
                if(center_offset != 0)
                {
                    float center_x = rangeMinX + ((float)(outCoors_cur -> z) + 0.5) * vsizeX;
                    float center_y = rangeMinY + ((float)(outCoors_cur -> y) + 0.5) * vsizeY;
                    float center_z = rangeMinZ + ((float)(outCoors_cur -> x) + 0.5) * vsizeZ;
                    for(int k = 0; k < num; ++k)
                    {
                        outVoxels_cur[k * outCols + inCols     + offset] = (outVoxels_cur[k * outCols    ] - center_x) / vsizeX;
                        outVoxels_cur[k * outCols + inCols + 1 + offset] = (outVoxels_cur[k * outCols + 1] - center_y) / vsizeY;
                        outVoxels_cur[k * outCols + inCols + 2 + offset] = (outVoxels_cur[k * outCols + 2] - center_z) / vsizeZ;
                    }
                }

                devValidOutputPointsBatch[old_num] = num;
            }
            entry_idx = entry->nextId;
        }
    }
}


__global__
void SupplementVoxelsHalfKernel(__half* outVoxels, int* devValidOutputPoints, int* dCounter,
                            int max_points, int maxValidOutput, int batchSize, int outCols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx ; i < batchSize * maxValidOutput * max_points; i += stride)
    {
        int cur_batch = i / (maxValidOutput * max_points);
        int cur_voxel = (i % (maxValidOutput * max_points)) / max_points;
        int cur_point = (i % (maxValidOutput * max_points)) % max_points;
        if(cur_voxel >= dCounter[cur_batch] || cur_point < devValidOutputPoints[cur_batch * maxValidOutput + cur_voxel]) continue;

        __half* cur_outVoxel = outVoxels + (cur_batch * maxValidOutput + cur_voxel) * max_points * outCols;
        for(int j = 0 ; j < outCols; ++j)
        {
            cur_outVoxel[cur_point * outCols + j] = cur_outVoxel[j];
        }
    }
}


__global__
void ExtractValidOutVoxelHalfKernel(const __half* inPoints, int* map_addr, int* map, HashEntry* list, int* devValidOutputCoors, __half* outVoxels,
                                    int* outCoors, int* devValidOutputPoints, int batchSize, int inCols, int outCols, int maxInputPoints, int maxValidOutput,
                                    int cluster_offset, int center_offset, int supplement, int max_points,
                                    float vsizeX, float vsizeY, float vsizeZ,
                                    float rangeMinX, float rangeMinY, float rangeMinZ)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx ; i < batchSize * maxValidOutput; i += stride)
    {
        const int curBatch = i / maxValidOutput;

        int hash_addr = map_addr[i];
        if(hash_addr<0) continue;

        int3* outCoorsBatch = ((int3*)outCoors) + curBatch * maxValidOutput;
        __half* outVoxelsBatch = outVoxels + curBatch * maxValidOutput * max_points * outCols;
        int* devValidOutputPointsBatch = devValidOutputPoints + curBatch * maxValidOutput;

        int entry_idx = map[hash_addr];
        while(entry_idx >= 0) {
            HashEntry* entry = list + entry_idx;
            if((entry->intCoor).y >= 0)
            {
                int old_num = atomicAdd(devValidOutputCoors + curBatch, 1);
                if(old_num >= maxValidOutput)
                {
                    devValidOutputCoors[curBatch] = maxValidOutput;
                    break;
                }

                int3* outCoors_cur = outCoorsBatch + old_num;
                __half* outVoxels_cur = outVoxelsBatch + old_num * outCols * max_points;

                outCoors_cur -> x = (entry->intCoor).x;
                outCoors_cur -> y = (entry->intCoor).y;
                outCoors_cur -> z = (entry->intCoor).z;

                int num = 1;
                for(int j = 0; j < inCols; ++j)
                    outVoxels_cur[j] = inPoints[entry_idx * inCols + j];

                int next_idx = entry->nextId;
                while(next_idx >= 0)
                {
                    HashEntry* entry_next = list + next_idx;
                    if((entry_next->intCoor).y == (entry->intCoor).y)
                    {
                        (entry_next->intCoor).y = -1;
                        if(num < max_points)
                        {
                            for(int j = 0; j < inCols; ++j)
                                outVoxels_cur[num * outCols + j] = inPoints[next_idx * inCols + j];
                            num += 1;
                        }
                    }
                    next_idx = entry_next -> nextId;
                }

                int offset = 0;
                if(cluster_offset != 0)
                {
                    offset = 3;

                    __half sum = __float2half(0.0f);
                    for(int k = 0; k < num; ++k)
                        sum = __hadd(sum, outVoxels_cur[k * outCols]);
                    __half mean = __hmul(sum, __float2half(1 / (float)num));
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols] = __hmul(__hsub(outVoxels_cur[k * outCols], mean), __float2half(1 / vsizeX));

                    sum = __float2half(0.0f);
                    for(int k = 0; k < num; ++k)
                        sum = __hadd(sum, outVoxels_cur[k * outCols + 1]);
                    mean = __hmul(sum, __float2half(1 / (float)num));
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols + 1] = __hmul(__hsub(outVoxels_cur[k * outCols + 1], mean), __float2half(1 / vsizeY));

                    sum = __float2half(0.0f);
                    for(int k = 0; k < num; ++k)
                        sum = __hadd(sum, outVoxels_cur[k * outCols + 2]);
                    mean = __hmul(sum, __float2half(1 / (float)num));
                    for(int k = 0; k < num; ++k)
                        outVoxels_cur[k * outCols + inCols + 2] = __hmul(__hsub(outVoxels_cur[k * outCols + 2], mean), __float2half(1 / vsizeZ));

                }
                if(center_offset != 0)
                {
                    __half center_x = __float2half(rangeMinX + ((float)(outCoors_cur -> z) + 0.5) * vsizeX);
                    __half center_y = __float2half(rangeMinY + ((float)(outCoors_cur -> y) + 0.5) * vsizeY);
                    __half center_z = __float2half(rangeMinZ + ((float)(outCoors_cur -> x) + 0.5) * vsizeZ);
                    for(int k = 0; k < num; ++k)
                    {
                        outVoxels_cur[k * outCols + inCols     + offset] = __hmul(__hsub(outVoxels_cur[k * outCols    ], center_x), __float2half(1 / vsizeX));
                        outVoxels_cur[k * outCols + inCols + 1 + offset] = __hmul(__hsub(outVoxels_cur[k * outCols + 1], center_y), __float2half(1 / vsizeY));
                        outVoxels_cur[k * outCols + inCols + 2 + offset] = __hmul(__hsub(outVoxels_cur[k * outCols + 2], center_z), __float2half(1 / vsizeZ));
                    }
                }

                devValidOutputPointsBatch[old_num] = num;
            }
            entry_idx = entry->nextId;
        }
    }
}


void cuda_points_to_voxel(const float* inPoints, const int* devNumValidPointsInput,
                         int* outCoords, int* devValidOutputPoints, float* outVoxels, int* dCounter,
                         int* map_tensor_rw, int* addr_tensor_rw, HashEntry* list,
                         std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size,
                         int batchSize, int inPointsNum, int inCols, int outCols,
                         int cluster_offset, int center_offset, int supplement,
                         int maxValidOutput, int max_points, const int value_map_z)
{
    const int maxInputPoints = inPointsNum / batchSize;

    InitializeHashMap(inPoints, devNumValidPointsInput, dCounter, map_tensor_rw, list, addr_tensor_rw,
                      batchSize, maxInputPoints, maxValidOutput, inCols, point_cloud_range, voxel_size, grid_size, value_map_z);

    int num_thread;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                    // maximum occupancy for a full device launch

    checkCudaErrors(cudaMemset(dCounter, 0, sizeof(int) * batchSize));

    num_thread = batchSize * maxValidOutput;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtractValidOutVoxelKernel, 0, num_thread);
    minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
    ExtractValidOutVoxelKernel<<<minGridSize, blockSize>>>(inPoints, addr_tensor_rw, map_tensor_rw, list, dCounter, outVoxels,
                                                            outCoords, devValidOutputPoints, batchSize, inCols, outCols, maxInputPoints, maxValidOutput,
                                                            cluster_offset, center_offset, supplement, max_points,
                                                            voxel_size[0], voxel_size[1],voxel_size[2],
                                                            point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]);

    if(supplement != 0)
    {
        num_thread = batchSize * maxValidOutput * max_points;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, SupplementVoxelsKernel, 0, num_thread);
        minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
        SupplementVoxelsKernel<<<minGridSize, blockSize>>>(outVoxels, devValidOutputPoints, dCounter, max_points, maxValidOutput, batchSize, outCols);
    }

}


void cuda_points_to_voxel_fp16(const __half* inPoints, const int* devNumValidPointsInput,
                         int* outCoords, int* devValidOutputPoints, __half* outVoxels, int* dCounter,
                         int* map_tensor_rw, int* addr_tensor_rw, HashEntry* list,
                         std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size,
                         int batchSize, int inPointsNum, int inCols, int outCols,
                         int cluster_offset, int center_offset, int supplement,
                         int maxValidOutput, int max_points, const int value_map_z)
{
    const int maxInputPoints = inPointsNum / batchSize;

    InitializeHashMapFp16(inPoints, devNumValidPointsInput, dCounter, map_tensor_rw, list, addr_tensor_rw,
                      batchSize, maxInputPoints, maxValidOutput, inCols, point_cloud_range, voxel_size, grid_size, value_map_z);

    int num_thread;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                    // maximum occupancy for a full device launch

    checkCudaErrors(cudaMemset(dCounter, 0, sizeof(int) * batchSize));

    num_thread = batchSize * maxValidOutput;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ExtractValidOutVoxelHalfKernel, 0, num_thread);
    minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
    ExtractValidOutVoxelHalfKernel<<<minGridSize, blockSize>>>(inPoints, addr_tensor_rw, map_tensor_rw, list, dCounter, outVoxels,
                                                            outCoords, devValidOutputPoints, batchSize, inCols, outCols, maxInputPoints, maxValidOutput,
                                                            cluster_offset, center_offset, supplement, max_points,
                                                            voxel_size[0], voxel_size[1],voxel_size[2],
                                                            point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]);

    if(supplement != 0)
    {
        num_thread = batchSize * maxValidOutput * max_points;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, SupplementVoxelsHalfKernel, 0, num_thread);
        minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
        SupplementVoxelsHalfKernel<<<minGridSize, blockSize>>>(outVoxels, devValidOutputPoints, dCounter, max_points, maxValidOutput, batchSize, outCols);
    }

}

} // namespace