#include "gpu_hash_map.h"
#include <iostream>
#define DELTA 0.01

namespace NAMESPACE
{
__global__
void InitializeMapEntries(const float* __restrict__  points,
                          const int* __restrict__  numValid,
                          int* validOutputVoxels,
                          float RangeMinX, float RangeMinY, float RangeMinZ,
                          float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                          int GridX, int GridY, int GridZ,
                          int* map, HashEntry* list,
                          int batchSize, int cols, int maxInputPoints, int maxOutputVoxels,
                          int* map_addr, const int value_map_z)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;;

    const float Intervel_Oz = (float)GridZ / (float)value_map_z + 0.001;

    for(int i = idx ; i < batchSize * maxInputPoints; i += stride)
    {   
        const int curBatch = i / maxInputPoints;
        const int curPoints = i % maxInputPoints;
        if(curPoints >= numValid[curBatch]) continue;

        const float* cur_point = points + i * cols;

        int bs = __float2int_rd(cur_point[0] + DELTA);
        int x  = __float2int_rd((cur_point[1] - RangeMinX) / VoxelSizeX);
        int y  = __float2int_rd((cur_point[2] - RangeMinY) / VoxelSizeY);
        int z  = __float2int_rd((cur_point[3] - RangeMinZ) / VoxelSizeZ);

        HashEntry* entry = list + i;

        if(bs!=curBatch || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        int4 coor;
        coor.x = bs;
        coor.y = z;
        coor.z = y;
        coor.w = x;

        int hash_idx = bs * GridX * GridY * value_map_z + (int)(z / Intervel_Oz) * GridX * GridY + x * GridY + y;

        entry -> intCoor = coor;
        int* address = map + hash_idx;
        int newVal = i;
        int curVal = *address;
        int assumed;

        do {
            assumed = curVal;
            curVal = atomicCAS(address, assumed, newVal);
        } while (assumed != curVal);

        entry -> nextId = curVal;

        if(curVal == -1)
        {
            int old_num = atomicAdd(validOutputVoxels + bs, 1);
            if(old_num < maxOutputVoxels)
                map_addr[bs * maxOutputVoxels + old_num] = hash_idx;
        }
    }
}


__global__
void InitializeMapEntriesFp16(const __half* __restrict__  points,
                              const int* __restrict__  numValid,
                              int* validOutputVoxels,
                              float RangeMinX, float RangeMinY, float RangeMinZ,
                              float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                              int GridX, int GridY, int GridZ,
                              int* map, HashEntry* list,
                              int batchSize, int cols, int maxInputPoints, int maxOutputVoxels,
                              int* map_addr, const int value_map_z)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;;

    const float Intervel_Oz = (float)GridZ / (float)value_map_z + 0.001;

    for(int i = idx ; i < batchSize * maxInputPoints; i += stride)
    {
        const int curBatch = i / maxInputPoints;
        const int curPoints = i % maxInputPoints;
        if(curPoints >= numValid[curBatch]) continue;

        const __half* cur_point = points + i * cols;

        int bs = __float2int_rd(__half2float(cur_point[0]) + DELTA);
        int x  = __float2int_rd((__half2float(cur_point[1]) - RangeMinX) / VoxelSizeX);
        int y  = __float2int_rd((__half2float(cur_point[2]) - RangeMinY) / VoxelSizeY);
        int z  = __float2int_rd((__half2float(cur_point[3]) - RangeMinZ) / VoxelSizeZ);

        HashEntry* entry = list + i;

        if(bs!=curBatch || x<0 || y<0 || z<0 || x>=GridX || y>=GridY || z>=GridZ) continue;

        int4 coor;
        coor.x = bs;
        coor.y = z;
        coor.z = y;
        coor.w = x;

        int hash_idx = bs * GridX * GridY * value_map_z + (int)(z / Intervel_Oz) * GridX * GridY + x * GridY + y;

        entry -> intCoor = coor;
        int* address = map + hash_idx;
        int newVal = i;
        int curVal = *address;
        int assumed;

        do {
            assumed = curVal;
            curVal = atomicCAS(address, assumed, newVal);
        } while (assumed != curVal);

        entry -> nextId = curVal;

        if(curVal == -1)
        {
            int old_num = atomicAdd(validOutputVoxels + bs, 1);
            if(old_num < maxOutputVoxels)
                map_addr[bs * maxOutputVoxels + old_num] = hash_idx;
        }
    }
}

void InitializeHashMap(const float* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
        int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
        std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z)
{

    int num_thread;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the 
                    // maximum occupancy for a full device launch

    num_thread = batchSize * maxInputPoints;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, InitializeMapEntries, 0, num_thread);
    minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
    InitializeMapEntries<<<minGridSize, blockSize>>>(points, numValid, validOutputVoxels,
                                                     point_cloud_range[0], point_cloud_range[1], point_cloud_range[2],
                                                     voxel_size       [0], voxel_size       [1], voxel_size       [2],
                                                     grid_size        [0], grid_size        [1], grid_size        [2],
                                                     map, list, batchSize, inCols, maxInputPoints,
                                                     maxOutputVoxels, map_addr, value_map_z);
}

void InitializeHashMapFp16(const __half* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
        int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
        std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z)
{

    int num_thread;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                    // maximum occupancy for a full device launch

    num_thread = batchSize * maxInputPoints;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, InitializeMapEntriesFp16, 0, num_thread);
    minGridSize = std::min(minGridSize, DivUp(num_thread, blockSize));
    InitializeMapEntriesFp16<<<minGridSize, blockSize>>>(points, numValid, validOutputVoxels,
                                                     point_cloud_range[0], point_cloud_range[1], point_cloud_range[2],
                                                     voxel_size       [0], voxel_size       [1], voxel_size       [2],
                                                     grid_size        [0], grid_size        [1], grid_size        [2],
                                                     map, list, batchSize, inCols, maxInputPoints,
                                                     maxOutputVoxels, map_addr, value_map_z);
}
}