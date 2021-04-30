#ifndef VOXELV1
#define VOXELV1
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <string>
#define NAMESPACE VoxelGeneratorV1Space
#define DEBUG

#ifdef USE_TORCH
#include <torch/serialize/tensor.h>
#endif

#if defined(DEBUG) || defined(_DEBUG)
#define checkCudaErrors_voxelv1(x) do { \
  cudaError_t result = x;\
  if (result != cudaSuccess) { \
    fprintf(stderr, "CUDA Runtime Error at %s\nerror type: %s\nerror path: %s:%d\n", #x, cudaGetErrorString(result), __FILE__, __LINE__); \
    exit(-1);\
  } \
} while(0)
#else
    #define checkCudaErrors_voxelv1(x) cudaError_t result = x
#endif

namespace NAMESPACE
{
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__
void OutputCoords(int* outCoor, int* Index, int oX, int oY, int oZ, int maxValidOutput,
                  int realNum, int N, float sample_stride, int batch_idx);


__global__
void computeRowOffsetOA(int* outCoords, int* RO, int oY, int N);

__global__
void computeColumnIdx(int* columnIdx, int*RO, int* outCoords, int oY, int oX, int N);

__inline__ __device__
int Access(int*& RO, int*& CI, int row, int col)
{
    int offsetS = RO[row];
    int offsetE = RO[row + 1];
    int counter = 0;
    bool find = false;
    for(int i = offsetS; i < offsetE; ++i)
    {
        if(CI[i] != col)
        {
            counter ++;
        }
        else
        {
            find = true;
            break;
        }
    }
    return find ?  (offsetS + counter) : -1;
}



void cuda_points_to_voxel(const float* inPoints, const int* devNumValidPointsInput, int* outPosIndexBuff,
                                 int* outCoords, float* outVoxels, int* outVoxelNum, int* dCounter,
                                 int GridX, int GridY, int GridZ,
                                 float RangeMinX, float RangeMinY, float RangeMinZ,
                                 float VoxelSizeX, float VoxelSizeY, float VoxelSizeZ,
                                 int batchSize, int inPointsNum, int inCols, int maxValidOutput,
                                 int maxPointsPerVoxel, int* TensorRowOffsetPtr, int* TensorColumnsPtr,
                                 int cluster_offset, int center_offset, int supplement, int cuda_idx);

void cuda_points_to_voxel_fp16(const __half* inPoints, const int* devNumValidPointsInput, int* outPosIndexBuff,
                                 int* outCoords, __half* outVoxels, int* outVoxelNum, int* dCounter,
                                 int GridX, int GridY, int GridZ,
                                 __half RangeMinX,  __half RangeMinY,  __half RangeMinZ,
                                 __half VoxelSizeX, __half VoxelSizeY, __half VoxelSizeZ,
                                 int batchSize, int inPointsNum, int inCols, int maxValidOutput,
                                 int maxPointsPerVoxel, int* TensorRowOffsetPtr, int* TensorColumnsPtr,
                                 int cluster_offset, int center_offset, int supplement, int cuda_idx);
}
#endif