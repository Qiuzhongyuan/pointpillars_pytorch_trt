#include "cuda_tookit.h"


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

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertFloat2Half);
    ConvertFloat2Half<<<minGridSize, blockSize>>>(in, out, num);
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

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertHalf2Float);
    ConvertHalf2Float<<<minGridSize, blockSize>>>(in, out, num);
}