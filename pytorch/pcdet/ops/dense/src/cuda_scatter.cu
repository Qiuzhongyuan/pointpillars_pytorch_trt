#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda_scatter.h"

namespace NAMESPACE
{
extern "C" __global__ void Scatter(const float *features_rw, const int *indices_rw, float *output_rw, 
                                    int spatialShape0, int spatialShape1, int spatialShape2,
                                    int num_voxels, int num_features)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_voxels; i += stride)
    {
        int4 coor = reinterpret_cast<const int4*>(indices_rw)[i];
        int output_vol = spatialShape0 * spatialShape1 * spatialShape2;

        //remove init -1.
        if(coor.x < 0 || coor.y < 0 || coor.z < 0 || coor.w < 0) continue;

        float *outPerBatch = output_rw + coor.x * num_features * output_vol;
        int offset = coor.y * spatialShape1 * spatialShape2 + coor.z * spatialShape2 + coor.w;

        for(int j = 0; j < num_features; ++j)
            outPerBatch[j * output_vol + offset] = features_rw[i * num_features + j];
	}

}

void cuda_scatter(const float *features_rw, const int *indices_rw,  float *output_rw, std::vector<int> spatialShape_rw,
                int num_voxels, int num_features)
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Scatter, 0, num_voxels));
    minGridSize = std::min(minGridSize, DivUp(num_voxels, blockSize));

    Scatter<<<minGridSize, blockSize>>>(features_rw, indices_rw, output_rw, spatialShape_rw[0], spatialShape_rw[1], spatialShape_rw[2], num_voxels, num_features);
    //cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    checkCudaErrors(error);

}


extern "C" __global__ void Scatter_Backward(const float *features_rw, const int *indices_rw,
                                            float *output_rw, int oX, int oY, int oZ,
                                            int num_voxels, int num_features)
{
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_voxels; i += stride)
    {
        int bs = indices_rw[i * 4];
        int x  = indices_rw[i * 4 + 1];
        int y  = indices_rw[i * 4 + 2];
        int z  = indices_rw[i * 4 + 3];

        //remove init -1.
        if(bs<0 || x<0 || y<0 || z<0) continue;

        // out shape: (bs, c, x, y, z)
        int output_vol = oX*oY*oZ;
        const float *inPerBatch = features_rw + bs * num_features * output_vol;
        int offset = x * oY * oZ + y * oZ + z;

        #pragma unroll
        for(int j = 0; j < num_features; ++j)
            output_rw[i * num_features + j] = inPerBatch[j * output_vol + offset];
	}


}
 
void cuda_scatter_backward(const float *features_rw, const int *indices_rw,  float *output_rw,
                            std::vector<int> spatialShape_rw, int num_voxels, int num_features)
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Scatter, 0, num_voxels));
    minGridSize = std::min(minGridSize, DivUp(num_voxels, blockSize));

    Scatter_Backward<<<minGridSize, blockSize>>>(features_rw, indices_rw, output_rw, spatialShape_rw[0], spatialShape_rw[1], spatialShape_rw[2], num_voxels, num_features);
    //cudaDeviceSynchronize();

}
 
}//namespace