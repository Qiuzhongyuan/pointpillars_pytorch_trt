#include "cuda_tookit.h"


__global__
void ConvertFloat2Half(float* in, __half* out, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num; i += stride)
    {
        out[i] = __float2half(in[i]);
    }
}


void convertFP32ToFP16(float* in, __half* out, int num)
{

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertFloat2Half, 0, num);
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

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ConvertHalf2Float, 0, num);
    ConvertHalf2Float<<<minGridSize, blockSize>>>(in, out, num);
}




__global__
void print_half_kernel(const __half* input, int stride, int rows, int cols)
{
    for(int i = 0;i < rows; ++i)
    {
        for(int j = 0;j < stride; ++j)
        {
            printf("%f ", __half2float(input[cols * i + j]));
        }
        printf("\n");
    }

}


void print_half(const __half* input, int stride, int rows, int cols)
{
    cudaDeviceSynchronize();
    print_half_kernel<<<1,1>>>(input, stride, rows, cols);
    cudaDeviceSynchronize();

}


__global__
void print_float_kernel(const float* input, int stride, int rows, int cols)
{
    for(int i = 0;i < rows; ++i)
    {
        for(int j = 0;j < stride; ++j)
        {
            printf("%f ", input[cols * i + j]);
        }
        printf("\n");
    }

}


void print_float(const float* input, int stride, int rows, int cols)
{
    cudaDeviceSynchronize();
    print_float_kernel<<<1,1>>>(input, stride, rows, cols);
    cudaDeviceSynchronize();

}





__global__
void compare_half_float_kernel(const float* input_float, const __half* input_half, int stride, int rows, int cols)
{
    float sum = 0.0;
    for(int i = 0;i < rows; ++i)
    {
        for(int j = 0;j < stride; ++j)
        {
            sum += abs(__half2float(input_half[cols * i + j]) - input_float[cols * i + j]);
        }
        
    }
    printf("\ncompare_half_float sum value: %f\n", sum);

}


void compare_half_float(const float* input_float, const __half* input_half, int stride, int rows, int cols)
{
    cudaDeviceSynchronize();
    compare_half_float_kernel<<<1,1>>>(input_float, input_half, stride, rows, cols);
    cudaDeviceSynchronize();

}