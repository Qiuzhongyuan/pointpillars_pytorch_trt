#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

void convertFP32ToFP16(float* in, __half* out, int num);
void convertFP16ToFP32(__half* in, float* out, int num);


void print_half(const __half* input, int stride, int rows, int cols);

void print_float(const float* input, int stride, int rows, int cols);

void compare_half_float(const float* input_float, const __half* input_half, int stride, int rows, int cols);