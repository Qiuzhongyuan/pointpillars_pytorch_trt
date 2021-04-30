#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

void convertFP32ToFP16(float* in, __half* out, int num);
void convertFP16ToFP32(__half* in, float* out, int num);