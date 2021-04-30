/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "VoxelGeneratorV1.h"
#include "serialize.hpp"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

const int voxelNum = 10000;

using namespace nvinfer1;
using nvinfer1::plugin::VoxelGeneratorV1;
using nvinfer1::plugin::VoxelGeneratorV1PluginCreator;

static const char* VoxelGeneratorV1_PLUGIN_VERSION{"001"};
static const char* VoxelGeneratorV1_PLUGIN_NAME{"VoxelGeneratorV1"};

PluginFieldCollection VoxelGeneratorV1PluginCreator::mFC{};
std::vector<PluginField> VoxelGeneratorV1PluginCreator::mPluginAttributes;


VoxelGeneratorV1::VoxelGeneratorV1(int batch_size, int max_num_points, int max_voxels, std::vector<float> point_cloud_range, std::vector<float> voxel_size, int center_offset, int cluster_offset, int supplement)
    : _batch_size(batch_size), _max_num_points(max_num_points), _max_voxels(max_voxels), _point_cloud_range(point_cloud_range), _voxel_size(voxel_size), _center_offset(center_offset),
     _cluster_offset(cluster_offset), _supplement(supplement)
{
}

VoxelGeneratorV1::VoxelGeneratorV1(const void* data, size_t length)
{
    deserialize_value(&data, &length, &_batch_size);
    deserialize_value(&data, &length, &_max_num_points);
    deserialize_value(&data, &length, &_max_voxels);
    deserialize_value(&data, &length, &_center_offset);
    deserialize_value(&data, &length, &_cluster_offset);
    deserialize_value(&data, &length, &_point_cloud_range);
    deserialize_value(&data, &length, &_voxel_size);
    deserialize_value(&data, &length, &_supplement);
}

VoxelGeneratorV1::~VoxelGeneratorV1()
{
}

int VoxelGeneratorV1::getNbOutputs() const
{
    return 4;
}

DimsExprs VoxelGeneratorV1::getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, nvinfer1::IExprBuilder& exprBuilder)
{
    int inCols = inputs[0].d[inputs[0].nbDims - 1]->getConstantValue();
    int numfeature = inCols - 1;
    int outfeature = numfeature;
    if (_center_offset!=0){
        outfeature+=3;
    }
    if(_cluster_offset!=0){
	    outfeature +=3;
    }
    if (index == 0){
        nvinfer1::DimsExprs voxelDims;
        voxelDims.nbDims = 3;
        voxelDims.d[0] = exprBuilder.constant(_max_voxels*_batch_size);
        voxelDims.d[1] = exprBuilder.constant(_max_num_points);
        voxelDims.d[2] = exprBuilder.constant(outfeature);
        return voxelDims;
    }
    else if (index == 1){
        nvinfer1::DimsExprs coorsDims;
        coorsDims.nbDims = 2;
        coorsDims.d[0] = exprBuilder.constant(_max_voxels*_batch_size);
        coorsDims.d[1] = exprBuilder.constant(4);
        return coorsDims;
    }
    else if (index == 2){
        nvinfer1::DimsExprs nppvDims;
        nppvDims.nbDims = 1;
        nppvDims.d[0] = exprBuilder.constant(_max_voxels*_batch_size);
        return nppvDims;
    }
    else if (index == 3){
        nvinfer1::DimsExprs voDims;
        voDims.nbDims = 1;
        voDims.d[0] = exprBuilder.constant(_batch_size);
        return voDims;
    }
}

int VoxelGeneratorV1::initialize()
{
    return STATUS_SUCCESS;
}

void VoxelGeneratorV1::terminate()
{
}

size_t VoxelGeneratorV1::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    int grid_size[3];
    for (int i=0; i < 3;i++){
        grid_size[i] = round((_point_cloud_range[3 + i] - _point_cloud_range[i]) / _voxel_size[i]);
    }
    int batchRowOffset = grid_size[0]*grid_size[1] + 1;

    Dims indim = inputs[0].dims;
    size_t N = indim.d[0];
    
    size_t totalSize = sizeof(int)*N + _batch_size*sizeof(int)*batchRowOffset + _batch_size*sizeof(int)*_max_voxels;
    return totalSize;
}

int VoxelGeneratorV1::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{    
    cudaDeviceSynchronize();
    //auto t1=std::chrono::steady_clock::now();

    /*if (DataType::kHALF == inputDesc->type){
        printf("-------------fp16--------------\n");
    }

    DataType pD = inputDesc[0].type;
    DataType vD = inputDesc[1].type;


    if (DataType::kHALF == pD){
        printf(" points is half\n");
    }else if (DataType::kFLOAT == pD){
        printf(" points is float\n");
    }

    if (DataType::kHALF == vD){
        printf(" vD is half\n");
    }else if (DataType::kINT32 == vD){
        printf(" vD is int32\n");
    }*/
    
    int const* ValidInput = static_cast<int const*>(inputs[1]);
    
    Dims outdims0 = outputDesc[0].dims;
    Dims outdims1 = outputDesc[1].dims;

    Dims points_dim = inputDesc[0].dims;
    int N = points_dim.d[0];
    int inCols = points_dim.d[1];
    int num_features = inCols-1;
    
	int outCols = num_features;
    if(_center_offset!=0) outCols +=3;
	if(_cluster_offset!=0){
	    outCols +=3;
	}
    
    int grid_size[3];
    for (int i=0; i < 3;i++){
        grid_size[i] = round((_point_cloud_range[3 + i] - _point_cloud_range[i]) / _voxel_size[i]);
    }
    int _total_voxel = grid_size[0]*grid_size[1]*grid_size[2];

    int cuda_idx = 0;
    // auto options = torch::TensorOptions({at::kCUDA, cuda_idx}).dtype(torch::kInt32);


    int *coors = static_cast<int*>(outputs[1]);
    int *num_points_per_voxel = static_cast<int*>(outputs[2]);
    int *ValidOutput = static_cast<int*>(outputs[3]);

    init_int_(coors, -1, _max_voxels*_batch_size*4);
    init_int_(num_points_per_voxel, 0, _max_voxels*_batch_size);
    init_int_(ValidOutput, 0, _batch_size);
    // thrust::fill(coors, coors+_max_voxels*_batch_size*4, -1); // thrust::device, 
    // thrust::fill(num_points_per_voxel, num_points_per_voxel+_max_voxels*_batch_size, 0);
    // thrust::fill(ValidOutput, ValidOutput+_batch_size, 0);

    // torch::Tensor prob_voxel_index = torch::zeros({N, }, options) - 1; //init -1

    //for CSR
    const int batchRowOffset = grid_size[0]*grid_size[1] + 1;

    void* prob_voxel_index_v = workspace;
    void* TensorRowOffset_v = workspace + sizeof(int)*N;
    void* TensorColunms_v = workspace + sizeof(int)*N + sizeof(int)*batchRowOffset*_batch_size;

    int* prob_voxel_index_ptr = static_cast<int*>(prob_voxel_index_v);
    int* TensorRowOffsetPtr = static_cast<int*>(TensorRowOffset_v);
    int* TensorColumnsPtr = static_cast<int*>(TensorColunms_v);

    init_int_(prob_voxel_index_ptr, -1, N);
    init_int_(TensorRowOffsetPtr, 0, batchRowOffset*_batch_size);
    init_int_(TensorColumnsPtr, 0, _max_voxels*_batch_size);
    // thrust::fill(TensorRowOffsetPtr, TensorRowOffsetPtr+batchRowOffset*_batch_size, 0);
    // thrust::fill(TensorColumnsPtr, TensorColumnsPtr+_max_voxels*_batch_size, 0);

    // if (DataType::kHALF == inputDesc[0].type){
#ifdef USE_FP16
        __half const* points = static_cast<__half const*>(inputs[0]);
        __half* voxels = static_cast<__half*>(outputs[0]);
        // __half const* points = (__half const*)inputs[0];
        // __half *voxels = (__half*)outputs[0];
        // init_temp<__half>(voxels, 0.0, _max_voxels*_batch_size*_max_num_points*outCols);
        init_float_half(voxels, (__half)0.0, _max_voxels*_batch_size*_max_num_points*outCols);
        VoxelGeneratorV1Space::cuda_points_to_voxel_fp16(points, ValidInput, prob_voxel_index_ptr,
                             coors, voxels, num_points_per_voxel, ValidOutput,
                             grid_size[0], grid_size[1], grid_size[2],
                             (__half)_point_cloud_range[0], (__half)_point_cloud_range[1], (__half)_point_cloud_range[2],
                             (__half)_voxel_size[0], (__half)_voxel_size[1], (__half)_voxel_size[2],
                             _batch_size, N, inCols, _max_voxels, _max_num_points,
                             TensorRowOffsetPtr, TensorColumnsPtr,
                             _cluster_offset, _center_offset, _supplement, cuda_idx);
#else
        float const* points = static_cast<float const*>(inputs[0]);
        float *voxels = static_cast<float*>(outputs[0]);
        init_float(voxels, 0.0, _max_voxels*_batch_size*_max_num_points*outCols);
        VoxelGeneratorV1Space::cuda_points_to_voxel(points, ValidInput, prob_voxel_index_ptr,
                             coors, voxels, num_points_per_voxel, ValidOutput,
                             grid_size[0], grid_size[1], grid_size[2],
                             _point_cloud_range[0], _point_cloud_range[1], _point_cloud_range[2],
                             _voxel_size[0], _voxel_size[1], _voxel_size[2],
                             _batch_size, N, inCols, _max_voxels, _max_num_points,
                             TensorRowOffsetPtr, TensorColumnsPtr,
                             _cluster_offset, _center_offset, _supplement, cuda_idx);
#endif

    // size_t allsize = _max_voxels*_batch_size*_max_num_points*outCols;
    // float* points_h = (float*)malloc(sizeof(int)*allsize);
    // // LOG_ERROR(cudaMallocHost((void**) &points_h, sizeof(int)*num_outshape_vol*batchSize));
    // cudaMemcpy(points_h, voxels, sizeof(int)*allsize, cudaMemcpyDeviceToHost);
    // for (int i =0; i<1000;i++){
    //     std::cout << " ^ " << points_h[i] << " & ";
    // }
    // free(points_h);
    // std::cout << " \n ---------------- " << std::endl;

    cudaDeviceSynchronize();
    //auto t10=std::chrono::steady_clock::now();
    //double engine_ms=std::chrono::duration<double,std::milli>(t10-t1).count();
    //std::cout << "********* VoxelGeneratorV1 time is: " << engine_ms << std::endl;
    return 0;
}

size_t VoxelGeneratorV1::getSerializationSize() const
{
    return serialized_size(_batch_size) + serialized_size(_max_num_points) + serialized_size(_cluster_offset) + serialized_size(_center_offset) +
    serialized_size(_max_voxels) + serialized_size(_point_cloud_range) + serialized_size(_voxel_size) + serialized_size(_supplement);
}

void VoxelGeneratorV1::serialize(void* buffer) const
{
    serialize_value(&buffer, _batch_size);
    serialize_value(&buffer, _max_num_points);
    serialize_value(&buffer, _max_voxels);
    serialize_value(&buffer, _point_cloud_range);
    serialize_value(&buffer, _voxel_size);
    serialize_value(&buffer, _cluster_offset);
    serialize_value(&buffer, _center_offset);
    serialize_value(&buffer, _supplement);
}

bool VoxelGeneratorV1::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF  || inOut[pos].type == nvinfer1::DataType::kINT32)
        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW); // && inOut[pos].type == inOut[0].type);
}

const char* VoxelGeneratorV1::getPluginType() const
{
    return VoxelGeneratorV1_PLUGIN_NAME;
}

const char* VoxelGeneratorV1::getPluginVersion() const
{
    return VoxelGeneratorV1_PLUGIN_VERSION;
}

void VoxelGeneratorV1::destroy()
{
    delete this;
}

IPluginV2DynamicExt* VoxelGeneratorV1::clone() const
{
    auto* plugin
        = new VoxelGeneratorV1(_batch_size, _max_num_points, _max_voxels, _point_cloud_range, _voxel_size, _center_offset, _cluster_offset, _supplement);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void VoxelGeneratorV1::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* VoxelGeneratorV1::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType VoxelGeneratorV1::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // if (index == 0 && inputTypes[0] == DataType::kHALF){
    //     printf("getOutputDataType  half");
    //     return DataType::kHALF;
    // }else if (index == 0 && inputTypes[0] == DataType::kFLOAT){
    //     printf("getOutputDataType  float");
    //     return DataType::kFLOAT;
    // }
    if (index == 0){
        return DataType::kFLOAT;
    }else if (index == 1){
        return DataType::kINT32;
    }else if (index == 2){
        return DataType::kINT32;
    }else if (index == 3){
        return DataType::kINT32;
    }
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void VoxelGeneratorV1::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void VoxelGeneratorV1::detachFromContext() {}

void VoxelGeneratorV1::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbOutputs == 4);
}



VoxelGeneratorV1PluginCreator::VoxelGeneratorV1PluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("batch_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_num_points", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_voxels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 6));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 3));
    mPluginAttributes.emplace_back(PluginField("center_offset", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("cluster_offset", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("supplement", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* VoxelGeneratorV1PluginCreator::getPluginName() const
{
    return VoxelGeneratorV1_PLUGIN_NAME;
}

const char* VoxelGeneratorV1PluginCreator::getPluginVersion() const
{
    return VoxelGeneratorV1_PLUGIN_VERSION;
}

const PluginFieldCollection* VoxelGeneratorV1PluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* VoxelGeneratorV1PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int batch_size;
    int max_num_points;
    int max_voxels;
    int center_offset;
    int cluster_offset;
    int supplement;
    std::vector<float> point_cloud_range;
    std::vector<float> voxel_size;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "batch_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            batch_size = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "max_num_points"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            max_num_points = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "center_offset"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            center_offset = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "cluster_offset"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            cluster_offset = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "supplement"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            supplement = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "max_voxels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            max_voxels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "point_cloud_range"))
        {
            const int size = fields[i].length;
            const float* a = reinterpret_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                point_cloud_range.push_back(a[j]);
                // a++;
            }
        }
        else if (!strcmp(attrName, "voxel_size"))
        {
            const int size = fields[i].length;
            const float* b = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                voxel_size.push_back(b[j]);
                // b++;
            }
        }
    }

    auto* plugin = new VoxelGeneratorV1(batch_size, max_num_points, max_voxels, point_cloud_range, voxel_size, center_offset, cluster_offset, supplement);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* VoxelGeneratorV1PluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2DynamicExt* plugin = new VoxelGeneratorV1(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}