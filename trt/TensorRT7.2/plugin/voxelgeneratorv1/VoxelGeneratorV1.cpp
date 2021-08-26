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


VoxelGeneratorV1::VoxelGeneratorV1(int batch_size, int max_num_points, int max_voxels, std::vector<float> point_cloud_range, std::vector<float> voxel_size, 
                                   int center_offset, int cluster_offset, int supplement, int use_fp16)
    : _batch_size(batch_size), _max_num_points(max_num_points), _max_voxels(max_voxels), _point_cloud_range(point_cloud_range), _voxel_size(voxel_size),
     _center_offset(center_offset), _cluster_offset(cluster_offset), _supplement(supplement), _use_fp16(use_fp16)
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
    deserialize_value(&data, &length, &_use_fp16);
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
    int numfeature = inCols;
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
        coorsDims.d[1] = exprBuilder.constant(3);
        return coorsDims;
    }
    else if (index == 2){
        nvinfer1::DimsExprs voDims;
        voDims.nbDims = 1;
        voDims.d[0] = exprBuilder.constant(_batch_size);
        return voDims;
    }
    else if (index == 3){
        nvinfer1::DimsExprs nppvDims;
        nppvDims.nbDims = 1;
        nppvDims.d[0] = exprBuilder.constant(_max_voxels*_batch_size);
        return nppvDims;
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
    
    float bev_map = (float)(grid_size[0] * grid_size[1] * sizeof(float)) / 1024 / 1024;
    int value_map_z = 40.0 / bev_map;

    value_map_z = std::max(value_map_z, 1);
    value_map_z = std::min(value_map_z, grid_size[2]);
    int mapsize = grid_size[0] * grid_size[1] * value_map_z;

    Dims indim = inputs[0].dims;
    size_t N = indim.d[0];

    const int listBytes = N * sizeof(VoxelGeneratorV1Space::HashEntry);
    
    size_t totalSize = _batch_size*sizeof(int)*mapsize + _batch_size*sizeof(int)*_max_voxels + listBytes;
    return totalSize;
}

int VoxelGeneratorV1::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{    
    cudaDeviceSynchronize();
    
    int const* ValidInput = static_cast<int const*>(inputs[1]);
    
    Dims points_dim = inputDesc[0].dims;
    int N = points_dim.d[0];
    int inCols = points_dim.d[1];
    int num_features = inCols;

	int outCols = num_features;
    if(_center_offset!=0) outCols +=3;
	if(_cluster_offset!=0){
	    outCols +=3;
	}
    
    std::vector<int> grid_size(3);
    for (int i=0; i < 3;i++){
        grid_size[i] = round((_point_cloud_range[3 + i] - _point_cloud_range[i]) / _voxel_size[i]);
    }

    int *coors = static_cast<int*>(outputs[1]);
    int *ValidOutput = static_cast<int*>(outputs[2]);
    int *num_points_per_voxel = static_cast<int*>(outputs[3]);

    init_int_(coors, -1, _max_voxels*_batch_size*3);
    init_int_(num_points_per_voxel, 0, _max_voxels*_batch_size);
    init_int_(ValidOutput, 0, _batch_size);

    float bev_map = (float)(grid_size[0] * grid_size[1] * sizeof(float)) / 1024 / 1024;
    int value_map_z = 40.0 / bev_map;

    value_map_z = std::max(value_map_z, 1);
    value_map_z = std::min(value_map_z, grid_size[2]);
    int mapsize = grid_size[0] * grid_size[1] * value_map_z;


    void* map_tensor_v = workspace;
    void* addr_tensor_v = workspace + sizeof(int)*mapsize*_batch_size;
    void* list_tensor_v = workspace + sizeof(int)*mapsize*_batch_size + _batch_size*sizeof(int)*_max_voxels;

    int* map_tensor = static_cast<int*>(map_tensor_v);
    int* addr_tensor = static_cast<int*>(addr_tensor_v);

    init_int_(map_tensor, -1, _batch_size*mapsize);
    init_int_(addr_tensor, -1, _batch_size*_max_voxels);

    VoxelGeneratorV1Space::HashEntry* list_tensor_rw = reinterpret_cast<VoxelGeneratorV1Space::HashEntry*>(list_tensor_v);


    if (_use_fp16){
        __half const* points = static_cast<__half const*>(inputs[0]);
        __half* voxels = static_cast<__half*>(outputs[0]);
        init_float_half(voxels, (__half)0.0, _max_voxels*_batch_size*_max_num_points*outCols);
        VoxelGeneratorV1Space::cuda_points_to_voxel_fp16(points, ValidInput,
                                            coors, num_points_per_voxel, voxels, ValidOutput,
                                            map_tensor, addr_tensor, list_tensor_rw,
                                            _point_cloud_range, _voxel_size, grid_size,
                                            _batch_size, N, inCols, outCols,
                                            _cluster_offset, _center_offset, _supplement, _max_voxels, 
                                            _max_num_points, value_map_z);
    }else{
        float const* points = static_cast<float const*>(inputs[0]);
        float *voxels = static_cast<float*>(outputs[0]);
        init_float(voxels, 0.0, _max_voxels*_batch_size*_max_num_points*outCols);
        VoxelGeneratorV1Space::cuda_points_to_voxel(points, ValidInput,
                                                   coors, num_points_per_voxel, voxels, ValidOutput,
                                                   map_tensor, addr_tensor, list_tensor_rw,
                                                   _point_cloud_range, _voxel_size, grid_size,
                                                   _batch_size, N, inCols, outCols,
                                                   _cluster_offset, _center_offset, _supplement, _max_voxels, 
                                                   _max_num_points, value_map_z);
    }
        
    cudaDeviceSynchronize();
    return 0;
}

size_t VoxelGeneratorV1::getSerializationSize() const
{
    return serialized_size(_batch_size) + serialized_size(_max_num_points) + serialized_size(_cluster_offset) + serialized_size(_center_offset) +
    serialized_size(_max_voxels) + serialized_size(_point_cloud_range) + serialized_size(_voxel_size) + serialized_size(_supplement) + serialized_size(_use_fp16);
}

void VoxelGeneratorV1::serialize(void* buffer) const
{
    serialize_value(&buffer, _batch_size);
    serialize_value(&buffer, _max_num_points);
    serialize_value(&buffer, _max_voxels);
    serialize_value(&buffer, _center_offset);
    serialize_value(&buffer, _cluster_offset);
    serialize_value(&buffer, _point_cloud_range);
    serialize_value(&buffer, _voxel_size);
    serialize_value(&buffer, _supplement);
    serialize_value(&buffer, _use_fp16);
}

bool VoxelGeneratorV1::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    switch (pos)
    {
        case 0: 
            if (_use_fp16){
                return in[0].type == nvinfer1::DataType::kHALF   && in[0].format == TensorFormat::kLINEAR;
            }else{
                return in[0].type == nvinfer1::DataType::kFLOAT  && in[0].format == TensorFormat::kLINEAR;
            }
        case 2: 
            if (_use_fp16){
                return out[0].type == nvinfer1::DataType::kHALF  && out[0].format == TensorFormat::kLINEAR;
            }else{
                return out[0].type == nvinfer1::DataType::kFLOAT && out[0].format == TensorFormat::kLINEAR;
            }

        case 1: return in[1].type == nvinfer1::DataType::kINT32  && in[1].format == TensorFormat::kLINEAR;
        case 3: return out[1].type == nvinfer1::DataType::kINT32 && out[1].format == TensorFormat::kLINEAR;
        case 4: return out[2].type == nvinfer1::DataType::kINT32 && out[2].format == TensorFormat::kLINEAR;
        case 5: return out[3].type == nvinfer1::DataType::kINT32 && out[3].format == TensorFormat::kLINEAR;
    }
    printf("invalid connection number\n");
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
        = new VoxelGeneratorV1(_batch_size, _max_num_points, _max_voxels, _point_cloud_range, _voxel_size, _center_offset, _cluster_offset, _supplement, _use_fp16);
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
        if (_use_fp16){
            return DataType::kHALF;
        }else{
            return DataType::kFLOAT;
        }
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
    mPluginAttributes.emplace_back(PluginField("use_fp16", nullptr, PluginFieldType::kINT32, 1));

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
    int use_fp16;
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
        else if (!strcmp(attrName, "use_fp16"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            use_fp16 = *(static_cast<const int*>(fields[i].data));
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

    auto* plugin = new VoxelGeneratorV1(batch_size, max_num_points, max_voxels, point_cloud_range, voxel_size, center_offset, cluster_offset, supplement, use_fp16);
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
