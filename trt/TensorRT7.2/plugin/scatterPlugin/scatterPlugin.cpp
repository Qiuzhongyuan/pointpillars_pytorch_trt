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
#include "scatterPlugin.h"
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

using namespace nvinfer1;
using nvinfer1::plugin::Dense;
using nvinfer1::plugin::DensePluginCreator;

static const char* Dense_PLUGIN_VERSION{"001"};
static const char* Dense_PLUGIN_NAME{"Dense"};

PluginFieldCollection DensePluginCreator::mFC{};
std::vector<PluginField> DensePluginCreator::mPluginAttributes;


Dense::Dense(int batch_size, std::vector<int> spatialShape, int channels, int use_fp16)
    : _batch_size(batch_size), _spatialShape(spatialShape), _channels(channels), _use_fp16(use_fp16)
{
}

Dense::Dense(const void* data, size_t length)
{
    deserialize_value(&data, &length, &_batch_size);
    deserialize_value(&data, &length, &_spatialShape);
    deserialize_value(&data, &length, &_channels);
    deserialize_value(&data, &length, &_use_fp16);

}

Dense::~Dense()
{
}

int Dense::getNbOutputs() const
{
    return 1;
}

DimsExprs Dense::getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs outdims;
    outdims.nbDims = 5;
    outdims.d[0] = exprBuilder.constant(_batch_size);
    outdims.d[1] = exprBuilder.constant(_channels);
    outdims.d[2] = exprBuilder.constant(_spatialShape[0]);
    outdims.d[3] = exprBuilder.constant(_spatialShape[1]);
    outdims.d[4] = exprBuilder.constant(_spatialShape[2]);
    return outdims;
}

int Dense::initialize()
{
    return STATUS_SUCCESS;
}

void Dense::terminate()
{
}

size_t Dense::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    Dims indim = inputs[0].dims;
    size_t inSize = indim.d[indim.nbDims-3]*indim.d[indim.nbDims-2];
    size_t outSize = _channels*_spatialShape[0]*_spatialShape[1]*_spatialShape[2];
    // return sizeof(float)*inSize+sizeof(float)*outSize;
    return 0;
}

int Dense::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    Dims featuredims = inputDesc[0].dims;
    int num_voxels = featuredims.d[0];
    int num_features = featuredims.d[1];
    size_t outsize = _batch_size*_channels*_spatialShape[0]*_spatialShape[1]*_spatialShape[2];

    int const* indices_rw = reinterpret_cast<int const*>(inputs[1]);

    // if (DataType::kHALF == inputDesc->type){
#ifdef USE_FP16
        //printf("DataType::kHALF == inputDesc->type\n\n\n\n");
        __half const* features_rw = reinterpret_cast<__half const*>(inputs[0]);
        __half* output_rw   = reinterpret_cast<__half*>(outputs[0]);
        init_float_half(output_rw, (__half)0.0, outsize);
	    DenseSpace::cuda_scatter_fp16(features_rw, indices_rw, output_rw, _spatialShape, num_voxels, num_features);
    // }else if(DataType::kFLOAT == inputDesc->type||DataType::kINT8 == inputDesc->type){
#else
        //printf("DataType::kFLOAT32 == inputDesc->type||DataType::kINT8 == inputDesc->type\n\n\n\n\n\n");
        float const* features_rw = reinterpret_cast<float const*>(inputs[0]);
        float* output_rw   = reinterpret_cast<float*>(outputs[0]);
        // init_zeros(output_rw, 0, outsize);
        init_float(output_rw, 0, outsize);
        DenseSpace::cuda_scatter(features_rw, indices_rw, output_rw, _spatialShape, num_voxels, num_features);

#endif

    //printf("***** after dense *******\n");
    return 0;
}

size_t Dense::getSerializationSize() const
{
    return serialized_size(_batch_size) + serialized_size(_spatialShape) + serialized_size(_channels) + serialized_size(_use_fp16);
}

void Dense::serialize(void* buffer) const
{
    serialize_value(&buffer, _batch_size);
    serialize_value(&buffer, _spatialShape);
    serialize_value(&buffer, _channels);
    serialize_value(&buffer, _use_fp16);
}

bool Dense::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kINT32)
        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW);// && inOut[pos].type == inOut[0].type);
}

const char* Dense::getPluginType() const
{
    return Dense_PLUGIN_NAME;
}

const char* Dense::getPluginVersion() const
{
    return Dense_PLUGIN_VERSION;
}

void Dense::destroy()
{
    delete this;
}

IPluginV2DynamicExt* Dense::clone() const
{
    auto* plugin
        = new Dense(_batch_size, _spatialShape, _channels, _use_fp16);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void Dense::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* Dense::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType Dense::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Dense::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Dense::detachFromContext() {}

void Dense::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbOutputs == 1);
}



DensePluginCreator::DensePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("batch_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatialShape", nullptr, PluginFieldType::kFLOAT32, 3));
    mPluginAttributes.emplace_back(PluginField("channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_fp16", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DensePluginCreator::getPluginName() const
{
    return Dense_PLUGIN_NAME;
}

const char* DensePluginCreator::getPluginVersion() const
{
    return Dense_PLUGIN_VERSION;
}

const PluginFieldCollection* DensePluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* DensePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int batch_size;
    std::vector<int> spatialShape;
    int channels;
    int use_fp16;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "batch_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            batch_size = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "spatialShape"))
        {
            const int size = fields[i].length;
            const int* a = reinterpret_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                spatialShape.push_back(a[j]);
                // a++;
            }
        }
        else if (!strcmp(attrName, "channels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            channels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "use_fp16"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            use_fp16 = *(static_cast<const int*>(fields[i].data));
        }
    }

    auto* plugin = new Dense(batch_size, spatialShape, channels, use_fp16);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* DensePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2DynamicExt* plugin = new Dense(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
