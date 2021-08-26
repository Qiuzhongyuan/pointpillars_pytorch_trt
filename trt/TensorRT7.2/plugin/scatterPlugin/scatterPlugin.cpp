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


Dense::Dense(std::vector<int> spatialShape, int use_fp16)
    : _spatialShape(spatialShape), _use_fp16(use_fp16)
{
}

Dense::Dense(const void* data, size_t length)
{
    deserialize_value(&data, &length, &_spatialShape);
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
    int num_features = (inputs[0].d[1])->getConstantValue();
    int batch_size = (inputs[2].d[0])->getConstantValue();

    nvinfer1::DimsExprs outdims;
    outdims.nbDims = 5;
    outdims.d[0] = exprBuilder.constant(batch_size);
    outdims.d[1] = exprBuilder.constant(num_features);
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
    return 0;
}

int Dense::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    Dims featuredims = inputDesc[0].dims;
    int num_voxels = featuredims.d[0];
    int num_features = featuredims.d[1];
    Dims validims = inputDesc[2].dims;
    int batch_size = validims.d[0];
    int max_voxels = num_voxels / batch_size;

    size_t outsize = batch_size*num_features*_spatialShape[0]*_spatialShape[1]*_spatialShape[2];

    int const* indices_rw = reinterpret_cast<int const*>(inputs[1]);
    int const* valid_rw = reinterpret_cast<int const*>(inputs[2]);

    if (_use_fp16){
        __half const* features_rw = reinterpret_cast<__half const*>(inputs[0]);
        __half* output_rw   = reinterpret_cast<__half*>(outputs[0]);
        init_float_half(output_rw, (__half)0.0, outsize);
	    DenseSpace::cuda_scatter_fp16(features_rw, indices_rw, valid_rw, output_rw, _spatialShape, max_voxels, batch_size, num_features);

    }else{
        float const* features_rw = reinterpret_cast<float const*>(inputs[0]);
        float* output_rw   = reinterpret_cast<float*>(outputs[0]);
        init_float(output_rw, 0, outsize);
        DenseSpace::cuda_scatter(features_rw, indices_rw, valid_rw, output_rw, _spatialShape, max_voxels, batch_size, num_features);
    }

    return 0;
}

size_t Dense::getSerializationSize() const
{
    return serialized_size(_spatialShape) + serialized_size(_use_fp16);
}

void Dense::serialize(void* buffer) const
{
    serialize_value(&buffer, _spatialShape);
    serialize_value(&buffer, _use_fp16);
}

bool Dense::supportsFormatCombination(
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
        case 3: 
            if (_use_fp16){
                return out[0].type == nvinfer1::DataType::kHALF  && out[0].format == TensorFormat::kLINEAR;
            }else{
                return out[0].type == nvinfer1::DataType::kFLOAT && out[0].format == TensorFormat::kLINEAR;
            }

        case 1: return in[1].type == nvinfer1::DataType::kINT32  && in[1].format == TensorFormat::kLINEAR;
        case 2: return in[2].type == nvinfer1::DataType::kINT32  && in[2].format == TensorFormat::kLINEAR;
    }
    printf("invalid connection number\n");
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
        = new Dense(_spatialShape, _use_fp16);
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
    if (_use_fp16){
        return DataType::kHALF;
    }else{
        return DataType::kFLOAT;
    }
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
    mPluginAttributes.emplace_back(PluginField("spatialShape", nullptr, PluginFieldType::kFLOAT32, 3));
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
    std::vector<int> spatialShape;
    int use_fp16;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "spatialShape"))
        {
            const int size = fields[i].length;
            const int* a = reinterpret_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                spatialShape.push_back(a[j]);
                // a++;
            }
        }
        else if (!strcmp(attrName, "use_fp16"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            use_fp16 = *(static_cast<const int*>(fields[i].data));
        }
    }

    auto* plugin = new Dense(spatialShape, use_fp16);
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
