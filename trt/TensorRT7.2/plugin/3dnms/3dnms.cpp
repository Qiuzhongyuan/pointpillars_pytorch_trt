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
#include "3dnms.h"
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

using namespace nvinfer1;
using nvinfer1::plugin::NMS3D;
using nvinfer1::plugin::NMS3DPluginCreator;

static const char* NMS3D_PLUGIN_VERSION{"001"};
static const char* NMS3D_PLUGIN_NAME{"NMS"};

PluginFieldCollection NMS3DPluginCreator::mFC{};
std::vector<PluginField> NMS3DPluginCreator::mPluginAttributes;


NMS3D::NMS3D(int nms_post_maxsize, int nms_pre_maxsize, float nms_thresh, float score_thresh, int use_bev, int batchSize, int use_fp16)
    : _nms_post_maxsize(nms_post_maxsize), _nms_pre_maxsize(nms_pre_maxsize), _nms_thresh(nms_thresh), _score_thresh(score_thresh), _use_bev(use_bev),
    _bs(batchSize), _use_fp16(use_fp16)
{
}

NMS3D::NMS3D(const void* data, size_t length)
{
    deserialize_value(&data, &length, &_nms_post_maxsize);
    deserialize_value(&data, &length, &_nms_pre_maxsize);
    deserialize_value(&data, &length, &_nms_thresh);
    deserialize_value(&data, &length, &_score_thresh);
    deserialize_value(&data, &length, &_use_bev);
    deserialize_value(&data, &length, &_bs);
    deserialize_value(&data, &length, &_use_fp16);
}

NMS3D::~NMS3D()
{
}

int NMS3D::getNbOutputs() const
{
    return 1;
}

DimsExprs NMS3D::getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, nvinfer1::IExprBuilder& exprBuilder)
{
    if (_use_bev){
        nvinfer1::DimsExprs outdims;
        outdims.nbDims = 3;
        outdims.d[0] = exprBuilder.constant(_bs);
        outdims.d[1] = exprBuilder.constant(9);
        outdims.d[2] = exprBuilder.constant(_nms_post_maxsize);
        return outdims;
    }else{
        nvinfer1::DimsExprs outdims;
        outdims.nbDims = 3;
        outdims.d[0] = exprBuilder.constant(_bs);
        outdims.d[1] = exprBuilder.constant(6);
        outdims.d[2] = exprBuilder.constant(_nms_post_maxsize);
        return outdims;
    }
    
}

int NMS3D::initialize()
{
    return STATUS_SUCCESS;
}

void NMS3D::terminate()
{
}

size_t NMS3D::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    Dims cls_dim = inputs[1].dims;
    Dims batch_box_dims = inputs[0].dims;
    int batch_size = batch_box_dims.d[0];
    int num_box = batch_box_dims.d[2];
    int num_cls = cls_dim.d[1];
    int num_out_info;
    if(_use_bev!=0){
        num_out_info = 9;
    }
    else{
        num_out_info = 6;
    }
    size_t totalsize = sizeof(int)*num_box + sizeof(float)*num_box + sizeof(int)*num_box + sizeof(int)*_nms_pre_maxsize +
     sizeof(int)*_nms_pre_maxsize*(num_out_info-2) + sizeof(int)*batch_size*2 + sizeof(float)*_nms_pre_maxsize*_nms_pre_maxsize +
     sizeof(float)*batch_size*_nms_post_maxsize*num_out_info + sizeof(float)*batch_size*num_box*(num_out_info-2) + 
     sizeof(float)*batch_size*num_box*num_cls;
    return totalsize;
}

int NMS3D::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    cudaDeviceSynchronize();
    //auto t1=std::chrono::steady_clock::now();
    Dims batch_box_dims = inputDesc[0].dims;
    Dims batch_cls_dims = inputDesc[1].dims;
    int batch_size = batch_box_dims.d[0];
    int num_box = batch_box_dims.d[2];
    int num_cls = batch_cls_dims.d[1];

    int num_out_info;
    if(_use_bev!=0){
        num_out_info = 9;
    }
    else{
        num_out_info = 6;
    }


    void* index_v = workspace;
    void* score_v = workspace + sizeof(int)*num_box;
    void* cls_type_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box;
    void* cls_temp_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2;
    void* box_temp_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize;
    void* pos_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize + sizeof(int)*_nms_pre_maxsize*(num_out_info-2);
    void* ious_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize + sizeof(int)*_nms_pre_maxsize*(num_out_info-2) + sizeof(int)*_bs*2;
    void* outfeature_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize + sizeof(int)*_nms_pre_maxsize*(num_out_info-2) + sizeof(int)*_bs*2 + 
     sizeof(float)*_nms_pre_maxsize*_nms_pre_maxsize;
    void* batch_box_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize + sizeof(int)*_nms_pre_maxsize*(num_out_info-2) + sizeof(int)*_bs*2 + 
     sizeof(float)*_nms_pre_maxsize*_nms_pre_maxsize + sizeof(float)*_bs*_nms_post_maxsize*num_out_info;
    void* batch_cls_v = workspace + sizeof(int)*num_box + sizeof(float)*num_box*2 + + sizeof(int)*_nms_pre_maxsize + sizeof(int)*_nms_pre_maxsize*(num_out_info-2)  + sizeof(int)*_bs*2 + 
     sizeof(float)*_nms_pre_maxsize*_nms_pre_maxsize + sizeof(float)*_bs*_nms_post_maxsize*num_out_info + sizeof(float)*_bs*num_box*(num_out_info-2);

    int* index = static_cast<int*>(index_v);
    float* score = static_cast<float*>(score_v);
    int* cls_type = static_cast<int*>(cls_type_v);
    int* cls_temp = static_cast<int*>(cls_temp_v);
    float* box_temp = static_cast<float*>(box_temp_v);
    int* pos = static_cast<int*>(pos_v);
    float* ious = static_cast<float*>(ious_v);
    float* outfeature = static_cast<float*>(outfeature_v);
    float* batch_box = static_cast<float*>(batch_box_v);
    float* batch_cls = static_cast<float*>(batch_cls_v);

    // LOG_ERROR(cudaMemset(index, 0, sizeof(int)*num_box*batch_size));
    // LOG_ERROR(cudaMemset(score, 0, sizeof(float)*num_box));
    // LOG_ERROR(cudaMemset(cls_type, 0, sizeof(int)*num_box));
    // LOG_ERROR(cudaMemset(cls_temp, 0, sizeof(int)*_nms_pre_maxsize));
    // LOG_ERROR(cudaMemset(box_temp, 0, sizeof(float)*_nms_pre_maxsize));
    // LOG_ERROR(cudaMemset(pos, 0, sizeof(int)*batch_size*2));
    // LOG_ERROR(cudaMemset(ious, 0, sizeof(float)*_nms_pre_maxsize*_nms_pre_maxsize));
    // LOG_ERROR(cudaMemset(batch_box, 0, sizeof(float)*batch_size*num_box*(num_out_info-2)));
    // LOG_ERROR(cudaMemset(batch_cls, 0, sizeof(float)*batch_size*num_box*num_cls));
    // printf("++++++++++++++++++++++++++++++++++++\n\n\n");
    init_int_(index, 0, num_box*batch_size);
    init_zeros(score, 0, num_box*batch_size);
    init_int_(cls_type, 0, num_box);
    init_int_(cls_temp, 0, _nms_pre_maxsize);
    init_zeros(box_temp, 0, _nms_pre_maxsize);
    init_int_(pos, 0, batch_size*2);
    init_zeros(ious, 0, _nms_pre_maxsize*_nms_pre_maxsize);
    init_zeros(batch_box, 0, batch_size*num_box*(num_out_info-2));
    init_zeros(batch_cls, 0, batch_size*num_box*num_cls);

    init_zeros(outfeature, -1, batch_size*_nms_post_maxsize*num_out_info);

    // if (DataType::kHALF == inputDesc->type){
#ifdef USE_FP16
        //  printf("DataType::kHALF == inputDesc->type\n\n\n\n\n\n");
        __half const* temp_box = static_cast<__half const*>(inputs[0]);
        __half const* temp_cls = static_cast<__half const*>(inputs[1]);
        half2float_custom(temp_box, batch_box, batch_size*num_box*(num_out_info-2));
        half2float_custom(temp_cls, batch_cls, batch_size*num_box*num_cls);
    // }else if(DataType::kFLOAT == inputDesc->type||DataType::kINT8 == inputDesc->type){
#else
        //printf("3dnms:  DataType::kFLOAT32 == inputDesc->type||DataType::kINT8 == inputDesc->type\n\n\n\n\n\n");
        float const* temp_box = static_cast<float const*>(inputs[0]);
        float const* temp_cls = static_cast<float const*>(inputs[1]);
        LOG_ERROR(cudaMemcpy(batch_box, temp_box, sizeof(float)*batch_size*num_box*(num_out_info-2), cudaMemcpyDeviceToDevice));
        LOG_ERROR(cudaMemcpy(batch_cls, temp_cls, sizeof(float)*batch_size*num_box*num_cls, cudaMemcpyDeviceToDevice));
    // }
#endif

    
    NMSSpace::cuda_nms(batch_box, batch_cls, score, cls_type, index, pos, cls_temp, box_temp, ious, outfeature, num_box,
     num_cls, _nms_pre_maxsize, _nms_post_maxsize, _nms_thresh, batch_size, _score_thresh, _use_bev);

 #ifdef USE_FP16
        __half *outfeat = static_cast<__half *>(outputs[0]);
        float2half_custom(outfeature, outfeat, batch_size*_nms_post_maxsize*num_out_info);
#else
        float *outfeat = static_cast<float *>(outputs[0]);
        LOG_ERROR(cudaMemcpy(outfeat, outfeature, sizeof(float)*batch_size*_nms_post_maxsize*num_out_info, cudaMemcpyDeviceToDevice));
#endif

    // LOG_ERROR(cudaFree(batch_box));
    // LOG_ERROR(cudaFree(batch_cls));
    // LOG_ERROR(cudaFree(outfeature));
    

    // LOG_ERROR(cudaFree(index));
    // LOG_ERROR(cudaFree(score));
    // LOG_ERROR(cudaFree(cls_type));
    // LOG_ERROR(cudaFree(cls_temp));
    // LOG_ERROR(cudaFree(box_temp));
    // LOG_ERROR(cudaFree(score_temp));
    // LOG_ERROR(cudaFree(ious));
    cudaDeviceSynchronize();
    //auto t10=std::chrono::steady_clock::now();
    //double engine_ms=std::chrono::duration<double,std::milli>(t10-t1).count();
    //std::cout << "********* NMS all time is: " << engine_ms << std::endl;
    return 0;
}

size_t NMS3D::getSerializationSize() const
{
    return serialized_size(_nms_post_maxsize) + serialized_size(_nms_pre_maxsize) + serialized_size(_nms_thresh)
     + serialized_size(_score_thresh) + serialized_size(_use_bev) + serialized_size(_bs) + serialized_size(_use_fp16);
}

void NMS3D::serialize(void* buffer) const
{
    serialize_value(&buffer, _nms_post_maxsize);
    serialize_value(&buffer, _nms_pre_maxsize);
    serialize_value(&buffer, _nms_thresh);
    serialize_value(&buffer, _score_thresh);
    serialize_value(&buffer, _use_bev);
    serialize_value(&buffer, _bs);
    serialize_value(&buffer, _use_fp16);
}

bool NMS3D::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW && inOut[pos].type == inOut[0].type);
}

const char* NMS3D::getPluginType() const
{
    return NMS3D_PLUGIN_NAME;
}

const char* NMS3D::getPluginVersion() const
{
    return NMS3D_PLUGIN_VERSION;
}

void NMS3D::destroy()
{
    delete this;
}

IPluginV2DynamicExt* NMS3D::clone() const
{
    auto* plugin
        = new NMS3D(_nms_post_maxsize, _nms_pre_maxsize, _nms_thresh, _score_thresh, _use_bev, _bs, _use_fp16);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void NMS3D::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* NMS3D::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType NMS3D::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void NMS3D::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void NMS3D::detachFromContext() {}

void NMS3D::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbOutputs == 1);
}



NMS3DPluginCreator::NMS3DPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("nms_post_maxsize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nms_pre_maxsize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nms_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_bev", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("batch_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_fp16", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NMS3DPluginCreator::getPluginName() const
{
    return NMS3D_PLUGIN_NAME;
}

const char* NMS3DPluginCreator::getPluginVersion() const
{
    return NMS3D_PLUGIN_VERSION;
}

const PluginFieldCollection* NMS3DPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* NMS3DPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int _nms_post_maxsize;
    int _nms_pre_maxsize;
    float _nms_thresh;
    float _score_thresh;
    int _use_bev;
    int _bs;
    int _use_fp16;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "nms_post_maxsize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            _nms_post_maxsize = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "nms_pre_maxsize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            _nms_pre_maxsize = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "nms_thresh"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            _nms_thresh = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "score_thresh"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            _score_thresh = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "use_bev"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            _use_bev = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "batch_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            _bs = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "use_fp16"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            _use_fp16 = *(static_cast<const int*>(fields[i].data));
        }
    }

    auto* plugin = new NMS3D(_nms_post_maxsize, _nms_pre_maxsize, _nms_thresh, _score_thresh, _use_bev, _bs, _use_fp16);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* NMS3DPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2DynamicExt* plugin = new NMS3D(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
