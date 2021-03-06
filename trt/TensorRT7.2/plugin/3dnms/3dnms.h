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
#ifndef TRT_3DNMS_PLUGIN_H
#define TRT_3DNMS_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>
#include <NMS_nova.h>
#include <cuda_fp16.h>
#include "kernel.h"

#define LOG_ERROR(status)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace nvinfer1
{
namespace plugin
{
class NMS3D : public nvinfer1::IPluginV2DynamicExt
{
public:

    NMS3D(int nms_post_maxsize, int nms_pre_maxsize, float nms_thresh, float score_thresh, int use_bev, int batchSize, int use_fp16);

    NMS3D(const void* data, size_t length);

    NMS3D() = delete;

    ~NMS3D() override;

    int getNbOutputs() const override;

    DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, nvinfer1::IExprBuilder& exprBuilder) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
              const void* const* inputs, void* const* outputs, 
              void* workspace, 
              cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) override;

    void detachFromContext() override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

private:
    // Weights copyToDevice(const void* hostData, size_t count);

    // void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;

    // Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    int _nms_post_maxsize;
    int _nms_pre_maxsize;
    float _nms_thresh;
    float _score_thresh;
    int _use_bev;
    int _bs;
    int _use_fp16;
    
    const char* mPluginNamespace;
};

class NMS3DPluginCreator : public BaseCreator
{
public:
    NMS3DPluginCreator();

    ~NMS3DPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_3DNMS_PLUGIN_H
