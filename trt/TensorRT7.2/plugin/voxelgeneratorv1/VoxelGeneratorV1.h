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
#ifndef TRT_VOXELGENERATORV1_PLUGIN_H
#define TRT_VOXELGENERATORV1_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_voxelV1_generator.h>
#include <cuda_fp16.h>
#include <kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
// #include <thrust/execution_policy.h>


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
class VoxelGeneratorV1 : public nvinfer1::IPluginV2DynamicExt
{
public:

    VoxelGeneratorV1(int batch_size, int max_num_points, int max_voxels, std::vector<float> point_cloud_range, std::vector<float> voxel_size, int center_offset, 
                     int cluster_offset, int supplement, int use_fp16);

    VoxelGeneratorV1(const void* data, size_t length);

    VoxelGeneratorV1() = delete;

    ~VoxelGeneratorV1() override;

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

    int _batch_size;
    int _max_num_points;
    int _max_voxels;
    int _center_offset;
    int _cluster_offset;
    int _supplement;
    int _use_fp16;
    std::vector<float> _point_cloud_range;
    std::vector<float> _voxel_size;
    
    const char* mPluginNamespace;
};

class VoxelGeneratorV1PluginCreator : public BaseCreator
{
public:
    VoxelGeneratorV1PluginCreator();

    ~VoxelGeneratorV1PluginCreator() override = default;

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

#endif // TRT_VOXELGENERATORV1_PLUGIN_H
