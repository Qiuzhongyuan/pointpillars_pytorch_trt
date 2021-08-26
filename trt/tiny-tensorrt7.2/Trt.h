/*
 * @Date: 2019-08-29 09:48:01
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-03-02 14:58:37
 */

#ifndef TRT_HPP
#define TRT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include "NvInfer.h"
#include "cuda_tookit.h"

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#define SAVEOUTPUT

extern std::vector<cv::String> binnames;
extern float totaltime;

template <typename Dtype>
Dtype round(Dtype r){
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};


// class PluginFactory;

class Trt {
public:
    /**
     * @description: default constructor, will initialize plugin factory with default parameters.
     */
    Trt();

    /**
     * @description: if you costomize some parameters, use this.
     */
    // Trt(TrtPluginParams params);

    ~Trt();

    /**
     * @description: create engine from onnx model
     * @onnxModel: path to onnx model
     * @engineFile: path to saved engien file will be load or save, if it's empty them will not
     *              save engine file
     * @maxBatchSize: max batch size for inference.
     * @return: 
     */
    void CreateEngine(
        const std::string& onnxModel,
        const std::string& engineFile,
        const std::vector<std::string>& customOutput,
        int maxBatchSize,
        int mode,
        const std::vector<std::vector<float>>& calibratorData);


    /**
     * @description: do inference on engine context, make sure you already copy your data to device memory,
     *               see DataTransfer and CopyFromHostToDevice etc.
     */
    void Forward_mult_FP32(std::vector<void*>& data, int save_flag);

    void Forward_mult_FP16(std::vector<void*>& data, int save_flag);
    

    /**
     * @description: async inference on engine context
     * @stream cuda stream for async inference and data transfer
     */
    void ForwardAsync(const cudaStream_t& stream);

    /**
     * @description: data transfer between host and device, for example befor Forward, you need
     *               copy input data from host to device, and after Forward, you need to transfer
     *               output result from device to host.
     * @data data for read and write.
     * @bindIndex binding data index, you can see this in CreateEngine log output.
     * @isHostToDevice 0 for device to host, 1 for host to device (host: cpu memory, device: gpu memory)
     */
    void DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice);

    /**
     * @description: async data tranfer between host and device, see above.
     * @stream cuda stream for async interface and data transfer.
     * @return: 
     */
    void DataTransferAsync(std::vector<float>& data, int bindIndex, bool isHostToDevice, cudaStream_t& stream);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream);
    
    void SetDevice(int device);

    int GetDevice() const;

    /**
     * @description: get max batch size of build engine.
     * @return: max batch size of build engine.
     */
    int GetMaxBatchSize() const;

    /**
     * @description: get binding data pointer in device. for example if you want to do some post processing
     *               on inference output but want to process them in gpu directly for efficiency, you can
     *               use this function to avoid extra data io
     * @return: pointer point to device memory.
     */
    void* GetBindingPtr(int bindIndex) const;

    /**
     * @description: get binding data size in byte, so maybe you need to divide it by sizeof(T) where T is data type
     *               like float.
     * @return: size in byte.
     */
    size_t GetBindingSize(int bindIndex) const;

    /**
     * @description: get binding dimemsions
     * @return: binding dimemsions, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_dims.html
     */
    nvinfer1::Dims GetBindingDims(int bindIndex) const;

    /**
     * @description: get binding data type
     * @return: binding data type, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacenvinfer1.html#afec8200293dc7ed40aca48a763592217
     */
    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

    std::vector<std::string> mBindingName;

protected:

    bool DeserializeEngine(const std::string& engineFile);

    void BuildEngine(nvinfer1::IBuilder* builder,
                      nvinfer1::INetworkDefinition* network,
                      const std::vector<std::vector<float>>& calibratorData,
                      int maxBatchSize,
                      int mode);

    bool BuildEngineWithCaffe(const std::string& prototxt, 
                    const std::string& caffeModel,
                    const std::string& engineFile,
                    const std::vector<std::string>& outputBlobName,
                    const std::vector<std::vector<float>>& calibratorData,
                    int maxBatchSize);

    bool BuildEngineWithOnnx(const std::string& onnxModel,
                     const std::string& engineFile,
                     const std::vector<std::string>& customOutput,
                     const std::vector<std::vector<float>>& calibratorData,
                     int maxBatchSize);

    bool BuildEngineWithUff(const std::string& uffModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& inputTensorName,
                      const std::vector<std::vector<int>>& inputDims,
                      const std::vector<std::string>& outputTensorName,
                      const std::vector<std::vector<float>>& calibratorData,
                      int maxBatchSize);
                     
    /**
     * description: Init resource such as device memory
     */
    void InitEngine();

    /**
     * description: save engine to engine file
     */
    void SaveEngine(const std::string& fileName);

protected:
    TrtLogger mLogger;

    // tensorrt run mode 0:fp32 1:fp16 2:int8
    int mRunMode;

    nvinfer1::ICudaEngine* mEngine = nullptr;

    nvinfer1::IExecutionContext* mContext = nullptr;

    // PluginFactory* mPluginFactory;

    nvinfer1::IRuntime* mRuntime = nullptr;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    float* out = nullptr;

    int mInputSize = 0;

    // batch size
    int mBatchSize; 
};

#endif
