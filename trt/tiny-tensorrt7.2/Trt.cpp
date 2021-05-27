/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2020-06-10 11:51:09
 * @LastEditors: zerollzeng
 */
#include "Trt.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "Int8EntropyCalibrator.h"
#include <fstream>
// #include "PluginFactory.h"
// #include "tensorflow/graph.pb.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

// #define pi 3.14159265358979323846
#define BOUNDARY_THRESH 0.05
#define VSLOT_MIN_DIST 0.044771278151623496
#define VSLOT_MAX_DIST 0.1099427457599304
#define HSLOT_MIN_DIST 0.15057789144568634
#define HSLOT_MAX_DIST 0.44449496544202816
#define SLOT_SUPPRESSION_DOT_PRODUCT_THRESH 0.8
#define BRIDGE_ANGLE_DIFF 0.23597706424922374
#define SEPARATOR_ANGLE_DIFF 0.42337349082331477
#define SHORT_SEPARATOR_LENGTH 0.199519231
#define LONG_SEPARATOR_LENGTH 0.46875
#define ENTRANCE_LENGTH_MASK 30.0
const float eps = 1e-8;

const float pi = 3.14159265358979323846;
using namespace std;

using namespace nvinfer1;

#ifndef SAVEOUTPUT
#define SAVEOUTPUT
#endif

Trt::Trt() {}

Trt::~Trt() {
    // if(mPluginFactory != nullptr) {
    //     delete mPluginFactory;
    //     mPluginFactory = nullptr;
    // }
    if(mContext != nullptr) {
        mContext->destroy();
        mContext = nullptr;
    }
    if(mEngine !=nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    for(size_t i=0;i<mBinding.size();i++) {
        safeCudaFree(mBinding[i]);
    }
}

void Trt::CreateEngine(
        const std::string& prototxt, 
        const std::string& caffeModel,
        const std::string& engineFile,
        const std::vector<std::string>& outputBlobName,
        int maxBatchSize,
        int mode,
        const std::vector<std::vector<float>>& calibratorData) {
    mRunMode = mode;
    spdlog::info("prototxt: {}",prototxt);
    spdlog::info("caffeModel: {}",caffeModel);
    spdlog::info("engineFile: {}",engineFile);
    spdlog::info("outputBlobName: ");
    for(size_t i=0;i<outputBlobName.size();i++) {
        std::cout << outputBlobName[i] << " ";
    }
    std::cout << std::endl;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithCaffe(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
    // Notice: close profiler
    //mContext->setProfiler(mProfiler);
}

void Trt::CreateEngine(
        const std::string& onnxModel,
        const std::string& engineFile,
        const std::vector<std::string>& customOutput,
        int maxBatchSize,
        int mode,
        const std::vector<std::vector<float>>& calibratorData) {
    mRunMode = mode;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithOnnx(onnxModel,engineFile,customOutput,calibratorData,maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    std::cout << "--------- before InitEngine " << std::endl;
    InitEngine();
}

void Trt::CreateEngine(
        const std::string& uffModel,
        const std::string& engineFile,
        const std::vector<std::string>& inputTensorNames,
        const std::vector<std::vector<int>>& inputDims,
        const std::vector<std::string>& outputTensorNames,
        int maxBatchSize,
        int mode,
        const std::vector<std::vector<float>>& calibratorData) {
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithUff(uffModel,engineFile,inputTensorNames,inputDims, outputTensorNames,calibratorData, maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
}

void Trt::Forward() {
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //mContext->execute(mBatchSize, &mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    spdlog::info("net forward takes {} ms", elapsedTime);
}

void Trt::Forward_data(float* data, int length) 
{
    std::ofstream outfile;
    outfile.open("aa.bin", std::ios::binary);
    // for (int i = 0; i < 200; i++){
    //     std::cout << "data value:  " << data[i] << std::endl;
    // }
    if (data == nullptr){
        std::cout << " data ptr is null " << std::endl;
    }
    if (mBinding[0] == nullptr){
        std::cout << " mBinding[0] ptr is null " << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(mBinding[0], data, sizeof(float)*length, cudaMemcpyHostToDevice));
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // mContext->execute(mBatchSize, &mBinding[0]);
    mContext->executeV2(&mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    spdlog::info("net forward takes {} ms", elapsedTime);
    out = (float*)malloc(mBindingSize[1]);
    std::cout << " out size: " << mBindingSize[1] << std::endl;
    CUDA_CHECK(cudaMemcpy(out, mBinding[3], mBindingSize[3], cudaMemcpyDeviceToHost));
    // for(int i = 0; i < mBindingSize[1]/4; i++){
    //     std::cout << " " << out[i] << " ";
    // }
    // std::cout << std::endl;
    outfile.write(reinterpret_cast<const char*>(out), mBindingSize[3]);
    outfile.close();
}

static int inum=0;
float totaltime=0.0;

void Trt::Forward_mult(std::vector<void*>& data) 
{
    __half* temp_half = nullptr;
    float* temp_float = nullptr;
    for (int i = 0; i < data.size()-1; i++)
    {
        
        CUDA_CHECK(cudaMalloc((void**)&temp_float, sizeof(float)*25000*5));
        CUDA_CHECK(cudaMemcpy(temp_float, data[i], sizeof(float)*25000*5, cudaMemcpyHostToDevice));
       
        printf("1\n");
        CUDA_CHECK(cudaMalloc((void**)&temp_half, sizeof(__half)*25000*5));
        convertFP32ToFP16(temp_float, temp_half, 25000*5);
        printf("mBindingSize[i]:  %d  \n",mBindingSize[i]);
        CUDA_CHECK(cudaMemcpy(mBinding[i], temp_half, mBindingSize[i], cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(temp_float));
        CUDA_CHECK(cudaFree(temp_half));
    }
    CUDA_CHECK(cudaMemcpy(mBinding[1], data[1], mBindingSize[1], cudaMemcpyHostToDevice));


    spdlog::info("\n\n net forward begin");
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // mContext->execute(mBatchSize, &mBinding[0]);
    mContext->executeV2(&mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[0;1;33;41m times %f\033[0m\n",elapsedTime);totaltime+=elapsedTime;

    string strclone = binnames[inum++];
    std::cout<<"binname :"<<strclone<<std::endl;
    for (int i = mInputSize; i < mBinding.size();i++)
    {

#ifdef SAVEOUTPUT
        std::ofstream outfile;
        char outname[2000];
        string str=strclone;str=str.substr(0,str.find_last_of("."));str=str+"_"+std::to_string(i-mInputSize)+".bin";
        str="../saveint8"+str.substr(str.find_last_of("/"));
        //str="."+str.substr(str.find_last_of("/"));
        std::cout<<"str"<<str<<std::endl;
        sprintf(outname, str.c_str(), i-mInputSize);
        outfile.open(outname, std::ios::binary);
#endif

        printf("out %d\n", mBindingSize[i]);
        float* out_t = (float*)malloc(mBindingSize[i] * 2);
        float* temp_float;
        CUDA_CHECK(cudaMalloc((void**)&temp_float, sizeof(float)*900));
        convertFP16ToFP32(static_cast<__half*>(mBinding[i]), temp_float, 900);
        CUDA_CHECK(cudaMemcpy(out_t, temp_float, sizeof(float)*900, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(temp_float));
        // CUDA_CHECK(cudaMemcpy(out_t, mBinding[i], mBindingSize[i], cudaMemcpyDeviceToHost));

        for(int j=0;j<mBindingSize[i]/2&&j<36;j++)
        {
            if(j%9==0)
                printf("\n");
            printf("%f ", out_t[j]);
        }
        printf("\n");

#ifdef SAVEOUTPUT
        outfile.write(reinterpret_cast<const char*>(out_t), mBindingSize[i]/2);
        outfile.close();
        free(out_t);
#endif

    }
    
}
void Trt::Forward_mult_FP32(std::vector<void*>& data) 
{
    for (int i = 0; i < data.size(); i++)
    {
       CUDA_CHECK(cudaMemcpy(mBinding[i], data[i], mBindingSize[i], cudaMemcpyHostToDevice));
    }

    spdlog::info("\n\n net forward begin");
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // mContext->execute(mBatchSize, &mBinding[0]);
    mContext->executeV2(&mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[0;1;33;41m times %f\033[0m\n",elapsedTime);totaltime+=elapsedTime;

    string strclone = binnames[inum++];
    std::cout<<"binname :"<<strclone<<std::endl;
    for (int i = mInputSize; i < mBinding.size();i++)
    {
#ifdef SAVEOUTPUT
        std::ofstream outfile;
        char outname[2000];
        string str=strclone;str=str.substr(0,str.find_last_of("."));str=str+"_"+std::to_string(i-mInputSize)+".bin";
        str="../saveint8"+str.substr(str.find_last_of("/"));
        //str="."+str.substr(str.find_last_of("/"));
        std::cout<<"outname: str"<<str<<std::endl;
        sprintf(outname, str.c_str(), i-mInputSize);
        outfile.open(outname, std::ios::binary);
#endif
        float* out_t = (float*)malloc(mBindingSize[i]);
        printf("mBindingSize[i] %d\n",mBindingSize[i]);
        CUDA_CHECK(cudaMemcpy(out_t, mBinding[i], mBindingSize[i], cudaMemcpyDeviceToHost));
        for(int j=0;j<mBindingSize[i]/sizeof(float)&&j<36;j++)
        {
            if(j%9==0)
                printf("\n");
            printf("%f ", out_t[j]);
        }
        printf("\n");
#ifdef SAVEOUTPUT
        outfile.write(reinterpret_cast<const char*>(out_t), mBindingSize[i]);
        outfile.close();
        free(out_t);
#endif
    } 
}



void Trt::ForwardAsync(const cudaStream_t& stream) {
    mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
}

void Trt::DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice) {
    if(isHostToDevice) {
        assert(data.size()*sizeof(float) <= mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpy(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
    }
}

void Trt::DataTransferAsync(std::vector<float>& data, int bindIndex, bool isHostToDevice, cudaStream_t& stream) {
    if(isHostToDevice) {
        assert(data.size()*sizeof(float) <= mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    } else {
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
    }
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
}

void Trt::SetDevice(int device) {
    spdlog::warn("warning: make sure save engine file match choosed device");
    CUDA_CHECK(cudaSetDevice(device));
}

int Trt::GetDevice() const { 
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if(device != -1) {
        return device;
    } else {
        spdlog::error("Get Device Error");
        return -1;
    }
}

int Trt::GetMaxBatchSize() const{
    return mBatchSize;
}

void* Trt::GetBindingPtr(int bindIndex) const {
    return mBinding[bindIndex];
}

size_t Trt::GetBindingSize(int bindIndex) const {
    return mBindingSize[bindIndex];
}

nvinfer1::Dims Trt::GetBindingDims(int bindIndex) const {
    return mBindingDims[bindIndex];
}

nvinfer1::DataType Trt::GetBindingDataType(int bindIndex) const {
    return mBindingDataType[bindIndex];
}

void Trt::SaveEngine(const std::string& fileName) {
    if(fileName == "") {
        spdlog::warn("empty engine file name, skip save");
        return;
    }
    if(mEngine != nullptr) {
        spdlog::info("save engine to {}...",fileName);
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open()) {
            spdlog::error("read create engine file {} failed",fileName);
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        data->destroy();
    } else {
        spdlog::error("engine is empty, save engine failed");
    }
}

bool Trt::DeserializeEngine(const std::string& engineFile) {
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if(in.is_open()) {
        spdlog::info("deserialize engine from {}",engineFile);
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        initLibNvInferPlugins(&mLogger, "");
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine = mRuntime->deserializeCudaEngine((void*)engineBuf.get(), bufCount, nullptr);
        assert(mEngine != nullptr);
        mBatchSize = mEngine->getMaxBatchSize();
        spdlog::info("max batch size of deserialized engine: {}",mEngine->getMaxBatchSize());
        mRuntime->destroy();
        return true;
    }
    return false;
}

void Trt::BuildEngine(nvinfer1::IBuilder* builder,
                      nvinfer1::INetworkDefinition* network,
                      const std::vector<std::vector<float>>& calibratorData,
                      int maxBatchSize,
                      int mode) {
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    Int8EntropyCalibrator* calibrator = nullptr;
    if (mRunMode == 2)
    {
        spdlog::info("set int8 inference mode");
        if (!builder->platformHasFastInt8())
        {
            spdlog::warn("Warning: current platform doesn't support int8 inference");
        }
        if (calibratorData.size() > 0 ){
            std::string calibratorName = "calibrator";
            std::cout << "create calibrator,Named:" << calibratorName << std::endl;
            calibrator = new Int8EntropyCalibrator(maxBatchSize,calibratorData,calibratorName,true);
        }
        // enum class BuilderFlag : int
        // {
        //     kFP16 = 0,         //!< Enable FP16 layer selection.
        //     kINT8 = 1,         //!< Enable Int8 layer selection.
        //     kDEBUG = 2,        //!< Enable debugging of layers via synchronizing after every layer.
        //     kGPU_FALLBACK = 3, //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.
        //     kSTRICT_TYPES = 4, //!< Enables strict type constraints.
        //     kREFIT = 5,        //!< Enable building a refittable engine.
        // };
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    }
    
    if (mRunMode == 1)
    {
        spdlog::info("setFp16Mode");
        if (!builder->platformHasFastFp16()) {
            spdlog::warn("the platform do not has fast for fp16");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        network->getInput(0)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(0)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(1)->setType(nvinfer1::DataType::kHALF);
    }
    builder->setMaxBatchSize(mBatchSize);
    
    // set the maximum GPU temporary memory which the engine can use at execution time.
    config->setMaxWorkspaceSize(1ULL << 32);
    // config->setMaxWorkspaceSize(1ULL << 29);
    
    spdlog::info("fp16 support: {}",builder->platformHasFastFp16 ());
    spdlog::info("int8 support: {}",builder->platformHasFastInt8 ());
    spdlog::info("Max batchsize: {}",builder->getMaxBatchSize());
    spdlog::info("Max workspace size: {}",config->getMaxWorkspaceSize());
    spdlog::info("Number of DLA core: {}",builder->getNbDLACores());
    //spdlog::info("Max DLA batchsize: {}",builder->getMaxDLABatchSize());
    //spdlog::info("Current use DLA core: {}",config->getDLACore()); // TODO: set DLA core
    spdlog::info("build engine...");
    std::cout << " ***** before buildEngineWithConfig " << std::endl;
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    std::cout << " ***** after buildEngineWithConfig " << std::endl;
    config->destroy();
    if(calibrator){
        delete calibrator;
        calibrator = nullptr;
    }
}

bool Trt::BuildEngineWithCaffe(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& engineFile,
                        const std::vector<std::string>& outputBlobName,
                        const std::vector<std::vector<float>>& calibratorData,
                        int maxBatchSize) {
    mBatchSize = maxBatchSize;
    spdlog::info("build caffe engine with {} and {}", prototxt, caffeModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);
    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
    // if(mPluginFactory != nullptr) {
    //     parser->setPluginFactoryV2(mPluginFactory);
    // }
    // Notice: change here to costom data type
    nvinfer1::DataType type = mRunMode==1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(prototxt.c_str(),caffeModel.c_str(),
                                                                            *network,type);
    
    for(auto& s : outputBlobName) {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }
    spdlog::info("Number of network layers: {}",network->getNbLayers());
    spdlog::info("Number of input: ", network->getNbInputs());
    std::cout << "Input layer: " << std::endl;
    for(int i = 0; i < network->getNbInputs(); i++) {
        std::cout << network->getInput(i)->getName() << " : ";
        Dims dims = network->getInput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j] << "x"; 
        }
        std::cout << "\b "  << std::endl;
    }
    spdlog::info("Number of output: {}",network->getNbOutputs());
    std::cout << "Output layer: " << std::endl;
    for(int i = 0; i < network->getNbOutputs(); i++) {
        std::cout << network->getOutput(i)->getName() << " : ";
        Dims dims = network->getOutput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j] << "x"; 
        }
        std::cout << "\b " << std::endl;
    }
    spdlog::info("parse network done");

    BuildEngine(builder, network, calibratorData, maxBatchSize, mRunMode);

    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);
    
    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}

bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& customOutput,
                      const std::vector<std::vector<float>>& calibratorData,
                      int maxBatchSize) {
    mBatchSize = maxBatchSize;
    spdlog::info("build onnx engine from {}...",onnxModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, mLogger);
    if(!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        spdlog::error("error: could not parse onnx engine");
        return false;
    }
    // change output 
    // nvinfer1::ITensor* origin_output = network->getOutput(network->getNbOutputs()-1);
    // network->unmarkOutput(*origin_output);
    // nvinfer1::ILayer* custom_output = network->getLayer(network->getNbLayers()-2);
    // nvinfer1::ITensor* output_tensor = custom_output->getOutput(0);
    // network->markOutput(*output_tensor);
    // change output 


    BuildEngine(builder, network, calibratorData, maxBatchSize, mRunMode);
    //SaveEngine(engineFile);

    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}

bool Trt::BuildEngineWithUff(const std::string& uffModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& inputTensorNames,
                      const std::vector<std::vector<int>>& inputDims,
                      const std::vector<std::string>& outputTensorNames,
                      const std::vector<std::vector<float>>& calibratorData,
                      int maxBatchSize) {
    mBatchSize = maxBatchSize;
    spdlog::info("build uff engine with {}...", uffModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    assert(parser != nullptr);
    assert(inputTensorNames.size() == inputDims.size());
    //parse input
    for(size_t i=0;i<inputTensorNames.size();i++) {
        nvinfer1::Dims dim;
        dim.nbDims = inputDims[i].size();
        for(int j=0;j<dim.nbDims;j++) {
            dim.d[j] = inputDims[i][j];
        }
        parser->registerInput(inputTensorNames[i].c_str(), dim, nvuffparser::UffInputOrder::kNCHW);
    }
    //parse output
    for(size_t i=0;i<outputTensorNames.size();i++) {
        parser->registerOutput(outputTensorNames[i].c_str());
    }
    if(!parser->parse(uffModel.c_str(), *network, nvinfer1::DataType::kFLOAT)) {
        spdlog::error("error: parse model failed");
    }
    BuildEngine(builder, network, calibratorData, maxBatchSize, mRunMode);
    spdlog::info("serialize engine to {}", engineFile);
    //SaveEngine(engineFile);
    
    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}

void Trt::InitEngine() {
    spdlog::info("init engine  ...");
    mContext = mEngine->createExecutionContext();
    spdlog::info("after engine  ...");
    assert(mContext != nullptr);

    spdlog::info("malloc device memory");
    int nbBindings = mEngine->getNbBindings();
    std::cout << "--------------- nbBingdings: " << nbBindings << std::endl;
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
        std::cout << " bindings:  " << volume(dims) << "  " << mBatchSize << "  " <<  getElementSize(dtype) << std::endl;
        int64_t totalSize = volume(dims) * mBatchSize * getElementSize(dtype);
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
        if(mEngine->bindingIsInput(i)) {
            spdlog::info("input: ");
        } else {
            spdlog::info("output: ");
        }
        spdlog::info("binding bindIndex: {}, name: {}, size in byte: {}",i,name,totalSize);
        spdlog::info("binding dims with {} dimemsion",dims.nbDims);
        for(int j=0;j<dims.nbDims;j++) {
            std::cout << dims.d[j] << " x ";
        }
        std::cout << "\b\b  "<< std::endl;
        mBinding[i] = safeCudaMalloc(totalSize);
        if(mEngine->bindingIsInput(i)) {
            mInputSize++;
        }
    }
}

float Trt::direction_diff(float direction_a, float direction_b){
    float diff = abs(direction_a - direction_b);
    if (diff < pi){
        return diff;
    }else{
        float out = 2*pi - diff;
        return out;
    }
}

int Trt::detemine_point_shape(MarkingPoint point, float* vector){
    float vec_direct = atan2f(vector[1], vector[0]);
    float vec_direct_up = atan2f(-vector[0], vector[1]);
    float vec_direct_down = atan2f(vector[0], -vector[1]);
    if (point.shape < 0.5){
        if (direction_diff(vec_direct, point.d) < BRIDGE_ANGLE_DIFF){
            return 3;
        }
        if (direction_diff(vec_direct_up, point.d) < SEPARATOR_ANGLE_DIFF){
            return 4;
        }
        if (direction_diff(vec_direct_down, point.d) < SEPARATOR_ANGLE_DIFF){
            return 2;
        }
    }else{
        if (direction_diff(vec_direct, point.d) < BRIDGE_ANGLE_DIFF){
            return 1;
        }
        if (direction_diff(vec_direct_up, point.d) < SEPARATOR_ANGLE_DIFF){
            return 5;
        }
    }
    return 0;
}

int Trt::pair_marking_points(MarkingPoint point_a, MarkingPoint point_b){
    float vector_ab[2] = {point_b.x-point_a.x, point_b.y-point_a.y};
    float vector_ba[2] = {point_a.x-point_b.x, point_a.y-point_b.y};
    float norm = powf((vector_ab[0]*vector_ab[0]+vector_ab[1]*vector_ab[1]), 0.5);
    for (int i = 0; i < 2; i++){
        vector_ab[i] = vector_ab[i] / norm;
        vector_ba[i] = vector_ba[i] / norm;
    }
    int point_shape_a = detemine_point_shape(point_a, vector_ab);
    int point_shape_b = detemine_point_shape(point_b, vector_ba);
    if (point_shape_a == 0 || point_shape_b == 0){
        return 0;
    }
    if (point_shape_a == 3 && point_shape_b == 3){
        return 0;
    }
    if (point_shape_a > 3 && point_shape_b > 3){
        return 0;
    }
    if (point_shape_a < 3 && point_shape_b < 3){
        return 0;
    }
    if (point_shape_a > 3){
        return 1;
    }
    if (point_shape_a < 3){
        return -1;
    }
    if (point_shape_a == 3 && point_shape_b < 3){
        return 1;
    }
    if (point_shape_a == 3 && point_shape_b > 3){
        return -1;
    }
}

bool Trt::pass_through_third_point(std::vector<MarkingPoint>& mk_v, int i, int j){
    float x_1 = mk_v[i].x;
    float y_1 = mk_v[i].y;
    float x_2 = mk_v[j].x;
    float y_2 = mk_v[j].y;
    for (int z = 0; z < mk_v.size(); z++){
        if (z == i || z == j)
            continue;
        float x_0 = mk_v[z].x;
        float y_0 = mk_v[z].y;
        float vec1[2] = {x_0-x_1, y_0-y_1};
        float vec2[2] = {x_2-x_0, y_2-y_0};
        float norm1 = powf((vec1[0]*vec1[0]+vec1[1]*vec1[1]), 0.5);
        float norm2 = powf((vec2[0]*vec2[0]+vec2[1]*vec2[1]), 0.5);
        float vec_sum = 0.0;
        for (int m = 0; m < 2; m++){
            vec_sum += (vec1[m] / norm1 + vec2[m] / norm2);
        }
        if (vec_sum > SLOT_SUPPRESSION_DOT_PRODUCT_THRESH)
            return true;
    }
    return false;
}

std::vector<Slot> Trt::inference_slots(std::vector<MarkingPoint>& mk_v){
    int num_detected = mk_v.size();
    std::vector<Slot> slots;
    for (int i = 0; i < num_detected-1; i++){
        for (int j = i+1; j < num_detected; j++){
            MarkingPoint point_i = mk_v[i];
            MarkingPoint point_j = mk_v[j];
            float distance = (point_i.x-point_j.x)*(point_i.x-point_j.x) + (point_i.y-point_j.y)*(point_i.y-point_j.y);
            if ((VSLOT_MIN_DIST <= distance && distance <= VSLOT_MAX_DIST) || (HSLOT_MIN_DIST <= distance && distance <= HSLOT_MAX_DIST)){
                if(pass_through_third_point(mk_v, i, j)){
                    continue;
                }
                int result = pair_marking_points(point_i, point_j);
                Slot slot;
                if (result == 1){
                    slot.x = i;
                    slot.y = j;
                    slots.push_back(slot);
                }
                if (result == -1){
                    slot.x = j;
                    slot.y = i;
                    slots.push_back(slot);
                }
            }
        }
    }
    return slots;
}

std::vector<predicted_point> Trt::non_maximum_suppression(std::vector<predicted_point>& pred_points){
    int size = pred_points.size();
    std::vector<int> suppressed(size, 0);
    for (int i = 0; i < size-1; i++){
        for (int j = i+1; j < size; j++){
            MarkingPoint temp_mk_i = pred_points[i].mp;
            MarkingPoint temp_mk_j = pred_points[j].mp;
            int i_x = (int)temp_mk_i.x;
            int i_y = (int)temp_mk_i.y;
            int j_x = (int)temp_mk_j.x;
            int j_y = (int)temp_mk_j.y;
            if (abs(j_x-i_x)< 0.0625 && abs(j_y-i_y) < 0.0625){
                int idx;
                pred_points[i].score < pred_points[j].score ? idx = i : idx = j;
                suppressed[idx] = 1;
            }
        }
    }
    int num = std::count(suppressed.begin(), suppressed.end(), 1);
    if (num > 0){
        std::vector<predicted_point> unsupres_pred_points;
        for (int i = 0; i < suppressed.size(); i++){
            if (suppressed[i] == 0){
                unsupres_pred_points.push_back(pred_points[i]);
            }
        }
        return unsupres_pred_points;
    }
    return pred_points;
}

std::vector<cv::Point2i> Trt::cal_point(MarkingPoint point, float entrance_len, int h, int w){
    std::vector<cv::Point2i> point_4;
    float cos_a = cosf(point.d);
    float sin_a = sinf(point.d);
    int p0_x = (int)round<float>(w*point.x);
    int p0_y = (int)round<float>(h*point.y);
    point_4.push_back(cv::Point2i(p0_x, p0_y));
    point_4.push_back(cv::Point2i((int)round<float>(p0_x + entrance_len * cos_a), (int)round<float>(p0_y + entrance_len * sin_a)));
    point_4.push_back(cv::Point2i((int)round<float>(p0_x - entrance_len * sin_a), (int)round<float>(p0_x - entrance_len * sin_a)));
    point_4.push_back(cv::Point2i((int)round<float>(p0_x + entrance_len * sin_a), (int)round<float>(p0_y - entrance_len * cos_a)));
    return point_4;
}

void Trt::plot_points(cv::Mat& image, std::vector<predicted_point>& pred_points){
    if (pred_points.size() < 1){
        return;
    }
    int height = image.rows;
    int width = image.cols;
    for (int i = 0;i < pred_points.size(); i++){
        float confidence = pred_points[i].score;
        MarkingPoint marking_point = pred_points[i].mp;
        std::vector<cv::Point2i> points_info = cal_point(marking_point, ENTRANCE_LENGTH_MASK, height, width);
        int p0_x = points_info[0].x;
        int p0_y = points_info[0].y;
        int p1_x = points_info[1].x;
        int p1_y = points_info[1].y;
        int p2_x = points_info[2].x;
        int p2_y = points_info[2].y;
        int p3_x = points_info[3].x;
        int p3_y = points_info[3].y;
        float p4_x = (p1_x - 10 * (p1_x - p0_x)/sqrtf((p1_x - p0_x)*(p1_x - p0_x) + (p1_y - p0_y)*(p1_y - p0_y)));
        float p4_y = (p1_y - 10 * (p1_y - p0_y)/sqrtf((p1_x - p0_x)*(p1_x - p0_x) + (p1_y - p0_y)*(p1_y - p0_y)));
        int p5_x = (int)(round<float>(p4_x - (float)p1_x)*cosf(pi/4) - (p4_y - (float)p1_y)*sinf(pi/4) + p1_x);
        int p5_y = (int)(round<float>(p4_x - (float)p1_x)*sinf(pi/4) + (p4_y - (float)p1_y)*cosf(pi/4) + p1_y);
        int p6_x = (int)(round<float>(p4_x - (float)p1_x)*cosf(-pi/4) - (p4_y - (float)p1_y)*sinf(-pi/4) + p1_x);
        int p6_y = (int)(round<float>(p4_x - (float)p1_x)*sinf(-pi/4) + (p4_y - (float)p1_y)*cosf(-pi/4) + p1_y);
        cv::line(image, cv::Point(p0_x, p0_y), cv::Point(p1_x, p1_y), cv::Scalar(0, 255, 0), 2);
        cv::line(image, cv::Point(p1_x, p1_y), cv::Point(p5_x, p5_y), cv::Scalar(0, 255, 0), 3);
        cv::line(image, cv::Point(p1_x, p1_y), cv::Point(p6_x, p6_y), cv::Scalar(0, 255, 0), 3);

        cv::circle(image, points_info[0], 1, cv::Scalar(0,0,255), 4);
        if (marking_point.shape > 0.5){
            char text[100];
            sprintf(text, "%s:p0", "L");
            cv::putText(image, text, points_info[0], cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 2);
        }else{
            char text[100];
            sprintf(text, "%s:p0", "T");
            cv::putText(image, text, points_info[0], cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 2);
        }
    }
}

std::vector<Slot> Trt::cal_slots(MarkingPoint point1, MarkingPoint point2, cv::Mat& image){
    Slot slot0;
    Slot slot1;
    Slot slot2;
    Slot slot3;
    Slot slot4;
    int height = image.rows;
    int width = image.cols;
    float dis = (point1.x-point2.x)*(point1.x-point2.x) + (point1.y-point2.y)*(point1.y-point2.y);
    float sep_len = 0.0;
    if (dis >= VSLOT_MIN_DIST && dis <= VSLOT_MAX_DIST){
        sep_len = LONG_SEPARATOR_LENGTH;
    }
    if (dis >= HSLOT_MIN_DIST && dis <= HSLOT_MAX_DIST){
        sep_len = SHORT_SEPARATOR_LENGTH;
    }
    float p0_x = width*point1.x;
    float p0_y = height*point1.y;
    slot0.x = (int)p0_x;
    slot0.y = (int)p0_y;
    float p1_x = width*point2.x;
    float p1_y = height*point2.y;
    slot1.x = (int)p1_x;
    slot1.y = (int)p1_y;
    float norm = powf(((p1_x-p0_x)*(p1_x-p0_x) + (p1_y-p0_y)*(p1_y-p0_y)), 0.5);
    std::cout << "norm: " << (p1_x-p0_x)*(p1_x-p0_x) + (p1_y-p0_y)*(p1_y-p0_y) << std::endl;
    float vec0 = (p1_x-p0_x) / norm;
    float vec1 = (p1_y-p0_y) / norm;
    int p2_x = (int)round<float>(p0_x + height * sep_len * vec1);
    int p2_y = (int)round<float>(p0_y - width * sep_len * vec0);
    std::cout << "width * sep_len * vec0: " << width * sep_len * vec0 << std::endl;
    slot2.x = p2_x;
    slot2.y = p2_y;
    int p3_x = (int)round<float>(p1_x + height * sep_len * vec1);
    int p3_y = (int)round<float>(p1_y - width * sep_len * vec0);
    slot3.x = p3_x;
    slot3.y = p3_y;
    int pm_x = (int)round<float>((p1_x+p2_x)/2);
    int pm_y = (int)round<float>((p1_y+p2_y)/2);
    slot4.x = pm_x;
    slot4.y = pm_y;
    std::vector<Slot> slots = {slot0, slot1, slot2, slot3, slot4};
    return slots;
}

void Trt::plot_slots(cv::Mat& image, std::vector<MarkingPoint>& marking_points, std::vector<Slot>& slots){
    if (marking_points.size() < 1 || slots.size() < 1){
        return;
    }
    int num = 0;
    for (int i = 0; i < slots.size(); i++){
        int a = slots[i].x;
        int b = slots[i].y;
        MarkingPoint point_a = marking_points[a];
        MarkingPoint point_b = marking_points[b];
        std::vector<Slot> slots_5 = cal_slots(point_a, point_b, image);
        for (int i = 0; i < slots_5.size(); i++){
            std::cout << " ******* slots_5: " << slots_5[i].x << "  " << slots_5[i].y << std::endl;
        }
        cv::line(image, cv::Point(slots_5[0].x, slots_5[0].y), cv::Point(slots_5[1].x, slots_5[1].y), cv::Scalar(255, 255, 0), 2);
        cv::line(image, cv::Point(slots_5[0].x, slots_5[0].y), cv::Point(slots_5[2].x, slots_5[2].y), cv::Scalar(255, 255, 0), 2);
        cv::line(image, cv::Point(slots_5[1].x, slots_5[1].y), cv::Point(slots_5[3].x, slots_5[3].y), cv::Scalar(255, 255, 0), 2);
        num+=1;
        char text[100];
        sprintf(text, "%d", num);
        cv::putText(image, text, cv::Point(slots_5[4].x, slots_5[4].y), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
    }
}

void Trt::Postprocess1(float thresh, cv::Mat& image){
    nvinfer1::Dims outdim = mBindingDims[1];
    int h = outdim.d[outdim.nbDims-2];
    int w = outdim.d[outdim.nbDims-1];
    int c = outdim.d[outdim.nbDims-3];
    // modify
    float* out_first = out;
    float* out_two = out+4*16*16*sizeof(float);
    for (int i = 0; i < 4*16*16;i++){
        float temp = out_first[i];
        out_first[i] = 1/(1+exp(-temp));
    }
    for (int i = 0; i < 2*16*16;i++){
        float temp = out_two[i];
        out_two[i] = tanh(temp);
    }

    std::vector<predicted_point> predicted_points;
    for (int i = 0; i < h; i++){
        for (int j = 0;j < w; j++){
            float tempvalue = out[i*w+j];
            if (tempvalue >= thresh){
                float xval = (j+out[2*h*w+i*w+j])/w;
                float yval = (i+out[3*h*w+i*w+j])/h;
                // std::cout << " in BOUNDARY_THRESH " << xval << " " << yval << " " << std::endl;
                if (xval >= BOUNDARY_THRESH && xval <= 1-BOUNDARY_THRESH && yval >= BOUNDARY_THRESH && yval <= 1-BOUNDARY_THRESH){
                    float cos_value = out[4*h*w+i*w+j];
                    float sin_value = out[5*h*w+i*w+j];
                    float direction = atan2f(sin_value, cos_value);
                    MarkingPoint mk;
                    mk.x = xval;
                    mk.y = yval;
                    mk.d = direction;
                    mk.shape = out[1*h*w+i*w+j];
                    predicted_point pp;
                    pp.score = out[i*w+j];
                    pp.mp = mk;
                    predicted_points.push_back(pp);
                }
            }
        }
    }
    // for (int i = 0; i < predicted_points.size(); i++){
    //     std::cout << " predicted_points : " << predicted_points[i].score << std::endl;
    // }
    // std::vector<predicted_point> nms_result = non_maximum_suppression(predicted_points);
    // for (int i = 0; i < nms_result.size(); i++){
    //     std::cout << " nms_result : " << nms_result[i].score << std::endl;
    // }
    std::vector<MarkingPoint> mk_v;
    for (int i = 0; i < predicted_points.size(); i++){
        mk_v.push_back(predicted_points[i].mp);
    }
    // for (int i = 0; i < nms_result.size(); i++){
    //     mk_v.push_back(nms_result[i].mp);
    // }
    std::vector<Slot> slots = inference_slots(mk_v);
    for (int i = 0; i < slots.size(); i++){
        std::cout << " slots: " << slots[i].x << " " << slots[i].y << std::endl;
    }
    plot_slots(image, mk_v, slots);
    plot_points(image, predicted_points);

    mBinding.clear();
    mBindingSize.clear();
    mBindingDims.clear();
    mBindingDataType.clear();
    if (out != nullptr){
        free(out);
        out = nullptr;
    }
}

std::vector<Points_xy> from_head_points(float x1, float y1, float x2, float y2, cv::Mat& img){
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    float x1_l = x1+24;
    float y1_l = y1+22;
    float x2_l = x2-24;
    float y2_l = y2-22;
    float x1_r = x2-24;
    float y1_r = y1+22;
    float x2_r = x1+24;
    float y2_r = y2-22;
    float k_l = (y2_l - y1_l) / (x2_l - x1_l);
    float k_r = (y1_r - y2_r) / (x1_r - x2_r);

    float sum_intensity_l = 0.0;
    float sum_intensity_r = 0.0;
    for (int i=0; i < (int)(x2_l-x1_l); i++){
        for (int k=-2;k<2;k++){
            int y = (int)(k_l * i + y1_l + k);
            int x = (int)(i + x1_l);
            if (y > 599){
                y = 599;
            }
            if (x > 599){
                x = 599;
            }
            if (y < 0){
                y = 0;
            }
            if (x < 0){
                x = 0;
            }
            sum_intensity_l+=gray_img.at<float>(y,x);
        }
    }
    for (int i=0; i < (int)(x1_r - x2_r); i++){
        for (int k=-2;k<2;k++){
            int y = (int)(k_r * i + y2_r + k);
            int x = (int)(i + x2_r);
            if (y > 599){
                y = 599;
            }
            if (x > 599){
                x = 599;
            }
            if (y < 0){
                y = 0;
            }
            if (x < 0){
                x = 0;
            }
            sum_intensity_r+=gray_img.at<float>(y,x);
        }
    }
    if (sum_intensity_l > sum_intensity_r){
        Points_xy points_l1;
        Points_xy points_l2;
        points_l1.x = x1_l;
        points_l1.x = y1_l;
        points_l2.x = x2_l;
        points_l2.x = y2_l;
        if (y2_l > y1_l){
            std::vector<Points_xy> outpoints = {points_l1, points_l2};
            return outpoints;
        }
        else{
            std::vector<Points_xy> outpoints = {points_l2, points_l1};
            return outpoints;
        }
    }
    else{
        Points_xy points_r1;
        Points_xy points_r2;
        points_r1.x = x1_l;
        points_r1.x = y1_l;
        points_r2.x = x2_l;
        points_r2.x = y2_l;
        if (y2_r > y1_r){
            std::vector<Points_xy> outpoints = {points_r1, points_r2};
            return outpoints;
        }
        else{
            std::vector<Points_xy> outpoints = {points_r2, points_r1};
            return outpoints;
        }
    }
}

int dcmp(float x)
{
    if(x > eps) return 1;
    return x < -eps ? -1 : 0;
}

float cross(Points_xy a,Points_xy b,Points_xy c) ///叉积
{
    return (a.x-c.x)*(b.y-c.y)-(b.x-c.x)*(a.y-c.y);
}
Points_xy intersection(Points_xy a,Points_xy b,Points_xy c,Points_xy d)
{
    Points_xy p = a;
    float t =((a.x-c.x)*(c.y-d.y)-(a.y-c.y)*(c.x-d.x))/((a.x-b.x)*(c.y-d.y)-(a.y-b.y)*(c.x-d.x));
    p.x +=(b.x-a.x)*t;
    p.y +=(b.y-a.y)*t;
    return p;
}
//计算多边形面积
float PolygonArea(Points_xy p[], int n)
{
    if(n < 3) return 0.0;
    float s = p[0].y * (p[n - 1].x - p[1].x);
    p[n] = p[0];
    for(int i = 1; i < n; ++ i)
        s += p[i].y * (p[i - 1].x - p[i + 1].x);
    return fabs(s * 0.5);
}
float CPIA(Points_xy a[], Points_xy b[], int na, int nb)//ConvexPolygonIntersectArea
{
    Points_xy p[20], tmp[20];
    int tn, sflag, eflag;
    a[na] = a[0], b[nb] = b[0];
    memcpy(p,b,sizeof(Points_xy)*(nb + 1));
    for(int i = 0; i < na && nb > 2; i++)
    {
        sflag = dcmp(cross(a[i + 1], p[0],a[i]));
        for(int j = tn = 0; j < nb; j++, sflag = eflag)
        {
            if(sflag>=0) tmp[tn++] = p[j];
            eflag = dcmp(cross(a[i + 1], p[j + 1],a[i]));
            if((sflag ^ eflag) == -2)
                tmp[tn++] = intersection(a[i], a[i + 1], p[j], p[j + 1]); ///求交点
        }
        memcpy(p, tmp, sizeof(Points_xy) * tn);
        nb = tn, p[nb] = p[0];
    }
    if(nb < 3) return 0.0;
    return PolygonArea(p, nb);
}
float SPIA(PS4 a_, PS4 b_, int na, int nb)///SimplePolygonIntersectArea 调用此函数
{
    std::vector<Points_xy> a = {a_.points1, a_.points2, a_.points3, a_.points4};
    std::vector<Points_xy> b = {b_.points1, b_.points2, b_.points3, b_.points4};
    int i, j;
    Points_xy t1[4], t2[4];
    float res = 0, num1, num2;
    a[na] = t1[0] = a[0], b[nb] = t2[0] = b[0];
    for(i = 2; i < na; i++)
    {
        t1[1] = a[i-1], t1[2] = a[i];
        num1 = dcmp(cross(t1[1], t1[2],t1[0]));
        if(num1 < 0){
            Points_xy temp = t1[1];
            t1[1] = t1[2];
            t1[2] = temp;
        }
        for(j = 2; j < nb; j++)
        {
            t2[1] = b[j - 1], t2[2] = b[j];
            num2 = dcmp(cross(t2[1], t2[2],t2[0]));
            if(num2 < 0){
                Points_xy temp = t2[1];
                t2[1] = t2[2];
                t2[2] = temp;
            }
            res += CPIA(t1, t2, 3, 3) * num1 * num2;
        }
    }
    return res;
}

float cal_iou(PS4 a, PS4 b, int na, int nb){
    float inter_area = SPIA(a, b, na, nb);
    Points_xy a_[4] = {a.points1, a.points2, a.points3, a.points4};
    Points_xy b_[4] = {b.points1, b.points2, b.points3, b.points4};
    float a_area = PolygonArea(a_, na);
    float b_area = PolygonArea(b_, nb);
    float iou = inter_area / (a_area+b_area-inter_area+eps);
    return iou;
}

std::vector<Points_xy> compute_two_points(float angle, Points_xy points1, Points_xy points2){
    Points_xy p1_p2;
    Points_xy point3;
    Points_xy point4;
    float depth = 125;
    p1_p2.x = points2.x - points1.x;
    p1_p2.y = points2.y - points1.y;
    float p1_p2_norm = powf((powf(p1_p2.x, 2) + powf(p1_p2.y, 2)), 0.5);
    if (p1_p2_norm < 200){
        depth = 250;
    }
    p1_p2.x = p1_p2.x / p1_p2_norm;
    p1_p2.y = p1_p2.y / p1_p2_norm;
    float sin_r = sinf(angle / 180.0 * pi);
    float cos_r = cosf(angle / 180.0 * pi);
    point3.x = (p1_p2.x * cos_r + p1_p2.y * sin_r ) * depth + points2.x;
    point3.y = (-p1_p2.x * sin_r + p1_p2.y * cos_r) * depth + points2.y;
    point4.x = (p1_p2.x * cos_r + p1_p2.y * sin_r ) * depth + points1.x;
    point4.y = (-p1_p2.x * sin_r + p1_p2.y * cos_r) * depth + points1.y;
    std::vector<Points_xy> out_p = {point3, point4};
    return out_p;
}


int segment(Points_xy p1, Points_xy p2, Points_xy p3, Points_xy p4){
    int D = 0;
    if (max(p1.x, p2.x)>=min(p3.x, p4.x) && max(p3.x, p4.x)>=min(p1.x, p2.x) && max(p1.y, p2.y)>=min(p3.y, p4.y) && max(p3.y, p4.y)>=min(p1.y, p2.y)){
        if ((cross(p1,p2,p3)*cross(p1,p2,p4)<=0 && cross(p3,p4,p1)*cross(p3,p4,p2)<=0)){
            D = 1;
        }
    }
    return D;
}

PS4 compute_four_points(float angle, Points_xy points1, Points_xy points2){
    std::vector<Points_xy> points34 = compute_two_points(angle, points1, points2);
    std::cout << "-------- " << points34[0].x << " " << points34[0].y << " " << points34[1].x << " " << points34[1].y << std::endl;
    PS4 bbx_ps;
    bbx_ps.points1 = points1;
    bbx_ps.points2 = points2;
    bbx_ps.points3 = points34[0];
    bbx_ps.points4 = points34[1];
    float point_12_min_x = min(points1.x, points2.x);
    float point_12_min_y = min(points1.y, points2.y);
    float point_12_max_x = max(points1.x, points2.x);
    float point_12_max_y = max(points1.y, points2.y);
    float diff_y = abs(points1.y - points2.y);
    std::vector<Points_xy> points34_ex = compute_two_points(angle, points2, points1);
    PS4 bbx_ps_ex;
    bbx_ps_ex.points1 = points1;
    bbx_ps_ex.points2 = points2;
    bbx_ps_ex.points3 = points34_ex[0];
    bbx_ps_ex.points4 = points34_ex[1];
    PS4 bbx_car;
    Points_xy p1 = {x:200.0, y:130.0};
    Points_xy p2 = {x:400.0, y:130.0};
    Points_xy p3 = {x:400.0, y:440.0};
    Points_xy p4 = {x:200.0, y:440.0};
    bbx_car = {p1, p2, p3, p4};
    PS4 bbx_car_real;
    Points_xy p5 = {x:250.0, y:180.0};
    Points_xy p6 = {x:350.0, y:180.0};
    Points_xy p7 = {x:350.0, y:390.0};
    Points_xy p8 = {x:250.0, y:390.0};
    bbx_car_real = {p5, p6, p7, p8};

    float iou_value = cal_iou(bbx_ps, bbx_car, 4, 4);
    float iou_value_ex = cal_iou(bbx_ps_ex, bbx_car, 4, 4);
    float iou_value_real = cal_iou(bbx_ps, bbx_car_real, 4, 4);
    float iou_value_ex_real = cal_iou(bbx_ps_ex, bbx_car_real, 4, 4);

    PS4 pts;
    if (iou_value < iou_value_ex){
        pts = {points1, points2, points34[0], points34[1]};
    }
    else{
        pts = {points2, points1, points34_ex[0], points34_ex[1]};
    }
    if ((diff_y < 70 && point_12_min_y > 300) || diff_y < 30){
        if ((point_12_max_y < 390 && point_12_max_y > 180) || (point_12_min_y > 180 && point_12_min_y < 390) || diff_y < 10){
            if (points34[0].y < 300 || points34[1].y < 300){
                pts = bbx_ps_ex;
            }
            else{
                pts = bbx_ps;
            }
            if (iou_value_real == 21000){
                pts = bbx_ps;
            }
            if (iou_value_ex_real == 21000){
                pts = bbx_ps_ex;
            }
        }
    }
    Points_xy rec_1 = {x:250, y:180};
    Points_xy rec_2 = {x:350, y:390};
    Points_xy rec_3 = {x:350, y:180};
    Points_xy rec_4 = {x:250, y:390};
    int label_inter_1 = segment(points1, points2, rec_1, rec_2);
    int label_inter_2 = segment(points1, points2, rec_3, rec_4);
    if (diff_y > 180){
        if (label_inter_1 || label_inter_2){
            if (points1.y < points2.y){
                if (points1.x > points2.x){
                    pts = bbx_ps;
                }
                else{
                    pts = bbx_ps_ex;
                }
            }
            else{
                if (points2.x > points1.x){
                    pts = bbx_ps_ex;
                }
                else{
                    pts = bbx_ps;
                }
            }
        }
        if (iou_value_real ==  21000 || iou_value_ex_real == 21000){
            if (point_12_min_x > 300){
                if (points34[0].x < 300 && points34[1].x < 300){
                    pts = bbx_ps_ex;
                }
                else{
                    pts = bbx_ps;
                }
            }
            if (point_12_max_x < 300){
                if (points34[0].x > 300 || points34[1].x > 300){
                    pts = bbx_ps_ex;
                }
                else{
                    pts = bbx_ps;
                }
            }
        }
    }
    return pts;
}

void Trt::Postprocess2(cv::Mat& image){
    nvinfer1::Dims outdim = mBindingDims[1];
    int all_boxes = outdim.d[outdim.nbDims-2];
    int len = outdim.d[outdim.nbDims-1];
    int bs = outdim.d[outdim.nbDims-3];

    float resize_ratio = 600.0/416.0;

    std::vector<Points_xy> points_xy_v;
    std::vector<PS> ps_v;

    int num_check = 0;
    int num_boxes = 0;
    for (int i=0; i < all_boxes; i++){
        float x1 = out[i*len+0];
        float y1 = out[i*len+1];
        float x2 = out[i*len+2];
        float y2 = out[i*len+3];
        if (x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0){
            break;
        }
        num_check+=1;
        int cls = (int)out[i*len+4];
        if (cls == 3){
            Points_xy points_xy;
            points_xy.x = (x1+x2)/2.0*resize_ratio;
            points_xy.y = (y1+y2)/2.0*resize_ratio;
            points_xy_v.push_back(points_xy);
        }
        else{
            num_boxes+=1;
        }
    }
    if (num_boxes > 0){
        for (int i = 0; i < num_check; i++){
            float x1 = out[i*len+0]*resize_ratio;
            float y1 = out[i*len+1]*resize_ratio;
            float x2 = out[i*len+2]*resize_ratio;
            float y2 = out[i*len+3]*resize_ratio;
            int cls = (int)out[i*len+4];
            Points_xy points1;
            Points_xy points2;
            PS per_ps;
            float angle = 129;
            if (cls != 3){
                std::vector<Points_xy> points_valid_xy;
                for (int j = 0; j < points_xy_v.size(); j++){
                    if (points_xy_v[j].x > x1 && points_xy_v[j].x < x2 && points_xy_v[j].y > y1 && points_xy_v[j].y < y2){
                        points_valid_xy.push_back(points_xy_v[j]);
                    }
                }
                if (points_valid_xy.size() == 2){
                    points1 = points_valid_xy[0];
                    points2 = points_valid_xy[1];
                }
                else{
                    std::vector<Points_xy> cal_points = from_head_points(x1,y1, x2, y2, image);
                    points1 = cal_points[0];
                    points2 = cal_points[1];
                }
                if (cls == 0){
                    angle = 90;
                }
                if (cls == 1){
                    angle = 67;
                }
                per_ps.points1 = points1;
                per_ps.points2 = points2;
                per_ps.angle = angle;
                ps_v.push_back(per_ps);
            }
        }
    }
    
    if (ps_v.size() == 0){
        std::cout << "no detections" << std::endl;
        return;
    }

    for (int i = 0; i < ps_v.size(); i++){
        PS4 ps4;
        ps4 = compute_four_points(ps_v[i].angle, ps_v[i].points1, ps_v[i].points2);
        cv::Point points[4];
        points[0] = cv::Point((int)ps4.points1.x, (int)ps4.points1.y);
        points[1] = cv::Point((int)ps4.points2.x, (int)ps4.points2.y);
        points[2] = cv::Point((int)ps4.points3.x, (int)ps4.points3.y);
        points[3] = cv::Point((int)ps4.points4.x, (int)ps4.points4.y);
        const cv::Point* pts[] = {points};
        int npts[] = {4};
        cv::polylines(image, pts, npts, 1, true, cv::Scalar(255), 3, 8, 0);
    }
    cv::imwrite("out.jpg", image);

    mBinding.clear();
    mBindingSize.clear();
    mBindingDims.clear();
    mBindingDataType.clear();
    if (out != nullptr){
        free(out);
        out = nullptr;
    }
}

template<typename T> 
std::vector<int> argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos2, int pos1) {return (array[pos1] < array[pos2]);});

	return array_index;
}

float calIOU_softNms(const BboxWithScore& bbox1,const BboxWithScore& bbox2)
{
    float iw = (std::min(bbox1.tx + bbox1.bx / 2.,bbox2.tx + bbox2.bx / 2.) -
                std::max(bbox1.tx - bbox1.bx / 2.,bbox2.tx - bbox2.bx / 2.));
    if (iw < 0)
    {
        return 0.;
    }

    float ih = (std::min(bbox1.ty + bbox1.by / 2.,bbox2.ty + bbox2.by / 2.) -
                std::max(bbox1.ty - bbox1.by / 2.,bbox2.ty - bbox2.by / 2.));

    if (ih < 0)
    {
        return 0.;
    }

    return iw * ih;
}

void softNms(std::vector<BboxWithScore>& bboxes,const int& method,
            const float& sigma,const float& iou_thre,const float& threshold)
{
    if (bboxes.empty())
    {
        return;
    }

    int N = bboxes.size();
    float max_score,max_pos,cur_pos,weight;
    BboxWithScore tmp_bbox,index_bbox;
    for (int i = 0; i < N; ++i)
    {
        max_score = bboxes[i].score;
        max_pos = i;
        tmp_bbox = bboxes[i];
        cur_pos = i + 1;

        while (cur_pos < N)
        {
            if (max_score < bboxes[cur_pos].score)
            {
                max_score = bboxes[cur_pos].score;
                max_pos = cur_pos;
            }
            cur_pos ++;
        }

        bboxes[i] = bboxes[max_pos];

        bboxes[max_pos] = tmp_bbox;
        tmp_bbox = bboxes[i];

        cur_pos = i + 1;

        while (cur_pos < N)
        {
            index_bbox = bboxes[cur_pos];

            // float area = index_bbox.bx * index_bbox.by;
            float iou = calIOU_softNms(tmp_bbox,index_bbox); 
            if (iou <= 0)
            {
                cur_pos++;
                continue;
            }
            // iou /= area;
            if (method == 1) // 当选择的是线性加权法
            {
                if (iou > iou_thre) //对于重叠率满足一定重叠率阈值的进行适当加权，而不像传统的NMS直接删除，softNMS只是进行降低分数
                {
                    weight = 1 - iou;
                } else
                {
                    weight = 1;
                }
            }else if (method == 2) //当选择的是高斯加权
            {
                weight = exp(-(iou * iou) / sigma);
            }else // original NMS
            {
                if (iou > iou_thre)
                {
                    weight = 0;
                }else
                {
                    weight = 1;
                }
            }
            bboxes[cur_pos].score *= weight;
            if (bboxes[cur_pos].score <= threshold)  //对最终的全部检测框的分数进行阈值筛选
            {
                bboxes[cur_pos] = bboxes[N - 1];
                N --;
                cur_pos = cur_pos - 1;
            }
            cur_pos++;
        }
    }

    bboxes.resize(N);
}

bool Traditinal_cmpScore(const BboxWithScore &lsh, const BboxWithScore &rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

void Traditinal_NMS(std::vector<BboxWithScore>& boundingBox_, const float overlap_threshold)
{

    if (boundingBox_.empty()){
        return;
    }
    //对各个候选框根据score的大小进行升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), Traditinal_cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    vector<int> vPick;
    int nPick = 0;
    multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i){
        vScores.insert(pair<float, int>(boundingBox_[i].score, i));
    }
    while (vScores.size() > 0){
        int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
        vPick[nPick] = last;
        nPick += 1;
        for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = max(boundingBox_.at(it_idx).tx, boundingBox_.at(last).tx);
            maxY = max(boundingBox_.at(it_idx).ty, boundingBox_.at(last).ty);
            minX = min(boundingBox_.at(it_idx).bx, boundingBox_.at(last).bx);
            minY = min(boundingBox_.at(it_idx).by, boundingBox_.at(last).by);
            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;

            IOU = (maxX * maxY) / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            if (IOU > overlap_threshold){
                it = vScores.erase(it);    
            }
            else{
                it++;
            }
        }
    }

    vPick.resize(nPick);
    vector<BboxWithScore> tmp_;
    tmp_.resize(nPick);
    for (int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}


std::vector<BboxWithScore> Trt::NMS(){
    nvinfer1::Dims outdim = mBindingDims[mBindingDims.size()-1];
    int numBox = outdim.d[outdim.nbDims-2];
    int numfeature = outdim.d[outdim.nbDims-1];

    std::vector<BboxWithScore> bboxes;

    float* score = (float*)malloc(sizeof(float)*numBox*(numfeature-1));
    for (int i =0; i < numBox; i++){
        BboxWithScore tempbox;
        float x_c = out[i*numfeature+0];
        float y_c = out[i*numfeature+1];
        float w_c = out[i*numfeature+2];
        float h_c = out[i*numfeature+3];
        float conf = out[i*numfeature+4];

        int cate = 0;
        float max_score = 0;
        for (int j = 5; j < numfeature; j++){
            if (out[i*numfeature+j] > max_score){
                max_score = out[i*numfeature+j];
                cate = j-5;
            }
        }
        tempbox.tx = x_c - w_c/2;
        tempbox.ty = y_c - h_c/2;
        tempbox.bx = x_c + w_c/2;
        tempbox.by = y_c + w_c/2;
        tempbox.area = w_c*h_c;
        tempbox.score = conf;
        tempbox.cate = cate;
        bboxes.push_back(tempbox);
    }

    softNms(bboxes, 1, 0.5, 0.5, 0.3);
    // Traditinal_NMS(bboxes, 0.5);
    std::cout << "number is: " << numBox << "  " << bboxes.size() << std::endl;
    for (int u = 0; u < bboxes.size(); u++){
        float score_ = bboxes[u].score;
        if (score_ < 0.3 ){
            continue;
        }
        std::cout << "tx: " << bboxes[u].tx << " ty: " << bboxes[u].ty << " bx: " << bboxes[u].bx << " by: " << bboxes[u].by << 
        " score: " << bboxes[u].score << " category: " << bboxes[u].cate << std::endl;
    }
    return bboxes;
}
