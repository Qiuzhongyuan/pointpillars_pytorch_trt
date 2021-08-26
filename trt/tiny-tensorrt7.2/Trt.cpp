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
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

using namespace std;
using namespace nvinfer1;


Trt::Trt() {
}

Trt::~Trt() {
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
    InitEngine();
}

static int inum=0;
float totaltime=0.0;


void Trt::Forward_mult_FP32(std::vector<void*>& data, int save_flag) 
{
    assert(data.size() != mInputSize && "The number of data input does not match the model input!! ");
    for (int i = 0; i < data.size(); i++)
    {
        printf("mBindingSize[%d] %d\n", i, mBindingSize[i]);
        CUDA_CHECK(cudaMemcpy(mBinding[i], data[i], mBindingSize[i], cudaMemcpyHostToDevice));
    }

    spdlog::info("\n\n net forward begin");
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mContext->executeV2(&mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[0;1;33;41m times %f\033[0m\n",elapsedTime);totaltime+=elapsedTime;

    string strclone = binnames[inum++];
    std::cout<<"binname :"<<strclone<<std::endl;
    for (int i = mInputSize; i < mBinding.size();i++)
    {   
        float* out_t = (float*)malloc(mBindingSize[i]);
        CUDA_CHECK(cudaMemcpy(out_t, mBinding[i], mBindingSize[i], cudaMemcpyDeviceToHost));
        for(int j=0;j<mBindingSize[i]/sizeof(float)&&j<45;j++)
        {
            if(j%9==0)
                printf("\n");
            printf("%f ", out_t[j]);
        }
        printf("\n");

        if (save_flag){
            std::ofstream outfile;
            char outname[2000];
            string str=strclone;str=str.substr(0,str.find_last_of("."));str=str+"_"+std::to_string(i-mInputSize)+".bin";
            str="../output"+str.substr(str.find_last_of("/"));
            //str="."+str.substr(str.find_last_of("/"));
            std::cout<<"outname: str: "<<str<<std::endl;
            sprintf(outname, str.c_str(), i-mInputSize);
            outfile.open(outname, std::ios::binary);
            outfile.write(reinterpret_cast<const char*>(out_t), mBindingSize[i]);
            outfile.close();
            free(out_t);
            out_t = nullptr;
        }
    } 
}

void Trt::Forward_mult_FP16(std::vector<void*>& data, int save_flag) 
{
    assert(data.size() != mInputSize && "The number of data input does not match the model input!! ");

    
    float* temp_float = nullptr;
    int mBindingSize_int = mBindingSize[0] / 2;
    printf("mBindingSize_int: %d\n", mBindingSize_int);

    CUDA_CHECK(cudaMalloc((void**)&temp_float, mBindingSize_int*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(temp_float, data[0], mBindingSize_int*sizeof(float), cudaMemcpyHostToDevice));
    // print_float(temp_float, 4, 20, 4);
    convertFP32ToFP16(temp_float, static_cast<__half*>(mBinding[0]), mBindingSize_int);
    // print_half(static_cast<__half*>(mBinding[0]), 4, 20, 4);

    // compare_half_float(temp_float, static_cast<__half*>(mBinding[0]), 4, 20000, 4);
    // printf("mBindingSize_int: %d\n", mBindingSize_int);
    CUDA_CHECK(cudaFree(temp_float));
    
    // printf("mBindingSize[1]: %d\n", mBindingSize[1] / 4);
    CUDA_CHECK(cudaMemcpy(mBinding[1], data[1], mBindingSize[1], cudaMemcpyHostToDevice));

    // std::cout << " output add: ";
    // for (int i = 0; i < mBinding.size(); i++){
    //     std::cout << mBinding[i]
    // }
    
    // printf("\nmBinding[0]): %d %d %d  %d  %d  %d \n",mBinding[0], mBinding[1], mBinding[2], mBinding[3], mBinding[4], mBinding[5]);

    spdlog::info("\n\n net forward begin");
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mContext->executeV2(&mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[0;1;33;41m times %f\033[0m\n",elapsedTime);totaltime+=elapsedTime;

    
    string strclone = binnames[inum++];
    std::cout<<"binname :"<<strclone<<std::endl;
    for (int i = mInputSize; i < mBinding.size(); ++i)
    {   
        int out_BindingSize_int = mBindingSize[i] / 2;
        printf("out BindingSize_int: %d %d\n", i - mInputSize, out_BindingSize_int);

        float* out_t = (float*)malloc(out_BindingSize_int * sizeof(float));
        float* temp_float;
        CUDA_CHECK(cudaMalloc((void**)&temp_float, out_BindingSize_int * sizeof(float)));
        convertFP16ToFP32(static_cast<__half*>(mBinding[i]), temp_float, out_BindingSize_int);
        CUDA_CHECK(cudaMemcpy(out_t, temp_float, out_BindingSize_int * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(temp_float));

        for(int j=0;j<out_BindingSize_int && j<90; ++j)
        {
            if(j % 9 == 0) printf("\n");
            printf("%f ", out_t[j]);
        }
        printf("\n");

        if (save_flag){
            std::ofstream outfile;
            char outname[2000];
            string str=strclone;str=str.substr(0,str.find_last_of("."));str=str+"_"+std::to_string(i-mInputSize)+".bin";
            str="../output"+str.substr(str.find_last_of("/"));
            //str="."+str.substr(str.find_last_of("/"));
            std::cout<<"outname: str: "<<str<<std::endl;
            sprintf(outname, str.c_str(), i-mInputSize);
            outfile.open(outname, std::ios::binary);
            outfile.write(reinterpret_cast<const char*>(out_t), mBindingSize[i]);
            outfile.close();
            free(out_t);
            out_t = nullptr;
        }
    } 
    // for (int i = mInputSize+1; i < mBinding.size(); ++i)
    // {   
    //     int out_BindingSize_int = mBindingSize[i]/4;
    //     printf("out BindingSize_int: %d %d\n", i - mInputSize, out_BindingSize_int);

    //     int* out_t = (int*)malloc(out_BindingSize_int * sizeof(int));

    //     CUDA_CHECK(cudaMemcpy(out_t, static_cast<int*>(mBinding[i]), out_BindingSize_int * sizeof(int), cudaMemcpyDeviceToHost));

    //     for(int j=0;j<out_BindingSize_int && j<100; ++j)
    //     {
    //         if(j % 3 == 0) printf("\n");
    //         printf("%d ", out_t[j]);
    //     }
    //     printf("\n");

    //     if (save_flag){
    //         std::ofstream outfile;
    //         char outname[2000];
    //         string str=strclone;str=str.substr(0,str.find_last_of("."));str=str+"_"+std::to_string(i-mInputSize)+".bin";
    //         str="../output"+str.substr(str.find_last_of("/"));
    //         //str="."+str.substr(str.find_last_of("/"));
    //         std::cout<<"outname: str: "<<str<<std::endl;
    //         sprintf(outname, str.c_str(), i-mInputSize);
    //         outfile.open(outname, std::ios::binary);
    //         outfile.write(reinterpret_cast<const char*>(out_t), mBindingSize[i]);
    //         outfile.close();
    //         free(out_t);
    //         out_t = nullptr;
    //     }
    // } 
    
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
        int numout = network->getNbOutputs();
        // network->getOutput(0)->setType(nvinfer1::DataType::kHALF);

        for (int n = 0; n < numout; n++){
            network->getOutput(n)->setType(nvinfer1::DataType::kHALF);
        }
    }
    builder->setMaxBatchSize(mBatchSize);
    
    config->setMaxWorkspaceSize(1ULL << 32);
    
    spdlog::info("fp16 support: {}",builder->platformHasFastFp16 ());
    spdlog::info("int8 support: {}",builder->platformHasFastInt8 ());
    spdlog::info("Max batchsize: {}",builder->getMaxBatchSize());
    spdlog::info("Max workspace size: {}",config->getMaxWorkspaceSize());
    spdlog::info("Number of DLA core: {}",builder->getNbDLACores());
    spdlog::info("Max DLA batchsize: {}",builder->getMaxDLABatchSize());
    spdlog::info("Current use DLA core: {}",config->getDLACore()); // TODO: set DLA core
    spdlog::info("build engine...");
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    config->destroy();
    if(calibrator){
        delete calibrator;
        calibrator = nullptr;
    }
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
    

    if(!customOutput[0].empty()) {
        spdlog::info("unmark original output...");
        for(int i=0;i<network->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = network->getOutput(i);
            network->unmarkOutput(*origin_output);
        }
        spdlog::info("mark custom output...");
        for(int i=0;i<network->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = network->getLayer(i);
            for(int j=0;j<custom_output->getNbOutputs();j++) {
                nvinfer1::ITensor* output_tensor = custom_output->getOutput(j);
                for(size_t k=0; k<customOutput.size();k++) {
                    std::string layer_name(output_tensor->getName());
                    if(layer_name == customOutput[k]) {
                        network->markOutput(*output_tensor);
                        break;
                    }
                }
            }

        }
    }

    BuildEngine(builder, network, calibratorData, maxBatchSize, mRunMode);
    SaveEngine(engineFile);

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
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
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
