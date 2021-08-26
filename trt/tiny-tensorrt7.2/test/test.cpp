/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2020-03-02 15:16:08
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <libgen.h>
#include <math.h>
#include <dirent.h>
#include <cuda_fp16.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

cv::Mat MEAN, STD, IMAGESIZE;
int MODE, USE_MESN, CUDA_INDEX, SAVE_IMAGE;
std::string ENGINE, OUTNODE;


void GetFileNames(std::string path, std::string rege, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            std::string tempname = ptr->d_name;
            if (strstr(tempname.c_str(), rege.c_str()) == nullptr){
                continue;
            }
            filenames.push_back(path + "/" + tempname);
        }
    }
    closedir(pDir);
}


template<typename T>
int loadBinmyself(const std::string& fileName, T* outBuffer, int bytes_size)
{
    std::cout<<std::endl<<"fileName:"<<fileName<<std::endl;
    std::fstream infile(fileName.c_str(), std::ios::in | std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (bytes_size < length){
        length = bytes_size;
    }
    printf(" in loadBinmyself num: %d\n", length/16);

    int stride = 16; 
    for (int i = 0; i < length; i += stride) 
        infile.read((char *)(outBuffer) + i, stride);

    
    // printf("   --------- %f \n", (float)outBuffer[0]);
    infile.close();
    return length / 16;
}




std::vector<cv::String> binnames;
void test_onnx(const std::string& onnxModelpath, const std::vector<std::string> &dataFile) {
    std::string engineFile = ENGINE;
    // const std::vector<std::string> customOutput;
    const std::vector<std::string> customOutput {OUTNODE};
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = MODE;

    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData);

    std::vector<cv::String> images;

    cv::glob(dataFile[0],images,true);//file dir

    size_t firstSize = onnx_net->GetBindingSize(0);
    size_t secondSize = onnx_net->GetBindingSize(1);

    nvinfer1::Dims first_in_dim = onnx_net->GetBindingDims(0);
    int first_shape_size = 1;
    for (int i = 0; i < first_in_dim.nbDims;i++){
        first_shape_size*=first_in_dim.d[i];
    }

    nvinfer1::Dims second_in_dim = onnx_net->GetBindingDims(1);
    int second_shape_size = 1;
    for (int i = 0; i < second_in_dim.nbDims;i++){
        second_shape_size*=second_in_dim.d[i];
    }

    void* indata = malloc(first_shape_size*sizeof(float));
    void* in1 = malloc(second_shape_size*sizeof(int));

    std::vector<void *> inputs;
    inputs.push_back(indata);
    inputs.push_back(in1);
    int points_num;
    int byteSize = first_shape_size*sizeof(float);

    binnames=images;
    for(int num=0;num<images.size();num++)
    {
        // memset(indata, 0.0, byteSize);
        points_num = loadBinmyself<float>(images[num], reinterpret_cast<float*>(inputs[0]), byteSize);   //do inference

        // std::ofstream outfile;
        // outfile.open("pre.bin", std::ios::binary);
        // outfile.write(reinterpret_cast<const char*>(indata), byteSize);
        // outfile.close();

        printf(" in points num: %d\n", points_num);

        (reinterpret_cast<int*>(inputs[1]))[0] = points_num;
        
		if (mode == 0){
            onnx_net->Forward_mult_FP32(inputs, SAVE_IMAGE);
        }else if (mode == 1){
            onnx_net->Forward_mult_FP16(inputs, SAVE_IMAGE);
        }
        printf(" \n inference is finished \n");
    }
    printf("\033[0;1;33;41m average time %f\033[0m\n",totaltime/images.size());   
    free(indata);
    free(in1);  
}

class InputParser{                                                              
    public:                                                                     
        InputParser (int &argc, char **argv){                                   
            for (int i=1; i < argc; ++i)                                        
                this->tokens.push_back(std::string(argv[i]));                   
        }                                                                       
        /// @author iain                                                                                                                                                                     
        const std::string& getCmdOption(const std::string &option) const{       
            std::vector<std::string>::const_iterator itr;                       
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option); 
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){      
                return *itr;                                                    
            }                                                                   
            static const std::string empty_string("");                          
            return empty_string;                                                
        }                                                                       
        /// @author iain                                                        
        bool cmdOptionExists(const std::string &option) const{                  
            return std::find(this->tokens.begin(), this->tokens.end(), option)  
                   != this->tokens.end();                                       
        }                                                                       
    private:                                                                    
        std::vector <std::string> tokens;                                       
};  

int main(int argc, char** argv) {
    cv::FileStorage fs("../test.xml", cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["MEAN"] >> MEAN;
        fs["STD"] >> STD;
        fs["IMAGESIZE"] >> IMAGESIZE;
        fs["MODE"] >> MODE;
        fs["USE_MESN"] >> USE_MESN;
        fs["CUDA_INDEX"] >> CUDA_INDEX;
        fs["SAVE_IMAGE"] >> SAVE_IMAGE;
        fs["ENGINE"] >> ENGINE;
        fs["OUTNODE"] >> OUTNODE;
        fs.release();
    }
    cudaSetDevice(CUDA_INDEX);
    InputParser cmdparams(argc, argv);

    const std::string& trt_path = cmdparams.getCmdOption("--onnx_path");
    const std::string& data_path = cmdparams.getCmdOption("--data_path");
    const std::string& data_path2 = cmdparams.getCmdOption("--data_path2");
    std::vector<std::string> allIn{data_path, data_path2};
    test_onnx(trt_path, allIn);
    
    return 0;
}
