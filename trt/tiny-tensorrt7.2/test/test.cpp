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

template<class T>
void loadBin(const std::string& fileName, T* outBuffer, int expand){
    std::fstream infile(fileName.c_str(), std::ios::in | std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (expand){
        T* inputBuffer = (T*)malloc(length);
        infile.read((char *)inputBuffer, length);
        int num = length / 16;
        for (int i = 0; i < num;i++){
            outBuffer[i*5+0] = 0.0;
            outBuffer[i*5+1] = inputBuffer[i*4+0];
            outBuffer[i*5+2] = inputBuffer[i*4+1];
            outBuffer[i*5+3] = inputBuffer[i*4+2];
            outBuffer[i*5+4] = inputBuffer[i*4+3];
        }
        free(inputBuffer);
    }else{
        infile.read((char *)outBuffer, length);
    }
    
    infile.close();
}


int loadBinmyself(const std::string& fileName, float* outBuffer, int expand)
{
    std::cout<<std::endl<<"fileName:"<<fileName<<std::endl;
    std::fstream infile(fileName.c_str(), std::ios::in | std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::cout<<"bin file length: "<<length<<std::endl;
    int num = length / 16;

    if (expand)
    {
        float* inputBuffer = (float*)malloc(length);
        infile.read((char *)inputBuffer, length);
        std::cout<<"num: "<<num<<std::endl;
        for (int i = 0; i<num&&i<25000;i++)
        {
            outBuffer[i*5+0] = 0.0;
            outBuffer[i*5+1] = inputBuffer[i*4+0];
            outBuffer[i*5+2] = inputBuffer[i*4+1];
            outBuffer[i*5+3] = inputBuffer[i*4+2];
            outBuffer[i*5+4] = inputBuffer[i*4+3];
        }
        if(num<25000)
        {
            //memset(outBuffer,0.0,sizeof(float)*(20285-num));
            for(int i=num;i<25000;i++)
            {
                outBuffer[i*5+0] = 0.0;
                outBuffer[i*5+1] = 0.0;
                outBuffer[i*5+2] = 0.0;
                outBuffer[i*5+3] = 1.1;
                outBuffer[i*5+4] = 0.0;
            }
        }
        free(inputBuffer);
    }else
    {
            //infile.read((char *)outBuffer, length);
            float* inputBuffer = (float*)malloc(length);
            infile.read((char *)inputBuffer, length);
            int num = length / 4;
            std::cout<<"num: "<<num<<std::endl;
            for (int i = 0; i<num;i++)
            {
                outBuffer[i] = inputBuffer[i];
            }
            free(inputBuffer);
    }  
    infile.close();
    return num;
}


cv::Mat padding(cv::Mat &in, int c, int h, int w, float value, int downright){
    int dim_diff;
    int pad1;
    int pad2;
    if (h <= w){
        dim_diff = w - h;
        cv::Mat out = cv::Mat::zeros(w, w, CV_32FC3);
        if (downright){
            pad1 = 0;
            pad2 = dim_diff;
        }else{
            pad1 = dim_diff / 2;
            pad2 = dim_diff - dim_diff / 2;
        }
        
        for (int i = pad1; i < w-pad2; i++){
            for (int j = 0; j < w; j++){
                for (int z = 0; z < c; z ++){
                    out.at<cv::Vec3f>(i,j)[z] = in.at<cv::Vec3f>(i-pad1,j)[z];
                }
            }
        }
        return out;
    }
    if (h > w){
        dim_diff = h - w;
        cv::Mat out = cv::Mat::zeros(h, h, CV_32FC3);
        if (downright){
            pad1 = 0;
            pad2 = dim_diff;
        }else{
            pad1 = dim_diff / 2;
            pad2 = dim_diff - dim_diff / 2;
        }
        for (int i = 0; i < h; i++){
            for (int j = pad1; j < h-pad2; j++){
                for (int z = 0; z < c; z ++){
                    out.at<cv::Vec3f>(i,j)[z] = in.at<cv::Vec3f>(i,j-pad1)[z];
                }
            }
        }
        return out;
    }

}

void test_caffe(
        const std::string& prototxt, 
        const std::string& caffeModel,
        const std::vector<std::string>& outputBlobName) {
    std::string engineFile = "";
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = 0;
    Trt* caffe_net = new Trt();
    caffe_net->CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, maxBatchSize, mode, calibratorData);
    caffe_net->Forward();
}

std::vector<cv::String> binnames;
void test_onnx(const std::string& onnxModelpath, const std::vector<std::string> &dataFile, const std::string& calibrate_path) {
    std::string engineFile = "";
    // const std::string onnxModelpath = "";
    const std::vector<std::string> customOutput;
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = MODE;
    std::vector<std::string> cali_files;
    GetFileNames(calibrate_path, "bin", cali_files);
    int INPUT_SIZE_INT8=25000*5;
    for (int i = 0; i < cali_files.size();i++)
    {
	    size_t perSize = INPUT_SIZE_INT8*sizeof(float);
        float* indata = (float*)malloc(perSize);
        std::cout<<"cali_files name:"<<cali_files[i]<<std::endl;
        loadBinmyself(cali_files[i], indata, 1);
        std::vector<float> temp_data(indata,indata+INPUT_SIZE_INT8);
        //temp_data.push_back(20285);
       	free(indata);
        calibratorData.push_back(temp_data);
    }

    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData);


    std::vector<cv::String> images;

    cv::glob(dataFile[0],images,true);//file dir
    /*std::string str;
    std::ifstream in("./val.txt");
    if(in.is_open())
    {
        while(!in.eof())
        {
            getline(in,str);
            images.push_back(dataFile[0]+"/"+str+".bin");
        }
    }*/

    binnames=images;
    for(int num=0;num<images.size();num++)
    {
        std::vector<void *> inputs;
        int points_num;
        for (int ind = 0; ind < 1; ind++)
        {
            size_t perSize = onnx_net->GetBindingSize(ind);
            std::cout<< "\nperSize "<<perSize<<std::endl;//250000
            float* indata=nullptr;
            if(mode==1)
                indata = (float*)malloc(perSize*2);
            else
                indata = (float*)malloc(perSize);

            if (ind==0)
            {
                points_num = loadBinmyself(images[num], indata, 1);   //do inference
            }
            inputs.push_back(indata);
		
            perSize = onnx_net->GetBindingSize(1);
            int* in1 = (int*)malloc(perSize);
            if (points_num > 25000){
                in1[0] = 25000;
            }else{
                in1[0] = points_num;
            }
            inputs.push_back(in1);
		
        }
        //float *a=static_cast<float*>(inputs[0]);

        if(mode==1)
            onnx_net->Forward_mult(inputs);//fp16
        else
            onnx_net->Forward_mult_FP32(inputs);//fp32 || int8
        free(static_cast<float*>(inputs[0]));
	    //free(inputs[1]);
    }

    printf("\033[0;1;33;41m average time %f\033[0m\n",totaltime/images.size());     
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
        fs.release();
    }
    cudaSetDevice(CUDA_INDEX);
    InputParser cmdparams(argc, argv);

    // const std::string& onnx_path = cmdparams.getCmdOption("--onnx_path");
    const std::string& trt_path = cmdparams.getCmdOption("--onnx_path");
    const std::string& data_path = cmdparams.getCmdOption("--data_path");
    const std::string& data_path2 = cmdparams.getCmdOption("--data_path2");
    const std::string& calibrate_path = cmdparams.getCmdOption("--calibrate_path");
    std::vector<std::string> allIn{data_path, data_path2};
    test_onnx(trt_path, allIn, calibrate_path);

    // const std::string& prototxt = cmdparams.getCmdOption("--prototxt");         
    // const std::string& caffemodel = cmdparams.getCmdOption("--caffemodel");     
    // const std::string& output_blob = cmdparams.getCmdOption("--output_blob");   
    // std::vector<std::string> outputBlobName;
    // outputBlobName.push_back(output_blob);
    // test_caffe(prototxt,caffemodel,outputBlobName);

    // test_uff("../models/frozen_inference_graph.uff");
    
    return 0;
}
