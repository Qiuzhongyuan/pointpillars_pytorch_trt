cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
project(tinytrt)

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" OFF)
option(BUILD_TEST "compile test" ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories (${CUDA_INCLUDE_DIRS})

include_directories(spdlog)
include_directories(./)
# include_directories(./plugin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        #-gencode arch=compute_20,code=sm_20
        #-gencode arch=compute_20,code=sm_21
	# -gencode arch=compute_30,code=sm_30
	# -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=compute_50
  -gencode arch=compute_61,code=compute_61
  -gencode arch=compute_70,code=compute_70
  -gencode arch=compute_75,code=compute_75
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )



# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

message(STATUS "
Generated gencode flags: ${CUDA_gencode_flags} 
BUILD_PYTHON : ${BUILD_PYTHON} 
BUILD_TEST : ${BUILD_TEST} 
ENABLE_PRELU_FP16 : ${ENABLE_PRELU_FP16} 
")


file(GLOB_RECURSE test_source
     Trt.cpp
     Int8EntropyCalibrator.cpp
     cuda_tookit.cu
      test/test.cpp
     )

cuda_add_executable(unit_test ${test_source})
target_compile_options(unit_test PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(unit_test nvinfer)
target_link_libraries(unit_test nvinfer_plugin)
target_link_libraries(unit_test nvparsers)
target_link_libraries(unit_test nvonnxparser)
target_link_libraries(unit_test nvcaffe_parser)
target_link_libraries(unit_test ${OpenCV_LIBS})
target_link_libraries(unit_test ${CUDART})
target_link_libraries(unit_test ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(unit_test ${CUDA_curand_LIBRARY})
target_link_libraries(unit_test ${CUDA_LIBRARIES})

set_property(TARGET unit_test PROPERTY CXX_STANDARD 14)


