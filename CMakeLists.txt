cmake_minimum_required(VERSION 3.23)
project(moe_tensorrt_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Override for your GPU, e.g. -DCMAKE_CUDA_ARCHITECTURES=89 or 90.
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

option(BUILD_CUSTOM_MOE_PLUGIN "Build the experimental CustomMoE TensorRT plugin" OFF)
option(USE_CUTLASS_GROUPED_GEMM "Use CUTLASS grouped GEMM inside the CustomMoE plugin" OFF)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include /usr/include /usr/local/include
  PATH_SUFFIXES include)
find_library(TENSORRT_LIB nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
  PATH_SUFFIXES lib lib64)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
  PATH_SUFFIXES lib lib64)
find_library(TENSORRT_ONNXPARSER_LIB nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
  PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TENSORRT_LIB}")

if(BUILD_CUSTOM_MOE_PLUGIN)
  if(USE_CUTLASS_GROUPED_GEMM AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include/cutlass/cutlass.h")
    message(FATAL_ERROR "USE_CUTLASS_GROUPED_GEMM=ON requires third_party/cutlass. Clone CUTLASS into third_party/cutlass.")
  endif()
  add_library(custom_moe_plugin SHARED
    plugins/custom_moe_plugin.cpp
    plugins/custom_moe_kernels.cu)
  target_include_directories(custom_moe_plugin PRIVATE ${TENSORRT_INCLUDE_DIR} plugins)
  if(USE_CUTLASS_GROUPED_GEMM)
    target_include_directories(custom_moe_plugin PRIVATE
      third_party/cutlass/include
      third_party/cutlass/tools/util/include)
    target_compile_definitions(custom_moe_plugin PRIVATE USE_CUTLASS_GROUPED_GEMM=1)
  endif()
  target_link_libraries(custom_moe_plugin PRIVATE CUDA::cudart ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB})
  target_compile_options(custom_moe_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  set_target_properties(custom_moe_plugin PROPERTIES OUTPUT_NAME "custom_moe_plugin")
endif()

add_executable(moe_trt_infer cpp/infer.cpp)
target_include_directories(moe_trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(moe_trt_infer PRIVATE CUDA::cudart ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB} dl)

add_executable(swin_cifar10_trt_infer cpp/swin_cifar10_infer.cpp)
target_include_directories(swin_cifar10_trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(swin_cifar10_trt_infer PRIVATE CUDA::cudart ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB})

install(TARGETS moe_trt_infer swin_cifar10_trt_infer RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
if(BUILD_CUSTOM_MOE_PLUGIN)
  install(TARGETS custom_moe_plugin RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
endif()
