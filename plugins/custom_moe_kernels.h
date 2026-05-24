#pragma once

#include <NvInfer.h>
#include <cstddef>
#include <cuda_runtime_api.h>

struct CustomMoeConfig {
    int in_features;
    int hidden_features;
    int num_experts;
    int top_k;
};

size_t getCustomMoeWorkspaceSize(const CustomMoeConfig& cfg, int64_t token_count, nvinfer1::DataType dtype);

cudaError_t enqueueCustomMoe(
    const CustomMoeConfig& cfg,
    nvinfer1::DataType dtype,
    int64_t token_count,
    const void* x,
    const float* router_w,
    const void* w1,
    const void* b1,
    const void* w2,
    const void* b2,
    void* y,
    void* workspace,
    cudaStream_t stream);
