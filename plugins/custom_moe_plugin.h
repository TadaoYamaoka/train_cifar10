#pragma once

#include "custom_moe_kernels.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <string>
#include <vector>

class CustomMoePlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit CustomMoePlugin(CustomMoeConfig cfg);
    CustomMoePlugin(const void* data, size_t length);

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext,
        cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;

private:
    CustomMoeConfig mCfg{};
    std::string mNamespace;
};

class CustomMoePluginCreator final : public nvinfer1::IPluginCreator {
public:
    CustomMoePluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFC{};
};
