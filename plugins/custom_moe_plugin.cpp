#include "custom_moe_plugin.h"

#include <cstring>
#include <iostream>

using namespace nvinfer1;

namespace {
constexpr char kPluginName[] = "CustomMoE";
constexpr char kPluginVersion[] = "1";
constexpr int kNbInputs = 6;

int64_t tokenCountFromDims(const Dims& d) {
    if (d.nbDims < 2) return -1;
    int64_t n = 1;
    for (int i = 0; i < d.nbDims - 1; ++i) {
        if (d.d[i] < 0) return -1;
        n *= d.d[i];
    }
    return n;
}

template <typename T>
void write(char*& dst, const T& v) {
    std::memcpy(dst, &v, sizeof(T));
    dst += sizeof(T);
}

template <typename T>
T read(const char*& src) {
    T v{};
    std::memcpy(&v, src, sizeof(T));
    src += sizeof(T);
    return v;
}

int getFieldInt(const PluginFieldCollection* fc, const char* name, int defaultValue) {
    if (!fc) return defaultValue;
    for (int i = 0; i < fc->nbFields; ++i) {
        const PluginField& f = fc->fields[i];
        if (std::strcmp(f.name, name) == 0 && f.data != nullptr) {
            if (f.type == PluginFieldType::kINT32) {
                return *static_cast<const int*>(f.data);
            }
        }
    }
    return defaultValue;
}

}  // namespace

CustomMoePlugin::CustomMoePlugin(CustomMoeConfig cfg) : mCfg(cfg) {}

CustomMoePlugin::CustomMoePlugin(const void* data, size_t length) {
    (void) length;
    const char* p = static_cast<const char*>(data);
    mCfg.in_features = read<int>(p);
    mCfg.hidden_features = read<int>(p);
    mCfg.num_experts = read<int>(p);
    mCfg.top_k = read<int>(p);
}

const char* CustomMoePlugin::getPluginType() const noexcept { return kPluginName; }
const char* CustomMoePlugin::getPluginVersion() const noexcept { return kPluginVersion; }
int32_t CustomMoePlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs CustomMoePlugin::getOutputDimensions(
    int32_t outputIndex,
    const DimsExprs* inputs,
    int32_t nbInputs,
    IExprBuilder& exprBuilder) noexcept {
    (void) exprBuilder;
    if (outputIndex != 0 || nbInputs != kNbInputs) {
        DimsExprs bad{};
        bad.nbDims = 0;
        return bad;
    }
    return inputs[0];
}

bool CustomMoePlugin::supportsFormatCombination(
    int32_t pos,
    const PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (nbInputs != kNbInputs || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs) return false;
    const auto& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR) return false;

    const DataType xType = inOut[0].type;
    const bool xTypeOk = xType == DataType::kHALF || xType == DataType::kFLOAT;
    if (!xTypeOk) return false;

    if (pos == 0) return true;
    if (pos == 1) return desc.type == DataType::kFLOAT;            // router_w stays FP32
    if (pos >= 2 && pos <= 5) return desc.type == xType;            // expert weights/biases
    if (pos == 6) return desc.type == xType;                        // output
    return false;
}

void CustomMoePlugin::configurePlugin(
    const DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
    (void) in;
    (void) nbInputs;
    (void) out;
    (void) nbOutputs;
}

size_t CustomMoePlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs,
    int32_t nbInputs,
    const PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    (void) outputs;
    (void) nbOutputs;
    if (nbInputs != kNbInputs) return 0;
    int64_t M = tokenCountFromDims(inputs[0].dims);
    if (M <= 0) return 0;
    return getCustomMoeWorkspaceSize(mCfg, M, inputs[0].type);
}

int32_t CustomMoePlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    (void) outputDesc;
    int64_t M = tokenCountFromDims(inputDesc[0].dims);
    if (M <= 0) return 1;
    const int d = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    if (d != mCfg.in_features) return 1;

    cudaError_t err = enqueueCustomMoe(
        mCfg,
        inputDesc[0].type,
        M,
        inputs[0],
        static_cast<const float*>(inputs[1]),
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        outputs[0],
        workspace,
        stream);
    return err == cudaSuccess ? 0 : 1;
}

size_t CustomMoePlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int);
}

void CustomMoePlugin::serialize(void* buffer) const noexcept {
    char* p = static_cast<char*>(buffer);
    write<int>(p, mCfg.in_features);
    write<int>(p, mCfg.hidden_features);
    write<int>(p, mCfg.num_experts);
    write<int>(p, mCfg.top_k);
}

void CustomMoePlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt* CustomMoePlugin::clone() const noexcept {
    auto* p = new CustomMoePlugin(mCfg);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void CustomMoePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

const char* CustomMoePlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType CustomMoePlugin::getOutputDataType(int32_t index, const DataType* inputTypes, int32_t nbInputs) const noexcept {
    (void) index;
    (void) nbInputs;
    return inputTypes[0];
}

void CustomMoePlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) noexcept {
    (void) cudnnContext;
    (void) cublasContext;
    (void) gpuAllocator;
}

void CustomMoePlugin::detachFromContext() noexcept {}
int32_t CustomMoePlugin::initialize() noexcept { return 0; }
void CustomMoePlugin::terminate() noexcept {}

CustomMoePluginCreator::CustomMoePluginCreator() {
    mFields.emplace_back(PluginField{"top_k", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"num_experts", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"in_features", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"hidden_features", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"activation", nullptr, PluginFieldType::kCHAR, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

const char* CustomMoePluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* CustomMoePluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
const PluginFieldCollection* CustomMoePluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* CustomMoePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    (void) name;
    CustomMoeConfig cfg{};
    cfg.top_k = getFieldInt(fc, "top_k", -1);
    cfg.num_experts = getFieldInt(fc, "num_experts", -1);
    cfg.in_features = getFieldInt(fc, "in_features", -1);
    cfg.hidden_features = getFieldInt(fc, "hidden_features", -1);
    if (cfg.top_k <= 0 || cfg.num_experts <= 0 || cfg.in_features <= 0 || cfg.hidden_features <= 0) {
        std::cerr << "CustomMoE plugin got invalid attributes. "
                  << "top_k=" << cfg.top_k << " num_experts=" << cfg.num_experts
                  << " in_features=" << cfg.in_features << " hidden_features=" << cfg.hidden_features << std::endl;
        return nullptr;
    }
    auto* plugin = new CustomMoePlugin(cfg);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* CustomMoePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    (void) name;
    auto* plugin = new CustomMoePlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void CustomMoePluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace ? libNamespace : "";
}

const char* CustomMoePluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(CustomMoePluginCreator);
