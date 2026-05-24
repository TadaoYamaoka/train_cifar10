#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

class Logger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) std::cerr << "[TRT] " << msg << '\n';
    }
};

#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

struct Args {
    std::string engine;
    std::string plugin;
    std::string input_file;
    std::string output_file;
    int batch = 1;
    int channels = 3;
    int height = 32;
    int width = 32;
    int warmup = 20;
    int iters = 200;
    bool cuda_graph = true;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--engine") a.engine = need("--engine");
        else if (k == "--plugin") a.plugin = need("--plugin");
        else if (k == "--input") a.input_file = need("--input");
        else if (k == "--output") a.output_file = need("--output");
        else if (k == "--batch") a.batch = std::stoi(need("--batch"));
        else if (k == "--channels") a.channels = std::stoi(need("--channels"));
        else if (k == "--height") a.height = std::stoi(need("--height"));
        else if (k == "--width") a.width = std::stoi(need("--width"));
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--iters") a.iters = std::stoi(need("--iters"));
        else if (k == "--no-cuda-graph") a.cuda_graph = false;
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: moe_trt_infer --engine model.engine [--plugin libcustom_moe_plugin.so] "
                      << "[--batch 1] [--input input.bin] [--output logits.bin] [--iters 200] [--no-cuda-graph]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    return a;
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    f.read(data.data(), static_cast<std::streamsize>(size));
    return data;
}

size_t dtypeSize(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case DataType::kBF16: return 2;
#endif
        default: throw std::runtime_error("unsupported tensor dtype");
    }
}

int64_t volume(const Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) throw std::runtime_error("dynamic dimension was not resolved");
        v *= d.d[i];
    }
    return v;
}

void fillRandomFloat(float* p, size_t n) {
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) p[i] = dist(gen);
}

void loadInputFloat(const std::string& file, float* host, size_t count) {
    if (file.empty()) {
        fillRandomFloat(host, count);
        return;
    }
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input file: " + file);
    f.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("input file size does not match expected FP32 tensor size");
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        Logger logger;
        initLibNvInferPlugins(&logger, "");

        void* pluginHandle = nullptr;
        if (!args.plugin.empty()) {
            pluginHandle = dlopen(args.plugin.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!pluginHandle) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
            std::cerr << "[INFO] loaded plugin " << args.plugin << '\n';
        }

        std::vector<char> engineData = readFile(args.engine);
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");

        int nb = engine->getNbIOTensors();
        std::vector<std::string> names;
        names.reserve(nb);
        std::string inputName;
        std::vector<std::string> outputNames;
        for (int i = 0; i < nb; ++i) {
            const char* name = engine->getIOTensorName(i);
            names.emplace_back(name);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) inputName = name;
            else outputNames.emplace_back(name);
        }
        if (inputName.empty()) throw std::runtime_error("no input tensor found");

        Dims inShape = engine->getTensorShape(inputName.c_str());
        if (inShape.nbDims != 4) throw std::runtime_error("expected NCHW input rank 4");
        inShape.d[0] = args.batch;
        inShape.d[1] = args.channels;
        inShape.d[2] = args.height;
        inShape.d[3] = args.width;
        if (!context->setInputShape(inputName.c_str(), inShape)) {
            throw std::runtime_error("setInputShape failed");
        }

        cudaStream_t stream{};
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        struct Buf { void* dev = nullptr; void* host = nullptr; size_t bytes = 0; DataType dtype{}; Dims shape{}; bool isInput = false; };
        std::unordered_map<std::string, Buf> bufs;
        for (const auto& name : names) {
            Buf b;
            b.dtype = engine->getTensorDataType(name.c_str());
            b.shape = context->getTensorShape(name.c_str());
            b.bytes = static_cast<size_t>(volume(b.shape)) * dtypeSize(b.dtype);
            b.isInput = engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
            CHECK_CUDA(cudaMalloc(&b.dev, b.bytes));
            CHECK_CUDA(cudaHostAlloc(&b.host, b.bytes, cudaHostAllocDefault));
            if (!context->setTensorAddress(name.c_str(), b.dev)) {
                throw std::runtime_error("setTensorAddress failed for " + name);
            }
            bufs.emplace(name, b);
            std::cerr << "[INFO] tensor " << name << " bytes=" << b.bytes
                      << " dtype=" << static_cast<int>(b.dtype) << " dims=[";
            for (int j = 0; j < b.shape.nbDims; ++j) std::cerr << (j ? "," : "") << b.shape.d[j];
            std::cerr << "]\n";
        }

        Buf& input = bufs.at(inputName);
        if (input.dtype != DataType::kFLOAT) {
            throw std::runtime_error("runner currently expects FP32 network input; rebuild ONNX with FP32 input");
        }
        loadInputFloat(args.input_file, static_cast<float*>(input.host), input.bytes / sizeof(float));

        auto launchOnce = [&]() {
            CHECK_CUDA(cudaMemcpyAsync(input.dev, input.host, input.bytes, cudaMemcpyHostToDevice, stream));
            if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
            for (const auto& outName : outputNames) {
                Buf& out = bufs.at(outName);
                CHECK_CUDA(cudaMemcpyAsync(out.host, out.dev, out.bytes, cudaMemcpyDeviceToHost, stream));
            }
        };

        for (int i = 0; i < args.warmup; ++i) launchOnce();
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        float elapsedMs = 0.0f;
        if (args.cuda_graph) {
            cudaGraph_t graph{};
            cudaGraphExec_t graphExec{};
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launchOnce();
            CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
#if CUDART_VERSION >= 13000
            CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, 0));
#else
            CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
#endif
            CHECK_CUDA(cudaEventRecord(start, stream));
            for (int i = 0; i < args.iters; ++i) CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
            CHECK_CUDA(cudaGraphExecDestroy(graphExec));
            CHECK_CUDA(cudaGraphDestroy(graph));
        } else {
            CHECK_CUDA(cudaEventRecord(start, stream));
            for (int i = 0; i < args.iters; ++i) launchOnce();
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        }

        std::cout << "avg_latency_ms=" << (elapsedMs / std::max(1, args.iters))
                  << " throughput_images_per_s=" << (1000.0 * args.batch * args.iters / elapsedMs) << '\n';

        if (!args.output_file.empty() && !outputNames.empty()) {
            const Buf& out = bufs.at(outputNames[0]);
            std::ofstream f(args.output_file, std::ios::binary);
            f.write(static_cast<const char*>(out.host), static_cast<std::streamsize>(out.bytes));
            std::cerr << "[OK] wrote " << args.output_file << '\n';
        }

        for (auto& kv : bufs) {
            if (kv.second.dev) cudaFree(kv.second.dev);
            if (kv.second.host) cudaFreeHost(kv.second.host);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        context.reset();
        engine.reset();
        runtime.reset();
        if (pluginHandle) dlclose(pluginHandle);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
}
