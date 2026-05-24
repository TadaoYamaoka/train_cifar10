#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace {

class Logger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) std::cerr << "[TRT] " << msg << '\n';
    }
};

#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while (0)

struct Args {
    std::string engine;
    std::string input;
    std::string output;
    int batch = 1;
    int channels = 3;
    int height = 32;
    int width = 32;
    int warmup = 20;
    int iters = 100;
    int topk = 5;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--engine") args.engine = need("--engine");
        else if (key == "--input") args.input = need("--input");
        else if (key == "--output") args.output = need("--output");
        else if (key == "--batch") args.batch = std::stoi(need("--batch"));
        else if (key == "--channels") args.channels = std::stoi(need("--channels"));
        else if (key == "--height") args.height = std::stoi(need("--height"));
        else if (key == "--width") args.width = std::stoi(need("--width"));
        else if (key == "--warmup") args.warmup = std::stoi(need("--warmup"));
        else if (key == "--iters") args.iters = std::stoi(need("--iters"));
        else if (key == "--topk") args.topk = std::stoi(need("--topk"));
        else if (key == "--help" || key == "-h") {
            std::cout << "Usage: swin_cifar10_trt_infer --engine swin.engine [--input input.bin] "
                      << "[--batch 1] [--output logits.bin] [--iters 100] [--topk 5]\n\n"
                      << "input.bin must be raw FP32 NCHW, normalized with CIFAR-10 eval mean/std.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.engine.empty()) throw std::runtime_error("--engine is required");
    if (args.batch <= 0) throw std::runtime_error("--batch must be positive");
    return args;
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), static_cast<std::streamsize>(size));
    return data;
}

size_t dtypeSize(DataType type) {
    switch (type) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case DataType::kBF16: return 2;
#endif
        default: throw std::runtime_error("unsupported TensorRT dtype");
    }
}

int64_t volume(const Dims& dims) {
    int64_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("dynamic dimension was not resolved");
        total *= dims.d[i];
    }
    return total;
}

void fillRandomInput(float* data, size_t count) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) data[i] = dist(gen);
}

void loadInput(const std::string& path, float* data, size_t count) {
    if (path.empty()) {
        fillRandomInput(data, count);
        return;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open input file: " + path);
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(file.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("input file size does not match expected FP32 NCHW tensor size");
    }
}

std::vector<float> tensorToFloatVector(const void* host, DataType dtype, size_t count) {
    std::vector<float> values(count);
    if (dtype == DataType::kFLOAT) {
        const float* ptr = static_cast<const float*>(host);
        std::copy(ptr, ptr + count, values.begin());
    } else if (dtype == DataType::kHALF) {
        const __half* ptr = static_cast<const __half*>(host);
        for (size_t i = 0; i < count; ++i) values[i] = __half2float(ptr[i]);
    } else {
        throw std::runtime_error("top-k printing supports FLOAT or HALF logits only");
    }
    return values;
}

void printTopK(const std::vector<float>& logits, int batch, int classes, int topk) {
    static const char* kLabels[] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    topk = std::min(topk, classes);
    for (int b = 0; b < batch; ++b) {
        const float* row = logits.data() + static_cast<size_t>(b) * classes;
        std::vector<int> order(classes);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + topk, order.end(), [&](int lhs, int rhs) {
            return row[lhs] > row[rhs];
        });
        float maxLogit = *std::max_element(row, row + classes);
        float denom = 0.0f;
        for (int c = 0; c < classes; ++c) denom += std::exp(row[c] - maxLogit);
        std::cout << "sample " << b << ':';
        for (int i = 0; i < topk; ++i) {
            int cls = order[i];
            float prob = std::exp(row[cls] - maxLogit) / denom;
            const char* label = cls >= 0 && cls < 10 ? kLabels[cls] : "class";
            std::cout << ' ' << label << '(' << cls << ")=" << std::fixed << std::setprecision(4) << prob;
        }
        std::cout << '\n';
    }
}

struct Buffer {
    void* device = nullptr;
    void* host = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    Dims shape{};
    bool input = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        Logger logger;
        initLibNvInferPlugins(&logger, "");

        std::vector<char> engineData = readFile(args.engine);
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");

        std::vector<std::string> tensorNames;
        std::string inputName;
        std::vector<std::string> outputNames;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            tensorNames.emplace_back(name);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) inputName = name;
            else outputNames.emplace_back(name);
        }
        if (inputName.empty() || outputNames.empty()) throw std::runtime_error("engine must have one input and at least one output");

        Dims inputShape = engine->getTensorShape(inputName.c_str());
        if (inputShape.nbDims != 4) throw std::runtime_error("expected NCHW input rank 4");
        inputShape.d[0] = args.batch;
        inputShape.d[1] = args.channels;
        inputShape.d[2] = args.height;
        inputShape.d[3] = args.width;
        if (!context->setInputShape(inputName.c_str(), inputShape)) throw std::runtime_error("setInputShape failed");

        cudaStream_t stream{};
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::unordered_map<std::string, Buffer> buffers;
        for (const std::string& name : tensorNames) {
            Buffer buffer;
            buffer.dtype = engine->getTensorDataType(name.c_str());
            buffer.shape = context->getTensorShape(name.c_str());
            buffer.bytes = static_cast<size_t>(volume(buffer.shape)) * dtypeSize(buffer.dtype);
            buffer.input = engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
            CHECK_CUDA(cudaMalloc(&buffer.device, buffer.bytes));
            CHECK_CUDA(cudaHostAlloc(&buffer.host, buffer.bytes, cudaHostAllocDefault));
            if (!context->setTensorAddress(name.c_str(), buffer.device)) {
                throw std::runtime_error("setTensorAddress failed for " + name);
            }
            buffers.emplace(name, buffer);
        }

        Buffer& input = buffers.at(inputName);
        if (input.dtype != DataType::kFLOAT) throw std::runtime_error("this runner expects FP32 network input");
        loadInput(args.input, static_cast<float*>(input.host), input.bytes / sizeof(float));

        auto enqueueOnce = [&]() {
            CHECK_CUDA(cudaMemcpyAsync(input.device, input.host, input.bytes, cudaMemcpyHostToDevice, stream));
            if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
            for (const std::string& outputName : outputNames) {
                Buffer& output = buffers.at(outputName);
                CHECK_CUDA(cudaMemcpyAsync(output.host, output.device, output.bytes, cudaMemcpyDeviceToHost, stream));
            }
        };

        for (int i = 0; i < args.warmup; ++i) enqueueOnce();
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < args.iters; ++i) enqueueOnce();
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::cout << "avg_latency_ms=" << (elapsedMs / std::max(1, args.iters))
                  << " throughput_images_per_s=" << (1000.0 * args.batch * args.iters / elapsedMs) << '\n';

        const std::string& firstOutputName = outputNames.front();
        const Buffer& firstOutput = buffers.at(firstOutputName);
        size_t outputCount = firstOutput.bytes / dtypeSize(firstOutput.dtype);
        if (outputCount % static_cast<size_t>(args.batch) != 0) {
            throw std::runtime_error("output size is not divisible by batch");
        }
        int classes = static_cast<int>(outputCount / static_cast<size_t>(args.batch));
        std::vector<float> logits = tensorToFloatVector(firstOutput.host, firstOutput.dtype, outputCount);
        printTopK(logits, args.batch, classes, args.topk);

        if (!args.output.empty()) {
            std::ofstream out(args.output, std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
            std::cerr << "[OK] wrote FP32 logits to " << args.output << '\n';
        }

        for (auto& item : buffers) {
            cudaFree(item.second.device);
            cudaFreeHost(item.second.host);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] " << exc.what() << '\n';
        return 1;
    }
}