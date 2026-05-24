#include "custom_moe_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <type_traits>

namespace {

constexpr int kMaxExperts = 128;
constexpr int kMaxTopK = 8;
constexpr size_t kAlign = 256;

inline size_t alignUp(size_t x, size_t a = kAlign) {
    return (x + a - 1) / a * a;
}

template <typename T>
__device__ inline float readScalar(const T* p) {
    return static_cast<float>(*p);
}

template <>
__device__ inline float readScalar<__half>(const __half* p) {
    return __half2float(*p);
}

template <typename T>
__device__ inline T writeScalar(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline __half writeScalar<__half>(float v) {
    return __float2half_rn(v);
}

template <typename T>
__device__ inline void atomicAddScalar(T* p, float v) {
    atomicAdd(p, static_cast<T>(v));
}

template <>
__device__ inline void atomicAddScalar<__half>(__half* p, float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd(p, __float2half_rn(v));
#else
    // Fallback for old architectures. TensorRT 10 support starts from newer GPUs,
    // but keep a defensive non-atomic store for compile-only bring-up.
    *p = __float2half_rn(__half2float(*p) + v);
#endif
}

__device__ inline float geluExact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

struct WorkspaceParts {
    int* topk_experts;
    float* topk_gates;
    int* counts;
    int* offsets;
    int* write_ptr;
    int* assignment_dst;
    int* token_sorted;
    int* expert_sorted;
    float* gate_sorted;
    void* x_sorted;
    void* h_sorted;
};

WorkspaceParts splitWorkspace(void* workspace, const CustomMoeConfig& cfg, int64_t M, nvinfer1::DataType dtype) {
    char* base = reinterpret_cast<char*>(workspace);
    size_t off = 0;
    const int64_t N = M * cfg.top_k;
    auto take = [&](size_t bytes) -> void* {
        off = alignUp(off);
        void* p = base + off;
        off += bytes;
        return p;
    };
    WorkspaceParts w{};
    w.topk_experts = reinterpret_cast<int*>(take(sizeof(int) * N));
    w.topk_gates = reinterpret_cast<float*>(take(sizeof(float) * N));
    w.counts = reinterpret_cast<int*>(take(sizeof(int) * cfg.num_experts));
    w.offsets = reinterpret_cast<int*>(take(sizeof(int) * (cfg.num_experts + 1)));
    w.write_ptr = reinterpret_cast<int*>(take(sizeof(int) * cfg.num_experts));
    w.assignment_dst = reinterpret_cast<int*>(take(sizeof(int) * N));
    w.token_sorted = reinterpret_cast<int*>(take(sizeof(int) * N));
    w.expert_sorted = reinterpret_cast<int*>(take(sizeof(int) * N));
    w.gate_sorted = reinterpret_cast<float*>(take(sizeof(float) * N));
    size_t elt = dtype == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);
    w.x_sorted = take(elt * N * cfg.in_features);
    w.h_sorted = take(elt * N * cfg.hidden_features);
    return w;
}

template <typename T>
__global__ void routeTopKKernel(
    const T* __restrict__ x,
    const float* __restrict__ router_w,
    int* __restrict__ topk_experts,
    float* __restrict__ topk_gates,
    int64_t M,
    int D,
    int E,
    int K) {
    int64_t token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= M) return;

    float logits[kMaxExperts];
    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        const float* rw = router_w + e * D;
        const T* xp = x + token * D;
        for (int d = 0; d < D; ++d) {
            acc += readScalar<T>(xp + d) * rw[d];
        }
        logits[e] = acc;
    }

    // Select top-k logits. This is equivalent to top-k over softmax probs.
    int best_e[kMaxTopK];
    float best_v[kMaxTopK];
    for (int k = 0; k < K; ++k) {
        best_e[k] = 0;
        best_v[k] = -3.402823466e38F;
    }
    for (int e = 0; e < E; ++e) {
        float v = logits[e];
        for (int k = 0; k < K; ++k) {
            if (v > best_v[k]) {
                for (int j = K - 1; j > k; --j) {
                    best_v[j] = best_v[j - 1];
                    best_e[j] = best_e[j - 1];
                }
                best_v[k] = v;
                best_e[k] = e;
                break;
            }
        }
    }

    float maxv = best_v[0];
    float denom = 0.0f;
    float exps[kMaxTopK];
    for (int k = 0; k < K; ++k) {
        exps[k] = expf(best_v[k] - maxv);
        denom += exps[k];
    }
    for (int k = 0; k < K; ++k) {
        int64_t idx = token * K + k;
        topk_experts[idx] = best_e[k];
        topk_gates[idx] = exps[k] / denom;
    }
}

__global__ void countExpertsKernel(
    const int* __restrict__ topk_experts,
    int* __restrict__ counts,
    int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(counts + topk_experts[i], 1);
}

__global__ void prefixAndResetKernel(
    const int* __restrict__ counts,
    int* __restrict__ offsets,
    int* __restrict__ write_ptr,
    int E) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int sum = 0;
    offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        sum += counts[e];
        offsets[e + 1] = sum;
    }
    for (int e = 0; e < E; ++e) {
        write_ptr[e] = offsets[e];
    }
}

__global__ void packAssignmentsKernel(
    const int* __restrict__ topk_experts,
    const float* __restrict__ topk_gates,
    int* __restrict__ write_ptr,
    int* __restrict__ assignment_dst,
    int* __restrict__ token_sorted,
    int* __restrict__ expert_sorted,
    float* __restrict__ gate_sorted,
    int64_t M,
    int K) {
    int64_t a = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = M * K;
    if (a >= N) return;
    int e = topk_experts[a];
    int dst = atomicAdd(write_ptr + e, 1);
    assignment_dst[a] = dst;
    token_sorted[dst] = static_cast<int>(a / K);
    expert_sorted[dst] = e;
    gate_sorted[dst] = topk_gates[a];
}

template <typename T>
__global__ void copyPackedXKernel(
    const T* __restrict__ x,
    const int* __restrict__ assignment_dst,
    T* __restrict__ x_sorted,
    int64_t M,
    int D,
    int K) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = M * K;
    int64_t total = N * static_cast<int64_t>(D);
    if (linear >= total) return;
    int d = static_cast<int>(linear % D);
    int64_t a = linear / D;
    int dst = assignment_dst[a];
    int64_t token = a / K;
    x_sorted[static_cast<int64_t>(dst) * D + d] = x[token * D + d];
}

template <typename T>
__global__ void fc1GeluKernel(
    const T* __restrict__ x_sorted,
    const T* __restrict__ w1,
    const T* __restrict__ b1,
    const int* __restrict__ expert_sorted,
    T* __restrict__ h_sorted,
    int64_t N,
    int D,
    int H) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * static_cast<int64_t>(H);
    if (linear >= total) return;
    int h = static_cast<int>(linear % H);
    int64_t row = linear / H;
    int e = expert_sorted[row];
    float acc = readScalar<T>(b1 + static_cast<int64_t>(e) * H + h);
    const T* xp = x_sorted + row * D;
    const T* wp = w1 + (static_cast<int64_t>(e) * D * H) + h;
    for (int d = 0; d < D; ++d) {
        acc += readScalar<T>(xp + d) * readScalar<T>(wp + static_cast<int64_t>(d) * H);
    }
    h_sorted[row * H + h] = writeScalar<T>(geluExact(acc));
}

template <typename T>
__global__ void fc2CombineKernel(
    const T* __restrict__ h_sorted,
    const T* __restrict__ w2,
    const T* __restrict__ b2,
    const int* __restrict__ token_sorted,
    const int* __restrict__ expert_sorted,
    const float* __restrict__ gate_sorted,
    T* __restrict__ y,
    int64_t N,
    int H,
    int D) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * static_cast<int64_t>(D);
    if (linear >= total) return;
    int d = static_cast<int>(linear % D);
    int64_t row = linear / D;
    int e = expert_sorted[row];
    int token = token_sorted[row];
    float acc = readScalar<T>(b2 + static_cast<int64_t>(e) * D + d);
    const T* hp = h_sorted + row * H;
    const T* wp = w2 + (static_cast<int64_t>(e) * H * D) + d;
    for (int h = 0; h < H; ++h) {
        acc += readScalar<T>(hp + h) * readScalar<T>(wp + static_cast<int64_t>(h) * D);
    }
    acc *= gate_sorted[row];
    atomicAddScalar<T>(y + static_cast<int64_t>(token) * D + d, acc);
}

template <typename T>
cudaError_t enqueueTyped(
    const CustomMoeConfig& cfg,
    int64_t M,
    const void* x_void,
    const float* router_w,
    const void* w1_void,
    const void* b1_void,
    const void* w2_void,
    const void* b2_void,
    void* y_void,
    void* workspace,
    cudaStream_t stream) {
    const T* x = static_cast<const T*>(x_void);
    const T* w1 = static_cast<const T*>(w1_void);
    const T* b1 = static_cast<const T*>(b1_void);
    const T* w2 = static_cast<const T*>(w2_void);
    const T* b2 = static_cast<const T*>(b2_void);
    T* y = static_cast<T*>(y_void);

    WorkspaceParts ws = splitWorkspace(workspace, cfg, M, std::is_same<T, __half>::value ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    const int64_t N = M * cfg.top_k;
    const int threads = 256;

    cudaMemsetAsync(ws.counts, 0, sizeof(int) * cfg.num_experts, stream);
    cudaMemsetAsync(y, 0, sizeof(T) * M * cfg.in_features, stream);

    routeTopKKernel<T><<<static_cast<unsigned>((M + threads - 1) / threads), threads, 0, stream>>>(
        x, router_w, ws.topk_experts, ws.topk_gates, M, cfg.in_features, cfg.num_experts, cfg.top_k);
    countExpertsKernel<<<static_cast<unsigned>((N + threads - 1) / threads), threads, 0, stream>>>(
        ws.topk_experts, ws.counts, N);
    prefixAndResetKernel<<<1, 1, 0, stream>>>(ws.counts, ws.offsets, ws.write_ptr, cfg.num_experts);
    packAssignmentsKernel<<<static_cast<unsigned>((N + threads - 1) / threads), threads, 0, stream>>>(
        ws.topk_experts, ws.topk_gates, ws.write_ptr, ws.assignment_dst, ws.token_sorted,
        ws.expert_sorted, ws.gate_sorted, M, cfg.top_k);
    copyPackedXKernel<T><<<static_cast<unsigned>((N * cfg.in_features + threads - 1) / threads), threads, 0, stream>>>(
        x, ws.assignment_dst, static_cast<T*>(ws.x_sorted), M, cfg.in_features, cfg.top_k);
    fc1GeluKernel<T><<<static_cast<unsigned>((N * cfg.hidden_features + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<T*>(ws.x_sorted), w1, b1, ws.expert_sorted, static_cast<T*>(ws.h_sorted),
        N, cfg.in_features, cfg.hidden_features);
    fc2CombineKernel<T><<<static_cast<unsigned>((N * cfg.in_features + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<T*>(ws.h_sorted), w2, b2, ws.token_sorted, ws.expert_sorted, ws.gate_sorted,
        y, N, cfg.hidden_features, cfg.in_features);
    return cudaGetLastError();
}

}  // namespace

size_t getCustomMoeWorkspaceSize(const CustomMoeConfig& cfg, int64_t token_count, nvinfer1::DataType dtype) {
    size_t off = 0;
    const int64_t N = token_count * cfg.top_k;
    auto add = [&](size_t bytes) {
        off = alignUp(off);
        off += bytes;
    };
    add(sizeof(int) * N);                          // topk_experts
    add(sizeof(float) * N);                        // topk_gates
    add(sizeof(int) * cfg.num_experts);            // counts
    add(sizeof(int) * (cfg.num_experts + 1));      // offsets
    add(sizeof(int) * cfg.num_experts);            // write_ptr
    add(sizeof(int) * N);                          // assignment_dst
    add(sizeof(int) * N);                          // token_sorted
    add(sizeof(int) * N);                          // expert_sorted
    add(sizeof(float) * N);                        // gate_sorted
    size_t elt = dtype == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);
    add(elt * N * cfg.in_features);                // x_sorted
    add(elt * N * cfg.hidden_features);            // h_sorted
    return alignUp(off);
}

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
    cudaStream_t stream) {
    if (cfg.num_experts > kMaxExperts || cfg.top_k > kMaxTopK || cfg.top_k <= 0) {
        return cudaErrorInvalidValue;
    }
    if (dtype == nvinfer1::DataType::kHALF) {
        return enqueueTyped<__half>(cfg, token_count, x, router_w, w1, b1, w2, b2, y, workspace, stream);
    }
    if (dtype == nvinfer1::DataType::kFLOAT) {
        return enqueueTyped<float>(cfg, token_count, x, router_w, w1, b1, w2, b2, y, workspace, stream);
    }
    return cudaErrorInvalidValue;
}
