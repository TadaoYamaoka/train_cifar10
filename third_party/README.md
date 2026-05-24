# Third-party dependencies

CUTLASS is not vendored in this package. Clone it when you enable the grouped
GEMM production path:

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

The included plugin builds without CUTLASS so that the ONNX export and C++
TensorRT integration can be verified first. The file
`plugins/cutlass_grouped_gemm_plan.cuh` describes the exact replacement point
for the FC1/FC2 kernels.
