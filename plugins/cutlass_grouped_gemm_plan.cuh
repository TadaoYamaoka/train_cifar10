#pragma once

/*
CUTLASS grouped GEMM integration notes for CustomMoE
====================================================

The default custom_moe_kernels.cu implementation is correctness-first: it uses
simple CUDA kernels for routing, packing, FC1, GELU, FC2 and combine. For
production throughput, CMake can enable an alternate path with
USE_CUTLASS_GROUPED_GEMM=ON. That path replaces fc1GeluKernel/fc2CombineKernel
with CUTLASS Grouped GEMM calls for FP16 plugin tensors and keeps the original
CUDA kernels as the fallback for other dtypes or non-CUTLASS builds.

Why CUTLASS Grouped GEMM fits this MoE:
  - After routing, token rows are packed contiguously by expert.
  - Each expert runs GEMM with a different M_e = number of assigned rows.
  - D and H are fixed for a layer, but M_e changes every inference.
  - CUTLASS grouped kernels accept multiple GEMM problems in one launch and
    schedule tiles persistently across the group.

Implemented pipeline:
  1. routeTopKKernel: produce topk_experts/topk_gates.
  2. countExpertsKernel + prefixAndResetKernel: produce offsets[e].
  3. packAssignmentsKernel + copyPackedXKernel: produce x_sorted grouped by expert.
  4. CUTLASS grouped GEMM FC1:
        A_e = x_sorted[offsets[e]:offsets[e+1], D]
        B_e = w1[e, D, H]
        C/D_e = h_sorted[offsets[e]:offsets[e+1], H]
    The current implementation uses a separate addBiasGeluKernel after GEMM.
  5. CUTLASS grouped GEMM FC2:
        A_e = h_sorted[offsets[e]:offsets[e+1], H]
        B_e = w2[e, H, D]
        C/D_e = y_sorted[offsets[e]:offsets[e+1], D]
    6. addBiasGateCombineKernel: add b2[e], multiply gate_sorted[row], and
      scatter/reduce y_sorted back to token-major output.

Key detail: do not copy counts/offsets to host. Instead, create the CUTLASS
problem-size array, pointer arrays and leading-dimension arrays in device
workspace with a small setup kernel. CUTLASS example 24_gemm_grouped stores
problem sizes, matrix pointers and leading dimensions in global memory arrays.

The setupGroupedGemmMetaKernel in custom_moe_kernels.cu follows this pattern:

  setup_grouped_gemm_meta<<<1, E>>>(
      offsets, x_sorted, w1, h_sorted,
      problem_sizes, ptr_A, ptr_B, ptr_C, ptr_D, lda, ldb, ldc, ldd,
      D, H);

  for expert e:
      M_e = offsets[e+1] - offsets[e]
      problem_sizes[e] = {M_e, H, D}
      ptr_A[e] = x_sorted + offsets[e] * D
      ptr_B[e] = w1 + e * D * H
      ptr_C[e] = h_sorted + offsets[e] * H
      ptr_D[e] = h_sorted + offsets[e] * H
      lda[e] = D
      ldb[e] = H
      ldc[e] = H
      ldd[e] = H

The current CUTLASS instantiation targets Ampere FP16:
  ElementA/B/C = cutlass::half_t
  ElementAccumulator = float
  LayoutA/B/C = cutlass::layout::RowMajor
  OpClassTensorOp, Sm80
  ThreadblockShape = GemmShape<128, 128, 32>
  WarpShape        = GemmShape<64, 64, 32>
  InstructionShape = GemmShape<16, 8, 16>

For Hopper/Blackwell, retune the architecture tag and tile shapes. Keep this as
a layer tactic: if M * top_k is very small, the dense-all-experts ONNX export or
the simple CUDA plugin may win; if M_e is large enough, CUTLASS grouped GEMM
should win.
*/
