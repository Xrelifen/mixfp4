// Simple int8 x int8 -> int32 GEMM (D = alpha * A * B + beta * C) using CUTLASS's
// classic device::Gemm API targeting Ampere int8 tensor cores (mma.m16n8k32.s8.s8.s32).
//
// A: MxK, row-major, int8_t
// B: KxN, column-major, int8_t
// C/D: MxN, row-major, int32_t
// accumulate in int32_t

#include <cstdlib>
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "helper.h"

using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue = int32_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// Tile shapes for Ampere int8 tensor cores (mma shape 16x8x32). Kept small enough
// that the mainloop's double-buffered shared memory (128x64 + 256x64) * 2 stages =
// 48KB fits within the default 48KB static shared memory limit, so no per-kernel
// opt-in is needed on smaller-shmem GPUs (e.g. GeForce Blackwell's 101KB budget).
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                          LayoutInputA,
                                          ElementInputB,
                                          LayoutInputB,
                                          ElementOutput,
                                          LayoutOutput,
                                          ElementAccumulator,
                                          MMAOp,
                                          SmArch,
                                          ShapeMMAThreadBlock,
                                          ShapeMMAWarp,
                                          ShapeMMAOp,
                                          EpilogueOp,
                                          SwizzleThreadBlock,
                                          NumStages>;

int run(int m, int n, int k, int warmup_iters, int bench_iters) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(8), ElementInputA(-8), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(8), ElementInputB(-8), 0);
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  ElementComputeEpilogue alpha(1);
  ElementComputeEpilogue beta(0);
  int split_k_slices = 1;

  typename Gemm::Arguments arguments{problem_size,
                                      tensor_a.device_ref(),
                                      tensor_b.device_ref(),
                                      tensor_c.device_ref(),
                                      tensor_d.device_ref(),
                                      {alpha, beta},
                                      split_k_slices};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness check against CUTLASS's device reference GEMM.
  cutlass::reference::device::Gemm<ElementInputA,
                                    LayoutInputA,
                                    ElementInputB,
                                    LayoutInputB,
                                    ElementOutput,
                                    LayoutOutput,
                                    ElementComputeEpilogue,
                                    ElementComputeEpilogue>
      gemm_device_reference;

  gemm_device_reference(problem_size,
                         alpha,
                         tensor_a.device_ref(),
                         tensor_b.device_ref(),
                         beta,
                         tensor_c.device_ref(),
                         tensor_ref_d.device_ref());
  CUDA_CHECK(cudaDeviceSynchronize());

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                         tensor_ref_d.host_view());

  std::cout << "Correctness: " << (passed ? "PASSED" : "FAILED") << std::endl;
  if (!passed) {
    return -1;
  }

  // Benchmark.
  auto result = run_benchmark(
      [&]() {
        status = gemm_op();
      },
      warmup_iters,
      bench_iters);
  CUTLASS_CHECK(status);

  double flops = 2.0 * double(m) * double(n) * double(k);
  double tflops = flops / (result.avg_runtime_ms / 1.0e3) / 1.0e12;

  std::cout << "Problem size: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "Avg latency: " << result.avg_runtime_ms << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOP/s (int8)" << std::endl;

  return 0;
}

int main(int argc, char const **args) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (props.major * 10 + props.minor < 80) {
    std::cerr << "This kernel requires a GPU with compute capability >= 8.0 (Ampere int8 tensor cores)."
               << std::endl;
    return 0;
  }

  int m = 4096, n = 4096, k = 4096;
  int warmup_iters = 10;
  int bench_iters = 50;

  if (argc >= 4) {
    m = std::atoi(args[1]);
    n = std::atoi(args[2]);
    k = std::atoi(args[3]);
  }
  if (argc >= 5) {
    bench_iters = std::atoi(args[4]);
  }

  return run(m, n, k, warmup_iters, bench_iters);
}
