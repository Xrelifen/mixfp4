// Simple fp8 (e4m3) x fp8 (e4m3) -> fp32 GEMM (D = alpha * A * B + beta * C) using CUTLASS's
// 3.x collective-builder API, natively targeting GeForce Blackwell (sm_120) tensor cores.
//
// A: MxK, row-major, e4m3
// B: KxN, column-major, e4m3
// C/D: MxN, column-major, float
// accumulate in float

#include <cstdlib>
#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = float;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;
using ElementScalar = float;

using TileShape = Shape<_128, _128, _64>;
// GeForce Blackwell (sm_120) tensor cores are single-SM only; unlike datacenter
// Blackwell (sm_100), there is no 2-SM cluster MMA, so this must stay 1x1x1.
using ClusterShape = Shape<_1, _1, _1>;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementScalar>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

template <class Element>
void initialize_block(cutlass::DeviceAllocation<Element> &block, uint64_t seed) {
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(2), Element(-2), 0);
}

int run(int m, int n, int k, int warmup_iters, int bench_iters) {
  auto problem_size = make_shape(m, n, k, 1);

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  cutlass::DeviceAllocation<ElementA> block_a(size_t(m) * k);
  cutlass::DeviceAllocation<ElementB> block_b(size_t(k) * n);
  cutlass::DeviceAllocation<ElementC> block_c(size_t(m) * n);
  cutlass::DeviceAllocation<ElementD> block_d(size_t(m) * n);
  cutlass::DeviceAllocation<ElementD> block_ref_d(size_t(m) * n);

  initialize_block(block_a, 2023);
  initialize_block(block_b, 2022);
  initialize_block(block_c, 2021);

  float alpha = 1.0f, beta = 0.0f;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_a.get(), stride_a, block_b.get(), stride_b},
      {{}, block_c.get(), stride_c, block_d.get(), stride_d},
      hw_info};
  arguments.epilogue.thread.alpha = alpha;
  arguments.epilogue.thread.beta = beta;

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness check against CUTLASS's device reference GEMM.
  cutlass::TensorRef ref_a(block_a.get(), LayoutA::packed({m, k}));
  cutlass::TensorRef ref_b(block_b.get(), LayoutB::packed({k, n}));
  cutlass::TensorRef ref_c(block_c.get(), LayoutC::packed({m, n}));
  cutlass::TensorRef ref_d(block_ref_d.get(), LayoutD::packed({m, n}));

  cutlass::reference::device::GemmComplex(
      {m, n, k}, ElementScalar(alpha),
      ref_a, cutlass::ComplexTransform::kNone,
      ref_b, cutlass::ComplexTransform::kNone,
      ElementScalar(beta), ref_c, ref_d,
      ElementAccumulator(0),
      1, size_t(m) * k, size_t(k) * n, size_t(m) * n, size_t(m) * n);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_d.get(), block_d.get(), block_d.size());

  std::cout << "Correctness: " << (passed ? "PASSED" : "FAILED") << std::endl;
  if (!passed) {
    return -1;
  }

  // Benchmark.
  cutlass::Status status;
  auto result = run_benchmark(
      [&]() { status = gemm_op.run(); }, warmup_iters, bench_iters);
  CUTLASS_CHECK(status);

  double flops = 2.0 * double(m) * double(n) * double(k);
  double tflops = flops / (result.avg_runtime_ms / 1.0e3) / 1.0e12;

  std::cout << "Problem size: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "Avg latency: " << result.avg_runtime_ms << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOP/s (fp8)" << std::endl;

  return 0;
}

int main(int argc, char const **args) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (props.major != 12) {
    std::cerr << "This kernel targets GeForce Blackwell (sm_120) tensor cores; "
                  "detected compute capability "
               << props.major << "." << props.minor << std::endl;
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
