// Simple NVFP4 (e2m1, block-scaled) x NVFP4 -> bf16 GEMM (D = alpha * A * B + beta * C) using
// CUTLASS's 3.x collective-builder API, natively targeting GeForce Blackwell (sm_120) block-scaled
// tensor cores (mma.sync.aligned.block_scale).
//
// NVFP4 packs 4-bit (e2m1) elements in blocks of 16, each block sharing one e4m3 scale factor
// (cutlass::nv_float4_t<float_e2m1_t>). This has 2x the throughput of MXFP8 and 4x fp8 on sm_120.
//
// A: MxK, row-major, nvfp4
// B: KxN, column-major, nvfp4
// C/D: MxN, bf16

#include <cstdlib>
#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "helper.h"

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;  // elements; a 32-wide e2m1 vector is 16 bytes.

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
// GeForce Blackwell (sm_120) tensor cores don't support TMA multicast, so cluster shape
// must stay 1x1x1 (unlike datacenter Blackwell sm_100, which can use e.g. 2x2x1 clusters).
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));

template <typename T>
auto make_iterator(T *ptr) {
  return cute::recast_ptr<T>(ptr);
}

// e2m1 data blocks use a small dynamic range; e4m3/e8m0 scale factors need a positive range.
template <typename Element, typename Layout>
void initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  constexpr int bits = cutlass::sizeof_bits<Element>::value;
  double scope_max, scope_min;
  if constexpr (bits <= 6) {
    scope_max = 2;
    scope_min = -2;
  } else if constexpr (cute::is_same_v<Element, cutlass::float_ue4m3_t> ||
                        cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
    scope_max = 4;
    scope_min = 1;
  } else {
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
}

int run(int m, int n, int k, int warmup_iters, int bench_iters) {
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  LayoutA layout_a = make_layout(make_shape(m, k, 1), stride_a);
  LayoutB layout_b = make_layout(make_shape(n, k, 1), stride_b);
  LayoutC layout_c = make_layout(make_shape(m, n, 1), stride_c);
  LayoutD layout_d = make_layout(make_shape(m, n, 1), stride_d);
  LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_a;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfa;
  cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_b;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfb;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_c;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_d;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_ref_d;

  block_a.reset(cutlass::make_Coord(size(layout_a)));
  block_b.reset(cutlass::make_Coord(size(layout_b)));
  block_c.reset(cutlass::make_Coord(size(layout_c)));
  block_d.reset(cutlass::make_Coord(size(layout_d)));
  block_ref_d.reset(cutlass::make_Coord(size(layout_d)));
  block_sfa.reset(cutlass::make_Coord(size(filter_zeros(layout_sfa))));
  block_sfb.reset(cutlass::make_Coord(size(filter_zeros(layout_sfb))));

  initialize_block(block_a.host_view(), 2021);
  initialize_block(block_b.host_view(), 2022);
  initialize_block(block_c.host_view(), 2023);
  initialize_block(block_sfa.host_view(), 2024);
  initialize_block(block_sfb.host_view(), 2025);

  block_a.sync_device();
  block_b.sync_device();
  block_c.sync_device();
  block_sfa.sync_device();
  block_sfb.sync_device();

  float alpha = 1.0f, beta = 0.0f;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {block_a.device_data(), stride_a, block_b.device_data(), stride_b,
       block_sfa.device_data(), layout_sfa, block_sfb.device_data(), layout_sfb},
      {{alpha, beta}, block_c.device_data(), stride_c, block_d.device_data(), stride_d}};

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness check against CUTLASS's generic (block-scaled) host reference GEMM.
  Tensor tensor_a = make_tensor(make_iterator(block_a.host_data()), layout_a);
  Tensor tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
  Tensor tensor_b = make_tensor(make_iterator(block_b.host_data()), layout_b);
  Tensor tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator, decltype(tensor_a), decltype(tensor_sfa), decltype(tensor_b),
      decltype(tensor_sfb)>
      mainloop_params{tensor_a, tensor_sfa, tensor_b, tensor_sfb};

  auto tensor_c = make_tensor(make_iterator(block_c.host_data()), layout_c);
  auto tensor_ref_d = make_tensor(make_iterator(block_ref_d.host_data()), layout_d);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator, ElementAccumulator, ElementAccumulator, decltype(tensor_c),
      decltype(tensor_ref_d)>
      epilogue_params{alpha, beta, tensor_c, tensor_ref_d};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  block_d.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_ref_d.host_view(), block_d.host_view());
  passed &= cutlass::reference::host::TensorNorm(block_ref_d.host_view()) > 0;
  passed &= cutlass::reference::host::TensorNorm(block_d.host_view()) > 0;

  std::cout << "Correctness: " << (passed ? "PASSED" : "FAILED") << std::endl;
  if (!passed) {
    return -1;
  }

  // Benchmark.
  cutlass::Status status;
  auto result = run_benchmark(
      [&]() {
        status = gemm_op.initialize(arguments, workspace.get());
        status = gemm_op.run();
      },
      warmup_iters, bench_iters);
  CUTLASS_CHECK(status);

  double flops = 2.0 * double(m) * double(n) * double(k);
  double tflops = flops / (result.avg_runtime_ms / 1.0e3) / 1.0e12;

  std::cout << "Problem size: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "Avg latency: " << result.avg_runtime_ms << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOP/s (nvfp4)" << std::endl;

  return 0;
}

int main(int argc, char const **args) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr << "This kernel targets GeForce Blackwell (sm_120/121) block-scaled tensor cores; "
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
