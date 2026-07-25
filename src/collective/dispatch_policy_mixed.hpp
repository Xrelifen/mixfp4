// A dispatch-policy tag structurally identical to CUTLASS's
// cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled (3rdparty/cutlass/include/cutlass/
// gemm/dispatch_policy.hpp:1487-1494), but a distinct type -- this is what lets
// cutlass::gemm::collective::CollectiveMma<MixedDispatchPolicy, ...> resolve to OUR forked
// mainloop specialization (sm120_blockscaled_mma_tma_mixed.hpp) instead of CUTLASS's vendored
// one, while reusing 100% of CollectiveBuilder's type derivation (see mixed_nvfp4_gemm.cu).
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120TmaWarpSpecializedBlockScaledMixed {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

} // namespace cutlass::gemm
