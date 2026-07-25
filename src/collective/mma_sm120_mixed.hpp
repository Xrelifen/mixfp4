// Mixed E0M3/E2M1 MMA atoms for SM120 block-scaled NVFP4 (m16n8k64, scale_vec::4X, UE4M3 scale).
//
// Structurally identical to CUTLASS's cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t,
// float_e2m1_t, float, float_ue4m3_t, 16> (3rdparty/cutlass/include/cute/arch/mma_sm120.hpp:3185-
// 3236 and its MMA_Traits at 3rdparty/cutlass/include/cute/atom/mma_traits_sm120.hpp:134-165) --
// same register layout, same thread/value layouts, so CuTe's generic partitioning/dispatch
// (make_tiled_mma, cute::gemm, mma_unpack) treats these identically to the stock atom.
//
// PTX has no e0m3 token, so all four sites below compile as plain e2m1 x e2m1; a post-compile
// SASS patch (scripts/patch_mixed_nvfp4_gemm.py) gives sites 1-3 their real target format. Site 0
// (flag_a=0, flag_b=0) is deliberately left unpatched/native.
//
// ---------------------------------------------------------------------------------------------
// WHY THE FORMAT CHOICE IS *NOT* MADE HERE
// ---------------------------------------------------------------------------------------------
// The obvious design -- one atom whose fma() branches four ways on a format flag -- is a large
// performance trap, and the reason is a single hardware fact:
//
//     A PREDICATED-OFF OMMA STILL CONSUMES A TENSOR-PIPE ISSUE SLOT.
//
// The stock SM120 nvfp4 mainloop is tensor-pipe bound: ncu reports sm__inst_executed_pipe_tensor
// at 84% of peak with `math_pipe_throttle` as the dominant warp stall. Throughput is therefore
// very nearly a linear function of how many OMMA instructions are *issued*, useful or not.
//
// ptxas runs its own if-conversion pass over whatever PTX it is handed, and a branch arm holding
// a lone OMMA is a textbook if-conversion candidate -- so it folds any per-mma format choice into
// `@P0 OMMA` / `@!P0 OMMA` pairs no matter how the branch was written. Measured directly: the
// four-way-branching atom's SASS contained 128 `@P0` + 128 `@!P0` OMMAs and *zero* unpredicated
// ones, and ncu measured sm__inst_executed_pipe_tensor at exactly 2x baseline (16,777,216 vs
// 8,388,608) -- one wasted tensor issue for every useful one, i.e. half the machine.
//
// Every per-mma encoding tried lands in the same trap or worse (4096^3, RTX 5090, vs stock
// nvfp4_gemm's 1208 TFLOP/s):
//   - noinline separate functions ......................  ~45 TFLOP/s (device-call ABI per mma)
//   - C++ if/else across 4 asm blocks .................. 508
//   - __shfl_sync to prove warp-uniformity ............. 412  (shuffles cost more than they save)
//   - trivially-uniform blockIdx condition ............. 604  ("hard to prove uniform" was never
//                                                             the problem)
//   - hand-written PTX with real `bra` labels .......... 477  (WARPSYNC.ALL count 264 vs base 3)
//   - same with `bra.uni` asserting non-divergence ..... 504  (WARPSYNC fell to 3, matching
//                                                             baseline exactly, yet throughput
//                                                             did not move -- WARPSYNC was a
//                                                             symptom, not the cost)
//   - flat 4-way compound predicates ................... 357  (the confirming case: no outer real
//                                                             branch leaves *four* predicated
//                                                             OMMAs, and 1208/357 ~ 4x)
//   - `brx.idx` indirect jump table .................... 296  (defeats if-conversion completely --
//                                                             256 OMMAs, zero predicated, tensor
//                                                             issues back to exactly 1x -- but
//                                                             each dispatch costs a constant-bank
//                                                             jump-table LDC + BRX + WARPSYNC.ALL)
//   - ptxas --allow-expensive-optimizations=false ...... 470  (predicates all four instead)
//
// The `brx.idx` result is the decisive one: it proves the tensor-issue model is right (removing
// the predication restored 1x tensor issues exactly) *and* that no per-mma branch encoding can
// win, because the only encoding ptxas will not if-convert is also the most expensive kind of
// branch the hardware has.
//
// So the format choice moved up one level, into the mainloop's gemm_kblock
// (sm120_blockscaled_mma_tma_mixed.hpp), where one branch selects among four *unconditional*
// atoms for a whole cute::gemm at a time. Each arm then holds 16 OMMAs -- far too large for
// ptxas to if-convert -- so exactly one OMMA is issued per useful MMA, and the branch is paid
// once per 16 of them instead of once each. See that file for the granularity this costs.
#pragma once

#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cutlass/numeric_types.h"

// Bit 7 of a UE4M3 scale byte: architecturally unused by the tensor core (verified on hardware in
// tests/mma_intrinsics) and never touched by CUTLASS's scale-factor dataflow, so it is free to
// carry the per-tile format tag.
#define MIXFP4_FLAG_MASK 0x80u

namespace cute {
namespace SM120::BLOCKSCALED {

// One of the four format sites, as an unconditional mma.sync. `Site` is (flag_a | flag_b << 1),
// matching the patcher's position -> format mapping:
//   0 -> e2m1 x e2m1 (native, left unpatched)   2 -> e2m1 x e0m3 (bit 15)
//   1 -> e0m3 x e2m1 (bit 14)                   3 -> e0m3 x e0m3 (bits 14 and 15)
//
// The leading `prmt.b32` is an identity permute that exists purely to keep the four sites'
// instruction sequences textually distinct. Without it the four arms of the mainloop's dispatch
// are byte-identical -- same source registers, same accumulators -- and ptxas tail-merges them
// back into a single shared OMMA, which would leave nothing to patch differently. prmt.b32 d,a,a,
// sel selects four bytes out of {a,a}, so every selector whose nibbles are (0|4),(1|5),(2|6),(3|7)
// in that order computes the identity; the four below are four such encodings. This also gives
// the patcher a reliable way to tell which site an OMMA belongs to (it reads the selector of the
// PRMT that defines each OMMA's SFA operand) rather than relying on address ordering.
template <int Site>
struct SM120_16x8x64_TN_VS_MixedSite {
  static_assert(Site >= 0 && Site < 4, "format site index must be in [0,4)");

  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  // Four identity selectors, one per site.
  static constexpr uint32_t kPrmtSelector[4] = {0x3210u, 0x3214u, 0x3254u, 0x3654u};

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint32_t const& sfa0,
      uint32_t const& sfb0) {
#if defined(__CUDA_ARCH__)
    constexpr uint16_t bid_a = 0;
    constexpr uint16_t tid_a = 0;
    constexpr uint16_t bid_b = 0;
    constexpr uint16_t tid_b = 0;
    constexpr uint32_t sel   = kPrmtSelector[Site];

    asm volatile(
        "{\n"
        ".reg .b32 %sfa_t;\n"
        "prmt.b32 %sfa_t, %14, %14, %20;\n"
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%sfa_t}, {%15, %16}, {%17}, {%18, %19};\n"
        "}\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa0), "h"(bid_a), "h"(tid_a),
          "r"(sfb0), "h"(bid_b), "h"(tid_b),
          "n"(sel));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x64_TN_VS_MixedSite on host.");
#endif
  }
};

// The type the kernel builds its TiledMma from. It is only ever used for partitioning -- the
// mainloop calls the four site atoms directly -- so which site it names is immaterial.
using SM120_16x8x64_TN_VS_Mixed = SM120_16x8x64_TN_VS_MixedSite<0>;

} // namespace SM120::BLOCKSCALED

// Same body as MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t,
// float, float_ue4m3_t, 16>> (mma_traits_sm120.hpp:134-165), hardcoded to that one instantiation
// instead of templated over element types, since these atoms only ever back that specific
// element/scale combination. Identical for all four sites -- that is what lets the mainloop swap
// atoms mid-kernel without repartitioning anything.
template <int Site>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS_MixedSite<Site>> {
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;

  using ValTypeD = float;
  using ValTypeC = float;

  using ValTypeSF = cutlass::float_ue4m3_t;
  constexpr static int SFVecSize = 16;

  using Shape_MNK = Shape<_16,_8,_64>;
  using ThrID     = Layout<_32>;

  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2>>,
                           Stride<Stride<_64,_1>,Stride<_8,_256>>>;
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>,
                           Stride<Stride<_8,_0,_1>,_16>>;
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,
                           Stride<Stride<_0,_1>, _8>>;
  using CLayout   = SM80_16x8_Row;
};

} // namespace cute
