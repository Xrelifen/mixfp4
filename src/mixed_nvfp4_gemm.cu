// Mixed E0M3/E2M1 block-scaled NVFP4 GEMM on SM120 (see
// /home/brian/.claude/plans/mutable-soaring-pascal.md for the full effort).
//
// Same problem setup as src/nvfp4_gemm.cu, but the mainloop is built through our forked
// src/collective/sm120_blockscaled_mma_tma_mixed.hpp (via a substituted DispatchPolicy tag), which
// selects per-granule between E2M1 and E0M3 decoding of each operand. The format flag rides in
// bit 7 of the UE4M3 scale bytes -- architecturally ignored by the tensor core, so it costs no
// extra storage or bandwidth.
//
// PTX cannot spell E0M3, so this binary must be run through scripts/patch_mixed_nvfp4_gemm.py
// before the E0M3 sites do anything: unpatched, all four dispatch sites are plain E2M1 x E2M1 and
// the kernel simply computes ordinary NVFP4.
//
// How CollectiveMainloop is assembled: rather than hand-deriving TiledMma/SmemLayoutAtoms/etc.
// (which involves intricate swizzle and stage-count arithmetic -- see
// cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl), we instantiate the
// *standard* CollectiveBuilder (StdMainloopBuilder below) to get all of that derivation for
// free -- its "using X = ..." lines are public member typedefs, not just internal plumbing, so
// they're directly readable as StdMainloopBuilder::X. We only substitute the two things that
// actually need to differ: the DispatchPolicy tag (which selects our CollectiveMma partial
// specialization) and the MMA atom.

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

#include "collective/dispatch_policy_mixed.hpp"
#include "collective/mma_sm120_mixed.hpp"
#include "collective/sm120_blockscaled_mma_tma_mixed.hpp"

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

// The CTA tile's N sets how many n-atoms a warp covers, and therefore how many B granules a given
// column granularity costs in dispatch bits. At N=128 a warp owns 8 n-atoms, so 16-column granules
// need 4 bits; halving to N=64 makes it 4 atoms and 2 bits, which is the difference between 64 arms
// (outlined) and 16 (clean) for a 16x16x128 granule. The tile is smaller, so the dispatch-free
// ceiling drops too -- measure both before concluding anything.
#ifndef MIXFP4_TILE_N
#define MIXFP4_TILE_N 128
#endif
using ThreadBlockShape = Shape<_128, cute::Int<MIXFP4_TILE_N>, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

// The standard builder: reused purely for its derived types (TiledMma, SmemLayoutAtoms, etc.),
// not instantiated as an actual mainloop -- CollectiveOp below deliberately isn't used.
using StdMainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>;

using MixedDispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaledMixed<
    StdMainloopBuilder::DispatchPolicy::Stages,
    StdMainloopBuilder::DispatchPolicy::SchedulerPipelineStageCount,
    typename StdMainloopBuilder::DispatchPolicy::ClusterShape,
    typename StdMainloopBuilder::DispatchPolicy::Schedule>;

// Our mixed atom in place of StdMainloopBuilder::TiledMma's stock atom, built from the exact same
// AtomLayoutMNK/PermTile{M,N,K} the builder derived (public members of the builder struct) -- so
// thread/value partitioning is identical to the stock kernel, only the atom's fma() differs.
using MixedMmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS_Mixed;
using MixedTiledMma = decltype(cute::make_tiled_mma(
    MixedMmaOp{},
    typename StdMainloopBuilder::AtomLayoutMNK{},
    Tile<typename StdMainloopBuilder::PermTileM,
         typename StdMainloopBuilder::PermTileN,
         typename StdMainloopBuilder::PermTileK>{}));

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    MixedDispatchPolicy,
    ThreadBlockShape,
    cute::tuple<typename StdMainloopBuilder::ElementA, typename StdMainloopBuilder::ElementSF>,
    typename StdMainloopBuilder::StridePairA,
    cute::tuple<typename StdMainloopBuilder::ElementB, typename StdMainloopBuilder::ElementSF>,
    typename StdMainloopBuilder::StridePairB,
    MixedTiledMma,
    typename StdMainloopBuilder::GmemTiledCopyPairA,
    typename StdMainloopBuilder::SmemLayoutAtomsA,
    typename StdMainloopBuilder::SmemCopyAtomsA,
    cute::identity,
    typename StdMainloopBuilder::GmemTiledCopyPairB,
    typename StdMainloopBuilder::SmemLayoutAtomsB,
    typename StdMainloopBuilder::SmemCopyAtomsB,
    cute::identity>;

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

// ------------------------------------------------------------------------------------------
// Mixed-format host reference
// ------------------------------------------------------------------------------------------
// CUTLASS's own block-scaled reference (Gemm3x + GettBlockScalingMainloopParams) decodes every
// operand as E2M1 and treats the whole scale byte as a UE4M3 magnitude. Neither holds here: bit
// 7 of the scale byte is our format tag (the tensor core ignores it -- verified on hardware in
// tests/mma_intrinsics), and a tagged granule is decoded by the tensor core under the E0M3
// codebook instead. So correctness needs its own reference.
//
// E2M1 and E0M3 index the *same* 4-bit nibble, just through different codebooks:
//   E2M1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6      (plus a sign bit)
//   E0M3 magnitudes: 0, 1,   2, 3,   4, 5, 6, 7      (plus a sign bit -- equal-spaced signed
//                                                     integers, i.e. sign-magnitude INT4)
// confirmed on this hardware in 3rdparty/sm120-e0m3-mma/RESULTS.md. Recovering the nibble index
// from an E2M1-decoded value and re-reading it under the E0M3 codebook therefore reproduces
// exactly what a patched E0M3 instruction computes, without needing raw sub-byte access.
static constexpr float kE2m1Magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static float reinterpret_e2m1_as_e0m3(float v) {
  float const mag = std::fabs(v);
  int nibble = 0;
  for (int i = 0; i < 8; ++i) {
    if (mag == kE2m1Magnitudes[i]) { nibble = i; break; }
  }
  return std::signbit(v) ? -float(nibble) : float(nibble);
}

// UE4M3 scale, with the format tag in bit 7 masked off exactly as the tensor core does.
static float decode_scale(uint8_t raw) {
  cutlass::float_ue4m3_t s{};
  s.raw() = uint8_t(raw & 0x7fu);
  return float(s);
}

// ------------------------------------------------------------------------------------------
// Where does a format granule actually live?
// ------------------------------------------------------------------------------------------
// The kernel reads one flag per granule from that granule's first MMA atom, so the host has to
// tag whichever rows/columns that atom covers. Those are NOT simply contiguous: the TiledMma
// carries a PermTileN of Layout<Shape<_8,_2,_2>, Stride<_1,_16,_8>>, which permutes the N
// dimension within each aligned 32-column block, so "n-atom j covers columns 8j..8j+7" is false.
// Assuming it silently produced correct results for every *uniform* tagging (all-E2M1, all-E0M3,
// all-A, all-B -- a permutation of a constant is that same constant) while corrupting every
// genuinely mixed one.
//
// Rather than re-derive the permutation by hand, ask CuTe: partition an identity tensor with the
// very same TiledMma the kernel uses and read off which coordinates each (thread, atom) touches.
// This is correct by construction and stays correct if the tile shape or warp layout changes.
//
// Returns a vector mapping each row (or column) of one CTA tile to a granule id.
template <bool IsA>
std::vector<int> build_granule_map(int atoms_per_granule) {
  MixedTiledMma tiled_mma;
  // Size the identity tensor from the CTA tile, exactly as the mainloop does when it calls
  // partition_fragment_A/B on sA/sB. NOT from tile_shape(tiled_mma): that reports the TiledMma's
  // own tile, whose N is the PermTileN extent (32) rather than the CTA tile's 128, which would
  // silently build a map covering only the first quarter of the columns and then apply it with
  // the wrong period.
  int const mn_extent = IsA ? int(size<0>(ThreadBlockShape{})) : int(size<1>(ThreadBlockShape{}));
  int const k_extent  = int(size<2>(ThreadBlockShape{}));

  std::vector<int> mn_to_granule(size_t(mn_extent), -1);
  Tensor id = make_identity_tensor(make_shape(mn_extent, k_extent));

  // A granule is identified by the *set* of rows/columns it covers, canonicalised to that set's
  // smallest member -- not by the warp that happens to touch it. With AtomLayoutMNK = 4x2 the
  // warp index encodes both an M and an N position, so warps 0 and 4 cover identical rows of A
  // (differing only in which columns of B they take). They must therefore share one A granule,
  // and they do: they read the same scale factors. Keying on the warp id instead would declare
  // that a conflict.
  // A granule is what a whole *warp* covers for one group of atoms -- the flag is read by all 32
  // lanes and drives a warp-wide branch, so the unit is the warp's coverage, not one lane's.
  // Accumulate per (warp, atom-group) across every thread of the warp first, then canonicalise
  // each set to its smallest member. Warps that cover identical rows (with AtomLayoutMNK = 4x2,
  // warps 0 and 4 do) land on the same representative and correctly share one granule.
  int const num_threads = int(size(typename MixedTiledMma::ThrLayoutVMNK{}));
  std::map<std::pair<int, int>, std::set<int>> coverage;  // (warp, group) -> rows/cols
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto frag = [&] {
      if constexpr (IsA) { return thr_mma.partition_A(id); }
      else               { return thr_mma.partition_B(id); }
    }();
    int const num_atoms = int(size<1>(frag));
    for (int atom = 0; atom < num_atoms; ++atom) {
      auto &dst = coverage[{tid / 32, atom / atoms_per_granule}];
      for (int v = 0; v < int(size<0>(frag)); ++v) {
        for (int kk = 0; kk < int(size<2>(frag)); ++kk) {
          int const mn = get<0>(frag(v, atom, kk));
          if (mn >= 0 && mn < mn_extent) { dst.insert(mn); }
        }
      }
    }
  }

  for (auto const &[key, covered] : coverage) {
    if (covered.empty()) { continue; }
    int const rep = *covered.begin();  // std::set is ordered, so this is the minimum
    for (int mn : covered) {
      if (mn_to_granule[size_t(mn)] == -1) {
        mn_to_granule[size_t(mn)] = rep;
      }
      else if (mn_to_granule[size_t(mn)] != rep) {
        std::cerr << "FATAL: " << (IsA ? "row " : "column ") << mn
                  << " belongs to two partially-overlapping format granules (representatives "
                  << mn_to_granule[size_t(mn)] << " and " << rep
                  << "). No host-side tagging can satisfy this kernel at "
                  << atoms_per_granule << " atoms per granule." << std::endl;
        std::exit(1);
      }
    }
  }
  for (int i = 0; i < mn_extent; ++i) {
    if (mn_to_granule[size_t(i)] == -1) {
      std::cerr << "FATAL: " << (IsA ? "row " : "column ") << i
                << " is not covered by any MMA atom" << std::endl;
      std::exit(1);
    }
  }
  return mn_to_granule;
}

enum class TagMode { kNone, kRandom, kAllA, kAllB, kAll };

static TagMode parse_tag_mode() {
  const char *e = std::getenv("MIXFP4_TAG");
  if (e == nullptr) { return TagMode::kRandom; }
  std::string s{e};
  if (s == "none" || s == "0") { return TagMode::kNone; }
  if (s == "a")                { return TagMode::kAllA; }
  if (s == "b")                { return TagMode::kAllB; }
  if (s == "all")              { return TagMode::kAll; }
  return TagMode::kRandom;
}

static const char *tag_mode_name(TagMode t) {
  switch (t) {
    case TagMode::kNone:   return "none (all E2M1 -- site 0 only)";
    case TagMode::kAllA:   return "all-A (E0M3 x E2M1 -- site 1 only)";
    case TagMode::kAllB:   return "all-B (E2M1 x E0M3 -- site 2 only)";
    case TagMode::kAll:    return "all (E0M3 x E0M3 -- site 3 only)";
    default:               return "random per granule (all four sites)";
  }
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

  // ------------------------------------------------------------------------------------------
  // Format tagging: set bit 7 of the scale bytes to mark E0M3 granules.
  // ------------------------------------------------------------------------------------------
  // The granule must match what the kernel honours (see
  // collective/sm120_blockscaled_mma_tma_mixed.hpp). The kernel reads one flag per granule from
  // that granule's first atom and applies it to the whole granule, so every scale byte in a
  // granule must carry the same bit. Tagging finer than the granule is not merely inaccurate:
  // below one atom it makes lanes of a warp disagree and runs mma.sync.aligned partially
  // converged, which is undefined behaviour. Build with -DMIXFP4_DEBUG_UNIFORMITY=1 to catch it.
  constexpr int kAGranuleRows = MIXFP4_A_ATOMS_PER_GRANULE * 16;  // atom M is 16
  constexpr int kBGranuleCols = MIXFP4_B_ATOMS_PER_GRANULE * 8;   // atom N is 8
  // The C++ path reads the flag once per k_tile (from k_block 0), so K granularity is the whole
  // k_tile. The generated PTX path dispatches per k_block and publishes its own K granule, which
  // is 64 -- one mma.sync. Tagging coarser than the kernel reads is always safe; finer is not.
// A and B can carry different K granules: joint-K-B spends its arm budget on giving B (the weight
// operand) a 64-element K granule while A stays at the k_tile.
#if defined(MIXFP4_K_GRANULE_A)
  constexpr int kKGranuleA    = MIXFP4_K_GRANULE_A;
#elif defined(MIXFP4_K_GRANULE)
  constexpr int kKGranuleA    = MIXFP4_K_GRANULE;
#else
  constexpr int kKGranuleA    = size<2>(ThreadBlockShape{});      // one k_tile
#endif
#if defined(MIXFP4_K_GRANULE_B)
  constexpr int kKGranuleB    = MIXFP4_K_GRANULE_B;
#elif defined(MIXFP4_K_GRANULE)
  constexpr int kKGranuleB    = MIXFP4_K_GRANULE;
#else
  constexpr int kKGranuleB    = size<2>(ThreadBlockShape{});
#endif
  static_assert(kKGranuleA > 0 && int(size<2>(ThreadBlockShape{})) % kKGranuleA == 0 &&
                kKGranuleB > 0 && int(size<2>(ThreadBlockShape{})) % kKGranuleB == 0,
                "K granule must divide the CTA tile's K");

  TagMode const tag_mode = parse_tag_mode();
  {
    // granule_map tells us, for each row/column within one CTA tile, which granule it belongs to
    // (accounting for the TiledMma's N permutation). Two rows/columns in the same granule of the
    // same tile and the same k_tile must always get the same flag.
    auto tag = [](auto tensor, int mn_extent, int k_extent, std::vector<int> const &granule_map,
                  int k_granule, uint64_t seed, bool force) {
      int const tile_mn = int(granule_map.size());
      for (int mn = 0; mn < mn_extent; ++mn) {
        int const granule_id =
            (mn / tile_mn) * 100000 + granule_map[size_t(mn % tile_mn)];
        for (int kk = 0; kk < k_extent; ++kk) {
          bool set = force;
          if (!force) {
            uint64_t h = seed;
            h = h * 6364136223846793005ull + uint64_t(granule_id) + 1;
            h ^= h >> 29;
            h = h * 6364136223846793005ull + uint64_t(kk / k_granule) + 1;
            h ^= h >> 29;
            set = (h & 1ull) != 0;
          }
          if (set) { tensor(mn, kk, 0).raw() |= 0x80; }
        }
      }
    };
    std::vector<int> const a_granule_map =
        build_granule_map<true>(MIXFP4_A_ATOMS_PER_GRANULE);
    std::vector<int> const b_granule_map =
        build_granule_map<false>(MIXFP4_B_ATOMS_PER_GRANULE);
    bool const rand_a  = (tag_mode == TagMode::kRandom);
    bool const rand_b  = (tag_mode == TagMode::kRandom);
    bool const force_a = (tag_mode == TagMode::kAllA || tag_mode == TagMode::kAll);
    bool const force_b = (tag_mode == TagMode::kAllB || tag_mode == TagMode::kAll);
    if (rand_a || force_a) {
      tag(make_tensor(block_sfa.host_data(), layout_sfa), m, k, a_granule_map, kKGranuleA,
          0x9e3779b97f4a7c15ull, force_a);
    }
    if (rand_b || force_b) {
      tag(make_tensor(block_sfb.host_data(), layout_sfb), n, k, b_granule_map, kKGranuleB,
          0xbf58476d1ce4e5b9ull, force_b);
    }
  }
  if (std::getenv("MIXFP4_PRINT_MAP") != nullptr) {
    auto show = [](const char *name, std::vector<int> const &mp) {
      std::cout << name << " (index -> granule rep):";
      for (size_t i = 0; i < mp.size(); ++i) {
        if (i % 16 == 0) { std::cout << "\n  [" << i << "]\t"; }
        std::cout << mp[i] << " ";
      }
      std::cout << std::endl;
    };
    show("A row map", build_granule_map<true>(MIXFP4_A_ATOMS_PER_GRANULE));
    show("B col map", build_granule_map<false>(MIXFP4_B_ATOMS_PER_GRANULE));
    return 0;
  }
  std::cout << "Format tagging: " << tag_mode_name(tag_mode)
            << "; granule = " << kAGranuleRows << " rows of A x " << kKGranuleA << " K"
            << ", " << kBGranuleCols << " cols of B x " << kKGranuleB << " K" << std::endl;

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

  // ------------------------------------------------------------------------------------------
  // Correctness against the mixed-format host reference.
  // ------------------------------------------------------------------------------------------
  Tensor tensor_a   = make_tensor(make_iterator(block_a.host_data()), layout_a);
  Tensor tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
  Tensor tensor_b   = make_tensor(make_iterator(block_b.host_data()), layout_b);
  Tensor tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);
  auto tensor_c     = make_tensor(make_iterator(block_c.host_data()), layout_c);

  // The O(M*N*K) host reference is the long pole past ~2k, and pure throughput runs do not need
  // it. MIXFP4_SKIP_REF=1 reports the kernel as unverified rather than silently claiming a pass.
  bool const skip_ref = [] {
    const char *e = std::getenv("MIXFP4_SKIP_REF");
    return e != nullptr && std::atoi(e) != 0;
  }();
  if (skip_ref) {
    std::cout << "Correctness: SKIPPED (MIXFP4_SKIP_REF=1)" << std::endl;
  }

  // Decode both operands to scaled floats first, honouring each granule's format, then do a
  // plain float GEMM. Decoding up front turns an O(M*N*K) inner loop full of sub-byte and
  // scale-factor address arithmetic into O(M*K + N*K) of it, which is what makes reference sizes
  // past ~1k tractable at all.
  double num = 0.0, den = 0.0, max_abs_err = 0.0;
  std::vector<float> a_dec(skip_ref ? 0 : size_t(m) * size_t(k));
  std::vector<float> b_dec(skip_ref ? 0 : size_t(n) * size_t(k));
  if (!skip_ref) {

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
      uint8_t const sf = tensor_sfa(i, kk, 0).raw();
      float v = float(static_cast<cutlass::float_e2m1_t>(tensor_a(i, kk, 0)));
      if (sf & 0x80u) { v = reinterpret_e2m1_as_e0m3(v); }
      a_dec[size_t(i) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
    }
  }
#pragma omp parallel for schedule(static)
  for (int j = 0; j < n; ++j) {
    for (int kk = 0; kk < k; ++kk) {
      uint8_t const sf = tensor_sfb(j, kk, 0).raw();
      float v = float(static_cast<cutlass::float_e2m1_t>(tensor_b(j, kk, 0)));
      if (sf & 0x80u) { v = reinterpret_e2m1_as_e0m3(v); }
      b_dec[size_t(j) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
    }
  }

  block_d.sync_host();
  auto tensor_d = make_tensor(make_iterator(block_d.host_data()), layout_d);

  // Compare by relative Frobenius norm rather than element-wise relative error. The GPU sums K
  // in a different order than the reference and rounds the result to bfloat16, so individual
  // elements whose accumulation nearly cancels can show a large relative error while the result
  // is entirely correct. A norm ratio is insensitive to that but still moves decisively (order
  // 1, not 1e-3) if any granule decodes under the wrong codebook.
#pragma omp parallel for schedule(static) reduction(+ : num, den) reduction(max : max_abs_err)
  for (int i = 0; i < m; ++i) {
    float const *ap = &a_dec[size_t(i) * size_t(k)];
    for (int j = 0; j < n; ++j) {
      float const *bp = &b_dec[size_t(j) * size_t(k)];
      float acc = 0.f;
      for (int kk = 0; kk < k; ++kk) { acc += ap[kk] * bp[kk]; }
      float const ref = alpha * acc + beta * float(static_cast<cutlass::bfloat16_t>(tensor_c(i, j, 0)));
      float const got = float(static_cast<cutlass::bfloat16_t>(tensor_d(i, j, 0)));
      double const d  = double(got) - double(ref);
      num += d * d;
      den += double(ref) * double(ref);
      max_abs_err = std::max(max_abs_err, std::fabs(d));
    }
  }

  }  // !skip_ref

  double const rel_err = (den > 0.0) ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
  // bfloat16 carries 8 mantissa bits, so per-element rounding alone is ~2^-9 relative; over a
  // K-length reduction with a different summation order the norm ratio lands a little above
  // that. 1e-2 is far below the ~1.0 a mis-decoded granule produces and far above the noise.
  if (!skip_ref) {
    bool const passed = (den > 0.0) && (rel_err < 1e-2);
    std::cout << "Correctness: " << (passed ? "PASSED" : "FAILED")
              << "  (relative Frobenius error " << rel_err
              << ", max abs error " << max_abs_err << ")" << std::endl;
    if (!passed) {
      return -1;
    }
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
  std::cout << "Throughput: " << tflops << " TFLOP/s (nvfp4-mixed-parity)" << std::endl;

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
