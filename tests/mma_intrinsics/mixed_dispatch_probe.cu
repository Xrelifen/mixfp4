// Stage 0 Step 2 PoC (see /home/brian/.claude/plans/mutable-soaring-pascal.md): proves the
// mechanism a real mixed-format GEMM mainloop would need -- a kernel containing FOUR
// independently-patchable mma.sync call sites, selected at runtime by a 2-bit predicate
// (flag_a, flag_b) extracted from a byte that only ONE lane physically holds, broadcast via
// __shfl_sync before a warp-uniform branch. This simulates CUTLASS's real SFALayout/SFBLayout
// constraint (only 16/32 lanes hold a given SFA value, 8/32 for SFB -- see
// cute/atom/mma_traits_sm120.hpp:158-162) instead of glossing over it.
//
// All four sites compile identically -- PTX has no e0m3 token, so every site initially declares
// plain e2m1 x e2m1. patch_dispatch_formats.py independently patches sites 1-3 to different A/B
// format combinations post-compile (site 0 stays native e2m1 x e2m1), so that after patching,
// (flag_a, flag_b) genuinely selects between four different hardware decodes at runtime:
//   flag_a=0 flag_b=0 -> site0 -> e2m1 x e2m1  (native, unpatched)
//   flag_a=1 flag_b=0 -> site1 -> e0m3 x e2m1  (bit 14 patched, matches patch_formats.py's A bit)
//   flag_a=0 flag_b=1 -> site2 -> e2m1 x e0m3  (bit 15 patched, matches patch_formats.py's B bit)
//   flag_a=1 flag_b=1 -> site3 -> e0m3 x e0m3  (both bits patched)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t status_ = (expr);                                                 \
    if (status_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(status_)); \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

namespace {

#define DEFINE_MMA_SITE(name)                                                    \
  __device__ __noinline__ void name(                                             \
      float& d0, float& d1, float& d2, float& d3,                                \
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,                        \
      uint32_t b0, uint32_t b1,                                                  \
      float c0, float c1, float c2, float c3,                                    \
      uint32_t sfa, uint32_t sfb) {                                              \
    constexpr uint16_t bid_a = 0;                                                \
    constexpr uint16_t tid_a = 0;                                                \
    constexpr uint16_t bid_b = 0;                                                \
    constexpr uint16_t tid_b = 0;                                                \
    asm volatile(                                                                \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."             \
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                              \
        "{%0, %1, %2, %3},"                                                      \
        "{%4, %5, %6, %7},"                                                      \
        "{%8, %9},"                                                              \
        "{%10, %11, %12, %13},"                                                  \
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"                                \
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                                 \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                                    \
          "r"(b0), "r"(b1),                                                      \
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),                                    \
          "r"(sfa), "h"(bid_a), "h"(tid_a),                                      \
          "r"(sfb), "h"(bid_b), "h"(tid_b));                                     \
  }

// Named so the site index is unambiguous in mangled symbol names (patch_dispatch_formats.py
// looks for the literal substrings "mma_site0".."mma_site3").
DEFINE_MMA_SITE(mma_site0)
DEFINE_MMA_SITE(mma_site1)
DEFINE_MMA_SITE(mma_site2)
DEFINE_MMA_SITE(mma_site3)
#undef DEFINE_MMA_SITE

uint32_t repeat_nibble(unsigned value) {
  value &= 0xfu;
  uint32_t packed = 0;
  for (int shift = 0; shift < 32; shift += 4) packed |= value << shift;
  return packed;
}

__global__ void mixed_dispatch_probe(uint32_t packed_a, uint32_t packed_b,
                                     uint32_t raw_sfa_byte, uint32_t raw_sfb_byte,
                                     unsigned owner_lane, float* output) {
  // Only `owner_lane` starts out holding the raw tag byte -- every other lane starts at 0 and
  // must receive it via shuffle, so a branch on it can't be warp-uniform "for free"; this forces
  // the same broadcast-before-branch step the real mainloop would need.
  uint32_t local_a = (threadIdx.x == owner_lane) ? raw_sfa_byte : 0u;
  uint32_t local_b = (threadIdx.x == owner_lane) ? raw_sfb_byte : 0u;
  uint32_t bcast_a = __shfl_sync(0xffffffffu, local_a, owner_lane);
  uint32_t bcast_b = __shfl_sync(0xffffffffu, local_b, owner_lane);
  bool flag_a = (bcast_a & 0x80u) != 0u;
  bool flag_b = (bcast_b & 0x80u) != 0u;

  // Stage 0 Step 1 already proved real hardware ignores bit 7 of the scale byte, so the still-
  // tagged byte is packed straight into the SF operand register without masking -- one byte
  // replicated across all 4 K16-group lanes, same convention as mma_probe.cu's pack_scale().
  uint32_t packed_sfa = bcast_a | (bcast_a << 8) | (bcast_a << 16) | (bcast_a << 24);
  uint32_t packed_sfb = bcast_b | (bcast_b << 8) | (bcast_b << 16) | (bcast_b << 24);

  float d0, d1, d2, d3;
  if (!flag_a && !flag_b) {
    mma_site0(d0, d1, d2, d3, packed_a, packed_a, packed_a, packed_a, packed_b, packed_b,
              0.0f, 0.0f, 0.0f, 0.0f, packed_sfa, packed_sfb);
  } else if (flag_a && !flag_b) {
    mma_site1(d0, d1, d2, d3, packed_a, packed_a, packed_a, packed_a, packed_b, packed_b,
              0.0f, 0.0f, 0.0f, 0.0f, packed_sfa, packed_sfb);
  } else if (!flag_a && flag_b) {
    mma_site2(d0, d1, d2, d3, packed_a, packed_a, packed_a, packed_a, packed_b, packed_b,
              0.0f, 0.0f, 0.0f, 0.0f, packed_sfa, packed_sfb);
  } else {
    mma_site3(d0, d1, d2, d3, packed_a, packed_a, packed_a, packed_a, packed_b, packed_b,
              0.0f, 0.0f, 0.0f, 0.0f, packed_sfa, packed_sfb);
  }

  int offset = static_cast<int>(threadIdx.x) * 4;
  output[offset + 0] = d0;
  output[offset + 1] = d1;
  output[offset + 2] = d2;
  output[offset + 3] = d3;
}

void usage(char const* program) {
  std::fprintf(stderr,
               "Usage: %s [a_nibble] [b_nibble] [sfa_byte] [sfb_byte] [owner_lane]\n"
               "  sfa_byte/sfb_byte: 0-255; bit 7 selects e0m3 for that operand at runtime\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 6) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  unsigned a_nibble = argc >= 2 ? std::strtoul(argv[1], nullptr, 0) : 1u;
  unsigned b_nibble = argc >= 3 ? std::strtoul(argv[2], nullptr, 0) : a_nibble;
  unsigned sfa_byte = argc >= 4 ? std::strtoul(argv[3], nullptr, 0) : 0x38u;
  unsigned sfb_byte = argc >= 5 ? std::strtoul(argv[4], nullptr, 0) : sfa_byte;
  unsigned owner_lane = argc >= 6 ? std::strtoul(argv[5], nullptr, 0) : 0u;
  if (a_nibble > 15 || b_nibble > 15 || sfa_byte > 255 || sfb_byte > 255 || owner_lane > 31) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 12) {
    std::fprintf(stderr, "This probe requires SM120/SM12x, found SM%d%d.\n", props.major, props.minor);
    return EXIT_FAILURE;
  }

  constexpr int kOutputCount = 32 * 4;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, kOutputCount * sizeof(float)));

  mixed_dispatch_probe<<<1, 32>>>(repeat_nibble(a_nibble), repeat_nibble(b_nibble),
                                  sfa_byte, sfb_byte, owner_lane, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(kOutputCount);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));

  auto [minimum, maximum] = std::minmax_element(output.begin(), output.end());
  bool flag_a = (sfa_byte & 0x80u) != 0u;
  bool flag_b = (sfb_byte & 0x80u) != 0u;
  std::printf("binary=%s gpu=%s sm=%d%d flag_a=%d flag_b=%d a=0x%x b=0x%x sfa_byte=0x%02x sfb_byte=0x%02x owner_lane=%u\n",
              argv[0], props.name, props.major, props.minor, flag_a, flag_b,
              a_nibble, b_nibble, sfa_byte, sfb_byte, owner_lane);
  std::printf("outputs=%d min=%.9g max=%.9g first=[%.9g %.9g %.9g %.9g]\n",
              kOutputCount, *minimum, *maximum, output[0], output[1], output[2], output[3]);

  bool all_equal = std::all_of(output.begin(), output.end(),
                               [&](float x) { return x == output.front(); });
  std::printf("all_equal=%s\n", all_equal ? "true" : "false");
  return all_equal ? EXIT_SUCCESS : EXIT_FAILURE;
}
