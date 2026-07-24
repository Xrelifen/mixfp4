// Block-scaled `mma.sync` intrinsic probe for SM120 (GeForce Blackwell).
//
// Exercises all three PTX-legal kind::mxf4nvf4.block_scale configs for m16n8k64 e2m1 x e2m1
// (confirmed against 3rdparty/cutlass/include/cute/arch/mma_sm120.hpp, gated by
// CUTE_ARCH_MXF4NVF4_*_MMA_ENABLED in 3rdparty/cutlass/include/cute/arch/config.hpp):
//
//   VEC4_UE4M3  scale_vec::4X   4 groups of K16, UE4M3 scale   (NVFP4; CUDA >= 12.8)
//   VEC4_UE8M0  scale_vec::4X   4 groups of K16, UE8M0 scale   (CUDA >= 13.1)
//   VEC2_UE8M0  scale_vec::2X   2 groups of K32, UE8M0 scale   (OCP MXFP4 block size; CUDA >= 12.8)
//
// PTX has no token for e0m3, so this binary only ever declares e2m1 x e2m1. E0M3 operands are
// reached afterwards by flipping two SASS bits post-compile; see patch_formats.py, ported from
// 3rdparty/sm120-e0m3-mma/patch_executable_formats.py.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define VEC4_UE4M3 1
#define VEC4_UE8M0 2
#define VEC2_UE8M0 3

#ifndef OMMA_SCALE_MODE
#define OMMA_SCALE_MODE VEC4_UE4M3
#endif

#if OMMA_SCALE_MODE == VEC4_UE4M3
#define OMMA_SCALE_TYPE_STR "ue4m3"
#define OMMA_SCALE_VEC_STR "4X"
#define OMMA_SCALE_GROUPS 4
#elif OMMA_SCALE_MODE == VEC4_UE8M0
#define OMMA_SCALE_TYPE_STR "ue8m0"
#define OMMA_SCALE_VEC_STR "4X"
#define OMMA_SCALE_GROUPS 4
#elif OMMA_SCALE_MODE == VEC2_UE8M0
#define OMMA_SCALE_TYPE_STR "ue8m0"
#define OMMA_SCALE_VEC_STR "2X"
#define OMMA_SCALE_GROUPS 2
#else
#error "Unknown OMMA_SCALE_MODE"
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t status_ = (expr);                                                 \
    if (status_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(status_)); \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

namespace {

__device__ __forceinline__ void omma_f4_f4(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb) {
  constexpr uint16_t bid_a = 0;
  constexpr uint16_t tid_a = 0;
  constexpr uint16_t bid_b = 0;
  constexpr uint16_t tid_b = 0;

#if OMMA_SCALE_MODE == VEC4_UE4M3
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
#elif OMMA_SCALE_MODE == VEC4_UE8M0
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
#elif OMMA_SCALE_MODE == VEC2_UE8M0
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
#endif
}

__global__ void nibble_probe(uint32_t packed_a, uint32_t packed_b,
                             uint32_t packed_scale_a,
                             uint32_t packed_scale_b, float* output) {
  float d0;
  float d1;
  float d2;
  float d3;
  omma_f4_f4(d0, d1, d2, d3,
             packed_a, packed_a, packed_a, packed_a,
             packed_b, packed_b,
             0.0f, 0.0f, 0.0f, 0.0f,
             packed_scale_a, packed_scale_b);

  int offset = static_cast<int>(threadIdx.x) * 4;
  output[offset + 0] = d0;
  output[offset + 1] = d1;
  output[offset + 2] = d2;
  output[offset + 3] = d3;
}

uint32_t repeat_nibble(unsigned value) {
  value &= 0xfu;
  uint32_t packed = 0;
  for (int shift = 0; shift < 32; shift += 4) {
    packed |= value << shift;
  }
  return packed;
}

// Packs one scale byte into the low `OMMA_SCALE_GROUPS` bytes of the 32-bit SF operand register,
// one byte per K group (K16 groups for scale_vec::4X, K32 groups for scale_vec::2X); unused high
// bytes are left zero. All groups get the same scale value, so the whole K64 operand is uniformly
// scaled and the expected dot product reduces to a simple K * a * b formula.
uint32_t pack_scale(unsigned value) {
  value &= 0xffu;
  uint32_t packed = 0;
  for (int i = 0; i < OMMA_SCALE_GROUPS; ++i) {
    packed |= value << (8 * i);
  }
  return packed;
}

[[maybe_unused]] float decode_ue4m3(unsigned code) {
  code &= 0x7fu;
  unsigned exponent = (code >> 3) & 0xfu;
  unsigned mantissa = code & 0x7u;
  if (exponent == 0) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0xfu && mantissa == 0x7u) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                    static_cast<int>(exponent) - 7);
}

[[maybe_unused]] float decode_ue8m0(unsigned code) {
  code &= 0xffu;
  if (code == 0xffu) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

float decode_scale(unsigned code) {
#if OMMA_SCALE_MODE == VEC4_UE4M3
  return decode_ue4m3(code);
#else
  return decode_ue8m0(code);
#endif
}

unsigned default_scale_code() {
#if OMMA_SCALE_MODE == VEC4_UE4M3
  return 0x38u;  // 1.0
#else
  return 127u;  // 1.0
#endif
}

unsigned max_scale_code() {
#if OMMA_SCALE_MODE == VEC4_UE4M3
  return 0xffu;  // relaxed for bit-7 hardware test (bit 7 is masked off by decode_ue4m3 for display)
#else
  return 0xfeu;
#endif
}

void usage(char const* program) {
  std::fprintf(stderr,
               "Usage: %s [a_nibble] [b_nibble] [sfa_%s] [sfb_%s]\n",
               program, OMMA_SCALE_TYPE_STR, OMMA_SCALE_TYPE_STR);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 5) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  unsigned a_nibble = argc >= 2 ? std::strtoul(argv[1], nullptr, 0) : 1u;
  unsigned b_nibble = argc >= 3 ? std::strtoul(argv[2], nullptr, 0) : a_nibble;
  unsigned scale_a_code = argc >= 4 ? std::strtoul(argv[3], nullptr, 0) : default_scale_code();
  unsigned scale_b_code = argc >= 5 ? std::strtoul(argv[4], nullptr, 0) : scale_a_code;
  if (a_nibble > 15 || b_nibble > 15 ||
      scale_a_code > max_scale_code() || scale_b_code > max_scale_code()) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 12) {
    std::fprintf(stderr, "This probe requires SM120/SM12x, found SM%d%d.\n",
                 props.major, props.minor);
    return EXIT_FAILURE;
  }

  constexpr int kOutputCount = 32 * 4;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, kOutputCount * sizeof(float)));

  nibble_probe<<<1, 32>>>(repeat_nibble(a_nibble), repeat_nibble(b_nibble),
                          pack_scale(scale_a_code), pack_scale(scale_b_code),
                          device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(kOutputCount);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));

  auto [minimum, maximum] = std::minmax_element(output.begin(), output.end());
  std::printf("binary=%s gpu=%s sm=%d%d scale_mode=scale_vec::%s.%s "
              "ptx_declared_format=e2m1_x_e2m1 "
              "a=0x%x b=0x%x sfa=0x%02x(%.9g) sfb=0x%02x(%.9g)\n",
              argv[0], props.name, props.major, props.minor,
              OMMA_SCALE_VEC_STR, OMMA_SCALE_TYPE_STR,
              a_nibble, b_nibble, scale_a_code, decode_scale(scale_a_code),
              scale_b_code, decode_scale(scale_b_code));
  std::printf("outputs=%d min=%.9g max=%.9g first=[%.9g %.9g %.9g %.9g]\n",
              kOutputCount, *minimum, *maximum,
              output[0], output[1], output[2], output[3]);

  bool all_equal = std::all_of(output.begin(), output.end(),
                               [&](float x) { return x == output.front(); });
  std::printf("all_equal=%s\n", all_equal ? "true" : "false");
  return all_equal ? EXIT_SUCCESS : EXIT_FAILURE;
}
