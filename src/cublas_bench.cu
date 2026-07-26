// cuBLASLt throughput baseline for the mixed-NVFP4 comparison.
//
// The other reference points in this project (src/nvfp4_gemm.cu, src/fp8_gemm.cu) are CUTLASS
// kernels, which answers "what does the same library cost without mixed formats". This answers
// the different and more practical question: what would you get from the vendor library if you
// were not writing a kernel at all.
//
// All modes run the TN layout the low-precision tensor cores require -- both operands K-major,
// op(A)=T, op(B)=N -- and produce bfloat16 D, matching mixed_nvfp4_gemm.cu.
//
// NOTE: this measures throughput only. The operand and scale-factor buffers are filled with
// arbitrary bytes; cuBLAS computes garbage at exactly the speed it would compute real data. Do
// not read any numerical meaning into it. Correctness of our kernel is established separately,
// against an E0M3-aware reference, in mixed_nvfp4_gemm.cu.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x)                                                                       \
  do {                                                                                      \
    cudaError_t e_ = (x);                                                                   \
    if (e_ != cudaSuccess) {                                                                \
      std::printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_));       \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_LT(x)                                                                         \
  do {                                                                                      \
    cublasStatus_t s_ = (x);                                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                      \
      std::printf("cublasLt error %s at %d: %d\n", #x, __LINE__, int(s_));                  \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

struct Mode {
  const char *name;
  cudaDataType_t ab_type;
  bool block_scaled;   // NVFP4-style per-16-element UE4M3 scales
};

// Returns TFLOP/s, or 0 if cuBLAS has no kernel for this combination on this GPU.
static double bench(cublasLtHandle_t lt, Mode const &mode, int m, int n, int k, int iters) {
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;

  cublasLtMatmulDesc_t desc = nullptr;
  CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Both operands K-major (TN), D column-major m x n.
  cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
  CHECK_LT(cublasLtMatrixLayoutCreate(&Ad, mode.ab_type, k, m, k));
  CHECK_LT(cublasLtMatrixLayoutCreate(&Bd, mode.ab_type, k, n, k));
  CHECK_LT(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, m));

  // Operand bytes. 4-bit types pack two elements per byte.
  size_t const elem_bits = (mode.ab_type == CUDA_R_4F_E2M1) ? 4
                         : (mode.ab_type == CUDA_R_8F_E4M3) ? 8
                                                            : 16;
  size_t const a_bytes = (size_t(m) * k * elem_bits) / 8;
  size_t const b_bytes = (size_t(n) * k * elem_bits) / 8;

  void *dA = nullptr, *dB = nullptr, *dD = nullptr, *dSFA = nullptr, *dSFB = nullptr, *ws = nullptr;
  size_t const ws_bytes = 128ull << 20;
  CHECK_CUDA(cudaMalloc(&dA, a_bytes));
  CHECK_CUDA(cudaMalloc(&dB, b_bytes));
  CHECK_CUDA(cudaMalloc(&dD, size_t(m) * n * 2));
  CHECK_CUDA(cudaMalloc(&ws, ws_bytes));
  CHECK_CUDA(cudaMemset(dA, 0x22, a_bytes));
  CHECK_CUDA(cudaMemset(dB, 0x22, b_bytes));

  if (mode.block_scaled) {
    // One UE4M3 byte per 16 elements along K, plus generous slack for whatever swizzled layout
    // cuBLAS expects -- only the read addresses matter for timing, not the values.
    size_t const sfa = (size_t(m) * k / 16) * 4 + (1 << 20);
    size_t const sfb = (size_t(n) * k / 16) * 4 + (1 << 20);
    CHECK_CUDA(cudaMalloc(&dSFA, sfa));
    CHECK_CUDA(cudaMalloc(&dSFB, sfb));
    CHECK_CUDA(cudaMemset(dSFA, 0x3c, sfa));
    CHECK_CUDA(cudaMemset(dSFB, 0x3c, sfb));
    int32_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSFA, sizeof(dSFA)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSFB, sizeof(dSFB)));
  }

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

  cublasLtMatmulHeuristicResult_t heur[8];
  int found = 0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 8, heur, &found);

  double tflops = 0.0;
  if (hs == CUBLAS_STATUS_SUCCESS && found > 0) {
    float alpha = 1.0f, beta = 0.0f;
    // Time every returned algorithm and keep the best -- the heuristic's first choice is not
    // always the fastest, and we want cuBLAS shown at its best.
    for (int a = 0; a < found; ++a) {
      auto run = [&] {
        return cublasLtMatmul(lt, desc, &alpha, dA, Ad, dB, Bd, &beta, dD, Cd, dD, Cd,
                              &heur[a].algo, ws, ws_bytes, nullptr);
      };
      if (run() != CUBLAS_STATUS_SUCCESS) { continue; }
      for (int i = 0; i < 5; ++i) { run(); }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t t0, t1;
      CHECK_CUDA(cudaEventCreate(&t0));
      CHECK_CUDA(cudaEventCreate(&t1));
      CHECK_CUDA(cudaEventRecord(t0));
      for (int i = 0; i < iters; ++i) { run(); }
      CHECK_CUDA(cudaEventRecord(t1));
      CHECK_CUDA(cudaEventSynchronize(t1));
      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
      CHECK_CUDA(cudaEventDestroy(t0));
      CHECK_CUDA(cudaEventDestroy(t1));
      double const f = 2.0 * m * n * k / ((ms / iters) / 1e3) / 1e12;
      if (f > tflops) { tflops = f; }
    }
  }

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulDescDestroy(desc);
  cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(ws);
  if (dSFA) { cudaFree(dSFA); }
  if (dSFB) { cudaFree(dSFB); }
  return tflops;
}

int main(int argc, char **argv) {
  cublasLtHandle_t lt;
  CHECK_LT(cublasLtCreate(&lt));

  Mode const modes[] = {
      {"bf16",  CUDA_R_16BF,     false},
      {"fp8",   CUDA_R_8F_E4M3,  false},
      {"nvfp4", CUDA_R_4F_E2M1,  true },
  };

  std::vector<std::array<int, 3>> shapes;
  if (argc >= 4) {
    shapes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])});
  } else {
    shapes = {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096},
              {8192, 8192, 8192}, {4096, 4096, 16384}, {8192, 8192, 2048},
              {16384, 16384, 2048}};
  }

  std::printf("%-20s %12s %12s %12s\n", "M x N x K", "cuBLAS-bf16", "cuBLAS-fp8", "cuBLAS-nvfp4");
  for (auto const &s : shapes) {
    char name[64];
    std::snprintf(name, sizeof(name), "%dx%dx%d", s[0], s[1], s[2]);
    std::printf("%-20s", name);
    for (auto const &mode : modes) {
      double t = bench(lt, mode, s[0], s[1], s[2], 50);
      if (t > 0) { std::printf(" %12.1f", t); }
      else       { std::printf(" %12s", "unsupported"); }
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  cublasLtDestroy(lt);
  return 0;
}
