cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(mixfp4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build only CUTLASS's headers + host/device reference utilities (tools/util) --
# skip its own examples/tests/profiler, which we don't need and which are slow to configure.
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)

add_subdirectory(3rdparty/cutlass)

add_executable(int8_gemm src/int8_gemm.cu)
cutlass_apply_standard_compile_options(int8_gemm)
target_link_libraries(int8_gemm PRIVATE nvidia::cutlass::tools::util)
target_include_directories(int8_gemm PRIVATE 3rdparty/cutlass/examples/common)

# sm_80 (Ampere) int8 tensor-core MMA is forward-compatible PTX; embedding it lets
# the driver JIT it for whatever GPU is actually present (e.g. sm_120 Blackwell).
cutlass_apply_cuda_gencode_flags(int8_gemm SM_ARCHS 80)

add_executable(fp8_gemm src/fp8_gemm.cu)
cutlass_apply_standard_compile_options(fp8_gemm)
target_link_libraries(fp8_gemm PRIVATE nvidia::cutlass::tools::util)
target_include_directories(fp8_gemm PRIVATE 3rdparty/cutlass/examples/common)

# sm_120a unlocks GeForce Blackwell's architecture-specific tensor-core MMA (the "a"
# suffix ISA extensions are not PTX-forward-compatible, so this only runs on sm_120).
cutlass_apply_cuda_gencode_flags(fp8_gemm SM_ARCHS 120a)

add_executable(nvfp4_gemm src/nvfp4_gemm.cu)
cutlass_apply_standard_compile_options(nvfp4_gemm)
target_link_libraries(nvfp4_gemm PRIVATE nvidia::cutlass::tools::util)
target_include_directories(nvfp4_gemm PRIVATE 3rdparty/cutlass/examples/common)
cutlass_apply_cuda_gencode_flags(nvfp4_gemm SM_ARCHS 120a)

add_executable(nvfp4_gemm_cute src/nvfp4_gemm_cute.cu)
cutlass_apply_standard_compile_options(nvfp4_gemm_cute)
target_link_libraries(nvfp4_gemm_cute PRIVATE nvidia::cutlass::tools::util)
target_include_directories(nvfp4_gemm_cute PRIVATE 3rdparty/cutlass/examples/common)
cutlass_apply_cuda_gencode_flags(nvfp4_gemm_cute SM_ARCHS 120a)
target_compile_options(nvfp4_gemm_cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(mixed_nvfp4_gemm src/mixed_nvfp4_gemm.cu)
cutlass_apply_standard_compile_options(mixed_nvfp4_gemm)
target_link_libraries(mixed_nvfp4_gemm PRIVATE nvidia::cutlass::tools::util)
target_include_directories(mixed_nvfp4_gemm PRIVATE 3rdparty/cutlass/examples/common src)
cutlass_apply_cuda_gencode_flags(mixed_nvfp4_gemm SM_ARCHS 120a)
target_compile_options(mixed_nvfp4_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# cuBLASLt throughput baseline (bf16 / fp8 / block-scaled nvfp4). Not a CUTLASS kernel -- it
# answers "what does the vendor library give you", which is the comparison that matters when
# deciding whether a custom mixed-format kernel is worth maintaining.
add_executable(cublas_bench src/cublas_bench.cu)
cutlass_apply_cuda_gencode_flags(cublas_bench SM_ARCHS 120a)
target_link_libraries(cublas_bench PRIVATE cublasLt cublas)
