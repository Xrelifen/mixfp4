#!/bin/bash
# Fast standalone rebuild of just mixed_nvfp4_gemm (bypasses cmake reconfigure of all of CUTLASS).
# Flags mirror build/CMakeFiles/mixed_nvfp4_gemm.dir/flags.make exactly, minus the
# compute_120a-PTX gencode (SASS-only build; nothing here JITs) which roughly halves build time.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$WT/build/mixed_nvfp4_gemm}"
EXTRA="${EXTRA:-}"

# CUTLASS's cmake generates a couple of headers (version info) into its build tree. Point at any
# already-configured build dir rather than reconfiguring one -- that reconfigure is the slow part
# this script exists to skip.
CUTLASS_GENERATED_INCLUDE="${CUTLASS_GENERATED_INCLUDE:-$WT/build/3rdparty/cutlass/include}"
if [ ! -d "$CUTLASS_GENERATED_INCLUDE" ]; then
  echo "error: no generated CUTLASS headers at $CUTLASS_GENERATED_INCLUDE" >&2
  echo "       configure cmake once, or set CUTLASS_GENERATED_INCLUDE to an existing build's" >&2
  echo "       3rdparty/cutlass/include directory." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
/usr/local/cuda-13.1/bin/nvcc \
  $EXTRA \
  -O3 -DNDEBUG -std=c++17 \
  "--generate-code=arch=compute_120a,code=[sm_120a]" \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
  --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DCUTLASS_SM100_FAMILY_ARCHS_ENABLED \
  -Xcompiler=-fno-strict-aliasing -Xcompiler=-fopenmp \
  -I"$WT/3rdparty/cutlass/examples/common" \
  -I"$WT/src" \
  -I"$WT/3rdparty/cutlass/include" \
  -I"$CUTLASS_GENERATED_INCLUDE" \
  -I"$WT/3rdparty/cutlass/tools/util/include" \
  -isystem /usr/local/cuda-13.1/include \
  "$WT/src/mixed_nvfp4_gemm.cu" -o "$OUT" \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lgomp
echo "built $OUT"
