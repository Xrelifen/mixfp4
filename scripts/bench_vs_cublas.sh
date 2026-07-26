#!/bin/bash
# The mixed E0M3/E2M1 kernel against cuBLAS (the vendor library) and the CUTLASS reference
# kernels, on the same shapes, back to back.
#
# cuBLAS-nvfp4 is the interesting column: cuBLAS 13.2 exposes the *same* block-scaled NVFP4 format
# this kernel uses (CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, 16-element blocks, UE4M3 scales), so
# it is a true like-for-like baseline -- what you would get today without writing a kernel, and
# without mixed formats.
#
# Refuses to run on a busy GPU; contention silently halves everything.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/home/brian/mixfp4/build
REPS="${REPS:-3}"

read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
if [ "$util" -ge 15 ] || [ "$mem" -ge 2000 ]; then
  echo "REFUSING: GPU busy (util=${util}% mem=${mem}MiB)."; exit 1
fi
echo "GPU clean (util=${util}% mem=${mem}MiB). Units: TFLOP/s, best of $REPS."
echo

best() {
  local b=0 t
  for _ in $(seq 1 "$REPS"); do
    t=$("$@" 2>/dev/null | grep -oP 'Throughput: \K[0-9.]+')
    [ -n "$t" ] && b=$(python3 -c "print(max($b,$t))")
  done
  echo "$b"
}

printf "%-20s %9s %9s %9s | %9s %9s | %9s\n" \
  "M x N x K" "cuBLAS" "cuBLAS" "cuBLAS" "CUTLASS" "CUTLASS" "mixed"
printf "%-20s %9s %9s %9s | %9s %9s | %9s\n" \
  "" "bf16" "fp8" "nvfp4" "fp8" "nvfp4" "E0M3/E2M1"
printf "%-20s %9s %9s %9s | %9s %9s | %9s\n" \
  "--------------------" "---------" "---------" "---------" "---------" "---------" "---------"

for sz in "1024 1024 1024" "2048 2048 2048" "4096 4096 4096" "8192 8192 8192" "4096 4096 16384" "8192 8192 2048" "16384 16384 2048"; do
  set -- $sz
  cb=$("$WT/build/cublas_bench" "$@" 2>/dev/null | tail -1)
  cb_bf16=$(awk '{print $2}' <<<"$cb")
  cb_fp8=$(awk  '{print $3}' <<<"$cb")
  cb_fp4=$(awk  '{print $4}' <<<"$cb")
  ct_fp8=$(best "$MAIN/fp8_gemm" "$@")
  ct_fp4=$(best "$MAIN/nvfp4_gemm" "$@")
  mx=$(best env MIXFP4_SKIP_REF=1 MIXFP4_TAG=random "$WT/build/mixed_patched" "$@")
  printf "%-20s %9s %9s %9s | %9.1f %9.1f | %9.1f\n" \
    "${1}x${2}x${3}" "$cb_bf16" "$cb_fp8" "$cb_fp4" "$ct_fp8" "$ct_fp4" "$mx"
done
echo
echo "mixed = E0M3/E2M1 patched, randomly tagged so all four format arms are live."
