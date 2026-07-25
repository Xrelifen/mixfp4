#!/bin/bash
# Throughput of the mixed E0M3/E2M1 kernel against stock NVFP4 and FP8, across shapes.
#
# Refuses to run on a busy GPU: every number here is a comparison, and a contended card silently
# halves them (observed during this work -- stock nvfp4_gemm reading 852 TFLOP/s instead of 1208
# while another process held 26GB).
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/home/brian/mixfp4/build
REPS="${REPS:-3}"

read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
if [ "$util" -ge 15 ] || [ "$mem" -ge 2000 ]; then
  echo "REFUSING: GPU busy (util=${util}% mem=${mem}MiB)."; exit 1
fi

best() {  # best of REPS runs of "$@"
  local b=0 t
  for _ in $(seq 1 "$REPS"); do
    t=$("$@" 2>/dev/null | grep -oP 'Throughput: \K[0-9.]+')
    [ -n "$t" ] && b=$(python3 -c "print(max($b,$t))")
  done
  echo "$b"
}

printf "%-22s %10s %10s %10s %10s %10s\n" "M x N x K" "FP8" "NVFP4" "mixed" "vs NVFP4" "vs FP8"
printf "%-22s %10s %10s %10s %10s %10s\n" "----------------------" "----------" "----------" "----------" "----------" "----------"
for sz in "1024 1024 1024" "2048 2048 2048" "4096 4096 4096" "8192 8192 8192" "4096 4096 16384" "8192 8192 2048" "16384 16384 2048"; do
  set -- $sz
  f8=$(best "$MAIN/fp8_gemm" "$@")
  f4=$(best "$MAIN/nvfp4_gemm" "$@")
  mx=$(best env MIXFP4_SKIP_REF=1 MIXFP4_TAG=random "$WT/build/g32_patched" "$@")
  python3 -c "
f8,f4,mx = $f8,$f4,$mx
name='${1}x${2}x${3}'
r4 = f'{100*(1-mx/f4):+.1f}%' if f4>0 else 'n/a'
r8 = f'{mx/f8:.2f}x' if f8>0 else 'n/a'
print(f'{name:<22} {f8:10.1f} {f4:10.1f} {mx:10.1f} {r4:>10} {r8:>10}')
"
done
echo
echo "FP8/NVFP4 = stock CUTLASS kernels. mixed = E0M3/E2M1 patched, randomly tagged so all four"
echo "format sites are live. 'vs NVFP4' is the overhead of mixed-format dispatch. Units: TFLOP/s."
