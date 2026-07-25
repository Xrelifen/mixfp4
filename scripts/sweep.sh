#!/bin/bash
# Clean-GPU measurement sweep. Refuses to run while another process is on the GPU -- every number
# in this investigation is a throughput comparison, and a contended GPU silently halves them
# (observed: stock nvfp4_gemm reading 852 TFLOP/s instead of 1208 while a 26GB job was resident).
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPS="${REPS:-3}"

read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
if [ "$util" -ge 15 ] || [ "$mem" -ge 2000 ]; then
  echo "REFUSING: GPU is busy (util=${util}% mem=${mem}MiB). Numbers would be meaningless."
  exit 1
fi
echo "GPU clean (util=${util}% mem=${mem}MiB); ${REPS} reps each, reporting best."
echo

run() {
  local label="$1" bin="$2"; shift 2
  [ -x "$bin" ] || { printf "%-34s (missing)\n" "$label"; return; }
  local best=0 t
  for _ in $(seq 1 "$REPS"); do
    t=$(env "$@" "$bin" 4096 4096 4096 2>/dev/null | grep -oP 'Throughput: \K[0-9.]+')
    [ -n "$t" ] && best=$(python3 -c "print(max($best,$t))")
  done
  printf "%-34s %8.1f TFLOP/s\n" "$label" "$best"
}

run "stock nvfp4_gemm (reference)"   /home/brian/mixfp4/build/nvfp4_gemm             DUMMY=1
run "ceiling (dispatch compiled out)" "$WT/build/mixed_ceiling"                      DUMMY=1
run "HOISTED per-k_tile, untagged"    "$WT/build/mixed_ktile"                        DUMMY=1
run "HOISTED per-k_tile, TAGGED"      "$WT/build/mixed_ktile"                        MIXFP4_TAG_FLAGS=1
echo "--- superseded per-mma-dispatch designs, for the record ---"
run "per-mma if/else (was current)"   /home/brian/mixfp4/build/mixed_nvfp4_gemm      DUMMY=1
run "per-mma brx.idx jump table"      "$WT/build/mixed_v2"                           DUMMY=1
run "per-cute::gemm nested if"        "$WT/build/mixed_hoist"                        DUMMY=1
run "per-cute::gemm switch"           "$WT/build/mixed_switch"                       DUMMY=1
