#!/bin/bash
# Granularity-vs-throughput sweep: what each format granule costs against stock NVFP4.
#
# Builds every variant from source, patches it, and measures. Refuses to run while another process
# is on the GPU -- every number here is a comparison, and a contended card silently halves them
# (observed twice during this work: stock nvfp4_gemm reading 725 and 852 TFLOP/s instead of 1207).
#
#   ./scripts/sweep.sh              # build everything, then measure
#   SKIP_BUILD=1 ./scripts/sweep.sh # measure binaries already in build/
#   REPS=5 SHAPE="8192 8192 8192" ./scripts/sweep.sh
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPS="${REPS:-5}"
SHAPE="${SHAPE:-4096 4096 4096}"
STOCK="${STOCK:-/home/brian/mixfp4/build/nvfp4_gemm}"

# label | granule | extra nvcc flags | generator args ("-" = C++ path, no generated header)
VARIANTS=(
  "ceiling (no dispatch)|-|-DMIXFP4_NO_DISPATCH=1|-"
  "C++ 8 arms (shipped default)|32x32x128||-"
  "C++ 8 arms, fine A|16x64x128|-DMIXFP4_A_ATOMS_PER_GRANULE=1 -DMIXFP4_B_ATOMS_PER_GRANULE=8|-"
  "PTX G=1|16x16x64|-DMIXFP4_PTX=1|--a-atoms 1 --b-atoms 2 --m-per-group 2 --n-per-group 8"
  "PTX G=2 (2x4 atoms)|16x8x64|-DMIXFP4_PTX=1|--a-atoms 1 --b-atoms 1 --m-per-group 2 --n-per-group 4"
  "PTX G=4 (2x2 atoms)|16x8x64|-DMIXFP4_PTX=1|--a-atoms 1 --b-atoms 1 --m-per-group 2 --n-per-group 2"
  "PTX G=8 (1x2 atoms)|16x8x64|-DMIXFP4_PTX=1|--a-atoms 1 --b-atoms 1 --m-per-group 1 --n-per-group 2"
)

slug() { echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]'; }

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "building ${#VARIANTS[@]} variants..."
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r label gran flags genargs <<< "$v"
    out="$WT/build/sweep_$(slug "$label")"
    [ "$genargs" = "-" ] || python3 "$WT/scripts/gen_mixed_mma_ptx.py" $genargs >/dev/null || exit 1
    EXTRA="$flags" "$WT/scripts/build_mixed.sh" "$out" >/dev/null 2>&1 \
      || { echo "  BUILD FAILED: $label"; continue; }
    # The ceiling has only site 0, so it is not patchable -- and does not need to be.
    if [ -z "${flags##*NO_DISPATCH*}" ]; then
      cp "$out" "${out}_patched"
    else
      python3 "$WT/scripts/patch_mixed_nvfp4_gemm.py" "$out" "${out}_patched" >/dev/null \
        || { echo "  PATCH FAILED: $label"; continue; }
    fi
    echo "  ok: $label"
  done
  echo
fi

read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
if [ "$util" -ge 15 ] || [ "$mem" -ge 2000 ]; then
  echo "REFUSING: GPU is busy (util=${util}% mem=${mem}MiB). Numbers would be meaningless."
  exit 1
fi
echo "GPU clean (util=${util}% mem=${mem}MiB); shape ${SHAPE}, ${REPS} reps each, reporting best."
echo

best() {  # best <tag> <bin>
  local b=0 t
  for _ in $(seq 1 "$REPS"); do
    t=$(env MIXFP4_SKIP_REF=1 MIXFP4_TAG="$1" "$2" $SHAPE 2>/dev/null | grep -oP 'Throughput: \K[0-9.]+')
    [ -n "$t" ] && b=$(python3 -c "print(max($b,$t))")
  done
  echo "$b"
}

stock=$(best none "$STOCK")
printf "%-30s %-11s %9s %9s %9s\n" "config" "granule" "TAG=none" "TAG=rand" "vs stock"
printf "%-30s %-11s %9s %9s %9s\n" "------------------------------" "-----------" "---------" "---------" "---------"
printf "%-30s %-11s %9s %9.1f %9s\n" "stock nvfp4_gemm" "-" "-" "$stock" "-"
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r label gran flags genargs <<< "$v"
  bin="$WT/build/sweep_$(slug "$label")_patched"
  [ -x "$bin" ] || { printf "%-30s %-11s %29s\n" "$label" "$gran" "(missing)"; continue; }
  n=$(best none "$bin"); r=$(best random "$bin")
  printf "%-30s %-11s %9.1f %9.1f %9s\n" "$label" "$gran" "$n" "$r" \
    "$(python3 -c "print(f'{100*(1-$r/$stock):+.1f}%' if $stock>0 else 'n/a')")"
done
echo
echo "TAG=none  pins every dispatch to arm 0: dispatch instruction cost with a hot code footprint."
echo "TAG=rand  walks every arm: adds the code-footprint / instruction-fetch cost."
