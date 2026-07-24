#!/usr/bin/env bash
# Builds and runs the SM120 block-scaled mma.sync intrinsic tests.
#
# Covers all three PTX-legal kind::mxf4nvf4.block_scale scale configs for m16n8k64 e2m1 x e2m1
# (see mma_probe.cu for where these are confirmed against 3rdparty/cutlass), crossed with all
# four A/B tensor-format combinations (e2m1 x e2m1 is native PTX; the other three come from the
# post-compile SASS patch documented in 3rdparty/sm120-e0m3-mma, applied here by patch_formats.py).
#
# Every observed GPU result is checked against expected_value.py, a from-scratch second
# implementation of the E2M1/E0M3 codebooks and UE4M3/UE8M0 scale decode -- not against
# mma_probe.cu's own decode code.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

cuda_root="${CUDA_HOME:-/usr/local/cuda}"
nvcc="${cuda_root}/bin/nvcc"
cuobjdump="${cuda_root}/bin/cuobjdump"
build_dir="build"

# scale_mode -> "scale_type default_sfa"
declare -A SCALE_TYPE=(
  [VEC4_UE4M3]=ue4m3
  [VEC4_UE8M0]=ue8m0
  [VEC2_UE8M0]=ue8m0
)
declare -A DEFAULT_SF=(
  [VEC4_UE4M3]=0x38
  [VEC4_UE8M0]=127
  [VEC2_UE8M0]=127
)
SCALE_MODES=(VEC4_UE4M3 VEC4_UE8M0 VEC2_UE8M0)
VARIANTS=(e2m1_e2m1 e0m3_e2m1 e2m1_e0m3 e0m3_e0m3)

# scale_vec::2X does not support the E0M3 format-select bits: cuobjdump still prints an E0M3
# mnemonic for the patched binary, but the GPU traps with "an illegal instruction was encountered"
# at every combination tried. Confirmed reproducible (3 reruns) and does not corrupt GPU state --
# see the correctness log for the raw fault text. Treated as an expected/known result, not a
# script bug.
UNSUPPORTED_RE='^VEC2_UE8M0:(e0m3_e2m1|e2m1_e0m3|e0m3_e0m3)$'

pass_count=0
fail_count=0
xfail_count=0

compile_mode() {
  local mode="$1"
  local out_dir="${build_dir}/probe/${mode}"
  mkdir -p "${out_dir}"
  "${nvcc}" -std=c++17 -O3 -DOMMA_SCALE_MODE="${mode}" \
    -gencode=arch=compute_120a,code=sm_120a \
    mma_probe.cu -o "${out_dir}/omma_e2m1_e2m1"
  python3 patch_formats.py "${out_dir}/omma_e2m1_e2m1" "${out_dir}" --cuobjdump "${cuobjdump}"
}

show_sass() {
  local binary="$1"
  echo "=== SASS: ${binary} ==="
  "${cuobjdump}" --dump-sass "${binary}" 2>/dev/null | grep -E 'OMMA|QMMA'
}

variant_formats() {
  # variant name -> "a_format b_format"
  case "$1" in
    e2m1_e2m1) echo "e2m1 e2m1" ;;
    e0m3_e2m1) echo "e0m3 e2m1" ;;
    e2m1_e0m3) echo "e2m1 e0m3" ;;
    e0m3_e0m3) echo "e0m3 e0m3" ;;
  esac
}

check_variant() {
  local mode="$1" variant="$2" a_nibble="$3" b_nibble="$4" sfa="$5" sfb="$6"
  local binary="${build_dir}/probe/${mode}/omma_${variant}"
  local scale_type="${SCALE_TYPE[${mode}]}"
  read -r a_format b_format < <(variant_formats "${variant}")
  local expected
  expected="$(python3 expected_value.py "${a_format}" "${b_format}" \
    "${a_nibble}" "${b_nibble}" "${scale_type}" "${sfa}" "${sfb}")"

  local raw_output run_status
  raw_output="$("${binary}" "${a_nibble}" "${b_nibble}" "${sfa}" "${sfb}" 2>&1)"
  run_status=$?

  local key="${mode}:${variant}"
  if [[ ${run_status} -ne 0 ]]; then
    if [[ "${key}" =~ ${UNSUPPORTED_RE} ]] && [[ "${raw_output}" == *"illegal instruction"* ]]; then
      printf "XFAIL %-14s %-10s a=%-4s b=%-4s -- traps with illegal instruction (expected: E0M3 unsupported under scale_vec::2X)\n" \
        "${mode}" "${variant}" "${a_nibble}" "${b_nibble}"
      xfail_count=$((xfail_count + 1))
      return 0
    fi
    printf "FAIL  %-14s %-10s a=%-4s b=%-4s -- binary exited %d: %s\n" \
      "${mode}" "${variant}" "${a_nibble}" "${b_nibble}" "${run_status}" "${raw_output}"
    fail_count=$((fail_count + 1))
    return 1
  fi

  local observed
  observed="$(printf '%s\n' "${raw_output}" | sed -n 's/.* min=\([^ ]*\).*/\1/p')"
  if ! awk -v a="${observed}" -v b="${expected}" 'BEGIN { exit !(a == b) }'; then
    printf "FAIL  %-14s %-10s a=%-4s b=%-4s -- observed=%s expected=%s\n" \
      "${mode}" "${variant}" "${a_nibble}" "${b_nibble}" "${observed}" "${expected}"
    fail_count=$((fail_count + 1))
    return 1
  fi
  printf "PASS  %-14s %-10s a=%-4s b=%-4s -- observed=%s\n" \
    "${mode}" "${variant}" "${a_nibble}" "${b_nibble}" "${observed}"
  pass_count=$((pass_count + 1))
}


check_bit7_invariant() {
  # UE4M3 scale bit 7 is architecturally unused (scales are non-negative; the real hardware masks
  # scale codes to 7 bits -- see decode_ue4m3() in mma_probe.cu). This is Stage 0 validation for
  # smuggling a per-block e0m3/e2m1 flag into that bit: confirm the GPU produces a bit-identical
  # result whether bit 7 is set or clear, for several nibble/scale combinations, symmetric and
  # asymmetric across A/B. VEC4_UE4M3 only -- UE8M0's 8 bits are all meaningful exponent, no spare
  # bit exists there.
  local mode="VEC4_UE4M3" variant="e2m1_e2m1"
  local binary="${build_dir}/probe/${mode}/omma_${variant}"
  local cases=(
    "7 7 0x38 0x38 0xb8 0xb8"
    "2 3 0x34 0x40 0xb4 0xc0"
    "1 5 0x40 0x38 0xc0 0x38"
    "4 6 0x38 0x38 0x38 0xb8"
  )
  local a b sfa_clear sfb_clear sfa_set sfb_set
  for case in "${cases[@]}"; do
    read -r a b sfa_clear sfb_clear sfa_set sfb_set <<<"${case}"
    local out_clear out_set obs_clear obs_set
    out_clear="$("${binary}" "${a}" "${b}" "${sfa_clear}" "${sfb_clear}" 2>&1)"
    out_set="$("${binary}" "${a}" "${b}" "${sfa_set}" "${sfb_set}" 2>&1)"
    obs_clear="$(printf '%s\n' "${out_clear}" | sed -n 's/.* min=\([^ ]*\).*/\1/p')"
    obs_set="$(printf '%s\n' "${out_set}" | sed -n 's/.* min=\([^ ]*\).*/\1/p')"
    if [[ "${obs_clear}" != "${obs_set}" ]]; then
      printf "FAIL  %-14s %-10s a=%-4s b=%-4s -- bit7 changed result: clear(sfa=%s,sfb=%s)=%s set(sfa=%s,sfb=%s)=%s\n" \
        "${mode}" "bit7" "${a}" "${b}" "${sfa_clear}" "${sfb_clear}" "${obs_clear}" "${sfa_set}" "${sfb_set}" "${obs_set}"
      fail_count=$((fail_count + 1))
      continue
    fi
    printf "PASS  %-14s %-10s a=%-4s b=%-4s -- bit7 ignored by hardware, result=%s (both bit7=0 and bit7=1)\n" \
      "${mode}" "bit7" "${a}" "${b}" "${obs_clear}"
    pass_count=$((pass_count + 1))
  done
}


compile_dispatch_probe() {
  local out_dir="${build_dir}/mixed"
  mkdir -p "${out_dir}"
  "${nvcc}" -std=c++17 -O3 \
    -gencode=arch=compute_120a,code=sm_120a \
    mixed_dispatch_probe.cu -o "${out_dir}/probe_baseline"
  python3 patch_dispatch_formats.py "${out_dir}/probe_baseline" "${out_dir}/probe_patched" \
    --cuobjdump "${cuobjdump}"
}

check_dispatch() {
  # Stage 0 Step 2: mixed_dispatch_probe.cu has 4 independently-patched mma.sync sites, selected
  # at runtime by (flag_a, flag_b) extracted from bit 7 of sfa_byte/sfb_byte via a shfl_sync
  # broadcast from a single owning lane. This proves the mechanism a real mixed E0M3/E2M1 GEMM
  # mainloop would need, independent of the much larger CUTLASS integration.
  local a_nibble="$1" b_nibble="$2" sfa_byte="$3" sfb_byte="$4" owner_lane="$5"
  local binary="${build_dir}/mixed/probe_patched"
  local flag_a flag_b a_format b_format
  flag_a=$(( (sfa_byte & 0x80) != 0 ))
  flag_b=$(( (sfb_byte & 0x80) != 0 ))
  a_format=$([[ ${flag_a} -eq 1 ]] && echo e0m3 || echo e2m1)
  b_format=$([[ ${flag_b} -eq 1 ]] && echo e0m3 || echo e2m1)
  local expected
  expected="$(python3 expected_value.py "${a_format}" "${b_format}" \
    "${a_nibble}" "${b_nibble}" ue4m3 "${sfa_byte}" "${sfb_byte}")"

  local raw_output observed
  raw_output="$("${binary}" "${a_nibble}" "${b_nibble}" "${sfa_byte}" "${sfb_byte}" "${owner_lane}" 2>&1)"
  observed="$(printf '%s\n' "${raw_output}" | sed -n 's/.* min=\([^ ]*\).*/\1/p')"
  if ! awk -v a="${observed}" -v b="${expected}" 'BEGIN { exit !(a == b) }'; then
    printf "FAIL  %-14s %-10s a=%-4s b=%-4s owner=%-3s -- observed=%s expected=%s\n" \
      "dispatch" "${a_format}x${b_format}" "${a_nibble}" "${b_nibble}" "${owner_lane}" "${observed}" "${expected}"
    fail_count=$((fail_count + 1))
    return 1
  fi
  printf "PASS  %-14s %-10s a=%-4s b=%-4s owner=%-3s -- observed=%s\n" \
    "dispatch" "${a_format}x${b_format}" "${a_nibble}" "${b_nibble}" "${owner_lane}" "${observed}"
  pass_count=$((pass_count + 1))
}

main() {
  mkdir -p "${build_dir}/logs"

  echo "### Compiling and patching all scale-mode baselines ###"
  for mode in "${SCALE_MODES[@]}"; do
    compile_mode "${mode}"
  done

  echo
  echo "### SASS decoder output for every (scale mode x tensor format) combination ###"
  for mode in "${SCALE_MODES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      show_sass "${build_dir}/probe/${mode}/omma_${variant}"
    done
  done

  echo
  echo "### Correctness: full codebook boundary values (nibble 7, unit scale) ###"
  for mode in "${SCALE_MODES[@]}"; do
    local_sfa="${DEFAULT_SF[${mode}]}"
    for variant in "${VARIANTS[@]}"; do
      check_variant "${mode}" "${variant}" 7 7 "${local_sfa}" "${local_sfa}"
    done
  done

  echo
  echo "### Correctness: E0M3 sign handling (nibble 0x9 = -1) ###"
  for mode in "${SCALE_MODES[@]}"; do
    local_sfa="${DEFAULT_SF[${mode}]}"
    check_variant "${mode}" e0m3_e0m3 9 1 "${local_sfa}" "${local_sfa}"
  done

  echo
  echo "### Correctness: non-unit, mismatched A/B scale factors ###"
  for mode in "${SCALE_MODES[@]}"; do
    case "${SCALE_TYPE[${mode}]}" in
      ue4m3) sfa=0x34; sfb=0x40 ;;   # 0.75, 2.0
      ue8m0) sfa=126;  sfb=128 ;;    # 0.5,  2.0
    esac
    for variant in "${VARIANTS[@]}"; do
      check_variant "${mode}" "${variant}" 2 3 "${sfa}" "${sfb}"
    done
  done

  echo
  echo "### Correctness: UE4M3 scale bit 7 is hardware-ignored (Stage 0 validation) ###"
  check_bit7_invariant

  echo
  echo "### Stage 0 Step 2: dual-path branch-and-patch dispatch (mixed_dispatch_probe) ###"
  compile_dispatch_probe
  show_sass "${build_dir}/mixed/probe_patched"
  check_dispatch 7 7 0x38 0x38 0
  check_dispatch 7 7 0xB8 0x38 0
  check_dispatch 7 7 0x38 0xB8 0
  check_dispatch 7 7 0xB8 0xB8 0
  check_dispatch 2 3 0xB4 0x40 17
  check_dispatch 2 3 0x34 0xC0 17
  check_dispatch 9 1 0xB8 0xB8 31

  echo
  echo "### Summary: ${pass_count} passed, ${fail_count} failed, ${xfail_count} expected-fail (E0M3 x scale_vec::2X) ###"
  [[ ${fail_count} -eq 0 ]]
}

main "$@" 2>&1 | tee "${build_dir}/logs/run_tests.txt"
exit "${PIPESTATUS[0]}"
