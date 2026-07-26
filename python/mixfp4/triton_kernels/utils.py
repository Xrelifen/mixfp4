"""Device helpers shared by the mixfp4 Triton kernels.

The structure here follows GemLite's ``triton_kernels/utils.py`` closely -- the constexpr-gated
masked load, the tile swizzles, the shared-memory estimator and the M-bucketing for autotune keys
are all the same idioms, because they solve the same problems.  What is new is
:func:`dequantize_mixfp4`, which decodes two codebooks in one gather.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..codebook import combined_lut  # noqa: F401  (re-exported for the forward wrappers)

GPU_COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


# --- loads -------------------------------------------------------------------------------------


@triton.jit
def load_ptr(ptrs, mask, eviction_policy, apply_mask: tl.constexpr, other=0.0):
    """Masked load whose mask operand disappears entirely when the extent is known even."""
    if apply_mask:
        return tl.load(ptrs, mask=mask, other=other, eviction_policy=eviction_policy)
    return tl.load(ptrs, eviction_policy=eviction_policy)


# --- tile swizzles -----------------------------------------------------------------------------


@triton.jit
def swizzle_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr):
    """Group-M swizzle: consecutive programs share B tiles, improving L2 reuse."""
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr):
    """M-fastest traversal.  Preferred when B is packed, so B tiles stay resident across pid_m."""
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    return pid_m, pid_n


# --- the mixed-format decode -------------------------------------------------------------------


@triton.jit
def dequantize_mixfp4(b_packed, sf_raw, lut, q_shift, out_dtype: tl.constexpr):
    """Decode a tile of mixed E2M1/E0M3 weights to ``out_dtype``.

    Args:
        b_packed: ``(BLOCK_K, BLOCK_N)`` uint8, two nibbles per byte along K.
        sf_raw: ``(BLOCK_K, BLOCK_N)`` uint8 scale bytes, already broadcast to element resolution.
            Bit 7 is the format tag, bits 0..6 the UE4M3 magnitude.
        lut: ``(BLOCK_K, 32)`` -- ``combined_lut`` broadcast, hoisted out of the K loop.
        q_shift: ``(BLOCK_K, 1)`` int32, 0 for even K indices and 4 for odd.  Loop-invariant.

    The whole point of the format lives in one expression: ``nibble + 16 * fmt``.  The tag selects
    which half of the 32-entry table to read from, so choosing a codebook per group costs one
    integer add and nothing else -- no branch, no predication, no duplicated code path.

    Bits 0..6 of a UE4M3 byte are laid out exactly like ``float8_e4m3fn`` with the sign clear
    (4 exponent bits, 3 mantissa bits, bias 7), so masking the tag off and bitcasting *is* the
    decode.  This matches what the tensor core does with the same byte on the CUDA path.
    """
    nibble = ((b_packed >> q_shift) & 0xF).to(tl.int32)
    fmt = (sf_raw >> 7).to(tl.int32)
    scale = (sf_raw & 0x7F).to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = tl.gather(lut, nibble + 16 * fmt, axis=1)
    return (value * scale).to(out_dtype)


@triton.jit
def load_lut(lut_ptr, BLOCK_K: tl.constexpr):
    """Hoist the 32-entry codebook table into registers, broadcast to gather's expected rank."""
    return tl.load(lut_ptr + tl.arange(0, 32), eviction_policy="evict_last")[None, :].broadcast_to(
        (BLOCK_K, 32))


# --- host-side autotune support ------------------------------------------------------------------


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def next_power_of_2(v: int) -> int:
    return 2 ** int(math.ceil(math.log2(v)))


def estimate_shared_memory_per_block(block_m, block_n, block_k, a_sizeof, num_stages) -> int:
    """Rough SMEM footprint of one pipelined stage set.

    A tile costs ``BM*BK`` at the activation's width.  B costs ``BK*BN/2`` packed plus the
    dequantised ``BK*BN`` at activation width, since the MMA reads the decoded values.  The scale
    tile is ``BK*BN`` bytes as loaded (it is broadcast to element resolution before use, which is
    redundant in *loads* but not in bytes -- see the note in the GEMM kernel).  The 1.15 factor
    covers alignment padding and the double-buffered pointers Triton keeps around it.
    """
    a_smem = block_m * block_k * a_sizeof
    b_smem = (block_k // 2) * block_n + block_k * block_n * a_sizeof
    sf_smem = (block_k // 16) * block_n
    return int((a_smem + b_smem + sf_smem) * max(num_stages, 1) * 1.15)


def get_gpu_shared_memory() -> int:
    from triton.runtime import driver
    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)


def gpu_supports_float16_acc(ref_gpus=("5090", "5080", "5070", "5060",
                                       "4090", "4080", "4070", "4060",
                                       "3090", "3080", "3070", "3060")) -> bool:
    """Consumer parts run fp16 accumulate at 2x fp32.  Datacenter parts do not."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_properties(0).name.lower()
    return any(g in name for g in ref_gpus)


def gpu_supports_bfloat16_atomicadd() -> bool:
    return GPU_COMPUTE_CAPABILITY >= 9


def _generate_m_lookup(max_m=4096, min_split=32, divisors=(2, 4)):
    """Powers of two plus interpolated midpoints, so nearby M values share a tuned config."""
    vals = set()
    i = 0
    while (v := 2 ** i) <= max_m:
        vals.add(v)
        nxt = 2 ** (i + 1)
        if v >= min_split and nxt <= max_m:
            for d in divisors:
                vals.add((v + nxt) // d)
        i += 1
    ordered = sorted(vals)
    lookup = [0] * (max_m + 1)
    for m in range(max_m + 1):
        lookup[m] = min((x for x in ordered if x >= m), default=max_m)
    return lookup


M_MAXVAL = 4096
M_MAPPING = _generate_m_lookup(M_MAXVAL)


def get_closest_m(m: int) -> int:
    """Round M up to the next autotune bucket, so M=100 and M=128 share one tuned config."""
    return M_MAPPING[m] if m <= M_MAXVAL else M_MAXVAL


NUM_SMS = (torch.cuda.get_device_properties("cuda").multi_processor_count
           if torch.cuda.is_available() else 1)
