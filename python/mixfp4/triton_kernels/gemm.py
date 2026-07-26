"""A16W4-mixfp4 GEMM: 16-bit activations against mixed E2M1/E0M3 4-bit weights.

The prefill / large-batch kernel.  One program owns a ``(BLOCK_M, BLOCK_N)`` output tile and walks
K, decoding each weight tile to the activation dtype before a normal ``tl.dot``.

Why software decode rather than the FP4 tensor core
---------------------------------------------------
``tl.dot_scaled`` can feed E2M1 nibbles straight to the block-scaled MMA, and for a pure-NVFP4
weight that is the right kernel.  It cannot express E0M3: the format is two bits of the *SASS
instruction encoding*, reachable only by patching a compiled cubin (which is exactly what
``scripts/patch_mixed_nvfp4_gemm.py`` does for the CUTLASS path).  Triton has no hook into that
stage, and no arithmetic identity maps one codebook onto the other -- E0M3's ``{0..7}`` and
E2M1's ``{0,.5,1,1.5,2,3,4,6}`` agree only up to index 4 under any single scale factor.

So the mixed path decodes in registers.  What that buys, beyond portability, is granularity: the
CUDA kernel's format granule is bounded below by one MMA atom's footprint (16 rows, and the
report's cost curve shows the arm count exploding past 8), while here the granule is one scale
group -- 16 elements of K by one output column.  See ``dequantize_mixfp4`` in utils.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..codebook import combined_lut
from .config import AUTOTUNE_MODE
from .tuning import build_configs, make_pruner
from .utils import dequantize_mixfp4, get_closest_m, linear_tile, load_lut, load_ptr

KEYS = ["M_CLOSEST", "N", "K"]
MATMUL_TYPE = "GEMM"


@triton.autotune(
    configs=build_configs(AUTOTUNE_MODE),
    key=KEYS,
    prune_configs_by={"early_config_prune": make_pruner(MATMUL_TYPE)},
    warmup=20, rep=50,
)
@triton.heuristics({
    "EVEN_M": lambda a: a["M"] % a["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda a: a["N"] % a["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, sf_ptr, c_ptr, lut_ptr,
    M, N, K, M_CLOSEST, meta_scale,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    a_sizeof: tl.constexpr,
    acc_dtype: tl.constexpr,
    data_contiguous: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, A_load_order: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # B is packed, so keeping pid_m fastest holds one B tile resident across the M sweep.
    pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if data_contiguous:
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Two nibbles share a byte and sixteen elements share a scale, so both pointer sets index K at
    # reduced resolution.  The repeated addresses cost no extra bandwidth -- the loads coalesce in
    # L1 exactly as GemLite's packed-weight path relies on -- and in exchange the group index stays
    # loop-invariant, so only the base pointer advances.
    b_ptrs = b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    s_ptrs = sf_ptr + (offs_k[:, None] // 16) * stride_sk + offs_n[None, :] * stride_sn
    q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]

    lut = load_lut(lut_ptr, BLOCK_SIZE_K)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=NUM_STAGES):
        if EVEN_K:
            a_mask, bs_mask = mask_m, mask_n
        else:
            k_valid = (k * BLOCK_SIZE_K + offs_k) < K
            a_mask = mask_m & k_valid[None, :]
            bs_mask = k_valid[:, None] & mask_n

        if A_load_order == 0:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))
        b = load_ptr(b_ptrs, bs_mask, "evict_first", not (EVEN_N and EVEN_K), other=0)
        if A_load_order == 1:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))
        # A masked-off scale byte reads as 0: format tag clear, UE4M3 magnitude zero.  Paired with
        # a zero nibble that contributes exactly nothing to the dot product.
        s = load_ptr(s_ptrs, bs_mask, "evict_last", not (EVEN_N and EVEN_K), other=0)
        if A_load_order == 2:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))

        b_deq = dequantize_mixfp4(b, s, lut, q_shift, a.dtype)
        acc = tl.dot(a, b_deq, acc=acc, out_dtype=acc_dtype)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        s_ptrs += (BLOCK_SIZE_K // 16) * stride_sk

    # One global renormaliser undoes the shift that packed the group scales into UE4M3's range.
    acc = acc.to(tl.float32) * meta_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def gemm_forward(x: Tensor, W_q: Tensor, scales: Tensor, meta_scale: float,
                 N: int, K: int, acc_dtype=None, output_dtype=None) -> Tensor:
    """``x @ dequantize(W_q, scales).T`` for 2-D ``x`` of shape ``[M, K]``."""
    M = x.shape[0]
    out_dtype = output_dtype or x.dtype
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    lut = combined_lut(x.device, torch.float32)
    acc = acc_dtype or tl.float32

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    gemm_kernel[grid](
        x, W_q, scales, out, lut,
        M, N, K, get_closest_m(M), meta_scale,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        a_sizeof=x.element_size(),
        acc_dtype=acc,
        data_contiguous=True,
    )
    return out
