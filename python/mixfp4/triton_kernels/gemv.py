"""A16W4-mixfp4 GEMV: the batch-size-1..4 decode kernel.

At M <= 4 there is no useful matrix multiply left.  ``tl.dot`` would pad M to 16 and throw away
three quarters of the tensor core, so this kernel reduces elementwise instead and puts the *entire*
K loop in the grid: one program owns one output row, one BLOCK_N slice, and one BLOCK_K slab, and
adds its partial sum atomically.  That is GemLite's GEMV shape, and it is right for the same
reason: token generation is bandwidth bound, so the only thing that matters is issuing enough
independent loads to saturate memory.

One structural note.  GemLite's fastest bs=1 kernel is ``gemv_revsplitK``, which unrolls two K
stages that share a single scale load.  That trick needs ``BLOCK_K * 2 <= group_size``; NVFP4's
group size is 16, so it would cap BLOCK_K at 8 and cost more in launch count than it saves in
metadata traffic.  It is a real technique that this format's group size simply does not admit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..codebook import combined_lut
from .config import AUTOTUNE_MODE
from .tuning import build_configs, make_pruner
from .utils import dequantize_mixfp4, load_lut

KEYS = ["M", "N", "K"]
MATMUL_TYPE = "GEMV"


@triton.autotune(
    configs=build_configs(AUTOTUNE_MODE, block_m=[1], block_n=[32, 64, 128, 256],
                          block_k=[16, 32, 64, 128], zero_output=True),
    key=KEYS,
    prune_configs_by={"early_config_prune": make_pruner(MATMUL_TYPE, force_block_m=1,
                                                        min_block_k=16, zero_output=True)},
    warmup=20, rep=50,
)
@triton.heuristics({
    "EVEN_N": lambda a: a["N"] % a["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def gemv_kernel(
    a_ptr, b_ptr, sf_ptr, c_ptr, lut_ptr,
    M, N, K, meta_scale,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    a_sizeof: tl.constexpr,
    acc_dtype: tl.constexpr,
    data_contiguous: tl.constexpr,
    atomic_mode: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, A_load_order: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = pid % M
    pid_n = pid // M

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    if data_contiguous:
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + pid_m * stride_am + offs_k * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    s_ptrs = sf_ptr + (offs_k[:, None] // 16) * stride_sk + offs_n[None, :] * stride_sn
    q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]

    lut = load_lut(lut_ptr, BLOCK_SIZE_K)

    if EVEN_K:
        if EVEN_N:
            a = tl.load(a_ptrs, eviction_policy="evict_last")
            b = tl.load(b_ptrs, eviction_policy="evict_first")
            s = tl.load(s_ptrs, eviction_policy="evict_last")
        else:
            n_valid = offs_n[None, :] < N
            a = tl.load(a_ptrs, eviction_policy="evict_last")
            b = tl.load(b_ptrs, mask=n_valid, other=0, eviction_policy="evict_first")
            s = tl.load(s_ptrs, mask=n_valid, other=0, eviction_policy="evict_last")
    else:
        k_valid = offs_k < K
        bs_mask = k_valid[:, None] & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=k_valid, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_ptrs, mask=bs_mask, other=0, eviction_policy="evict_first")
        s = tl.load(s_ptrs, mask=bs_mask, other=0, eviction_policy="evict_last")

    b_deq = dequantize_mixfp4(b, s, lut, q_shift, acc_dtype)
    acc = tl.sum(a[:, None].to(acc_dtype) * b_deq, axis=0, keep_dims=True)
    acc = acc.to(tl.float32) * meta_scale

    c_ptrs = c_ptr + pid_m * stride_cm + offs_n[None, :] * stride_cn
    out = acc.to(c_ptr.dtype.element_ty)
    if EVEN_N:
        tl.atomic_add(c_ptrs, out, sem=atomic_mode)
    else:
        tl.atomic_add(c_ptrs, out, mask=offs_n[None, :] < N, sem=atomic_mode)


def gemv_forward(x: Tensor, W_q: Tensor, scales: Tensor, meta_scale: float,
                 N: int, K: int, acc_dtype=None, output_dtype=None) -> Tensor:
    """``x @ dequantize(W_q, scales).T`` for very small M, accumulating atomically over K."""
    M = x.shape[0]
    out_dtype = output_dtype or x.dtype
    # Accumulate in fp32 regardless of the output dtype.  Every program contributes one partial sum
    # per K slab, so a 4096-deep K at BLOCK_K=64 rounds 64 times into the same address; bf16's
    # 8-bit mantissa loses ~5x accuracy that way (measured) and fp16's 10-bit loses ~6x.  At the
    # M <= 4 this kernel serves, the fp32 buffer and its cast are a few tens of KB against megabytes
    # of weight traffic -- far too cheap to trade accuracy for.
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    lut = combined_lut(x.device, torch.float32)

    grid = lambda meta: (M * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                         triton.cdiv(K, meta["BLOCK_SIZE_K"]))
    gemv_kernel[grid](
        x, W_q, scales, out, lut,
        M, N, K, meta_scale,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        a_sizeof=x.element_size(),
        acc_dtype=acc_dtype or tl.float32,
        data_contiguous=True,
        atomic_mode="relaxed",
    )
    return out.to(out_dtype)
