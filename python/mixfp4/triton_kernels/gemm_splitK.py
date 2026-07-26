"""A16W4-mixfp4 split-K GEMM: the small-batch decode kernel.

At batch sizes between roughly 4 and 64 there are not enough output tiles to fill the machine --
a 4096x4096 layer at M=8 is one row of tiles.  Splitting the K reduction across programs trades an
atomic accumulation for occupancy, which is the right trade whenever the kernel is bandwidth bound,
as weight-only quantised matmuls at low M always are.

The body is the GEMM's body with a strided K walk; the only real difference is the epilogue.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..codebook import combined_lut
from .config import AUTOTUNE_MODE
from .tuning import build_configs, make_pruner
from .utils import dequantize_mixfp4, get_closest_m, load_lut, load_ptr

KEYS = ["M_CLOSEST", "N", "K"]
MATMUL_TYPE = "GEMM_SPLITK"


@triton.autotune(
    configs=build_configs(AUTOTUNE_MODE, split_k=(1, 2, 4, 8), block_m=[16, 32, 64]),
    key=KEYS,
    prune_configs_by={"early_config_prune": make_pruner(MATMUL_TYPE, with_split_k=True)},
    warmup=20, rep=50,
)
@triton.heuristics({
    "EVEN_M": lambda a: a["M"] % a["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda a: a["N"] % a["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda a: a["K"] % (a["BLOCK_SIZE_K"] * a["SPLIT_K"]) == 0,
})
@triton.jit
def gemm_splitK_kernel(
    a_ptr, b_ptr, sf_ptr, c_ptr, lut_ptr,
    M, N, K, M_CLOSEST, meta_scale,
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
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # This program starts at its own slice and strides by the full split width.  BLOCK_SIZE_K is a
    # multiple of 16, so q_shift and the group offset stay loop-invariant despite the stride.
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    if data_contiguous:
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    s_ptrs = sf_ptr + (offs_k[:, None] // 16) * stride_sk + offs_n[None, :] * stride_sn
    q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]

    lut = load_lut(lut_ptr, BLOCK_SIZE_K)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K), num_stages=NUM_STAGES):
        if EVEN_K:
            a_mask, bs_mask = mask_m, mask_n
        else:
            k_valid = (k * BLOCK_SIZE_K * SPLIT_K + offs_k) < K
            a_mask = mask_m & k_valid[None, :]
            bs_mask = k_valid[:, None] & mask_n

        if A_load_order == 0:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))
        b = load_ptr(b_ptrs, bs_mask, "evict_first", not (EVEN_N and EVEN_K), other=0)
        if A_load_order == 1:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))
        s = load_ptr(s_ptrs, bs_mask, "evict_last", not (EVEN_N and EVEN_K), other=0)
        if A_load_order == 2:
            a = load_ptr(a_ptrs, a_mask, "evict_last", not (EVEN_M and EVEN_K))

        b_deq = dequantize_mixfp4(b, s, lut, q_shift, a.dtype)
        acc = tl.dot(a, b_deq, acc=acc, out_dtype=acc_dtype)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 2) * stride_bk
        s_ptrs += (BLOCK_SIZE_K * SPLIT_K // 16) * stride_sk

    acc = acc.to(tl.float32) * meta_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    out = acc.to(c_ptr.dtype.element_ty)
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, out, mask=c_mask, sem=atomic_mode)
    else:
        tl.store(c_ptrs, out, mask=c_mask)


def gemm_splitK_forward(x: Tensor, W_q: Tensor, scales: Tensor, meta_scale: float,
                        N: int, K: int, acc_dtype=None, output_dtype=None) -> Tensor:
    """``x @ dequantize(W_q, scales).T`` with the K reduction split across programs."""
    M = x.shape[0]
    out_dtype = output_dtype or x.dtype
    # fp32 accumulation buffer: with SPLIT_K > 1 the output address is a reduction target, and
    # rounding each partial sum to the output dtype costs an order of magnitude of accuracy.
    # Zeroed here as well as in the config pre-hook, because a SPLIT_K == 1 config stores rather
    # than accumulates and so carries no pre-hook.
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    lut = combined_lut(x.device, torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                         meta["SPLIT_K"])
    gemm_splitK_kernel[grid](
        x, W_q, scales, out, lut,
        M, N, K, get_closest_m(M), meta_scale,
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
