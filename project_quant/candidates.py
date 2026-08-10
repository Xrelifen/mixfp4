"""Construction of independently scaled E2M1 and E0M3 K16 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .scale import ScaleRule, build_scaled_candidate, compute_global_scale


SCALE_GROUP_SIZE = 16
TAIL_POLICY = (
    "right_zero_pad_only_the_final_partial_K16_scale_block_then_unpad; "
    "partial_N_or_M_and_K_format_regions_use_only_real_K16_blocks"
)


@dataclass
class CandidateBlocks:
    original_shape: tuple[int, ...]
    source_dtype: torch.dtype
    operand_role: str
    rows: int
    k: int
    groups: int
    padded_k: int
    tail_padding: int
    grouped_original: torch.Tensor
    e2_dequant: torch.Tensor
    e0_dequant: torch.Tensor
    e2_codes: torch.Tensor
    e0_codes: torch.Tensor
    e2_scales: torch.Tensor
    e0_scales: torch.Tensor
    e2_errors: torch.Tensor
    e0_errors: torch.Tensor
    e2_scale_expansion_ids: torch.Tensor
    e0_scale_expansion_ids: torch.Tensor
    global_scale: torch.Tensor
    scale_rule: str


def validate_layout(x: torch.Tensor, *, scale_group_size: int, operand_role: str) -> None:
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")
    if x.ndim < 2:
        raise ValueError("x must preserve a row dimension and a final K dimension")
    if operand_role == "weight_b" and x.ndim != 2:
        raise ValueError("weight_b requires the exact [N,K] matrix layout")
    if operand_role not in {"weight_b", "activation_a"}:
        raise ValueError("operand_role must be 'weight_b' or 'activation_a'")
    if scale_group_size != SCALE_GROUP_SIZE:
        raise ValueError("the primary MixFP4 study fixes scale_group_size=16")
    if x.shape[-1] <= 0:
        raise ValueError("K must be positive")


@torch.no_grad()
def build_candidates(
    x: torch.Tensor,
    *,
    scale_group_size: int = SCALE_GROUP_SIZE,
    operand_role: Literal["weight_b", "activation_a"] = "weight_b",
    scale_rule: ScaleRule = "standard",
) -> CandidateBlocks:
    validate_layout(x, scale_group_size=scale_group_size, operand_role=operand_role)
    original_shape = tuple(x.shape)
    k = x.shape[-1]
    rows = x.numel() // k
    source = x.detach().reshape(rows, k).to(torch.float32)
    tail_padding = (-k) % scale_group_size
    padded = F.pad(source, (0, tail_padding)) if tail_padding else source
    grouped = padded.reshape(rows, -1, scale_group_size)
    global_scale = compute_global_scale(grouped, scale_rule)
    e2 = build_scaled_candidate(
        grouped, format_name="e2m1", global_scale=global_scale, scale_rule=scale_rule
    )
    e0 = build_scaled_candidate(
        grouped, format_name="e0m3", global_scale=global_scale, scale_rule=scale_rule
    )
    return CandidateBlocks(
        original_shape=original_shape,
        source_dtype=x.dtype,
        operand_role=operand_role,
        rows=rows,
        k=k,
        groups=grouped.shape[1],
        padded_k=padded.shape[1],
        tail_padding=tail_padding,
        grouped_original=grouped,
        e2_dequant=e2.dequant,
        e0_dequant=e0.dequant,
        e2_codes=e2.codes,
        e0_codes=e0.codes,
        e2_scales=e2.scales,
        e0_scales=e0.scales,
        e2_errors=e2.errors,
        e0_errors=e0.errors,
        e2_scale_expansion_ids=e2.expansion_ids,
        e0_scale_expansion_ids=e0.expansion_ids,
        global_scale=global_scale,
        scale_rule=scale_rule,
    )
