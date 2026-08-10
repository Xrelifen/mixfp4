"""Public layout-preserving MixFP4 quantization API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .candidates import SCALE_GROUP_SIZE, TAIL_POLICY, CandidateBlocks, build_candidates
from .metrics import calculate_region_metrics, tensor_error_metrics
from .region import normalize_region, region_shape
from .scale import ScaleRule
from .selector import Selection, mse_selection, output_aware_selection


@dataclass
class GranularityResult:
    dequant: torch.Tensor
    format_ids: torch.Tensor
    oracle_format_ids: torch.Tensor
    selected_scales: torch.Tensor
    selected_scale_expansion_ids: torch.Tensor
    e2_scales: torch.Tensor
    e0_scales: torch.Tensor
    e2_errors: torch.Tensor
    e0_errors: torch.Tensor
    summary: dict[str, Any]
    regions: list[dict[str, Any]]


def _forced_selection(
    candidates: CandidateBlocks,
    selection: Selection,
    *,
    use_e0: bool,
) -> Selection:
    ids = torch.full_like(selection.selected_ids, use_e0)
    return Selection(
        selected_ids=ids,
        mse_ids=selection.mse_ids,
        sensitivity_ids=selection.sensitivity_ids,
        mse_region_e2=selection.mse_region_e2,
        mse_region_e0=selection.mse_region_e0,
        sensitivity_region_e2=selection.sensitivity_region_e2,
        sensitivity_region_e0=selection.sensitivity_region_e0,
        sensitivity_block_e2=selection.sensitivity_block_e2,
        sensitivity_block_e0=selection.sensitivity_block_e0,
    )


@torch.no_grad()
def quant_mixfp4_granularity(
    x: torch.Tensor,
    *,
    scale_group_size: int = SCALE_GROUP_SIZE,
    format_region: str = "oracle16",
    operand_role: Literal["weight_b", "activation_a"] = "weight_b",
    selector: Literal["mse", "output_aware", "activation_aware"] = "mse",
    scale_rule: ScaleRule = "standard",
    calibration_stats: dict[str, Any] | None = None,
    transform: Any = None,
    permutation: torch.Tensor | None = None,
    return_stats: bool = False,
    collect_regions: bool = False,
    region_sample_limit: int | None = None,
) -> torch.Tensor | GranularityResult:
    """Quantize with K16 scales while varying only the datatype decision region.

    Activation leading dimensions are folded into the matrix M axis, never
    across K.  Weight permutations are returned in their permuted layout so a
    model adapter must prove/fold the inverse.  Transforms similarly act along K
    and must be paired on the other operand by the adapter.
    """
    working = x
    if permutation is not None:
        if operand_role != "weight_b" or x.ndim != 2:
            raise ValueError("permutation is defined only for [N,K] weights")
        if permutation.numel() != x.shape[0]:
            raise ValueError("permutation length must equal N")
        working = working.index_select(0, permutation.to(working.device))
    if transform is not None:
        from .rotation import apply_transform

        working = apply_transform(working, transform)

    mode = normalize_region(format_region, operand_role)
    candidates = build_candidates(
        working,
        scale_group_size=scale_group_size,
        operand_role=operand_role,
        scale_rule=scale_rule,
    )
    shape = region_shape(mode, operand_role, candidates.rows, candidates.groups)
    if selector in {"output_aware", "activation_aware"}:
        if operand_role != "weight_b" or mode != "n8k64":
            raise ValueError("output-aware selection is defined for N8K64 weights")
        if calibration_stats is None:
            raise ValueError("output-aware selection requires calibration_stats")
        selection = output_aware_selection(candidates, shape, calibration_stats)
        selector_name = "output_aware"
    elif selector == "mse":
        selection = mse_selection(candidates, shape)
        selector_name = "mse"
    else:
        raise ValueError(f"unknown selector={selector!r}")

    forced_format = None
    if mode == "all_e2m1":
        selection = _forced_selection(candidates, selection, use_e0=False)
        forced_format = "e2m1"
    elif mode == "all_e0m3":
        selection = _forced_selection(candidates, selection, use_e0=True)
        forced_format = "e0m3"
    elif mode == "oracle16":
        oracle = candidates.e0_errors < candidates.e2_errors
        selection = Selection(
            selected_ids=oracle,
            mse_ids=oracle,
            sensitivity_ids=None,
            mse_region_e2=candidates.e2_errors,
            mse_region_e0=candidates.e0_errors,
            sensitivity_region_e2=None,
            sensitivity_region_e0=None,
            sensitivity_block_e2=None,
            sensitivity_block_e0=None,
        )

    selected = torch.where(
        selection.selected_ids[..., None], candidates.e0_dequant, candidates.e2_dequant
    )
    selected_scales = torch.where(
        selection.selected_ids, candidates.e0_scales, candidates.e2_scales
    )
    selected_expansions = torch.where(
        selection.selected_ids,
        candidates.e0_scale_expansion_ids,
        candidates.e2_scale_expansion_ids,
    )
    flat = selected.reshape(candidates.rows, candidates.padded_k)
    if candidates.tail_padding:
        flat = flat[:, : candidates.k]
    dequant = flat.reshape(candidates.original_shape).to(candidates.source_dtype)
    if not return_stats:
        return dequant

    region_summary, records = calculate_region_metrics(
        candidates,
        selection,
        shape,
        mode=mode,
        selector_name=selector_name,
        collect_records=collect_regions,
        region_sample_limit=region_sample_limit,
        forced_format=forced_format,
    )
    oracle_ids = candidates.e0_errors < candidates.e2_errors
    margin = candidates.e2_errors - candidates.e0_errors
    selected_count = int(selection.selected_ids.sum().item())
    summary: dict[str, Any] = {
        **tensor_error_metrics(working, dequant),
        **region_summary,
        "scale_group_size": scale_group_size,
        "scale_rule": scale_rule,
        "format_granularity": mode,
        "format_region_rows": shape.rows,
        "format_region_k16_groups": shape.k16_groups,
        "format_region_k_values": shape.k_values,
        "operand_role": operand_role,
        "selector": selector_name,
        "tail_policy": TAIL_POLICY,
        "tail_padding_values": candidates.tail_padding,
        "rows": candidates.rows,
        "N_or_M": candidates.rows,
        "K": candidates.k,
        "numel": working.numel(),
        "num_scale_blocks": candidates.rows * candidates.groups,
        "e0_count": selected_count,
        "e2_count": selection.selected_ids.numel() - selected_count,
        "e0_ratio": float(selection.selected_ids.float().mean().item()),
        "e2_ratio": float((~selection.selected_ids).float().mean().item()),
        "oracle_e0_count": int(oracle_ids.sum().item()),
        "oracle_e2_count": oracle_ids.numel() - int(oracle_ids.sum().item()),
        "oracle_e0_ratio": float(oracle_ids.float().mean().item()),
        "mean_format_margin": float(margin.mean().item()),
        "mean_signed_format_margin": float(margin.mean().item()),
        "mean_abs_format_margin": float(margin.abs().mean().item()),
        "median_format_margin": float(margin.median().item()),
        "oracle_mse": float(torch.minimum(candidates.e2_errors, candidates.e0_errors).sum().item() / working.numel()),
        "constrained_mse": float(region_summary["constrained_error"] / working.numel()),
        "selected_mse": float(region_summary["selected_weight_error"] / working.numel()),
        "global_scale": float(candidates.global_scale.item()),
        "four_over_six_e2_expanded_ratio": float(candidates.e2_scale_expansion_ids.float().mean().item()),
        "four_over_six_e0_expanded_ratio": float(candidates.e0_scale_expansion_ids.float().mean().item()),
        "transform": getattr(transform, "name", transform) if transform is not None else "identity",
        "permutation_applied": permutation is not None,
    }
    e2_total = candidates.e2_errors.sum()
    summary["baseline_normalized_regret"] = float(
        (torch.tensor(region_summary["granularity_regret"], device=e2_total.device) / e2_total.clamp_min(1e-30)).item()
    )
    return GranularityResult(
        dequant=dequant,
        format_ids=selection.selected_ids.to(torch.uint8),
        oracle_format_ids=oracle_ids.to(torch.uint8),
        selected_scales=selected_scales,
        selected_scale_expansion_ids=selected_expansions,
        e2_scales=candidates.e2_scales,
        e0_scales=candidates.e0_scales,
        e2_errors=candidates.e2_errors,
        e0_errors=candidates.e0_errors,
        summary=summary,
        regions=records,
    )
