"""Tensor, K16-block, and true 2-D coarse-region diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch

from .candidates import CandidateBlocks
from .region import RegionShape, aggregate_regions, iter_region_bounds
from .selector import Selection


def tensor_error_metrics(source: torch.Tensor, dequant: torch.Tensor) -> dict[str, float]:
    x = source.detach().float().reshape(-1)
    q = dequant.detach().float().reshape(-1)
    delta = q - x
    error_energy = torch.dot(delta, delta)
    source_energy = torch.dot(x, x)
    q_energy = torch.dot(q, q)
    tiny = torch.finfo(torch.float32).tiny
    cosine = torch.dot(x, q) / torch.sqrt(source_energy * q_energy).clamp_min(tiny)
    return {
        "mse": float((error_energy / x.numel()).item()),
        "nmse": float((error_energy / source_energy.clamp_min(tiny)).item()),
        "relative_l2": float(torch.sqrt(error_energy / source_energy.clamp_min(tiny)).item()),
        "cosine_similarity": float(cosine.item()),
        "cosine_error": float((1 - cosine).item()),
        "mean_absolute_error": float(delta.abs().mean().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "max_absolute_error": float(delta.abs().max().item()),
        "max_abs_error": float(delta.abs().max().item()),
    }


def _sample_region_indices(total: int, limit: int | None) -> set[int]:
    if limit is None or total <= limit:
        return set(range(total))
    return set(
        torch.linspace(0, total - 1, steps=limit, dtype=torch.float64)
        .round()
        .long()
        .unique()
        .tolist()
    )


def calculate_region_metrics(
    candidates: CandidateBlocks,
    selection: Selection,
    shape: RegionShape,
    *,
    mode: str,
    selector_name: str,
    collect_records: bool,
    region_sample_limit: int | None,
    forced_format: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    e2 = candidates.e2_errors
    e0 = candidates.e0_errors
    oracle_ids = e0 < e2
    margins = e2 - e0
    p_region = aggregate_regions(margins.clamp_min(0), shape)
    n_region = aggregate_regions((-margins).clamp_min(0), shape)
    identity_regret = torch.minimum(p_region, n_region)
    oracle_region = aggregate_regions(torch.minimum(e2, e0), shape)
    mse_e2_region = aggregate_regions(e2, shape)
    mse_e0_region = aggregate_regions(e0, shape)
    mse_ids_region = mse_e0_region < mse_e2_region
    mse_constrained_region = torch.minimum(mse_e2_region, mse_e0_region)
    mse_regret = mse_constrained_region - oracle_region
    identity_residual = mse_regret - identity_regret

    selected_error_blocks = torch.where(selection.selected_ids, e0, e2)
    selected_error_region = aggregate_regions(selected_error_blocks, shape)
    selected_weight_regret = selected_error_region - oracle_region
    valid_counts = aggregate_regions(torch.ones_like(e2), shape)
    e0_counts = aggregate_regions(oracle_ids.to(e2.dtype), shape)
    e2_counts = valid_counts - e0_counts
    homogeneity = torch.maximum(e0_counts, e2_counts) / valid_counts
    margin_conflict = identity_regret / (p_region + n_region + 1e-30)
    margin_homogeneity = 1 - margin_conflict
    abs_margin_region = aggregate_regions(margins.abs(), shape)
    signed_margin_region = aggregate_regions(margins, shape)
    selector_disagreement = selection.selected_ids != selection.mse_ids
    selector_disagreement_region = aggregate_regions(selector_disagreement.to(e2.dtype), shape)
    changed_margins = margins[selector_disagreement]

    sensitivity_regret = None
    sensitivity_oracle_region = None
    sensitivity_coarse_region = None
    sensitivity_selected_region = None
    sensitivity_margin = None
    if selection.sensitivity_block_e2 is not None and selection.sensitivity_block_e0 is not None:
        se2 = selection.sensitivity_block_e2
        se0 = selection.sensitivity_block_e0
        sensitivity_margin = se2 - se0
        sensitivity_oracle_region = aggregate_regions(torch.minimum(se2, se0), shape)
        sensitivity_coarse_region = torch.minimum(
            aggregate_regions(se2, shape), aggregate_regions(se0, shape)
        )
        sensitivity_regret = sensitivity_coarse_region - sensitivity_oracle_region
        sensitivity_selected_region = aggregate_regions(
            torch.where(selection.selected_ids, se0, se2), shape
        )

    # Fixed-format baselines do not claim the R=min(P,N) coarse optimum.
    if forced_format is None:
        constrained_total = mse_constrained_region.sum()
        granularity_regret_total = mse_regret.sum()
    else:
        constrained_total = selected_error_region.sum()
        granularity_regret_total = selected_weight_regret.sum()
    oracle_total = oracle_region.sum()
    selected_total = selected_error_region.sum()
    flattened_h = homogeneity.double().flatten().cpu()
    flattened_c = margin_conflict.double().flatten().cpu()
    # This is the conflict ratio after weighting regions by total absolute
    # margin.  Keep it distinct from the arithmetic mean of per-region ratios.
    weighted_conflict = identity_regret.sum() / (p_region.sum() + n_region.sum() + 1e-30)
    summary: dict[str, Any] = {
        "num_format_regions": int(homogeneity.numel()),
        "oracle_error": float(oracle_total.item()),
        "constrained_error": float(constrained_total.item()),
        "granularity_regret": float(granularity_regret_total.item()),
        "normalized_regret": float((granularity_regret_total / oracle_total.clamp_min(1e-30)).item()),
        "mse_optimal_constrained_error": float(mse_constrained_region.sum().item()),
        "mse_optimal_granularity_regret": float(mse_regret.sum().item()),
        "identity_regret_min_pn": float(identity_regret.sum().item()),
        "identity_max_abs_residual": float(identity_residual.abs().max().item()),
        "selected_weight_error": float(selected_total.item()),
        "selected_weight_mse_regret": float((selected_total - oracle_total).item()),
        "selected_weight_normalized_regret": float(
            ((selected_total - oracle_total) / oracle_total.clamp_min(1e-30)).item()
        ),
        "mean_homogeneity": float(flattened_h.mean().item()),
        "mean_count_homogeneity": float(flattened_h.mean().item()),
        "median_homogeneity": float(flattened_h.median().item()),
        "p10_homogeneity": float(torch.quantile(flattened_h, 0.10).item()),
        "p90_homogeneity": float(torch.quantile(flattened_h, 0.90).item()),
        "mean_margin_conflict": float(flattened_c.mean().item()),
        "margin_weighted_conflict": float(weighted_conflict.item()),
        "mean_margin_homogeneity": float((1 - flattened_c).mean().item()),
        "selector_disagreement_count": int(selector_disagreement.sum().item()),
        "selector_disagreement_ratio": float(selector_disagreement.float().mean().item()),
        "selector_changed_mean_format_margin": (
            float(changed_margins.mean().item()) if changed_margins.numel() else 0.0
        ),
        "selector_changed_mean_abs_format_margin": (
            float(changed_margins.abs().mean().item()) if changed_margins.numel() else 0.0
        ),
    }
    if sensitivity_regret is not None:
        sensitivity_oracle_total = sensitivity_oracle_region.sum()
        sensitivity_coarse_total = sensitivity_coarse_region.sum()
        sensitivity_selected_total = sensitivity_selected_region.sum()
        summary.update(
            {
                "sensitivity_oracle_error": float(sensitivity_oracle_total.item()),
                "sensitivity_constrained_error": float(sensitivity_coarse_total.item()),
                "sensitivity_regret": float(sensitivity_regret.sum().item()),
                "sensitivity_normalized_regret": float(
                    (sensitivity_regret.sum() / sensitivity_oracle_total.clamp_min(1e-30)).item()
                ),
                "selected_sensitivity_error": float(sensitivity_selected_total.item()),
                "selected_sensitivity_regret": float(
                    (sensitivity_selected_total - sensitivity_oracle_total).item()
                ),
            }
        )
    if selection.sensitivity_region_e2 is not None and selection.sensitivity_region_e0 is not None:
        full_selected = torch.where(
            selection.sensitivity_region_e0 < selection.sensitivity_region_e2,
            selection.sensitivity_region_e0,
            selection.sensitivity_region_e2,
        )
        full_mse_selected = torch.where(
            mse_ids_region,
            selection.sensitivity_region_e0,
            selection.sensitivity_region_e2,
        )
        summary.update(
            {
                "calibration_output_sse_selected": float(full_selected.sum().item()),
                "calibration_output_sse_mse_selector": float(full_mse_selected.sum().item()),
                "calibration_output_sse_all_e2": float(selection.sensitivity_region_e2.sum().item()),
                "calibration_output_sse_all_e0": float(selection.sensitivity_region_e0.sum().item()),
            }
        )

    if not collect_records:
        return summary, []
    bounds = list(iter_region_bounds(candidates.rows, candidates.groups, shape))
    sampled = _sample_region_indices(len(bounds), region_sample_limit)
    records: list[dict[str, Any]] = []
    region_columns = math.ceil(candidates.groups / shape.k16_groups)
    for flat_index, (r0, r1, g0, g1) in enumerate(bounds):
        if flat_index not in sampled:
            continue
        rr = flat_index // region_columns
        rg = flat_index % region_columns
        count = int(valid_counts[rr, rg].item())
        actual_id = bool(selection.selected_ids[r0, g0].item())
        mse_id = bool(mse_ids_region[rr, rg].item())
        sensitivity_id = (
            bool(selection.sensitivity_ids[r0, g0].item())
            if selection.sensitivity_ids is not None
            else None
        )
        record: dict[str, Any] = {
            "operand_role": candidates.operand_role,
            "format_granularity": mode,
            "region_n_start": r0,
            "region_k_start": g0 * 16,
            "region_n_size": r1 - r0,
            "region_k_size": min(candidates.k, g1 * 16) - g0 * 16,
            "num_k16_blocks": count,
            "oracle_e0_count": int(e0_counts[rr, rg].item()),
            "oracle_e2_count": int(e2_counts[rr, rg].item()),
            "homogeneity": float(homogeneity[rr, rg].item()),
            "count_homogeneity": float(homogeneity[rr, rg].item()),
            "P_G": float(p_region[rr, rg].item()),
            "N_G": float(n_region[rr, rg].item()),
            "R_G": float(identity_regret[rr, rg].item()),
            "margin_conflict": float(margin_conflict[rr, rg].item()),
            "margin_homogeneity": float(margin_homogeneity[rr, rg].item()),
            "signed_format_margin": float(signed_margin_region[rr, rg].item() / count),
            "mean_abs_margin": float(abs_margin_region[rr, rg].item() / count),
            "max_abs_margin": float(margins[r0:r1, g0:g1].abs().max().item()),
            "oracle_error": float(oracle_region[rr, rg].item()),
            "constrained_error": float(
                (selected_error_region if forced_format is not None else mse_constrained_region)[rr, rg].item()
            ),
            "regret": float(
                (selected_weight_regret if forced_format is not None else mse_regret)[rr, rg].item()
            ),
            "mse_optimal_regret": float(mse_regret[rr, rg].item()),
            "identity_regret_min_pn": float(identity_regret[rr, rg].item()),
            "identity_residual": float(identity_residual[rr, rg].item()),
            "selected_weight_error": float(selected_error_region[rr, rg].item()),
            "selected_weight_mse_regret": float(selected_weight_regret[rr, rg].item()),
            "normalized_regret": float(
                ((selected_weight_regret if forced_format is not None else mse_regret)[rr, rg]
                 / oracle_region[rr, rg].clamp_min(1e-30)).item()
            ),
            "selected_format": "e0m3" if actual_id else "e2m1",
            "selected_format_mse": "e0m3" if mse_id else "e2m1",
            "selected_format_sensitivity": (
                "e0m3" if sensitivity_id else "e2m1" if sensitivity_id is not None else None
            ),
            "selector_disagreement": actual_id != mse_id,
            "selector_disagreement_block_count": int(selector_disagreement_region[rr, rg].item()),
            "sampled_region": len(sampled) < len(bounds),
            "total_regions_in_tensor": len(bounds),
        }
        if sensitivity_regret is not None:
            record.update(
                {
                    "sensitivity_oracle_error": float(sensitivity_oracle_region[rr, rg].item()),
                    "sensitivity_constrained_error": float(sensitivity_coarse_region[rr, rg].item()),
                    "sensitivity_regret": float(sensitivity_regret[rr, rg].item()),
                    "selected_sensitivity_error": float(sensitivity_selected_region[rr, rg].item()),
                    "mean_sensitivity_margin": float(
                        sensitivity_margin[r0:r1, g0:g1].mean().item()
                    ),
                }
            )
        records.append(record)
    return summary, records
