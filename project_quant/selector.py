"""MSE and calibration-output-aware coarse datatype selectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .candidates import CandidateBlocks
from .region import RegionShape, aggregate_regions, expand_region_values


@dataclass
class Selection:
    selected_ids: torch.Tensor
    mse_ids: torch.Tensor
    sensitivity_ids: torch.Tensor | None
    mse_region_e2: torch.Tensor
    mse_region_e0: torch.Tensor
    sensitivity_region_e2: torch.Tensor | None
    sensitivity_region_e0: torch.Tensor | None
    sensitivity_block_e2: torch.Tensor | None
    sensitivity_block_e0: torch.Tensor | None


def mse_selection(candidates: CandidateBlocks, shape: RegionShape) -> Selection:
    e2 = aggregate_regions(candidates.e2_errors, shape)
    e0 = aggregate_regions(candidates.e0_errors, shape)
    region_ids = e0 < e2  # ties retain E2, matching both pinned references.
    ids = expand_region_values(region_ids, rows=candidates.rows, groups=candidates.groups, shape=shape)
    return Selection(ids, ids, None, e2, e0, None, None, None, None)


def _padded_deltas(candidates: CandidateBlocks) -> tuple[torch.Tensor, torch.Tensor, int]:
    k64_blocks = math.ceil(candidates.k / 64)
    padded_k = k64_blocks * 64
    source = candidates.grouped_original.reshape(candidates.rows, -1)
    e2 = candidates.e2_dequant.reshape(candidates.rows, -1)
    e0 = candidates.e0_dequant.reshape(candidates.rows, -1)
    source = F.pad(source, (0, padded_k - source.shape[1]))
    e2 = F.pad(e2, (0, padded_k - e2.shape[1]))
    e0 = F.pad(e0, (0, padded_k - e0.shape[1]))
    return (source - e2).reshape(candidates.rows, k64_blocks, 64), (
        source - e0
    ).reshape(candidates.rows, k64_blocks, 64), k64_blocks


def sensitivity_scores(
    candidates: CandidateBlocks, hessian_k64: torch.Tensor, shape: RegionShape
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full-region and additive K16-diagonal calibration output SSE."""
    if shape.k16_groups != 4:
        raise ValueError("output-aware selection currently requires a K64 format region")
    delta_e2, delta_e0, k64_blocks = _padded_deltas(candidates)
    hessian = hessian_k64.to(device=delta_e2.device, dtype=torch.float32)
    if hessian.ndim != 3 or hessian.shape[1:] != (64, 64):
        raise ValueError("calibration_stats['hessian_k64'] must have shape [ceil(K/64),64,64]")
    if hessian.shape[0] < k64_blocks:
        hessian = F.pad(hessian, (0, 0, 0, 0, 0, k64_blocks - hessian.shape[0]))
    hessian = hessian[:k64_blocks]

    padded_rows = math.ceil(candidates.rows / shape.rows) * shape.rows
    d2 = F.pad(delta_e2, (0, 0, 0, 0, 0, padded_rows - candidates.rows))
    d0 = F.pad(delta_e0, (0, 0, 0, 0, 0, padded_rows - candidates.rows))
    d2 = d2.reshape(-1, shape.rows, k64_blocks, 64)
    d0 = d0.reshape(-1, shape.rows, k64_blocks, 64)
    region_e2 = torch.einsum("rnki,kij,rnkj->rk", d2, hessian, d2)
    region_e0 = torch.einsum("rnki,kij,rnkj->rk", d0, hessian, d0)

    # K16 block-diagonal scores provide an additive sensitivity-margin metric.
    block_e2 = torch.empty_like(candidates.e2_errors)
    block_e0 = torch.empty_like(candidates.e0_errors)
    for group in range(candidates.groups):
        k64 = group // 4
        offset = (group % 4) * 16
        valid = min(16, candidates.k - group * 16)
        h16 = hessian[k64, offset : offset + valid, offset : offset + valid]
        delta2 = delta_e2[:, k64, offset : offset + valid]
        delta0 = delta_e0[:, k64, offset : offset + valid]
        block_e2[:, group] = torch.einsum("ni,ij,nj->n", delta2, h16, delta2)
        block_e0[:, group] = torch.einsum("ni,ij,nj->n", delta0, h16, delta0)
    return region_e2, region_e0, block_e2, block_e0


def output_aware_selection(
    candidates: CandidateBlocks,
    shape: RegionShape,
    calibration_stats: dict,
) -> Selection:
    if "hessian_k64" not in calibration_stats:
        raise ValueError("output-aware selection requires calibration_stats['hessian_k64']")
    mse = mse_selection(candidates, shape)
    region_e2, region_e0, block_e2, block_e0 = sensitivity_scores(
        candidates, calibration_stats["hessian_k64"], shape
    )
    region_ids = region_e0 < region_e2
    ids = expand_region_values(region_ids, rows=candidates.rows, groups=candidates.groups, shape=shape)
    return Selection(
        selected_ids=ids,
        mse_ids=mse.mse_ids,
        sensitivity_ids=ids,
        mse_region_e2=mse.mse_region_e2,
        mse_region_e0=mse.mse_region_e0,
        sensitivity_region_e2=region_e2,
        sensitivity_region_e0=region_e0,
        sensitivity_block_e2=block_e2,
        sensitivity_block_e0=block_e0,
    )
