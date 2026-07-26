"""Round-to-nearest fits.

``rtn`` is the obvious baseline: give each group the smallest scale that keeps its largest weight
representable, then round every weight to the nearest codebook entry.  It is what standard NVFP4
quantisation does.

``rtn_search`` additionally sweeps a small window of scale multipliers and keeps the best.  The
amax-derived scale is optimal only if the group's largest element is the one worth preserving; when
a group has one outlier, deliberately clipping it can lower total error a lot.  This is the same
idea as GemLite's ``window_size`` search in its MXFP weight quantiser.
"""

from __future__ import annotations

import torch

from ..codebook import dequantize_nibbles, quantize_nibbles
from . import CODEBOOK_MAX, FitResult, QuantConfig, lp_error, register

#: Multipliers tried by ``rtn_search``.  Below 1.0 clips the group's extremes; above 1.0 gives the
#: outlier more headroom at the cost of resolution everywhere else.
SEARCH_GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05)


def _fit_at_scale(grouped: torch.Tensor, scale: torch.Tensor, fmt_t: torch.Tensor):
    safe = scale.clamp(min=torch.finfo(scale.dtype).tiny).unsqueeze(-1)
    nibbles = quantize_nibbles(grouped / safe, fmt_t)
    recon = dequantize_nibbles(nibbles, fmt_t) * safe
    return nibbles, recon


@register("rtn")
def fit_rtn(grouped: torch.Tensor, fmt: int, cfg: QuantConfig) -> FitResult:
    fmt_t = torch.full((), fmt, dtype=torch.int32, device=grouped.device)
    scale = grouped.abs().amax(-1) / CODEBOOK_MAX[fmt]
    nibbles, recon = _fit_at_scale(grouped, scale, fmt_t)
    return FitResult(nibbles, scale, lp_error(grouped - recon, cfg.p), grouped)


@register("rtn_search")
def fit_rtn_search(grouped: torch.Tensor, fmt: int, cfg: QuantConfig) -> FitResult:
    fmt_t = torch.full((), fmt, dtype=torch.int32, device=grouped.device)
    base = grouped.abs().amax(-1) / CODEBOOK_MAX[fmt]

    best_err = None
    best_scale = base
    best_nibbles = None
    for multiplier in SEARCH_GRID:
        scale = base * multiplier
        nibbles, recon = _fit_at_scale(grouped, scale, fmt_t)
        err = lp_error(grouped - recon, cfg.p)
        if best_err is None:
            best_err, best_scale, best_nibbles = err, scale, nibbles
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, scale, best_scale)
            best_nibbles = torch.where(take.unsqueeze(-1), nibbles, best_nibbles)
    return FitResult(best_nibbles, best_scale, best_err, grouped)
