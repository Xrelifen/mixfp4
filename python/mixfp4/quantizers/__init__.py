"""Pluggable weight-quantisation methods for the mixed E2M1/E0M3 format.

Two axes, deliberately orthogonal:

  **format policy** -- which codebook a group uses (``mixed`` decides per group, ``e2m1`` pins
  everything to standard NVFP4, ``e0m3`` to sign-magnitude INT4).

  **method** -- how the codes and the scale are fitted once a codebook is chosen (``rtn``
  round-to-nearest, ``hqq`` half-quadratic proximal optimisation).

Keeping them separate is what makes the ablation meaningful: ``e2m1 + rtn`` is exactly ordinary
NVFP4, so any difference against ``mixed + rtn`` is attributable to the format choice alone, and
any difference against ``e2m1 + hqq`` to the optimiser alone.

Register a new method with :func:`register`; it needs to return codes, a per-group scale, a
per-group error, and the target it wants codes re-derived from when the scale is later rounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from ..codebook import E0M3_MAX, E2M1_MAX, FMT_E0M3, FMT_E2M1, GROUP_SIZE

#: Largest representable magnitude, indexed by format id.
CODEBOOK_MAX = {FMT_E2M1: E2M1_MAX, FMT_E0M3: E0M3_MAX}

FORMAT_POLICIES = {
    "mixed": (FMT_E2M1, FMT_E0M3),   # decide per group by error
    "e2m1": (FMT_E2M1,),             # standard NVFP4
    "nvfp4": (FMT_E2M1,),            # alias, for readable comparison tables
    "e0m3": (FMT_E0M3,),             # sign-magnitude INT4
}


@dataclass
class QuantConfig:
    """Everything that controls a quantisation run."""

    method: str = "rtn"
    format_policy: str = "mixed"
    group_size: int = GROUP_SIZE

    #: Exponent of the selection/error norm.  ``2`` is plain least squares; HQQ argues for p < 1,
    #: which tolerates a few large errors in exchange for making most errors tiny.
    p: float = 2.0

    #: Coarsen the per-group format decision to blocks of this many output columns / K groups, so
    #: the resulting checkpoint is also legal for the CUDA path (whose granule cannot go below one
    #: MMA atom's footprint).
    granule_n: int = 1
    granule_k: int = 1

    # --- HQQ-style optimiser ---
    iters: int = 20
    beta: float = 1e1
    kappa: float = 1.01
    #: Exponent of the shrinkage prox.  HQQ's default is 0.7.
    lp_norm: float = 0.7

    def candidates(self) -> tuple[int, ...]:
        if self.format_policy not in FORMAT_POLICIES:
            raise ValueError(f"unknown format_policy {self.format_policy!r}; "
                             f"choose from {sorted(FORMAT_POLICIES)}")
        return FORMAT_POLICIES[self.format_policy]


@dataclass
class FitResult:
    """What a method returns for one codebook, over every group of one weight matrix."""

    nibbles: torch.Tensor   # [N, G, group_size] uint8
    scale: torch.Tensor     # [N, G] float32, the continuous per-group scale
    error: torch.Tensor     # [N, G] float32, this fit's objective value against the true weights
    target: torch.Tensor    # [N, G, group_size] float32, what codes are derived from

    # ``target`` is not always the weights.  HQQ's whole mechanism is that it fits codes to a
    # *corrected* tensor ``W - W_e``; re-deriving codes from ``W`` after the scale is rounded to
    # UE4M3 would silently discard the optimisation.


class Method(Protocol):
    def __call__(self, grouped: torch.Tensor, fmt: int, cfg: QuantConfig) -> FitResult: ...


_REGISTRY: dict[str, Method] = {}


def register(name: str) -> Callable[[Method], Method]:
    def wrap(fn: Method) -> Method:
        _REGISTRY[name] = fn
        return fn
    return wrap


def get_method(name: str) -> Method:
    if name not in _REGISTRY:
        raise ValueError(f"unknown quantisation method {name!r}; "
                         f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def lp_error(residual: torch.Tensor, p: float) -> torch.Tensor:
    """Per-group ``sum |residual|^p``.  The comparison metric for format selection."""
    if p == 2.0:
        return residual.pow(2).sum(-1)
    return residual.abs().pow(p).sum(-1)


from . import hqq, rtn  # noqa: E402,F401  (import for the side effect of registering)
