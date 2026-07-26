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


@dataclass(frozen=True)
class Candidate:
    """One way to encode a group: a codebook, and the value its amax is mapped onto.

    ``cmax`` is normally the codebook's largest magnitude, so the group's peak weight lands on the
    top entry.  Mapping onto a *smaller* entry deliberately gives up the top of the codebook in
    exchange for a more uniform spacing over the values that remain -- which is the whole idea
    behind Four Over Six (arXiv:2512.02010).
    """

    fmt: int          # written to bit 7 of the scale byte
    cmax: float       # the codebook value this group's amax is mapped onto
    name: str


#: Standard NVFP4: amax maps to E2M1's top entry, 6.
E2M1_6 = Candidate(FMT_E2M1, 6.0, "e2m1@6")
#: Four Over Six: amax maps to 4 instead, giving up +-6 and +-5.  E2M1's steps are 0.5 below 2,
#: 1 up to 4, then 2 up to 6, so the top of the range is where its resolution is worst; a value
#: that should land near 5 has nowhere to go.  Capping at 4 removes that gap, at the cost of one
#: representable level.
E2M1_4 = Candidate(FMT_E2M1, 4.0, "e2m1@4")
#: mixfp4's alternative: a uniformly spaced codebook, amax onto 7.
E0M3_7 = Candidate(FMT_E0M3, 7.0, "e0m3@7")

FORMAT_POLICIES = {
    # --- single candidate: no per-group choice at all ---
    "e2m1": (E2M1_6,),                    # standard NVFP4
    "nvfp4": (E2M1_6,),                   # alias, for readable comparison tables
    "e0m3": (E0M3_7,),                    # sign-magnitude INT4 everywhere
    "e2m1-4": (E2M1_4,),                  # every block capped at 4 -- the paper's Table 3 control

    # --- per-group choice ---
    "mixed": (E2M1_6, E0M3_7),            # mixfp4: choose the codebook
    "nvfp4-46": (E2M1_6, E2M1_4),         # Four Over Six: choose the cap, one codebook
    "mixed-46": (E2M1_6, E2M1_4, E0M3_7),  # both axes at once
}

# Note that Four Over Six needs no format-level support whatsoever.  The per-group scale is stored
# explicitly as a UE4M3 byte, so a block capped at 4 simply stores amax/4; nothing downstream needs
# to know which cap was used, and the dequantiser is unchanged.  It composes freely with the
# codebook tag in bit 7, which is why "mixed-46" is a legal three-way choice rather than a conflict.
#
# It also sidesteps the paper's Section 3.1 caveat.  There, the tensor-wide scale is derived as
# max(|X|) / (M_FP4 * M_FP8) assuming a single M_FP4, so a block wanting amax/4 would need a block
# scale 1.5x larger than the formula allows -- 448 * 6/4 = 672 overflows E4M3, forcing M_FP8 to be
# dropped from 448 to 256.  This implementation instead takes the global normaliser from the max of
# the per-group scales actually selected, so the largest lands on 448 by construction whatever cap
# produced it, and no headroom is wasted.


@dataclass
class QuantConfig:
    """Everything that controls a quantisation run."""

    method: str = "rtn"
    format_policy: str = "mixed"
    group_size: int = GROUP_SIZE

    #: Exponent of the selection/error norm.  ``2`` is plain least squares (Four Over Six's
    #: preferred rule); ``1`` is MAE; ``inf`` is worst-case error; HQQ argues for p < 1, which
    #: tolerates a few large errors in exchange for making most errors tiny.
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

    def candidates(self) -> tuple["Candidate", ...]:
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
    def __call__(self, grouped: torch.Tensor, cand: "Candidate",
                 cfg: QuantConfig) -> FitResult: ...


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
    """Per-group ``sum |residual|^p``, the metric candidates are compared on.

    ``p = 2`` is mean squared error, ``p = 1`` is mean absolute error, and ``p = inf`` is the
    largest single error in the group -- the three selection rules Four Over Six ablates in its
    Table 4, where it reports MSE working best.
    """
    if p == float("inf"):
        return residual.abs().amax(-1)
    if p == 2.0:
        return residual.pow(2).sum(-1)
    return residual.abs().pow(p).sum(-1)


from . import hqq, rtn  # noqa: E402,F401  (import for the side effect of registering)
