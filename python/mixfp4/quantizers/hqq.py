"""HQQ-style half-quadratic quantisation, adapted to a symmetric 4-bit codebook.

HQQ (Badri & Shaji, Mobius Labs -- the method GemLite was built to serve) minimises

    || W - dequant(quant(W)) ||_p    with p < 1

instead of the usual least squares.  A sub-1 exponent tolerates a few large errors in exchange for
making most errors very small, which suits weight matrices where a handful of outliers otherwise
drag the scale up and coarsen every other value in the group.  The minimisation is done by
half-quadratic splitting: introduce a slack variable for the error, alternate a closed-form
proximal (shrinkage) step on it with a closed-form update of the quantisation meta-parameters, and
anneal the coupling ``beta`` upward each iteration.

**The adaptation.**  Stock HQQ optimises the *zero-point* of an asymmetric affine quantiser.  Both
codebooks here are symmetric sign-magnitude, so there is no zero-point to move -- the only free
meta-parameter is the per-group scale.  The zero-point update is therefore replaced by a
least-squares scale update, which for a fixed set of codes has the closed form

    s* = <W - W_e, C> / <C, C>

where ``C`` holds the codebook values of the current codes.  Everything else -- the shrinkage
operator, the annealing schedule, the defaults ``p = 0.7``, ``beta = 10``, ``kappa = 1.01`` -- is
HQQ's.  Calling this "HQQ" outright would overclaim; it is HQQ's machinery on a different
meta-parameter.

Because the iteration is not monotone in the true error, the best iterate is kept rather than the
last.
"""

from __future__ import annotations

import torch

from ..codebook import dequantize_nibbles, quantize_nibbles
from . import CODEBOOK_MAX, FitResult, QuantConfig, lp_error, register


def shrink_lp(x: torch.Tensor, beta: float, p: float) -> torch.Tensor:
    """Generalised soft-thresholding: the proximal operator of ``|.|^p / beta``.

    For ``p < 1`` the threshold ``|x|^(p-1) / beta`` grows without bound as ``|x| -> 0``, which is
    exactly the point -- small errors are annihilated, large ones survive nearly untouched.  The
    magnitude is floored before exponentiation so that the singularity at zero yields 0 rather
    than NaN.
    """
    magnitude = x.abs()
    if p == 1.0:
        threshold = 1.0 / beta
    else:
        floor = torch.finfo(x.dtype).tiny
        threshold = (1.0 / beta) * magnitude.clamp(min=floor).pow(p - 1.0)
    return x.sign() * (magnitude - threshold).clamp(min=0.0)


@register("hqq")
def fit_hqq(grouped: torch.Tensor, fmt: int, cfg: QuantConfig) -> FitResult:
    fmt_t = torch.full((), fmt, dtype=torch.int32, device=grouped.device)
    tiny = torch.finfo(grouped.dtype).tiny

    scale = (grouped.abs().amax(-1) / CODEBOOK_MAX[fmt]).clamp(min=tiny)
    slack = torch.zeros_like(grouped)          # W_e, the error slack variable
    beta = cfg.beta

    best_error = None
    best_nibbles = None
    best_scale = scale
    best_target = grouped

    for _ in range(max(1, cfg.iters)):
        # 1. Fit codes to the corrected target rather than to W itself.  This is the step that
        #    makes HQQ more than a scale search: the slack absorbs the outliers, so the codes are
        #    chosen for the bulk of the distribution.
        target = grouped - slack
        codes = quantize_nibbles(target / scale.unsqueeze(-1), fmt_t)
        values = dequantize_nibbles(codes, fmt_t)

        # 2. Closed-form least-squares scale for those codes.  Groups whose codes are all zero
        #    have no scale information; leave theirs alone.
        denom = (values * values).sum(-1)
        numer = (target * values).sum(-1)
        scale = torch.where(denom > 0, numer / denom.clamp(min=tiny), scale).clamp(min=tiny)

        # 3. Proximal step, measured against the true weights.
        recon = values * scale.unsqueeze(-1)
        residual = grouped - recon
        slack = shrink_lp(residual, beta, cfg.lp_norm)
        beta *= cfg.kappa

        # 4. Track the best iterate; the objective is not monotone under annealing.
        error = lp_error(residual, cfg.p)
        if best_error is None:
            best_error, best_nibbles, best_scale, best_target = error, codes, scale, target
        else:
            take = error < best_error
            best_error = torch.where(take, error, best_error)
            best_nibbles = torch.where(take.unsqueeze(-1), codes, best_nibbles)
            best_scale = torch.where(take, scale, best_scale)
            best_target = torch.where(take.unsqueeze(-1), target, best_target)

    return FitResult(best_nibbles, best_scale, best_error, best_target)
