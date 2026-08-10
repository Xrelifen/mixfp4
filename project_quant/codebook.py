"""Numerical E2M1 and project E0M3 reference codebooks."""

from __future__ import annotations

import torch


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0M3_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """NVFP4-RaZeR-compatible E2M1 RTN, saturated to ±6.

    The upstream implementation rounds the private mantissa with
    ``floor(abs(x) + 0.5)`` rather than PyTorch's ties-to-even ``round``.  The
    distinction is preserved here for exact baseline regression.
    """
    magnitude = x.abs()
    nonzero_for_log = magnitude + (magnitude == 0).to(magnitude.dtype)
    private_exp = torch.floor(torch.log2(nonzero_for_log)).clamp(min=0)
    mantissa = magnitude / torch.pow(2.0, private_exp) * 2.0
    mantissa_q = torch.floor(mantissa + 0.5)
    quantized = mantissa_q * torch.pow(2.0, private_exp) / 2.0
    return x.sign() * quantized.clamp(max=6.0)


def quantize_e2m1_fouroversix_reference(x: torch.Tensor) -> torch.Tensor:
    """Exact nearest-rounding arithmetic used by pinned FourOverSix."""
    magnitude = x.abs()
    step1 = torch.round(2 * magnitude) / 2
    step2 = torch.round(magnitude)
    step3 = 2 * torch.round(magnitude / 2)
    mask1 = magnitude < 2
    mask2 = magnitude < 4
    return x.sign() * (
        step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)
    )


def quantize_e0m3(x: torch.Tensor) -> torch.Tensor:
    """Project E0M3: sign magnitude 0, ±1, …, ±7 (never −8)."""
    return x.sign() * x.abs().round().clamp(max=7.0)


def numerical_levels(quantizer, *, minimum: float = -10, maximum: float = 10) -> list[float]:
    probe = torch.linspace(minimum, maximum, 80001, dtype=torch.float32)
    return sorted(set(float(value) for value in quantizer(probe).abs().tolist()))
