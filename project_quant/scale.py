"""K16 candidate-specific scale construction and 4Over6-style composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .codebook import quantize_e0m3, quantize_e2m1


STANDARD_E4M3_MAX = 448.0
FOUROVERSIX_GLOBAL_E4M3_MAX = 256.0
E4M3_MIN_NORMAL_OR_SUBNORMAL = 2**-9
ScaleRule = Literal["standard", "four_over_six"]


@dataclass
class ScaledCandidate:
    codes: torch.Tensor
    dequant: torch.Tensor
    scales: torch.Tensor
    errors: torch.Tensor
    expansion_ids: torch.Tensor


def compute_global_scale(grouped: torch.Tensor, scale_rule: ScaleRule) -> torch.Tensor:
    """Return the tensor-level decode scale shared by both datatype candidates."""
    amax = grouped.abs().amax()
    denominator = 6.0 * (
        STANDARD_E4M3_MAX if scale_rule == "standard" else FOUROVERSIX_GLOBAL_E4M3_MAX
    )
    # Upstream emits NaNs on an all-zero tensor; the zero-safe value below is
    # mathematically equivalent after quantization and explicitly tested.
    return torch.where(amax > 0, amax / denominator, torch.ones_like(amax))


def _e4m3(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float8_e4m3fn).to(torch.float32)


def _one_scale_candidate(
    grouped: torch.Tensor,
    scaled: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    qmax: float,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
    expansion: float,
    standard_clamp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ideal = scaled.abs().amax(dim=-1, keepdim=True) / qmax * expansion
    if standard_clamp:
        ideal = ideal.clamp(min=E4M3_MIN_NORMAL_OR_SUBNORMAL, max=STANDARD_E4M3_MAX)
    scale = _e4m3(ideal)
    normalized = torch.where(scale != 0, scaled / scale, torch.zeros_like(scaled))
    codes = quantizer(normalized)
    dequant = codes * scale * global_scale
    errors = ((dequant - grouped) ** 2).sum(dim=-1)
    return codes, dequant, scale.squeeze(-1), errors


def _four_over_six_scale_candidate(
    grouped: torch.Tensor,
    *,
    qmax: float,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
    expansion: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct one candidate with the pinned reference's arithmetic order.

    The operation order matters for values exactly on E2M1 rounding boundaries:
    ``(x / global) / scale`` is algebraically equivalent to the canonical
    ``x * (1 / (decode_scale * scale))`` but not floating-point equivalent on
    large real tensors.  This mirrors FourOverSix's PyTorch reference while
    allowing the project E0 candidate to substitute only its explicit qmax and
    codebook.
    """

    amax = grouped.abs().amax().float()
    denominator = torch.tensor(
        6.0 * FOUROVERSIX_GLOBAL_E4M3_MAX,
        dtype=amax.dtype,
        device=amax.device,
    )
    if bool(amax == 0):
        ideal = torch.zeros_like(grouped[..., 0])
    else:
        encode_scale = denominator / amax
        ideal = grouped.abs().amax(dim=-1) / qmax * encode_scale
    ideal = ideal * expansion
    scale = _e4m3(ideal)
    decode_scale = torch.where(amax > 0, 1.0 / (denominator / amax), torch.zeros_like(amax))
    normalized = torch.where(
        scale[..., None] != 0,
        grouped * (1.0 / (decode_scale * scale[..., None])),
        torch.zeros_like(grouped),
    )
    codes = quantizer(normalized)
    dequant = codes * scale[..., None] * amax / denominator
    errors = ((dequant - grouped) ** 2).sum(dim=-1)
    return codes, dequant, scale, errors


def _canonical_four_over_six_e2_candidate(
    grouped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the pinned FourOverSix E2 path verbatim.

    CUDA's float8 conversion and scalar-expression lowering can differ at an
    exact rounding boundary even when an algebraically equivalent local
    expression is used.  Composition reduction is a hard invariant, so the E2
    candidate deliberately calls the pinned PyTorch reference implementation.
    The project-local implementation remains in use for the symmetric E0
    extension, whose semantics are necessarily project-defined.
    """
    from fouroversix.quantize.pytorch.reference import (
        nvfp4_fouroversix_block_scaled_quantization,
        quantize_to_nvfp4,
    )
    from fouroversix.utils import RoundStyle, ScaleRule as FourOverSixScaleRule

    blocks = grouped.reshape(-1, grouped.shape[-1]).float()
    amax = blocks.abs().amax().float()
    codes, selected_scales = nvfp4_fouroversix_block_scaled_quantization(
        blocks,
        amax,
        scale_rule=FourOverSixScaleRule.mse,
        round_style=RoundStyle.nearest,
    )
    _, base_scales = quantize_to_nvfp4(
        blocks,
        amax,
        scale_rule=FourOverSixScaleRule.mse,
        round_style=RoundStyle.nearest,
    )
    _, expanded_scales = quantize_to_nvfp4(
        blocks,
        amax,
        scale_rule=FourOverSixScaleRule.mse,
        round_style=RoundStyle.nearest,
        scale_expansion_factor=1.5,
    )
    codes = codes.reshape_as(grouped)
    scales = selected_scales.reshape(grouped.shape[:-1]).float()
    dequant = (
        codes.to(amax.dtype)
        * scales[..., None].to(amax.dtype)
        * amax
        / torch.tensor(
            6.0 * FOUROVERSIX_GLOBAL_E4M3_MAX,
            dtype=amax.dtype,
            device=amax.device,
        )
    )
    errors = ((dequant - grouped) ** 2).sum(dim=-1)
    expanded = (
        (selected_scales.reshape(-1) == expanded_scales.reshape(-1))
        & (selected_scales.reshape(-1) != base_scales.reshape(-1))
    ).reshape(grouped.shape[:-1])
    return codes, dequant, scales, errors, expanded.to(torch.uint8)


def build_scaled_candidate(
    grouped: torch.Tensor,
    *,
    format_name: Literal["e2m1", "e0m3"],
    global_scale: torch.Tensor,
    scale_rule: ScaleRule,
) -> ScaledCandidate:
    """Build one independently scaled datatype candidate per K16 block.

    ``four_over_six`` applies the published 1.5 scale expansion symmetrically
    inside each datatype.  E2 uses qmax=6 and therefore reduces exactly to the
    canonical FourOverSix PyTorch reference.  E0 uses qmax=7 with the same
    expansion ratio; this is a project composition, not a published baseline.
    """
    if format_name == "e2m1":
        qmax, quantizer = 6.0, quantize_e2m1
    elif format_name == "e0m3":
        qmax, quantizer = 7.0, quantize_e0m3
    else:  # pragma: no cover - the Literal and public callers constrain this.
        raise ValueError(f"unknown format {format_name!r}")
    scaled = grouped / global_scale
    if scale_rule == "standard":
        codes, dequant, scales, errors = _one_scale_candidate(
            grouped,
            scaled,
            global_scale,
            qmax=qmax,
            quantizer=quantizer,
            expansion=1.0,
            standard_clamp=True,
        )
        expansion_ids = torch.zeros_like(errors, dtype=torch.uint8)
        return ScaledCandidate(codes, dequant, scales, errors, expansion_ids)
    if scale_rule != "four_over_six":
        raise ValueError(f"unknown scale_rule={scale_rule!r}")

    if format_name == "e2m1":
        codes, dequant, scales, errors, expansion_ids = _canonical_four_over_six_e2_candidate(
            grouped
        )
        return ScaledCandidate(codes, dequant, scales, errors, expansion_ids)

    base = _four_over_six_scale_candidate(
        grouped,
        qmax=qmax,
        quantizer=quantizer,
        expansion=1.0,
    )
    expanded = _four_over_six_scale_candidate(
        grouped,
        qmax=qmax,
        quantizer=quantizer,
        expansion=1.5,
    )
    choose_expanded = expanded[3] < base[3]
    choose_values = choose_expanded[..., None]
    codes = torch.where(choose_values, expanded[0], base[0])
    dequant = torch.where(choose_values, expanded[1], base[1])
    scales = torch.where(choose_expanded, expanded[2], base[2])
    errors = torch.where(choose_expanded, expanded[3], base[3])
    return ScaledCandidate(codes, dequant, scales, errors, choose_expanded.to(torch.uint8))
