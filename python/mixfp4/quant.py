"""Weight quantiser for the mixed E2M1/E0M3 4-bit format.

Produces exactly what ``mixfp4.triton_kernels`` consumes:

  ``W_q``        ``[K // 2, N]`` uint8   -- nibbles packed over K, two per byte
  ``scales``     ``[K // 16, N]`` uint8  -- bit 7 = format tag, bits 0..6 = UE4M3 magnitude
  ``meta_scale`` scalar float32          -- global renormaliser, applied once after the K loop

The fitting itself is delegated to :mod:`mixfp4.quantizers`, which separates *which codebook* a
group uses from *how* its codes and scale are fitted.  This module owns the parts common to every
method: per-group format selection, the two-level scale, and packing.

The two-level scale is the standard NVFP4 recipe.  UE4M3 spans roughly 2^-9 to 448, so a single
global fp32 factor normalises the per-group scales into that window; the kernel undoes it once,
after the K loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook import (
    ELEMENTS_PER_SAMPLE,
    GROUP_SIZE,
    UE4M3_MAX,
    decode_ue4m3,
    dequantize_nibbles,
    encode_ue4m3,
    pack_nibbles_over_k,
    quantize_nibbles,
    unpack_nibbles_over_k,
)
from .quantizers import (  # noqa: F401
    E0M3_7,
    E2M1_6,
    Candidate,
    FitResult,
    QuantConfig,
    available_methods,
    get_method,
    lp_error,
)


@dataclass
class MixFP4Tensor:
    """A quantised weight matrix plus the metadata the kernels and the packer need."""

    W_q: torch.Tensor        # [K // 2, N] uint8
    scales: torch.Tensor     # [K // 16, N] uint8, bit 7 tagged
    meta_scale: float        # multiply the accumulator by this once, after the K loop
    shape: tuple[int, int]   # (N, K) of the original weight
    e0m3_fraction: float     # share of groups that chose E0M3, for reporting

    @property
    def N(self) -> int:
        return self.shape[0]

    @property
    def K(self) -> int:
        return self.shape[1]

    @property
    def is_uniform_e2m1(self) -> bool:
        """True when no group chose E0M3, i.e. this is an ordinary NVFP4 tensor."""
        return self.e0m3_fraction == 0.0

    def to(self, device) -> "MixFP4Tensor":
        return MixFP4Tensor(self.W_q.to(device), self.scales.to(device), self.meta_scale,
                            self.shape, self.e0m3_fraction)


def _coarsen(errors: list[torch.Tensor], granule_n: int, granule_k: int) -> torch.Tensor:
    """Decide the format over a block of groups by pooling their errors, then broadcast back.

    The Triton kernels read one flag per scale byte and so support the finest possible granule
    (one output column by 16 K elements).  The CUDA path cannot: its format lives in the
    instruction encoding, so a granule is at best one MMA atom's footprint.  Pooling here lets a
    single checkpoint serve both backends.
    """
    stacked = torch.stack(errors)                     # [F, N, G]
    if granule_n == 1 and granule_k == 1:
        return stacked.argmin(0).to(torch.int32)
    _, n, kg = stacked.shape
    assert n % granule_n == 0 and kg % granule_k == 0, (
        f"granule ({granule_n}, {granule_k}) must divide the group grid ({n}, {kg})")
    blocks = stacked.reshape(-1, n // granule_n, granule_n, kg // granule_k, granule_k)
    winner = blocks.sum((2, 4)).argmin(0)             # [N/gn, G/gk]
    return (winner.unsqueeze(1).unsqueeze(3)
            .expand(n // granule_n, granule_n, kg // granule_k, granule_k)
            .reshape(n, kg).to(torch.int32))


def quantize_mixfp4(weight: torch.Tensor,
                    *,
                    config: QuantConfig | None = None,
                    method: str | None = None,
                    format_policy: str | None = None,
                    force_format: str | None = None,
                    granule_n: int | None = None,
                    granule_k: int | None = None,
                    **method_kwargs) -> MixFP4Tensor:
    """Quantise an ``[N, K]`` weight matrix (``nn.Linear.weight`` layout) to mixed FP4.

    Args:
        weight: ``[out_features, in_features]``, any float dtype.
        config: a :class:`~mixfp4.quantizers.QuantConfig`.  Any of the keyword arguments below
            override fields on it.
        method: ``"rtn"``, ``"rtn_search"``, ``"hqq"``, or anything else registered.
        format_policy: ``"mixed"`` (choose per group), ``"e2m1"``/``"nvfp4"``, or ``"e0m3"``.
        force_format: legacy alias for ``format_policy``; also accepts ``"random"``, which tags
            groups at random.  Uniform tagging is correct by accident under a permuted layout, so
            the kernel tests need a genuinely mixed pattern to exercise the per-group path.
        granule_n, granule_k: coarsen the format decision to blocks of this many output columns
            and 16-element K groups.  Defaults to the finest granule the format allows.
    """
    assert weight.dim() == 2, f"expected a 2-D weight, got shape {tuple(weight.shape)}"
    cfg = config or QuantConfig()
    randomise = force_format == "random"
    if method is not None:
        cfg.method = method
    if format_policy is not None:
        cfg.format_policy = format_policy
    elif force_format is not None and not randomise:
        cfg.format_policy = force_format
    if granule_n is not None:
        cfg.granule_n = granule_n
    if granule_k is not None:
        cfg.granule_k = granule_k
    for key, value in method_kwargs.items():
        setattr(cfg, key, value)

    fit = _fit_and_select(weight, cfg, randomise=randomise)
    n, k = weight.shape
    packed = pack_nibbles_over_k(fit.nibbles.reshape(n, k))
    scale_bytes = (fit.scale_codes | (fit.flags.to(torch.uint8) << 7)).t().contiguous()

    return MixFP4Tensor(
        W_q=packed,
        scales=scale_bytes,
        meta_scale=1.0 / fit.meta_norm,
        shape=(n, k),
        e0m3_fraction=fit.flags.float().mean().item(),
    )


@dataclass
class _Selected:
    """The result of fitting and selecting, before it is either packed or dequantised."""

    nibbles: torch.Tensor      # [N, G, group_size] uint8
    flags: torch.Tensor        # [N, G] int32, the chosen codebook per group
    scale_codes: torch.Tensor  # [N, G] uint8, UE4M3 with bit 7 clear
    effective: torch.Tensor    # [N, G] float32, decoded scale / meta_norm
    meta_norm: float


def _fit_and_select(x: torch.Tensor, cfg: QuantConfig, *, randomise: bool = False) -> _Selected:
    """Run every candidate over ``x`` and keep the best per group.

    Shared by the weight path (which packs the result) and the activation path (which dequantises
    it immediately).  Both need identical arithmetic, including the UE4M3 rounding of the scale --
    simulating NVFP4 without that rounding would flatter it.
    """
    n, k = x.shape
    assert k % cfg.group_size == 0, (
        f"K={k} must be a multiple of the scale group size {cfg.group_size}")

    grouped = x.detach().float().reshape(n, k // cfg.group_size, cfg.group_size)
    fit_method = get_method(cfg.method)

    candidates = (E2M1_6, E0M3_7) if randomise else cfg.candidates()
    fits: list[FitResult] = [fit_method(grouped, cand, cfg) for cand in candidates]

    # ``index`` selects which fit each group takes; ``flags`` is the format id that fit represents,
    # which is what gets written to bit 7 of the scale byte.
    if randomise:
        index = torch.randint(0, len(fits), fits[0].scale.shape,
                              device=grouped.device, dtype=torch.long)
    elif len(fits) == 1:
        index = torch.zeros(fits[0].scale.shape, device=grouped.device, dtype=torch.long)
    else:
        # Selection uses each fit's own objective on the *continuous* scale.  The stored scale is
        # then rounded to UE4M3, a perturbation small enough not to reorder the choice.
        index = _coarsen([f.error for f in fits], cfg.granule_n, cfg.granule_k).long()

    flags = torch.tensor([c.fmt for c in candidates],
                         device=grouped.device, dtype=torch.int32)[index]

    def pick(field: str) -> torch.Tensor:
        values = torch.stack([getattr(f, field) for f in fits])
        idx = index
        while idx.dim() < values.dim() - 1:
            idx = idx.unsqueeze(-1)
        return values.gather(0, idx.expand(values.shape[1:]).unsqueeze(0)).squeeze(0)

    ideal = pick("scale")
    target = pick("target")

    # Normalise the group scales into UE4M3's representable window.  An all-zero tensor has no
    # scale to speak of; fall back to 1.0 so the reciprocal stays finite.
    gmax = ideal.max()
    meta_norm = (UE4M3_MAX / gmax).item() if gmax > 0 else 1.0

    scale_codes = encode_ue4m3(ideal * meta_norm)
    # Re-derive codes against the scale that will actually be applied -- the rounded UE4M3 value
    # times the global meta_scale -- and against the method's own target, which for HQQ is the
    # slack-corrected tensor rather than the tensor itself.  Refitting from the raw values here
    # would silently discard the optimisation; fitting against the unrounded scale would bake in
    # an error nothing downstream can undo.
    effective = decode_ue4m3(scale_codes) / meta_norm
    safe = effective.clamp(min=torch.finfo(effective.dtype).tiny).unsqueeze(-1)
    nibbles = quantize_nibbles(target / safe, flags.unsqueeze(-1))
    nibbles = torch.where((effective == 0).unsqueeze(-1), torch.zeros_like(nibbles), nibbles)

    return _Selected(nibbles, flags, scale_codes, effective, meta_norm)


def simulate_quantization(x: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quantise and immediately dequantise a 2-D tensor, skipping the packing step.

    The activation path.  Activations are quantised per forward pass and thrown away, so packing
    them into nibbles and unpacking again is pure waste -- but every other step, including the
    UE4M3 scale rounding, has to match the weight path exactly for the simulation to be honest.
    """
    fit = _fit_and_select(x, cfg)
    values = dequantize_nibbles(fit.nibbles, fit.flags.unsqueeze(-1))
    return (values * fit.effective.unsqueeze(-1)).reshape(x.shape)


def dequantize_mixfp4(q: MixFP4Tensor, dtype=torch.float32) -> torch.Tensor:
    """Reconstruct the ``[N, K]`` weight matrix.  The reference the kernels are checked against."""
    nibbles = unpack_nibbles_over_k(q.W_q)                      # [N, K]
    scale_bytes = q.scales.t()                                  # [N, K // 16]
    scale = decode_ue4m3(scale_bytes) * q.meta_scale
    flags = (scale_bytes >> 7).to(torch.int32)
    values = dequantize_nibbles(nibbles.reshape(q.N, -1, GROUP_SIZE), flags.unsqueeze(-1))
    return (values * scale.unsqueeze(-1)).reshape(q.N, q.K).to(dtype)


def quantize_dequantize(weight: torch.Tensor, **kwargs) -> torch.Tensor:
    """Simulated quantisation: round-trip a weight through the format, in its original dtype.

    This is what the perplexity harness uses.  It isolates the question "how much accuracy does
    this format and method cost" from "which kernel runs the matmul", and lets both be measured
    independently.
    """
    q = quantize_mixfp4(weight, **kwargs)
    return dequantize_mixfp4(q, dtype=weight.dtype).to(weight.device)


def quantization_error(weight: torch.Tensor, *, p: float = 2.0, **kwargs) -> dict[str, float]:
    """Relative error of each format policy on one weight matrix, for side-by-side comparison."""
    out: dict[str, float] = {}
    reference = weight.float()
    denom = max(reference.norm().item(), 1e-30)
    for name, policy in (("e2m1", "e2m1"), ("e0m3", "e0m3"), ("mixed", "mixed")):
        q = quantize_mixfp4(weight, format_policy=policy, **kwargs)
        recon = dequantize_mixfp4(q).to(weight.device)
        out[name] = (reference - recon).norm().item() / denom
        if policy == "mixed":
            out["e0m3_fraction"] = q.e0m3_fraction
    return out
