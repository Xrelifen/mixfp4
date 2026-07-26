"""The two 4-bit codebooks, the UE4M3 scale format, and the bit-7 format tag.

This is the host-side mirror of what the Triton kernels do in registers, and it is deliberately
written to agree with the CUDA path's conventions exactly:

  * the codebooks match ``src/mixed_nvfp4_gemm.cu`` (``kE2m1Magnitudes``) and the independent
    reference in ``tests/mma_intrinsics/expected_value.py``;
  * the format tag lives in bit 7 of the UE4M3 scale byte, the same channel as
    ``MIXFP4_FLAG_MASK`` in ``src/collective/mma_sm120_mixed.hpp``.

Both codebooks index the *same* 4-bit nibble; only the decode differs.  E2M1 is log-spaced with a
2-bit exponent, E0M3 is sign-magnitude INT4 -- equally spaced.  Which one a group of weights
prefers is a property of that group's distribution, which is the whole point of mixing them.

Why bit 7 is free
-----------------
A UE4M3 scale is unsigned: 4 exponent bits and 3 mantissa bits, stored in a byte whose top bit is
architecturally unused, and which the tensor core masks off before decoding.  The CUDA path
verifies this on real hardware (``check_bit7_invariant`` in tests/mma_intrinsics/run_tests.sh).
So the tag costs zero extra storage and zero extra bandwidth.

Note this is specific to NVFP4/UE4M3.  MXFP4's UE8M0 scale uses all eight bits for the exponent,
so there is no spare bit and this tagging scheme does not carry over -- see docs/triton_backend.md.
"""

from __future__ import annotations

import torch

# --- the two codebooks, index 0..15 = (sign << 3) | magnitude ------------------------------------

E2M1_CODEBOOK = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

E0M3_CODEBOOK = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                 -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]

#: Largest magnitude each codebook can represent.  A group's ideal scale is ``amax / CODEBOOK_MAX``.
E2M1_MAX = 6.0
E0M3_MAX = 7.0

#: Rounding thresholds for E2M1 magnitudes -- the midpoints of ``E2M1_CODEBOOK[:8]``.
E2M1_THRESHOLDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

#: Format ids, and the offset each one contributes to the combined lookup table.
FMT_E2M1 = 0
FMT_E0M3 = 1

#: Bit 7 of a scale byte selects the codebook.  Same value as ``MIXFP4_FLAG_MASK``.
FLAG_MASK = 0x80
#: The remaining seven bits are the UE4M3 magnitude, exactly what the tensor core decodes.
CODE_MASK = 0x7F

#: NVFP4 scale-group length along K.  UE4M3 scales, one per 16 elements.
GROUP_SIZE = 16
#: Nibbles packed per byte.
ELEMENTS_PER_SAMPLE = 2
#: Largest finite value of the 7-bit UE4M3 field, used to normalise the scale range.
UE4M3_MAX = 448.0

_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def combined_lut(device, dtype=torch.float32) -> torch.Tensor:
    """The 32-entry table the kernels gather from: ``[E2M1_CODEBOOK, E0M3_CODEBOOK]``.

    Concatenating the two codebooks is what makes runtime format selection free.  A group's
    format flag is not a branch and not a predicate -- it is ``16 * fmt`` added to the nibble
    before the gather.  Selecting a codebook and selecting an entry become one operation.

    Contrast the CUDA path, where the format is two bits of the *instruction* encoding: there,
    choosing per granule means choosing among instructions, which forced a whole specialised
    dispatch tree (see docs/mixed_nvfp4_report.md).  Software decode pays a constant cost per
    element and gets arbitrary granularity for free.
    """
    key = (torch.device(device), dtype)
    cached = _LUT_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(E2M1_CODEBOOK + E0M3_CODEBOOK, dtype=dtype, device=device)
        _LUT_CACHE[key] = cached
    return cached


# --- UE4M3 scale codec ---------------------------------------------------------------------------


def encode_ue4m3(values: torch.Tensor) -> torch.Tensor:
    """Round non-negative floats to UE4M3 and return the raw bytes, bit 7 clear.

    ``float8_e4m3fn`` has the same 4-exponent/3-mantissa field as UE4M3 and the same bias of 7, so
    a positive ``e4m3fn`` value *is* a UE4M3 code with the sign bit clear.  Clamping first keeps
    the cast away from ``e4m3fn``'s NaN encoding, which is what an out-of-range value would
    otherwise become.
    """
    clamped = values.clamp(min=0.0, max=UE4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def decode_ue4m3(codes: torch.Tensor) -> torch.Tensor:
    """Decode raw scale bytes, masking off the format tag exactly as the tensor core does."""
    return (codes & CODE_MASK).view(torch.float8_e4m3fn).float()


def split_scale_bytes(scale_bytes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed scale bytes into ``(decoded_scale, format_flag)``."""
    return decode_ue4m3(scale_bytes), (scale_bytes >> 7).to(torch.int32)


# --- nibble quantisers ---------------------------------------------------------------------------


def quantize_nibbles(normalized: torch.Tensor, fmt: torch.Tensor) -> torch.Tensor:
    """Round already-scaled weights to 4-bit codes under a per-element choice of codebook.

    ``normalized`` is ``w / scale``; ``fmt`` broadcasts against it and selects E0M3 where true.
    """
    magnitude = normalized.abs()
    thresholds = torch.tensor(E2M1_THRESHOLDS, dtype=magnitude.dtype, device=magnitude.device)
    mag_e2m1 = torch.bucketize(magnitude, thresholds)
    mag_e0m3 = magnitude.round().clamp_(0, 7).to(mag_e2m1.dtype)
    mag = torch.where(fmt.to(torch.bool), mag_e0m3, mag_e2m1)
    return (mag | (normalized < 0).to(mag.dtype) * 8).to(torch.uint8)


def dequantize_nibbles(nibbles: torch.Tensor, fmt: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_nibbles`, before the scale is reapplied."""
    lut = combined_lut(nibbles.device)
    index = nibbles.to(torch.long) + 16 * fmt.to(torch.long).expand_as(nibbles)
    return lut[index]


# --- nibble packing ------------------------------------------------------------------------------


def pack_nibbles_over_k(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack ``[N, K]`` nibbles into the ``[K // 2, N]`` uint8 layout the kernels read.

    Two consecutive K elements share a byte, the even one in the low nibble -- the same
    little-endian-within-the-word convention GemLite's ``pack_weights_over_cols`` uses, so a
    weight tensor packed by either project is readable by the other.
    """
    n, k = nibbles.shape
    assert k % ELEMENTS_PER_SAMPLE == 0, f"K={k} must be even to pack two nibbles per byte"
    pair = nibbles.reshape(n, k // 2, 2)
    packed = pair[..., 0] | (pair[..., 1] << 4)
    return packed.to(torch.uint8).t().contiguous()


def unpack_nibbles_over_k(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles_over_k`: ``[K // 2, N]`` uint8 -> ``[N, K]`` nibbles."""
    k_half, n = packed.shape
    by_n = packed.t()
    lo, hi = by_n & 0xF, (by_n >> 4) & 0xF
    return torch.stack((lo, hi), dim=-1).reshape(n, k_half * 2)
