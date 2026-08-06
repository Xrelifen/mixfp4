"""Independent CPU references for the NativeMix SM120 study.

This module intentionally shares no implementation with the GPU decoder,
legacy probe helper, CUTLASS, or PyTorch.  The E0M3 mapping is a *candidate*
mapping until an exhaustive SM120 observation agrees with every nibble.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E0M3_MAGNITUDES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def _validated_nibble(nibble: int) -> int:
    if isinstance(nibble, bool) or not isinstance(nibble, int):
        raise TypeError("nibble must be an integer")
    if not 0 <= nibble <= 0xF:
        raise ValueError("nibble must be in [0x0, 0xf]")
    return nibble


def _decode_sign_magnitude(nibble: int, magnitudes: Sequence[float]) -> float:
    nibble = _validated_nibble(nibble)
    magnitude = magnitudes[nibble & 0x7]
    return math.copysign(magnitude, -1.0 if nibble & 0x8 else 1.0)


def decode_e2m1(nibble: int) -> float:
    """Decode the public NVFP4 E2M1 nibble, preserving negative zero."""

    return _decode_sign_magnitude(nibble, _E2M1_MAGNITUDES)


def decode_candidate_e0m3(nibble: int) -> float:
    """Decode the candidate latent E0M3 sign-magnitude integer lattice.

    This is a hypothesis/reference table, not proof of SM120 behavior.
    """

    return _decode_sign_magnitude(nibble, _E0M3_MAGNITUDES)


def block_scale_ue4m3(code: int) -> float:
    """Decode the seven-bit unsigned E4M3 NVFP4 block-scale field.

    Bit 7 is rejected instead of silently masked because the CPU reference
    represents the stored dequant scale, not a proposed format tag.  Code 0x7f
    is the unique NaN encoding of the positive E4M3FN field; all other codes
    are finite.
    """

    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("UE4M3 code must be an integer")
    if not 0 <= code <= 0x7F:
        raise ValueError("UE4M3 scale code must be in [0x00, 0x7f]")
    if code == 0x7F:
        return math.nan
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * (2.0**-9)
    return (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def decode_nibble(nibble: int, fmt: str) -> float:
    if fmt == "e2m1":
        return decode_e2m1(nibble)
    if fmt == "e0m3_candidate":
        return decode_candidate_e0m3(nibble)
    raise ValueError(f"unsupported format: {fmt}")


def dot_reference(
    a_nibbles: Sequence[int],
    b_nibbles: Sequence[int],
    *,
    a_format: str,
    b_format: str,
    a_scale: float = 1.0,
    b_scale: float = 1.0,
    accumulator: float = 0.0,
) -> float:
    """Reference dot product with decoded operands and an initial accumulator."""

    if len(a_nibbles) != len(b_nibbles):
        raise ValueError("dot operands must have equal length")
    total = float(accumulator)
    for a_code, b_code in zip(a_nibbles, b_nibbles, strict=True):
        a_value = decode_nibble(a_code, a_format) * a_scale
        b_value = decode_nibble(b_code, b_format) * b_scale
        total += a_value * b_value
    return total


def matmul_reference(
    a_nibbles: Sequence[Sequence[int]],
    b_nibbles: Sequence[Sequence[int]],
    *,
    a_format: str,
    b_format: str,
    a_scale: float = 1.0,
    b_scale: float = 1.0,
    accumulator: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Small, dependency-free matrix product used for structured ISA vectors."""

    if not a_nibbles or not b_nibbles:
        raise ValueError("matrices must be non-empty")
    k = len(a_nibbles[0])
    if any(len(row) != k for row in a_nibbles):
        raise ValueError("A must be rectangular")
    if len(b_nibbles) != k:
        raise ValueError("A columns must equal B rows")
    n = len(b_nibbles[0])
    if any(len(row) != n for row in b_nibbles):
        raise ValueError("B must be rectangular")
    m = len(a_nibbles)
    if accumulator is None:
        accumulator = [[0.0 for _ in range(n)] for _ in range(m)]
    if len(accumulator) != m or any(len(row) != n for row in accumulator):
        raise ValueError("accumulator must have shape M x N")

    result: list[list[float]] = []
    for row_index in range(m):
        row: list[float] = []
        for column_index in range(n):
            b_column = [b_nibbles[k_index][column_index] for k_index in range(k)]
            row.append(
                dot_reference(
                    a_nibbles[row_index],
                    b_column,
                    a_format=a_format,
                    b_format=b_format,
                    a_scale=a_scale,
                    b_scale=b_scale,
                    accumulator=accumulator[row_index][column_index],
                )
            )
        result.append(row)
    return result
