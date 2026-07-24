#!/usr/bin/env python3
"""Independently recomputes the expected mma_probe.cu dot product.

Deliberately a second, from-scratch implementation of the E2M1/E0M3 codebooks and the
UE4M3/UE8M0 scale decode -- it must never import mma_probe.cu's own C++ decode logic, so that
run_tests.sh is cross-checking the GPU result against an independent reference rather than the
program's self-reported arithmetic.

Usage: expected_value.py <a_format> <b_format> <a_nibble> <b_nibble> <scale_type> <sfa> <sfb>
  a_format/b_format: e2m1 | e0m3
  scale_type: ue4m3 | ue8m0
  a_nibble/b_nibble/sfa/sfb: integers (0x.. accepted)

All four K groups share one scale value (mma_probe.cu packs one byte into every group), so the
full K=64 dot product is just K * (a_value * sfa) * (b_value * sfb).
"""

import sys

E2M1_CODEBOOK = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
E0M3_CODEBOOK = [0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7]
K = 64


def decode_ue4m3(code):
    code &= 0x7F
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0 ** -9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def decode_ue8m0(code):
    code &= 0xFF
    return 2.0 ** (code - 127)


def main():
    a_format, b_format, a_nibble, b_nibble, scale_type, sfa_code, sfb_code = sys.argv[1:8]
    a_nibble, b_nibble = int(a_nibble, 0), int(b_nibble, 0)
    sfa_code, sfb_code = int(sfa_code, 0), int(sfb_code, 0)

    a_book = E0M3_CODEBOOK if a_format == "e0m3" else E2M1_CODEBOOK
    b_book = E0M3_CODEBOOK if b_format == "e0m3" else E2M1_CODEBOOK
    decode_scale = decode_ue8m0 if scale_type == "ue8m0" else decode_ue4m3

    a_value = a_book[a_nibble] * decode_scale(sfa_code)
    b_value = b_book[b_nibble] * decode_scale(sfb_code)
    print(repr(K * a_value * b_value))


if __name__ == "__main__":
    main()
