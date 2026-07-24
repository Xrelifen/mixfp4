#!/usr/bin/env python3
"""Patch validated SM120 OMMA format-select bits in copies of an e2m1 x e2m1 baseline executable.

Generalized port of 3rdparty/sm120-e0m3-mma/patch_executable_formats.py: that version only
patched the scale_vec::4X + UE4M3 (NVFP4) OMMA encoding. This one matches any block-scaled
kind::mxf4nvf4 OMMA mnemonic (UE4M3 or UE8M0 scale, 4X or 2X grouping) so the same E0M3 swap can
be verified across all three PTX-legal scale configs.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import struct
import subprocess


FIELD_SHIFT = 14
FORMATS = {
    "e0m3_e2m1": 0b01,
    "e2m1_e0m3": 0b10,
    "e0m3_e0m3": 0b11,
}


def replace_all(data: bytes, needle: bytes, replacement: bytes) -> tuple[bytes, int]:
    count = data.count(needle)
    if count == 0:
        raise RuntimeError("baseline OMMA instruction encoding was not found")
    return data.replace(needle, replacement), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument(
        "--cuobjdump",
        type=pathlib.Path,
        default=pathlib.Path("/usr/local/cuda/bin/cuobjdump"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=list(FORMATS),
        help="format variants to create",
    )
    args = parser.parse_args()

    data = args.baseline.read_bytes()
    sass = subprocess.run(
        [str(args.cuobjdump), "--dump-sass", str(args.baseline)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    # Matches OMMA.SF.<k>.F32.E2M1.E2M1.<scale suffix> for any scale type/grouping, e.g.
    # ".UE4M3.4X", ".E8.4X" (UE8M0 4X), or ".E8" (UE8M0 2X, no suffix in this disassembler).
    pattern = re.compile(
        r"OMMA\.SF\.\d+\.F32\.E2M1\.E2M1\.\S*[^\n]*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/\s*\n\s*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/"
    )
    encodings = {(int(a, 16), int(b, 16)) for a, b in pattern.findall(sass)}
    if not encodings:
        raise RuntimeError("cuobjdump found no baseline E2M1 x E2M1 OMMA instructions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.formats:
        format_bits = FORMATS[name]
        patched = data
        count = 0
        patched_words: list[str] = []
        for first_word, second_word in encodings:
            if ((second_word >> FIELD_SHIFT) & 0b11) != 0:
                raise RuntimeError(
                    f"expected baseline E2M1 selector bits to be zero, "
                    f"got second_word=0x{second_word:016x}"
                )
            patched_word = second_word | (format_bits << FIELD_SHIFT)
            needle = struct.pack("<QQ", first_word, second_word)
            replacement = struct.pack("<QQ", first_word, patched_word)
            patched, replacement_count = replace_all(patched, needle, replacement)
            count += replacement_count
            patched_words.append(f"0x{patched_word:016x}")
        output = args.output_dir / f"omma_{name}"
        output.write_bytes(patched)
        os.chmod(output, args.baseline.stat().st_mode)
        print(
            f"{name}: format_bits=0b{format_bits:02b} replacements={count} "
            f"second_words={','.join(patched_words)} output={output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
