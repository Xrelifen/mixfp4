#!/usr/bin/env python3
"""Offset-aware SASS patcher for mixed_dispatch_probe.cu.

patch_formats.py (used elsewhere in this test suite) blanket-replaces every occurrence of one
instruction encoding with the same patched encoding -- correct when a binary contains one
*logical* instruction site repeated by a runtime loop, but wrong here: mixed_dispatch_probe.cu
contains four *distinct* static mma.sync call sites that must each be patched to a *different*
target format, and (as observed empirically) at least two of them share byte-identical encodings,
so value-based matching can't tell them apart.

Occurrences are matched to call sites by ascending file offset. cuobjdump disassembles the
(single) kernel function linearly -- mma_site0..3 get inlined into it despite __noinline__, so
there is exactly one "Function :" block -- meaning OMMA occurrences appear in the disassembly in
the same left-to-right order as their relative addresses within that function, which is the same
order their raw 16-byte encodings appear at ascending file offsets. This ordering is unambiguous
only because everything lives in one function; it would not generalize as-is to a binary with
multiple functions each containing one site.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import struct
import subprocess


FIELD_SHIFT = 14
FORMAT_BITS = {
    "e2m1_e2m1": 0b00,
    "e0m3_e2m1": 0b01,
    "e2m1_e0m3": 0b10,
    "e0m3_e0m3": 0b11,
}


def find_all_offsets(data: bytes, needle: bytes) -> list[int]:
    offsets = []
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--cuobjdump", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda/bin/cuobjdump")
    )
    parser.add_argument(
        "--site-formats", nargs=4, metavar=("SITE0", "SITE1", "SITE2", "SITE3"),
        default=["e2m1_e2m1", "e0m3_e2m1", "e2m1_e0m3", "e0m3_e0m3"],
        choices=list(FORMAT_BITS),
        help="target format for site0..site3, in ascending-address order",
    )
    args = parser.parse_args()

    data = bytearray(args.baseline.read_bytes())
    sass = subprocess.run(
        [str(args.cuobjdump), "--dump-sass", str(args.baseline)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout

    func_headers = re.findall(r"^\s*Function : (\S+)\s*$", sass, re.MULTILINE)
    if len(func_headers) != 1:
        raise RuntimeError(
            f"expected exactly 1 function in the disassembly (mma_site0..3 inlined into it), "
            f"found {len(func_headers)}: {func_headers}"
        )

    pattern = re.compile(
        r"OMMA\.SF\.\d+\.F32\.E2M1\.E2M1\.\S*[^\n]*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/\s*\n\s*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/"
    )
    matches = pattern.findall(sass)
    if len(matches) != 4:
        raise RuntimeError(f"expected exactly 4 OMMA occurrences in the disassembly, found {len(matches)}")

    # Collect (offset, first_word, second_word) across every DISTINCT byte pattern seen (there may
    # be 1-4 distinct patterns among the 4 sites -- we observed 2 in practice), then globally sort
    # by file offset. That reconstructs left-to-right call-site order regardless of how many
    # distinct encodings exist.
    occurrences = []
    seen_values = set()
    for first_hex, second_hex in matches:
        value = (int(first_hex, 16), int(second_hex, 16))
        if value in seen_values:
            continue
        seen_values.add(value)
        needle = struct.pack("<QQ", *value)
        for offset in find_all_offsets(bytes(data), needle):
            occurrences.append((offset, *value))

    occurrences.sort(key=lambda t: t[0])
    if len(occurrences) != 4:
        raise RuntimeError(
            f"expected exactly 4 raw-byte occurrences across {len(seen_values)} distinct "
            f"instruction values, found {len(occurrences)} total: {occurrences}"
        )

    for site_index, ((offset, first_word, second_word), target_name) in enumerate(
        zip(occurrences, args.site_formats)
    ):
        if ((second_word >> FIELD_SHIFT) & 0b11) != 0:
            raise RuntimeError(
                f"site{site_index} at offset 0x{offset:x}: expected baseline E2M1 selector bits "
                f"to be zero, got second_word=0x{second_word:016x}"
            )
        format_bits = FORMAT_BITS[target_name]
        patched_word = second_word | (format_bits << FIELD_SHIFT)
        data[offset:offset + 16] = struct.pack("<QQ", first_word, patched_word)
        print(f"site{site_index}: offset=0x{offset:x} format={target_name} "
              f"format_bits=0b{format_bits:02b} patched_second_word=0x{patched_word:016x}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(data))
    os.chmod(args.output, args.baseline.stat().st_mode)
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
