"""Generate deterministic CPU-only ISA reference artifacts.

Large CSVs are generated mechanically from the reviewed independent reference.
Native observation columns remain NOT_RUN when no SM120 result is supplied.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from cpu_reference import decode_candidate_e0m3, decode_e2m1, dot_reference


BLOCK_REASON = "SM120 GPU unavailable"


def display_value(value: float) -> str:
    if value == 0.0:
        return "-0" if math.copysign(1.0, value) < 0 else "+0"
    return format(value, ".17g")


def write_decode_table(path: Path, fmt: str) -> None:
    decoder = decode_e2m1 if fmt == "e2m1" else decode_candidate_e0m3
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=(
                "nibble",
                "sign",
                "decoded_value",
                "zero_kind",
                "cpu_expected",
                "gpu_observed",
                "absolute_error",
                "pass_fail",
                "blocking_reason",
            ),
        )
        writer.writeheader()
        for nibble in range(16):
            value = decoder(nibble)
            zero_kind = "negative_zero" if value == 0.0 and math.copysign(1.0, value) < 0 else (
                "positive_zero" if value == 0.0 else "not_zero"
            )
            writer.writerow(
                {
                    "nibble": f"0x{nibble:x}",
                    "sign": "negative" if nibble & 0x8 else "positive",
                    "decoded_value": display_value(value),
                    "zero_kind": zero_kind,
                    "cpu_expected": display_value(value),
                    "gpu_observed": "",
                    "absolute_error": "",
                    "pass_fail": "NOT_RUN",
                    "blocking_reason": BLOCK_REASON,
                }
            )


def write_truth_table(path: Path) -> None:
    combinations = (
        ("e2m1", "e2m1"),
        ("e0m3_candidate", "e2m1"),
        ("e2m1", "e0m3_candidate"),
        ("e0m3_candidate", "e0m3_candidate"),
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=(
                "a_format",
                "b_format",
                "a_nibble",
                "b_nibble",
                "k",
                "a_scale",
                "b_scale",
                "accumulator_initial",
                "cpu_expected",
                "gpu_observed",
                "absolute_error",
                "pass_fail",
                "blocking_reason",
            ),
        )
        writer.writeheader()
        for a_format, b_format in combinations:
            for a_nibble in range(16):
                for b_nibble in range(16):
                    expected = dot_reference(
                        [a_nibble] * 64,
                        [b_nibble] * 64,
                        a_format=a_format,
                        b_format=b_format,
                    )
                    writer.writerow(
                        {
                            "a_format": a_format,
                            "b_format": b_format,
                            "a_nibble": f"0x{a_nibble:x}",
                            "b_nibble": f"0x{b_nibble:x}",
                            "k": 64,
                            "a_scale": 1.0,
                            "b_scale": 1.0,
                            "accumulator_initial": 0.0,
                            "cpu_expected": display_value(expected),
                            "gpu_observed": "",
                            "absolute_error": "",
                            "pass_fail": "NOT_RUN",
                            "blocking_reason": BLOCK_REASON,
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isa_dir = args.output / "isa_characterization"
    isa_dir.mkdir(parents=True, exist_ok=True)
    write_decode_table(isa_dir / "e2m1_decode_table.csv", "e2m1")
    write_decode_table(isa_dir / "e0m3_decode_table.csv", "e0m3_candidate")
    write_truth_table(isa_dir / "mixed_operand_truth_table.csv")


if __name__ == "__main__":
    main()
