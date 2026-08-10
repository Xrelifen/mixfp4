"""True two-dimensional hardware-format regions and explicit tail handling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionShape:
    rows: int
    k16_groups: int

    @property
    def k_values(self) -> int:
        return self.k16_groups * 16


def normalize_region(mode: str, operand_role: str) -> str:
    value = mode.lower().replace("-", "_")
    for prefix in ("mixfp4_", "nvfp4_mixfp4_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    aliases = {
        "nvfp4": "all_e2m1",
        "all_e2": "all_e2m1",
        "e2": "all_e2m1",
        "all_e0": "all_e0m3",
        "e0": "all_e0m3",
        "oracle16_a": "oracle16",
        "k64_row_a": "k64_row",
    }
    value = aliases.get(value, value)
    if operand_role == "activation_a" and value == "k64_row_a":
        value = "k64_row"
    return value


def region_shape(mode: str, operand_role: str, rows: int, groups: int) -> RegionShape:
    mode = normalize_region(mode, operand_role)
    if mode == "oracle16":
        return RegionShape(1, 1)
    if mode in {"all_e2m1", "all_e0m3", "layer"}:
        return RegionShape(rows, groups)
    weight = {
        "k32_row": RegionShape(1, 2),
        "k64_row": RegionShape(1, 4),
        "n8k16": RegionShape(8, 1),
        "n2k64": RegionShape(2, 4),
        "n4k64": RegionShape(4, 4),
        "n8k64": RegionShape(8, 4),
        "n16k64": RegionShape(16, 4),
        "n32k64": RegionShape(32, 4),
        "n64k64": RegionShape(64, 4),
    }
    activation = {
        "k64_row": RegionShape(1, 4),
        "m16k16": RegionShape(16, 1),
        "m4k64": RegionShape(4, 4),
        "m8k64": RegionShape(8, 4),
        "m16k64": RegionShape(16, 4),
        "m32k64": RegionShape(32, 4),
    }
    table = weight if operand_role == "weight_b" else activation
    if mode not in table:
        raise ValueError(f"format_region={mode!r} is invalid for {operand_role}")
    return table[mode]


def aggregate_regions(values: torch.Tensor, shape: RegionShape) -> torch.Tensor:
    """Sum [row,K16] values over real 2-D regions; padded cells contribute zero."""
    if values.ndim != 2:
        raise ValueError("region values must have [row,K16-group] shape")
    rows, groups = values.shape
    padded_rows = math.ceil(rows / shape.rows) * shape.rows
    padded_groups = math.ceil(groups / shape.k16_groups) * shape.k16_groups
    padded = F.pad(values, (0, padded_groups - groups, 0, padded_rows - rows))
    return padded.reshape(
        padded_rows // shape.rows,
        shape.rows,
        padded_groups // shape.k16_groups,
        shape.k16_groups,
    ).sum(dim=(1, 3))


def expand_region_values(
    values: torch.Tensor,
    *,
    rows: int,
    groups: int,
    shape: RegionShape,
) -> torch.Tensor:
    expanded = values[:, None, :, None].expand(-1, shape.rows, -1, shape.k16_groups)
    expanded = expanded.reshape(values.shape[0] * shape.rows, values.shape[1] * shape.k16_groups)
    return expanded[:rows, :groups]


def iter_region_bounds(rows: int, groups: int, shape: RegionShape):
    for row_start in range(0, rows, shape.rows):
        for group_start in range(0, groups, shape.k16_groups):
            yield (
                row_start,
                min(row_start + shape.rows, rows),
                group_start,
                min(group_start + shape.k16_groups, groups),
            )
