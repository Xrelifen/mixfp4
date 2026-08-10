"""Normalized block-Hadamard transform bank and calibration objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransformSpec:
    name: str
    block_size: int | None
    seed: int | None = None
    deployability: str = "potentially_foldable"


TRANSFORM_BANK = (
    TransformSpec("identity", None, deployability="exact_foldable"),
    TransformSpec("H16", 16),
    TransformSpec("H32", 32),
    TransformSpec("H64", 64),
    TransformSpec("H128", 128),
    TransformSpec("random_signed_H64_seed0", 64, 0),
    TransformSpec("random_signed_H64_seed1", 64, 1),
    TransformSpec("random_signed_H64_seed2", 64, 2),
    TransformSpec("random_signed_H64_seed3", 64, 3),
)


def get_transform(value: str | TransformSpec) -> TransformSpec:
    if isinstance(value, TransformSpec):
        return value
    for spec in TRANSFORM_BANK:
        if spec.name.lower() == value.lower():
            return spec
    raise ValueError(f"unknown transform {value!r}")


def signed_vector(size: int, seed: int, *, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64).mul(2).sub(1)
    return signs.to(device=device, dtype=dtype)


def fwht(x: torch.Tensor) -> torch.Tensor:
    size = x.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError("FWHT final dimension must be a positive power of two")
    result = x
    stride = 1
    while stride < size:
        paired = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = paired[..., 0, :], paired[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape_as(result)
        stride *= 2
    return result / math.sqrt(size)


def apply_transform(x: torch.Tensor, value: str | TransformSpec) -> torch.Tensor:
    spec = get_transform(value)
    if spec.block_size is None:
        return x
    size = spec.block_size
    full_k = x.shape[-1] // size * size
    if full_k == 0:
        return x
    blocks = x[..., :full_k].reshape(*x.shape[:-1], full_k // size, size)
    if spec.seed is not None:
        blocks = blocks * signed_vector(size, spec.seed, device=x.device, dtype=x.dtype)
    transformed = fwht(blocks).reshape(*x.shape[:-1], full_k)
    if full_k < x.shape[-1]:
        transformed = torch.cat((transformed, x[..., full_k:]), dim=-1)
    return transformed


def hadamard_matrix(size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix / math.sqrt(size)


def dense_transform_matrix(
    value: str | TransformSpec, k: int, *, device=None, dtype=torch.float32
) -> torch.Tensor:
    spec = get_transform(value)
    matrix = torch.eye(k, device=device, dtype=dtype)
    if spec.block_size is None:
        return matrix
    size = spec.block_size
    base = hadamard_matrix(size, device=device, dtype=dtype)
    if spec.seed is not None:
        base = torch.diag(signed_vector(size, spec.seed, device=device, dtype=dtype)) @ base
    for start in range(0, k // size * size, size):
        matrix[start : start + size, start : start + size] = base
    return matrix


@torch.no_grad()
def verify_rotation_equivalence(
    x: torch.Tensor, weight: torch.Tensor, value: str | TransformSpec
) -> dict[str, float | bool | str]:
    spec = get_transform(value)
    reference = x.float() @ weight.float().T
    rotated = apply_transform(x.float(), spec) @ apply_transform(weight.float(), spec).T
    delta = rotated - reference
    relative = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    return {
        "transform": spec.name,
        "deployability": spec.deployability,
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(relative.item()),
        "passed": bool(relative <= 2e-5),
    }


def normalized_objective(output_errors: torch.Tensor, regrets: torch.Tensor, lam: float) -> torch.Tensor:
    def normalize(values: torch.Tensor) -> torch.Tensor:
        span = values.max() - values.min()
        if span <= torch.maximum(values.abs().max(), torch.tensor(1.0, device=values.device)) * 1e-12:
            return torch.zeros_like(values)
        return (values - values.min()) / span

    return normalize(output_errors) + lam * normalize(regrets)
