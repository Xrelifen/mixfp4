"""Exact block-K64 Linear-input covariance collection for Phase B selectors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .modeling import ModelPreparation, named_linears
from .rotation import TransformSpec, apply_transform


@dataclass
class CovarianceCollector:
    """Collect cumulative mean X^T X per K64 block without duplicate projection work."""

    model: nn.Module
    preparation: ModelPreparation
    accumulators: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    module_to_canonical: dict[str, str] = field(default_factory=dict)
    hooks: list[Any] = field(default_factory=list)
    _current_inputs: dict[tuple[int, int, tuple[int, ...], str], str] = field(default_factory=dict)
    _held_inputs: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name, linear in named_linears(self.model):
            spec = self.preparation.transform_by_module.get(name)
            self.hooks.append(linear.register_forward_pre_hook(self._hook(name, spec)))

    def begin_sequence(self) -> None:
        self._current_inputs.clear()
        self._held_inputs.clear()

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._held_inputs.clear()

    def _hook(self, name: str, spec: TransformSpec | None):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            raw = inputs[0]
            spec_name = spec.name if spec is not None else "identity"
            key = (id(raw), raw.data_ptr(), tuple(raw.shape), spec_name)
            self._held_inputs.append(raw)
            if key in self._current_inputs:
                canonical = self._current_inputs[key]
                prior = self.module_to_canonical.setdefault(name, canonical)
                if prior != canonical:
                    raise RuntimeError(f"calibration alias changed for {name}: {prior} -> {canonical}")
                return inputs

            canonical = name
            self._current_inputs[key] = canonical
            prior = self.module_to_canonical.setdefault(name, canonical)
            if prior != canonical:
                raise RuntimeError(f"calibration canonical changed for {name}: {prior} -> {canonical}")

            x = apply_transform(raw, spec) if spec is not None else raw
            x = x.detach().float().reshape(-1, x.shape[-1])
            # K128 covariance retains the cross-K64 terms needed to derive exact
            # covariances after every transform in the Phase-B bank, including H128.
            k128_blocks = math.ceil(x.shape[-1] / 128)
            if x.shape[-1] != k128_blocks * 128:
                x = F.pad(x, (0, k128_blocks * 128 - x.shape[-1]))
            blocks = x.reshape(-1, k128_blocks, 128)
            covariance = torch.einsum("tki,tkj->kij", blocks, blocks)
            if canonical not in self.accumulators:
                self.accumulators[canonical] = covariance
                self.sample_counts[canonical] = blocks.shape[0]
            else:
                self.accumulators[canonical].add_(covariance)
                self.sample_counts[canonical] += blocks.shape[0]
            return inputs

        return hook

    def payload(self, *, calibration_sequences: int, metadata: dict[str, Any]) -> dict[str, Any]:
        means128 = {
            name: value.detach().cpu().div(float(self.sample_counts[name])).contiguous()
            for name, value in self.accumulators.items()
        }
        means64 = {
            name: torch.stack((value[:, :64, :64], value[:, 64:, 64:]), dim=1).reshape(-1, 64, 64).contiguous()
            for name, value in means128.items()
        }
        module_names = {name for name, _ in named_linears(self.model)}
        if set(self.module_to_canonical) != module_names:
            missing = sorted(module_names - set(self.module_to_canonical))
            raise RuntimeError(f"calibration did not observe every Linear input: {missing}")
        return {
            "schema_version": 1,
            "calibration_sequences": calibration_sequences,
            "hessian_definition": "mean over calibration tokens of block-diagonal K128 X^T X; exact K64 diagonals also stored",
            "scale_group_size": 16,
            "format_region": "N8K64",
            "module_to_canonical": dict(sorted(self.module_to_canonical.items())),
            "sample_counts_by_canonical": dict(sorted(self.sample_counts.items())),
            "hessian_by_canonical": means64,
            "hessian_k128_by_canonical": means128,
            "metadata": metadata,
        }


def save_covariances(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(block)
    digest = hash_value.hexdigest()
    return digest


def load_calibration_by_module(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    canonical = value["hessian_by_canonical"]
    by_module = {
        module: {"hessian_k64": canonical[source]}
        for module, source in value["module_to_canonical"].items()
    }
    return by_module, value
