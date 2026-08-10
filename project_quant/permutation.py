"""Margin-aware N8 channel packing and exact foldable permutations."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import torch
from torch import nn


def k64_sums(values: torch.Tensor) -> torch.Tensor:
    rows, groups = values.shape
    padding = (-groups) % 4
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    return values.reshape(rows, -1, 4).sum(dim=-1)


def signed_margin_signatures(e2_errors: torch.Tensor, e0_errors: torch.Tensor) -> torch.Tensor:
    if e2_errors.shape != e0_errors.shape or e2_errors.ndim != 2:
        raise ValueError("candidate errors must have matching [N,K16-block] shapes")
    return k64_sums(e2_errors - e0_errors)


def sort_by_e0_ratio(e2_errors: torch.Tensor, e0_errors: torch.Tensor) -> torch.Tensor:
    ratio = (e0_errors < e2_errors).float().mean(dim=-1)
    return torch.argsort(ratio, descending=True, stable=True)


def margin_vector_clustering(e2_errors: torch.Tensor, e0_errors: torch.Tensor) -> torch.Tensor:
    """Deterministic locality clustering from full signed-margin signatures.

    Rows are bucketed by several deterministic signed projections, then ordered
    within buckets by margin energy and E0 ratio.  This bounded projection
    clustering scales to 14k-wide MLPs while retaining the full K64 signature;
    it is deliberately distinct from a scalar E0-ratio sort.
    """
    signature = signed_margin_signatures(e2_errors, e0_errors).detach().float().cpu().numpy()
    n, width = signature.shape
    if width == 0:
        return torch.arange(n, device=e2_errors.device)
    norm = np.linalg.norm(signature, axis=1, keepdims=True)
    normalized = signature / np.maximum(norm, 1e-30)
    positions = np.arange(width, dtype=np.float64) + 0.5
    projections = []
    for frequency in (1, 2, 3, 5, 8, 13):
        vector = np.cos(np.pi * frequency * positions / width)
        projections.append(normalized @ vector)
    projection = np.stack(projections, axis=1)
    sign_code = np.zeros(n, dtype=np.int64)
    for index in range(projection.shape[1]):
        sign_code |= (projection[:, index] >= 0).astype(np.int64) << index
    e0_ratio = (e0_errors < e2_errors).float().mean(dim=1).detach().cpu().numpy()
    energy = norm[:, 0]
    # np.lexsort uses the final key as primary. Stable original indices resolve
    # all ties and make the packing byte-for-byte deterministic.
    order = np.lexsort((np.arange(n), -energy, -e0_ratio, projection[:, 2], projection[:, 1], projection[:, 0], sign_code))
    return torch.from_numpy(order.astype(np.int64)).to(e2_errors.device)


def greedy_min_regret_n8(
    e2_errors: torch.Tensor,
    e0_errors: torch.Tensor,
    *,
    window: int = 64,
    initial_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedily assemble N8 groups minimizing exact K64 MSE regret."""
    if e2_errors.shape != e0_errors.shape or e2_errors.ndim != 2:
        raise ValueError("candidate errors must have matching [N,K16-block] shapes")
    if window < 8:
        raise ValueError("window must be at least 8")
    if initial_order is None:
        initial_order = margin_vector_clustering(e2_errors, e0_errors)
    order = initial_order.detach().cpu().numpy().astype(np.int64)
    e2 = k64_sums(e2_errors).detach().float().cpu().numpy()
    e0 = k64_sums(e0_errors).detach().float().cpu().numpy()
    oracle = np.minimum(e2, e0).sum(axis=1)
    packed: list[int] = []
    for window_start in range(0, len(order), window):
        remaining = order[window_start : window_start + window].tolist()
        while remaining:
            group = [remaining.pop(0)]
            sum_e2 = e2[group[0]].copy()
            sum_e0 = e0[group[0]].copy()
            oracle_sum = float(oracle[group[0]])
            while remaining and len(group) < 8:
                candidates = np.asarray(remaining, dtype=np.int64)
                constrained = np.minimum(
                    sum_e2[None, :] + e2[candidates], sum_e0[None, :] + e0[candidates]
                ).sum(axis=1)
                regret = constrained - (oracle_sum + oracle[candidates])
                selected_offset = int(np.argmin(regret))
                selected = remaining.pop(selected_offset)
                group.append(selected)
                sum_e2 += e2[selected]
                sum_e0 += e0[selected]
                oracle_sum += float(oracle[selected])
            packed.extend(group)
    result = torch.tensor(packed, device=e2_errors.device, dtype=torch.long)
    if result.numel() != e2_errors.shape[0] or torch.unique(result).numel() != result.numel():
        raise RuntimeError("N8 packing failed to produce an exact permutation")
    return result


def choose_permutation(
    method: str,
    e2_errors: torch.Tensor,
    e0_errors: torch.Tensor,
    *,
    sensitivity_e2: torch.Tensor | None = None,
    sensitivity_e0: torch.Tensor | None = None,
) -> torch.Tensor:
    method = method.lower()
    if method in {"none", "no_permutation"}:
        return torch.arange(e2_errors.shape[0], device=e2_errors.device)
    if method in {"e0_ratio_sort", "sort_by_e0_ratio"}:
        return sort_by_e0_ratio(e2_errors, e0_errors)
    if method == "margin_vector_clustering":
        return margin_vector_clustering(e2_errors, e0_errors)
    if method in {"greedy_min_regret_n8", "greedy_min_regret_n8_packing"}:
        return greedy_min_regret_n8(e2_errors, e0_errors)
    if method == "sensitivity_weighted_greedy_n8":
        if sensitivity_e2 is None or sensitivity_e0 is None:
            raise ValueError("sensitivity_weighted_greedy_n8 requires sensitivity candidate errors")
        initial = margin_vector_clustering(sensitivity_e2, sensitivity_e0)
        return greedy_min_regret_n8(sensitivity_e2, sensitivity_e0, initial_order=initial)
    raise ValueError(f"unknown permutation method {method!r}")


@torch.no_grad()
def permute_linear_output(linear: nn.Linear, permutation: torch.Tensor) -> torch.Tensor:
    """Permute rows/bias and return the explicit inverse (upper-bound path)."""
    permutation = permutation.to(linear.weight.device)
    if permutation.numel() != linear.out_features:
        raise ValueError("output permutation size must match out_features")
    linear.weight.data = linear.weight.data[permutation].contiguous()
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[permutation].contiguous()
    return torch.argsort(permutation)


def inverse_output_hook(inverse: torch.Tensor):
    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return output.index_select(-1, inverse.to(output.device))

    return hook


@torch.no_grad()
def apply_foldable_mlp_permutation(mlp: nn.Module, permutation: torch.Tensor) -> None:
    """Fold one intermediate permutation across gate/up rows and down columns."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(mlp, name, None), nn.Linear):
            raise TypeError("foldable permutation requires gate_proj/up_proj/down_proj Linear modules")
    gate: nn.Linear = mlp.gate_proj
    up: nn.Linear = mlp.up_proj
    down: nn.Linear = mlp.down_proj
    permutation = permutation.to(gate.weight.device)
    if (
        permutation.numel() != gate.out_features
        or up.out_features != gate.out_features
        or down.in_features != gate.out_features
    ):
        raise ValueError("inconsistent gated-MLP intermediate dimensions")
    gate.weight.data = gate.weight.data[permutation].contiguous()
    up.weight.data = up.weight.data[permutation].contiguous()
    if gate.bias is not None:
        gate.bias.data = gate.bias.data[permutation].contiguous()
    if up.bias is not None:
        up.bias.data = up.bias.data[permutation].contiguous()
    down.weight.data = down.weight.data[:, permutation].contiguous()


@torch.no_grad()
def apply_foldable_ffn_permutation(
    first: nn.Linear, second: nn.Linear, permutation: torch.Tensor
) -> None:
    """Fold a permutation through an exact y=W2 activation(W1 x) motif."""
    permutation = permutation.to(first.weight.device)
    if first.out_features != second.in_features or permutation.numel() != first.out_features:
        raise ValueError("FFN projections do not share the requested intermediate dimension")
    first.weight.data = first.weight.data[permutation].contiguous()
    if first.bias is not None:
        first.bias.data = first.bias.data[permutation].contiguous()
    second.weight.data = second.weight.data[:, permutation].contiguous()


@torch.no_grad()
def apply_foldable_glumbconv_permutation(module: nn.Module, permutation: torch.Tensor) -> None:
    """Fold one hidden-channel permutation through SANA's GLUMBConv motif.

    The inverted 1x1 convolution and its depthwise convolution contain two
    corresponding hidden-channel halves.  Both halves receive the same
    permutation, while the pointwise output convolution receives the matching
    input-column permutation.
    """
    inverted = getattr(module, "conv_inverted", None)
    depth = getattr(module, "conv_depth", None)
    point = getattr(module, "conv_point", None)
    if not all(isinstance(value, nn.Conv2d) for value in (inverted, depth, point)):
        raise TypeError("foldable GLUMBConv permutation requires three Conv2d modules")
    hidden = point.in_channels
    if (
        inverted.kernel_size != (1, 1)
        or point.kernel_size != (1, 1)
        or inverted.out_channels != 2 * hidden
        or depth.in_channels != 2 * hidden
        or depth.out_channels != 2 * hidden
        or depth.groups != 2 * hidden
        or permutation.numel() != hidden
    ):
        raise ValueError("inconsistent GLUMBConv hidden-channel dimensions")
    permutation = permutation.to(inverted.weight.device)
    paired = torch.cat((permutation, permutation + hidden))
    inverted.weight.data = inverted.weight.data[paired].contiguous()
    if inverted.bias is not None:
        inverted.bias.data = inverted.bias.data[paired].contiguous()
    depth.weight.data = depth.weight.data[paired].contiguous()
    if depth.bias is not None:
        depth.bias.data = depth.bias.data[paired].contiguous()
    point.weight.data = point.weight.data[:, permutation].contiguous()


def combined_candidate_errors(
    results: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = list(results)
    if not pairs:
        raise ValueError("at least one candidate-error pair is required")
    return torch.stack([pair[0] for pair in pairs]).sum(0), torch.stack(
        [pair[1] for pair in pairs]
    ).sum(0)


def permutation_sha256(permutation: torch.Tensor) -> str:
    array = permutation.detach().cpu().numpy().astype(np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()
