"""SANA/DeepCompressor adapter using the shared MixFP4 numerical core."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import SanaPipeline
from torch import nn

from ..artifacts import ROOT
from ..candidates import build_candidates
from ..metrics import tensor_error_metrics
from ..modeling import aggregate_activation_metrics, aggregate_weight_metrics, quantize_weight_tensor
from ..permutation import (
    apply_foldable_glumbconv_permutation,
    choose_permutation,
    inverse_output_hook,
    permutation_sha256,
    permute_linear_output,
)
from ..region import aggregate_regions, region_shape
from ..rotation import TransformSpec, apply_transform, get_transform, verify_rotation_equivalence
from ..selector import sensitivity_scores


SANA_MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
SANA_REVISION = "ac0da2ff55fbe434795be0dce883042e4d49e2fc"
SANA_TRANSFORMER_REVISION = SANA_REVISION
SANA_VAE_REVISION = SANA_REVISION
SANA_TEXT_ENCODER_REVISION = SANA_REVISION
SANA_SNAPSHOT = Path(
    "/share2/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/"
    f"snapshots/{SANA_REVISION}"
)


@dataclass
class DiffusionPreparation:
    layer_metrics: list[dict[str, Any]] = field(default_factory=list)
    region_metrics: list[dict[str, Any]] = field(default_factory=list)
    margin_samples: list[dict[str, Any]] = field(default_factory=list)
    equivalence_checks: list[dict[str, Any]] = field(default_factory=list)
    permutation_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_modules: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[Any] = field(default_factory=list)
    transform_by_module: dict[str, TransformSpec] = field(default_factory=dict)


def load_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def load_sana_pipeline(*, with_vae: bool = True) -> SanaPipeline:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "use_safetensors": True,
    }
    if not with_vae:
        kwargs["vae"] = None
    pipeline = SanaPipeline.from_pretrained(str(SANA_SNAPSHOT), **kwargs)
    pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=True)
    pipeline.transformer.eval()
    if pipeline.text_encoder is not None:
        pipeline.text_encoder.eval()
    return pipeline


def sana_pipeline_provenance(pipeline: SanaPipeline) -> dict[str, Any]:
    """Return the exact component/scheduler provenance persisted by every run."""
    scheduler_config = json.loads(
        json.dumps(dict(pipeline.scheduler.config), default=lambda value: str(value))
    )
    component_blobs: dict[str, str | None] = {}
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = SANA_SNAPSHOT / relative
        if path.is_symlink():
            component_blobs[relative] = Path(path.readlink()).name
        elif path.exists():
            component_blobs[relative] = "regular_file_in_pinned_snapshot"
        else:
            component_blobs[relative] = None
    return {
        "model_id": SANA_MODEL_ID,
        "model_revision": SANA_REVISION,
        "transformer_revision": SANA_TRANSFORMER_REVISION,
        "vae_revision": SANA_VAE_REVISION,
        "text_encoder_revision": SANA_TEXT_ENCODER_REVISION,
        "scheduler_class": type(pipeline.scheduler).__name__,
        "scheduler_config": scheduler_config,
        "component_blob_ids": component_blobs,
        "pag_scale": 0.0,
        "pag_enabled": False,
    }


def is_matrix_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or (
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
        and module.groups == 1
    )


def named_matrix_modules(model: nn.Module):
    for name, module in model.named_modules():
        if is_matrix_module(module):
            yield name, module


def matrix_dimensions(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return module.out_features, module.in_features
    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
        return module.out_channels, module.in_channels
    raise TypeError(f"unsupported matrix module {type(module).__name__}")


def matrix_weight(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.data
    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
        return module.weight.data[:, :, 0, 0]
    raise TypeError(f"unsupported matrix module {type(module).__name__}")


def set_matrix_weight(module: nn.Module, value: torch.Tensor) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data = value.contiguous()
    elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
        module.weight.data = value[:, :, None, None].contiguous()
    else:
        raise TypeError(f"unsupported matrix module {type(module).__name__}")


def _n8k64_regret_from_errors(e2: torch.Tensor, e0: torch.Tensor) -> float:
    shape = region_shape("n8k64", "weight_b", e2.shape[0], e2.shape[1])
    oracle = torch.minimum(e2, e0).sum()
    constrained = torch.minimum(
        aggregate_regions(e2, shape), aggregate_regions(e0, shape)
    ).sum()
    return float((constrained - oracle).item())


def _matrix_float(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    bias = module.bias.float() if module.bias is not None else None
    if isinstance(module, nn.Linear):
        return F.linear(value.float(), module.weight.float(), bias)
    if isinstance(module, nn.Conv2d):
        return F.conv2d(
            value.float(),
            module.weight.float(),
            bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    raise TypeError(f"unsupported matrix module {type(module).__name__}")


@torch.no_grad()
def _permute_matrix_output(module: nn.Module, permutation: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return permute_linear_output(module, permutation)
    if isinstance(module, nn.Conv2d) and is_matrix_module(module):
        permutation = permutation.to(module.weight.device)
        if permutation.numel() != module.out_channels:
            raise ValueError("output permutation size must match Conv2d out_channels")
        module.weight.data = module.weight.data[permutation].contiguous()
        if module.bias is not None:
            module.bias.data = module.bias.data[permutation].contiguous()
        return torch.argsort(permutation)
    raise TypeError(f"unsupported matrix module {type(module).__name__}")


def _inverse_matrix_output_hook(module: nn.Module, inverse: torch.Tensor):
    if isinstance(module, nn.Linear):
        return inverse_output_hook(inverse)

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return output.index_select(1, inverse.to(output.device))

    return hook


def matrix_input(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return value if isinstance(module, nn.Linear) else value.movedim(1, -1)


def restore_matrix_input(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return value if isinstance(module, nn.Linear) else value.movedim(-1, 1)


def diffusion_layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)transformer_blocks\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def diffusion_module_type(name: str) -> str:
    mapping = {
        ".attn1.to_q": "self_attn_q",
        ".attn1.to_k": "self_attn_k",
        ".attn1.to_v": "self_attn_v",
        ".attn1.to_out": "self_attn_out",
        ".attn2.to_q": "cross_attn_q",
        ".attn2.to_k": "cross_attn_k",
        ".attn2.to_v": "cross_attn_v",
        ".attn2.to_out": "cross_attn_out",
        ".ff.conv_inverted": "ffn_inverted_1x1",
        ".ff.conv_point": "ffn_output_1x1",
    }
    for fragment, label in mapping.items():
        if fragment in name:
            return label
    if name.startswith("caption_projection"):
        return "caption_projection"
    if name.startswith("time_embed"):
        return "time_embedding"
    if name == "patch_embed.proj":
        return "patch_projection"
    if name == "proj_out":
        return "output_projection"
    return "other_matrix"


def _layer_row(name: str, module: nn.Module, result) -> dict[str, Any]:
    summary = result.summary
    n, k = matrix_dimensions(module)
    keys = (
        "format_granularity", "scale_rule", "scale_group_size", "num_scale_blocks",
        "num_format_regions", "e0_ratio", "e2_ratio", "oracle_e0_ratio",
        "mean_homogeneity", "median_homogeneity", "p10_homogeneity", "p90_homogeneity",
        "mean_format_margin", "mean_abs_format_margin", "median_format_margin",
        "oracle_mse", "constrained_mse", "granularity_regret", "normalized_regret",
        "baseline_normalized_regret", "mse", "nmse", "relative_l2", "cosine_error",
        "max_abs_error", "mean_abs_error", "selector_disagreement_ratio",
        "selector_changed_mean_format_margin",
        "selector_changed_mean_abs_format_margin",
        "mean_margin_conflict", "margin_weighted_conflict", "mean_margin_homogeneity",
        "selected_mse", "selected_weight_mse_regret", "sensitivity_regret",
        "sensitivity_oracle_error", "sensitivity_constrained_error",
        "selected_sensitivity_error", "selected_sensitivity_regret", "calibration_output_sse_selected",
        "calibration_output_sse_mse_selector", "calibration_output_sse_all_e2",
        "calibration_output_sse_all_e0", "numel",
    )
    row = {key: summary.get(key) for key in keys}
    row.update(
        {
            "layer_idx": diffusion_layer_index(name),
            "module_name": name,
            "module_type": diffusion_module_type(name),
            "module_class": type(module).__name__,
            "N": n,
            "K": k,
        }
    )
    return row


@torch.no_grad()
def quantize_sana_weights(
    transformer: nn.Module,
    weight_mode: str,
    preparation: DiffusionPreparation,
    *,
    selector: str = "mse",
    calibration_by_module: dict[str, dict[str, Any]] | None = None,
) -> None:
    if weight_mode in {"high_precision", "fp16", "bf16"}:
        return
    for name, module in named_matrix_modules(transformer):
        source = matrix_weight(module)
        calibration = calibration_by_module.get(name) if calibration_by_module else None
        collect = weight_mode in {"n8k64", "n8k16", "k64_row", "layer"} or "n8k64" in weight_mode or selector != "mse"
        result = quantize_weight_tensor(
            source,
            weight_mode,
            selector=selector,
            calibration_stats=calibration,
            collect_regions=collect,
        )
        set_matrix_weight(module, result.dequant)
        preparation.layer_metrics.append(_layer_row(name, module, result))
        for region in result.regions:
            preparation.region_metrics.append(
                {
                    "module_name": name,
                    "module_type": diffusion_module_type(name),
                    "layer_idx": diffusion_layer_index(name),
                    **region,
                }
            )
        margins = (result.e2_errors - result.e0_errors).reshape(-1)
        if margins.numel() > 256:
            indices = torch.linspace(0, margins.numel() - 1, 256, device=margins.device).round().long()
            margins = margins[indices]
        preparation.margin_samples.extend(
            {"module_name": name, "module_type": diffusion_module_type(name), "margin": value}
            for value in margins.detach().float().cpu().tolist()
        )
        del result
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Conv2d) and not is_matrix_module(module):
            preparation.skipped_modules.append(
                {
                    "module_name": name,
                    "module_class": type(module).__name__,
                    "reason": "non_1x1_or_grouped_convolution_is_not_a_GEMM_NK_hardware_proxy",
                    "shape": list(module.weight.shape),
                    "groups": module.groups,
                    "kernel_size": list(module.kernel_size),
                }
            )


class DiffusionActivationHooks:
    def __init__(
        self,
        transformer: nn.Module,
        activation_mode: str,
        preparation: DiffusionPreparation,
        *,
        max_stats_calls_per_module: int = 2,
    ) -> None:
        self.mode = activation_mode
        self.preparation = preparation
        self.max_stats = max_stats_calls_per_module
        self.calls: dict[str, int] = {}
        self.metrics: list[dict[str, Any]] = []
        for name, module in named_matrix_modules(transformer):
            spec = preparation.transform_by_module.get(name)
            if activation_mode in {"high_precision", "fp16", "bf16"} and spec is None:
                continue
            preparation.hooks.append(module.register_forward_pre_hook(self._hook(name, module, spec)))

    def _hook(self, name: str, module: nn.Module, spec: TransformSpec | None):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            value = matrix_input(module, inputs[0])
            if spec is not None:
                value = apply_transform(value, spec)
            if self.mode not in {"high_precision", "fp16", "bf16"}:
                from ..core import GranularityResult, quant_mixfp4_granularity

                count = self.calls.get(name, 0)
                collect = count < self.max_stats
                result = quant_mixfp4_granularity(
                    value,
                    format_region=self.mode,
                    operand_role="activation_a",
                    return_stats=collect,
                    collect_regions=collect,
                    region_sample_limit=64,
                )
                if collect:
                    assert isinstance(result, GranularityResult)
                    self.metrics.append(
                        {
                            "module_name": name,
                            "module_type": diffusion_module_type(name),
                            "layer_idx": diffusion_layer_index(name),
                            "call_index": count,
                            **result.summary,
                        }
                    )
                    for region in result.regions:
                        self.preparation.region_metrics.append(
                            {
                                "module_name": name,
                                "module_type": diffusion_module_type(name),
                                "layer_idx": diffusion_layer_index(name),
                                "activation_call_index": count,
                                **region,
                            }
                        )
                    value = result.dequant
                else:
                    assert isinstance(result, torch.Tensor)
                    value = result
                self.calls[name] = count + 1
            return (restore_matrix_input(module, value), *inputs[1:])

        return hook


@torch.no_grad()
def apply_sana_rotations(
    transformer: nn.Module,
    rotation: str,
    preparation: DiffusionPreparation,
    per_module_rotation: dict[str, str] | None = None,
) -> None:
    if rotation == "identity" and not per_module_rotation:
        return
    for name, module in named_matrix_modules(transformer):
        selected = per_module_rotation.get(name, rotation) if per_module_rotation else rotation
        spec = get_transform(selected)
        if spec.name == "identity":
            continue
        weight = matrix_weight(module)
        x = torch.randn((2, weight.shape[1]), generator=torch.Generator().manual_seed(0)).to(weight.device)
        check = verify_rotation_equivalence(x, weight, spec)
        check.update({"module_name": name, "transform": spec.name, "deployability": spec.deployability})
        preparation.equivalence_checks.append(check)
        if not check["passed"]:
            raise RuntimeError(f"SANA rotation equivalence failed for {name}: {check}")
        set_matrix_weight(module, apply_transform(weight, spec))
        preparation.transform_by_module[name] = spec


@torch.no_grad()
def prepare_sana_permutations(
    transformer: nn.Module,
    method: str,
    preparation: DiffusionPreparation,
    *,
    calibration_by_module: dict[str, dict[str, Any]] | None = None,
) -> None:
    if method in {"none", "no_permutation"}:
        return
    if method.startswith("all_linear_"):
        local = method.removeprefix("all_linear_")
        for name, module in named_matrix_modules(transformer):
            candidates = build_candidates(matrix_weight(module))
            kwargs: dict[str, torch.Tensor] = {}
            if local == "sensitivity_weighted_greedy_n8":
                if calibration_by_module is None or name not in calibration_by_module:
                    raise ValueError(
                        "sensitivity-weighted SANA all-matrix packing requires matched calibration"
                    )
                shape = region_shape("n8k64", "weight_b", candidates.rows, candidates.groups)
                _, _, sensitivity_e2, sensitivity_e0 = sensitivity_scores(
                    candidates, calibration_by_module[name]["hessian_k64"], shape
                )
                kwargs = {
                    "sensitivity_e2": sensitivity_e2,
                    "sensitivity_e0": sensitivity_e0,
                }
            permutation = choose_permutation(
                local, candidates.e2_errors, candidates.e0_errors, **kwargs
            )
            regret_before = _n8k64_regret_from_errors(
                candidates.e2_errors, candidates.e0_errors
            )
            regret_after = _n8k64_regret_from_errors(
                candidates.e2_errors.index_select(0, permutation),
                candidates.e0_errors.index_select(0, permutation),
            )
            generator = torch.Generator(device="cpu").manual_seed(0)
            if isinstance(module, nn.Linear):
                x = torch.randn(
                    (2, 3, module.in_features), generator=generator, dtype=torch.float32
                ).to(module.weight.device)
                output_axis = -1
            else:
                x = torch.randn(
                    (1, module.in_channels, 3, 3), generator=generator, dtype=torch.float32
                ).to(module.weight.device)
                output_axis = 1
            reference = _matrix_float(module, x)
            inverse = _permute_matrix_output(module, permutation)
            actual = _matrix_float(module, x).index_select(output_axis, inverse)
            delta = actual - reference
            relative = float(
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(reference).clamp_min(1e-30)
            )
            if relative > 2e-6:
                raise RuntimeError(f"SANA all-matrix inverse equivalence failed for {name}: {relative}")
            preparation.hooks.append(
                module.register_forward_hook(_inverse_matrix_output_hook(module, inverse))
            )
            preparation.permutation_records.append(
                {
                    "module_name": name, "method": method, "deployability": "upper_bound_only",
                    "size": permutation.numel(), "permutation_sha256": permutation_sha256(permutation),
                    "permutation": permutation.detach().cpu().tolist(),
                    "module_class": type(module).__name__,
                    "equivalence_relative_l2": relative,
                    "equivalence_max_abs": float(delta.abs().max().item()),
                    "n8k64_regret_before": regret_before,
                    "n8k64_regret_after": regret_after,
                    "packing_objective_scope": "single_matrix_output_rows",
                }
            )
        return
    if method.startswith("foldable_sana_ffn_"):
        local = method.removeprefix("foldable_sana_ffn_")
        for block_index, block in enumerate(transformer.transformer_blocks):
            ff = block.ff
            weight = ff.conv_inverted.weight.data[:, :, 0, 0]
            hidden = ff.conv_point.in_channels
            candidates = build_candidates(weight)
            point_candidates_before = build_candidates(ff.conv_point.weight.data[:, :, 0, 0])
            e2 = candidates.e2_errors[:hidden] + candidates.e2_errors[hidden:]
            e0 = candidates.e0_errors[:hidden] + candidates.e0_errors[hidden:]
            kwargs: dict[str, torch.Tensor] = {}
            if local == "sensitivity_weighted_greedy_n8":
                name = f"transformer_blocks.{block_index}.ff.conv_inverted"
                if calibration_by_module is None or name not in calibration_by_module:
                    raise ValueError("sensitivity-weighted SANA packing requires calibration")
                shape = region_shape("n8k64", "weight_b", candidates.rows, candidates.groups)
                _, _, se2, se0 = sensitivity_scores(candidates, calibration_by_module[name]["hessian_k64"], shape)
                kwargs = {"sensitivity_e2": se2[:hidden] + se2[hidden:], "sensitivity_e0": se0[:hidden] + se0[hidden:]}
            permutation = choose_permutation(local, e2, e0, **kwargs)
            x = torch.randn((1, ff.conv_inverted.in_channels, 3, 3), generator=torch.Generator().manual_seed(0)).to(
                device=weight.device, dtype=weight.dtype
            )
            reference = ff(x).float()
            inverted_regret_before = _n8k64_regret_from_errors(e2, e0)
            point_regret_before = _n8k64_regret_from_errors(
                point_candidates_before.e2_errors, point_candidates_before.e0_errors
            )
            apply_foldable_glumbconv_permutation(ff, permutation)
            actual = ff(x).float()
            delta = actual - reference
            relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30))
            if relative > 3e-5:
                raise RuntimeError(f"SANA foldable FFN equivalence failed at block {block_index}: {relative}")
            point_candidates_after = build_candidates(ff.conv_point.weight.data[:, :, 0, 0])
            inverted_regret_after = _n8k64_regret_from_errors(
                e2.index_select(0, permutation), e0.index_select(0, permutation)
            )
            point_regret_after = _n8k64_regret_from_errors(
                point_candidates_after.e2_errors, point_candidates_after.e0_errors
            )
            preparation.permutation_records.append(
                {
                    "module_name": f"transformer_blocks.{block_index}.ff", "method": method,
                    "deployability": "exact_foldable", "size": permutation.numel(),
                    "permutation_sha256": permutation_sha256(permutation),
                    "permutation": permutation.detach().cpu().tolist(), "equivalence_relative_l2": relative,
                    "equivalence_max_abs": float(delta.abs().max()),
                    "inverted_pair_n8k64_regret_before": inverted_regret_before,
                    "inverted_pair_n8k64_regret_after": inverted_regret_after,
                    "point_n8k64_regret_before": point_regret_before,
                    "point_n8k64_regret_after": point_regret_after,
                    "motif_n8k64_regret_before": inverted_regret_before + point_regret_before,
                    "motif_n8k64_regret_after": inverted_regret_after + point_regret_after,
                    "packing_objective_scope": (
                        "paired_inverted_output_row_regret_optimized; "
                        "point_input_column_fold_regret_measured"
                    ),
                }
            )
        return
    raise ValueError(f"unsupported SANA permutation method {method!r}")


@dataclass
class DiffusionCovarianceCollector:
    model: nn.Module
    preparation: DiffusionPreparation
    accumulators: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    module_to_canonical: dict[str, str] = field(default_factory=dict)
    hooks: list[Any] = field(default_factory=list)
    max_rows_per_invocation: int = 64
    enabled: bool = True
    sampled_row_count: int = 0
    observed_row_count: int = 0
    _current_inputs: dict[tuple[int, int, tuple[int, ...], str], str] = field(default_factory=dict)
    _held_inputs: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name, module in named_matrix_modules(self.model):
            spec = self.preparation.transform_by_module.get(name)
            self.hooks.append(module.register_forward_pre_hook(self._hook(name, module, spec)))

    def begin_sample(self) -> None:
        self._current_inputs.clear()
        self._held_inputs.clear()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._current_inputs.clear()
            self._held_inputs.clear()

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._held_inputs.clear()

    def _hook(self, name: str, module: nn.Module, spec: TransformSpec | None):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if not self.enabled:
                return inputs
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            raw = inputs[0]
            key = (id(raw), raw.data_ptr(), tuple(raw.shape), spec.name if spec else "identity")
            self._held_inputs.append(raw)
            if key in self._current_inputs:
                canonical = self._current_inputs[key]
                prior = self.module_to_canonical.setdefault(name, canonical)
                if prior != canonical:
                    raise RuntimeError(f"diffusion calibration alias changed for {name}")
                return inputs
            canonical = name
            self._current_inputs[key] = canonical
            prior = self.module_to_canonical.setdefault(name, canonical)
            if prior != canonical:
                raise RuntimeError(f"diffusion calibration canonical changed for {name}")
            value = matrix_input(module, raw)
            value = apply_transform(value, spec) if spec is not None else value
            x = value.detach().float().reshape(-1, value.shape[-1])
            self.observed_row_count += x.shape[0]
            if self.max_rows_per_invocation > 0 and x.shape[0] > self.max_rows_per_invocation:
                indices = torch.linspace(
                    0,
                    x.shape[0] - 1,
                    steps=self.max_rows_per_invocation,
                    device=x.device,
                ).round().long()
                x = x.index_select(0, indices)
            self.sampled_row_count += x.shape[0]
            blocks_count = math.ceil(x.shape[-1] / 128)
            x = F.pad(x, (0, blocks_count * 128 - x.shape[-1]))
            blocks = x.reshape(-1, blocks_count, 128)
            covariance = torch.einsum("tki,tkj->kij", blocks, blocks)
            if canonical not in self.accumulators:
                self.accumulators[canonical] = covariance
                self.sample_counts[canonical] = blocks.shape[0]
            else:
                self.accumulators[canonical].add_(covariance)
                self.sample_counts[canonical] += blocks.shape[0]
            return inputs

        return hook

    def payload(self, *, calibration_samples: int, metadata: dict[str, Any]) -> dict[str, Any]:
        means128 = {
            name: value.detach().cpu().div(float(self.sample_counts[name])).contiguous()
            for name, value in self.accumulators.items()
        }
        means64 = {
            name: torch.stack((value[:, :64, :64], value[:, 64:, 64:]), dim=1).reshape(-1, 64, 64).contiguous()
            for name, value in means128.items()
        }
        names = {name for name, _ in named_matrix_modules(self.model)}
        if set(self.module_to_canonical) != names:
            missing = sorted(names - set(self.module_to_canonical))
            raise RuntimeError(f"diffusion calibration missed matrix modules: {missing}")
        return {
            "schema_version": 1, "calibration_samples": calibration_samples,
            "hessian_definition": "mean over observed spatial/token rows of block-diagonal K128 X^T X; exact K64 diagonals stored",
            "scale_group_size": 16, "format_region": "N8K64",
            "module_to_canonical": dict(sorted(self.module_to_canonical.items())),
            "sample_counts_by_canonical": dict(sorted(self.sample_counts.items())),
            "hessian_by_canonical": means64, "hessian_k128_by_canonical": means128,
            "activation_row_sampling": {
                "policy": "deterministic_evenly_spaced_rows_per_module_invocation",
                "max_rows_per_invocation": self.max_rows_per_invocation,
                "observed_rows_across_module_invocations": self.observed_row_count,
                "sampled_rows_across_module_invocations": self.sampled_row_count,
            },
            "metadata": metadata,
        }


def load_diffusion_calibration_by_module(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    canonical = value["hessian_by_canonical"]
    return ({module: {"hessian_k64": canonical[source]} for module, source in value["module_to_canonical"].items()}, value)


def prompt_conditioning(pipe: SanaPipeline, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    positive, positive_mask, negative, negative_mask = pipe.encode_prompt(
        prompt=prompt, negative_prompt="", do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=torch.device("cuda:0"), clean_caption=True,
        max_sequence_length=300,
    )
    return torch.cat((negative, positive), 0), torch.cat((negative_mask, positive_mask), 0)


@torch.inference_mode()
def denoise_from_conditioning(
    pipe: SanaPipeline,
    *,
    conditioning: torch.Tensor,
    attention_mask: torch.Tensor,
    initial_latents: torch.Tensor,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    capture_steps: set[int] | None = None,
    timestep_callback=None,
) -> dict[str, Any]:
    pipe.scheduler.set_timesteps(num_steps, device=torch.device("cuda:0"))
    latents = initial_latents.to(device="cuda:0", dtype=conditioning.dtype).clone()
    trajectory = [latents.detach().cpu()]
    predictions: dict[int, torch.Tensor] = {}
    inputs: dict[int, torch.Tensor] = {}
    start = time.monotonic()
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        model_input = torch.cat((latents, latents), 0).to(conditioning.dtype)
        expanded_timestep = timestep.expand(model_input.shape[0]).to(latents.dtype)
        if timestep_callback is not None:
            timestep_callback(step_index, int(timestep.item()), model_input, conditioning, attention_mask)
        output = pipe.transformer(
            model_input, encoder_hidden_states=conditioning, encoder_attention_mask=attention_mask,
            timestep=expanded_timestep, return_dict=False,
        )[0].float()
        unconditioned, text = output.chunk(2)
        guided = unconditioned + guidance_scale * (text - unconditioned)
        if capture_steps and step_index in capture_steps:
            inputs[step_index] = latents.detach().cpu()
            predictions[step_index] = guided.detach().cpu()
        latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]
        trajectory.append(latents.detach().cpu())
    return {
        "latents": latents, "trajectory": torch.stack(trajectory), "predictions": predictions,
        "inputs": inputs, "timesteps": [int(value.item()) for value in pipe.scheduler.timesteps],
        "wall_time_seconds": time.monotonic() - start,
    }


def proxy_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    return tensor_error_metrics(reference, actual)


__all__ = [
    "DiffusionActivationHooks", "DiffusionCovarianceCollector", "DiffusionPreparation",
    "SANA_MODEL_ID", "SANA_REVISION", "SANA_SNAPSHOT", "SANA_TEXT_ENCODER_REVISION",
    "SANA_TRANSFORMER_REVISION", "SANA_VAE_REVISION", "aggregate_activation_metrics",
    "aggregate_weight_metrics", "apply_sana_rotations", "denoise_from_conditioning",
    "diffusion_layer_index", "diffusion_module_type", "load_diffusion_calibration_by_module",
    "load_prompt_rows", "load_sana_pipeline", "named_matrix_modules", "prepare_sana_permutations",
    "prompt_conditioning", "proxy_metrics", "quantize_sana_weights", "sana_pipeline_provenance",
]
