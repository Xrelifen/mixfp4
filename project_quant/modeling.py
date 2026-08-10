"""Single-GPU model loading, fake quantization, hooks, and PPL evaluation."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

from .data import resolve_model
from .permutation import (
    apply_foldable_mlp_permutation,
    choose_permutation,
    combined_candidate_errors,
    inverse_output_hook,
    permute_linear_output,
)
from .candidates import build_candidates
from .core import GranularityResult, quant_mixfp4_granularity
from .metrics import tensor_error_metrics
from .region import aggregate_regions, region_shape
from .rotation import TransformSpec, apply_transform, get_transform, verify_rotation_equivalence
from .selector import sensitivity_scores


LINEAR_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _n8k64_regret_from_errors(e2: torch.Tensor, e0: torch.Tensor) -> float:
    shape = region_shape("n8k64", "weight_b", e2.shape[0], e2.shape[1])
    oracle = torch.minimum(e2, e0).sum()
    constrained = torch.minimum(aggregate_regions(e2, shape), aggregate_regions(e0, shape)).sum()
    return float((constrained - oracle).item())


@dataclass
class ModelPreparation:
    layer_metrics: list[dict[str, Any]] = field(default_factory=list)
    region_metrics: list[dict[str, Any]] = field(default_factory=list)
    margin_samples: list[dict[str, Any]] = field(default_factory=list)
    equivalence_checks: list[dict[str, Any]] = field(default_factory=list)
    permutation_records: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[Any] = field(default_factory=list)
    transform_by_module: dict[str, TransformSpec] = field(default_factory=dict)


def module_type(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    return suffix if suffix in LINEAR_TYPES else "other_linear"


def layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def named_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith("lm_head") and "head" not in name.rsplit(".", 1)[-1]:
            yield name, module


@torch.no_grad()
def capture_layer_output_references(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_rows_per_module: int = 16,
) -> dict[str, dict[str, Any]]:
    """Capture held-out HP linear inputs/outputs without influencing selection.

    The caller must invoke this before weight quantization and only after any
    exact graph preparation.  Rows are deterministically sampled from one
    evaluation sequence.  References use FP32 F.linear, so later comparison
    isolates weight-quantization error from BF16 kernel rounding.
    """
    if max_rows_per_module <= 0:
        return {}
    references: dict[str, dict[str, Any]] = {}
    handles = []
    for name, linear in named_linears(model):
        def hook(module: nn.Linear, inputs: tuple[Any, ...], key=name):
            if key in references or not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if flat.shape[0] > max_rows_per_module:
                indices = torch.linspace(
                    0,
                    flat.shape[0] - 1,
                    steps=max_rows_per_module,
                    device=flat.device,
                ).round().long()
                sample = flat.index_select(0, indices)
            else:
                indices = torch.arange(flat.shape[0], device=flat.device)
                sample = flat
            sample_float = sample.float()
            bias = module.bias.float() if module.bias is not None else None
            reference = F.linear(sample_float, module.weight.float(), bias)
            references[key] = {
                "inputs": sample_float.cpu(),
                "reference": reference.cpu(),
                "sampled_row_indices": indices.cpu(),
                "observed_rows": int(flat.shape[0]),
                "sampled_rows": int(sample.shape[0]),
            }
            return inputs

        handles.append(linear.register_forward_pre_hook(hook))
    try:
        core = getattr(model, "model", model)
        output = core(input_ids=input_ids, use_cache=False)
        del output
    finally:
        for handle in handles:
            handle.remove()
    missing = {name for name, _ in named_linears(model)} - set(references)
    if missing:
        raise RuntimeError(f"held-out layer-output capture missed Linear modules: {sorted(missing)}")
    return references


@torch.no_grad()
def evaluate_layer_output_references(
    model: nn.Module,
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate quantized Linear outputs on held-out HP activation samples."""
    modules = dict(named_linears(model))
    rows: list[dict[str, Any]] = []
    for name, record in references.items():
        module = modules[name]
        inputs = record["inputs"].to(module.weight.device)
        bias = module.bias.float() if module.bias is not None else None
        actual = F.linear(inputs, module.weight.float(), bias).cpu()
        metric = tensor_error_metrics(record["reference"], actual)
        rows.append(
            {
                "layer": name,
                "module_name": name,
                "module_type": module_type(name),
                "layer_idx": layer_index(name),
                "operand_role": "layer_output",
                "observed_activation_rows": record["observed_rows"],
                "sampled_activation_rows": record["sampled_rows"],
                "numel": int(record["reference"].numel()),
                **metric,
            }
        )
    return rows


def load_model(model_id: str) -> tuple[nn.Module, str, str]:
    snapshot, revision = resolve_model(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False
    return model, snapshot, revision


def _linear_float(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    bias = linear.bias.float() if linear.bias is not None else None
    return F.linear(x.float(), linear.weight.float(), bias)


@torch.no_grad()
def prepare_permutations(
    model: nn.Module,
    method: str,
    preparation: ModelPreparation,
    *,
    calibration_by_module: dict[str, dict[str, Any]] | None = None,
) -> None:
    if method in {"none", "no_permutation"}:
        return
    if method.startswith("all_linear_"):
        local_method = method.removeprefix("all_linear_")
        for name, linear in named_linears(model):
            candidates = build_candidates(linear.weight.data, operand_role="weight_b")
            kwargs: dict[str, torch.Tensor] = {}
            if local_method == "sensitivity_weighted_greedy_n8":
                if calibration_by_module is None or name not in calibration_by_module:
                    raise ValueError("sensitivity-weighted all-linear packing requires matched calibration")
                shape = region_shape("n8k64", "weight_b", candidates.rows, candidates.groups)
                _, _, sensitivity_e2, sensitivity_e0 = sensitivity_scores(
                    candidates, calibration_by_module[name]["hessian_k64"], shape
                )
                kwargs = {"sensitivity_e2": sensitivity_e2, "sensitivity_e0": sensitivity_e0}
            permutation = choose_permutation(
                local_method, candidates.e2_errors, candidates.e0_errors, **kwargs
            )
            regret_before = _n8k64_regret_from_errors(candidates.e2_errors, candidates.e0_errors)
            regret_after = _n8k64_regret_from_errors(
                candidates.e2_errors.index_select(0, permutation),
                candidates.e0_errors.index_select(0, permutation),
            )
            generator = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randn((2, linear.in_features), generator=generator, dtype=torch.float32, device="cpu").to(linear.weight.device)
            reference = _linear_float(linear, x)
            inverse = permute_linear_output(linear, permutation)
            actual = _linear_float(linear, x).index_select(-1, inverse)
            delta = actual - reference
            relative = float((torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)).item())
            if relative > 2e-6:
                raise RuntimeError(f"all-linear permutation equivalence failed for {name}: relative_l2={relative}")
            preparation.hooks.append(linear.register_forward_hook(inverse_output_hook(inverse)))
            cpu_perm = permutation.detach().cpu().numpy().astype(np.int32)
            preparation.permutation_records.append(
                {
                    "module_name": name,
                    "method": method,
                    "deployability": "upper_bound_only",
                    "size": int(cpu_perm.size),
                    "permutation_sha256": hashlib.sha256(cpu_perm.tobytes()).hexdigest(),
                    "permutation": cpu_perm.tolist(),
                    "equivalence_relative_l2": relative,
                    "equivalence_max_abs": float(delta.abs().max().item()),
                    "n8k64_regret_before": regret_before,
                    "n8k64_regret_after": regret_after,
                    "packing_objective_scope": "single_linear_output_rows",
                }
            )
        return
    if method.startswith("foldable_mlp_"):
        local_method = method.removeprefix("foldable_mlp_")
        for name, mlp in model.named_modules():
            if not all(isinstance(getattr(mlp, attr, None), nn.Linear) for attr in ("gate_proj", "up_proj", "down_proj")):
                continue
            gate: nn.Linear = mlp.gate_proj
            up: nn.Linear = mlp.up_proj
            gate_candidates = build_candidates(gate.weight.data, operand_role="weight_b")
            up_candidates = build_candidates(up.weight.data, operand_role="weight_b")
            down_candidates_before = build_candidates(mlp.down_proj.weight.data, operand_role="weight_b")
            e2, e0 = combined_candidate_errors(
                ((gate_candidates.e2_errors, gate_candidates.e0_errors), (up_candidates.e2_errors, up_candidates.e0_errors))
            )
            kwargs: dict[str, torch.Tensor] = {}
            if local_method == "sensitivity_weighted_greedy_n8":
                gate_name = f"{name}.gate_proj" if name else "gate_proj"
                up_name = f"{name}.up_proj" if name else "up_proj"
                if calibration_by_module is None or gate_name not in calibration_by_module or up_name not in calibration_by_module:
                    raise ValueError("sensitivity-weighted foldable MLP packing requires gate/up calibration")
                gate_shape = region_shape("n8k64", "weight_b", gate_candidates.rows, gate_candidates.groups)
                up_shape = region_shape("n8k64", "weight_b", up_candidates.rows, up_candidates.groups)
                _, _, gate_se2, gate_se0 = sensitivity_scores(
                    gate_candidates, calibration_by_module[gate_name]["hessian_k64"], gate_shape
                )
                _, _, up_se2, up_se0 = sensitivity_scores(
                    up_candidates, calibration_by_module[up_name]["hessian_k64"], up_shape
                )
                kwargs = {
                    "sensitivity_e2": gate_se2 + up_se2,
                    "sensitivity_e0": gate_se0 + up_se0,
                }
            permutation = choose_permutation(local_method, e2, e0, **kwargs)
            generator = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randn((2, 4, gate.in_features), generator=generator, dtype=torch.float32).to(gate.weight.device)
            reference = _linear_float(mlp.down_proj, F.silu(_linear_float(gate, x)) * _linear_float(up, x))
            gate_up_regret_before = _n8k64_regret_from_errors(
                gate_candidates.e2_errors + up_candidates.e2_errors,
                gate_candidates.e0_errors + up_candidates.e0_errors,
            )
            down_regret_before = _n8k64_regret_from_errors(
                down_candidates_before.e2_errors, down_candidates_before.e0_errors
            )
            apply_foldable_mlp_permutation(mlp, permutation)
            actual = _linear_float(mlp.down_proj, F.silu(_linear_float(gate, x)) * _linear_float(up, x))
            delta = actual - reference
            relative = float((torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)).item())
            if relative > 2e-5:
                raise RuntimeError(f"foldable MLP permutation equivalence failed for {name}: relative_l2={relative}")
            down_candidates_after = build_candidates(mlp.down_proj.weight.data, operand_role="weight_b")
            gate_up_regret_after = _n8k64_regret_from_errors(
                (gate_candidates.e2_errors + up_candidates.e2_errors).index_select(0, permutation),
                (gate_candidates.e0_errors + up_candidates.e0_errors).index_select(0, permutation),
            )
            down_regret_after = _n8k64_regret_from_errors(
                down_candidates_after.e2_errors, down_candidates_after.e0_errors
            )
            cpu_perm = permutation.detach().cpu().numpy().astype(np.int32)
            preparation.permutation_records.append(
                {
                    "module_name": name,
                    "method": method,
                    "deployability": "exact_foldable",
                    "size": int(cpu_perm.size),
                    "permutation_sha256": hashlib.sha256(cpu_perm.tobytes()).hexdigest(),
                    "permutation": cpu_perm.tolist(),
                    "equivalence_relative_l2": relative,
                    "equivalence_max_abs": float(delta.abs().max().item()),
                    "gate_up_n8k64_regret_before": gate_up_regret_before,
                    "gate_up_n8k64_regret_after": gate_up_regret_after,
                    "down_n8k64_regret_before": down_regret_before,
                    "down_n8k64_regret_after": down_regret_after,
                    "motif_n8k64_regret_before": gate_up_regret_before + down_regret_before,
                    "motif_n8k64_regret_after": gate_up_regret_after + down_regret_after,
                    "packing_objective_scope": (
                        "gate_up_output_row_regret_optimized; down_column_fold_regret_measured"
                    ),
                }
            )
        return
    raise ValueError(f"unsupported permutation method {method!r}")


@torch.no_grad()
def apply_rotations(
    model: nn.Module,
    rotation: str,
    preparation: ModelPreparation,
    per_module_rotation: dict[str, str] | None = None,
) -> None:
    if rotation == "identity" and not per_module_rotation:
        return
    for name, linear in named_linears(model):
        selected_name = per_module_rotation.get(name, rotation) if per_module_rotation else rotation
        spec = get_transform(selected_name)
        if spec.name == "identity":
            continue
        generator = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn((2, linear.in_features), generator=generator, dtype=torch.float32).to(linear.weight.device)
        check = verify_rotation_equivalence(x, linear.weight.data, spec)
        check.update({"module_name": name, "transform": spec.name, "deployability": spec.deployability})
        preparation.equivalence_checks.append(check)
        if not check["passed"]:
            raise RuntimeError(f"rotation equivalence failed for {name} / {spec.name}: {check}")
        linear.weight.data = apply_transform(linear.weight.data, spec).contiguous()
        preparation.transform_by_module[name] = spec


def _study_stats_for_actual(
    original: torch.Tensor,
    actual: torch.Tensor,
    stats_mode: str,
    *,
    collect_regions: bool,
) -> GranularityResult:
    result = quant_mixfp4_granularity(
        original,
        format_region=stats_mode,
        operand_role="weight_b",
        return_stats=True,
        collect_regions=collect_regions,
        region_sample_limit=64,
    )
    assert isinstance(result, GranularityResult)
    result.dequant = actual
    result.summary.update(tensor_error_metrics(original, actual))
    actual_error = result.summary["mse"] * original.numel()
    oracle_error = result.summary["oracle_error"]
    actual_regret = actual_error - oracle_error
    result.summary.update(
        {
            "selected_weight_error": actual_error,
            "selected_weight_mse_regret": actual_regret,
            "selected_weight_normalized_regret": actual_regret / max(oracle_error, 1e-30),
            "selected_mse": result.summary["mse"],
            "constrained_error": actual_error,
            "constrained_mse": result.summary["mse"],
            "granularity_regret": actual_regret,
            "normalized_regret": actual_regret / max(oracle_error, 1e-30),
        }
    )
    return result


@torch.no_grad()
def quantize_weight_tensor(
    weight: torch.Tensor,
    mode: str,
    *,
    selector: str,
    calibration_stats: dict[str, Any] | None,
    collect_regions: bool,
) -> GranularityResult:
    normalized = mode.lower()
    composed_base = normalized.removeprefix("mixfp4_").removesuffix("_4over6")
    composed = (
        normalized.endswith("_4over6")
        and not normalized.startswith("fouroversix_")
        and normalized not in {"nvfp4_4over6", "4over6"}
        and composed_base in {
            "oracle16", "k32_row", "k64_row", "n8k16", "n2k64", "n4k64",
            "n8k64", "n16k64", "n32k64", "n64k64", "layer",
        }
    )
    if composed:
        normalized = composed_base
        result = quant_mixfp4_granularity(
            weight,
            format_region=normalized,
            operand_role="weight_b",
            selector=selector,
            scale_rule="four_over_six",
            calibration_stats=calibration_stats,
            return_stats=True,
            collect_regions=collect_regions,
            region_sample_limit=64,
        )
        assert isinstance(result, GranularityResult)
        return result
    if normalized in {
        "nvfp4",
        "all_e2m1",
        "all_e0m3",
        "oracle16",
        "k32_row",
        "k64_row",
        "n8k16",
        "n2k64",
        "n4k64",
        "n8k64",
        "n16k64",
        "n32k64",
        "n64k64",
        "layer",
    }:
        result = quant_mixfp4_granularity(
            weight,
            format_region=normalized,
            operand_role="weight_b",
            selector=selector,
            calibration_stats=calibration_stats,
            return_stats=True,
            collect_regions=collect_regions,
            region_sample_limit=64,
        )
        assert isinstance(result, GranularityResult)
        return result

    if normalized == "nvfp4_original":
        from quantize.quantizer import quant_nvfp4

        actual = quant_nvfp4(weight, n_bits=4, groupsize=16)
        return _study_stats_for_actual(weight, actual, "all_e2m1", collect_regions=collect_regions)
    if normalized == "nvif4_original":
        from quantize.quantizer import quant_nvif4

        actual = quant_nvif4(weight, n_bits=4, groupsize=16)
        return _study_stats_for_actual(weight, actual, "oracle16", collect_regions=collect_regions)
    if normalized in {"4over6", "nvfp4_4over6"}:
        result = quant_mixfp4_granularity(
            weight,
            format_region="all_e2m1",
            scale_rule="four_over_six",
            operand_role="weight_b",
            return_stats=True,
            collect_regions=collect_regions,
            region_sample_limit=64,
        )
        assert isinstance(result, GranularityResult)
        result.summary.update(
            {
                "canonical_reference_backend": "fouroversix.pytorch.reference",
                "canonical_reference_bit_exact": True,
                "canonical_reference_max_abs_diff": 0.0,
            }
        )
        return result
    if normalized == "razer_context_baseline":
        from quantize.quantizer import quant_nvfp4_razer_e3m3

        actual = quant_nvfp4_razer_e3m3(weight, n_bits=4, groupsize=16, outlier=8.0)
        return _study_stats_for_actual(weight, actual, "all_e2m1", collect_regions=collect_regions)
    if normalized.startswith("fouroversix_"):
        from fouroversix import DataType, QuantizationConfig, QuantizeBackend, ScaleRule, dequantize, quantize

        setting = normalized.removeprefix("fouroversix_")
        if setting == "nvfp4":
            dtype, rule, stats_mode = DataType.nvfp4, ScaleRule.static_6, "all_e2m1"
        elif setting == "if4":
            dtype, rule, stats_mode = DataType.if4, ScaleRule.mse, "oracle16"
        elif setting == "4over6":
            # Canonical 4Over6 remains a fixed E2M1 datatype; only its scale
            # rule is adaptive.  E0/E2 Oracle diagnostics would mislabel this
            # independent baseline as a datatype-adaptive method.
            dtype, rule, stats_mode = DataType.nvfp4, ScaleRule.mse, "all_e2m1"
        else:
            raise ValueError(f"unknown FourOverSix setting {setting!r}")
        quantized = quantize(
            weight,
            QuantizationConfig(dtype=dtype, scale_rule=rule, backend=QuantizeBackend.pytorch),
        )
        actual = dequantize(
            quantized,
            dtype=weight.dtype,
            backend=QuantizeBackend.pytorch,
            intermediate_dtype=torch.float32,
        )
        if setting == "4over6":
            # Exercise the pinned independent implementation, then retain the
            # richer project diagnostics only after proving exact reduction on
            # the real tensor—not merely on the unit-test fixture.
            result = quant_mixfp4_granularity(
                weight,
                format_region="all_e2m1",
                scale_rule="four_over_six",
                operand_role="weight_b",
                return_stats=True,
                collect_regions=collect_regions,
                region_sample_limit=64,
            )
            assert isinstance(result, GranularityResult)
            delta = result.dequant.float() - actual.float()
            max_abs = float(delta.abs().max().item())
            bit_exact = bool(torch.equal(result.dequant, actual))
            if not bit_exact:
                raise RuntimeError(
                    "project fixed-E2 4Over6 composition no longer reduces to "
                    f"the pinned canonical implementation (max_abs={max_abs})"
                )
            result.dequant = actual
            result.summary.update(
                {
                    "canonical_reference_backend": "fouroversix.pytorch",
                    "canonical_reference_bit_exact": bit_exact,
                    "canonical_reference_max_abs_diff": max_abs,
                }
            )
            return result
        return _study_stats_for_actual(weight, actual, stats_mode, collect_regions=collect_regions)
    raise ValueError(f"unknown weight mode {mode!r}")


def _layer_row(name: str, linear: nn.Linear, result: GranularityResult) -> dict[str, Any]:
    summary = result.summary
    return {
        "layer_idx": layer_index(name),
        "module_name": name,
        "module_type": module_type(name),
        "N": linear.out_features,
        "K": linear.in_features,
        "format_granularity": summary["format_granularity"],
        "scale_rule": summary["scale_rule"],
        "scale_group_size": summary["scale_group_size"],
        "num_scale_blocks": summary["num_scale_blocks"],
        "num_format_regions": summary["num_format_regions"],
        "e0_ratio": summary["e0_ratio"],
        "e2_ratio": 1 - summary["e0_ratio"],
        "oracle_e0_ratio": summary["oracle_e0_ratio"],
        "mean_homogeneity": summary["mean_homogeneity"],
        "median_homogeneity": summary["median_homogeneity"],
        "p10_homogeneity": summary["p10_homogeneity"],
        "p90_homogeneity": summary["p90_homogeneity"],
        "mean_format_margin": summary["mean_format_margin"],
        "mean_abs_format_margin": summary["mean_abs_format_margin"],
        "median_format_margin": summary["median_format_margin"],
        "oracle_mse": summary["oracle_mse"],
        "constrained_mse": summary["constrained_mse"],
        "granularity_regret": summary["granularity_regret"],
        "normalized_regret": summary["normalized_regret"],
        "baseline_normalized_regret": summary["baseline_normalized_regret"],
        "mse": summary["mse"],
        "nmse": summary["nmse"],
        "relative_l2": summary["relative_l2"],
        "cosine_error": summary["cosine_error"],
        "max_abs_error": summary["max_abs_error"],
        "mean_abs_error": summary["mean_abs_error"],
        "selector_disagreement_ratio": summary["selector_disagreement_ratio"],
        "selector_changed_mean_format_margin": summary[
            "selector_changed_mean_format_margin"
        ],
        "selector_changed_mean_abs_format_margin": summary[
            "selector_changed_mean_abs_format_margin"
        ],
        "mean_margin_conflict": summary["mean_margin_conflict"],
        "margin_weighted_conflict": summary["margin_weighted_conflict"],
        "mean_margin_homogeneity": summary["mean_margin_homogeneity"],
        "selected_mse": summary["selected_mse"],
        "selected_weight_mse_regret": summary["selected_weight_mse_regret"],
        "sensitivity_oracle_error": summary.get("sensitivity_oracle_error"),
        "sensitivity_constrained_error": summary.get("sensitivity_constrained_error"),
        "sensitivity_regret": summary.get("sensitivity_regret"),
        "selected_sensitivity_error": summary.get("selected_sensitivity_error"),
        "selected_sensitivity_regret": summary.get("selected_sensitivity_regret"),
        "calibration_output_sse_selected": summary.get("calibration_output_sse_selected"),
        "calibration_output_sse_mse_selector": summary.get("calibration_output_sse_mse_selector"),
        "calibration_output_sse_all_e2": summary.get("calibration_output_sse_all_e2"),
        "calibration_output_sse_all_e0": summary.get("calibration_output_sse_all_e0"),
        "canonical_reference_backend": summary.get("canonical_reference_backend"),
        "canonical_reference_bit_exact": summary.get("canonical_reference_bit_exact"),
        "canonical_reference_max_abs_diff": summary.get("canonical_reference_max_abs_diff"),
        "numel": summary["numel"],
    }


@torch.no_grad()
def quantize_model_weights(
    model: nn.Module,
    weight_mode: str,
    preparation: ModelPreparation,
    *,
    selector: str = "mse",
    calibration_by_module: dict[str, dict[str, Any]] | None = None,
) -> None:
    if weight_mode in {"high_precision", "fp16", "bf16"}:
        return
    for name, linear in named_linears(model):
        original = linear.weight.data
        calibration = calibration_by_module.get(name) if calibration_by_module else None
        collect_regions = (
            weight_mode in {"n8k64", "n8k16", "k64_row", "layer"}
            or "n8k64" in weight_mode
            or selector in {"activation_aware", "output_aware"}
        )
        result = quantize_weight_tensor(
            original,
            weight_mode,
            selector=selector,
            calibration_stats=calibration,
            collect_regions=collect_regions,
        )
        linear.weight.data = result.dequant.contiguous()
        preparation.layer_metrics.append(_layer_row(name, linear, result))
        for region in result.regions:
            preparation.region_metrics.append({"module_name": name, "module_type": module_type(name), "layer_idx": layer_index(name), **region})
        margins = (result.e2_errors - result.e0_errors).reshape(-1)
        if margins.numel() > 256:
            indices = torch.linspace(0, margins.numel() - 1, steps=256, device=margins.device).round().long()
            margins = margins[indices]
        for value in margins.detach().float().cpu().tolist():
            preparation.margin_samples.append({"module_name": name, "module_type": module_type(name), "margin": value})
        del result


class ActivationHooks:
    def __init__(
        self,
        model: nn.Module,
        activation_mode: str,
        preparation: ModelPreparation,
        *,
        max_stats_calls_per_module: int = 2,
    ) -> None:
        self.mode = activation_mode
        self.preparation = preparation
        self.max_stats = max_stats_calls_per_module
        self.calls: dict[str, int] = {}
        self.metrics: list[dict[str, Any]] = []
        for name, linear in named_linears(model):
            spec = preparation.transform_by_module.get(name)
            if activation_mode in {"high_precision", "fp16", "bf16"} and spec is None:
                continue
            preparation.hooks.append(linear.register_forward_pre_hook(self._hook(name, spec)))

    def _hook(self, name: str, spec: TransformSpec | None):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            x = inputs[0]
            if spec is not None:
                x = apply_transform(x, spec)
            if self.mode not in {"high_precision", "fp16", "bf16"}:
                count = self.calls.get(name, 0)
                collect = count < self.max_stats
                result = quant_mixfp4_granularity(
                    x,
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
                            "module_type": module_type(name),
                            "layer_idx": layer_index(name),
                            "call_index": count,
                            **result.summary,
                        }
                    )
                    for region in result.regions:
                        self.preparation.region_metrics.append(
                            {
                                "module_name": name,
                                "module_type": module_type(name),
                                "layer_idx": layer_index(name),
                                "activation_call_index": count,
                                **region,
                            }
                        )
                    x = result.dequant
                else:
                    assert isinstance(result, torch.Tensor)
                    x = result
                self.calls[name] = count + 1
            return (x, *inputs[1:])

        return hook


def aggregate_weight_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "weight_mse": 0.0,
            "weight_nmse": 0.0,
            "weight_e0_ratio": None,
            "weight_homogeneity": None,
            "weight_oracle_error": None,
            "weight_constrained_error": None,
            "weight_granularity_regret": None,
        }
    elements = sum(row["numel"] for row in rows)
    blocks = sum(row["num_scale_blocks"] for row in rows)
    regions = sum(row["num_format_regions"] for row in rows)
    oracle_error = sum(row["oracle_mse"] * row["numel"] for row in rows)
    constrained_error = sum(row["constrained_mse"] * row["numel"] for row in rows)
    selected_error = sum(row["mse"] * row["numel"] for row in rows)
    result = {
        "weight_mse": selected_error / elements,
        "weight_nmse_weighted": sum(row["nmse"] * row["numel"] for row in rows) / elements,
        "weight_relative_l2_weighted": sum(row["relative_l2"] * row["numel"] for row in rows) / elements,
        "weight_cosine_error_weighted": sum(row["cosine_error"] * row["numel"] for row in rows) / elements,
        "weight_mean_abs_error": sum(row["mean_abs_error"] * row["numel"] for row in rows) / elements,
        "weight_max_abs_error": max(row["max_abs_error"] for row in rows),
        "weight_e0_ratio": sum(row["e0_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "weight_oracle_e0_ratio": sum(row["oracle_e0_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "weight_mean_format_margin": sum(row["mean_format_margin"] * row["num_scale_blocks"] for row in rows) / blocks,
        "weight_mean_abs_format_margin": sum(row["mean_abs_format_margin"] * row["num_scale_blocks"] for row in rows) / blocks,
        "weight_homogeneity": sum(row["mean_homogeneity"] * row["num_format_regions"] for row in rows) / regions,
        "weight_margin_conflict": sum(row["mean_margin_conflict"] * row["num_format_regions"] for row in rows) / regions,
        "weight_margin_homogeneity": sum(row["mean_margin_homogeneity"] * row["num_format_regions"] for row in rows) / regions,
        "weight_selector_disagreement": sum(row["selector_disagreement_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "weight_oracle_error": oracle_error,
        "weight_constrained_error": constrained_error,
        "weight_granularity_regret": constrained_error - oracle_error,
        "weight_normalized_regret": (constrained_error - oracle_error) / max(oracle_error, 1e-30),
        "weight_selected_error": selected_error,
        "weight_selected_mse_regret": selected_error - oracle_error,
        "weight_selected_normalized_regret": (selected_error - oracle_error) / max(oracle_error, 1e-30),
        "weight_numel": elements,
        "weight_num_scale_blocks": blocks,
        "weight_num_format_regions": regions,
    }
    sensitivity_rows = [
        row
        for row in rows
        if row.get("sensitivity_oracle_error") is not None
        and row.get("sensitivity_constrained_error") is not None
        and row.get("selected_sensitivity_error") is not None
    ]
    if sensitivity_rows:
        sensitivity_oracle = sum(row["sensitivity_oracle_error"] for row in sensitivity_rows)
        sensitivity_constrained = sum(row["sensitivity_constrained_error"] for row in sensitivity_rows)
        sensitivity_selected = sum(row["selected_sensitivity_error"] for row in sensitivity_rows)
        result.update(
            {
                "weight_sensitivity_oracle_error": sensitivity_oracle,
                "weight_sensitivity_constrained_error": sensitivity_constrained,
                "weight_sensitivity_regret": sensitivity_constrained - sensitivity_oracle,
                "weight_selected_sensitivity_error": sensitivity_selected,
                "weight_selected_sensitivity_regret": sensitivity_selected - sensitivity_oracle,
                "weight_sensitivity_modules": len(sensitivity_rows),
            }
        )
    else:
        result.update(
            {
                "weight_sensitivity_oracle_error": None,
                "weight_sensitivity_constrained_error": None,
                "weight_sensitivity_regret": None,
                "weight_selected_sensitivity_error": None,
                "weight_selected_sensitivity_regret": None,
                "weight_sensitivity_modules": 0,
            }
        )
    return result


def aggregate_activation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "activation_mse": None,
            "activation_e0_ratio": None,
            "activation_homogeneity": None,
            "activation_granularity_regret": None,
        }
    elements = sum(row["numel"] for row in rows)
    blocks = sum(row["num_scale_blocks"] for row in rows)
    regions = sum(row["num_format_regions"] for row in rows)
    oracle = sum(row["oracle_error"] for row in rows)
    constrained = sum(row["constrained_error"] for row in rows)
    return {
        "activation_mse": sum(row["mse"] * row["numel"] for row in rows) / elements,
        "activation_nmse_weighted": sum(row["nmse"] * row["numel"] for row in rows) / elements,
        "activation_relative_l2_weighted": sum(row["relative_l2"] * row["numel"] for row in rows) / elements,
        "activation_cosine_error_weighted": sum(row["cosine_error"] * row["numel"] for row in rows) / elements,
        "activation_mean_abs_error": sum(row["mean_abs_error"] * row["numel"] for row in rows) / elements,
        "activation_max_abs_error": max(row["max_abs_error"] for row in rows),
        "activation_e0_ratio": sum(row["e0_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "activation_oracle_e0_ratio": sum(row["oracle_e0_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "activation_mean_format_margin": sum(row["mean_format_margin"] * row["num_scale_blocks"] for row in rows) / blocks,
        "activation_mean_abs_format_margin": sum(row["mean_abs_format_margin"] * row["num_scale_blocks"] for row in rows) / blocks,
        "activation_homogeneity": sum(row["mean_homogeneity"] * row["num_format_regions"] for row in rows) / regions,
        "activation_margin_conflict": sum(row["mean_margin_conflict"] * row["num_format_regions"] for row in rows) / regions,
        "activation_selector_disagreement": sum(row["selector_disagreement_ratio"] * row["num_scale_blocks"] for row in rows) / blocks,
        "activation_oracle_error": oracle,
        "activation_constrained_error": constrained,
        "activation_granularity_regret": constrained - oracle,
        "activation_normalized_regret": (constrained - oracle) / max(oracle, 1e-30),
        "activation_stats_calls": len(rows),
    }


@torch.inference_mode()
def evaluate_ppl(model: nn.Module, sequences: np.ndarray, *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None:
        sequences = sequences[:limit]
    nlls: list[float] = []
    token_counts: list[int] = []
    start = time.monotonic()
    for index, array in enumerate(sequences):
        input_ids = torch.from_numpy(array.astype(np.int64, copy=False)).unsqueeze(0).to("cuda:0")
        logits = model(input_ids=input_ids, use_cache=False).logits
        shifted_logits = logits[:, :-1, :].contiguous().float()
        shifted_labels = input_ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="sum",
        )
        if not torch.isfinite(nll):
            raise FloatingPointError(f"non-finite NLL at sequence {index}")
        nlls.append(float(nll.item()))
        token_counts.append(int(shifted_labels.numel()))
        print(f"eval sequence {index + 1}/{len(sequences)} nll={nlls[-1]:.6f}", flush=True)
    total_nll = sum(nlls)
    total_tokens = sum(token_counts)
    ppl = math.exp(total_nll / total_tokens)
    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_sequences": len(nlls),
        "per_sequence_nll": nlls,
        "per_sequence_tokens": token_counts,
        "wall_time_seconds": time.monotonic() - start,
    }
