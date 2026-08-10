#!/usr/bin/env python3
"""Measure SANA activation preference and coarse regret across denoising time."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.adapters.deepcompressor_diffusion import (  # noqa: E402
    DiffusionActivationHooks,
    DiffusionPreparation,
    SANA_MODEL_ID,
    SANA_REVISION,
    diffusion_layer_index,
    diffusion_module_type,
    load_sana_pipeline,
    matrix_input,
    matrix_weight,
    named_matrix_modules,
    proxy_metrics,
    quantize_sana_weights,
)
from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402
from project_quant.core import GranularityResult, quant_mixfp4_granularity  # noqa: E402


DEFAULT_REFERENCE = "artifacts/03_phase_a/diffusion/reference/current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--reference-pointer", default=DEFAULT_REFERENCE)
    parser.add_argument("--prompt-limit", type=int, default=2)
    parser.add_argument("--max-activation-rows", type=int, default=64)
    parser.add_argument("--weight-mode", default="n8k64")
    parser.add_argument("--activation-mode", default="m16k64")
    parser.add_argument(
        "--module-fragments",
        nargs="+",
        default=[".attn1.to_q", ".ff.conv_inverted"],
        help="representative matrix motifs sampled in every transformer block",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sample_rows(value: torch.Tensor, limit: int) -> torch.Tensor:
    flat = value.detach().reshape(-1, value.shape[-1])
    if limit > 0 and flat.shape[0] > limit:
        indices = torch.linspace(0, flat.shape[0] - 1, steps=limit, device=flat.device).round().long()
        flat = flat.index_select(0, indices)
    return flat.contiguous()


def covariance_k64(value: torch.Tensor) -> torch.Tensor:
    k64 = math.ceil(value.shape[-1] / 64)
    padded = F.pad(value.float(), (0, k64 * 64 - value.shape[-1]))
    blocks = padded.reshape(-1, k64, 64)
    return torch.einsum("tki,tkj->kij", blocks, blocks) / max(blocks.shape[0], 1)


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    count = min(left.numel(), right.numel())
    if count < 2:
        return 1.0
    x = left[:count].double()
    y = right[:count].double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator <= 1e-30:
        return 1.0 if torch.allclose(left[:count], right[:count]) else 0.0
    return float((torch.dot(x, y) / denominator).item())


@torch.inference_mode()
def guided_forward(pipe, case: dict[str, Any], step_index: int) -> torch.Tensor:
    conditioning = case["conditioning"].to("cuda:0")
    attention_mask = case["attention_mask"].to("cuda:0")
    latent = case["teacher_forced_inputs"][step_index].to("cuda:0")
    model_input = torch.cat((latent, latent), 0).to(conditioning.dtype)
    timestep = torch.tensor(case["timesteps"][step_index], device="cuda:0", dtype=latent.dtype).expand(2)
    output = pipe.transformer(
        model_input,
        encoder_hidden_states=conditioning,
        encoder_attention_mask=attention_mask,
        timestep=timestep,
        return_dict=False,
    )[0].float()
    unconditioned, text = output.chunk(2)
    return unconditioned + float(case["guidance_scale"]) * (text - unconditioned)


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config.update({"domain": "diffusion", "model": SANA_MODEL_ID})
    ledger = ExperimentLedger(args.experiment_id, "phase_b_timestep", config)
    try:
        pointer = json.loads(resolve(args.reference_pointer).read_text(encoding="utf-8"))
        records = json.loads(resolve(pointer["reference_cases"]).read_text(encoding="utf-8"))[: args.prompt_limit]
        cases = [torch.load(resolve(row["path"]), map_location="cpu", weights_only=True) for row in records]
        if not cases:
            raise ValueError("no SANA reference cases selected")
        pipe = load_sana_pipeline(with_vae=False)
        selected = {
            name: module
            for name, module in named_matrix_modules(pipe.transformer)
            if any(fragment in name for fragment in args.module_fragments)
        }
        if not selected:
            raise ValueError("module fragments selected no SANA matrix modules")

        rows: list[dict[str, Any]] = []
        margins_rows: list[dict[str, Any]] = []
        reference_ids: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        hp_outputs: dict[tuple[int, int], torch.Tensor] = {}
        for prompt_index, case in enumerate(cases):
            prompt_id = case["prompt_record"]["prompt_id"]
            for step_index in case["capture_steps"]:
                captured: dict[str, torch.Tensor] = {}
                handles = []
                for name, module in selected.items():
                    def hook(current_module, inputs, key=name):
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            captured[key] = sample_rows(
                                matrix_input(current_module, inputs[0]), args.max_activation_rows
                            ).detach().cpu()

                    handles.append(module.register_forward_pre_hook(hook))
                try:
                    hp_outputs[(prompt_index, step_index)] = guided_forward(pipe, case, step_index).detach().cpu()
                finally:
                    for handle in handles:
                        handle.remove()

                for name, module in selected.items():
                    activation = captured[name].to("cuda:0")
                    activation_result = quant_mixfp4_granularity(
                        activation,
                        format_region="m16k64",
                        operand_role="activation_a",
                        return_stats=True,
                        collect_regions=False,
                    )
                    assert isinstance(activation_result, GranularityResult)
                    hessian = covariance_k64(activation)
                    weight_result = quant_mixfp4_granularity(
                        matrix_weight(module),
                        format_region="n8k64",
                        operand_role="weight_b",
                        selector="output_aware",
                        calibration_stats={"hessian_k64": hessian},
                        return_stats=True,
                        collect_regions=False,
                    )
                    assert isinstance(weight_result, GranularityResult)
                    margin = (activation_result.e2_errors - activation_result.e0_errors).reshape(-1).detach().cpu()
                    key = (prompt_id, name)
                    if key not in reference_ids:
                        reference_ids[key] = (
                            activation_result.oracle_format_ids.detach().cpu(),
                            activation_result.format_ids.detach().cpu(),
                            margin,
                        )
                    ref_oracle, ref_coarse, ref_margin = reference_ids[key]
                    oracle_ids = activation_result.oracle_format_ids.detach().cpu().reshape(-1)
                    coarse_ids = activation_result.format_ids.detach().cpu().reshape(-1)
                    oracle_count = min(ref_oracle.numel(), oracle_ids.numel())
                    coarse_count = min(ref_coarse.numel(), coarse_ids.numel())
                    row = {
                        "experiment_id": args.experiment_id,
                        "model": SANA_MODEL_ID,
                        "promptset": pointer["reference_cases"],
                        "prompt_id": prompt_id,
                        "timestep": case["timesteps"][step_index],
                        "timestep_index": step_index,
                        "layer": diffusion_layer_index(name),
                        "module": name,
                        "module_type": diffusion_module_type(name),
                        "e0_ratio": activation_result.summary["oracle_e0_ratio"],
                        "coarse_e0_ratio": activation_result.summary["e0_ratio"],
                        "selector_agreement_vs_reference_step": float(
                            (ref_oracle.reshape(-1)[:oracle_count] == oracle_ids[:oracle_count]).float().mean().item()
                        ),
                        "coarse_selector_agreement_vs_reference_step": float(
                            (ref_coarse.reshape(-1)[:coarse_count] == coarse_ids[:coarse_count]).float().mean().item()
                        ),
                        "margin_correlation_vs_reference_step": correlation(ref_margin, margin),
                        "mean_margin": activation_result.summary["mean_format_margin"],
                        "mean_abs_margin": activation_result.summary["mean_abs_format_margin"],
                        "mean_homogeneity": activation_result.summary["mean_homogeneity"],
                        "mean_margin_conflict": activation_result.summary["mean_margin_conflict"],
                        "granularity_regret": activation_result.summary["granularity_regret"],
                        "normalized_regret": activation_result.summary["normalized_regret"],
                        "sensitivity_regret": weight_result.summary.get("sensitivity_regret"),
                        "sensitivity_normalized_regret": weight_result.summary.get("sensitivity_normalized_regret"),
                        "weight_selector_disagreement": weight_result.summary["selector_disagreement_ratio"],
                        "proxy_nmse": None,
                    }
                    rows.append(row)
                    sampled_margin = margin
                    if sampled_margin.numel() > 64:
                        indices = torch.linspace(0, sampled_margin.numel() - 1, 64).round().long()
                        sampled_margin = sampled_margin.index_select(0, indices)
                    margins_rows.extend(
                        {
                            "experiment_id": args.experiment_id,
                            "prompt_id": prompt_id,
                            "timestep_index": step_index,
                            "timestep": case["timesteps"][step_index],
                            "module": name,
                            "signed_margin": float(value),
                        }
                        for value in sampled_margin.tolist()
                    )
                    del activation_result, weight_result, activation, hessian
                print(f"timestep preference prompt={prompt_id} step={step_index}", flush=True)

        preparation = DiffusionPreparation()
        quantize_sana_weights(pipe.transformer, args.weight_mode, preparation)
        activation_hooks = DiffusionActivationHooks(pipe.transformer, args.activation_mode, preparation)
        proxy_by_key: dict[tuple[str, int], float] = {}
        for prompt_index, case in enumerate(cases):
            prompt_id = case["prompt_record"]["prompt_id"]
            for step_index in case["capture_steps"]:
                actual = guided_forward(pipe, case, step_index).detach().cpu()
                metric = proxy_metrics(hp_outputs[(prompt_index, step_index)], actual)
                proxy_by_key[(prompt_id, step_index)] = metric["nmse"]
        for row in rows:
            row["proxy_nmse"] = proxy_by_key[(row["prompt_id"], row["timestep_index"])]

        frame = pd.DataFrame(rows)
        classifications: dict[str, str] = {}
        for name, group in frame.groupby("module"):
            mean_agreement = float(group["selector_agreement_vs_reference_step"].mean())
            e0_range = float(group["e0_ratio"].max() - group["e0_ratio"].min())
            conflict_range = float(group["mean_margin_conflict"].max() - group["mean_margin_conflict"].min())
            if mean_agreement >= 0.90 and e0_range <= 0.10:
                label = "stable_format_preference"
            elif mean_agreement >= 0.75 and conflict_range <= 0.10:
                label = "stable_coarse_locality"
            else:
                label = "unstable_timestep_dependent"
            classifications[name] = label
        frame["classification"] = frame["module"].map(classifications)
        frame.to_csv(ledger.directory / "timestep_metrics.csv", index=False)
        pd.DataFrame(margins_rows).to_csv(ledger.directory / "timestep_margin_samples.csv", index=False)
        pd.DataFrame(preparation.layer_metrics).to_csv(ledger.directory / "per_layer_metrics.csv", index=False)
        pd.DataFrame(preparation.region_metrics).to_csv(ledger.directory / "format_region_metrics.csv", index=False)
        pd.DataFrame(activation_hooks.metrics).to_csv(ledger.directory / "activation_metrics.csv", index=False)
        atomic_json(ledger.directory / "layer_classifications.json", classifications)
        class_counts = pd.Series(classifications).value_counts().to_dict()
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_timestep",
            "domain": "diffusion",
            "status": "completed",
            "model": SANA_MODEL_ID,
            "model_revision": SANA_REVISION,
            "dataset_or_promptset": pointer["reference_cases"],
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "quantization_mode": f"W={args.weight_mode}/A={args.activation_mode}",
            "scale_group_size": 16,
            "selector": "mse",
            "permutation": "none",
            "rotation": "identity",
            "num_prompts": len(cases),
            "timestep_indices": sorted(set(frame["timestep_index"].tolist())),
            "timesteps": sorted(set(frame["timestep"].tolist()), reverse=True),
            "selected_module_fragments": args.module_fragments,
            "num_selected_modules": len(selected),
            "classification_counts": class_counts,
            "mean_selector_agreement": float(frame["selector_agreement_vs_reference_step"].mean()),
            "mean_margin_correlation": float(frame["margin_correlation_vs_reference_step"].mean()),
            "mean_proxy_nmse": float(frame["proxy_nmse"].mean()),
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
