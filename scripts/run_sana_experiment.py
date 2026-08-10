#!/usr/bin/env python3
"""Run one SANA proxy/trajectory experiment using fixed HP reference cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.adapters.deepcompressor_diffusion import (  # noqa: E402
    DiffusionActivationHooks,
    DiffusionPreparation,
    SANA_MODEL_ID,
    SANA_REVISION,
    aggregate_activation_metrics,
    aggregate_weight_metrics,
    apply_sana_rotations,
    denoise_from_conditioning,
    diffusion_layer_index,
    load_diffusion_calibration_by_module,
    load_sana_pipeline,
    prepare_sana_permutations,
    proxy_metrics,
    quantize_sana_weights,
    sana_pipeline_provenance,
)
from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402


DEFAULT_REFERENCE = "artifacts/03_phase_a/diffusion/reference/current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--weight-mode", default="high_precision")
    parser.add_argument("--activation-mode", default="high_precision")
    parser.add_argument("--selector", choices=("mse", "activation_aware", "output_aware"), default="mse")
    parser.add_argument("--calibration-file")
    parser.add_argument("--calibration-size", type=int, default=0)
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    parser.add_argument("--reference-pointer", default=DEFAULT_REFERENCE)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--layer-error-prompts", type=int, default=2)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def capture_block_outputs(pipe, case: dict[str, Any], step_index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values: dict[str, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(pipe.transformer.transformer_blocks):
        name = f"transformer_blocks.{index}"

        def hook(_module, _inputs, output, key=name):
            tensor = output[0] if isinstance(output, tuple) else output
            values[key] = tensor.detach().float().cpu()

        handles.append(block.register_forward_hook(hook))
    try:
        output = guided_forward(pipe, case, step_index).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return output, values


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {"mse": None, "nmse": None, "relative_l2": None, "cosine_error": None, "mean_abs_error": None, "max_abs_error": None}
    return {
        key: (max(row[key] for row in rows) if key == "max_abs_error" else sum(row[key] for row in rows) / len(rows))
        for key in ("mse", "nmse", "relative_l2", "cosine_error", "mean_abs_error", "max_abs_error")
    }


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config.update({"domain": "diffusion", "model": SANA_MODEL_ID})
    ledger = ExperimentLedger(args.experiment_id, args.phase, config)
    try:
        pointer = json.loads(resolve(args.reference_pointer).read_text())
        case_records = json.loads(resolve(pointer["reference_cases"]).read_text())
        if args.prompt_limit is not None:
            case_records = case_records[: args.prompt_limit]
        cases = [torch.load(resolve(record["path"]), map_location="cpu", weights_only=True) for record in case_records]
        for case in cases:
            if case.get("trajectory_storage_dtype") != "float32" or case["trajectory"].dtype != torch.float32:
                raise ValueError(
                    "SANA reference trajectory must be stored in FP32; the earlier BF16 "
                    "reference has a non-zero high-precision trajectory error floor"
                )
        pipe = load_sana_pipeline(with_vae=False)
        pipeline_provenance = sana_pipeline_provenance(pipe)
        atomic_json(ledger.directory / "pipeline_provenance.json", pipeline_provenance)

        # Same-GPU high-precision block references avoid attributing GPU rounding
        # differences to quantization in the per-layer proxy.
        block_references: dict[tuple[int, int], tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
        for prompt_index, case in enumerate(cases[: args.layer_error_prompts]):
            for step_index in case["capture_steps"]:
                block_references[(prompt_index, step_index)] = capture_block_outputs(pipe, case, step_index)

        calibration_by_module = None
        calibration_metadata = None
        if args.calibration_file:
            calibration_by_module, payload = load_diffusion_calibration_by_module(resolve(args.calibration_file))
            calibration_metadata = payload.get("metadata", {})
            observed_size = payload.get("calibration_samples")
            if args.calibration_size and observed_size != args.calibration_size:
                raise ValueError(f"calibration size mismatch: requested {args.calibration_size}, file has {observed_size}")
        if args.selector != "mse" and calibration_by_module is None:
            raise ValueError("output-aware selector requires --calibration-file")

        preparation = DiffusionPreparation()
        prepare_sana_permutations(
            pipe.transformer,
            args.permutation,
            preparation,
            calibration_by_module=calibration_by_module,
        )
        rotation_map = json.loads(resolve(args.rotation_map).read_text()) if args.rotation_map else None
        apply_sana_rotations(pipe.transformer, args.rotation, preparation, rotation_map)
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        if args.selector != "mse":
            expected_permutation = calibration_metadata.get("permutation", "none")
            expected_rotation = calibration_metadata.get("rotation", "identity")
            if expected_permutation != args.permutation or expected_rotation != rotation_label:
                raise ValueError(
                    "output-aware SANA calibration provenance mismatch: "
                    f"file has permutation={expected_permutation}, rotation={expected_rotation}; "
                    f"experiment requests permutation={args.permutation}, rotation={rotation_label}"
                )
        quantize_sana_weights(
            pipe.transformer,
            args.weight_mode,
            preparation,
            selector=args.selector,
            calibration_by_module=calibration_by_module,
        )
        activation_hooks = DiffusionActivationHooks(pipe.transformer, args.activation_mode, preparation)

        per_layer_output: list[dict[str, Any]] = []
        teacher_metrics: list[dict[str, Any]] = []
        for (prompt_index, step_index), (reference_output, reference_blocks) in block_references.items():
            actual_output, actual_blocks = capture_block_outputs(pipe, cases[prompt_index], step_index)
            output_row = proxy_metrics(reference_output, actual_output)
            teacher_metrics.append(
                {
                    "prompt_id": cases[prompt_index]["prompt_record"]["prompt_id"],
                    "step_index": step_index,
                    "timestep": cases[prompt_index]["timesteps"][step_index],
                    **output_row,
                }
            )
            for name, reference in reference_blocks.items():
                row = proxy_metrics(reference, actual_blocks[name])
                per_layer_output.append(
                    {
                        "experiment_id": args.experiment_id,
                        "model": SANA_MODEL_ID,
                        "prompt_id": cases[prompt_index]["prompt_record"]["prompt_id"],
                        "step_index": step_index,
                        "timestep": cases[prompt_index]["timesteps"][step_index],
                        "layer": name,
                        "layer_idx": diffusion_layer_index(name),
                        **row,
                    }
                )

        trajectory_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            result = denoise_from_conditioning(
                pipe,
                conditioning=case["conditioning"].to("cuda:0"),
                attention_mask=case["attention_mask"].to("cuda:0"),
                initial_latents=case["initial_latents"],
                num_steps=int(case["num_steps"]),
                guidance_scale=float(case["guidance_scale"]),
                capture_steps=set(case["capture_steps"]),
            )
            prompt_id = case["prompt_record"]["prompt_id"]
            for trajectory_index in range(result["trajectory"].shape[0]):
                row = proxy_metrics(case["trajectory"][trajectory_index].float(), result["trajectory"][trajectory_index].float())
                trajectory_rows.append(
                    {
                        "experiment_id": args.experiment_id,
                        "model": SANA_MODEL_ID,
                        "prompt_id": prompt_id,
                        "trajectory_index": trajectory_index,
                        "step_index": trajectory_index - 1,
                        "timestep": case["timesteps"][trajectory_index - 1] if trajectory_index else None,
                        **row,
                    }
                )
            for step_index in case["capture_steps"]:
                row = proxy_metrics(case["predictions"][step_index], result["predictions"][step_index])
                prediction_rows.append(
                    {
                        "experiment_id": args.experiment_id,
                        "model": SANA_MODEL_ID,
                        "prompt_id": prompt_id,
                        "step_index": step_index,
                        "timestep": case["timesteps"][step_index],
                        **row,
                    }
                )
            final_rows.append(proxy_metrics(case["trajectory"][-1].float(), result["trajectory"][-1].float()))
            print(f"SANA proxy {index + 1}/{len(cases)} {prompt_id}", flush=True)

        for row in preparation.layer_metrics:
            row.update({"experiment_id": args.experiment_id, "domain": "diffusion", "model": SANA_MODEL_ID})
        for row in preparation.region_metrics:
            row.update({"experiment_id": args.experiment_id, "domain": "diffusion", "model": SANA_MODEL_ID})
        for row in preparation.margin_samples:
            row.update({"experiment_id": args.experiment_id, "domain": "diffusion", "model": SANA_MODEL_ID})
        for row in activation_hooks.metrics:
            row.update({"experiment_id": args.experiment_id, "domain": "diffusion", "model": SANA_MODEL_ID})

        pd.DataFrame(preparation.layer_metrics).to_csv(ledger.directory / "per_layer_metrics.csv", index=False)
        pd.DataFrame(preparation.region_metrics).to_csv(ledger.directory / "format_region_metrics.csv", index=False)
        pd.DataFrame(preparation.margin_samples).to_csv(ledger.directory / "format_margin_samples.csv", index=False)
        pd.DataFrame(activation_hooks.metrics).to_csv(ledger.directory / "activation_metrics.csv", index=False)
        pd.DataFrame(per_layer_output).to_csv(ledger.directory / "per_layer_output_metrics.csv", index=False)
        pd.DataFrame(teacher_metrics).to_csv(ledger.directory / "teacher_forced_proxy_metrics.csv", index=False)
        pd.DataFrame(trajectory_rows).to_csv(ledger.directory / "trajectory_metrics.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(ledger.directory / "timestep_proxy_metrics.csv", index=False)
        atomic_json(ledger.directory / "equivalence_checks.json", preparation.equivalence_checks)
        atomic_json(ledger.directory / "permutations.json", preparation.permutation_records)
        atomic_json(ledger.directory / "skipped_modules.json", preparation.skipped_modules)
        atomic_json(ledger.directory / "calibration_metadata.json", calibration_metadata)

        prediction_aggregate = aggregate_metric_rows(prediction_rows)
        trajectory_aggregate = aggregate_metric_rows(trajectory_rows)
        final_aggregate = aggregate_metric_rows(final_rows)
        weight_aggregate = aggregate_weight_metrics(preparation.layer_metrics)
        activation_aggregate = aggregate_activation_metrics(activation_hooks.metrics)
        scale_rule = preparation.layer_metrics[0].get("scale_rule", "none") if preparation.layer_metrics else "none"
        summary = {
            "experiment_id": args.experiment_id,
            "phase": args.phase,
            "domain": "diffusion",
            "status": "completed",
            "model": SANA_MODEL_ID,
            "model_revision": SANA_REVISION,
            "transformer_revision": pipeline_provenance["transformer_revision"],
            "vae_revision": pipeline_provenance["vae_revision"],
            "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
            "scheduler_class": pipeline_provenance["scheduler_class"],
            "scheduler_config": pipeline_provenance["scheduler_config"],
            "num_inference_steps": int(cases[0]["num_steps"]),
            "guidance_scale": float(cases[0]["guidance_scale"]),
            "pag_scale": pipeline_provenance["pag_scale"],
            "height": int(cases[0]["height"]),
            "width": int(cases[0]["width"]),
            "dataset_or_promptset": pointer["reference_cases"],
            "prompt_manifest": pointer["reference_cases"],
            "reference_experiment_id": pointer["experiment_id"],
            "reference_attempt_id": pointer["attempt_id"],
            "reference_trajectory_storage_dtype": cases[0]["trajectory_storage_dtype"],
            "num_proxy_prompts": len(cases),
            "quantization_mode": f"W={args.weight_mode}/A={args.activation_mode}",
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "weight_format_granularity": args.weight_mode,
            "activation_format_granularity": args.activation_mode,
            "scale_rule": scale_rule,
            "scale_group_size": 16,
            "selector": args.selector,
            "permutation": args.permutation,
            "rotation": rotation_label,
            "rotation_map": args.rotation_map,
            "calibration_size": args.calibration_size,
            "calibration_file": args.calibration_file,
            "four_over_six_mode": "canonical" if args.weight_mode in {"4over6", "nvfp4_4over6", "fouroversix_4over6"} else "composed" if args.weight_mode.endswith("_4over6") else "none",
            "proxy_mse": prediction_aggregate["mse"],
            "proxy_nmse": prediction_aggregate["nmse"],
            "proxy_relative_l2": prediction_aggregate["relative_l2"],
            "proxy_cosine_error": prediction_aggregate["cosine_error"],
            "latent_trajectory_mse": trajectory_aggregate["mse"],
            "latent_trajectory_nmse": trajectory_aggregate["nmse"],
            "final_latent_mse": final_aggregate["mse"],
            "final_latent_nmse": final_aggregate["nmse"],
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
            **weight_aggregate,
            **activation_aggregate,
        }
        atomic_json(
            ledger.directory / "raw_metrics.json",
            {
                "summary": summary,
                "prediction_aggregate": prediction_aggregate,
                "trajectory_aggregate": trajectory_aggregate,
                "final_latent_aggregate": final_aggregate,
            },
        )
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
