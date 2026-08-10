#!/usr/bin/env python3
"""Capture cumulative 32/128/256-prompt SANA output-aware statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.adapters.deepcompressor_diffusion import (  # noqa: E402
    DiffusionCovarianceCollector,
    DiffusionPreparation,
    SANA_MODEL_ID,
    SANA_REVISION,
    apply_sana_rotations,
    load_prompt_rows,
    load_diffusion_calibration_by_module,
    load_sana_pipeline,
    prepare_sana_permutations,
    sana_pipeline_provenance,
)
from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402
from project_quant.calibration import save_covariances  # noqa: E402


DEFAULT_PROMPTS = "artifacts/00_environment/diffusion_prompts/calibration_qdiff_256.jsonl"
POINTER = ROOT / "artifacts" / "04_phase_b" / "selector" / "calibration" / "sana_current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-activation-rows", type=int, default=64)
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--packing-calibration-file")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    return parser.parse_args()


def encode_batch(pipe, rows: list[dict]):
    positive, positive_mask, negative, negative_mask = pipe.encode_prompt(
        prompt=[row["prompt"] for row in rows],
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=torch.device("cuda:0"),
        clean_caption=True,
        max_sequence_length=300,
    )
    return torch.cat((negative, positive), 0), torch.cat((negative_mask, positive_mask), 0)


@torch.inference_mode()
def run_group(pipe, collector: DiffusionCovarianceCollector, rows: list[dict], target_step: int, args) -> None:
    conditioning, mask = encode_batch(pipe, rows)
    latents = torch.stack(
        [
            torch.randn(
                (pipe.transformer.config.in_channels, 32, 32),
                generator=torch.Generator(device="cpu").manual_seed(int(row["seed"])),
                dtype=torch.float32,
            ).to(torch.bfloat16)
            for row in rows
        ]
    ).to("cuda:0")
    pipe.scheduler.set_timesteps(args.steps, device=torch.device("cuda:0"))
    for step_index, timestep in enumerate(pipe.scheduler.timesteps[: target_step + 1]):
        collector.set_enabled(step_index == target_step)
        if step_index == target_step:
            collector.begin_sample()
        model_input = torch.cat((latents, latents), 0).to(conditioning.dtype)
        expanded = timestep.expand(model_input.shape[0]).to(latents.dtype)
        output = pipe.transformer(
            model_input,
            encoder_hidden_states=conditioning,
            encoder_attention_mask=mask,
            timestep=expanded,
            return_dict=False,
        )[0].float()
        unconditioned, text = output.chunk(2)
        guided = unconditioned + args.guidance_scale * (text - unconditioned)
        latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]
    collector.set_enabled(False)


def main() -> int:
    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or any(value not in {32, 128, 256} for value in checkpoints):
        raise ValueError("SANA calibration checkpoints must be a nonempty subset of 32, 128, and 256")
    config = vars(args).copy()
    config.update({"domain": "diffusion", "model": SANA_MODEL_ID})
    ledger = ExperimentLedger(args.experiment_id, "phase_b_selector_calibration", config)
    collector = None
    try:
        rows = load_prompt_rows(args.prompt_file)[: max(checkpoints)]
        if len(rows) != max(checkpoints):
            raise ValueError("calibration prompt file is shorter than 256")
        pipe = load_sana_pipeline(with_vae=False)
        pipeline_provenance = sana_pipeline_provenance(pipe)
        atomic_json(ledger.directory / "pipeline_provenance.json", pipeline_provenance)
        preparation = DiffusionPreparation()
        packing_calibration = None
        packing_metadata = None
        if args.packing_calibration_file:
            packing_calibration, packing_payload = load_diffusion_calibration_by_module(
                Path(args.packing_calibration_file)
            )
            packing_metadata = packing_payload.get("metadata", {})
        prepare_sana_permutations(
            pipe.transformer,
            args.permutation,
            preparation,
            calibration_by_module=packing_calibration,
        )
        rotation_map = json.loads(Path(args.rotation_map).read_text()) if args.rotation_map else None
        apply_sana_rotations(pipe.transformer, args.rotation, preparation, rotation_map)
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        collector = DiffusionCovarianceCollector(
            pipe.transformer,
            preparation,
            max_rows_per_invocation=args.max_activation_rows,
        )
        collector.set_enabled(False)
        pipe.scheduler.set_timesteps(args.steps, device=torch.device("cuda:0"))
        representative_steps = [0, args.steps // 2, args.steps - 1]
        representative_timesteps = [int(pipe.scheduler.timesteps[index].item()) for index in representative_steps]
        files: dict[str, dict] = {}
        assignments: list[dict] = []
        processed = 0
        prior = 0
        for checkpoint in checkpoints:
            chunk = rows[prior:checkpoint]
            # Each checkpoint increment is split nearly equally among early,
            # middle, and late actual HP trajectories.
            groups_by_step = {step: [] for step in representative_steps}
            for offset, row in enumerate(chunk):
                assigned_step = representative_steps[offset % 3]
                groups_by_step[assigned_step].append(row)
                assignments.append(
                    {
                        "calibration_index": prior + offset,
                        "checkpoint_increment_end": checkpoint,
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "seed": row["seed"],
                        "step_index": assigned_step,
                        "timestep": representative_timesteps[
                            representative_steps.index(assigned_step)
                        ],
                    }
                )
            for target_step in representative_steps:
                group_rows = groups_by_step[target_step]
                for start in range(0, len(group_rows), args.batch_size):
                    batch = group_rows[start : start + args.batch_size]
                    run_group(pipe, collector, batch, target_step, args)
                    processed += len(batch)
                    print(
                        f"SANA calibration checkpoint={checkpoint} target_step={target_step} "
                        f"batch={len(batch)} cumulative_processed={prior + processed}",
                        flush=True,
                    )
            processed = 0
            prior = checkpoint
            metadata = {
                "model": SANA_MODEL_ID,
                "model_revision": SANA_REVISION,
                "transformer_revision": pipeline_provenance["transformer_revision"],
                "vae_revision": pipeline_provenance["vae_revision"],
                "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
                "scheduler_class": pipeline_provenance["scheduler_class"],
                "scheduler_config": pipeline_provenance["scheduler_config"],
                "pag_scale": pipeline_provenance["pag_scale"],
                "latent_resolution": [32, 32],
                "image_resolution": [1024, 1024],
                "prompt_file": args.prompt_file,
                "prompt_ids": [row["prompt_id"] for row in rows[:checkpoint]],
                "prompt_seeds": [row["seed"] for row in rows[:checkpoint]],
                "prompt_timestep_assignments": assignments[:checkpoint],
                "num_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "representative_step_indices": representative_steps,
                "representative_timesteps": representative_timesteps,
                "assignment": "within each cumulative increment, prompt offset modulo 3 selects early/mid/late",
                "trajectory_policy": "run the HP denoising trajectory through the assigned target step; collect only that step",
                "permutation": args.permutation,
                "packing_calibration_file": args.packing_calibration_file,
                "packing_calibration_metadata": packing_metadata,
                "rotation": rotation_label,
                "rotation_map": args.rotation_map,
                "evaluation_disjoint": True,
            }
            payload = collector.payload(calibration_samples=checkpoint, metadata=metadata)
            path = ledger.directory / f"calibration_{checkpoint}.pt"
            digest = save_covariances(path, payload)
            files[str(checkpoint)] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "num_canonical_inputs": len(payload["hessian_by_canonical"]),
                "num_modules": len(payload["module_to_canonical"]),
                "activation_row_sampling": payload["activation_row_sampling"],
            }
            del payload

        collector.close()
        collector = None
        atomic_json(ledger.directory / "calibration_sample_manifest.json", assignments)
        atomic_json(ledger.directory / "calibration_files.json", files)
        atomic_json(ledger.directory / "equivalence_checks.json", preparation.equivalence_checks)
        atomic_json(ledger.directory / "permutations.json", preparation.permutation_records)
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_selector_calibration",
            "domain": "diffusion",
            "status": "completed",
            "model": SANA_MODEL_ID,
            "model_revision": SANA_REVISION,
            "transformer_revision": pipeline_provenance["transformer_revision"],
            "vae_revision": pipeline_provenance["vae_revision"],
            "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
            "scheduler_class": pipeline_provenance["scheduler_class"],
            "scheduler_config": pipeline_provenance["scheduler_config"],
            "pag_scale": pipeline_provenance["pag_scale"],
            "calibration_sizes": checkpoints,
            "prompt_file": args.prompt_file,
            "dataset_or_promptset": args.prompt_file,
            "calibration_sample_manifest": str(
                (ledger.directory / "calibration_sample_manifest.json").relative_to(ROOT)
            ),
            "representative_step_indices": representative_steps,
            "representative_timesteps": representative_timesteps,
            "max_activation_rows_per_module_invocation": args.max_activation_rows,
            "permutation": args.permutation,
            "packing_calibration_file": args.packing_calibration_file,
            "rotation": rotation_label,
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
            "files": files,
        }
        import pandas as pd

        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        ledger.complete(summary)
        pointer = {
            "experiment_id": args.experiment_id,
            "attempt_id": ledger.attempt_id,
            "artifact_dir": str(ledger.directory.relative_to(ROOT)),
            "permutation": args.permutation,
            "rotation": rotation_label,
            "files": files,
        }
        pointer_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            f"sana_{args.permutation}_{rotation_label}".lower(),
        ).strip("_")
        atomic_json(POINTER.parent / f"{pointer_slug}_current.json", pointer)
        if args.permutation == "none" and rotation_label == "identity":
            atomic_json(POINTER, pointer)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if collector is not None:
            collector.close()
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
