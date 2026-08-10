#!/usr/bin/env python3
"""Capture fixed high-precision SANA trajectories and a real determinism check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.adapters.deepcompressor_diffusion import (  # noqa: E402
    SANA_MODEL_ID,
    SANA_REVISION,
    denoise_from_conditioning,
    load_prompt_rows,
    load_sana_pipeline,
    prompt_conditioning,
    sana_pipeline_provenance,
)
from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402


DEFAULT_PROMPTS = "artifacts/00_environment/diffusion_prompts/proxy_qdiff_16.jsonl"
REFERENCE_POINTER = ROOT / "artifacts" / "03_phase_a" / "diffusion" / "reference" / "current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def numpy_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config["domain"] = "diffusion"
    config["model"] = SANA_MODEL_ID
    ledger = ExperimentLedger(args.experiment_id, "phase0_diffusion_reference", config)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        rows = load_prompt_rows(args.prompt_file)
        if args.limit is not None:
            rows = rows[: args.limit]
        pipe = load_sana_pipeline(with_vae=True)
        pipeline_provenance = sana_pipeline_provenance(pipe)
        atomic_json(ledger.directory / "pipeline_provenance.json", pipeline_provenance)
        capture_steps = {
            round(index * (args.steps - 1) / 5)
            for index in range(6)
        }
        records: list[dict] = []
        case_dir = ledger.directory / "proxy_cases"
        case_dir.mkdir()
        for index, row in enumerate(rows):
            conditioning, attention_mask = prompt_conditioning(pipe, row["prompt"])
            initial = torch.randn(
                (1, pipe.transformer.config.in_channels, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor),
                generator=torch.Generator(device="cpu").manual_seed(int(row["seed"])),
                dtype=torch.float32,
            ).to(torch.bfloat16)
            result = denoise_from_conditioning(
                pipe,
                conditioning=conditioning,
                attention_mask=attention_mask,
                initial_latents=initial,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                capture_steps=capture_steps,
            )
            payload = {
                "schema_version": 1,
                "prompt_record": row,
                "conditioning": conditioning.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "initial_latents": initial.cpu(),
                # Preserve the scheduler state in FP32.  The transformer sees
                # BF16 inputs, but FlowMatch/DPMSolver updates promote latents
                # to FP32; storing those states as BF16 creates a non-zero HP
                # trajectory floor that can be mistaken for quantization loss.
                "trajectory": result["trajectory"].to(torch.float32),
                "predictions": {key: value.to(torch.float32) for key, value in result["predictions"].items()},
                "teacher_forced_inputs": {key: value.to(torch.float32) for key, value in result["inputs"].items()},
                "trajectory_storage_dtype": "float32",
                "prediction_storage_dtype": "float32",
                "teacher_forced_input_storage_dtype": "float32",
                "timesteps": result["timesteps"],
                "capture_steps": sorted(capture_steps),
                "num_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "model_revision": SANA_REVISION,
                "transformer_revision": pipeline_provenance["transformer_revision"],
                "vae_revision": pipeline_provenance["vae_revision"],
                "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
                "scheduler_class": pipeline_provenance["scheduler_class"],
                "scheduler_config": pipeline_provenance["scheduler_config"],
                "pag_scale": pipeline_provenance["pag_scale"],
            }
            path = case_dir / f"{index:04d}_{row['prompt_id']}.pt"
            torch.save(payload, path)
            records.append(
                {
                    "index": index,
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "path": str(path.relative_to(ROOT)),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "initial_latents_sha256": tensor_sha256(initial),
                    "final_latents_sha256": tensor_sha256(result["trajectory"][-1]),
                    "timesteps": result["timesteps"],
                    "wall_time_seconds": result["wall_time_seconds"],
                }
            )
            print(f"reference proxy {index + 1}/{len(rows)} {row['prompt_id']}", flush=True)

        # Replay the exact serialized FP32 trajectory contract before using it
        # as the formal proxy reference.  This catches accidental reference
        # compression independently of image-level determinism.
        first_case = torch.load(
            ROOT / records[0]["path"], map_location="cpu", weights_only=True
        )
        replay = denoise_from_conditioning(
            pipe,
            conditioning=first_case["conditioning"].to("cuda:0"),
            attention_mask=first_case["attention_mask"].to("cuda:0"),
            initial_latents=first_case["initial_latents"],
            num_steps=int(first_case["num_steps"]),
            guidance_scale=float(first_case["guidance_scale"]),
            capture_steps=set(first_case["capture_steps"]),
        )
        trajectory_delta = replay["trajectory"].float() - first_case["trajectory"].float()
        prediction_deltas = [
            replay["predictions"][step].float() - first_case["predictions"][step].float()
            for step in first_case["capture_steps"]
        ]
        trajectory_replay = {
            "trajectory_storage_dtype": first_case["trajectory_storage_dtype"],
            "trajectory_bit_exact": bool(
                torch.equal(replay["trajectory"].float(), first_case["trajectory"].float())
            ),
            "trajectory_max_abs_difference": float(trajectory_delta.abs().max().item()),
            "trajectory_mse_difference": float(trajectory_delta.square().mean().item()),
            "prediction_bit_exact": bool(
                all(float(delta.abs().max().item()) == 0.0 for delta in prediction_deltas)
            ),
            "prediction_max_abs_difference": float(
                max(delta.abs().max().item() for delta in prediction_deltas)
            ),
        }
        if args.formal and not (
            trajectory_replay["trajectory_bit_exact"]
            and trajectory_replay["prediction_bit_exact"]
        ):
            raise RuntimeError(
                f"formal serialized SANA trajectory replay is not bit-exact: {trajectory_replay}"
            )

        # Mandatory real fixed-prompt/fixed-seed pipeline determinism test.
        first = rows[0]
        images = []
        for _ in range(2):
            output = pipe(
                prompt=first["prompt"],
                negative_prompt="",
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=torch.Generator(device="cpu").manual_seed(int(first["seed"])),
                output_type="np",
                clean_caption=True,
                use_resolution_binning=True,
                max_sequence_length=300,
            ).images
            images.append(np.asarray(output))
        delta = images[1].astype(np.float64) - images[0].astype(np.float64)
        determinism = {
            "model": SANA_MODEL_ID,
            "model_revision": SANA_REVISION,
            "transformer_revision": pipeline_provenance["transformer_revision"],
            "vae_revision": pipeline_provenance["vae_revision"],
            "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
            "scheduler_class": pipeline_provenance["scheduler_class"],
            "scheduler_config": pipeline_provenance["scheduler_config"],
            "pag_scale": pipeline_provenance["pag_scale"],
            "prompt_id": first["prompt_id"],
            "prompt": first["prompt"],
            "seed": first["seed"],
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "first_sha256": numpy_sha256(images[0]),
            "second_sha256": numpy_sha256(images[1]),
            "bit_exact": bool(np.array_equal(images[0], images[1])),
            "max_abs_difference": float(np.abs(delta).max()),
            "mse_difference": float(np.square(delta).mean()),
            "serialized_trajectory_replay": trajectory_replay,
            "gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        atomic_json(ledger.directory / "diffusion_determinism.json", determinism)
        atomic_json(ROOT / "artifacts" / "02_tests" / "diffusion_determinism.json", determinism)
        atomic_json(ledger.directory / "reference_cases.json", records)
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase0_diffusion_reference",
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
            "prompt_file": args.prompt_file,
            "num_prompts": len(rows),
            "num_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "capture_steps": sorted(capture_steps),
            "scheduler_timesteps": records[0]["timesteps"],
            "trajectory_storage_dtype": "float32",
            "prediction_storage_dtype": "float32",
            "teacher_forced_input_storage_dtype": "float32",
            "determinism_bit_exact": determinism["bit_exact"],
            "determinism_max_abs_difference": determinism["max_abs_difference"],
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        atomic_json(ledger.directory / "raw_metrics.json", {"summary": summary, "determinism": determinism})
        import pandas as pd

        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        pointer = {
            "experiment_id": args.experiment_id,
            "attempt_id": ledger.attempt_id,
            "artifact_dir": str(ledger.directory.relative_to(ROOT)),
            "reference_cases": str((ledger.directory / "reference_cases.json").relative_to(ROOT)),
            "formal": args.formal,
            "num_prompts": len(rows),
            "model_revision": SANA_REVISION,
            "vae_revision": pipeline_provenance["vae_revision"],
            "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
            "scheduler_class": pipeline_provenance["scheduler_class"],
            "trajectory_storage_dtype": "float32",
            "prediction_storage_dtype": "float32",
            "teacher_forced_input_storage_dtype": "float32",
        }
        if args.formal:
            atomic_json(REFERENCE_POINTER, pointer)
        else:
            atomic_json(ledger.directory / "reference_pointer.json", pointer)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
