#!/usr/bin/env python3
"""Generate fixed SANA image sets and compute paired image metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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
    load_diffusion_calibration_by_module,
    load_prompt_rows,
    load_sana_pipeline,
    prepare_sana_permutations,
    quantize_sana_weights,
    sana_pipeline_provenance,
)
from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402


DEFAULT_SCREENING = "artifacts/00_environment/diffusion_prompts/screening_mjhq_128.jsonl"
REFERENCE_POINTER = ROOT / "artifacts" / "03_phase_a" / "diffusion" / "images" / "screening_reference_current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--prompt-file", default=DEFAULT_SCREENING)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--weight-mode", default="high_precision")
    parser.add_argument("--activation-mode", default="high_precision")
    parser.add_argument("--selector", choices=("mse", "activation_aware", "output_aware"), default="mse")
    parser.add_argument("--calibration-file")
    parser.add_argument("--calibration-size", type=int, default=0)
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--reference-pointer", default=str(REFERENCE_POINTER.relative_to(ROOT)))
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--compute-image-reward", action="store_true")
    parser.add_argument("--compute-clip-score", action="store_true")
    parser.add_argument("--compute-fid", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def import_image_reward():
    """Import ImageReward against current Transformers without changing its semantics.

    ImageReward 1.5 imports three helpers from ``modeling_utils`` as they were
    located in Transformers 4.27.  Current Transformers exposes the identical
    helpers from ``pytorch_utils``.  Supplying those compatibility aliases lets
    us retain the project's pinned modern Transformers version used by SANA.
    """

    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
    import ImageReward as reward_module

    return reward_module


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config.update({"domain": "diffusion", "model": SANA_MODEL_ID})
    ledger = ExperimentLedger(args.experiment_id, args.phase, config)
    try:
        rows = load_prompt_rows(args.prompt_file)[: args.count]
        if len(rows) != args.count:
            raise ValueError(f"prompt file contains {len(rows)} rows, expected {args.count}")
        calibration_by_module = None
        calibration_metadata = None
        if args.calibration_file:
            calibration_by_module, payload = load_diffusion_calibration_by_module(resolve(args.calibration_file))
            calibration_metadata = payload.get("metadata", {})
            if args.calibration_size and payload.get("calibration_samples") != args.calibration_size:
                raise ValueError("calibration size mismatch")
        if args.selector != "mse" and calibration_by_module is None:
            raise ValueError("output-aware image experiment requires calibration")

        pipe = load_sana_pipeline(with_vae=True)
        pipeline_provenance = sana_pipeline_provenance(pipe)
        atomic_json(ledger.directory / "pipeline_provenance.json", pipeline_provenance)
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
                    "output-aware image calibration provenance mismatch: "
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
        image_dir = ledger.directory / "images"
        image_dir.mkdir()
        image_records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            start = time.monotonic()
            image = pipe(
                prompt=row["prompt"],
                negative_prompt="",
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=torch.Generator(device="cpu").manual_seed(int(row["seed"])),
                output_type="pil",
                clean_caption=True,
                use_resolution_binning=True,
                max_sequence_length=300,
            ).images[0]
            path = image_dir / f"{index:04d}_{row['prompt_id']}.png"
            image.save(path, format="PNG", compress_level=6)
            image_records.append(
                {
                    "index": index,
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "seed": row["seed"],
                    "category": row.get("category"),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                    "generation_wall_time_seconds": time.monotonic() - start,
                }
            )
            print(f"SANA image {index + 1}/{len(rows)} {row['prompt_id']}", flush=True)
        atomic_json(ledger.directory / "image_manifest.json", image_records)

        paired_rows: list[dict[str, Any]] = []
        metric_failures: list[dict[str, str]] = []
        lpips_model = None
        if args.compute_lpips and not args.write_reference:
            try:
                import lpips

                lpips_model = lpips.LPIPS(net="alex").to("cuda:0").eval()
            except BaseException as error:
                metric_failures.append({"metric": "lpips", "error_type": type(error).__name__, "error": str(error)})
        reference_by_id = None
        if not args.write_reference:
            pointer = json.loads(resolve(args.reference_pointer).read_text())
            reference_records = json.loads(resolve(pointer["image_manifest"]).read_text())
            reference_by_id = {row["prompt_id"]: row for row in reference_records}
            if set(reference_by_id) != {row["prompt_id"] for row in image_records}:
                raise ValueError("reference and generated prompt IDs do not match exactly")
            for row in image_records:
                generated = image_array(resolve(row["path"]))
                reference = image_array(resolve(reference_by_id[row["prompt_id"]]["path"]))
                delta = generated - reference
                metric = {
                    "experiment_id": args.experiment_id,
                    "prompt_id": row["prompt_id"],
                    "psnr": float(peak_signal_noise_ratio(reference, generated, data_range=1.0)),
                    "ssim": float(structural_similarity(reference, generated, channel_axis=2, data_range=1.0)),
                    "pixel_mse": float(np.square(delta).mean()),
                    "lpips": None,
                }
                if lpips_model is not None:
                    ref_tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to("cuda:0") * 2 - 1
                    gen_tensor = torch.from_numpy(generated).permute(2, 0, 1).unsqueeze(0).to("cuda:0") * 2 - 1
                    with torch.inference_mode():
                        metric["lpips"] = float(lpips_model(ref_tensor, gen_tensor).item())
                paired_rows.append(metric)
            pd.DataFrame(paired_rows).to_csv(ledger.directory / "paired_image_metrics.csv", index=False)

        reward_rows: list[dict[str, Any]] = []
        if args.compute_image_reward:
            try:
                RM = import_image_reward()

                reward_model = RM.load("ImageReward-v1.0", device="cuda:0")
                for row in image_records:
                    reward_rows.append(
                        {
                            "experiment_id": args.experiment_id,
                            "prompt_id": row["prompt_id"],
                            "image_reward": float(reward_model.score(row["prompt"], str(resolve(row["path"])))),
                        }
                    )
                pd.DataFrame(reward_rows).to_csv(ledger.directory / "image_reward.csv", index=False)
            except BaseException as error:
                metric_failures.append({"metric": "image_reward", "error_type": type(error).__name__, "error": str(error)})

        clip_rows: list[dict[str, Any]] = []
        if args.compute_clip_score:
            try:
                import clip

                clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda:0")
                clip_model.eval()
                for row in image_records:
                    tokens = clip.tokenize([row["prompt"]], truncate=True).to("cuda:0")
                    pixels = clip_preprocess(Image.open(resolve(row["path"])).convert("RGB")).unsqueeze(0).to("cuda:0")
                    with torch.inference_mode():
                        image_features = clip_model.encode_image(pixels)
                        text_features = clip_model.encode_text(tokens)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        score = 100.0 * (image_features @ text_features.T).item()
                    clip_rows.append(
                        {
                            "experiment_id": args.experiment_id,
                            "prompt_id": row["prompt_id"],
                            "clip_score": float(score),
                        }
                    )
                pd.DataFrame(clip_rows).to_csv(ledger.directory / "clip_score.csv", index=False)
            except BaseException as error:
                metric_failures.append({"metric": "clip_score", "error_type": type(error).__name__, "error": str(error)})

        fid = None
        if args.compute_fid and not args.write_reference:
            try:
                from cleanfid import fid as clean_fid

                reference_dir = resolve(pointer["artifact_dir"]) / "images"
                fid = float(clean_fid.compute_fid(str(image_dir), str(reference_dir), device=torch.device("cuda:0")))
            except BaseException as error:
                metric_failures.append({"metric": "fid", "error_type": type(error).__name__, "error": str(error)})
        atomic_json(ledger.directory / "metric_failures.json", metric_failures)
        atomic_json(ledger.directory / "calibration_metadata.json", calibration_metadata)
        atomic_json(ledger.directory / "equivalence_checks.json", preparation.equivalence_checks)
        atomic_json(ledger.directory / "permutations.json", preparation.permutation_records)
        atomic_json(ledger.directory / "skipped_modules.json", preparation.skipped_modules)
        pd.DataFrame(preparation.layer_metrics).to_csv(ledger.directory / "per_layer_metrics.csv", index=False)
        pd.DataFrame(preparation.region_metrics).to_csv(ledger.directory / "format_region_metrics.csv", index=False)
        pd.DataFrame(activation_hooks.metrics).to_csv(ledger.directory / "activation_metrics.csv", index=False)

        def mean(key: str, values: list[dict]) -> float | None:
            available = [float(row[key]) for row in values if row.get(key) is not None and math.isfinite(float(row[key]))]
            return sum(available) / len(available) if available else None

        weight_aggregate = aggregate_weight_metrics(preparation.layer_metrics)
        activation_aggregate = aggregate_activation_metrics(activation_hooks.metrics)
        summary = {
            "experiment_id": args.experiment_id,
            "phase": args.phase,
            "domain": "diffusion",
            "status": "completed" if not metric_failures else "completed_with_metric_failures",
            "model": SANA_MODEL_ID,
            "model_revision": SANA_REVISION,
            "transformer_revision": pipeline_provenance["transformer_revision"],
            "vae_revision": pipeline_provenance["vae_revision"],
            "text_encoder_revision": pipeline_provenance["text_encoder_revision"],
            "scheduler_class": pipeline_provenance["scheduler_class"],
            "scheduler_config": pipeline_provenance["scheduler_config"],
            "pag_scale": pipeline_provenance["pag_scale"],
            "dataset_or_promptset": args.prompt_file,
            "prompt_manifest": args.prompt_file,
            "num_images": len(rows),
            "quantization_mode": f"W={args.weight_mode}/A={args.activation_mode}",
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "weight_format_granularity": args.weight_mode,
            "activation_format_granularity": args.activation_mode,
            "scale_rule": preparation.layer_metrics[0].get("scale_rule", "none") if preparation.layer_metrics else "none",
            "scale_group_size": 16,
            "selector": args.selector,
            "permutation": args.permutation,
            "rotation": rotation_label,
            "rotation_map": args.rotation_map,
            "calibration_size": args.calibration_size,
            "four_over_six_mode": "canonical" if args.weight_mode in {"4over6", "nvfp4_4over6", "fouroversix_4over6"} else "composed" if args.weight_mode.endswith("_4over6") else "none",
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "psnr": mean("psnr", paired_rows),
            "ssim": mean("ssim", paired_rows),
            "lpips": mean("lpips", paired_rows),
            "image_reward": mean("image_reward", reward_rows),
            "clip_score": mean("clip_score", clip_rows),
            "fid": fid,
            "metric_failures": metric_failures,
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
            **weight_aggregate,
            **activation_aggregate,
        }
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        if args.write_reference:
            atomic_json(
                resolve(args.reference_pointer),
                {
                    "experiment_id": args.experiment_id,
                    "attempt_id": ledger.attempt_id,
                    "artifact_dir": str(ledger.directory.relative_to(ROOT)),
                    "image_manifest": str((ledger.directory / "image_manifest.json").relative_to(ROOT)),
                    "prompt_file": args.prompt_file,
                    "prompt_file_sha256": file_sha256(resolve(args.prompt_file)),
                    "model_revision": SANA_REVISION,
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
