#!/usr/bin/env python3
"""Materialize stable SANA Phase-A proxy and image-screening queues."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "00_environment" / "queues" / "specs"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def proxy_job(
    experiment_id: str,
    phase: str,
    weight_mode: str,
    activation_mode: str = "high_precision",
    *,
    prompt_limit: int | None = None,
) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_sana_experiment.py",
        "--experiment-id", experiment_id,
        "--phase", phase,
        "--weight-mode", weight_mode,
        "--activation-mode", activation_mode,
    ]
    if prompt_limit is not None:
        command.extend(("--prompt-limit", str(prompt_limit)))
    return {
        "experiment_id": experiment_id,
        # The formal high-precision trajectory/image references are captured
        # on A6000.  Keep every main Phase-A SANA comparison on that GPU class
        # so cross-architecture BF16 kernel differences are not folded into
        # quantization error; the explicit cross-GPU queue measures those
        # differences separately.
        "gpu_kind": "a6000",
        "log_file": f"artifacts/03_phase_a/diffusion/logs/{experiment_id}.log",
        "command": command,
    }


def image_job(experiment_id: str, weight_mode: str, activation_mode: str = "high_precision") -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "gpu_kind": "a6000",
        "log_file": f"artifacts/03_phase_a/diffusion/logs/{experiment_id}.log",
        "command": [
            "python3", "scripts/run_sana_images.py",
            "--experiment-id", experiment_id,
            "--phase", "phase_a2_diffusion_images",
            "--count", "128",
            "--steps", "20",
            "--weight-mode", weight_mode,
            "--activation-mode", activation_mode,
            "--compute-lpips",
            "--compute-image-reward",
            "--compute-clip-score",
        ],
    }


def write(name: str, jobs: list[dict[str, Any]], purpose: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": timestamp(),
                "purpose": purpose,
                "num_jobs": len(jobs),
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    write(
        "phase0_sana_smoke",
        [
            {
                "experiment_id": "phase0_smoke_sana_reference_a6000",
                "gpu_kind": "a6000",
                "log_file": "artifacts/03_phase_a/diffusion/logs/phase0_smoke_sana_reference_a6000.log",
                "command": [
                    "python3", "scripts/capture_sana_reference.py",
                    "--experiment-id", "phase0_smoke_sana_reference_a6000",
                    "--limit", "1", "--steps", "2",
                ],
            }
        ],
        "Single-prompt/two-step SANA load, trajectory, VAE, and determinism smoke test",
    )
    write(
        "phase0_sana_reference",
        [
            {
                "experiment_id": "phase0_sana_reference_proxy16_steps20_fp32_trajectory",
                "gpu_kind": "a6000",
                "log_file": "artifacts/03_phase_a/diffusion/logs/phase0_sana_reference_proxy16_steps20_fp32_trajectory.log",
                "command": [
                    "python3", "scripts/capture_sana_reference.py",
                    "--experiment-id", "phase0_sana_reference_proxy16_steps20_fp32_trajectory",
                    "--steps", "20", "--formal",
                ],
            }
        ],
        "Formal fixed 16-prompt SANA high-precision trajectory reference",
    )
    write(
        "phase_a_sana_image_reference",
        [
            {
                "experiment_id": "phasea_sana_images_high_precision_reference_128",
                "gpu_kind": "a6000",
                "log_file": "artifacts/03_phase_a/diffusion/logs/phasea_sana_images_high_precision_reference_128.log",
                "command": [
                    "python3", "scripts/run_sana_images.py",
                    "--experiment-id", "phasea_sana_images_high_precision_reference_128",
                    "--phase", "phase_a2_diffusion_images", "--count", "128", "--steps", "20",
                    "--weight-mode", "high_precision", "--activation-mode", "high_precision",
                    "--write-reference", "--compute-image-reward", "--compute-clip-score",
                ],
            }
        ],
        "Formal fixed 128-prompt SANA high-precision image reference and absolute scores",
    )
    weight_modes = (
        "high_precision", "nvfp4", "all_e0m3", "oracle16", "k64_row",
        "n8k16", "n8k64", "fouroversix_4over6", "oracle16_4over6",
        "n8k64_4over6",
    )
    write(
        "phase_a2_sana_proxy",
        [
            proxy_job(
                f"phasea2_sana_proxy_{mode}_fp32ref",
                "phase_a2_diffusion",
                mode,
            )
            for mode in weight_modes
        ],
        "SANA-1.6B fixed 16-prompt W4A16 trajectory/proxy sweep",
    )
    w4a4 = (
        ("high_precision", "high_precision"),
        ("nvfp4", "all_e2m1"),
        ("all_e2m1", "all_e2m1"),
        ("all_e2m1", "oracle16_a"),
        ("oracle16", "oracle16_a"),
        ("n8k64", "all_e2m1"),
        ("all_e2m1", "m16k64"),
        ("n8k64", "m16k64"),
    )
    write(
        "phase_a3_sana_proxy",
        [
            proxy_job(
                f"phasea3_sana_proxy_w_{weight}_a_{activation}",
                "phase_a3_diffusion",
                weight,
                activation,
            )
            for weight, activation in w4a4
        ],
        "SANA-1.6B mandatory W4A4 one/two-side proxy comparisons",
    )
    screening = (
        ("nvfp4", "high_precision"),
        ("all_e0m3", "high_precision"),
        ("oracle16", "high_precision"),
        ("n8k64", "high_precision"),
        ("fouroversix_4over6", "high_precision"),
        ("n8k64", "m16k64"),
    )
    write(
        "phase_a_sana_images",
        [image_job(f"phasea_sana_images_w_{weight}_a_{activation}", weight, activation) for weight, activation in screening],
        "SANA fixed 128-prompt paired image screening after the HP reference image set",
    )
    print(json.dumps({path.name: json.loads(path.read_text())["num_jobs"] for path in sorted(OUTPUT.glob("*sana*.json"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
