#!/usr/bin/env python3
"""Build Phase-B queues from completed, provenance-matched calibration pointers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "artifacts" / "00_environment" / "queues" / "specs"
CALIBRATION_DIR = ROOT / "artifacts" / "04_phase_b" / "selector" / "calibration"
MODELS = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B", "meta_llama_llama_3_1_8b_none_identity_current.json"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "qwen_qwen3_8b_none_identity_current.json"),
}


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write(name: str, jobs: list[dict[str, Any]], purpose: str) -> None:
    SPECS.mkdir(parents=True, exist_ok=True)
    (SPECS / f"{name}.json").write_text(
        json.dumps({"schema_version": 1, "generated_at": timestamp(), "purpose": purpose, "num_jobs": len(jobs), "jobs": jobs}, indent=2, sort_keys=True) + "\n"
    )


def llm_job(
    experiment_id: str,
    phase: str,
    model: str,
    dataset: str,
    *,
    selector: str = "mse",
    permutation: str = "none",
    rotation: str = "identity",
    calibration_file: str | None = None,
    calibration_size: int = 0,
    rotation_map: str | None = None,
    layer_output_error_rows: int = 0,
) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_llm_experiment.py", "--experiment-id", experiment_id,
        "--phase", phase, "--model", model, "--dataset", dataset,
        "--weight-mode", "n8k64", "--activation-mode", "high_precision",
        "--selector", selector, "--permutation", permutation, "--rotation", rotation,
        "--calibration-size", str(calibration_size),
    ]
    if calibration_file:
        command.extend(("--calibration-file", calibration_file))
    if rotation_map:
        command.extend(("--rotation-map", rotation_map))
    if layer_output_error_rows:
        command.extend(("--layer-output-error-rows", str(layer_output_error_rows)))
    category = "selector" if "selector" in phase else "permutation" if "permutation" in phase else "rotation" if "rotation" in phase else "combined"
    return {
        "experiment_id": experiment_id,
        "gpu_kind": "a6000",
        "log_file": f"artifacts/04_phase_b/{category}/logs/{experiment_id}.log",
        "command": command,
    }


def calibration_jobs() -> list[dict[str, Any]]:
    jobs = []
    for slug, (model, _) in MODELS.items():
        experiment_id = f"phaseb1_calibration_{slug}_identity"
        jobs.append(
            {
                "experiment_id": experiment_id,
                "gpu_kind": "a6000",
                "log_file": f"artifacts/04_phase_b/selector/logs/{experiment_id}.log",
                "command": [
                    "python3", "scripts/capture_calibration.py", "--experiment-id", experiment_id,
                    "--model", model, "--rotation", "identity", "--permutation", "none",
                    "--checkpoints", "32", "128", "256", "--seq-len", "128", "--seed", "314159",
                ],
            }
        )
    experiment_id = "phaseb1_calibration_sana_identity"
    jobs.append(
        {
            "experiment_id": experiment_id,
            "gpu_kind": "a6000",
            "log_file": f"artifacts/04_phase_b/selector/logs/{experiment_id}.log",
            "command": [
                "python3", "scripts/capture_sana_calibration.py", "--experiment-id", experiment_id,
                "--checkpoints", "32", "128", "256", "--steps", "20", "--guidance-scale", "4.5",
                "--batch-size", "8", "--max-activation-rows", "64",
            ],
        }
    )
    return jobs


def sana_job(
    experiment_id: str,
    phase: str,
    *,
    selector: str = "mse",
    permutation: str = "none",
    rotation: str = "identity",
    calibration_file: str | None = None,
    calibration_size: int = 0,
) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_sana_experiment.py", "--experiment-id", experiment_id,
        "--phase", phase, "--weight-mode", "n8k64", "--activation-mode", "high_precision",
        "--selector", selector, "--permutation", permutation, "--rotation", rotation,
        "--calibration-size", str(calibration_size),
    ]
    if calibration_file:
        command.extend(("--calibration-file", calibration_file))
    category = "selector" if "selector" in phase else "permutation" if "permutation" in phase else "rotation"
    return {
        "experiment_id": experiment_id,
        "gpu_kind": "a6000",
        "log_file": f"artifacts/04_phase_b/{category}/logs/{experiment_id}.log",
        "command": command,
    }


def load_calibrations() -> dict[str, dict[str, Any]]:
    result = {}
    for slug, (_, pointer_name) in MODELS.items():
        path = CALIBRATION_DIR / pointer_name
        if not path.exists():
            raise FileNotFoundError(f"missing calibration pointer {path.relative_to(ROOT)}")
        result[slug] = json.loads(path.read_text(encoding="utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    write("phase_b1_calibration", calibration_jobs(), "Identity-layout LLM and timestep-stratified SANA calibration")
    if args.calibration_only:
        print("wrote phase_b1_calibration.json")
        return 0
    calibration = load_calibrations()
    sana_pointer_path = CALIBRATION_DIR / "sana_current.json"
    if not sana_pointer_path.exists():
        raise FileNotFoundError(f"missing calibration pointer {sana_pointer_path.relative_to(ROOT)}")
    sana_calibration = json.loads(sana_pointer_path.read_text(encoding="utf-8"))

    selector_jobs: list[dict[str, Any]] = []
    for slug, (model, _) in MODELS.items():
        pointer = calibration[slug]
        for dataset in ("wikitext", "c4"):
            selector_jobs.append(
                llm_job(
                    f"phaseb1_selector_{slug}_{dataset}_mse_reference",
                    "phase_b_selector",
                    model,
                    dataset,
                    layer_output_error_rows=16,
                )
            )
        for size in (32, 128, 256):
            file = pointer["files"][str(size)]["path"]
            for dataset in ("wikitext", "c4"):
                selector_jobs.append(
                    llm_job(
                        f"phaseb1_selector_{slug}_{dataset}_cal{size}", "phase_b_selector", model, dataset,
                        selector="output_aware", calibration_file=file, calibration_size=size,
                        layer_output_error_rows=16,
                    )
                )
    write("phase_b1_llm_selector", selector_jobs, "Ours-S LLM calibration-size stability")
    sana_selector_jobs = [
        sana_job(
            f"phaseb1_selector_sana_cal{size}",
            "phase_b_selector_diffusion",
            selector="output_aware",
            calibration_file=sana_calibration["files"][str(size)]["path"],
            calibration_size=size,
        )
        for size in (32, 128, 256)
    ]
    write("phase_b1_sana_selector", sana_selector_jobs, "Ours-S SANA calibration-size stability")

    permutation_jobs: list[dict[str, Any]] = []
    methods = (
        "no_permutation",
        "all_linear_sort_by_e0_ratio",
        "all_linear_margin_vector_clustering",
        "all_linear_greedy_min_regret_n8",
        "all_linear_sensitivity_weighted_greedy_n8",
        "foldable_mlp_sort_by_e0_ratio",
        "foldable_mlp_margin_vector_clustering",
        "foldable_mlp_greedy_min_regret_n8",
        "foldable_mlp_sensitivity_weighted_greedy_n8",
    )
    for slug, (model, _) in MODELS.items():
        identity_file = calibration[slug]["files"]["256"]["path"]
        for method in methods:
            calibration_file = identity_file if "sensitivity_weighted" in method else None
            calibration_size = 256 if calibration_file else 0
            for dataset in ("wikitext", "c4"):
                permutation_jobs.append(
                    llm_job(
                        f"phaseb2_permutation_{slug}_{dataset}_{method}", "phase_b_permutation", model, dataset,
                        permutation=method, calibration_file=calibration_file, calibration_size=calibration_size,
                    )
                )
    write("phase_b2_llm_permutation", permutation_jobs, "Ours-P upper-bound and exact-foldable LLM variants")
    sana_permutation_methods = (
        "no_permutation",
        "all_linear_sort_by_e0_ratio",
        "all_linear_margin_vector_clustering",
        "all_linear_greedy_min_regret_n8",
        "all_linear_sensitivity_weighted_greedy_n8",
        "foldable_sana_ffn_sort_by_e0_ratio",
        "foldable_sana_ffn_margin_vector_clustering",
        "foldable_sana_ffn_greedy_min_regret_n8",
        "foldable_sana_ffn_sensitivity_weighted_greedy_n8",
    )
    sana_identity = sana_calibration["files"]["256"]["path"]
    sana_permutation_jobs = [
        sana_job(
            f"phaseb2_permutation_sana_{method}",
            "phase_b_permutation_diffusion",
            permutation=method,
            calibration_file=sana_identity if "sensitivity_weighted" in method else None,
            calibration_size=256 if "sensitivity_weighted" in method else 0,
        )
        for method in sana_permutation_methods
    ]
    write("phase_b2_sana_permutation", sana_permutation_jobs, "Ours-P SANA upper-bound and exact-foldable FFN variants")

    rotation_jobs: list[dict[str, Any]] = []
    transforms = (
        "identity", "H16", "H32", "H64", "H128",
        "random_signed_H64_seed0", "random_signed_H64_seed1",
        "random_signed_H64_seed2", "random_signed_H64_seed3",
    )
    for slug, (model, _) in MODELS.items():
        for transform in transforms:
            for dataset in ("wikitext", "c4"):
                rotation_jobs.append(
                    llm_job(
                        f"phaseb3_rotation_{slug}_{dataset}_{transform}", "phase_b_rotation", model, dataset,
                        rotation=transform,
                    )
                )
    write("phase_b3_llm_rotation_bank", rotation_jobs, "Ours-R fixed transform bank; run only after the N/K gate")
    sana_rotation_jobs = [
        sana_job(
            f"phaseb3_rotation_sana_{transform}",
            "phase_b_rotation_diffusion",
            rotation=transform,
        )
        for transform in transforms
    ]
    write("phase_b3_sana_rotation_bank", sana_rotation_jobs, "Ours-R SANA transform bank; run only after the N/K gate")

    rotation_selection_jobs: list[dict[str, Any]] = []
    rotation_selection_ids: dict[str, str] = {}
    for slug, (model, _) in MODELS.items():
        selection_id = f"phaseb3_rotation_select_{slug}_cal256"
        rotation_selection_ids[slug] = selection_id
        rotation_selection_jobs.append(
            {
                "experiment_id": selection_id,
                "gpu_kind": "a6000",
                "log_file": f"artifacts/04_phase_b/rotation/logs/{selection_id}.log",
                "command": [
                    "python3", "scripts/select_rotation_map.py", "--experiment-id", selection_id,
                    "--domain", "llm", "--model", model,
                    "--calibration-file", calibration[slug]["files"]["256"]["path"],
                ],
            }
        )
    sana_selection_id = "phaseb3_rotation_select_sana_cal256"
    rotation_selection_jobs.append(
        {
            "experiment_id": sana_selection_id,
            "gpu_kind": "a6000",
            "log_file": f"artifacts/04_phase_b/rotation/logs/{sana_selection_id}.log",
            "command": [
                "python3", "scripts/select_rotation_map.py", "--experiment-id", sana_selection_id,
                "--domain", "diffusion", "--calibration-file", sana_identity,
            ],
        }
    )
    write(
        "phase_b3_rotation_selection",
        rotation_selection_jobs,
        "Calibration-only per-module transform selection for lambda=0,0.1,1,10; run after the N/K gate",
    )

    selected_llm_jobs: list[dict[str, Any]] = []
    selected_sana_jobs: list[dict[str, Any]] = []
    for selector_name in ("mse", "output_aware"):
        for lam, lam_slug in ((0.0, "0"), (0.1, "0p1"), (1.0, "1"), (10.0, "10")):
            for slug, (model, _) in MODELS.items():
                selection_id = rotation_selection_ids[slug]
                map_path = f"artifacts/04_phase_b/rotation/maps/{selection_id}_{selector_name}_lambda_{lam_slug}.json"
                for dataset in ("wikitext", "c4"):
                    selected_llm_jobs.append(
                        llm_job(
                            f"phaseb3_rotation_selected_{slug}_{dataset}_{selector_name}_lambda_{lam_slug}",
                            "phase_b_rotation_selected", model, dataset,
                            rotation_map=map_path,
                        )
                    )
            sana_map = f"artifacts/04_phase_b/rotation/maps/{sana_selection_id}_{selector_name}_lambda_{lam_slug}.json"
            selected_sana_jobs.append(
                sana_job(
                    f"phaseb3_rotation_selected_sana_{selector_name}_lambda_{lam_slug}",
                    "phase_b_rotation_selected_diffusion",
                    rotation="identity",
                )
            )
            selected_sana_jobs[-1]["command"].extend(("--rotation-map", sana_map))
    write(
        "phase_b3_llm_rotation_selected",
        selected_llm_jobs,
        "Evaluation of calibration-selected ordinary and granularity-aware per-module transform maps",
    )
    write(
        "phase_b3_sana_rotation_selected",
        selected_sana_jobs,
        "SANA evaluation of calibration-selected ordinary and granularity-aware per-module transform maps",
    )
    timestep_id = "phaseb4_sana_timestep_stability_proxy8"
    write(
        "phase_b4_sana_timestep",
        [
            {
                "experiment_id": timestep_id,
                "gpu_kind": "a6000",
                "log_file": f"artifacts/04_phase_b/timestep/logs/{timestep_id}.log",
                "command": [
                    "python3", "scripts/analyze_sana_timesteps.py",
                    "--experiment-id", timestep_id,
                    "--prompt-limit", "8",
                    "--weight-mode", "n8k64",
                    "--activation-mode", "m16k64",
                ],
            }
        ],
        "SANA early/mid/late plus intermediate timestep preference, locality, regret, and proxy stability",
    )
    print(
        json.dumps(
            {
                "llm_selector": len(selector_jobs), "sana_selector": len(sana_selector_jobs),
                "llm_permutation": len(permutation_jobs), "sana_permutation": len(sana_permutation_jobs),
                "llm_rotation": len(rotation_jobs), "sana_rotation": len(sana_rotation_jobs),
                "rotation_selection": len(rotation_selection_jobs),
                "llm_rotation_selected": len(selected_llm_jobs),
                "sana_rotation_selected": len(selected_sana_jobs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
