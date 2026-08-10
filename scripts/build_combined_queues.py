#!/usr/bin/env python3
"""Build provenance-matched Phase-B combination calibration/evaluation queues."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "artifacts" / "00_environment" / "queues" / "specs"
CALIBRATION = ROOT / "artifacts" / "04_phase_b" / "selector" / "calibration"
DEFAULT_NOMINATION = ROOT / "artifacts" / "04_phase_b" / "combined" / "phase_b_nominations.json"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write(name: str, jobs: list[dict[str, Any]], purpose: str) -> None:
    SPECS.mkdir(parents=True, exist_ok=True)
    (SPECS / f"{name}.json").write_text(
        json.dumps({"schema_version": 1, "generated_at": timestamp(), "purpose": purpose, "num_jobs": len(jobs), "jobs": jobs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def rotation_parts(config: dict[str, Any]) -> tuple[str, str | None, str]:
    rotation_map = config.get("rotation_map")
    rotation = config.get("rotation", "identity")
    label = f"per_module:{Path(rotation_map).stem}" if rotation_map else rotation
    return rotation, rotation_map, label


def pointer_path(model: str, permutation: str, rotation_label: str, *, sana: bool = False) -> Path:
    name = f"sana_{permutation}_{rotation_label}" if sana else f"{model}_{permutation}_{rotation_label}"
    return CALIBRATION / f"{slug(name)}_current.json"


def load_pointer(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"matched calibration pointer missing: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_jobs(nomination: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for model, choices in nomination["llm"].items():
        for label, use_p, use_r in (("p", True, False), ("r", False, True), ("pr", True, True)):
            permutation = choices["P"]["permutation"] if use_p else "none"
            rotation, rotation_map, _ = rotation_parts(choices["R"] if use_r else {})
            experiment_id = f"phaseb5_calibration_{choices['slug']}_{label}"
            command = [
                "python3", "scripts/capture_calibration.py", "--experiment-id", experiment_id,
                "--model", model, "--permutation", permutation, "--rotation", rotation,
                "--checkpoints", "256", "--seq-len", "128", "--seed", "314159",
            ]
            if "sensitivity_weighted" in permutation:
                command.extend(("--packing-calibration-file", choices["S"]["calibration_file"]))
            if rotation_map:
                command.extend(("--rotation-map", rotation_map))
            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "gpu_kind": "a6000",
                    "log_file": f"artifacts/04_phase_b/combined/logs/{experiment_id}.log",
                    "command": command,
                }
            )

    choices = nomination["sana"]
    for label, use_p, use_r in (("p", True, False), ("r", False, True), ("pr", True, True)):
        permutation = choices["P"]["permutation"] if use_p else "none"
        rotation, rotation_map, _ = rotation_parts(choices["R"] if use_r else {})
        experiment_id = f"phaseb5_calibration_sana_{label}"
        command = [
            "python3", "scripts/capture_sana_calibration.py", "--experiment-id", experiment_id,
            "--permutation", permutation, "--rotation", rotation, "--checkpoints", "256",
            "--steps", "20", "--guidance-scale", "4.5", "--batch-size", "8",
            "--max-activation-rows", "64",
        ]
        if "sensitivity_weighted" in permutation:
            command.extend(("--packing-calibration-file", choices["S"]["calibration_file"]))
        if rotation_map:
            command.extend(("--rotation-map", rotation_map))
        jobs.append(
            {
                "experiment_id": experiment_id,
                "gpu_kind": "a6000",
                "log_file": f"artifacts/04_phase_b/combined/logs/{experiment_id}.log",
                "command": command,
            }
        )
    return jobs


def llm_eval_job(
    experiment_id: str,
    model: str,
    dataset: str,
    *,
    weight_mode: str = "n8k64",
    activation_mode: str = "high_precision",
    selector: str = "mse",
    permutation: str = "none",
    rotation: str = "identity",
    rotation_map: str | None = None,
    calibration_file: str | None = None,
) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_llm_experiment.py", "--experiment-id", experiment_id,
        "--phase", "phase_b_combined", "--model", model, "--dataset", dataset,
        "--weight-mode", weight_mode, "--activation-mode", activation_mode,
        "--selector", selector, "--permutation", permutation, "--rotation", rotation,
        "--calibration-size", "256" if calibration_file else "0",
    ]
    if rotation_map:
        command.extend(("--rotation-map", rotation_map))
    if calibration_file:
        command.extend(("--calibration-file", calibration_file))
    return {
        "experiment_id": experiment_id,
        "gpu_kind": "a6000",
        "log_file": f"artifacts/04_phase_b/combined/logs/{experiment_id}.log",
        "command": command,
    }


def sana_eval_job(experiment_id: str, **config) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_sana_experiment.py", "--experiment-id", experiment_id,
        "--phase", "phase_b_combined_diffusion", "--weight-mode", config.get("weight_mode", "n8k64"),
        "--activation-mode", config.get("activation_mode", "high_precision"),
        "--selector", config.get("selector", "mse"), "--permutation", config.get("permutation", "none"),
        "--rotation", config.get("rotation", "identity"),
        "--calibration-size", "256" if config.get("calibration_file") else "0",
    ]
    if config.get("rotation_map"):
        command.extend(("--rotation-map", config["rotation_map"]))
    if config.get("calibration_file"):
        command.extend(("--calibration-file", config["calibration_file"]))
    return {
        "experiment_id": experiment_id,
        "gpu_kind": "a6000",
        "log_file": f"artifacts/04_phase_b/combined/logs/{experiment_id}.log",
        "command": command,
    }


def endpoint_configs(model: str, choices: dict[str, Any], *, sana: bool) -> dict[str, dict[str, Any]]:
    p = choices["P"]["permutation"]
    r, rotation_map, r_label = rotation_parts(choices["R"])
    identity_cal = choices["S"]["calibration_file"]
    p_cal = load_pointer(pointer_path(model, p, "identity", sana=sana))["files"]["256"]["path"]
    r_cal = load_pointer(pointer_path(model, "none", r_label, sana=sana))["files"]["256"]["path"]
    pr_cal = load_pointer(pointer_path(model, p, r_label, sana=sana))["files"]["256"]["path"]
    return {
        "raw": {},
        "s": {"selector": "output_aware", "calibration_file": identity_cal},
        "p": {"permutation": p},
        "r": {"rotation": r, "rotation_map": rotation_map},
        "sp": {"selector": "output_aware", "permutation": p, "calibration_file": p_cal},
        "sr": {"selector": "output_aware", "rotation": r, "rotation_map": rotation_map, "calibration_file": r_cal},
        "pr": {"permutation": p, "rotation": r, "rotation_map": rotation_map},
        "spr": {"selector": "output_aware", "permutation": p, "rotation": r, "rotation_map": rotation_map, "calibration_file": pr_cal},
    }


def evaluation_jobs(nomination: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    llm_jobs: list[dict] = []
    for model, choices in nomination["llm"].items():
        configs = endpoint_configs(model, choices, sana=False)
        for label, config in configs.items():
            for dataset in ("wikitext", "c4"):
                experiment_id = f"phaseb5_combined_{choices['slug']}_{dataset}_{label}"
                llm_jobs.append(llm_eval_job(experiment_id, model, dataset, **config))
        # The W4A4 endpoint keeps activation adaptation raw and applies Ours to
        # weights only; this is well-defined and explicitly labeled.
        for dataset in ("wikitext", "c4"):
            experiment_id = f"phaseb5_combined_{choices['slug']}_{dataset}_spr_w4a4"
            llm_jobs.append(llm_eval_job(experiment_id, model, dataset, activation_mode="m16k64", **configs["spr"]))
        for label, config in (("4over6_raw", configs["raw"]), ("4over6_spr", configs["spr"])):
            for dataset in ("wikitext", "c4"):
                experiment_id = f"phaseb5_combined_{choices['slug']}_{dataset}_{label}"
                llm_jobs.append(llm_eval_job(experiment_id, model, dataset, weight_mode="n8k64_4over6", **config))

    sana_choices = nomination["sana"]
    sana_configs = endpoint_configs("sana", sana_choices, sana=True)
    sana_jobs = [sana_eval_job(f"phaseb5_combined_sana_{label}", **config) for label, config in sana_configs.items()]
    sana_jobs.append(sana_eval_job("phaseb5_combined_sana_spr_w4a4", activation_mode="m16k64", **sana_configs["spr"]))
    sana_jobs.append(sana_eval_job("phaseb5_combined_sana_4over6_raw", weight_mode="n8k64_4over6", **sana_configs["raw"]))
    sana_jobs.append(sana_eval_job("phaseb5_combined_sana_4over6_spr", weight_mode="n8k64_4over6", **sana_configs["spr"]))
    return llm_jobs, sana_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--nominations", type=Path, default=DEFAULT_NOMINATION)
    args = parser.parse_args()
    path = args.nominations if args.nominations.is_absolute() else ROOT / args.nominations
    nomination = json.loads(path.read_text(encoding="utf-8"))
    if args.stage == "calibration":
        jobs = calibration_jobs(nomination)
        write("phase_b5_combination_calibration", jobs, "Matched 256-sample calibration for P, R, and P+R layouts")
        print(json.dumps({"calibration_jobs": len(jobs)}, indent=2))
        return 0
    llm, sana = evaluation_jobs(nomination)
    write("phase_b5_llm_combined", llm, "Controlled raw/S/P/R/SP/SR/PR/SPR and matched 4Over6 LLM endpoints")
    write("phase_b5_sana_combined", sana, "Controlled raw/S/P/R/SP/SR/PR/SPR and matched 4Over6 SANA endpoints")
    print(json.dumps({"llm_jobs": len(llm), "sana_jobs": len(sana)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
