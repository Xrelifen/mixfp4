#!/usr/bin/env python3
"""Build matched A6000/RTX 6000 Ada numerical-reproducibility queues."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "artifacts" / "00_environment" / "queues" / "specs"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write(name: str, jobs: list[dict], purpose: str) -> None:
    SPECS.mkdir(parents=True, exist_ok=True)
    (SPECS / f"{name}.json").write_text(
        json.dumps(
            {"schema_version": 1, "generated_at": timestamp(), "purpose": purpose, "num_jobs": len(jobs), "jobs": jobs},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-weight-mode")
    parser.add_argument("--best-selector", default="mse")
    parser.add_argument("--best-permutation", default="none")
    parser.add_argument("--best-rotation", default="identity")
    parser.add_argument("--best-calibration-file")
    parser.add_argument("--best-calibration-size", type=int, default=0)
    parser.add_argument("--best-rotation-map")
    args = parser.parse_args()

    modes = [(name, name, "mse", "none", "identity", None, 0, None) for name in (
        "high_precision", "nvfp4", "oracle16", "n8k64"
    )]
    if args.best_weight_mode:
        modes.append(
            (
                "best_phase_b", args.best_weight_mode, args.best_selector,
                args.best_permutation, args.best_rotation, args.best_calibration_file,
                args.best_calibration_size, args.best_rotation_map,
            )
        )

    llm_jobs = []
    sana_jobs = []
    for kind in ("a6000", "ada"):
        for label, weight, selector, permutation, rotation, calibration, size, rotation_map in modes:
            llm_id = f"cross_gpu_llama32_1b_{kind}_{label}"
            llm_command = [
                "python3", "scripts/run_llm_experiment.py", "--experiment-id", llm_id,
                "--phase", "cross_gpu_llm", "--model", "meta-llama/Llama-3.2-1B",
                "--dataset", "wikitext", "--eval-limit", "8", "--weight-mode", weight,
                "--activation-mode", "high_precision", "--selector", selector,
                "--permutation", permutation, "--rotation", rotation,
                "--calibration-size", str(size),
            ]
            if calibration:
                llm_command.extend(("--calibration-file", calibration))
            if rotation_map:
                llm_command.extend(("--rotation-map", rotation_map))
            llm_jobs.append(
                {
                    "experiment_id": llm_id,
                    "gpu_kind": kind,
                    "log_file": f"artifacts/05_cross_gpu/logs/{llm_id}.log",
                    "command": llm_command,
                }
            )

            sana_id = f"cross_gpu_sana_{kind}_{label}"
            sana_command = [
                "python3", "scripts/run_sana_experiment.py", "--experiment-id", sana_id,
                "--phase", "cross_gpu_diffusion", "--prompt-limit", "2", "--layer-error-prompts", "1",
                "--weight-mode", weight, "--activation-mode", "high_precision", "--selector", selector,
                "--permutation", permutation, "--rotation", rotation, "--calibration-size", str(size),
            ]
            if calibration:
                sana_command.extend(("--calibration-file", calibration))
            if rotation_map:
                sana_command.extend(("--rotation-map", rotation_map))
            sana_jobs.append(
                {
                    "experiment_id": sana_id,
                    "gpu_kind": kind,
                    "log_file": f"artifacts/05_cross_gpu/logs/{sana_id}.log",
                    "command": sana_command,
                }
            )
    write("cross_gpu_llm", llm_jobs, "Matched smoke-LLM numerical results on one A6000 and one RTX 6000 Ada")
    write("cross_gpu_sana", sana_jobs, "Matched two-prompt SANA numerical results on one A6000 and one RTX 6000 Ada")
    print(json.dumps({"llm_jobs": len(llm_jobs), "sana_jobs": len(sana_jobs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
