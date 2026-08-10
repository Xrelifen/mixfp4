#!/usr/bin/env python3
"""Materialize stable, reviewable Phase-A LLM GPU job queues."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "00_environment" / "queues" / "specs"

MODELS = {
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def llm_job(
    *,
    experiment_id: str,
    phase: str,
    model: str,
    dataset: str,
    weight_mode: str,
    activation_mode: str = "high_precision",
    eval_limit: int | None = None,
    gpu_kind: str = "a6000",
    selector: str = "mse",
    permutation: str = "none",
    rotation: str = "identity",
) -> dict[str, Any]:
    command = [
        "python3", "scripts/run_llm_experiment.py",
        "--experiment-id", experiment_id,
        "--phase", phase,
        "--model", model,
        "--dataset", dataset,
        "--weight-mode", weight_mode,
        "--activation-mode", activation_mode,
        "--selector", selector,
        "--permutation", permutation,
        "--rotation", rotation,
    ]
    if eval_limit is not None:
        command.extend(("--eval-limit", str(eval_limit)))
    return {
        "experiment_id": experiment_id,
        "gpu_kind": gpu_kind,
        "log_file": f"artifacts/03_phase_a/llm/logs/{experiment_id}.log",
        "command": command,
    }


def write(name: str, jobs: list[dict[str, Any]], purpose: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": timestamp(),
        "purpose": purpose,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }
    (OUTPUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def smoke_jobs() -> list[dict[str, Any]]:
    modes = (
        "nvfp4_original",
        "nvif4_original",
        "razer_context_baseline",
        "fouroversix_nvfp4",
        "fouroversix_4over6",
        "fouroversix_if4",
        "all_e0m3",
        "oracle16",
        "k64_row",
        "n8k16",
        "n8k64",
        "oracle16_4over6",
        "n8k64_4over6",
    )
    return [
        llm_job(
            experiment_id=f"phase0_smoke_llama32_1b_wikitext_{mode}_a6000",
            phase="phase0_llm",
            model="meta-llama/Llama-3.2-1B",
            dataset="wikitext",
            weight_mode=mode,
            eval_limit=1,
            gpu_kind="a6000",
        )
        for mode in modes
    ]


def phase_a0_jobs() -> list[dict[str, Any]]:
    modes = (
        "high_precision", "nvfp4", "all_e0m3", "oracle16", "k64_row",
        "n8k16", "n8k64", "fouroversix_4over6", "oracle16_4over6",
        "n8k64_4over6",
    )
    jobs: list[dict[str, Any]] = []
    for model_slug, model in MODELS.items():
        for mode in modes:
            jobs.append(
                llm_job(
                    experiment_id=f"phasea0_{model_slug}_wikitext_n8_{mode}",
                    phase="phase_a0_llm",
                    model=model,
                    dataset="wikitext",
                    weight_mode=mode,
                    eval_limit=8,
                )
            )
    return jobs


def phase_a1_jobs() -> list[dict[str, Any]]:
    modes = (
        "high_precision", "nvfp4", "all_e0m3", "oracle16", "k32_row",
        "k64_row", "n8k16", "n2k64", "n4k64", "n8k64", "n16k64",
        "n32k64", "n64k64", "layer", "fouroversix_nvfp4",
        "fouroversix_4over6", "nvfp4_4over6", "razer_context_baseline",
        "nvif4_original", "fouroversix_if4", "oracle16_4over6",
        "n8k64_4over6",
    )
    jobs: list[dict[str, Any]] = []
    for model_slug, model in MODELS.items():
        for dataset in ("wikitext", "c4"):
            for mode in modes:
                jobs.append(
                    llm_job(
                        experiment_id=f"phasea1_{model_slug}_{dataset}_{mode}",
                        phase="phase_a1_llm",
                        model=model,
                        dataset=dataset,
                        weight_mode=mode,
                    )
                )
    return jobs


def phase_a3_jobs() -> list[dict[str, Any]]:
    comparisons = (
        ("oracle16", "oracle16_a"),
        ("oracle16", "all_e2m1"),
        ("all_e2m1", "oracle16_a"),
        ("n8k64", "all_e2m1"),
        ("all_e2m1", "m16k64"),
        ("n8k64", "m16k64"),
        ("n8k64", "all_e0m3"),
        ("all_e0m3", "m16k64"),
        ("all_e0m3", "all_e0m3"),
        ("nvfp4", "all_e2m1"),
    )
    # Cover every required activation-region definition in a consistent
    # one-side study in addition to the eight mandated paired comparisons.
    activation_sweep = tuple(
        ("all_e2m1", mode)
        for mode in ("k64_row_a", "m16k16", "m4k64", "m8k64", "m32k64")
    )
    jobs: list[dict[str, Any]] = []
    for model_slug, model in MODELS.items():
        for dataset in ("wikitext", "c4"):
            for weight_mode, activation_mode in comparisons + activation_sweep:
                label = f"w_{weight_mode}_a_{activation_mode}"
                jobs.append(
                    llm_job(
                        experiment_id=f"phasea3_{model_slug}_{dataset}_{label}",
                        phase="phase_a3_llm",
                        model=model,
                        dataset=dataset,
                        weight_mode=weight_mode,
                        activation_mode=activation_mode,
                    )
                )
    return jobs


def main() -> int:
    write("phase0_smoke", smoke_jobs(), "Phase-0 reference and MixFP4 smoke baselines")
    write("phase_a0_llm", phase_a0_jobs(), "Fast 8-sequence N/K feasibility gate")
    write("phase_a1_llm", phase_a1_jobs(), "Full fixed-slice LLM W4A16 sweep")
    write("phase_a3_llm", phase_a3_jobs(), "Full fixed-slice LLM W4A4 one/two-side study")
    print(json.dumps({path.name: json.loads(path.read_text())["num_jobs"] for path in sorted(OUTPUT.glob("*.json"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
