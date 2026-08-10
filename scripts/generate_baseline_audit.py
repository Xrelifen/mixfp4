#!/usr/bin/env python3
"""Generate the Phase-0 reproduction audit from immutable completed attempts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
OUT = ROOT / "artifacts" / "01_repo_audit"


def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    value = frame[available].copy()
    if value.empty:
        return "_No completed measurements._"
    for column in value.columns:
        if pd.api.types.is_float_dtype(value[column]):
            value[column] = value[column].map(
                lambda item: "" if pd.isna(item) else f"{item:.10g}"
            )
    header = "| " + " | ".join(value.columns) + " |"
    divider = "| " + " | ".join("---" for _ in value.columns) + " |"
    body = [
        "| " + " | ".join(str(item) for item in row) + " |"
        for row in value.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *body))


def main() -> int:
    master = pd.read_csv(FINAL / "master_results.csv")
    smoke_all = master[
        master["experiment_id"].astype(str).str.startswith("phase0_smoke_llama32_1b_")
    ].copy()
    superseded_path = FINAL / "superseded_results.csv"
    superseded_all = (
        pd.read_csv(superseded_path)
        if superseded_path.exists() and superseded_path.stat().st_size
        else pd.DataFrame()
    )
    superseded_smoke = (
        superseded_all[
            superseded_all["experiment_id"].astype(str).str.startswith(
                "phase0_smoke_llama32_1b_"
            )
        ].copy()
        if not superseded_all.empty
        else pd.DataFrame()
    )
    corrected_modes = {"nvfp4_4over6", "oracle16_4over6", "n8k64_4over6"}
    smoke_all["baseline_validity"] = "current"
    smoke_all["superseded_by"] = ""
    for mode in corrected_modes:
        mode_rows = smoke_all[smoke_all["weight_format_mode"] == mode]
        validated = mode_rows[
            mode_rows["experiment_id"].astype(str).str.contains("reduction_validated", regex=False)
        ]
        if validated.empty:
            continue
        validated_id = str(validated.sort_values("end_time").iloc[-1]["experiment_id"])
        old_index = mode_rows.index.difference(validated.index)
        smoke_all.loc[old_index, "baseline_validity"] = "superseded_arithmetic_order"
        smoke_all.loc[old_index, "superseded_by"] = validated_id
        smoke_all.loc[validated.index, "baseline_validity"] = "validated_current"
    smoke = smoke_all[smoke_all["baseline_validity"] != "superseded_arithmetic_order"].copy()
    if not superseded_smoke.empty:
        superseded_smoke["baseline_validity"] = "superseded_arithmetic_order"
        if "superseded_by_experiment_id" in superseded_smoke:
            superseded_smoke["superseded_by"] = superseded_smoke[
                "superseded_by_experiment_id"
            ]
    formal_sana = master[
        master["experiment_id"].astype(str).isin(
            (
                "phase0_smoke_sana_reference_a6000",
                "phase0_sana_reference_proxy16_steps20_fp32_trajectory",
            )
        )
    ].copy()
    # Include the first completed SANA quantized proxy endpoints as Phase-0
    # numerical reproduction evidence once Phase A2 has populated them.
    sana_baselines = master[
        master["phase"].astype(str).eq("phase_a2_diffusion")
        & master["weight_format_mode"].astype(str).isin(
            ("high_precision", "nvfp4", "fouroversix_4over6")
        )
    ].copy()
    table = pd.concat(
        (smoke_all, superseded_smoke, formal_sana, sana_baselines),
        ignore_index=True,
    )
    table.to_csv(OUT / "phase0_baseline_results.csv", index=False)
    pd.concat((smoke, formal_sana, sana_baselines), ignore_index=True).to_csv(
        OUT / "phase0_current_baseline_results.csv", index=False
    )

    by_mode = {
        row.weight_format_mode: row
        for row in smoke.itertuples()
        if isinstance(row.weight_format_mode, str)
    }
    comparisons: list[dict] = []
    for label, left, right in (
        ("project NVFP4 vs original RaZeR NVFP4", "nvfp4", "nvfp4_original"),
        ("project Oracle16 vs original RaZeR NVIF4", "oracle16", "nvif4_original"),
        ("project canonical E2 4Over6 vs pinned public FourOverSix", "nvfp4_4over6", "fouroversix_4over6"),
        ("project Oracle16 vs pinned FourOverSix IF4", "oracle16", "fouroversix_if4"),
    ):
        if left in by_mode and right in by_mode:
            comparisons.append(
                {
                    "comparison": label,
                    "left_mode": left,
                    "right_mode": right,
                    "ppl_left": by_mode[left].ppl,
                    "ppl_right": by_mode[right].ppl,
                    "ppl_difference": by_mode[left].ppl - by_mode[right].ppl,
                    "weight_mse_left": by_mode[left].weight_mse,
                    "weight_mse_right": by_mode[right].weight_mse,
                    "weight_mse_difference": by_mode[left].weight_mse
                    - by_mode[right].weight_mse,
                }
            )
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame.to_csv(OUT / "phase0_cross_repo_comparisons.csv", index=False)

    determinism_path = ROOT / "artifacts" / "02_tests" / "diffusion_determinism.json"
    determinism = (
        json.loads(determinism_path.read_text(encoding="utf-8"))
        if determinism_path.exists()
        else None
    )
    report = """# Phase 0 baseline reproduction

All rows below are real guarded measurements from immutable attempt directories. PPL values in the smoke table use one fixed WikiText-2 sequence and are correctness checks, not headline quality results; the Phase A1 tables use the complete persisted slices.

## Current LLM smoke baselines

{llm_table}

## Preserved superseded attempts

The rows below exposed a floating-point operation-order mismatch in the first
project 4Over6 reconstruction. They remain immutable provenance, but are not
used in any comparison or decision.

{superseded_table}

## Cross-repository numerical comparisons

{comparison_table}

The project NVFP4 path is bit-exact with the original RaZeR NVFP4 path. The small project-Oracle/NVIF4 difference is fully diagnosed in `razer_real_weight_semantics.json`: 1,117/60,817,408 real K16 choices differ solely from float32 selector-reduction order. Canonical fixed-E2 4Over6 is independently reduced bit-exactly in `four_over_six_reduction.json`; project E0 composition endpoints remain explicitly labeled as project composition.

## SANA fixed-reference reproduction

{sana_table}

Fixed-prompt/fixed-seed image determinism: `{determinism}`. Exact scheduler/component/prompt/seed provenance is stored alongside the SANA reference attempt and in `artifacts/02_tests/diffusion_determinism.json`.
""".format(
        llm_table=markdown(
            smoke.sort_values("weight_format_mode"),
            [
                "experiment_id",
                "weight_format_mode",
                "baseline_validity",
                "ppl",
                "weight_mse",
                "weight_e0_ratio",
                "weight_granularity_regret",
                "physical_gpu_index",
                "gpu_type",
            ],
        ),
        superseded_table=markdown(
            superseded_smoke.sort_values("weight_format_mode")
            if not superseded_smoke.empty
            else superseded_smoke,
            [
                "experiment_id",
                "weight_format_mode",
                "baseline_validity",
                "superseded_by",
                "ppl",
                "weight_mse",
            ],
        ),
        comparison_table=markdown(
            comparisons_frame,
            [
                "comparison",
                "ppl_left",
                "ppl_right",
                "ppl_difference",
                "weight_mse_difference",
            ],
        ),
        sana_table=markdown(
            pd.concat((formal_sana, sana_baselines), ignore_index=True),
            [
                "experiment_id",
                "weight_format_mode",
                "num_proxy_prompts",
                "proxy_nmse",
                "latent_trajectory_nmse",
                "physical_gpu_index",
                "gpu_type",
            ],
        ),
        determinism=(
            f"bit_exact={determinism.get('bit_exact')}, steps={determinism.get('steps')}, "
            f"max_abs_difference={determinism.get('max_abs_difference')}"
            if determinism
            else "not yet measured"
        ),
    )
    (OUT / "baseline_reproduction.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "llm_smoke_rows": len(smoke),
                "preserved_superseded_llm_smoke_rows": int(
                    len(superseded_smoke)
                ),
                "sana_rows": len(formal_sana) + len(sana_baselines),
                "comparison_rows": len(comparisons_frame),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
