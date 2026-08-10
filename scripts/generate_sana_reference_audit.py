#!/usr/bin/env python3
"""Audit the FP32 SANA trajectory-reference correction and matched reruns."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
OUT = ROOT / "artifacts" / "01_repo_audit"


def load(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def table(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "reference_generation", "experiment_id", "weight_format_mode",
            "proxy_nmse", "latent_trajectory_nmse", "final_latent_nmse",
            "physical_gpu_index", "gpu_type",
        )
        if column in frame
    ]
    if frame.empty:
        return "_No matched measurements yet._"
    value = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(item) else str(item) for item in row) + " |"
        for row in value.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *body))


def main() -> int:
    current = load(FINAL / "master_results.csv")
    old = load(FINAL / "superseded_results.csv")
    current = current[
        current["experiment_id"].astype(str).str.match(
            r"phasea2_sana_proxy_.*_fp32ref$"
        )
    ].copy() if not current.empty else current
    old = old[
        old["experiment_id"].astype(str).isin(
            {
                "phasea2_sana_proxy_high_precision",
                "phasea2_sana_proxy_nvfp4",
                "phasea2_sana_proxy_all_e0m3",
            }
        )
    ].copy() if not old.empty else old
    if not old.empty:
        old["reference_generation"] = "superseded_bf16_trajectory"
    if not current.empty:
        current["reference_generation"] = "validated_fp32_trajectory"
    combined = pd.concat((old, current), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "sana_reference_storage_comparison.csv", index=False)

    determinism_path = ROOT / "artifacts" / "02_tests" / "diffusion_determinism.json"
    determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
    replay = determinism.get("serialized_trajectory_replay", {})
    hp_old = old[old.get("weight_format_mode", pd.Series(index=old.index)).eq("high_precision")]
    hp_new = current[current.get("weight_format_mode", pd.Series(index=current.index)).eq("high_precision")]
    old_floor = None if hp_old.empty else float(hp_old.iloc[-1]["latent_trajectory_nmse"])
    new_floor = None if hp_new.empty else float(hp_new.iloc[-1]["latent_trajectory_nmse"])
    report = f"""# SANA reference-storage correction

The first formal trajectory files cast scheduler latents to BF16.  That kept
teacher-forced denoiser predictions valid, but introduced a high-precision
trajectory NMSE floor of `{old_floor}`.  Those attempts are preserved in
`superseded_results.csv` and excluded from decisions.

The corrected reference stores trajectories, teacher-forced inputs, and
predictions in FP32.  Its serialized replay is bit-exact:

- trajectory bit-exact: `{replay.get('trajectory_bit_exact')}`
- trajectory max absolute difference: `{replay.get('trajectory_max_abs_difference')}`
- prediction bit-exact: `{replay.get('prediction_bit_exact')}`
- prediction max absolute difference: `{replay.get('prediction_max_abs_difference')}`
- corrected HP trajectory NMSE floor: `{new_floor}`

## Preserved and corrected measurements

{table(combined)}

Machine-readable source: `artifacts/01_repo_audit/sana_reference_storage_comparison.csv`.
"""
    (OUT / "sana_reference_storage_correction.md").write_text(report, encoding="utf-8")
    print(json.dumps({"old_rows": len(old), "current_rows": len(current), "replay": replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
