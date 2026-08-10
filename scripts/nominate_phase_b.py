#!/usr/bin/env python3
"""Nominate deployable Ours-S/P/R components from individual Phase-B results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
OUTPUT = ROOT / "artifacts" / "04_phase_b" / "combined" / "phase_b_nominations.json"
CALIBRATION = ROOT / "artifacts" / "04_phase_b" / "selector" / "calibration"


MODEL_SLUGS = {
    "meta-llama/Llama-3.1-8B": "llama31_8b",
    "Qwen/Qwen3-8B": "qwen3_8b",
}


def load_master() -> pd.DataFrame:
    path = FINAL / "master_results.csv"
    if not path.exists():
        raise FileNotFoundError("aggregate master_results.csv before nomination")
    return pd.read_csv(path)


def row_config(row: pd.Series) -> dict[str, Any]:
    def value(name: str, default=None):
        item = row.get(name, default)
        return default if pd.isna(item) else item

    return {
        "source_experiment_id": value("experiment_id"),
        "weight_mode": value("weight_format_mode", "n8k64"),
        "selector": value("selector", "mse"),
        "permutation": value("permutation", "none"),
        "rotation": value("rotation", "identity") if not str(value("rotation", "identity")).startswith("per_module:") else "identity",
        "rotation_label": value("rotation", "identity"),
        "rotation_map": value("rotation_map"),
        "ppl": value("ppl"),
        "proxy_nmse": value("proxy_nmse"),
    }


def best_llm(frame: pd.DataFrame, model: str, kind: str) -> dict[str, Any]:
    rows = frame[(frame["model"] == model) & frame["ppl"].notna()].copy()
    if kind == "P":
        rows = rows[
            rows["phase"].astype(str).str.contains("permutation", na=False)
            & rows["permutation"].astype(str).str.startswith("foldable_mlp_")
        ]
        key = "permutation"
    else:
        rows = rows[rows["phase"].astype(str).str.contains("rotation", na=False)]
        rows = rows[
            rows["rotation"].astype(str).ne("identity")
            | rows.get("rotation_map", pd.Series(index=rows.index)).notna()
        ]
        key = "rotation"
    if rows.empty:
        raise RuntimeError(f"no deployable {kind} rows completed for {model}")
    grouped = rows.groupby(key, dropna=False)["ppl"].mean().sort_values()
    winner = grouped.index[0]
    candidates = rows[rows[key].fillna("<none>") == ("<none>" if pd.isna(winner) else winner)]
    if candidates.empty:  # rotation maps share a display label but retain path per row.
        candidates = rows.loc[[rows["ppl"].idxmin()]]
    representative = candidates.sort_values("ppl").iloc[0]
    result = row_config(representative)
    result["selection_mean_ppl"] = float(grouped.iloc[0])
    result["selection_rows"] = int(len(candidates))
    return result


def best_sana(frame: pd.DataFrame, kind: str) -> dict[str, Any]:
    rows = frame[(frame["domain"] == "diffusion") & frame["proxy_nmse"].notna()].copy()
    if kind == "P":
        rows = rows[
            rows["phase"].astype(str).str.contains("permutation", na=False)
            & rows["permutation"].astype(str).str.startswith("foldable_sana_ffn_")
        ]
    else:
        rows = rows[
            rows["phase"].astype(str).str.contains("rotation", na=False)
            & (
                rows["rotation"].astype(str).ne("identity")
                | rows.get("rotation_map", pd.Series(index=rows.index)).notna()
            )
        ]
    if rows.empty:
        raise RuntimeError(f"no deployable SANA {kind} rows completed")
    return row_config(rows.sort_values("proxy_nmse").iloc[0])


def pointer(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    master = load_master()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "selection_policy": (
            "P: lowest mean WikiText-2/C4 PPL among exact-foldable variants; "
            "R: lowest mean PPL/proxy NMSE among calibrated block transforms; "
            "S: fixed output-aware selector with the predeclared 256-sample calibration"
        ),
        "llm": {},
        "sana": {},
    }
    for model, slug in MODEL_SLUGS.items():
        identity_pointer = pointer(
            CALIBRATION / ("meta_llama_llama_3_1_8b_none_identity_current.json" if slug == "llama31_8b" else "qwen_qwen3_8b_none_identity_current.json")
        )
        payload["llm"][model] = {
            "slug": slug,
            "S": {
                "selector": "output_aware",
                "calibration_size": 256,
                "calibration_file": identity_pointer["files"]["256"]["path"],
            },
            "P": best_llm(master, model, "P"),
            "R": best_llm(master, model, "R"),
        }
    sana_identity = pointer(CALIBRATION / "sana_current.json")
    payload["sana"] = {
        "S": {
            "selector": "output_aware",
            "calibration_size": 256,
            "calibration_file": sana_identity["files"]["256"]["path"],
        },
        "P": best_sana(master, "P"),
        "R": best_sana(master, "R"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
