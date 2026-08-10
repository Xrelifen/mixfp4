#!/usr/bin/env python3
"""Materialize Phase-A N/K decomposition and Oracle-retention tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
OUTPUT = ROOT / "artifacts" / "03_phase_a" / "summaries"


def load(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(FINAL / name)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = load("master_results.csv")
    layers = load("per_layer_metrics.csv")
    decomposition = pd.DataFrame()
    if not master.empty and not layers.empty:
        metadata = master[["experiment_id", "domain", "model", "dataset_or_promptset", "weight_format_mode"]]
        source = layers.merge(metadata, on=["experiment_id", "domain", "model"], how="inner")
        source = source[
            source["operand_role"].astype(str).eq("weight_b")
            & source["weight_format_mode"].isin(("k64_row", "n8k16", "n8k64"))
        ]
        keys = ["domain", "model", "dataset_or_promptset", "layer_idx", "module_name", "module_type"]
        if not source.empty:
            grouped = source.groupby(keys + ["weight_format_mode"], dropna=False, as_index=False).agg(
                granularity_regret=("granularity_regret", "mean"),
                normalized_regret=("normalized_regret", "mean"),
                mean_homogeneity=("mean_homogeneity", "mean"),
                mean_margin_conflict=("mean_margin_conflict", "mean"),
            )
            regret = grouped.pivot_table(index=keys, columns="weight_format_mode", values="granularity_regret").reset_index()
            normalized = grouped.pivot_table(index=keys, columns="weight_format_mode", values="normalized_regret").reset_index()
            for mode in ("k64_row", "n8k16", "n8k64"):
                if mode not in regret:
                    regret[mode] = np.nan
                if mode not in normalized:
                    normalized[mode] = np.nan
            decomposition = regret.rename(
                columns={
                    "k64_row": "regret_oracle_to_k64",
                    "n8k16": "regret_oracle_to_n8k16",
                    "n8k64": "regret_oracle_to_n8k64",
                }
            )
            decomposition["n_k_interaction_residual"] = (
                decomposition["regret_oracle_to_n8k64"]
                - decomposition["regret_oracle_to_k64"]
                - decomposition["regret_oracle_to_n8k16"]
            )
            normalized = normalized.rename(
                columns={
                    "k64_row": "normalized_regret_k64",
                    "n8k16": "normalized_regret_n8k16",
                    "n8k64": "normalized_regret_n8k64",
                }
            )
            decomposition = decomposition.merge(normalized, on=keys, how="left")
    decomposition.to_csv(OUTPUT / "n_k_conflict_decomposition.csv", index=False)

    retention = pd.DataFrame()
    if not master.empty:
        retention = master[
            master["weight_format_mode"].isin(("oracle16", "k64_row", "n8k16", "n8k64"))
        ][
            [
                "experiment_id", "phase", "domain", "model", "dataset_or_promptset",
                "weight_format_mode", "activation_format_mode", "ppl", "proxy_nmse",
                "ppl_delta_vs_nvfp4", "oracle_gain_retention", "weight_granularity_regret",
            ]
        ]
    retention.to_csv(OUTPUT / "oracle_gain_retention.csv", index=False)

    gate = {
        "num_decomposition_rows": len(decomposition),
        "num_retention_rows": len(retention),
        "oracle_gate_ready": bool(
            not master.empty
            and len(
                master[
                    master["phase"].astype(str).eq("phase_a1_llm")
                    & master["weight_format_mode"].isin(("nvfp4", "oracle16"))
                ]
            )
            >= 8
        ),
        "n_k_gate_ready": bool(
            not decomposition.empty
            and decomposition[["regret_oracle_to_k64", "regret_oracle_to_n8k16", "regret_oracle_to_n8k64"]].notna().all(axis=1).any()
        ),
        "rotation_may_start_only_after_n_k_gate_ready": True,
    }
    (OUTPUT / "phase_a_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
