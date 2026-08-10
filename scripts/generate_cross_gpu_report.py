#!/usr/bin/env python3
"""Compare matched A6000 and RTX 6000 Ada cross-GPU experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
OUTPUT = ROOT / "artifacts" / "05_cross_gpu"


def load(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(FINAL / name)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def comparison(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    source = frame[(frame["domain"] == domain) & frame["phase"].astype(str).str.startswith("cross_gpu")].copy()
    if source.empty:
        return pd.DataFrame()
    source["architecture"] = source["gpu_type"].map(
        lambda value: "a6000" if value == "NVIDIA RTX A6000" else "6000ada" if "6000 Ada" in str(value) else str(value)
    )
    keys = [
        "model", "dataset_or_promptset", "weight_format_mode", "activation_format_mode",
        "selector", "permutation", "rotation",
    ]
    metrics = [
        column for column in (
            "ppl", "proxy_nmse", "weight_mse", "weight_e0_ratio",
            "weight_homogeneity", "weight_margin_conflict", "weight_granularity_regret",
        ) if column in source
    ]
    rows = []
    for key, group in source.groupby(keys, dropna=False):
        by_arch = {row["architecture"]: row for _, row in group.iterrows()}
        row = dict(zip(keys, key, strict=True))
        for metric in metrics:
            left = by_arch.get("a6000", {}).get(metric, np.nan)
            right = by_arch.get("6000ada", {}).get(metric, np.nan)
            row[f"{metric}_a6000"] = left
            row[f"{metric}_6000ada"] = right
            row[f"{metric}_absolute_difference"] = right - left if pd.notna(left) and pd.notna(right) else np.nan
            row[f"{metric}_relative_difference"] = (
                (right - left) / max(abs(left), 1e-30) if pd.notna(left) and pd.notna(right) else np.nan
            )
        row["matched"] = "a6000" in by_arch and "6000ada" in by_arch
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No matched cross-GPU measurements completed._"
    columns = [column for column in frame if column in {
        "model", "dataset_or_promptset", "weight_format_mode", "activation_format_mode", "matched",
        "ppl_a6000", "ppl_6000ada", "ppl_absolute_difference",
        "proxy_nmse_a6000", "proxy_nmse_6000ada", "proxy_nmse_absolute_difference",
        "weight_e0_ratio_absolute_difference", "weight_granularity_regret_relative_difference",
    }]
    value = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join("" if pd.isna(item) else str(item) for item in row) + " |" for row in value.itertuples(index=False, name=None)]
    return "\n".join((header, separator, *body))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = load("master_results.csv")
    llm = comparison(master, "llm")
    diffusion = comparison(master, "diffusion")
    llm.to_csv(OUTPUT / "llm_a6000_vs_6000ada.csv", index=False)
    diffusion.to_csv(OUTPUT / "diffusion_a6000_vs_6000ada.csv", index=False)
    report = f"""# A6000 vs RTX 6000 Ada Numerical Reproducibility

These comparisons concern fake/reference quantization numerics only. They are not native FP4 performance comparisons.

## LLM

{markdown_table(llm)}

## SANA

{markdown_table(diffusion)}

Selector differences should be interpreted together with signed-margin samples: differences confined to near-zero-margin regions are expected to have negligible objective cost. Exact permutation/rotation labels are included in the matching keys.
"""
    (OUTPUT / "reproducibility_report.md").write_text(report)
    print(f"LLM matched rows={int(llm.get('matched', pd.Series(dtype=bool)).sum())}; diffusion matched rows={int(diffusion.get('matched', pd.Series(dtype=bool)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
