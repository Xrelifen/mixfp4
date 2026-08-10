#!/usr/bin/env python3
"""Generate every required plot together with its exact CSV source table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
PLOTS = FINAL / "plots"


def load(name: str) -> pd.DataFrame:
    path = FINAL / name
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)


def save_source(name: str, source: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    source.to_csv(PLOTS / f"{name}.csv", index=False)


def empty(ax: plt.Axes, name: str) -> None:
    ax.text(0.5, 0.5, "No completed measurements", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(name.replace("_", " "))
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_plot(name: str, source: pd.DataFrame, x: str, y: str, color: str | None = None) -> None:
    save_source(name, source)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    valid = source.copy()
    if x in valid:
        valid[x] = numeric(valid, x)
    if y in valid:
        valid[y] = numeric(valid, y)
    valid = valid.dropna(subset=[x, y]) if x in valid and y in valid else pd.DataFrame()
    if valid.empty:
        empty(ax, name)
    else:
        if color and color in valid:
            for label, group in valid.groupby(color, dropna=False):
                ax.scatter(group[x], group[y], alpha=0.72, s=24, label=str(label))
            if valid[color].nunique(dropna=False) <= 12:
                ax.legend(fontsize=7)
        else:
            ax.scatter(valid[x], valid[y], alpha=0.72, s=24)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.set_title(name.replace("_", " "))
    fig.savefig(PLOTS / f"{name}.png", dpi=180)
    plt.close(fig)


def line_plot(name: str, source: pd.DataFrame, x: str, y: str, group: str | None = None) -> None:
    save_source(name, source)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    valid = source.copy()
    if y in valid:
        valid[y] = numeric(valid, y)
    valid = valid.dropna(subset=[x, y]) if x in valid and y in valid else pd.DataFrame()
    if valid.empty:
        empty(ax, name)
    else:
        if group and group in valid:
            for label, part in valid.groupby(group, dropna=False):
                part = part.reset_index(drop=True)
                ax.plot(np.arange(len(part)), part[y], marker="o", linewidth=1.2, label=str(label))
            ax.set_xticks(np.arange(len(valid.groupby(group).__iter__().__next__()[1])))
            first = next(iter(valid.groupby(group)))[1]
            ax.set_xticklabels(first[x].astype(str), rotation=45, ha="right", fontsize=7)
            if valid[group].nunique(dropna=False) <= 12:
                ax.legend(fontsize=7)
        else:
            ax.plot(np.arange(len(valid)), valid[y], marker="o", linewidth=1.2)
            ax.set_xticks(np.arange(len(valid)))
            ax.set_xticklabels(valid[x].astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.set_title(name.replace("_", " "))
    fig.savefig(PLOTS / f"{name}.png", dpi=180)
    plt.close(fig)


def mean_layers(layers: pd.DataFrame) -> pd.DataFrame:
    if layers.empty:
        return pd.DataFrame()
    numeric_columns = [
        "granularity_regret", "sensitivity_regret", "mean_homogeneity",
        "mean_margin_conflict", "e0_ratio", "nmse",
    ]
    for column in numeric_columns:
        if column in layers:
            layers[column] = numeric(layers, column)
    columns = [column for column in numeric_columns if column in layers]
    return layers.groupby("experiment_id", as_index=False)[columns].mean() if columns else pd.DataFrame()


def conflict_table(master: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    if master.empty or layers.empty:
        return pd.DataFrame(columns=("model", "dataset_or_promptset", "module_name", "regret_k_only", "regret_n_only", "regret_nk", "interaction_residual"))
    modes = master[["experiment_id", "model", "dataset_or_promptset", "weight_format_mode"]]
    source = layers.merge(modes, on=["experiment_id", "model"], how="inner")
    source = source[source["weight_format_mode"].isin(("k64_row", "n8k16", "n8k64"))]
    if source.empty:
        return pd.DataFrame()
    grouped = source.groupby(["model", "dataset_or_promptset", "module_name", "weight_format_mode"], as_index=False)["granularity_regret"].mean()
    pivot = grouped.pivot_table(index=["model", "dataset_or_promptset", "module_name"], columns="weight_format_mode", values="granularity_regret").reset_index()
    for mode in ("k64_row", "n8k16", "n8k64"):
        if mode not in pivot:
            pivot[mode] = np.nan
    pivot = pivot.rename(columns={"k64_row": "regret_k_only", "n8k16": "regret_n_only", "n8k64": "regret_nk"})
    pivot["interaction_residual"] = pivot["regret_nk"] - pivot["regret_k_only"] - pivot["regret_n_only"]
    return pivot


def main() -> int:
    master = load("master_results.csv")
    layers = load("per_layer_metrics.csv")
    timesteps = load("timestep_metrics.csv")
    llm = master[(master.get("domain") == "llm") & master.get("ppl", pd.Series(index=master.index)).notna()].copy() if not master.empty else pd.DataFrame()
    diffusion = master[master.get("domain") == "diffusion"].copy() if not master.empty else pd.DataFrame()
    layer_means = mean_layers(layers.copy())
    experiment_metrics = master.merge(layer_means, on="experiment_id", how="left", suffixes=("", "_layer")) if not master.empty else pd.DataFrame()

    plot_specs: list[tuple[str, pd.DataFrame, str, str, str | None, str]] = []
    plot_specs.extend(
        [
            ("ppl_vs_granularity", llm, "weight_format_mode", "ppl", "model", "line"),
            ("relative_ppl_delta_vs_nvfp4", llm, "weight_format_mode", "ppl_delta_vs_nvfp4", "model", "line"),
            ("oracle_gain_retention_vs_granularity", llm, "weight_format_mode", "oracle_gain_retention", "model", "line"),
            ("regret_vs_granularity", experiment_metrics, "weight_format_mode", "granularity_regret", "model", "line"),
            ("homogeneity_vs_regret", layers, "mean_homogeneity", "granularity_regret", "module_type", "scatter"),
            ("margin_conflict_vs_regret", layers, "mean_margin_conflict", "granularity_regret", "module_type", "scatter"),
            ("regret_vs_ppl_delta", experiment_metrics, "granularity_regret", "ppl_delta_vs_nvfp4", "model", "scatter"),
            ("sensitivity_regret_vs_ppl_delta", experiment_metrics, "sensitivity_regret", "ppl_delta_vs_nvfp4", "model", "scatter"),
            ("e0_ratio_by_layer", layers, "layer_idx", "e0_ratio", "module_type", "line"),
            ("margin_conflict_by_layer", layers, "layer_idx", "mean_margin_conflict", "module_type", "line"),
        ]
    )
    conflict = conflict_table(master, layers)
    plot_specs.append(("n_conflict_vs_k_conflict", conflict, "regret_k_only", "regret_n_only", "model", "scatter"))

    n8 = layers[layers.get("format_granularity", pd.Series(index=layers.index)).astype(str).eq("n8k64")].copy() if not layers.empty else pd.DataFrame()
    save_source("n8k64_regret_layer_heatmap", n8)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if n8.empty or not {"layer_idx", "module_type", "granularity_regret"}.issubset(n8):
        empty(ax, "n8k64_regret_layer_heatmap")
    else:
        table = n8.pivot_table(index="module_type", columns="layer_idx", values="granularity_regret", aggfunc="mean")
        image = ax.imshow(np.log10(table.to_numpy(dtype=float) + 1e-30), aspect="auto", cmap="magma")
        ax.set_yticks(np.arange(len(table.index)), labels=table.index)
        ax.set_xticks(np.arange(len(table.columns)), labels=table.columns, fontsize=6)
        ax.set_xlabel("layer")
        ax.set_title("N8K64 log10 granularity regret")
        fig.colorbar(image, ax=ax, label="log10 regret")
    fig.savefig(PLOTS / "n8k64_regret_layer_heatmap.png", dpi=180)
    plt.close(fig)

    selector = experiment_metrics[experiment_metrics.get("selector", pd.Series(index=experiment_metrics.index)).astype(str).ne("mse") | experiment_metrics.get("phase", pd.Series(index=experiment_metrics.index)).astype(str).str.contains("selector", na=False)] if not experiment_metrics.empty else pd.DataFrame()
    permutation = experiment_metrics[experiment_metrics.get("permutation", pd.Series(index=experiment_metrics.index)).astype(str).ne("none")] if not experiment_metrics.empty else pd.DataFrame()
    rotation = experiment_metrics[experiment_metrics.get("rotation", pd.Series(index=experiment_metrics.index)).astype(str).ne("identity")] if not experiment_metrics.empty else pd.DataFrame()
    combined = experiment_metrics[experiment_metrics.get("phase", pd.Series(index=experiment_metrics.index)).astype(str).str.contains("combined", na=False)] if not experiment_metrics.empty else pd.DataFrame()
    four = experiment_metrics[experiment_metrics.get("four_over_six_mode", pd.Series(index=experiment_metrics.index)).astype(str).ne("none")] if not experiment_metrics.empty else pd.DataFrame()
    plot_specs.extend(
        [
            ("selector_comparison", selector, "selector", "ppl", "model", "line"),
            ("permutation_comparison", permutation, "permutation", "ppl", "model", "line"),
            ("rotation_ppl_vs_regret", rotation, "granularity_regret", "ppl", "rotation", "scatter"),
            ("rotation_homogeneity_vs_regret", rotation, "mean_homogeneity", "granularity_regret", "rotation", "scatter"),
            ("combined_ours_comparison", combined, "quantization_mode", "ppl", "model", "line"),
            ("four_over_six_composition_comparison", four, "weight_format_mode", "ppl", "model", "line"),
            ("llm_cross_model_comparison", llm, "weight_format_mode", "ppl_delta_vs_nvfp4", "model", "line"),
            ("sana_proxy_vs_granularity", diffusion, "weight_format_mode", "proxy_nmse", "activation_format_mode", "line"),
            ("sana_image_metrics_vs_granularity", diffusion, "weight_format_mode", "lpips", "activation_format_mode", "line"),
            ("sana_timestep_preference_stability", timesteps, "timestep_index", "selector_agreement_vs_reference_step", "module", "line"),
            ("sana_timestep_regret", timesteps, "timestep_index", "granularity_regret", "module", "line"),
        ]
    )

    cross = experiment_metrics[experiment_metrics.get("phase", pd.Series(index=experiment_metrics.index)).astype(str).str.startswith("cross_gpu")] if not experiment_metrics.empty else pd.DataFrame()
    plot_specs.append(("a6000_vs_6000ada_numerical_comparison", cross, "gpu_type", "ppl", "weight_format_mode", "line"))

    for name, source, x, y, color, kind in plot_specs:
        if kind == "scatter":
            scatter_plot(name, source, x, y, color)
        else:
            line_plot(name, source, x, y, color)
    print(f"generated {len(plot_specs) + 1} plots in {PLOTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
