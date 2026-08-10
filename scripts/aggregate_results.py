#!/usr/bin/env python3
"""Aggregate immutable attempts into the mandatory final machine-readable tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FINAL = ARTIFACTS / "06_final"
MANIFEST = FINAL / "experiment_manifest.jsonl"
SUPERSEDED = ARTIFACTS / "00_environment" / "superseded_runs.json"

MASTER_COLUMNS = (
    "experiment_id", "phase", "domain", "model", "model_revision",
    "dataset_or_promptset", "quantization_mode", "weight_format_mode",
    "activation_format_mode", "weight_format_granularity",
    "activation_format_granularity", "scale_rule", "scale_group_size",
    "selector", "permutation", "rotation", "calibration_size",
    "four_over_six_mode", "gpu_index", "physical_gpu_index",
    "logical_gpu_index", "gpu_type", "num_eval_sequences", "num_images",
    "ppl", "ppl_delta_vs_nvfp4", "proxy_delta_vs_nvfp4",
    "oracle_gain_retention", "oracle_proxy_gain_retention", "image_reward",
    "clip_score", "lpips", "psnr", "ssim", "fid", "downstream_mean_accuracy",
    "proxy_mse", "proxy_nmse",
    "proxy_relative_l2", "proxy_cosine_error", "latent_trajectory_nmse",
    "weight_mse", "weight_nmse_weighted", "weight_e0_ratio",
    "weight_homogeneity", "weight_margin_conflict", "weight_oracle_error",
    "weight_constrained_error", "weight_granularity_regret",
    "activation_mse", "activation_e0_ratio", "activation_homogeneity",
    "activation_margin_conflict", "activation_granularity_regret", "status",
    "artifact_dir", "attempt_id", "start_time", "end_time",
)

PER_LAYER_COLUMNS = (
    "experiment_id", "domain", "model", "layer_idx", "module_name",
    "module_type", "operand_role", "call_index", "N", "M", "K",
    "format_granularity", "num_scale_blocks", "num_format_regions",
    "e0_ratio", "e2_ratio", "mean_homogeneity", "mean_margin_conflict",
    "margin_weighted_conflict", "mean_format_margin", "mean_abs_format_margin",
    "oracle_mse", "constrained_mse", "granularity_regret", "normalized_regret",
    "sensitivity_regret", "selected_sensitivity_regret", "nmse",
    "relative_l2", "cosine_error", "mse", "mean_abs_error", "max_abs_error",
    "selector_disagreement_ratio", "artifact_dir",
)

REGION_COLUMNS = (
    "experiment_id", "domain", "model", "layer", "module",
    "module_name", "module_type", "operand_role", "activation_call_index",
    "region_n_start", "region_k_start", "region_n_size", "region_k_size",
    "num_k16_blocks", "oracle_e0_count", "oracle_e2_count", "homogeneity",
    "P_G", "N_G", "R_G", "margin_conflict", "oracle_error",
    "constrained_error", "regret", "normalized_regret", "selected_format",
    "selected_format_mse", "selected_format_sensitivity", "mean_abs_margin",
    "max_abs_margin", "sensitivity_regret", "selector_disagreement",
    "artifact_dir",
)

TIMESTEP_COLUMNS = (
    "experiment_id", "model", "promptset", "timestep", "timestep_index",
    "layer", "module", "e0_ratio", "selector_agreement_vs_reference_step",
    "mean_margin", "mean_homogeneity", "mean_margin_conflict",
    "granularity_regret", "sensitivity_regret", "proxy_nmse", "classification",
    "artifact_dir",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def superseded_entries() -> dict[str, dict[str, Any]]:
    if not SUPERSEDED.exists():
        return {}
    payload = json.loads(SUPERSEDED.read_text(encoding="utf-8"))
    return {str(row["experiment_id"]): row for row in payload.get("entries", [])}


def completed_attempts(*, include_superseded: bool = False) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(MANIFEST):
        if row.get("status") != "completed" or not row.get("summary_path"):
            continue
        current = latest.get(str(row["experiment_id"]))
        if current is None or str(row.get("end_time", "")) > str(current.get("end_time", "")):
            latest[str(row["experiment_id"])] = row
    if not include_superseded:
        excluded = set(superseded_entries())
        latest = {key: row for key, row in latest.items() if key not in excluded}
    return sorted(latest.values(), key=lambda row: (str(row.get("end_time", "")), row["experiment_id"]))


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def infer_domain(record: dict[str, Any], summary: dict[str, Any]) -> str:
    if summary.get("domain"):
        return str(summary["domain"])
    artifact_dir = str(record.get("artifact_dir", ""))
    return "diffusion" if "/diffusion/" in artifact_dir else "llm"


def normalize_summary(record: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    value = dict(summary)
    value["experiment_id"] = record["experiment_id"]
    value["phase"] = record.get("phase", value.get("phase"))
    value["domain"] = infer_domain(record, value)
    value["dataset_or_promptset"] = value.get("dataset_or_promptset") or value.get("dataset") or value.get("prompt_manifest")
    value["physical_gpu_index"] = value.get("physical_gpu_index", value.get("gpu_index"))
    value["gpu_index"] = value.get("gpu_index", value.get("physical_gpu_index"))
    value["artifact_dir"] = record.get("artifact_dir")
    value["attempt_id"] = record.get("attempt_id")
    value["start_time"] = record.get("start_time")
    value["end_time"] = record.get("end_time")
    value["status"] = value.get("status", "completed")
    return value


def baseline_key(row: pd.Series) -> tuple[Any, ...]:
    domain = row.get("domain")
    if domain == "llm":
        return (
            row.get("model"), row.get("dataset_or_promptset"), row.get("num_eval_sequences"),
            row.get("activation_format_mode"),
        )
    return (
        row.get("model"), row.get("dataset_or_promptset"),
        row.get("num_proxy_prompts", row.get("num_images")),
        row.get("activation_format_mode"),
    )


def add_ppl_comparisons(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master
    master = master.copy()
    master["ppl_delta_vs_nvfp4"] = np.nan
    master["proxy_delta_vs_nvfp4"] = np.nan
    master["oracle_gain_retention"] = np.nan
    master["oracle_proxy_gain_retention"] = np.nan
    llm = master[(master["domain"] == "llm") & master["ppl"].notna()]
    nvfp4: dict[tuple[Any, ...], float] = {}
    oracle: dict[tuple[Any, ...], float] = {}
    for _, row in llm.iterrows():
        key = baseline_key(row)
        mode = row.get("weight_format_mode")
        if mode == "nvfp4":
            nvfp4[key] = float(row["ppl"])
        elif mode == "oracle16" and row.get("activation_format_mode") in {"high_precision", "fp16", "bf16"}:
            oracle[key] = float(row["ppl"])
    for index, row in llm.iterrows():
        key = baseline_key(row)
        base = nvfp4.get(key)
        if base is None:
            continue
        ppl = float(row["ppl"])
        master.at[index, "ppl_delta_vs_nvfp4"] = ppl - base
        ideal = oracle.get(key)
        if ideal is not None and abs(base - ideal) > 1e-12:
            master.at[index, "oracle_gain_retention"] = (base - ppl) / (base - ideal)

    diffusion = master[(master["domain"] == "diffusion") & master["proxy_nmse"].notna()]
    proxy_nvfp4: dict[tuple[Any, ...], float] = {}
    proxy_oracle: dict[tuple[Any, ...], float] = {}
    for _, row in diffusion.iterrows():
        key = baseline_key(row)
        mode = row.get("weight_format_mode")
        if mode == "nvfp4":
            proxy_nvfp4[key] = float(row["proxy_nmse"])
        elif mode == "oracle16" and row.get("activation_format_mode") in {
            "high_precision", "fp16", "bf16"
        }:
            proxy_oracle[key] = float(row["proxy_nmse"])
    for index, row in diffusion.iterrows():
        key = baseline_key(row)
        base = proxy_nvfp4.get(key)
        if base is None:
            continue
        proxy = float(row["proxy_nmse"])
        master.at[index, "proxy_delta_vs_nvfp4"] = proxy - base
        ideal = proxy_oracle.get(key)
        if ideal is not None and abs(base - ideal) > 1e-30:
            retention = (base - proxy) / (base - ideal)
            master.at[index, "oracle_proxy_gain_retention"] = retention
            master.at[index, "oracle_gain_retention"] = retention
    return master


def aligned(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    ordered = list(columns) + [column for column in result.columns if column not in columns]
    return result[ordered]


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    layer_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    timestep_frames: list[pd.DataFrame] = []
    downstream_frames: list[pd.DataFrame] = []
    attempts = completed_attempts()
    for record in attempts:
        artifact_dir = ROOT / record["artifact_dir"]
        summary_path = ROOT / record["summary_path"]
        if not summary_path.exists():
            continue
        summary = normalize_summary(record, json.loads(summary_path.read_text(encoding="utf-8")))
        summaries.append(summary)
        domain = summary["domain"]
        model = summary.get("model")

        layer = read_csv(artifact_dir / "per_layer_metrics.csv")
        if not layer.empty:
            layer["experiment_id"] = record["experiment_id"]
            layer["domain"] = domain
            layer["model"] = model
            layer["operand_role"] = layer.get("operand_role", "weight_b")
            layer["artifact_dir"] = record["artifact_dir"]
            layer_frames.append(layer)
        activation = read_csv(artifact_dir / "activation_metrics.csv")
        if not activation.empty:
            activation["experiment_id"] = record["experiment_id"]
            activation["domain"] = domain
            activation["model"] = model
            activation["operand_role"] = "activation_a"
            activation["M"] = activation.get("N_or_M", activation.get("rows"))
            activation["artifact_dir"] = record["artifact_dir"]
            layer_frames.append(activation)
        layer_output = read_csv(artifact_dir / "per_layer_output_metrics.csv")
        if not layer_output.empty:
            layer_output["experiment_id"] = record["experiment_id"]
            layer_output["domain"] = domain
            layer_output["model"] = model
            layer_output["operand_role"] = "layer_output"
            if "module_name" not in layer_output and "layer" in layer_output:
                layer_output["module_name"] = layer_output["layer"]
            if "module_type" not in layer_output:
                layer_output["module_type"] = "other_linear"
            layer_output["artifact_dir"] = record["artifact_dir"]
            layer_frames.append(layer_output)
        regions = read_csv(artifact_dir / "format_region_metrics.csv")
        if not regions.empty:
            regions["experiment_id"] = record["experiment_id"]
            regions["domain"] = domain
            regions["model"] = model
            regions["layer"] = regions.get("layer_idx")
            regions["module"] = regions.get("module_name")
            regions["artifact_dir"] = record["artifact_dir"]
            region_frames.append(regions)
        timesteps = read_csv(artifact_dir / "timestep_metrics.csv")
        if not timesteps.empty:
            timesteps["experiment_id"] = record["experiment_id"]
            timesteps["model"] = model
            timesteps["promptset"] = summary.get("dataset_or_promptset")
            timesteps["artifact_dir"] = record["artifact_dir"]
            timestep_frames.append(timesteps)
        downstream = read_csv(artifact_dir / "downstream_metrics.csv")
        if not downstream.empty:
            downstream["artifact_dir"] = record["artifact_dir"]
            downstream["attempt_id"] = record.get("attempt_id")
            downstream_frames.append(downstream)

    master = aligned(pd.DataFrame(summaries), MASTER_COLUMNS)
    master = add_ppl_comparisons(master)
    master = aligned(master, MASTER_COLUMNS)
    master.to_csv(FINAL / "master_results.csv", index=False)
    master[master["domain"] == "llm"].to_csv(FINAL / "llm_results.csv", index=False)
    master[master["domain"] == "diffusion"].to_csv(FINAL / "diffusion_results.csv", index=False)

    layers = aligned(pd.concat(layer_frames, ignore_index=True) if layer_frames else pd.DataFrame(), PER_LAYER_COLUMNS)
    regions = aligned(pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame(), REGION_COLUMNS)
    timesteps = aligned(pd.concat(timestep_frames, ignore_index=True) if timestep_frames else pd.DataFrame(), TIMESTEP_COLUMNS)
    layers.to_csv(FINAL / "per_layer_metrics.csv", index=False)
    regions.to_csv(FINAL / "format_region_metrics.csv", index=False)
    timesteps.to_csv(FINAL / "timestep_metrics.csv", index=False)
    downstream = pd.concat(downstream_frames, ignore_index=True) if downstream_frames else pd.DataFrame(
        columns=("experiment_id", "model", "task", "metric_name", "metric", "num_fewshot", "limit", "artifact_dir", "attempt_id")
    )
    downstream.to_csv(FINAL / "downstream_results.csv", index=False)

    superseded = superseded_entries()
    superseded_summaries: list[dict[str, Any]] = []
    for record in completed_attempts(include_superseded=True):
        if record["experiment_id"] not in superseded:
            continue
        summary_path = ROOT / record["summary_path"]
        if not summary_path.exists():
            continue
        row = normalize_summary(
            record, json.loads(summary_path.read_text(encoding="utf-8"))
        )
        row.update(superseded[record["experiment_id"]])
        superseded_summaries.append(row)
    pd.DataFrame(superseded_summaries).to_csv(
        FINAL / "superseded_results.csv", index=False
    )

    report = {
        "completed_experiments": len(master),
        "llm_experiments": int((master["domain"] == "llm").sum()) if not master.empty else 0,
        "diffusion_experiments": int((master["domain"] == "diffusion").sum()) if not master.empty else 0,
        "per_layer_rows": len(layers),
        "format_region_rows": len(regions),
        "timestep_rows": len(timesteps),
        "downstream_rows": len(downstream),
        "superseded_completed_experiments_excluded": len(superseded_summaries),
        "superseded_registry": str(SUPERSEDED.relative_to(ROOT)),
        "nonfinite_numeric_cells_are_serialized_as_empty": True,
    }
    (FINAL / "aggregation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
