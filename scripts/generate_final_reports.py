#!/usr/bin/env python3
"""Generate measurement-driven summary, decision, limitations, and handoff files."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FINAL = ARTIFACTS / "06_final"

GO_SCHEMA = {
    "oracle16_has_meaningful_gain": None,
    "llm_weight_n8k64_gain_retention": None,
    "llm_activation_m16k64_gain_retention": None,
    "diffusion_weight_n8k64_gain_retention": None,
    "diffusion_activation_m16k64_gain_retention": None,
    "dominant_conflict_axis_llm": None,
    "dominant_conflict_axis_diffusion": None,
    "sensitivity_selector_recovers_gap": None,
    "permutation_recovers_gap": None,
    "rotation_recovers_gap": None,
    "best_ours_method": None,
    "four_over_six_composition_valid": None,
    "ours_complements_four_over_six": None,
    "preferred_adaptive_operand_llm": None,
    "preferred_adaptive_operand_diffusion": None,
    "diffusion_timestep_stability": None,
    "cross_model_generalization": None,
    "cross_domain_generalization": None,
    "cross_repo_validation": None,
    "cross_gpu_numerical_stability": None,
    "recommended_phase_c": None,
    "recommended_design": None,
    "confidence": None,
    "blocking_issues": [],
}


def load_csv(name: str) -> pd.DataFrame:
    path = FINAL / name
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def latest_failure_records(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(values):
        key = str(row.get("attempt_id") or f"legacy_{index}")
        latest[key] = row
    return list(latest.values())


def markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int = 40) -> str:
    available = [column for column in columns if column in frame]
    value = frame[available].head(limit).copy() if available else pd.DataFrame()
    if value.empty:
        return "_No completed measurements for this table._"
    for column in value.select_dtypes(include=["float", "float64", "float32"]).columns:
        value[column] = value[column].map(lambda item: "" if pd.isna(item) else f"{item:.7g}")
    header = "| " + " | ".join(value.columns) + " |"
    separator = "| " + " | ".join("---" for _ in value.columns) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in value.itertuples(index=False, name=None)]
    return "\n".join((header, separator, *body))


def mode_rows(frame: pd.DataFrame, modes: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty or "weight_format_mode" not in frame:
        return pd.DataFrame()
    return frame[frame["weight_format_mode"].astype(str).isin(modes)].copy()


def full_main_llm(llm: pd.DataFrame) -> pd.DataFrame:
    if llm.empty:
        return llm
    value = llm[
        llm["model"].astype(str).isin(("meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B"))
        & llm["phase"].astype(str).eq("phase_a1_llm")
    ].copy()
    return value


def paired_ppl(frame: pd.DataFrame, mode_a: str, mode_b: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    keys = ["model", "dataset_or_promptset", "num_eval_sequences"]
    left = frame[frame["weight_format_mode"] == mode_a][keys + ["ppl"]].rename(columns={"ppl": f"ppl_{mode_a}"})
    right = frame[frame["weight_format_mode"] == mode_b][keys + ["ppl"]].rename(columns={"ppl": f"ppl_{mode_b}"})
    return left.merge(right, on=keys, how="inner")


def mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def build_go_no_go(master: pd.DataFrame, layers: pd.DataFrame, timesteps: pd.DataFrame) -> dict[str, Any]:
    go = dict(GO_SCHEMA)
    go["blocking_issues"] = []
    llm = full_main_llm(master[master["domain"] == "llm"] if not master.empty else pd.DataFrame())
    oracle_pairs = paired_ppl(llm, "nvfp4", "oracle16")
    if len(oracle_pairs) >= 4:
        relative = (oracle_pairs["ppl_nvfp4"] - oracle_pairs["ppl_oracle16"]) / oracle_pairs["ppl_nvfp4"]
        # Operational reporting threshold, fixed before inspecting full data:
        # positive in >=3/4 settings and mean relative PPL gain >=0.1%.
        go["oracle16_has_meaningful_gain"] = bool((relative > 0).sum() >= 3 and relative.mean() >= 0.001)
        go["cross_model_generalization"] = bool(
            oracle_pairs.assign(gain=relative).groupby("model")["gain"].mean().gt(0).all()
        )
    else:
        go["blocking_issues"].append("full Llama/Qwen WikiText-2+C4 NVFP4/Oracle16 pairs incomplete")

    coarse = llm[llm["weight_format_mode"] == "n8k64"] if not llm.empty else pd.DataFrame()
    retention = pd.to_numeric(coarse.get("oracle_gain_retention", pd.Series(dtype=float)), errors="coerce").dropna()
    if not retention.empty:
        go["llm_weight_n8k64_gain_retention"] = float(retention.mean())

    if not layers.empty and "format_granularity" in layers:
        weight_layers = layers[layers.get("operand_role", "weight_b").astype(str).eq("weight_b")]
        k = mean_or_none(weight_layers[weight_layers["format_granularity"] == "k64_row"]["granularity_regret"])
        n = mean_or_none(weight_layers[weight_layers["format_granularity"] == "n8k16"]["granularity_regret"])
        if k is not None and n is not None:
            go["dominant_conflict_axis_llm"] = "K" if k > 1.2 * n else "N" if n > 1.2 * k else "mixed_N_K"

    selector_rows = master[master["phase"].astype(str).str.contains("selector", na=False)] if not master.empty else pd.DataFrame()
    permutation_rows = master[master["phase"].astype(str).str.contains("permutation", na=False)] if not master.empty else pd.DataFrame()
    rotation_rows = master[master["phase"].astype(str).str.contains("rotation", na=False) & master["ppl"].notna()] if not master.empty else pd.DataFrame()
    go["sensitivity_selector_recovers_gap"] = None if selector_rows.empty else bool(
        pd.to_numeric(selector_rows.get("ppl_delta_vs_nvfp4"), errors="coerce").min() < 0
    )
    go["permutation_recovers_gap"] = None if permutation_rows.empty else bool(
        pd.to_numeric(permutation_rows.get("ppl_delta_vs_nvfp4"), errors="coerce").min() < 0
    )
    go["rotation_recovers_gap"] = None if rotation_rows.empty else bool(
        pd.to_numeric(rotation_rows.get("ppl_delta_vs_nvfp4"), errors="coerce").min() < 0
    )

    composed_test = ROOT / "artifacts" / "02_tests" / "four_over_six_reduction.json"
    if composed_test.exists():
        payload = json.loads(composed_test.read_text(encoding="utf-8"))
        go["four_over_six_composition_valid"] = bool(
            payload.get("passed", payload.get("fixed_e2_reduces_to_canonical", False))
        )
    go["cross_repo_validation"] = bool(
        (ROOT / "artifacts" / "02_tests" / "cross_repo_validation.md").exists()
        and go["four_over_six_composition_valid"] is True
    )

    if not timesteps.empty and "classification" in timesteps:
        counts = timesteps.drop_duplicates("module")["classification"].value_counts(normalize=True)
        unstable = float(counts.get("unstable_timestep_dependent", 0.0))
        go["diffusion_timestep_stability"] = "materially_unstable" if unstable > 0.25 else "mostly_stable"

    cross = master[master["phase"].astype(str).str.startswith("cross_gpu", na=False)] if not master.empty else pd.DataFrame()
    if not cross.empty and cross["gpu_type"].nunique() >= 2:
        go["cross_gpu_numerical_stability"] = "measured; see reproducibility_report.md"

    required_domains = (
        not llm.empty
        and len(oracle_pairs) >= 4
        and not master[(master["domain"] == "diffusion") & master["proxy_nmse"].notna()].empty
    )
    go["cross_domain_generalization"] = None if not required_domains else bool(go["oracle16_has_meaningful_gain"])
    if not required_domains:
        go["blocking_issues"].append("mandatory main-model cross-domain result matrix incomplete")

    if required_domains and go["oracle16_has_meaningful_gain"] is not None:
        recovered = any(go[key] is True for key in (
            "sensitivity_selector_recovers_gap", "permutation_recovers_gap", "rotation_recovers_gap"
        ))
        go["recommended_phase_c"] = bool(go["oracle16_has_meaningful_gain"] and (recovered or (go["llm_weight_n8k64_gain_retention"] or 0) >= 0.7))
        go["confidence"] = "medium" if go["recommended_phase_c"] else "medium_to_high_no_go"
    return go


def table_section(title: str, frame: pd.DataFrame, columns: list[str]) -> str:
    return f"## {title}\n\n{markdown_table(frame, columns)}\n"


def build_summary(master: pd.DataFrame, layers: pd.DataFrame, timesteps: pd.DataFrame) -> str:
    llm = master[master["domain"] == "llm"] if not master.empty else pd.DataFrame()
    diffusion = master[master["domain"] == "diffusion"] if not master.empty else pd.DataFrame()
    conflict = load_csv("plots/n_conflict_vs_k_conflict.csv")
    sections = [
        "# Phase A/B Results Summary\n\nAll values below come from completed immutable experiment attempts. Empty tables denote work that is blocked or not yet completed; they are not imputed.\n",
        table_section("1. HP vs NVFP4 vs canonical 4Over6 vs E0 vs Oracle16", mode_rows(master, ("high_precision", "nvfp4", "fouroversix_4over6", "nvfp4_4over6", "all_e0m3", "oracle16")), ["domain", "model", "dataset_or_promptset", "weight_format_mode", "ppl", "proxy_nmse", "lpips", "psnr"]),
        table_section("2. LLM PPL across format granularities", llm, ["model", "dataset_or_promptset", "weight_format_mode", "activation_format_mode", "ppl", "ppl_delta_vs_nvfp4"]),
        table_section("3. N-only vs K-only conflict decomposition", conflict, ["model", "dataset_or_promptset", "module_name", "regret_k_only", "regret_n_only", "regret_nk", "interaction_residual"]),
        table_section("4. N8K64 Oracle gain retention", mode_rows(llm, ("oracle16", "n8k64")), ["model", "dataset_or_promptset", "weight_format_mode", "ppl", "ppl_delta_vs_nvfp4", "oracle_gain_retention", "weight_granularity_regret"]),
        table_section("5. M16K64 activation retention", llm[llm.get("activation_format_mode", pd.Series(index=llm.index)).astype(str).isin(("oracle16_a", "oracle16", "m16k64"))] if not llm.empty else pd.DataFrame(), ["model", "dataset_or_promptset", "weight_format_mode", "activation_format_mode", "ppl", "activation_granularity_regret"]),
        table_section("6. One-side vs two-side adaptive", llm[llm.get("phase", pd.Series(index=llm.index)).astype(str).eq("phase_a3_llm")] if not llm.empty else pd.DataFrame(), ["model", "dataset_or_promptset", "weight_format_mode", "activation_format_mode", "ppl"]),
        table_section("7. Margin conflict vs regret/PPL", layers, ["experiment_id", "model", "module_type", "mean_margin_conflict", "granularity_regret", "nmse"]),
        table_section("8. Activation-aware selector effect", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.contains("selector", na=False)] if not master.empty else pd.DataFrame(), ["model", "dataset_or_promptset", "selector", "calibration_size", "ppl", "proxy_nmse", "weight_selector_disagreement"]),
        table_section("9. Permutation effect", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.contains("permutation", na=False)] if not master.empty else pd.DataFrame(), ["model", "permutation", "ppl", "proxy_nmse", "weight_granularity_regret"]),
        table_section("10. Rotation effect", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.contains("rotation", na=False)] if not master.empty else pd.DataFrame(), ["model", "rotation", "ppl", "proxy_nmse", "weight_granularity_regret"]),
        table_section("11. Best combined Ours", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.contains("combined", na=False)] if not master.empty else pd.DataFrame(), ["model", "dataset_or_promptset", "quantization_mode", "selector", "permutation", "rotation", "ppl", "proxy_nmse"]),
        table_section("12. SANA W4A16 proxy/image results", diffusion[diffusion.get("activation_format_mode", pd.Series(index=diffusion.index)).astype(str).eq("high_precision")] if not diffusion.empty else pd.DataFrame(), ["weight_format_mode", "proxy_nmse", "latent_trajectory_nmse", "lpips", "psnr", "image_reward"]),
        table_section("13. SANA W4A4 selected results", diffusion[~diffusion.get("activation_format_mode", pd.Series(index=diffusion.index)).astype(str).eq("high_precision")] if not diffusion.empty else pd.DataFrame(), ["weight_format_mode", "activation_format_mode", "proxy_nmse", "lpips", "psnr", "image_reward"]),
        table_section("14. Diffusion timestep stability", timesteps, ["module", "timestep_index", "timestep", "e0_ratio", "selector_agreement_vs_reference_step", "mean_margin_conflict", "granularity_regret", "sensitivity_regret", "proxy_nmse", "classification"]),
        table_section("15. MixFP4+4Over6 composition validity/results", master[master.get("four_over_six_mode", pd.Series(index=master.index)).astype(str).ne("none")] if not master.empty else pd.DataFrame(), ["model", "dataset_or_promptset", "weight_format_mode", "four_over_six_mode", "ppl", "proxy_nmse"]),
        table_section("16. MixFP4+Ours vs MixFP4+4Over6+Ours", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.contains("combined", na=False)] if not master.empty else pd.DataFrame(), ["model", "weight_format_mode", "four_over_six_mode", "selector", "permutation", "rotation", "ppl", "proxy_nmse"]),
        table_section("17. Cross-model/domain consistency", master, ["domain", "model", "dataset_or_promptset", "weight_format_mode", "ppl_delta_vs_nvfp4", "oracle_gain_retention", "proxy_nmse"]),
        "## 18. Cross-repo consistency\n\nSee `artifacts/01_repo_audit/quantization_semantics.md`, `artifacts/02_tests/cross_repo_validation.md`, and the fixed-E2 reduction measurements in `artifacts/02_tests/four_over_six_reduction.json`.\n",
        table_section("19. Cross-GPU consistency", master[master.get("phase", pd.Series(index=master.index)).astype(str).str.startswith("cross_gpu", na=False)] if not master.empty else pd.DataFrame(), ["domain", "model", "gpu_type", "weight_format_mode", "ppl", "proxy_nmse", "weight_e0_ratio", "weight_granularity_regret"]),
    ]
    return "\n".join(sections)


def build_decision(go: dict[str, Any], master: pd.DataFrame) -> str:
    def answer(key: str) -> str:
        value = go.get(key)
        return "Undetermined from incomplete measurements." if value is None else str(value)

    return f"""# Strict Phase A/B Decision Report

This report distinguishes fake/reference quantization accuracy from native FP4 performance. No native SM120 speedup is claimed.

## A. Does Oracle16 meaningfully improve NVFP4?

{answer('oracle16_has_meaningful_gain')}

## B. How much Oracle gain survives N8K64/M16K64?

- LLM weight N8K64 mean retention: {answer('llm_weight_n8k64_gain_retention')}
- LLM activation M16K64 mean retention: {answer('llm_activation_m16k64_gain_retention')}
- Diffusion weight N8K64 mean retention: {answer('diffusion_weight_n8k64_gain_retention')}
- Diffusion activation M16K64 mean retention: {answer('diffusion_activation_m16k64_gain_retention')}

Raw deltas are in `master_results.csv` and Table 4/5 of `results_summary.md`.

## C. What causes the loss?

- LLM dominant conflict axis: {answer('dominant_conflict_axis_llm')}
- Diffusion dominant conflict axis: {answer('dominant_conflict_axis_diffusion')}
- Diffusion timestep stability: {answer('diffusion_timestep_stability')}

## D. Does Ours recover the loss?

- Ours-S: {answer('sensitivity_selector_recovers_gap')}
- Ours-P: {answer('permutation_recovers_gap')}
- Ours-R: {answer('rotation_recovers_gap')}
- Best method: {answer('best_ours_method')}

The complete NVFP4 / canonical 4Over6 / Oracle16 / raw coarse / coarse+Ours ladder is tabulated in `results_summary.md`.

## E. Is Ours complementary to 4Over6?

Composition validity: {answer('four_over_six_composition_valid')}. Complementarity: {answer('ours_complements_four_over_six')}.

## F. Does the result generalize?

- Cross-model: {answer('cross_model_generalization')}
- Cross-domain: {answer('cross_domain_generalization')}
- Cross-repo: {answer('cross_repo_validation')}
- Cross-GPU: {answer('cross_gpu_numerical_stability')}

## G. Which operand should be adaptive in a future SM120 kernel?

- LLM: {answer('preferred_adaptive_operand_llm')}
- Diffusion: {answer('preferred_adaptive_operand_diffusion')}

## H. Is RTX 5090 / SM120 Phase C justified?

{answer('recommended_phase_c')}

## I. Strongest evidence against continuing

{'; '.join(go['blocking_issues']) if go['blocking_issues'] else 'See the worst cross-model/domain result and the canonical 4Over6 comparison in results_summary.md.'}
"""


def build_limitations(master: pd.DataFrame, failed: list[dict[str, Any]], go: dict[str, Any]) -> str:
    counts = pd.Series([row.get("failure_class", "unknown") for row in failed]).value_counts().to_dict()
    lines = [
        "# Limitations and Blockers",
        "",
        "A6000/RTX 6000 Ada runs are reference/fake quantization accuracy experiments. They cannot establish native SM120 FP4 throughput, metadata cost, transform cost, or end-to-end speedup.",
        "",
        f"Completed experiment IDs currently aggregated: {len(master)}.",
        f"Preserved failed attempts: {len(failed)} ({json.dumps(counts, sort_keys=True)}).",
        "",
    ]
    if go["blocking_issues"]:
        lines.extend(("## Outstanding completion blockers", ""))
        lines.extend(f"- {item}" for item in go["blocking_issues"])
        lines.append("")
    if failed:
        lines.extend(("## Failed attempts", "", "| experiment_id | class | error |", "| --- | --- | --- |"))
        for row in failed:
            error = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")[:240]
            lines.append(f"| {row.get('experiment_id')} | {row.get('failure_class')} | {error} |")
    return "\n".join(lines) + "\n"


def build_handoff(go: dict[str, Any], master: pd.DataFrame) -> str:
    recommendation = go.get("recommended_phase_c")
    design = go.get("recommended_design") or "No design frozen until all Phase A/B gates are measured."
    return f"""# Phase C Handoff — RTX 5090 / SM120 (Not Executed)

Phase-C recommendation from measured Phase A/B gates: {recommendation}

- Exact best format rule: {design}
- Scale rule/granularity: candidate-specific K16 scales; selected standard or validated 4Over6 rule must be copied exactly from the winning row.
- Weight format granularity: N8×K64 unless the decision report rejects it.
- Activation format granularity: M16×K64 when adaptive; otherwise fixed E2M1.
- Preferred adaptive operand: LLM={go.get('preferred_adaptive_operand_llm')}; diffusion={go.get('preferred_adaptive_operand_diffusion')}.
- Selector: {go.get('best_ours_method')}.
- Permutation/transform: use only the exact-foldable winner; upper-bound-only variants are excluded from a deployable recommendation.
- Metadata: one datatype selector per hardware region plus one scale per K16 block; no timestep metadata unless the stability study justifies it.
- LLM test cases: pinned Llama-3.1-8B and Qwen3-8B revisions, persisted WikiText-2 and C4 slices.
- SANA test cases: pinned SANA-1.6B revision, persisted proxy/MJHQ prompts and seeds, 20 FlowMatch steps, guidance 4.5, 1024×1024.
- Expected reference metrics: use exact winning rows in `master_results.csv`; do not substitute rounded narrative values.
- Numerical tolerance: selector counts exact away from ties; end-to-end BF16/fake-quant metrics should reproduce within cross-GPU tolerances documented in `05_cross_gpu/reproducibility_report.md`.
- RTX 5090 work remaining: instruction semantics, native E0M3 encoding, selector metadata loading, K16 scale path, transform folding, correctness, kernel coverage, throughput, memory, power, and end-to-end latency.
- Suggested integration: FourOverSix reference semantics → CUTLASS SM120 block-scaled GEMM; use Nunchaku/DeepCompressor graph integration for SANA only after kernel correctness.

No native performance claim is made from A6000/RTX 6000 Ada results.
"""


def build_reproduction(manifest: list[dict[str, Any]]) -> str:
    completed: dict[str, dict[str, Any]] = {}
    for row in manifest:
        if row.get("status") == "completed" and row.get("command_shell"):
            completed[str(row["experiment_id"])] = row
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "# Each command re-runs through the mandatory dynamic GPU guard."]
    for experiment_id, row in sorted(completed.items()):
        command = row["command_shell"]
        lines.extend(("", f"# {experiment_id}", f"python3 scripts/gpu_guard.py --experiment-id repro_{experiment_id} --gpu-kind any -- {command}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    master = load_csv("master_results.csv")
    layers = load_csv("per_layer_metrics.csv")
    timesteps = load_csv("timestep_metrics.csv")
    failed = latest_failure_records(read_jsonl(FINAL / "failed_runs.jsonl"))
    manifest = read_jsonl(FINAL / "experiment_manifest.jsonl")
    go = build_go_no_go(master, layers, timesteps)
    (FINAL / "go_no_go.json").write_text(json.dumps(go, indent=2, sort_keys=True) + "\n")
    (FINAL / "results_summary.md").write_text(build_summary(master, layers, timesteps))
    (FINAL / "decision_report.md").write_text(build_decision(go, master))
    (FINAL / "limitations.md").write_text(build_limitations(master, failed, go))
    (FINAL / "phase_c_handoff.md").write_text(build_handoff(go, master))
    reproduction = FINAL / "reproduction_commands.sh"
    reproduction.write_text(build_reproduction(manifest))
    reproduction.chmod(0o755)
    print(json.dumps({"completed": len(master), "failed": len(failed), "recommended_phase_c": go["recommended_phase_c"], "blockers": go["blocking_issues"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
