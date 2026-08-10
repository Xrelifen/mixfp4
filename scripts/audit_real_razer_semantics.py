#!/usr/bin/env python3
"""Quantify real-weight project/ RaZeR differences and identify their cause."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstreams" / "NVFP4-RaZeR"))

from project_quant import build_candidates, quant_mixfp4_granularity  # noqa: E402
from project_quant.artifacts import atomic_json, timestamp  # noqa: E402
from quantize.quantizer import quant_nvfp4, quant_nvif4  # noqa: E402


MODEL = "meta-llama/Llama-3.2-1B"
REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
SNAPSHOT = Path(
    "/share2/huggingface/hub/models--meta-llama--Llama-3.2-1B/"
    f"snapshots/{REVISION}"
)
OUT = ROOT / "artifacts" / "01_repo_audit"


@torch.no_grad()
def main() -> int:
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    rows: list[dict] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.endswith("lm_head"):
            continue
        weight = module.weight.detach()
        candidates = build_candidates(weight, operand_role="weight_b")
        project = quant_mixfp4_granularity(
            weight,
            format_region="oracle16",
            return_stats=True,
        )
        upstream_e2 = quant_nvfp4(weight, n_bits=4, groupsize=16)
        upstream_if = quant_nvif4(weight, n_bits=4, groupsize=16)

        # RaZeR compares candidate residuals before multiplying by the shared
        # global decode scale. The project compares fully dequantized residuals.
        # These objectives are algebraically proportional but not bitwise
        # equivalent in float32 reductions near a zero format margin.
        scaled = candidates.grouped_original / candidates.global_scale
        e2_scaled_error = (
            (candidates.e2_codes * candidates.e2_scales[..., None] - scaled) ** 2
        ).sum(dim=-1)
        e0_scaled_error = (
            (candidates.e0_codes * candidates.e0_scales[..., None] - scaled) ** 2
        ).sum(dim=-1)
        upstream_selector = e0_scaled_error < e2_scaled_error
        project_selector = project.format_ids.bool()
        disagreement = upstream_selector != project_selector
        project_margin = candidates.e2_errors - candidates.e0_errors
        scaled_margin = e2_scaled_error - e0_scaled_error
        if disagreement.any():
            disagreement_abs_margin_max = float(project_margin[disagreement].abs().max())
            disagreement_scaled_abs_margin_max = float(scaled_margin[disagreement].abs().max())
        else:
            disagreement_abs_margin_max = 0.0
            disagreement_scaled_abs_margin_max = 0.0
        delta_e2 = project.e2_errors.new_tensor(0.0)
        # Keep the all-E2 regression separate from selector differences.
        project_e2 = quant_mixfp4_granularity(weight, format_region="all_e2m1")
        delta_e2 = project_e2.float() - upstream_e2.float()
        delta_if = project.dequant.float() - upstream_if.float()
        rows.append(
            {
                "module_name": name,
                "N": weight.shape[0],
                "K": weight.shape[1],
                "num_k16_blocks": project_selector.numel(),
                "selector_disagreement_count": int(disagreement.sum()),
                "selector_disagreement_ratio": float(disagreement.float().mean()),
                "disagreement_project_abs_margin_max": disagreement_abs_margin_max,
                "disagreement_scaled_abs_margin_max": disagreement_scaled_abs_margin_max,
                "nvfp4_bit_exact": bool(torch.equal(project_e2, upstream_e2)),
                "nvfp4_max_abs_difference": float(delta_e2.abs().max()),
                "oracle_vs_nvif_bit_exact": bool(torch.equal(project.dequant, upstream_if)),
                "oracle_vs_nvif_max_abs_difference": float(delta_if.abs().max()),
                "oracle_vs_nvif_different_value_count": int((delta_if != 0).sum()),
            }
        )
        print(name, rows[-1]["selector_disagreement_count"], flush=True)

    frame = pd.DataFrame(rows)
    csv_path = OUT / "razer_real_weight_semantics.csv"
    frame.to_csv(csv_path, index=False)
    total_blocks = int(frame["num_k16_blocks"].sum())
    total_disagreement = int(frame["selector_disagreement_count"].sum())
    summary = {
        "captured_at": timestamp(),
        "model": MODEL,
        "model_revision": REVISION,
        "device": "cpu",
        "num_linear_modules": len(frame),
        "num_k16_blocks": total_blocks,
        "all_e2_nvfp4_bit_exact": bool(frame["nvfp4_bit_exact"].all()),
        "all_e2_nvfp4_max_abs_difference": float(frame["nvfp4_max_abs_difference"].max()),
        "oracle_nvif_bit_exact": bool(frame["oracle_vs_nvif_bit_exact"].all()),
        "oracle_nvif_max_abs_difference": float(frame["oracle_vs_nvif_max_abs_difference"].max()),
        "selector_disagreement_count": total_disagreement,
        "selector_disagreement_ratio": total_disagreement / total_blocks,
        "cause": (
            "RaZeR NVIF4 sums candidate squared errors in globally normalized space; "
            "the project Oracle16 sums errors after applying the shared global decode scale. "
            "The objectives are algebraically proportional, but float32 reduction rounding "
            "can flip exact near-zero format margins. Candidate scales, E2/E0 codebooks, "
            "K16 grouping, saturation, and strict E2 tie-breaking are otherwise aligned."
        ),
        "source_table": str(csv_path.relative_to(ROOT)),
    }
    atomic_json(OUT / "razer_real_weight_semantics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
