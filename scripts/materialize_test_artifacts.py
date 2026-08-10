#!/usr/bin/env python3
"""Materialize numerical evidence behind the semantic pytest suite."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstreams" / "NVFP4-RaZeR"))

from project_quant import build_candidates, quant_mixfp4_granularity  # noqa: E402
from project_quant.artifacts import atomic_json, code_fingerprint, timestamp  # noqa: E402
from project_quant.codebook import (  # noqa: E402
    E0M3_LEVELS,
    E2M1_LEVELS,
    numerical_levels,
    quantize_e0m3,
    quantize_e2m1,
)
from project_quant.permutation import (  # noqa: E402
    apply_foldable_ffn_permutation,
    apply_foldable_glumbconv_permutation,
    apply_foldable_mlp_permutation,
)
from project_quant.rotation import TRANSFORM_BANK, verify_rotation_equivalence  # noqa: E402


OUT = ROOT / "artifacts" / "02_tests"


def seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def q(x: torch.Tensor, mode: str, role: str = "weight_b", collect: bool = False, scale_rule: str = "standard"):
    return quant_mixfp4_granularity(
        x,
        format_region=mode,
        operand_role=role,
        scale_rule=scale_rule,
        return_stats=True,
        collect_regions=collect,
    )


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    captured = timestamp()

    observed_e2 = numerical_levels(quantize_e2m1, minimum=-8, maximum=8)
    observed_e0 = numerical_levels(quantize_e0m3)
    atomic_json(
        OUT / "codebook_validation.json",
        {
            "captured_at": captured,
            "e2m1_expected_positive_magnitudes": list(E2M1_LEVELS),
            "e2m1_observed_positive_magnitudes": observed_e2,
            "e2m1_exact": observed_e2 == list(E2M1_LEVELS),
            "e0m3_expected_positive_magnitudes": list(E0M3_LEVELS),
            "e0m3_observed_positive_magnitudes": observed_e0,
            "e0m3_exact": observed_e0 == list(E0M3_LEVELS),
            "e0m3_saturation_probe": quantize_e0m3(torch.tensor([-100.0, 100.0])).tolist(),
            "minus_eight_present": -8.0 in quantize_e0m3(torch.linspace(-100, 100, 100001)).unique().tolist(),
        },
    )

    x = seeded((19, 131), 10)
    modes = ("oracle16", "k32_row", "k64_row", "n8k16", "n2k64", "n4k64", "n8k64", "n16k64", "layer")
    mode_values = {mode: q(x, mode, collect=mode in {"n8k16", "n8k64"}) for mode in modes}
    candidates = build_candidates(x)
    oracle = mode_values["oracle16"]
    all_e2 = q(x, "all_e2m1")
    all_e0 = q(x, "all_e0m3")
    margin = candidates.e2_errors - candidates.e0_errors
    nested_x = seeded((64, 128), 11)
    nested_modes = ("oracle16", "k32_row", "k64_row", "n2k64", "n4k64", "n8k64", "n16k64", "n32k64", "n64k64", "layer")
    nested_errors = {mode: q(nested_x, mode).summary["constrained_error"] for mode in nested_modes}
    layout_cases = {
        "N8K64": q(seeded((9, 80), 12), "n8k64", collect=True),
        "N8K16": q(seeded((9, 80), 13), "n8k16", collect=True),
        "M16K64": q(seeded((2, 17, 70), 14), "m16k64", "activation_a", True),
        "M16K16": q(seeded((2, 17, 70), 15), "m16k16", "activation_a", True),
    }
    layout_payload = {}
    for name, value in layout_cases.items():
        layout_payload[name] = {
            "rows": value.summary["rows"],
            "K": value.summary["K"],
            "scale_group_size": value.summary["scale_group_size"],
            "format_region_rows": value.summary["format_region_rows"],
            "format_region_k_values": value.summary["format_region_k_values"],
            "num_regions": value.summary["num_format_regions"],
            "region_bounds": [
                {
                    "row_start": row["region_n_start"],
                    "k_start": row["region_k_start"],
                    "row_size": row["region_n_size"],
                    "k_size": row["region_k_size"],
                    "real_k16_blocks": row["num_k16_blocks"],
                }
                for row in value.regions
            ],
        }
    tail_records = []
    for shape, role, mode in (((7, 19), "weight_b", "n8k64"), ((3, 5, 19), "activation_a", "m16k64")):
        value = q(seeded(shape, 16), mode, role, True)
        tail_records.append(
            {
                "input_shape": list(shape),
                "output_shape": list(value.dequant.shape),
                "operand_role": role,
                "mode": mode,
                "tail_padding_values": value.summary["tail_padding_values"],
                "tail_policy": value.summary["tail_policy"],
                "last_region_real_k_size": value.regions[-1]["region_k_size"],
            }
        )
    atomic_json(
        OUT / "granularity_invariants.json",
        {
            "captured_at": captured,
            "scale_group_size": 16,
            "candidate_scale_shape": list(candidates.e2_scales.shape),
            "candidate_expected_scale_shape": [19, math.ceil(131 / 16)],
            "oracle_error": oracle.summary["selected_weight_error"],
            "all_e2_error": all_e2.summary["selected_weight_error"],
            "all_e0_error": all_e0.summary["selected_weight_error"],
            "oracle_le_all_e2": oracle.summary["selected_weight_error"] <= all_e2.summary["selected_weight_error"],
            "oracle_le_all_e0": oracle.summary["selected_weight_error"] <= all_e0.summary["selected_weight_error"],
            "constrained_errors": {mode: value.summary["constrained_error"] for mode, value in mode_values.items()},
            "all_constrained_ge_oracle": all(value.summary["constrained_error"] + 2e-5 >= oracle.summary["oracle_error"] for value in mode_values.values()),
            "format_margin_positive_iff_e0_wins": bool(torch.equal(margin > 0, candidates.e0_errors < candidates.e2_errors)),
            "r_g_identity_max_abs_residual": mode_values["n8k64"].summary["identity_max_abs_residual"],
            "nested_mode_order": list(nested_modes),
            "nested_constrained_errors": nested_errors,
            "nested_monotonic": all(b + 2e-5 >= a for a, b in zip(nested_errors.values(), list(nested_errors.values())[1:])),
            "layouts": layout_payload,
            "tail_cases": tail_records,
        },
    )

    det_x1 = seeded((8, 64), 123)
    det_x2 = seeded((8, 64), 123)
    det_1, det_2 = q(det_x1, "n8k64"), q(det_x2, "n8k64")
    atomic_json(
        OUT / "determinism.json",
        {
            "captured_at": captured,
            "seed": 123,
            "inputs_equal": bool(torch.equal(det_x1, det_x2)),
            "dequant_equal": bool(torch.equal(det_1.dequant, det_2.dequant)),
            "format_ids_equal": bool(torch.equal(det_1.format_ids, det_2.format_ids)),
            "dequant_sha256": tensor_digest(det_1.dequant),
            "format_ids_sha256": tensor_digest(det_1.format_ids),
        },
    )

    from quantize.quantizer import quant_nvfp4, quant_nvif4

    baseline_x = seeded((11, 64), 17).to(torch.bfloat16)
    ours_nv = quant_mixfp4_granularity(baseline_x, format_region="all_e2m1")
    ours_if = quant_mixfp4_granularity(baseline_x, format_region="oracle16")
    ref_nv, ref_if = quant_nvfp4(baseline_x, n_bits=4, groupsize=16), quant_nvif4(baseline_x, n_bits=4, groupsize=16)
    atomic_json(
        OUT / "razer_baseline_regression.json",
        {
            "captured_at": captured,
            "input_seed": 17,
            "input_shape": [11, 64],
            "nvfp4_bit_exact": bool(torch.equal(ours_nv, ref_nv)),
            "nvfp4_max_abs_difference": float((ours_nv.float() - ref_nv.float()).abs().max()),
            "nvif4_bit_exact": bool(torch.equal(ours_if, ref_if)),
            "nvif4_max_abs_difference": float((ours_if.float() - ref_if.float()).abs().max()),
            "upstream_commit": "78be7fc78857b635ef789edf817cab77cc7f4a01",
        },
    )

    from fouroversix import DataType, QuantizationConfig, QuantizeBackend, dequantize, quantize
    from fouroversix.quantize.pytorch.reference import nvfp4_fouroversix_block_scaled_quantization
    from fouroversix.utils import RoundStyle, ScaleRule

    composition_x = seeded((16, 128), 18).to(torch.bfloat16)
    amax = composition_x.abs().max().float()
    values, scales = nvfp4_fouroversix_block_scaled_quantization(
        composition_x.float().reshape(-1, 16), amax, round_style=RoundStyle.nearest, scale_rule=ScaleRule.mse
    )
    reference = (values.float() * scales.float().reshape(-1, 1) * amax / (6 * 256)).reshape_as(composition_x).to(composition_x.dtype)
    composed_e2 = quant_mixfp4_granularity(composition_x, format_region="all_e2m1", scale_rule="four_over_six")
    difference = composed_e2.float() - reference.float()
    public_quantized = quantize(
        composition_x,
        QuantizationConfig(
            dtype=DataType.nvfp4,
            scale_rule=ScaleRule.mse,
            backend=QuantizeBackend.pytorch,
        ),
    )
    public_reference = dequantize(
        public_quantized,
        dtype=composition_x.dtype,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    public_difference = composed_e2.float() - public_reference.float()
    boundary = torch.zeros((128, 64), dtype=torch.bfloat16)
    boundary[0, 0] = 5.625
    boundary[1, :16] = torch.tensor(
        [
            1.2891, -0.9062, 0.4258, -0.3438, 1.3984, -1.8125, 2.7500, 1.7969,
            -0.7227, -2.0625, -0.3262, -2.3438, -0.4883, -0.6875, 0.2168, -1.0547,
        ],
        dtype=torch.bfloat16,
    )
    boundary_quantized = quantize(
        boundary,
        QuantizationConfig(
            dtype=DataType.nvfp4,
            scale_rule=ScaleRule.mse,
            backend=QuantizeBackend.pytorch,
        ),
    )
    boundary_reference = dequantize(
        boundary_quantized,
        dtype=boundary.dtype,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    boundary_composed = quant_mixfp4_granularity(
        boundary, format_region="all_e2m1", scale_rule="four_over_six"
    )
    boundary_difference = boundary_composed.float() - boundary_reference.float()
    reduction_passed = bool(
        torch.equal(composed_e2, reference)
        and torch.equal(composed_e2, public_reference)
        and torch.equal(boundary_composed, boundary_reference)
    )
    real_model_root = (
        ROOT
        / "artifacts"
        / "03_phase_a"
        / "llm"
        / "raw"
        / "phase0_smoke_llama32_1b_wikitext_fouroversix_4over6_a6000"
    )
    real_summaries = sorted(real_model_root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    real_model_evidence = None
    if real_summaries:
        real_summary_path = real_summaries[-1]
        real_summary = json.loads(real_summary_path.read_text(encoding="utf-8"))
        layer_path = real_summary_path.parent / "per_layer_metrics.csv"
        import pandas as pd

        layer = pd.read_csv(layer_path)
        exact = layer["canonical_reference_bit_exact"].fillna(False).astype(bool)
        max_difference = pd.to_numeric(
            layer["canonical_reference_max_abs_diff"], errors="coerce"
        ).max()
        real_model_evidence = {
            "experiment_id": real_summary["experiment_id"],
            "model": real_summary["model"],
            "model_revision": real_summary["model_revision"],
            "physical_gpu_index": real_summary["physical_gpu_index"],
            "gpu_type": real_summary["gpu_type"],
            "num_linear_tensors": int(len(layer)),
            "all_linear_tensors_bit_exact": bool(exact.all()),
            "max_abs_difference": float(max_difference),
            "ppl_smoke": float(real_summary["ppl"]),
            "artifact_dir": str(real_summary_path.parent.relative_to(ROOT)),
        }
        reduction_passed = reduction_passed and real_model_evidence[
            "all_linear_tensors_bit_exact"
        ]
    atomic_json(
        OUT / "four_over_six_reduction.json",
        {
            "captured_at": captured,
            "reference": "pinned FourOverSix public PyTorch quantize/dequantize plus low-level nvfp4_fouroversix_block_scaled_quantization",
            "input_seed": 18,
            "input_shape": [16, 128],
            "num_values_compared": composition_x.numel(),
            "bit_exact": bool(torch.equal(composed_e2, reference)),
            "max_abs_difference": float(difference.abs().max()),
            "mse_difference": float(difference.square().mean()),
            "public_api_bit_exact": bool(torch.equal(composed_e2, public_reference)),
            "public_api_max_abs_difference": float(public_difference.abs().max()),
            "boundary_input_shape": list(boundary.shape),
            "boundary_public_api_bit_exact": bool(torch.equal(boundary_composed, boundary_reference)),
            "boundary_public_api_max_abs_difference": float(boundary_difference.abs().max()),
            "arithmetic_order": "canonical x*(1/(decode_scale*block_scale)) and code*block_scale*amax/(6*256)",
            "tolerance_atol": 0.0,
            "tolerance_rtol": 0.0,
            "fixed_e2_reduces_to_canonical": reduction_passed,
            "real_model_reduction": real_model_evidence,
            "composition_classification": "validated project composition; only the fixed-E2 reduction is canonical",
            "fouroversix_commit": "dadfad6901d473a734fe71e0b082e70ee993e23a",
        },
    )

    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(16, 24, bias=True)
            self.up_proj = nn.Linear(16, 24, bias=True)
            self.down_proj = nn.Linear(24, 16, bias=True)

        def forward(self, value):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))

    class DiffusionFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(16, 24, bias=True)
            self.second = nn.Linear(24, 16, bias=True)

        def forward(self, value):
            return self.second(torch.nn.functional.gelu(self.first(value)))

    class SanaGLUMBConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.nonlinearity = nn.SiLU()
            self.conv_inverted = nn.Conv2d(16, 48, 1)
            self.conv_depth = nn.Conv2d(48, 48, 3, padding=1, groups=48)
            self.conv_point = nn.Conv2d(24, 16, 1, bias=False)

        def forward(self, value):
            value = self.nonlinearity(self.conv_inverted(value))
            value = self.conv_depth(value)
            value, gate = torch.chunk(value, 2, dim=1)
            return self.conv_point(value * self.nonlinearity(gate))

    permutation_rows = []
    for name, model, seed in (("llm_gated_mlp", GatedMLP(), 21), ("diffusion_ffn", DiffusionFFN(), 22)):
        generator = torch.Generator().manual_seed(seed)
        value = torch.randn((2, 7, 16), generator=generator)
        before = model(value)
        permutation = torch.randperm(24, generator=generator)
        if name == "llm_gated_mlp":
            apply_foldable_mlp_permutation(model, permutation)
        else:
            apply_foldable_ffn_permutation(model.first, model.second, permutation)
        after = model(value)
        delta = after - before
        permutation_rows.append(
            {
                "motif": name,
                "seed": seed,
                "deployability": "exact_foldable",
                "max_abs_difference": float(delta.detach().abs().max()),
                "relative_l2": float(torch.linalg.vector_norm(delta.detach()) / torch.linalg.vector_norm(before.detach())),
                "passed": bool(torch.allclose(after, before, rtol=1e-6, atol=1e-7)),
            }
        )
    glumb = SanaGLUMBConv()
    glumb_generator = torch.Generator().manual_seed(220)
    glumb_input = torch.randn((2, 16, 5, 7), generator=glumb_generator)
    glumb_before = glumb(glumb_input)
    apply_foldable_glumbconv_permutation(glumb, torch.randperm(24, generator=glumb_generator))
    glumb_after = glumb(glumb_input)
    glumb_delta = glumb_after - glumb_before
    permutation_rows.append(
        {
            "motif": "sana_glumbconv",
            "seed": 220,
            "deployability": "exact_foldable",
            "max_abs_difference": float(glumb_delta.detach().abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(glumb_delta.detach())
                / torch.linalg.vector_norm(glumb_before.detach()).clamp_min(1e-30)
            ),
            "passed": bool(torch.allclose(glumb_after, glumb_before, rtol=1e-6, atol=2e-7)),
        }
    )
    atomic_json(OUT / "permutation_equivalence.json", {"captured_at": captured, "checks": permutation_rows})

    rotation_x = seeded((11, 150), 25)
    rotation_w = seeded((19, 150), 125)
    rotation_rows = [verify_rotation_equivalence(rotation_x, rotation_w, spec) for spec in TRANSFORM_BANK]
    for row, spec in zip(rotation_rows, TRANSFORM_BANK):
        row["transform"] = spec.name
        row["deployability"] = spec.deployability
    atomic_json(OUT / "rotation_equivalence.json", {"captured_at": captured, "checks": rotation_rows})

    junit_path = OUT / "unit_tests.xml"
    junit = ET.parse(junit_path).getroot() if junit_path.exists() else None
    suites = [junit] if junit is not None and junit.tag == "testsuite" else (
        list(junit) if junit is not None else []
    )
    junit_summary = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites) if junit is not None else None
        for key in ("tests", "failures", "errors", "skipped")
    }
    report = {
        "captured_at": captured,
        "command": "python3 -m pytest -q tests --junitxml=artifacts/02_tests/unit_tests.xml",
        "junit": junit_summary,
        "passed": junit_summary.get("tests") if junit_summary.get("failures") == 0 and junit_summary.get("errors") == 0 else None,
        "code_fingerprint_sha256": code_fingerprint(),
        "artifacts": sorted(str(path.relative_to(ROOT)) for path in OUT.glob("*.json")),
        "diffusion_fixed_prompt_seed_determinism": (
            json.loads((OUT / "diffusion_determinism.json").read_text(encoding="utf-8"))
            if (OUT / "diffusion_determinism.json").exists()
            else "pending real SANA pipeline check; not counted as passed"
        ),
    }
    atomic_json(OUT / "test_report.json", report)
    (OUT / "unit_test_report.txt").write_text(
        f"Command: {report['command']}\nTests: {junit_summary['tests']}\nFailures: {junit_summary['failures']}\nErrors: {junit_summary['errors']}\nSkipped: {junit_summary['skipped']}\n"
    )


if __name__ == "__main__":
    main()
