from __future__ import annotations

import math

import pytest
import torch

from project_quant import build_candidates, quant_mixfp4_granularity
from project_quant.codebook import (
    E0M3_LEVELS,
    E2M1_LEVELS,
    numerical_levels,
    quantize_e0m3,
    quantize_e2m1,
)


def seeded(shape: tuple[int, ...], seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def result(x: torch.Tensor, mode: str, *, role: str = "weight_b", collect: bool = False):
    value = quant_mixfp4_granularity(
        x,
        format_region=mode,
        operand_role=role,
        return_stats=True,
        collect_regions=collect,
    )
    return value


def test_e2m1_codebook() -> None:
    assert numerical_levels(quantize_e2m1, minimum=-8, maximum=8) == list(E2M1_LEVELS)


def test_e0m3_sign_magnitude_codebook_and_no_minus_eight() -> None:
    assert numerical_levels(quantize_e0m3) == list(E0M3_LEVELS)
    probe = quantize_e0m3(torch.tensor([-100.0, 100.0]))
    assert probe.tolist() == [-7.0, 7.0]


def test_k16_scale_group_is_invariant() -> None:
    x = seeded((9, 70))
    candidates = build_candidates(x)
    quantized = result(x, "n8k64")
    assert candidates.e2_scales.shape == (9, math.ceil(70 / 16))
    assert candidates.e0_scales.shape == (9, math.ceil(70 / 16))
    assert quantized.summary["scale_group_size"] == 16
    assert quantized.summary["num_scale_blocks"] == 9 * math.ceil(70 / 16)
    with pytest.raises(ValueError, match="fixes scale_group_size=16"):
        quant_mixfp4_granularity(x, scale_group_size=64)


def test_oracle_not_worse_than_either_fixed_format() -> None:
    x = seeded((17, 79), 3)
    oracle = result(x, "oracle16")
    e2 = result(x, "all_e2m1")
    e0 = result(x, "all_e0m3")
    assert oracle.summary["selected_weight_error"] <= e2.summary["selected_weight_error"] + 1e-6
    assert oracle.summary["selected_weight_error"] <= e0.summary["selected_weight_error"] + 1e-6


@pytest.mark.parametrize(
    "mode",
    ["k32_row", "k64_row", "n8k16", "n2k64", "n4k64", "n8k64", "n16k64", "layer"],
)
def test_mse_constrained_error_not_below_oracle(mode: str) -> None:
    quantized = result(seeded((16, 128), 5), mode)
    assert quantized.summary["constrained_error"] + 1e-5 >= quantized.summary["oracle_error"]
    assert quantized.summary["granularity_regret"] >= -1e-5


def test_format_margin_sign() -> None:
    candidates = build_candidates(seeded((32, 128), 9))
    margin = candidates.e2_errors - candidates.e0_errors
    oracle = candidates.e0_errors < candidates.e2_errors
    assert torch.equal(margin > 0, oracle)
    assert torch.all(margin[oracle] > 0)
    assert torch.all(margin[~oracle] <= 0)


def test_r_g_equals_min_p_g_n_g() -> None:
    quantized = result(seeded((19, 131), 10), "n8k64", collect=True)
    assert quantized.summary["identity_max_abs_residual"] <= 2e-5
    for region in quantized.regions:
        assert region["R_G"] == pytest.approx(min(region["P_G"], region["N_G"]), abs=2e-5)
        assert region["mse_optimal_regret"] == pytest.approx(region["R_G"], abs=2e-5)


def test_nested_mse_granularity_monotonicity() -> None:
    x = seeded((64, 128), 11)
    modes = [
        "oracle16",
        "k32_row",
        "k64_row",
        "n2k64",
        "n4k64",
        "n8k64",
        "n16k64",
        "n32k64",
        "n64k64",
        "layer",
    ]
    errors = [result(x, mode).summary["constrained_error"] for mode in modes]
    assert all(right + 2e-5 >= left for left, right in zip(errors, errors[1:]))


def test_true_n8k64_weight_grouping() -> None:
    quantized = result(seeded((9, 80), 12), "n8k64", collect=True)
    assert [(r["region_n_start"], r["region_k_start"]) for r in quantized.regions] == [
        (0, 0),
        (0, 64),
        (8, 0),
        (8, 64),
    ]
    assert quantized.regions[0]["num_k16_blocks"] == 32
    assert quantized.regions[-1]["num_k16_blocks"] == 1
    assert torch.all(quantized.format_ids[:8, :4] == quantized.format_ids[0, 0])


def test_true_n8k16_weight_grouping() -> None:
    quantized = result(seeded((9, 80), 13), "n8k16", collect=True)
    assert len(quantized.regions) == 10
    assert quantized.regions[0]["num_k16_blocks"] == 8
    assert quantized.regions[-1]["num_k16_blocks"] == 1
    assert torch.all(quantized.format_ids[:8, 0] == quantized.format_ids[0, 0])


def test_true_m16k64_activation_grouping() -> None:
    quantized = result(seeded((2, 17, 70), 14), "m16k64", role="activation_a", collect=True)
    assert quantized.summary["rows"] == 34
    assert len(quantized.regions) == 6
    assert quantized.regions[0]["num_k16_blocks"] == 64
    assert torch.all(quantized.format_ids[:16, :4] == quantized.format_ids[0, 0])


def test_true_m16k16_activation_grouping() -> None:
    quantized = result(seeded((2, 17, 70), 15), "m16k16", role="activation_a", collect=True)
    assert len(quantized.regions) == 15
    assert quantized.regions[0]["num_k16_blocks"] == 16
    assert quantized.regions[-1]["num_k16_blocks"] == 2
    assert torch.all(quantized.format_ids[:16, 0] == quantized.format_ids[0, 0])


@pytest.mark.parametrize(
    "shape,role,mode",
    [((7, 19), "weight_b", "n8k64"), ((3, 5, 19), "activation_a", "m16k64")],
)
def test_tail_dimensions_are_shape_preserving(shape, role, mode) -> None:
    x = seeded(shape, 16)
    quantized = result(x, mode, role=role, collect=True)
    assert quantized.dequant.shape == x.shape
    assert quantized.summary["tail_padding_values"] == 13
    assert "partial" in quantized.summary["tail_policy"]
    assert quantized.regions[-1]["region_k_size"] == 19


def test_deterministic_seeds() -> None:
    first = result(seeded((8, 64), 123), "n8k64")
    second = result(seeded((8, 64), 123), "n8k64")
    assert torch.equal(first.dequant, second.dequant)
    assert torch.equal(first.format_ids, second.format_ids)
    assert first.summary == second.summary


def test_zero_tensor_is_finite_and_exact() -> None:
    x = torch.zeros((9, 70), dtype=torch.bfloat16)
    for mode in ("all_e2m1", "all_e0m3", "oracle16", "n8k64"):
        quantized = result(x, mode)
        assert torch.equal(quantized.dequant, x)
        assert torch.isfinite(quantized.dequant).all()


def test_razer_nvfp4_and_nvif4_baseline_regression() -> None:
    from quantize.quantizer import quant_nvfp4, quant_nvif4

    x = seeded((11, 64), 17, torch.bfloat16)
    reference_e2 = quant_nvfp4(x, n_bits=4, groupsize=16)
    reference_if = quant_nvif4(x, n_bits=4, groupsize=16)
    ours_e2 = quant_mixfp4_granularity(x, format_region="all_e2m1")
    ours_if = quant_mixfp4_granularity(x, format_region="oracle16")
    torch.testing.assert_close(ours_e2, reference_e2, rtol=0, atol=0)
    torch.testing.assert_close(ours_if, reference_if, rtol=0, atol=0)


def test_four_over_six_fixed_e2_reduces_to_canonical_reference() -> None:
    from fouroversix import DataType, QuantizationConfig, QuantizeBackend, dequantize, quantize
    from fouroversix.quantize.pytorch.reference import nvfp4_fouroversix_block_scaled_quantization
    from fouroversix.utils import RoundStyle, ScaleRule

    x = seeded((16, 128), 18, torch.bfloat16)
    amax = x.abs().max().float()
    values, scales = nvfp4_fouroversix_block_scaled_quantization(
        x.float().reshape(-1, 16),
        amax,
        round_style=RoundStyle.nearest,
        scale_rule=ScaleRule.mse,
    )
    reference = (
        values.float().reshape(-1, 16)
        * scales.float().reshape(-1, 1)
        * amax
        / (6 * 256)
    ).reshape_as(x).to(x.dtype)
    ours = quant_mixfp4_granularity(
        x,
        format_region="all_e2m1",
        scale_rule="four_over_six",
    )
    torch.testing.assert_close(ours, reference, rtol=0, atol=0)

    # This crafted real-scale boundary distinguishes the exact canonical
    # arithmetic order from the merely algebraically equivalent implementation
    # that originally passed the small random fixture.
    boundary = torch.zeros((128, 64), dtype=torch.bfloat16)
    boundary[0, 0] = 5.625
    boundary[1, :16] = torch.tensor(
        [
            1.2891, -0.9062, 0.4258, -0.3438, 1.3984, -1.8125, 2.7500, 1.7969,
            -0.7227, -2.0625, -0.3262, -2.3438, -0.4883, -0.6875, 0.2168, -1.0547,
        ],
        dtype=torch.bfloat16,
    )
    packed = quantize(
        boundary,
        QuantizationConfig(
            dtype=DataType.nvfp4,
            scale_rule=ScaleRule.mse,
            backend=QuantizeBackend.pytorch,
        ),
    )
    canonical = dequantize(
        packed,
        dtype=boundary.dtype,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    composed = quant_mixfp4_granularity(
        boundary,
        format_region="all_e2m1",
        scale_rule="four_over_six",
    )
    torch.testing.assert_close(composed, canonical, rtol=0, atol=0)


@pytest.mark.parametrize("label", ["oracle16_4over6", "mixfp4_oracle16_4over6", "n8k64_4over6"])
def test_composed_four_over_six_public_experiment_labels(label: str) -> None:
    from project_quant.modeling import quantize_weight_tensor

    x = seeded((9, 80), 19, torch.bfloat16)
    actual = quantize_weight_tensor(
        x,
        label,
        selector="mse",
        calibration_stats=None,
        collect_regions=True,
    )
    mode = "n8k64" if "n8k64" in label else "oracle16"
    reference = quant_mixfp4_granularity(
        x,
        format_region=mode,
        scale_rule="four_over_six",
        return_stats=True,
        collect_regions=True,
        region_sample_limit=64,
    )
    torch.testing.assert_close(actual.dequant, reference.dequant, rtol=0, atol=0)
    assert actual.summary["scale_rule"] == "four_over_six"
