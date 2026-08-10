from __future__ import annotations

import torch
from torch import nn

from project_quant.adapters.deepcompressor_diffusion import (
    DiffusionActivationHooks,
    DiffusionPreparation,
    prepare_sana_permutations,
)
from project_quant.modeling import ActivationHooks, ModelPreparation, prepare_permutations
from project_quant.permutation import (
    apply_foldable_ffn_permutation,
    apply_foldable_glumbconv_permutation,
    apply_foldable_mlp_permutation,
    choose_permutation,
    inverse_output_hook,
    permute_linear_output,
)
from project_quant.rotation import TRANSFORM_BANK, apply_transform, dense_transform_matrix, verify_rotation_equivalence


class GatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(16, 24, bias=True)
        self.up_proj = nn.Linear(16, 24, bias=True)
        self.down_proj = nn.Linear(24, 16, bias=True)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DiffusionFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(16, 24, bias=True)
        self.second = nn.Linear(24, 16, bias=True)

    def forward(self, x):
        return self.second(torch.nn.functional.gelu(self.first(x)))


class SanaGLUMBConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(16, 48, 1)
        self.conv_depth = nn.Conv2d(48, 48, 3, padding=1, groups=48)
        self.conv_point = nn.Conv2d(24, 16, 1, bias=False)

    def forward(self, x):
        value = self.nonlinearity(self.conv_inverted(x))
        value = self.conv_depth(value)
        value, gate = torch.chunk(value, 2, dim=1)
        return self.conv_point(value * self.nonlinearity(gate))


def test_all_linear_explicit_inverse_permutation_equivalence() -> None:
    generator = torch.Generator().manual_seed(20)
    linear = nn.Linear(23, 17, bias=True)
    x = torch.randn((3, 5, 23), generator=generator)
    reference = linear(x)
    permutation = torch.randperm(17, generator=generator)
    inverse = permute_linear_output(linear, permutation)
    handle = linear.register_forward_hook(inverse_output_hook(inverse))
    try:
        actual = linear(x)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)


def test_high_precision_mlp_permutation_equivalence() -> None:
    generator = torch.Generator().manual_seed(21)
    model = GatedMLP()
    x = torch.randn((2, 7, 16), generator=generator)
    reference = model(x)
    apply_foldable_mlp_permutation(model, torch.randperm(24, generator=generator))
    torch.testing.assert_close(model(x), reference, rtol=1e-6, atol=1e-7)


def test_high_precision_diffusion_ffn_permutation_equivalence() -> None:
    generator = torch.Generator().manual_seed(22)
    model = DiffusionFFN()
    x = torch.randn((2, 7, 16), generator=generator)
    reference = model(x)
    apply_foldable_ffn_permutation(model.first, model.second, torch.randperm(24, generator=generator))
    torch.testing.assert_close(model(x), reference, rtol=1e-6, atol=1e-7)


def test_high_precision_sana_glumbconv_permutation_equivalence() -> None:
    generator = torch.Generator().manual_seed(220)
    model = SanaGLUMBConv()
    x = torch.randn((2, 16, 5, 7), generator=generator)
    reference = model(x)
    apply_foldable_glumbconv_permutation(model, torch.randperm(24, generator=generator))
    torch.testing.assert_close(model(x), reference, rtol=1e-6, atol=2e-7)


def test_permutation_methods_are_exact_and_deterministic() -> None:
    generator = torch.Generator().manual_seed(23)
    e2, e0 = torch.rand((73, 20), generator=generator), torch.rand((73, 20), generator=generator)
    for method in ("no_permutation", "sort_by_e0_ratio", "margin_vector_clustering", "greedy_min_regret_n8"):
        first = choose_permutation(method, e2, e0)
        second = choose_permutation(method, e2, e0)
        assert torch.equal(first, second)
        assert torch.equal(torch.sort(first).values, torch.arange(73))
    sensitivity_e2 = e2.square()
    sensitivity_e0 = e0.square()
    permutation = choose_permutation(
        "sensitivity_weighted_greedy_n8",
        e2,
        e0,
        sensitivity_e2=sensitivity_e2,
        sensitivity_e0=sensitivity_e0,
    )
    assert torch.equal(torch.sort(permutation).values, torch.arange(73))


def test_rotation_dense_equivalence_and_orthogonality() -> None:
    generator = torch.Generator().manual_seed(24)
    x = torch.randn((7, 150), generator=generator)
    for spec in TRANSFORM_BANK:
        dense = dense_transform_matrix(spec, 150)
        torch.testing.assert_close(apply_transform(x, spec), x @ dense, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(dense.T @ dense, torch.eye(150), rtol=2e-6, atol=2e-6)


def test_high_precision_rotation_equivalence() -> None:
    generator = torch.Generator().manual_seed(25)
    x = torch.randn((11, 150), generator=generator)
    weight = torch.randn((19, 150), generator=generator)
    for spec in TRANSFORM_BANK:
        check = verify_rotation_equivalence(x, weight, spec)
        assert check["passed"], check


def test_llm_activation_hook_persists_sampled_regions() -> None:
    model = nn.Sequential(nn.Linear(19, 13, bias=False))
    preparation = ModelPreparation()
    hooks = ActivationHooks(model, "m16k64", preparation, max_stats_calls_per_module=1)
    output = model(torch.randn(2, 7, 19))
    assert output.shape == (2, 7, 13)
    assert len(hooks.metrics) == 1
    assert preparation.region_metrics
    assert all(row["operand_role"] == "activation_a" for row in preparation.region_metrics)


def test_diffusion_activation_hook_preserves_conv_layout_and_regions() -> None:
    model = nn.Sequential(nn.Conv2d(19, 13, kernel_size=1, bias=False))
    preparation = DiffusionPreparation()
    hooks = DiffusionActivationHooks(model, "m16k64", preparation, max_stats_calls_per_module=1)
    output = model(torch.randn(2, 19, 3, 5))
    assert output.shape == (2, 13, 3, 5)
    assert len(hooks.metrics) == 1
    assert preparation.region_metrics
    assert all(row["operand_role"] == "activation_a" for row in preparation.region_metrics)


def test_sensitivity_weighted_foldable_mlp_uses_calibration() -> None:
    class CalibratedMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(64, 16, bias=False)
            self.up_proj = nn.Linear(64, 16, bias=False)
            self.down_proj = nn.Linear(16, 64, bias=False)

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = CalibratedMLP()

    model = Wrapper()
    calibration = {
        "mlp.gate_proj": {"hessian_k64": torch.eye(64).unsqueeze(0)},
        "mlp.up_proj": {"hessian_k64": torch.eye(64).unsqueeze(0)},
    }
    preparation = ModelPreparation()
    prepare_permutations(
        model,
        "foldable_mlp_sensitivity_weighted_greedy_n8",
        preparation,
        calibration_by_module=calibration,
    )
    assert preparation.permutation_records
    assert preparation.permutation_records[0]["deployability"] == "exact_foldable"


def test_sana_all_matrix_sensitivity_packing_covers_linear_and_conv() -> None:
    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 16)
            self.conv = nn.Conv2d(64, 16, 1)

    model = TinyTransformer()
    calibration = {
        name: {"hessian_k64": torch.eye(64).unsqueeze(0)}
        for name in ("linear", "conv")
    }
    preparation = DiffusionPreparation()
    prepare_sana_permutations(
        model,
        "all_linear_sensitivity_weighted_greedy_n8",
        preparation,
        calibration_by_module=calibration,
    )
    assert {row["module_class"] for row in preparation.permutation_records} == {
        "Linear",
        "Conv2d",
    }
    assert all(row["n8k64_regret_after"] >= -1e-7 for row in preparation.permutation_records)
    assert all(row["deployability"] == "upper_bound_only" for row in preparation.permutation_records)


def test_sana_foldable_sensitivity_packing_records_motif_regret() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ff = SanaGLUMBConv()

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block()])

    model = TinyTransformer()
    calibration = {
        "transformer_blocks.0.ff.conv_inverted": {
            "hessian_k64": torch.eye(64).unsqueeze(0)
        }
    }
    # The toy motif's K=16 requires four padded K64 Hessian blocks only in the
    # shared selector representation; one physical K64 block is sufficient.
    preparation = DiffusionPreparation()
    prepare_sana_permutations(
        model,
        "foldable_sana_ffn_sensitivity_weighted_greedy_n8",
        preparation,
        calibration_by_module=calibration,
    )
    record = preparation.permutation_records[0]
    assert record["deployability"] == "exact_foldable"
    assert record["motif_n8k64_regret_before"] >= -1e-7
    assert record["motif_n8k64_regret_after"] >= -1e-7
