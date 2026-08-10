from __future__ import annotations

import torch
from torch import nn

from project_quant.calibration import CovarianceCollector
from project_quant.modeling import (
    ModelPreparation,
    capture_layer_output_references,
    evaluate_layer_output_references,
)


class SharedInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(70, 8, bias=False)
        self.k_proj = nn.Linear(70, 8, bias=False)
        self.v_proj = nn.Linear(70, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class TinyHeldOutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(5, 7, bias=True)
        self.down_proj = nn.Linear(7, 5, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        del use_cache
        return self.down_proj(torch.nn.functional.silu(self.q_proj(input_ids.float())))


def test_covariance_collection_is_exact_deduplicated_and_cumulative() -> None:
    generator = torch.Generator().manual_seed(26)
    model = SharedInputModel()
    collector = CovarianceCollector(model, ModelPreparation())
    first = torch.randn((2, 3, 70), generator=generator)
    second = torch.randn((1, 5, 70), generator=generator)
    try:
        collector.begin_sequence()
        model(first)
        payload_one = collector.payload(calibration_sequences=1, metadata={})
        assert set(payload_one["hessian_by_canonical"]) == {"q_proj"}
        assert payload_one["module_to_canonical"] == {
            "k_proj": "q_proj",
            "q_proj": "q_proj",
            "v_proj": "q_proj",
        }
        padded = torch.nn.functional.pad(first.reshape(-1, 70), (0, 58))
        blocks = padded.reshape(-1, 1, 128)
        expected = torch.einsum("tki,tkj->kij", blocks, blocks) / blocks.shape[0]
        torch.testing.assert_close(payload_one["hessian_k128_by_canonical"]["q_proj"], expected)

        collector.begin_sequence()
        model(second)
        payload_two = collector.payload(calibration_sequences=2, metadata={})
        all_values = torch.cat((first.reshape(-1, 70), second.reshape(-1, 70)), dim=0)
        all_padded = torch.nn.functional.pad(all_values, (0, 58)).reshape(-1, 1, 128)
        expected_all = torch.einsum("tki,tkj->kij", all_padded, all_padded) / all_padded.shape[0]
        torch.testing.assert_close(payload_two["hessian_k128_by_canonical"]["q_proj"], expected_all)
        assert payload_two["sample_counts_by_canonical"]["q_proj"] == 11
    finally:
        collector.close()


def test_held_out_layer_output_references_are_deterministic_and_isolate_weight_error() -> None:
    torch.manual_seed(91)
    model = TinyHeldOutModel().eval()
    held_out = torch.randn((2, 9, 5), generator=torch.Generator().manual_seed(92))

    first = capture_layer_output_references(model, held_out, max_rows_per_module=4)
    second = capture_layer_output_references(model, held_out, max_rows_per_module=4)
    assert set(first) == {"q_proj", "down_proj"}
    for name in first:
        assert first[name]["sampled_rows"] == 4
        torch.testing.assert_close(first[name]["inputs"], second[name]["inputs"], rtol=0, atol=0)
        torch.testing.assert_close(first[name]["reference"], second[name]["reference"], rtol=0, atol=0)

    pristine = evaluate_layer_output_references(model, first)
    assert all(row["mse"] == 0.0 for row in pristine)

    with torch.no_grad():
        model.q_proj.weight[0, 0] += 0.25
    perturbed = {row["module_name"]: row for row in evaluate_layer_output_references(model, first)}
    assert perturbed["q_proj"]["mse"] > 0.0
    assert perturbed["down_proj"]["mse"] == 0.0
    assert perturbed["q_proj"]["sampled_activation_rows"] == 4
