#!/usr/bin/env python3
"""Run one traceable single-GPU PPL experiment through the mandatory guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstreams" / "NVFP4-RaZeR"))

from project_quant.artifacts import ExperimentLedger, atomic_json
from project_quant.calibration import load_calibration_by_module
from project_quant.data import load_sequences, prepare_c4, prepare_wikitext
from project_quant.modeling import (
    ActivationHooks,
    ModelPreparation,
    aggregate_activation_metrics,
    aggregate_weight_metrics,
    apply_rotations,
    capture_layer_output_references,
    evaluate_layer_output_references,
    evaluate_ppl,
    load_model,
    prepare_permutations,
    quantize_model_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("wikitext", "c4"), required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--c4-count", type=int, default=256)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--weight-mode", default="high_precision")
    parser.add_argument("--activation-mode", default="high_precision")
    parser.add_argument("--selector", choices=("mse", "activation_aware", "output_aware"), default="mse")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--calibration-size", type=int, default=0)
    parser.add_argument("--calibration-file")
    parser.add_argument("--rotation-map")
    parser.add_argument(
        "--layer-output-error-rows",
        type=int,
        default=0,
        help="deterministic held-out activation rows per Linear for isolated output error",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config["domain"] = "llm"
    ledger = ExperimentLedger(args.experiment_id, args.phase, config)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(False)
        record = (
            prepare_wikitext(args.model, args.seq_len)
            if args.dataset == "wikitext"
            else prepare_c4(args.model, args.seq_len, args.c4_count, 0)
        )
        sequences = load_sequences(record)
        model, snapshot, revision = load_model(args.model)
        calibration_by_module = None
        calibration_metadata = None
        if args.calibration_file:
            calibration_by_module, calibration_payload = load_calibration_by_module(args.calibration_file)
            calibration_metadata = calibration_payload.get("metadata", {})
            if args.calibration_size and calibration_payload["calibration_sequences"] != args.calibration_size:
                raise ValueError("--calibration-size does not match calibration file")
        if args.selector in {"activation_aware", "output_aware"} and calibration_by_module is None:
            raise ValueError("output-aware selector requires --calibration-file")
        preparation = ModelPreparation()
        prepare_permutations(
            model,
            args.permutation,
            preparation,
            calibration_by_module=calibration_by_module,
        )
        rotation_map = json.loads(Path(args.rotation_map).read_text()) if args.rotation_map else None
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        apply_rotations(model, args.rotation, preparation, per_module_rotation=rotation_map)
        if args.selector in {"activation_aware", "output_aware"}:
            expected_permutation = calibration_metadata.get("permutation", "none")
            expected_rotation = calibration_metadata.get("rotation", "identity")
            if expected_permutation != args.permutation or expected_rotation != rotation_label:
                raise ValueError(
                    "output-aware calibration provenance mismatch: "
                    f"file has permutation={expected_permutation}, rotation={expected_rotation}; "
                    f"experiment requests permutation={args.permutation}, rotation={rotation_label}"
                )
        layer_output_references = {}
        layer_output_reference_metadata = None
        if args.layer_output_error_rows:
            if args.permutation not in {"none", "no_permutation"} or rotation_label != "identity":
                raise ValueError(
                    "held-out layer-output capture is currently restricted to identity layout; "
                    "combined layouts use matched calibration-output SSE"
                )
            heldout_array = sequences[0]
            heldout_ids = torch.from_numpy(
                heldout_array.astype("int64", copy=False)
            ).unsqueeze(0).to("cuda:0")
            layer_output_references = capture_layer_output_references(
                model,
                heldout_ids,
                max_rows_per_module=args.layer_output_error_rows,
            )
            del heldout_ids
            layer_output_reference_metadata = {
                "selection_use": "evaluation_only_never_used_by_selector",
                "dataset_manifest_key": record["key"],
                "dataset_manifest_sha256": record["data_sha256"],
                "evaluation_sequence_index": 0,
                "token_ids_sha256": hashlib.sha256(heldout_array.tobytes()).hexdigest(),
                "sequence_length": int(heldout_array.size),
                "rows_per_module": args.layer_output_error_rows,
            }
        quantize_model_weights(
            model,
            args.weight_mode,
            preparation,
            selector=args.selector,
            calibration_by_module=calibration_by_module,
        )
        layer_output_metrics = evaluate_layer_output_references(
            model, layer_output_references
        )
        layer_output_references.clear()
        activation_hooks = ActivationHooks(model, args.activation_mode, preparation)
        ppl = evaluate_ppl(model, sequences, limit=args.eval_limit)

        for row in preparation.layer_metrics:
            row.update({"experiment_id": args.experiment_id, "model": args.model})
        for row in preparation.region_metrics:
            row.update({"experiment_id": args.experiment_id, "model": args.model})
        for row in preparation.margin_samples:
            row.update({"experiment_id": args.experiment_id, "model": args.model})
        for row in activation_hooks.metrics:
            row.update({"experiment_id": args.experiment_id, "model": args.model})
        for row in layer_output_metrics:
            row.update({"experiment_id": args.experiment_id, "model": args.model})

        pd.DataFrame(preparation.layer_metrics).to_csv(ledger.directory / "per_layer_metrics.csv", index=False)
        pd.DataFrame(preparation.region_metrics).to_csv(ledger.directory / "format_region_metrics.csv", index=False)
        pd.DataFrame(preparation.margin_samples).to_csv(ledger.directory / "format_margin_samples.csv", index=False)
        pd.DataFrame(activation_hooks.metrics).to_csv(ledger.directory / "activation_metrics.csv", index=False)
        pd.DataFrame(layer_output_metrics).to_csv(
            ledger.directory / "per_layer_output_metrics.csv", index=False
        )
        atomic_json(ledger.directory / "equivalence_checks.json", preparation.equivalence_checks)
        atomic_json(ledger.directory / "permutations.json", preparation.permutation_records)
        atomic_json(ledger.directory / "ppl_raw.json", ppl)
        atomic_json(ledger.directory / "calibration_metadata.json", calibration_metadata)
        atomic_json(
            ledger.directory / "layer_output_reference_metadata.json",
            layer_output_reference_metadata,
        )

        weight_aggregate = aggregate_weight_metrics(preparation.layer_metrics)
        activation_aggregate = aggregate_activation_metrics(activation_hooks.metrics)
        layer_output_elements = sum(row["numel"] for row in layer_output_metrics)
        layer_output_aggregate = {
            "layer_output_mse": (
                sum(row["mse"] * row["numel"] for row in layer_output_metrics)
                / layer_output_elements
                if layer_output_elements
                else None
            ),
            "layer_output_nmse_weighted": (
                sum(row["nmse"] * row["numel"] for row in layer_output_metrics)
                / layer_output_elements
                if layer_output_elements
                else None
            ),
            "layer_output_relative_l2_weighted": (
                sum(row["relative_l2"] * row["numel"] for row in layer_output_metrics)
                / layer_output_elements
                if layer_output_elements
                else None
            ),
            "layer_output_cosine_error_weighted": (
                sum(row["cosine_error"] * row["numel"] for row in layer_output_metrics)
                / layer_output_elements
                if layer_output_elements
                else None
            ),
            "layer_output_max_abs_error": (
                max(row["max_abs_error"] for row in layer_output_metrics)
                if layer_output_metrics
                else None
            ),
            "layer_output_num_modules": len(layer_output_metrics),
            "layer_output_sample_rows_per_module": args.layer_output_error_rows,
        }
        scale_rule = (
            preparation.layer_metrics[0].get("scale_rule", "standard")
            if preparation.layer_metrics
            else "none"
        )
        summary = {
            "experiment_id": args.experiment_id,
            "phase": args.phase,
            "domain": "llm",
            "model": args.model,
            "model_revision": revision,
            "model_snapshot": snapshot,
            "dataset": args.dataset,
            "dataset_or_promptset": args.dataset,
            "dataset_manifest_key": record["key"],
            "dataset_manifest_sha256": record["data_sha256"],
            "quantization_mode": f"W={args.weight_mode}/A={args.activation_mode}",
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "weight_format_granularity": args.weight_mode,
            "activation_format_granularity": args.activation_mode,
            "scale_group_size": 16,
            "scale_rule": scale_rule,
            "selector": args.selector,
            "rotation": rotation_label,
            "rotation_seed": int(args.rotation.rsplit("seed", 1)[1]) if "seed" in args.rotation else None,
            "rotation_map": args.rotation_map,
            "permutation": args.permutation,
            "calibration_size": args.calibration_size,
            "calibration_file": args.calibration_file,
            "layer_output_reference_metadata": layer_output_reference_metadata,
            "gpu_index": int(__import__("os").environ["MIXFP4_PHYSICAL_GPU"]),
            "physical_gpu_index": int(__import__("os").environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(__import__("os").environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": __import__("os").environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
            "ppl": ppl["ppl"],
            "num_eval_sequences": ppl["num_sequences"],
            "num_eval_tokens": ppl["total_tokens"],
            "eval_wall_time_seconds": ppl["wall_time_seconds"],
            "status": "completed",
            "four_over_six_mode": "canonical" if args.weight_mode in {"4over6", "nvfp4_4over6", "fouroversix_4over6"} else "composed" if args.weight_mode.endswith("_4over6") else "none",
            **weight_aggregate,
            **activation_aggregate,
            **layer_output_aggregate,
        }
        atomic_json(
            ledger.directory / "raw_metrics.json",
            {
                "ppl": ppl,
                "weights": weight_aggregate,
                "activations": activation_aggregate,
                "layer_outputs": layer_output_aggregate,
            },
        )
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
