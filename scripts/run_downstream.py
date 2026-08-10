#!/usr/bin/env python3
"""Evaluate a completed finalist configuration on the mandatory zero-shot task set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402
from project_quant.calibration import load_calibration_by_module  # noqa: E402
from project_quant.modeling import (  # noqa: E402
    ActivationHooks,
    ModelPreparation,
    apply_rotations,
    load_model,
    prepare_permutations,
    quantize_model_weights,
)


TASKS = ("arc_easy", "hellaswag", "piqa", "winogrande")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight-mode", required=True)
    parser.add_argument("--activation-mode", default="high_precision")
    parser.add_argument("--selector", choices=("mse", "activation_aware", "output_aware"), default="mse")
    parser.add_argument("--calibration-file")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=float)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config["domain"] = "llm"
    ledger = ExperimentLedger(args.experiment_id, "phase_b_downstream", config)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model, snapshot, revision = load_model(args.model)
        calibration_by_module = None
        calibration_metadata = None
        if args.calibration_file:
            calibration_by_module, calibration_payload = load_calibration_by_module(args.calibration_file)
            calibration_metadata = calibration_payload.get("metadata", {})
        preparation = ModelPreparation()
        prepare_permutations(
            model,
            args.permutation,
            preparation,
            calibration_by_module=calibration_by_module,
        )
        rotation_map = json.loads(Path(args.rotation_map).read_text()) if args.rotation_map else None
        apply_rotations(model, args.rotation, preparation, per_module_rotation=rotation_map)
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        if args.selector in {"activation_aware", "output_aware"} and calibration_by_module is None:
            raise ValueError("output-aware selection requires --calibration-file")
        if args.selector in {"activation_aware", "output_aware"}:
            if calibration_metadata.get("permutation", "none") != args.permutation or calibration_metadata.get("rotation", "identity") != rotation_label:
                raise ValueError("downstream output-aware calibration provenance does not match permutation/rotation")
        quantize_model_weights(
            model,
            args.weight_mode,
            preparation,
            selector=args.selector,
            calibration_by_module=calibration_by_module,
        )
        activation_hooks = ActivationHooks(model, args.activation_mode, preparation)
        harness_model = HFLM(
            pretrained=model,
            tokenizer=snapshot,
            backend="causal",
            device="cuda:0",
            dtype=torch.bfloat16,
            batch_size=args.batch_size,
            trust_remote_code=True,
        )
        result = simple_evaluate(
            model=harness_model,
            tasks=list(TASKS),
            num_fewshot=0,
            batch_size=args.batch_size,
            device="cuda:0",
            limit=args.limit,
            bootstrap_iters=1000,
            log_samples=False,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
        clean = jsonable(result)
        atomic_json(ledger.directory / "lm_eval_results.json", clean)
        atomic_json(ledger.directory / "calibration_metadata.json", calibration_metadata)
        atomic_json(ledger.directory / "equivalence_checks.json", preparation.equivalence_checks)
        atomic_json(ledger.directory / "permutations.json", preparation.permutation_records)
        pd.DataFrame(preparation.layer_metrics).to_csv(ledger.directory / "per_layer_metrics.csv", index=False)
        pd.DataFrame(preparation.region_metrics).to_csv(ledger.directory / "format_region_metrics.csv", index=False)
        pd.DataFrame(activation_hooks.metrics).to_csv(ledger.directory / "activation_metrics.csv", index=False)
        metrics = clean.get("results", {})
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_downstream",
            "domain": "llm",
            "status": "completed",
            "model": args.model,
            "model_revision": revision,
            "model_snapshot": snapshot,
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "selector": args.selector,
            "rotation": rotation_label,
            "permutation": args.permutation,
            "tasks": list(TASKS),
            "limit": args.limit,
            "task_metrics": metrics,
            "gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        atomic_json(ledger.directory / "raw_metrics.json", clean)
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
