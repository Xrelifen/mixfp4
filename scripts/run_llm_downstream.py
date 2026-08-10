#!/usr/bin/env python3
"""Evaluate a selected LLM endpoint on the requested downstream task suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstreams" / "NVFP4-RaZeR"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight-mode", required=True)
    parser.add_argument("--activation-mode", default="high_precision")
    parser.add_argument("--selector", choices=("mse", "output_aware", "activation_aware"), default="mse")
    parser.add_argument("--calibration-file")
    parser.add_argument("--calibration-size", type=int, default=0)
    parser.add_argument("--permutation", default="none")
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "hellaswag", "piqa", "winogrande"])
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=float)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def metric_for_task(value: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in ("acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,none"):
        if key in value:
            return key, float(value[key])
    for key, item in value.items():
        if not key.endswith("_stderr,none") and isinstance(item, (float, int)):
            return key, float(item)
    return None, None


def main() -> int:
    args = parse_args()
    config = vars(args).copy()
    config["domain"] = "llm"
    ledger = ExperimentLedger(args.experiment_id, "phase_b_combined_downstream", config)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model, snapshot, revision = load_model(args.model)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
        calibration = None
        calibration_metadata = None
        if args.calibration_file:
            calibration, payload = load_calibration_by_module(resolve(args.calibration_file))
            calibration_metadata = payload.get("metadata", {})
            observed = payload.get("calibration_sequences")
            if args.calibration_size and observed != args.calibration_size:
                raise ValueError("calibration size mismatch")
        if args.selector != "mse" and calibration is None:
            raise ValueError("output-aware downstream endpoint requires calibration")
        preparation = ModelPreparation()
        prepare_permutations(model, args.permutation, preparation, calibration_by_module=calibration)
        rotation_map = json.loads(resolve(args.rotation_map).read_text(encoding="utf-8")) if args.rotation_map else None
        apply_rotations(model, args.rotation, preparation, per_module_rotation=rotation_map)
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        if args.selector != "mse":
            if calibration_metadata.get("permutation", "none") != args.permutation:
                raise ValueError("calibration permutation provenance mismatch")
            if calibration_metadata.get("rotation", "identity") != rotation_label:
                raise ValueError("calibration rotation provenance mismatch")
        quantize_model_weights(
            model,
            args.weight_mode,
            preparation,
            selector=args.selector,
            calibration_by_module=calibration,
        )
        ActivationHooks(model, args.activation_mode, preparation)

        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager

        evaluator_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device="cuda:0",
        )
        result = lm_eval.simple_evaluate(
            model=evaluator_model,
            tasks=args.tasks,
            num_fewshot=0,
            batch_size=args.batch_size,
            limit=args.limit,
            task_manager=TaskManager(),
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
        atomic_json(ledger.directory / "lm_eval_results.json", result)
        task_rows: list[dict[str, Any]] = []
        for task, values in result.get("results", {}).items():
            metric_name, metric = metric_for_task(values)
            task_rows.append(
                {
                    "experiment_id": args.experiment_id,
                    "model": args.model,
                    "task": task,
                    "metric_name": metric_name,
                    "metric": metric,
                    "num_fewshot": 0,
                    "limit": args.limit,
                }
            )
        pd.DataFrame(task_rows).to_csv(ledger.directory / "downstream_metrics.csv", index=False)
        available = [row["metric"] for row in task_rows if row["metric"] is not None]
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_combined_downstream",
            "domain": "llm",
            "status": "completed",
            "model": args.model,
            "model_revision": revision,
            "model_snapshot": snapshot,
            "dataset_or_promptset": "+".join(args.tasks),
            "quantization_mode": f"W={args.weight_mode}/A={args.activation_mode}",
            "weight_format_mode": args.weight_mode,
            "activation_format_mode": args.activation_mode,
            "selector": args.selector,
            "permutation": args.permutation,
            "rotation": rotation_label,
            "calibration_size": args.calibration_size,
            "tasks": args.tasks,
            "downstream_mean_accuracy": sum(available) / len(available) if available else None,
            "downstream_task_metrics": {row["task"]: row["metric"] for row in task_rows},
            "num_fewshot": 0,
            "evaluation_limit": args.limit,
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
