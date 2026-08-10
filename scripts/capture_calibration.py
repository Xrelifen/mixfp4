#!/usr/bin/env python3
"""Capture cumulative 32/128/256-sequence activation covariances on one guarded GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402
from project_quant.calibration import CovarianceCollector, load_calibration_by_module, save_covariances  # noqa: E402
from project_quant.data import load_sequences, prepare_calibration  # noqa: E402
from project_quant.modeling import ModelPreparation, apply_rotations, load_model, prepare_permutations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rotation", default="identity")
    parser.add_argument("--rotation-map")
    parser.add_argument("--permutation", default="none")
    parser.add_argument(
        "--packing-calibration-file",
        help="identity-layout calibration used only to choose a sensitivity-weighted permutation",
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("calibration checkpoints must be positive")
    config = vars(args).copy()
    config["domain"] = "llm"
    ledger = ExperimentLedger(args.experiment_id, "phase_b_calibration", config)
    collector: CovarianceCollector | None = None
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        record = prepare_calibration(args.model, seq_len=args.seq_len, count=max(checkpoints), seed=args.seed)
        sequences = load_sequences(record)
        calibration_samples = []
        source_entries = record.get("entries", [])
        for index, sequence in enumerate(sequences[: max(checkpoints)]):
            source = source_entries[index] if index < len(source_entries) else {}
            calibration_samples.append(
                {
                    "calibration_sample_id": (
                        f"c4_index_{source.get('dataset_index', 'unknown')}_"
                        f"offset_{source.get('token_offset', 'unknown')}"
                    ),
                    "calibration_index": index,
                    "dataset_index": source.get("dataset_index"),
                    "token_offset": source.get("token_offset"),
                    "source_token_count": source.get("source_token_count"),
                    "token_ids_sha256": hashlib.sha256(sequence.tobytes()).hexdigest(),
                    "sequence_length": int(sequence.size),
                }
            )
        atomic_json(ledger.directory / "calibration_sample_manifest.json", calibration_samples)
        model, snapshot, revision = load_model(args.model)
        preparation = ModelPreparation()
        packing_calibration = None
        packing_metadata = None
        if args.packing_calibration_file:
            packing_calibration, packing_payload = load_calibration_by_module(args.packing_calibration_file)
            packing_metadata = packing_payload.get("metadata", {})
        prepare_permutations(
            model,
            args.permutation,
            preparation,
            calibration_by_module=packing_calibration,
        )
        rotation_map = json.loads(Path(args.rotation_map).read_text()) if args.rotation_map else None
        apply_rotations(model, args.rotation, preparation, per_module_rotation=rotation_map)
        rotation_label = args.rotation if not args.rotation_map else f"per_module:{Path(args.rotation_map).stem}"
        collector = CovarianceCollector(model, preparation)
        core = getattr(model, "model", model)
        files: dict[str, dict[str, object]] = {}
        with torch.inference_mode():
            for index, array in enumerate(sequences[: max(checkpoints)]):
                collector.begin_sequence()
                input_ids = torch.from_numpy(array.astype("int64", copy=False)).unsqueeze(0).to("cuda:0")
                output = core(input_ids=input_ids, use_cache=False)
                del output, input_ids
                completed = index + 1
                print(f"calibration sequence {completed}/{max(checkpoints)}", flush=True)
                if completed in checkpoints:
                    metadata = {
                        "model": args.model,
                        "model_revision": revision,
                        "model_snapshot": snapshot,
                        "rotation": rotation_label,
                        "rotation_map": args.rotation_map,
                        "permutation": args.permutation,
                        "packing_calibration_file": args.packing_calibration_file,
                        "packing_calibration_metadata": packing_metadata,
                        "dataset_manifest_key": record["key"],
                        "dataset_sha256": record["data_sha256"],
                        "calibration_seed": args.seed,
                        "sequence_length": args.seq_len,
                        "calibration_sample_ids": [
                            row["calibration_sample_id"]
                            for row in calibration_samples[:completed]
                        ],
                    }
                    payload = collector.payload(calibration_sequences=completed, metadata=metadata)
                    path = ledger.directory / f"calibration_{completed}.pt"
                    digest = save_covariances(path, payload)
                    files[str(completed)] = {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": digest,
                        "num_canonical_inputs": len(payload["hessian_by_canonical"]),
                        "num_modules": len(payload["module_to_canonical"]),
                    }
                    del payload
        collector.close()
        collector = None
        atomic_json(ledger.directory / "calibration_files.json", files)
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_calibration",
            "domain": "llm",
            "status": "completed",
            "model": args.model,
            "model_revision": revision,
            "model_snapshot": snapshot,
            "rotation": rotation_label,
            "rotation_map": args.rotation_map,
            "permutation": args.permutation,
            "calibration_sizes": checkpoints,
            "dataset_manifest_key": record["key"],
            "dataset_manifest_sha256": record["data_sha256"],
            "dataset_or_promptset": record["key"],
            "calibration_sample_manifest": str(
                (ledger.directory / "calibration_sample_manifest.json").relative_to(ROOT)
            ),
            "exact_samples_saved": True,
            "gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
            "files": files,
        }
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        ledger.complete(summary)
        pointer_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            f"{args.model}_{args.permutation}_{rotation_label}".lower(),
        ).strip("_")
        atomic_json(
            ROOT / "artifacts" / "04_phase_b" / "selector" / "calibration" / f"{pointer_slug}_current.json",
            {
                "experiment_id": args.experiment_id,
                "attempt_id": ledger.attempt_id,
                "artifact_dir": str(ledger.directory.relative_to(ROOT)),
                "model": args.model,
                "model_revision": revision,
                "permutation": args.permutation,
                "rotation": rotation_label,
                "files": files,
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if collector is not None:
            collector.close()
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
