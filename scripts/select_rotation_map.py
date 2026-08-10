#!/usr/bin/env python3
"""Select per-module Phase-B transforms using calibration data only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import ExperimentLedger, atomic_json  # noqa: E402
from project_quant.core import GranularityResult, quant_mixfp4_granularity  # noqa: E402
from project_quant.modeling import load_model, named_linears  # noqa: E402
from project_quant.rotation import (  # noqa: E402
    TRANSFORM_BANK,
    dense_transform_matrix,
    get_transform,
    normalized_objective,
    verify_rotation_equivalence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--domain", choices=("llm", "diffusion"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 1.0, 10.0])
    parser.add_argument("--transforms", nargs="+", default=[spec.name for spec in TRANSFORM_BANK])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def transformed_hessian64(hessian128: torch.Tensor, transform_name: str, device: torch.device) -> torch.Tensor:
    spec = get_transform(transform_name)
    covariance = hessian128.to(device=device, dtype=torch.float32)
    transform = dense_transform_matrix(spec, 128, device=device, dtype=torch.float32)
    rotated = torch.einsum("ia,kij,jb->kab", transform, covariance, transform)
    return torch.stack((rotated[:, :64, :64], rotated[:, 64:, 64:]), dim=1).reshape(-1, 64, 64)


def slug_lambda(value: float) -> str:
    return (f"{value:g}").replace("-", "m").replace(".", "p")


def main() -> int:
    args = parse_args()
    if len(set(args.transforms)) != len(args.transforms):
        raise ValueError("transform bank contains duplicates")
    for value in args.transforms:
        get_transform(value)
    calibration_path = resolve(args.calibration_file)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if "hessian_k128_by_canonical" not in calibration:
        raise ValueError("rotation selection requires K128 calibration covariances")
    metadata = calibration.get("metadata", {})
    if metadata.get("rotation", "identity") != "identity" or metadata.get("permutation", "none") != "none":
        raise ValueError("rotation bank selection requires identity/no-permutation source calibration")

    config = vars(args).copy()
    config["calibration_file"] = str(calibration_path.relative_to(ROOT))
    ledger = ExperimentLedger(args.experiment_id, "phase_b_rotation_calibration", config)
    try:
        if args.domain == "llm":
            if not args.model:
                raise ValueError("--model is required for LLM rotation selection")
            model, snapshot, revision = load_model(args.model)
            modules = [(name, module.weight.data) for name, module in named_linears(model)]
            model_id = args.model
        else:
            from project_quant.adapters.deepcompressor_diffusion import (
                SANA_MODEL_ID,
                SANA_REVISION,
                load_sana_pipeline,
                matrix_weight,
                named_matrix_modules,
            )

            pipe = load_sana_pipeline(with_vae=False)
            modules = [(name, matrix_weight(module)) for name, module in named_matrix_modules(pipe.transformer)]
            model_id = SANA_MODEL_ID
            revision = SANA_REVISION
            snapshot = None

        mapping = calibration["module_to_canonical"]
        hessians128 = calibration["hessian_k128_by_canonical"]
        missing = sorted(name for name, _ in modules if name not in mapping)
        if missing:
            raise ValueError(f"calibration lacks {len(missing)} modules: {missing[:8]}")

        rows: list[dict[str, Any]] = []
        maps: dict[str, dict[float, dict[str, str]]] = {
            "mse": {value: {} for value in args.lambdas},
            "output_aware": {value: {} for value in args.lambdas},
        }
        for module_index, (name, weight) in enumerate(modules):
            module_rows: list[dict[str, Any]] = []
            h128 = hessians128[mapping[name]]
            for transform_name in args.transforms:
                h64 = transformed_hessian64(h128, transform_name, weight.device)
                result = quant_mixfp4_granularity(
                    weight,
                    format_region="n8k64",
                    operand_role="weight_b",
                    selector="output_aware",
                    calibration_stats={"hessian_k64": h64},
                    transform=get_transform(transform_name),
                    return_stats=True,
                    collect_regions=False,
                )
                assert isinstance(result, GranularityResult)
                check_input = torch.randn(
                    (2, min(weight.shape[1], 4096)),
                    generator=torch.Generator(device="cpu").manual_seed(0),
                    dtype=torch.float32,
                )
                # Use a leading slice only when K is very large; all transform
                # sizes divide 4096 and the exact operation is block-local.
                check = verify_rotation_equivalence(
                    check_input.to(weight.device),
                    weight[:, : check_input.shape[1]],
                    transform_name,
                )
                summary = result.summary
                row = {
                    "experiment_id": args.experiment_id,
                    "domain": args.domain,
                    "model": model_id,
                    "module_name": name,
                    "module_index": module_index,
                    "N": weight.shape[0],
                    "K": weight.shape[1],
                    "transform": transform_name,
                    "deployability": get_transform(transform_name).deployability,
                    "calibration_output_error_mse_selector": summary["calibration_output_sse_mse_selector"],
                    "calibration_output_error_output_selector": summary["calibration_output_sse_selected"],
                    "granularity_regret": summary["granularity_regret"],
                    "sensitivity_regret": summary.get("sensitivity_regret"),
                    "homogeneity": summary["mean_homogeneity"],
                    "margin_conflict": summary["mean_margin_conflict"],
                    "equivalence_relative_l2": check["relative_l2_error"],
                    "equivalence_passed": check["passed"],
                }
                if not check["passed"]:
                    raise RuntimeError(f"rotation equivalence failed for {name}/{transform_name}: {check}")
                module_rows.append(row)
                del result, h64

            mse_output = torch.tensor(
                [row["calibration_output_error_mse_selector"] for row in module_rows], dtype=torch.float64
            )
            out_output = torch.tensor(
                [row["calibration_output_error_output_selector"] for row in module_rows], dtype=torch.float64
            )
            mse_regret = torch.tensor([row["granularity_regret"] for row in module_rows], dtype=torch.float64)
            sensitivity_regret = torch.tensor(
                [row["sensitivity_regret"] for row in module_rows], dtype=torch.float64
            )
            for lam in args.lambdas:
                mse_objective = normalized_objective(mse_output, mse_regret, lam)
                output_objective = normalized_objective(out_output, sensitivity_regret, lam)
                mse_index = int(torch.argmin(mse_objective).item())
                output_index = int(torch.argmin(output_objective).item())
                maps["mse"][lam][name] = module_rows[mse_index]["transform"]
                maps["output_aware"][lam][name] = module_rows[output_index]["transform"]
                for index, row in enumerate(module_rows):
                    row[f"mse_objective_lambda_{lam:g}"] = float(mse_objective[index].item())
                    row[f"output_objective_lambda_{lam:g}"] = float(output_objective[index].item())
                    row[f"mse_selected_lambda_{lam:g}"] = index == mse_index
                    row[f"output_selected_lambda_{lam:g}"] = index == output_index
            rows.extend(module_rows)
            print(f"rotation selection module {module_index + 1}/{len(modules)} {name}", flush=True)

        pd.DataFrame(rows).to_csv(ledger.directory / "rotation_candidate_metrics.csv", index=False)
        map_records: dict[str, str] = {}
        map_dir = ROOT / "artifacts" / "04_phase_b" / "rotation" / "maps"
        for selector_name, by_lambda in maps.items():
            for lam, value in by_lambda.items():
                path = map_dir / f"{args.experiment_id}_{selector_name}_lambda_{slug_lambda(lam)}.json"
                atomic_json(path, value)
                map_records[f"{selector_name}_lambda_{lam:g}"] = str(path.relative_to(ROOT))
        atomic_json(ledger.directory / "rotation_maps.json", map_records)
        atomic_json(ledger.directory / "calibration_metadata.json", metadata)
        summary = {
            "experiment_id": args.experiment_id,
            "phase": "phase_b_rotation_calibration",
            "domain": args.domain,
            "status": "completed",
            "model": model_id,
            "model_revision": revision,
            "model_snapshot": snapshot,
            "calibration_file": str(calibration_path.relative_to(ROOT)),
            "calibration_size": calibration.get("calibration_sequences", calibration.get("calibration_samples")),
            "transforms": args.transforms,
            "lambdas": args.lambdas,
            "num_modules": len(modules),
            "map_files": map_records,
            "all_equivalence_checks_passed": all(row["equivalence_passed"] for row in rows),
            "physical_gpu_index": int(os.environ["MIXFP4_PHYSICAL_GPU"]),
            "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")),
            "gpu_uuid": os.environ.get("MIXFP4_GPU_UUID"),
            "gpu_type": torch.cuda.get_device_name(0),
        }
        pd.DataFrame([summary]).to_csv(ledger.directory / "summary_row.csv", index=False)
        atomic_json(ledger.directory / "raw_metrics.json", summary)
        ledger.complete(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        ledger.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
