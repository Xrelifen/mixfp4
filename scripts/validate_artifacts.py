#!/usr/bin/env python3
"""Validate the mandatory provenance contract for latest experiment attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
MANIFEST = FINAL / "experiment_manifest.jsonl"


def rows() -> list[dict[str, Any]]:
    values = []
    if not MANIFEST.exists():
        return values
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def latest_attempt_states() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows():
        attempt = row.get("attempt_id")
        if not attempt:
            continue
        current = latest.get(str(attempt))
        if current is None or str(row.get("end_time", "")) >= str(current.get("end_time", "")):
            latest[str(attempt)] = row
    return sorted(latest.values(), key=lambda row: str(row.get("start_time", "")))


def main() -> int:
    findings: list[dict[str, Any]] = []
    for row in latest_attempt_states():
        status = str(row.get("status"))
        directory_value = row.get("artifact_dir")
        if not directory_value:
            findings.append({"attempt_id": row.get("attempt_id"), "issue": "missing_artifact_dir"})
            continue
        directory = ROOT / str(directory_value)
        required = ["config.json", "command.txt", "status.json", "stdout_stderr.log"]
        if status == "completed":
            required.extend(("raw_metrics.json", "summary.json", "summary_row.csv"))
        for name in required:
            path = directory / name
            if not path.exists():
                findings.append(
                    {
                        "attempt_id": row.get("attempt_id"),
                        "experiment_id": row.get("experiment_id"),
                        "status": status,
                        "issue": f"missing_file:{name}",
                        "artifact_dir": str(directory_value),
                    }
                )
        for field in ("start_time", "code_fingerprint_sha256", "command_shell"):
            if row.get(field) in (None, ""):
                findings.append(
                    {
                        "attempt_id": row.get("attempt_id"),
                        "experiment_id": row.get("experiment_id"),
                        "status": status,
                        "issue": f"missing_manifest_field:{field}",
                        "artifact_dir": str(directory_value),
                    }
                )
        if status in {"completed", "failed"} and row.get("end_time") in (None, ""):
            findings.append(
                {
                    "attempt_id": row.get("attempt_id"),
                    "experiment_id": row.get("experiment_id"),
                    "status": status,
                    "issue": "missing_manifest_field:end_time",
                    "artifact_dir": str(directory_value),
                }
            )
        if status == "completed":
            summary_path = directory / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for field in ("physical_gpu_index", "logical_gpu_index", "gpu_uuid", "gpu_type"):
                    if summary.get(field) in (None, ""):
                        findings.append(
                            {
                                "attempt_id": row.get("attempt_id"),
                                "experiment_id": row.get("experiment_id"),
                                "status": status,
                                "issue": f"missing_summary_gpu_field:{field}",
                                "artifact_dir": str(directory_value),
                            }
                        )
                if summary.get("model_revision") in (None, "") and summary.get("phase") not in {
                    "phase0_tests"
                }:
                    findings.append(
                        {
                            "attempt_id": row.get("attempt_id"),
                            "experiment_id": row.get("experiment_id"),
                            "status": status,
                            "issue": "missing_summary_field:model_revision",
                            "artifact_dir": str(directory_value),
                        }
                    )
                if not any(
                    summary.get(field) not in (None, "")
                    for field in (
                        "dataset_or_promptset", "dataset", "prompt_file",
                        "calibration_file", "calibration_sample_manifest",
                    )
                ):
                    findings.append(
                        {
                            "attempt_id": row.get("attempt_id"),
                            "experiment_id": row.get("experiment_id"),
                            "status": status,
                            "issue": "missing_dataset_prompt_or_calibration_provenance",
                            "artifact_dir": str(directory_value),
                        }
                    )
    frame = pd.DataFrame(findings)
    FINAL.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FINAL / "artifact_validation_issues.csv", index=False)
    report = {
        "latest_attempts_checked": len(latest_attempt_states()),
        "issues": len(findings),
        "passed": not findings,
        "issues_csv": "artifacts/06_final/artifact_validation_issues.csv",
    }
    (FINAL / "artifact_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
