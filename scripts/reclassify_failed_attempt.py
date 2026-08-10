#!/usr/bin/env python3
"""Append an auditable correction for a previously classified failed attempt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import FAILED, MANIFEST, append_jsonl, atomic_json, timestamp  # noqa: E402


CLASSES = (
    "code_bug", "OOM", "model_access", "dataset_or_network", "dependency",
    "numerical_issue", "GPU_became_occupied", "unsupported_semantics",
    "runtime_excessive", "unknown",
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--failure-class", required=True, choices=CLASSES)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    matches = [row for row in rows(FAILED) if row.get("attempt_id") == args.attempt_id]
    if not matches:
        raise ValueError(f"failed attempt not found: {args.attempt_id}")
    original = matches[-1]
    corrected = {
        **original,
        "failure_class": args.failure_class,
        "original_failure_class": original.get("failure_class"),
        "failure_reclassified": True,
        "failure_reclassification_reason": args.reason,
        "reclassified_at": timestamp(),
    }
    artifact_dir = ROOT / corrected["artifact_dir"]
    atomic_json(
        artifact_dir / "failure_reclassification.json",
        {
            "attempt_id": args.attempt_id,
            "original_failure_class": original.get("failure_class"),
            "corrected_failure_class": args.failure_class,
            "reason": args.reason,
            "reclassified_at": corrected["reclassified_at"],
        },
    )
    atomic_json(artifact_dir / "status.json", corrected)
    append_jsonl(MANIFEST, corrected)
    append_jsonl(FAILED, corrected)
    print(json.dumps(corrected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
