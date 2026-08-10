#!/usr/bin/env python3
"""Record a non-ledger failure without losing its command or diagnostic text."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import FAILED, MANIFEST, append_jsonl, atomic_json, code_fingerprint, timestamp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--failure-class",
        required=True,
        choices=(
            "code_bug",
            "OOM",
            "model_access",
            "dataset_or_network",
            "dependency",
            "numerical_issue",
            "GPU_became_occupied",
            "unsupported_semantics",
            "runtime_excessive",
            "unknown",
        ),
    )
    parser.add_argument("--command", required=True)
    parser.add_argument("--error", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--domain", default="infrastructure")
    parser.add_argument("--exit-code", type=int, default=1)
    parser.add_argument("--historical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = timestamp()
    attempt_id = f"{args.experiment_id}__{now.replace(':', '').replace('+', '_')}"
    directory = ROOT / "artifacts" / "00_environment" / "failures" / args.experiment_id / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    command = shlex.split(args.command)
    config = {
        "domain": args.domain,
        "historical_record_reconstructed_from_observed_terminal_output": args.historical,
    }
    record = {
        "attempt_id": attempt_id,
        "experiment_id": args.experiment_id,
        "phase": args.phase,
        "domain": args.domain,
        "status": "failed",
        "start_time": now,
        "end_time": now,
        "exit_code": args.exit_code,
        "failure_class": args.failure_class,
        "error_type": args.error.split(":", 1)[0],
        "error": args.error,
        "command": command,
        "command_shell": args.command,
        "artifact_dir": str(directory.relative_to(ROOT)),
        "config_path": str((directory / "config.json").relative_to(ROOT)),
        "stdout_stderr_path": str((directory / "stdout_stderr.log").relative_to(ROOT)),
        "code_fingerprint_sha256": code_fingerprint(),
        "historical_record_reconstructed_from_observed_terminal_output": args.historical,
    }
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "status.json", record)
    (directory / "command.txt").write_text(args.command + "\n")
    (directory / "stdout_stderr.log").write_text(args.log.rstrip() + "\n")
    append_jsonl(MANIFEST, record)
    append_jsonl(FAILED, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
