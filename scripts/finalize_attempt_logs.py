#!/usr/bin/env python3
"""Materialize experiment-local stdout/stderr logs from immutable launcher logs.

Run only after GPU queues have stopped.  The guard writes each launch to a
unique attempt log; ledgers initially use a symlink because the child creates
its attempt directory after logging starts.  This finalizer replaces only that
symlink with a byte-for-byte local copy and never removes the guard-owned log.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "06_final"
MANIFEST = FINAL / "experiment_manifest.jsonl"


def records() -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        return []
    values = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def main() -> int:
    report: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records():
        attempt_id = str(row.get("attempt_id", ""))
        artifact_value = row.get("artifact_dir")
        if not attempt_id or not artifact_value or attempt_id in seen:
            continue
        seen.add(attempt_id)
        directory = ROOT / str(artifact_value)
        destination = directory / "stdout_stderr.log"
        launcher_value = row.get("launcher_log")
        source = Path(str(launcher_value)) if launcher_value else None
        if source is not None and not source.is_absolute():
            source = ROOT / source
        item = {
            "attempt_id": attempt_id,
            "experiment_id": row.get("experiment_id"),
            "artifact_dir": str(artifact_value),
            "launcher_log": str(source) if source else None,
        }
        if destination.is_symlink():
            resolved = destination.resolve(strict=False)
            if not resolved.exists():
                item["status"] = "missing_symlink_target"
            else:
                temporary = destination.with_suffix(".log.materializing")
                shutil.copy2(resolved, temporary)
                destination.unlink()
                os.replace(temporary, destination)
                item["status"] = "materialized_symlink"
                item["bytes"] = destination.stat().st_size
        elif destination.exists():
            item["status"] = "already_materialized"
            item["bytes"] = destination.stat().st_size
        elif source is not None and source.exists():
            shutil.copy2(source, destination)
            item["status"] = "materialized_from_manifest"
            item["bytes"] = destination.stat().st_size
        else:
            item["status"] = "missing_source"
        report.append(item)
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "attempt_log_materialization.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    missing = [row for row in report if row["status"].startswith("missing")]
    print(json.dumps({"attempts": len(report), "missing": len(missing)}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
