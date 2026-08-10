"""Atomic provenance and failure ledgers shared by every experiment."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FINAL = ARTIFACTS / "06_final"
MANIFEST = FINAL / "experiment_manifest.jsonl"
FAILED = FINAL / "failed_runs.jsonl"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"ERROR({result.returncode}): {result.stdout.strip()}"


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for base in (ROOT / "project_quant", ROOT / "scripts", ROOT / "tests"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    physical = os.environ.get("MIXFP4_PHYSICAL_GPU")
    gpu_name = None
    gpu_uuid = None
    if physical is not None:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical}",
                "--query-gpu=name,uuid",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode == 0 and "," in result.stdout:
            gpu_name, gpu_uuid = (part.strip() for part in result.stdout.split(",", 1))
    return {
        "workspace_commit": git(ROOT, "rev-parse", "HEAD"),
        "workspace_dirty": bool(git(ROOT, "status", "--porcelain=v1")),
        "nvfp4_razer_commit": git(ROOT / "upstreams" / "NVFP4-RaZeR", "rev-parse", "HEAD"),
        "nvfp4_razer_dirty": bool(git(ROOT / "upstreams" / "NVFP4-RaZeR", "status", "--porcelain=v1")),
        "fouroversix_commit": git(ROOT / "upstreams" / "fouroversix", "rev-parse", "HEAD"),
        "fouroversix_dirty": bool(git(ROOT / "upstreams" / "fouroversix", "status", "--porcelain=v1")),
        "deepcompressor_commit": git(ROOT / "upstreams" / "deepcompressor", "rev-parse", "HEAD"),
        "sana_commit": git(ROOT / "upstreams" / "Sana", "rev-parse", "HEAD"),
        "code_fingerprint_sha256": code_fingerprint(),
        "python": sys.version,
        "physical_gpu_index": int(physical) if physical is not None else None,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_index": int(os.environ.get("MIXFP4_LOGICAL_GPU", "0")) if physical is not None else None,
    }


def classify_failure(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    if "out of memory" in message or "cuda oom" in message:
        return "OOM"
    if "gated repo" in message or "401" in message or "403" in message or "not authorized" in message:
        return "model_access"
    if "dataset" in message or "connection" in message or "offline" in message:
        return "dataset_or_network"
    if "no module named" in message or "dependency" in message or "version" in message:
        return "dependency"
    if "nan" in message or "inf" in message or "numerical" in message:
        return "numerical_issue"
    if "occupied" in message or "compliant gpu" in message:
        return "GPU_became_occupied"
    if "unsupported" in message or "not implemented" in message:
        return "unsupported_semantics"
    if "timeout" in message or "runtime excessive" in message:
        return "runtime_excessive"
    if isinstance(error, (AssertionError, ValueError, RuntimeError, TypeError)):
        return "code_bug"
    return "unknown"


class ExperimentLedger:
    """Create immutable per-attempt files and append global status records."""

    def __init__(self, experiment_id: str, phase: str, config: dict[str, Any]) -> None:
        self.experiment_id = experiment_id
        self.phase = phase
        self.config = config
        domain = str(config.get("domain", "llm"))
        if phase.startswith(("phase0", "phase_a")):
            base = ARTIFACTS / "03_phase_a" / domain / "raw"
        elif phase.startswith("phase_b"):
            category = next(
                (name for name in ("selector", "permutation", "rotation", "timestep", "combined") if name in phase),
                "combined",
            )
            base = ARTIFACTS / "04_phase_b" / category / "raw"
        else:
            base = ARTIFACTS / "05_cross_gpu" / "raw"
        self.start_time = timestamp()
        self.attempt_id = f"{experiment_id}__{self.start_time.replace(':', '').replace('+', '_')}"
        self.directory = base / experiment_id / self.attempt_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.record = {
            "attempt_id": self.attempt_id,
            "experiment_id": experiment_id,
            "phase": phase,
            "status": "running",
            "start_time": self.start_time,
            "command": sys.argv,
            "command_shell": os.environ.get("MIXFP4_CHILD_COMMAND_SHELL", shlex.join(sys.argv)),
            "python_argv_shell": shlex.join(sys.argv),
            "config_path": str((self.directory / "config.json").relative_to(ROOT)),
            "artifact_dir": str(self.directory.relative_to(ROOT)),
            "launcher_log": os.environ.get("MIXFP4_LAUNCH_LOG"),
            **provenance(),
        }
        atomic_json(self.directory / "config.json", config)
        atomic_json(self.directory / "status.json", self.record)
        (self.directory / "command.txt").write_text(self.record["command_shell"] + "\n")
        # The GPU guard owns the combined process log and starts writing it
        # before this child process creates its experiment directory.  Keep an
        # immutable, experiment-local pointer to that exact stdout/stderr file
        # so measurements are never only recoverable from terminal output.
        launcher_log = self.record.get("launcher_log")
        if launcher_log:
            target = Path(str(launcher_log)).resolve()
            link = self.directory / "stdout_stderr.log"
            try:
                link.symlink_to(target)
            except FileExistsError:
                pass
        append_jsonl(MANIFEST, self.record)

    def complete(self, summary: dict[str, Any]) -> None:
        end = timestamp()
        summary_path = str((self.directory / "summary.json").relative_to(ROOT))
        summary_row = self.directory / "summary_row.csv"
        self.record.update(
            {
                "status": "completed",
                "exit_code": 0,
                "end_time": end,
                "summary_path": summary_path,
                "summary_row_path": (
                    str(summary_row.relative_to(ROOT)) if summary_row.exists() else None
                ),
                "model": summary.get("model"),
                "model_revision": summary.get("model_revision"),
                "dataset_or_promptset": summary.get("dataset_or_promptset")
                or summary.get("dataset")
                or summary.get("prompt_file"),
                "dataset_manifest_key": summary.get("dataset_manifest_key"),
                "dataset_manifest_sha256": summary.get("dataset_manifest_sha256"),
                "prompt_manifest": summary.get("prompt_manifest")
                or summary.get("prompt_file"),
                "physical_gpu_index": summary.get(
                    "physical_gpu_index", self.record.get("physical_gpu_index")
                ),
                "logical_gpu_index": summary.get(
                    "logical_gpu_index", self.record.get("logical_gpu_index")
                ),
                "gpu_type": summary.get("gpu_type", self.record.get("gpu_name")),
                "gpu_uuid": summary.get("gpu_uuid", self.record.get("gpu_uuid")),
            }
        )
        atomic_json(self.directory / "summary.json", summary)
        atomic_json(self.directory / "status.json", self.record)
        append_jsonl(MANIFEST, self.record)

    def fail(self, error: BaseException) -> None:
        end = timestamp()
        trace = traceback.format_exc()
        (self.directory / "stderr.log").write_text(trace)
        failure = {
            **self.record,
            "status": "failed",
            "end_time": end,
            "exit_code": 1,
            "failure_class": classify_failure(error),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_path": str((self.directory / "stderr.log").relative_to(ROOT)),
        }
        self.record = failure
        atomic_json(self.directory / "status.json", failure)
        append_jsonl(MANIFEST, failure)
        append_jsonl(FAILED, failure)
