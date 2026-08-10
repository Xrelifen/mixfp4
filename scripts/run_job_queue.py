#!/usr/bin/env python3
"""Resume a JSON job queue while every CUDA child is launched by gpu_guard.

The controller itself never imports CUDA.  Exit 75 means the guard refused a
launch because the shared-machine policy changed; that job stays pending and
is retried after a bounded polling interval.  A child failure is preserved by
the experiment ledger and is not silently retried.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "06_final" / "experiment_manifest.jsonl"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload["jobs"] if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise TypeError("job file must contain a list or {'jobs': [...]} object")
    identifiers: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("experiment_id"), str):
            raise TypeError("every job requires a string experiment_id")
        if job["experiment_id"] in identifiers:
            raise ValueError(f"duplicate experiment_id {job['experiment_id']!r}")
        identifiers.add(job["experiment_id"])
        if not isinstance(job.get("command"), list) or not all(
            isinstance(value, str) for value in job["command"]
        ):
            raise TypeError(f"job {job['experiment_id']} requires a string-list command")
        if job.get("gpu_kind", "any") not in {"any", "a6000", "ada"}:
            raise ValueError(f"job {job['experiment_id']} has invalid gpu_kind")
    return jobs


def latest_statuses() -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not MANIFEST.exists():
        return statuses
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("experiment_id") and row.get("status"):
            statuses[str(row["experiment_id"])] = str(row["status"])
    return statuses


def conflicting_schedulers(worktree: Path | None) -> list[dict[str, Any]]:
    if worktree is None:
        return []
    expected = str(worktree.resolve())
    owner_uid = os.getuid()
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != owner_uid:
                continue
            cwd = os.readlink(entry / "cwd")
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if cwd == expected and ("matrix.py" in command or "run_job_queue.py" in command):
            matches.append({"pid": int(entry.name), "cwd": cwd, "command": command.strip()})
    return sorted(matches, key=lambda row: row["pid"])


def guard_command(job: dict[str, Any], guard_log: Path) -> list[str]:
    result = [
        "python3",
        "scripts/gpu_guard.py",
        "--experiment-id",
        job["experiment_id"],
        "--gpu-kind",
        job.get("gpu_kind", "any"),
        "--log-file",
        str(guard_log.relative_to(ROOT)),
    ]
    if job.get("gpu_index") is not None:
        result.extend(("--gpu-index", str(job["gpu_index"])))
    result.append("--")
    result.extend(job["command"])
    return result


def command_option(command: list[str], option: str, default: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return default


def classify_nonledger_failure(log_text: str) -> str:
    value = log_text.lower()
    if "out of memory" in value or "cuda oom" in value:
        return "OOM"
    if "401" in value or "403" in value or "gated repo" in value:
        return "model_access"
    if "no module named" in value or "dependency" in value:
        return "dependency"
    if "dataset" in value and ("offline" in value or "connection" in value):
        return "dataset_or_network"
    if "unsupported" in value or "not implemented" in value:
        return "unsupported_semantics"
    return "unknown"


def preserve_nonledger_failure(
    job: dict[str, Any], *, exit_code: int, output_path: Path
) -> None:
    """Cover SIGKILL/interpreter failures that occur before a ledger can fail."""
    if latest_statuses().get(job["experiment_id"]) == "failed":
        return
    text = output_path.read_text(encoding="utf-8", errors="replace")
    command = job["command"]
    phase = command_option(command, "--phase", "queue_child_before_ledger")
    domain = "diffusion" if any("sana" in value.lower() for value in command) else "llm"
    record_command = [
        "python3",
        "scripts/record_failure.py",
        "--experiment-id",
        job["experiment_id"],
        "--phase",
        phase,
        "--failure-class",
        classify_nonledger_failure(text),
        "--command",
        shlex.join(command),
        "--error",
        f"guarded child exited {exit_code} before writing a failed ledger record",
        "--log",
        text[-80_000:],
        "--domain",
        domain,
        "--exit-code",
        str(exit_code),
    ]
    subprocess.run(record_command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_file", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--queue-id")
    parser.add_argument(
        "--conflicting-worktree",
        type=Path,
        default=Path("/home/JaaaaaA_l/mixfp4"),
        help="wait while a sibling worktree has a live GPU matrix scheduler",
    )
    parser.add_argument("--external-quiet-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers must be 1, 2, or 3")
    if not 2 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 2 and 60")
    if not 0 <= args.external_quiet_seconds <= 600:
        parser.error("--external-quiet-seconds must be between 0 and 600")
    source = args.job_file if args.job_file.is_absolute() else ROOT / args.job_file
    jobs = read_jobs(source)
    queue_id = args.queue_id or source.stem
    queue_dir = ROOT / "artifacts" / "00_environment" / "queues" / queue_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    event_log = queue_dir / "events.jsonl"
    guard_output_dir = queue_dir / "guard_output"
    guard_output_dir.mkdir(exist_ok=True)

    completed = {key for key, value in latest_statuses().items() if value == "completed"}
    pending = [job for job in jobs if job["experiment_id"] not in completed]
    running: dict[int, tuple[dict[str, Any], subprocess.Popen[str], Any, Path]] = {}
    failed: list[str] = []
    append_jsonl(
        event_log,
        {
            "timestamp": timestamp(),
            "event": "queue_started",
            "queue_id": queue_id,
            "job_file": str(source.relative_to(ROOT)),
            "total_jobs": len(jobs),
            "skipped_completed": len(jobs) - len(pending),
            "workers": args.workers,
        },
    )

    blocked_since: float | None = None
    next_launch_time = 0.0
    external_quiet_since: float | None = None
    last_conflict_signature: tuple[int, ...] | None = None
    while pending or running:
        for pid, (job, process, output_handle, output_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            output_handle.close()
            del running[pid]
            event = {
                "timestamp": timestamp(),
                "event": "guard_process_finished",
                "experiment_id": job["experiment_id"],
                "exit_code": code,
                "guard_output": str(output_path.relative_to(ROOT)),
            }
            append_jsonl(event_log, event)
            if code == 75:
                pending.insert(0, job)
                blocked_since = blocked_since or time.monotonic()
                next_launch_time = max(next_launch_time, time.monotonic() + args.poll_seconds)
            elif code != 0:
                preserve_nonledger_failure(job, exit_code=code, output_path=output_path)
                failed.append(job["experiment_id"])
                blocked_since = None
            else:
                blocked_since = None

        conflicts = conflicting_schedulers(args.conflicting_worktree)
        signature = tuple(row["pid"] for row in conflicts)
        if conflicts:
            external_quiet_since = None
            if signature != last_conflict_signature:
                append_jsonl(
                    event_log,
                    {
                        "timestamp": timestamp(),
                        "event": "waiting_for_conflicting_scheduler",
                        "conflicting_worktree": str(args.conflicting_worktree),
                        "processes": conflicts,
                    },
                )
        elif external_quiet_since is None:
            external_quiet_since = time.monotonic()
            append_jsonl(
                event_log,
                {
                    "timestamp": timestamp(),
                    "event": "external_quiet_interval_started",
                    "required_seconds": args.external_quiet_seconds,
                },
            )
        last_conflict_signature = signature
        external_stable = (
            external_quiet_since is not None
            and time.monotonic() - external_quiet_since >= args.external_quiet_seconds
        )

        launched = False
        while (
            pending
            and len(running) < args.workers
            and time.monotonic() >= next_launch_time
            and external_stable
        ):
            job = pending.pop(0)
            # A previous invocation may have completed this job while this
            # controller was waiting for a shared-machine slot.
            if latest_statuses().get(job["experiment_id"]) == "completed":
                append_jsonl(
                    event_log,
                    {"timestamp": timestamp(), "event": "skipped_completed", "experiment_id": job["experiment_id"]},
                )
                continue
            guard_log = ROOT / job.get(
                "log_file",
                f"artifacts/00_environment/launcher_logs/{job['experiment_id']}.log",
            )
            output_path = guard_output_dir / f"{job['experiment_id']}.log"
            output_handle = output_path.open("a", encoding="utf-8")
            command = guard_command(job, guard_log)
            output_handle.write(f"\n# {timestamp()}\n$ {shlex.join(command)}\n")
            output_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
            )
            running[process.pid] = (job, process, output_handle, output_path)
            append_jsonl(
                event_log,
                {
                    "timestamp": timestamp(),
                    "event": "guard_process_started",
                    "experiment_id": job["experiment_id"],
                    "guard_pid": process.pid,
                    "command": command,
                },
            )
            launched = True
            # Give the guard time to reserve/recheck before another local
            # controller worker competes for a slot.
            time.sleep(1.0)

        if pending or running:
            if not launched:
                if pending and time.monotonic() < next_launch_time:
                    time.sleep(min(2.0, max(0.1, next_launch_time - time.monotonic())))
                else:
                    time.sleep(2.0)

    summary = {
        "timestamp": timestamp(),
        "event": "queue_finished",
        "queue_id": queue_id,
        "total_jobs": len(jobs),
        "failed_jobs": failed,
        "status": "completed" if not failed else "completed_with_failures",
    }
    append_jsonl(event_log, summary)
    (queue_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
