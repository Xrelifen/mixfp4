#!/usr/bin/env python3
"""Exclusively launch one process on a policy-compliant project GPU.

The guard performs the four mandatory NVIDIA inspections twice: once while
selecting a device and once immediately before ``exec``.  It treats every
visible compute process as an occupancy conflict (including same-user jobs),
maps UUIDs back to physical indices, records process ownership/cwd where
possible, and counts MixFP4 jobs in sibling worktrees toward the global
three-GPU project limit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import pwd
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "artifacts" / "00_environment"
USAGE_LOG = ENV_DIR / "gpu_usage_log.jsonl"
LOCAL_LOCK_DIR = ENV_DIR / "gpu_locks"
GLOBAL_LOCK_DIR = Path(f"/tmp/mixfp4_phase_ab_gpu_guard_{os.getuid()}")
GLOBAL_MUTEX = GLOBAL_LOCK_DIR / "reservation.mutex"
# The older sibling worktree predates the /tmp project-wide lock.  Mirroring
# our live reservations into its lock directory makes its own guard count our
# jobs toward the same three-GPU limit, closing the only remaining race between
# independently started schedulers.  The path can be overridden or disabled
# (empty string) without changing experiment commands.
_DEFAULT_COMPAT_LOCK = ROOT.parent / "mixfp4" / "artifacts" / "00_environment" / "gpu_locks"
_COMPAT_RAW = os.environ.get("MIXFP4_COMPAT_LOCK_DIRS", str(_DEFAULT_COMPAT_LOCK))
COMPAT_LOCK_DIRS = tuple(
    Path(value).resolve()
    for value in _COMPAT_RAW.split(os.pathsep)
    if value and Path(value).resolve() != LOCAL_LOCK_DIR.resolve()
)
MAX_PROJECT_GPUS = 3
EXPECTED_MODELS = {
    0: "NVIDIA RTX A6000",
    1: "NVIDIA RTX A6000",
    2: "NVIDIA RTX A6000",
    3: "NVIDIA RTX A6000",
    4: "NVIDIA RTX 6000 Ada Generation",
    5: "NVIDIA RTX 6000 Ada Generation",
    6: "NVIDIA RTX 6000 Ada Generation",
}


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


def run_capture(command: list[str], *, required: bool = True) -> dict[str, Any]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(f"$ {shlex.join(command)}", flush=True)
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if required and result.returncode != 0:
        raise RuntimeError(f"command exited {result.returncode}: {shlex.join(command)}")
    return {"command": command, "exit_code": result.returncode, "output": result.stdout}


def parse_int(value: str) -> int:
    digits = "".join(character for character in value if character.isdigit())
    return int(digits) if digits else 0


def process_details(pid: int) -> dict[str, Any]:
    ps = run_capture(["ps", "-o", "user=,pid=,cmd=", "-p", str(pid)], required=False)
    owner = "unknown"
    cwd = None
    try:
        owner = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError, KeyError, PermissionError):
        pass
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError, OSError):
        pass
    command = ps["output"].strip()
    return {"owner": owner, "cwd": cwd, "ps": command}


def is_project_process(process: dict[str, Any]) -> bool:
    if process.get("owner") != pwd.getpwuid(os.getuid()).pw_name:
        return False
    cwd = process.get("cwd") or ""
    command = process.get("ps") or ""
    return (
        cwd.startswith("/home/JaaaaaA_l/mixfp4")
        or "mixfp4" in command.lower()
        or "granularity_study" in command
        or "run_experiment.py" in command
    )


def inspect_gpus() -> dict[str, Any]:
    """Run all required inspections and return a UUID-resolved snapshot."""
    full = run_capture(["nvidia-smi"])
    gpu_query = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    compute_query = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        required=False,
    )
    pmon = run_capture(["nvidia-smi", "pmon", "-c", "1"], required=False)

    inventory: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for row in csv.reader(gpu_query["output"].splitlines()):
        if len(row) < 6:
            continue
        index = int(row[0].strip())
        gpu = {
            "physical_index": index,
            "name": row[1].strip(),
            "uuid": row[2].strip(),
            "memory_used_mib": parse_int(row[3]),
            "memory_total_mib": parse_int(row[4]),
            "utilization_gpu_percent": parse_int(row[5]),
            "expected_name": EXPECTED_MODELS.get(index),
        }
        gpu["mapping_valid"] = gpu["expected_name"] == gpu["name"]
        gpu["kind"] = "a6000" if index <= 3 else "ada" if index <= 6 else "unsupported"
        inventory[index] = gpu
        uuid_to_index[gpu["uuid"]] = index

    processes: list[dict[str, Any]] = []
    for row in csv.reader(compute_query["output"].splitlines()):
        if len(row) < 4 or not row[1].strip().isdigit():
            continue
        pid = int(row[1].strip())
        details = process_details(pid)
        process = {
            "gpu_uuid": row[0].strip(),
            "physical_index": uuid_to_index.get(row[0].strip()),
            "pid": pid,
            "process_name": row[2].strip(),
            "used_memory": row[3].strip(),
            **details,
        }
        process["is_project_process"] = is_project_process(process)
        processes.append(process)

    for index, gpu in inventory.items():
        gpu["compute_processes"] = [p for p in processes if p["physical_index"] == index]

    return {
        "timestamp": timestamp(),
        "inventory": inventory,
        "processes": processes,
        "raw": {
            "nvidia_smi": full,
            "gpu_query": gpu_query,
            "compute_query": compute_query,
            "pmon": pmon,
        },
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_global_reservations() -> dict[int, dict[str, Any]]:
    GLOBAL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    reservations: dict[int, dict[str, Any]] = {}
    for path in GLOBAL_LOCK_DIR.glob("gpu_*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            index = int(value["physical_gpu_index"])
            guard_pid = int(value["guard_pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            append_jsonl(
                USAGE_LOG,
                {"timestamp": timestamp(), "event": "malformed_global_lock", "path": str(path)},
            )
            continue
        if process_alive(guard_pid):
            reservations[index] = value
        else:
            path.unlink(missing_ok=True)
            append_jsonl(
                USAGE_LOG,
                {
                    "timestamp": timestamp(),
                    "event": "removed_stale_global_lock",
                    "physical_gpu_index": index,
                    "guard_pid": guard_pid,
                },
            )
    return reservations


def live_compatible_reservations() -> dict[int, dict[str, Any]]:
    """Read live reservations made by compatible sibling-worktree guards."""

    reservations: dict[int, dict[str, Any]] = {}
    for directory in COMPAT_LOCK_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob("gpu_*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                raw_index = value["physical_gpu_index"] if "physical_gpu_index" in value else value["gpu_index"]
                index = int(raw_index)
                guard_pid = int(value["guard_pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if process_alive(guard_pid):
                value = {**value, "compatible_lock_path": str(path)}
                reservations[index] = value
    return reservations


def cleanup_stale_local_locks() -> None:
    LOCAL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for path in LOCAL_LOCK_DIR.glob("gpu_*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            guard_pid = int(value["guard_pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not process_alive(guard_pid):
            path.unlink(missing_ok=True)
            append_jsonl(
                USAGE_LOG,
                {
                    "timestamp": timestamp(),
                    "event": "removed_stale_local_lock",
                    "path": str(path),
                    "guard_pid": guard_pid,
                },
            )


def all_reservations() -> dict[int, dict[str, Any]]:
    cleanup_stale_local_locks()
    value = live_compatible_reservations()
    value.update(live_global_reservations())
    return value


def unlink_owned_lock(path: Path | None) -> None:
    """Remove only a reservation that is still owned by this guard PID."""

    if path is None or not path.exists():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("guard_pid", -1)) == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def evaluate(snapshot: dict[str, Any], reservations: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    state: dict[int, dict[str, Any]] = {}
    for index, raw_gpu in snapshot["inventory"].items():
        gpu = dict(raw_gpu)
        reasons: list[str] = []
        if index not in EXPECTED_MODELS:
            reasons.append("physical_index_not_allowed")
        if not gpu["mapping_valid"]:
            reasons.append("gpu_type_mapping_mismatch")
        if gpu["compute_processes"]:
            reasons.append("compute_process_present")
        if index in reservations:
            reasons.append("global_project_reservation_present")
        if not gpu["compute_processes"] and gpu["memory_used_mib"] > 256:
            reasons.append("unattributed_memory_occupancy")
        if not gpu["compute_processes"] and gpu["utilization_gpu_percent"] > 5:
            reasons.append("unattributed_gpu_utilization")
        gpu["available"] = not reasons
        gpu["reasons"] = reasons
        gpu["reservation"] = reservations.get(index)
        state[index] = gpu
    return state


def active_project_gpu_indices(snapshot: dict[str, Any], reservations: dict[int, dict[str, Any]]) -> set[int]:
    indices = {
        int(process["physical_index"])
        for process in snapshot["processes"]
        if process["physical_index"] is not None and process["is_project_process"]
    }
    indices.update(reservations)
    return indices


def choose_and_reserve(
    experiment_id: str,
    requested_index: int | None,
    requested_kind: str,
) -> tuple[dict[str, Any], list[Path]]:
    GLOBAL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with GLOBAL_MUTEX.open("a", encoding="utf-8") as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
        compat_mutexes: list[Any] = []
        lock_paths: list[Path] = []
        try:
            # Lock compatible schedulers before inspecting or counting their
            # reservations.  They cannot add a third reservation between our
            # count and mirror-lock creation.
            for directory in sorted(COMPAT_LOCK_DIRS, key=str):
                if not directory.exists():
                    continue
                mutex_handle = (directory / "reservation.mutex").open("a", encoding="utf-8")
                fcntl.flock(mutex_handle.fileno(), fcntl.LOCK_EX)
                compat_mutexes.append(mutex_handle)

            snapshot = inspect_gpus()
            reservations = all_reservations()
            state = evaluate(snapshot, reservations)
            active_indices = active_project_gpu_indices(snapshot, reservations)
            decision: dict[str, Any] = {
                "timestamp": timestamp(),
                "event": "selection",
                "experiment_id": experiment_id,
                "requested_gpu_index": requested_index,
                "requested_gpu_kind": requested_kind,
                "active_project_gpu_indices": sorted(active_indices),
                "active_project_gpu_count": len(active_indices),
                "max_project_gpus": MAX_PROJECT_GPUS,
                "compatible_lock_directories": [str(path) for path in COMPAT_LOCK_DIRS],
                "gpus": state,
                "inspection": snapshot["raw"],
            }
            if len(active_indices) >= MAX_PROJECT_GPUS:
                decision["decision"] = "rejected_project_gpu_limit"
                append_jsonl(USAGE_LOG, decision)
                raise RuntimeError(
                    f"project-wide GPU limit reached: {len(active_indices)}/{MAX_PROJECT_GPUS} "
                    f"on physical GPUs {sorted(active_indices)}"
                )
            candidates = [
                candidate
                for index, candidate in sorted(state.items())
                if candidate["available"]
                and (requested_index is None or index == requested_index)
                and (requested_kind == "any" or candidate["kind"] == requested_kind)
            ]
            if not candidates:
                decision["decision"] = "rejected_no_compliant_gpu"
                append_jsonl(USAGE_LOG, decision)
                raise RuntimeError("no compliant unoccupied GPU matches this request")
            gpu = candidates[0]
            lock_value = {
                "schema_version": 1,
                "guard_pid": os.getpid(),
                "workspace": str(ROOT),
                "experiment_id": experiment_id,
                "physical_gpu_index": gpu["physical_index"],
                "logical_gpu_index": 0,
                "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"],
                "user": pwd.getpwuid(os.getuid()).pw_name,
                "reserved_at": timestamp(),
            }
            global_path = GLOBAL_LOCK_DIR / f"gpu_{gpu['physical_index']}.json"
            local_path = LOCAL_LOCK_DIR / f"gpu_{gpu['physical_index']}.json"
            lock_paths = [global_path, local_path]
            lock_paths.extend(
                directory / f"gpu_{gpu['physical_index']}.json"
                for directory in COMPAT_LOCK_DIRS
                if directory.exists()
            )
            for path in lock_paths:
                value = dict(lock_value)
                if path.parent in COMPAT_LOCK_DIRS:
                    value["gpu_index"] = gpu["physical_index"]
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(value, sort_keys=True) + "\n")
            decision["decision"] = "reserved"
            decision["selected_gpu"] = gpu
            decision["reservation"] = lock_value
            append_jsonl(USAGE_LOG, decision)
        except BaseException:
            for path in lock_paths:
                unlink_owned_lock(path)
            raise
        finally:
            for handle in reversed(compat_mutexes):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        fcntl.flock(mutex.fileno(), fcntl.LOCK_UN)
    return gpu, lock_paths


def immediate_recheck(experiment_id: str, gpu: dict[str, Any]) -> dict[str, Any]:
    """Mandatory second inspection performed immediately before child launch."""
    snapshot = inspect_gpus()
    reservations = all_reservations()
    state = evaluate(snapshot, {i: r for i, r in reservations.items() if i != gpu["physical_index"]})
    selected = state[gpu["physical_index"]]
    active_indices = active_project_gpu_indices(snapshot, reservations)
    valid = selected["available"] and len(active_indices) <= MAX_PROJECT_GPUS
    record = {
        "timestamp": timestamp(),
        "event": "immediate_pre_exec_recheck",
        "experiment_id": experiment_id,
        "physical_gpu_index": gpu["physical_index"],
        "logical_gpu_index": 0,
        "active_project_gpu_indices": sorted(active_indices),
        "gpus": state,
        "inspection": snapshot["raw"],
        "decision": "approved" if valid else "rejected",
    }
    append_jsonl(USAGE_LOG, record)
    if not valid:
        raise RuntimeError(
            f"GPU {gpu['physical_index']} became occupied or project limit changed before exec"
        )
    return snapshot


def inspect_only(experiment_id: str, requested_index: int | None, requested_kind: str) -> int:
    snapshot = inspect_gpus()
    reservations = all_reservations()
    state = evaluate(snapshot, reservations)
    active_indices = active_project_gpu_indices(snapshot, reservations)
    append_jsonl(
        USAGE_LOG,
        {
            "timestamp": timestamp(),
            "event": "inspection_only",
            "experiment_id": experiment_id,
            "requested_gpu_index": requested_index,
            "requested_gpu_kind": requested_kind,
            "active_project_gpu_indices": sorted(active_indices),
            "gpus": state,
            "inspection": snapshot["raw"],
        },
    )
    print(json.dumps({"active_project_gpu_indices": sorted(active_indices), "gpus": state}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="inspection")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-kind", choices=("any", "a6000", "ada"), default="any")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="quiet interval before the final full pre-exec inspection",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu_index is not None and args.gpu_index not in EXPECTED_MODELS:
        parser.error("--gpu-index must be a physical index from 0 through 6")
    if not 0 <= args.settle_seconds <= 30:
        parser.error("--settle-seconds must be between 0 and 30")
    if args.inspect_only:
        return inspect_only(args.experiment_id, args.gpu_index, args.gpu_kind)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    gpu: dict[str, Any] | None = None
    reservation_paths: list[Path] = []
    child: subprocess.Popen[str] | None = None
    header: dict[str, Any] = {}
    exit_code = 75
    try:
        gpu, reservation_paths = choose_and_reserve(
            args.experiment_id, args.gpu_index, args.gpu_kind
        )
        immediate_recheck(args.experiment_id, gpu)
        if args.settle_seconds:
            append_jsonl(
                USAGE_LOG,
                {
                    "timestamp": timestamp(),
                    "event": "pre_exec_stabilization_started",
                    "experiment_id": args.experiment_id,
                    "physical_gpu_index": gpu["physical_index"],
                    "settle_seconds": args.settle_seconds,
                },
            )
            time.sleep(args.settle_seconds)
        # This third complete inspection is intentionally the last operation
        # before building the environment and execing the child.  It catches
        # non-cooperating sibling-worktree schedulers that race our lock.
        immediate_recheck(args.experiment_id, gpu)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu["physical_index"])
        environment["MIXFP4_PHYSICAL_GPU"] = str(gpu["physical_index"])
        environment["MIXFP4_LOGICAL_GPU"] = "0"
        environment["MIXFP4_GPU_UUID"] = gpu["uuid"]
        environment["MIXFP4_CHILD_COMMAND_SHELL"] = shlex.join(command)
        environment["PYTHONUNBUFFERED"] = "1"
        requested_log_path = (
            args.log_file or ENV_DIR / "launcher_logs" / f"{args.experiment_id}.log"
        )
        if not requested_log_path.is_absolute():
            requested_log_path = ROOT / requested_log_path
        launch_stamp = timestamp().replace(":", "").replace("+", "_")
        log_path = (
            requested_log_path.parent
            / "attempt_logs"
            / (
                f"{requested_log_path.stem}__{launch_stamp}__guard{os.getpid()}"
                f"{requested_log_path.suffix or '.log'}"
            )
        )
        environment["MIXFP4_LAUNCH_LOG"] = str(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        requested_log_path.parent.mkdir(parents=True, exist_ok=True)
        with requested_log_path.open("a", encoding="utf-8") as index_log:
            index_log.write(
                json.dumps(
                    {
                        "timestamp": timestamp(),
                        "event": "immutable_attempt_log_created",
                        "experiment_id": args.experiment_id,
                        "attempt_log": str(log_path),
                        "guard_pid": os.getpid(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        header = {
            "timestamp": timestamp(),
            "event": "process_started",
            "experiment_id": args.experiment_id,
            "command": command,
            "command_shell": shlex.join(command),
            "physical_gpu_index": gpu["physical_index"],
            "logical_gpu_index": 0,
            "cuda_visible_devices": str(gpu["physical_index"]),
            "gpu_name": gpu["name"],
            "gpu_uuid": gpu["uuid"],
            "guard_pid": os.getpid(),
            "user": pwd.getpwuid(os.getuid()).pw_name,
            "log_file": str(log_path),
        }
        with log_path.open("a", encoding="utf-8") as log:
            log.write("# gpu_guard launch\n" + json.dumps(header, indent=2, sort_keys=True) + "\n")
            log.flush()
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            append_jsonl(USAGE_LOG, {**header, "child_pid": child.pid})
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            exit_code = child.wait()
        append_jsonl(
            USAGE_LOG,
            {**header, "event": "process_finished", "child_pid": child.pid, "exit_code": exit_code, "end_time": timestamp()},
        )
        return exit_code
    except Exception as error:  # noqa: BLE001 - every refusal/failure must be persisted.
        append_jsonl(
            USAGE_LOG,
            {
                "timestamp": timestamp(),
                "event": "launch_failed_or_blocked",
                "experiment_id": args.experiment_id,
                "physical_gpu_index": gpu["physical_index"] if gpu else args.gpu_index,
                "logical_gpu_index": 0 if gpu else None,
                "command": command,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        print(f"gpu_guard: {type(error).__name__}: {error}", file=sys.stderr)
        return 75
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        for path in reservation_paths:
            unlink_owned_lock(path)
        if gpu is not None:
            append_jsonl(
                USAGE_LOG,
                {
                    "timestamp": timestamp(),
                    "event": "reservation_released",
                    "experiment_id": args.experiment_id,
                    "physical_gpu_index": gpu["physical_index"],
                    "logical_gpu_index": 0,
                },
            )


if __name__ == "__main__":
    raise SystemExit(main())
