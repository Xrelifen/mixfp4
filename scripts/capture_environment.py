#!/usr/bin/env python3
"""Capture exact repository, dependency, host, and GPU provenance."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "artifacts" / "00_environment"
FINAL_DIR = ROOT / "artifacts" / "06_final"
REPOSITORIES = {
    "workspace": ROOT,
    "nvfp4_razer": ROOT / "upstreams" / "NVFP4-RaZeR",
    "fouroversix": ROOT / "upstreams" / "fouroversix",
    "deepcompressor": ROOT / "upstreams" / "deepcompressor",
    "sana": ROOT / "upstreams" / "Sana",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd or ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as error:
        return {"command": command, "exit_code": 127, "output": str(error)}
    return {"command": command, "exit_code": result.returncode, "output": result.stdout.rstrip()}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def preserve_previous(path: Path, capture_time: str) -> None:
    if not path.exists():
        return
    history = path.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = capture_time.replace(":", "").replace("+", "_")
    shutil.copy2(path, history / f"{path.stem}_{stamp}{path.suffix}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path: Path, *arguments: str) -> str:
    result = run(["git", *arguments], cwd=path)
    if result["exit_code"]:
        return f"ERROR({result['exit_code']}): {result['output']}"
    return result["output"]


def git_present(path: Path) -> bool:
    return run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path)["exit_code"] == 0


def parse_submodules(path: Path) -> list[dict[str, Any]]:
    raw = git(path, "submodule", "status", "--recursive")
    if raw.startswith("ERROR"):
        return [{"error": raw}]
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            continue
        state = line[0]
        fields = line[1:].split()
        records.append(
            {
                "state": state,
                "sha": fields[0] if fields else None,
                "path": fields[1] if len(fields) > 1 else None,
                "description": " ".join(fields[2:]) if len(fields) > 2 else None,
            }
        )
    return records


def repository_manifest(name: str, path: Path) -> dict[str, Any]:
    if not path.exists() or not git_present(path):
        return {"name": name, "path": str(path), "present": False}
    remotes: list[dict[str, str]] = []
    for line in git(path, "remote", "-v").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            remotes.append(
                {"name": fields[0], "url": fields[1], "direction": fields[2].strip("()")}
            )
    status = git(path, "status", "--porcelain=v1", "--untracked-files=normal").splitlines()
    return {
        "name": name,
        "path": str(path),
        "present": True,
        "branch": git(path, "branch", "--show-current"),
        "head_sha": git(path, "rev-parse", "HEAD"),
        "head_commit": {
            "sha": git(path, "show", "-s", "--format=%H", "HEAD"),
            "author_time": git(path, "show", "-s", "--format=%aI", "HEAD"),
            "author": git(path, "show", "-s", "--format=%an <%ae>", "HEAD"),
            "subject": git(path, "show", "-s", "--format=%s", "HEAD"),
        },
        "remotes": remotes,
        "dirty": bool(status),
        "status_porcelain": status,
        "submodules": parse_submodules(path),
    }


def patch_record(name: str, path: Path) -> dict[str, Any]:
    if not path.exists() or not git_present(path):
        return {"name": name, "present": False}
    patch_dir = ENV_DIR / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    tracked_patch = git(path, "diff", "--binary", "--no-ext-diff", "HEAD")
    patch_path = patch_dir / f"{name}.patch"
    atomic_text(patch_path, tracked_patch + ("\n" if tracked_patch else ""))
    untracked_raw = git(path, "ls-files", "--others", "--exclude-standard")
    untracked_records: list[dict[str, Any]] = []
    # A nested upstream clone is represented by its own manifest, not recursively
    # hashed as workspace-local source.
    for relative in untracked_raw.splitlines():
        candidate = path / relative
        if candidate.is_file():
            untracked_records.append(
                {"path": relative, "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
            )
    return {
        "name": name,
        "present": True,
        "tracked_patch": str(patch_path.relative_to(ROOT)),
        "tracked_patch_bytes": len(tracked_patch.encode()),
        "tracked_patch_sha256": sha256_bytes(tracked_patch.encode()),
        "untracked_files": untracked_records,
    }


def package_versions() -> dict[str, str | None]:
    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "datasets": "datasets",
        "diffusers": "diffusers",
        "accelerate": "accelerate",
        "huggingface-hub": "huggingface-hub",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scikit-image": "scikit-image",
        "pytest": "pytest",
        "lpips": "lpips",
        "ImageReward": "image-reward",
        "openai-clip": "openai-clip",
        "clean-fid": "clean-fid",
        "timm": "timm",
        "fouroversix": "fouroversix",
        "deepcompressor": "deepcompressor",
    }
    versions: dict[str, str | None] = {}
    for label, package in packages.items():
        try:
            versions[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    result = run(command)
    rows: list[dict[str, Any]] = []
    if result["exit_code"] == 0:
        for line in result["output"].splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 6:
                rows.append(
                    {
                        "physical_index": int(fields[0]),
                        "name": fields[1],
                        "uuid": fields[2],
                        "pci_bus_id": fields[3],
                        "memory_total": fields[4],
                        "driver_version": fields[5],
                    }
                )
    return {"captured_at": now(), "command": command, "exit_code": result["exit_code"], "gpus": rows}


def main() -> int:
    capture_time = now()
    for directory in (
        ENV_DIR,
        ROOT / "artifacts" / "01_repo_audit",
        ROOT / "artifacts" / "02_tests",
        ROOT / "artifacts" / "03_phase_a" / "llm",
        ROOT / "artifacts" / "03_phase_a" / "diffusion",
        ROOT / "artifacts" / "04_phase_b" / "selector",
        ROOT / "artifacts" / "04_phase_b" / "permutation",
        ROOT / "artifacts" / "04_phase_b" / "rotation",
        ROOT / "artifacts" / "04_phase_b" / "timestep",
        ROOT / "artifacts" / "04_phase_b" / "combined",
        ROOT / "artifacts" / "05_cross_gpu",
        FINAL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for ledger in (
        ENV_DIR / "gpu_usage_log.jsonl",
        FINAL_DIR / "experiment_manifest.jsonl",
        FINAL_DIR / "failed_runs.jsonl",
    ):
        if not ledger.exists():
            ledger.write_text("", encoding="utf-8")

    repo_path = ENV_DIR / "repo_manifest.json"
    patch_path = ENV_DIR / "patch_manifest.json"
    preserve_previous(repo_path, capture_time)
    preserve_previous(patch_path, capture_time)
    atomic_json(
        repo_path,
        {
            "schema_version": 1,
            "captured_at": capture_time,
            "note": "Exact branches and SHAs are recorded; pre-existing workspace deletions are preserved.",
            "repositories": [repository_manifest(name, path) for name, path in REPOSITORIES.items()],
        },
    )
    atomic_json(
        patch_path,
        {
            "schema_version": 1,
            "captured_at": capture_time,
            "repositories": [patch_record(name, path) for name, path in REPOSITORIES.items()],
        },
    )

    freeze = run([sys.executable, "-m", "pip", "freeze"])
    atomic_text(ENV_DIR / "pip_freeze.txt", freeze["output"] + "\n")
    inventory = gpu_inventory()
    atomic_json(ENV_DIR / "gpu_inventory.json", inventory)
    credentials = {
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "HUGGING_FACE_HUB_TOKEN": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    commands = {
        "uname": run(["uname", "-a"]),
        "os_release": run(["cat", "/etc/os-release"]),
        "nvidia_smi": run(["nvidia-smi"]),
        "nvcc": run(["nvcc", "--version"]),
        "git": run(["git", "--version"]),
        "disk_home": run(["df", "-h", str(ROOT)]),
    }
    try:
        import torch

        torch_cuda_runtime = torch.version.cuda
        cuda_available = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001 - audit must survive broken dependencies.
        torch_cuda_runtime = f"ERROR: {type(error).__name__}: {error}"
        cuda_available = False
    environment = {
        "schema_version": 1,
        "captured_at": capture_time,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_versions(),
        "torch_cuda_runtime": torch_cuda_runtime,
        "cuda_available": cuda_available,
        "credential_presence_only": credentials,
        "cache_paths": {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE")
        },
        "commands": commands,
        "gpu_inventory_file": "artifacts/00_environment/gpu_inventory.json",
    }
    atomic_json(ENV_DIR / "environment.json", environment)
    lines = [
        f"captured_at={capture_time}",
        f"hostname={environment['hostname']}",
        f"platform={environment['platform']}",
        f"python={sys.version.splitlines()[0]}",
        f"python_executable={sys.executable}",
        f"torch_cuda_runtime={torch_cuda_runtime}",
        f"cuda_available={cuda_available}",
        "",
        "[packages]",
        *[f"{key}={value}" for key, value in sorted(environment["packages"].items())],
        "",
        "[commands]",
    ]
    for name, result in commands.items():
        lines.extend((f"$ {shlex_join(result['command'])}", result["output"], ""))
    atomic_text(ENV_DIR / "environment.txt", "\n".join(lines).rstrip() + "\n")
    print(f"captured environment at {capture_time}")
    return 0


def shlex_join(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


if __name__ == "__main__":
    raise SystemExit(main())
