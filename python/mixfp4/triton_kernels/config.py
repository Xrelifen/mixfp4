"""Autotune settings and the persistent config cache.

Mirrors GemLite's split between a per-matmul-type autotune *mode* and a JSON cache of the winning
configs, because the same problem applies here: the config space is large enough that autotuning
from scratch on every process start is unacceptable for serving, but the best config genuinely
varies with (M bucket, N, K).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch

#: ``"default"`` (a handful of configs), ``"fast"`` (a curated list) or ``"max"`` (full sweep).
AUTOTUNE_MODE = os.environ.get("MIXFP4_AUTOTUNE", "fast")

#: Where tuned configs are read from and written to.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

_FILE_LOCK = threading.Lock()

#: ``{matmul_type: {signature: config_dict}}``
CONFIG_CACHE: dict[str, dict[str, dict]] = {}


def signature(m: int, n: int, k: int) -> str:
    """Cache key.  M is bucketed so nearby batch sizes share a config; N and K are exact."""
    from .utils import get_closest_m
    return str((get_closest_m(m), n, k))


def device_tag() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_properties(0).name.lower().replace(" ", "_")


def load_config(path: str | os.PathLike | None = None) -> bool:
    """Load a tuned-config JSON.  Defaults to the file matching this GPU, if one is shipped."""
    global CONFIG_CACHE
    if path is None:
        tag = device_tag()
        candidates = sorted(CONFIG_DIR.glob("*.json")) if CONFIG_DIR.is_dir() else []
        # Longest matching tag wins, so "rtx_5090" beats a hypothetical "rtx_50".
        matches = sorted((c for c in candidates if c.stem in tag), key=lambda c: -len(c.stem))
        if not matches:
            return False
        path = matches[0]
    path = Path(path)
    if not path.is_file():
        return False
    with _FILE_LOCK:
        CONFIG_CACHE = json.loads(path.read_text())
    return True


def save_config(path: str | os.PathLike) -> None:
    """Persist whatever has been tuned so far."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FILE_LOCK:
        path.write_text(json.dumps(CONFIG_CACHE, indent=1, sort_keys=True))


def cache_kernel_config(kernel, matmul_type: str, keys) -> int:
    """Harvest the autotuner's winners into :data:`CONFIG_CACHE`.

    Triton keys ``kernel.cache`` by the tuple of autotune key values; we re-key by the bucketed
    signature so a later run with a nearby M hits the same entry.
    """
    from .utils import get_closest_m
    bucket = CONFIG_CACHE.setdefault(matmul_type, {})
    added = 0
    for device_cache in _iter_caches(kernel):
        for key, cfg in device_cache.items():
            values = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            if not {"M", "N", "K"} <= values.keys():
                continue
            sig = str((get_closest_m(values["M"]), values["N"], values["K"]))
            entry = dict(cfg.all_kwargs()) if hasattr(cfg, "all_kwargs") else dict(cfg.kwargs)
            entry.setdefault("num_warps", getattr(cfg, "num_warps", 4))
            entry.setdefault("num_stages", getattr(cfg, "num_stages", 3))
            bucket[sig] = {k: v for k, v in entry.items() if not callable(v)}
            added += 1
    return added


def _iter_caches(kernel):
    cache = getattr(kernel, "cache", {})
    # Triton >= 3.2 keys the autotuner cache by device first.
    if cache and all(isinstance(v, dict) for v in cache.values()):
        return list(cache.values())
    return [cache]


def cached_config(matmul_type: str, m: int, n: int, k: int) -> dict | None:
    return CONFIG_CACHE.get(matmul_type, {}).get(signature(m, n, k))


def reset_config() -> None:
    CONFIG_CACHE.clear()


# Best-effort: pick up a shipped config for this GPU at import time.
try:  # pragma: no cover - depends on the machine
    load_config()
except Exception:
    pass
