"""Persisted, model-specific WikiText/C4 evaluation and calibration slices."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .artifacts import ROOT, atomic_json


DATA_DIR = ROOT / "artifacts" / "00_environment" / "eval_data"
MANIFEST_PATH = ROOT / "artifacts" / "00_environment" / "dataset_manifest.json"
MODEL_MANIFEST_PATH = ROOT / "artifacts" / "00_environment" / "model_manifest.json"
HF_HUB_CACHE = os.environ.get("HUGGINGFACE_HUB_CACHE", "/share2/huggingface/hub")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "/share2/huggingface/datasets")
C4_VALIDATION_ARROW = Path(
    os.environ.get(
        "MIXFP4_C4_VALIDATION_ARROW",
        "/share2/huggingface/datasets/allenai___c4/default-4cb1202b07c0cd0b/0.0.0/"
        "607bd4c8450a42878aa9ddc051a65a055450ef87/c4-validation.arrow",
    )
)


def slug(value: str) -> str:
    return value.replace("/", "--").replace(".", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(model_id: str) -> tuple[str, str]:
    snapshot = Path(snapshot_download(model_id, cache_dir=HF_HUB_CACHE, local_files_only=True))
    revision = snapshot.name
    return str(snapshot), revision


def load_tokenizer(model_id: str):
    snapshot, revision = resolve_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True, use_fast=True)
    return tokenizer, snapshot, revision


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": {}}
    return json.loads(path.read_text())


def _save_record(path: Path, key: str, record: dict[str, Any]) -> None:
    manifest = _load_manifest(path)
    manifest.setdefault("records", {})[key] = record
    atomic_json(path, manifest)


def register_model(model_id: str, snapshot: str, revision: str) -> None:
    config_path = Path(snapshot) / "config.json"
    record = {
        "model_id": model_id,
        "snapshot_path": snapshot,
        "revision": revision,
        "config_sha256": sha256_file(config_path),
    }
    _save_record(MODEL_MANIFEST_PATH, model_id, record)


def _c4_dataset():
    if C4_VALIDATION_ARROW.exists():
        return Dataset.from_file(str(C4_VALIDATION_ARROW))
    return load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        cache_dir=HF_DATASETS_CACHE,
    )


def prepare_wikitext(model_id: str, seq_len: int = 2048) -> dict[str, Any]:
    tokenizer, snapshot, revision = load_tokenizer(model_id)
    register_model(model_id, snapshot, revision)
    key = f"{model_id}|wikitext2|seq{seq_len}"
    path = DATA_DIR / f"{slug(model_id)}_{revision[:12]}_wikitext2_seq{seq_len}.npz"
    if path.exists():
        manifest = _load_manifest(MANIFEST_PATH)
        return manifest["records"][key]
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=HF_DATASETS_CACHE)
    text = "\n\n".join(dataset["text"])
    ids = tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"].reshape(-1).astype(np.int32)
    usable = (ids.size // seq_len) * seq_len
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input_ids=ids[:usable].reshape(-1, seq_len))
    record = {
        "key": key,
        "dataset": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_snapshot": snapshot,
        "sequence_length": seq_len,
        "num_sequences": usable // seq_len,
        "source_policy": "NVFP4-RaZeR-compatible concatenated WikiText-2 test text",
        "data_file": str(path.relative_to(ROOT)),
        "data_sha256": sha256_file(path),
    }
    _save_record(MANIFEST_PATH, key, record)
    return record


def _sample_c4_tokens(tokenizer, dataset, *, count: int, seq_len: int, seed: int, excluded_indices: set[int]) -> tuple[np.ndarray, list[dict[str, int]]]:
    rng = random.Random(seed)
    arrays: list[np.ndarray] = []
    entries: list[dict[str, int]] = []
    used = set(excluded_indices)
    attempts = 0
    while len(arrays) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("unable to find enough long C4 samples")
        index = rng.randrange(len(dataset))
        if index in used:
            continue
        ids = np.asarray(tokenizer(dataset[index]["text"], add_special_tokens=False)["input_ids"], dtype=np.int32)
        if ids.size < seq_len + 1:
            continue
        offset = rng.randrange(ids.size - seq_len + 1)
        arrays.append(ids[offset : offset + seq_len])
        entries.append({"dataset_index": index, "token_offset": offset, "source_token_count": int(ids.size)})
        used.add(index)
    return np.stack(arrays), entries


def prepare_c4(model_id: str, seq_len: int = 2048, count: int = 256, seed: int = 0) -> dict[str, Any]:
    tokenizer, snapshot, revision = load_tokenizer(model_id)
    register_model(model_id, snapshot, revision)
    key = f"{model_id}|c4|seq{seq_len}|n{count}|seed{seed}"
    path = DATA_DIR / f"{slug(model_id)}_{revision[:12]}_c4_seq{seq_len}_n{count}_seed{seed}.npz"
    if path.exists():
        return _load_manifest(MANIFEST_PATH)["records"][key]
    dataset = _c4_dataset()
    arrays, entries = _sample_c4_tokens(tokenizer, dataset, count=count, seq_len=seq_len, seed=seed, excluded_indices=set())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input_ids=arrays)
    record = {
        "key": key,
        "dataset": "c4",
        "dataset_config": "allenai/c4 validation shard 0/8",
        "split": "validation",
        "dataset_fingerprint": dataset._fingerprint,
        "source_arrow_file": str(C4_VALIDATION_ARROW) if C4_VALIDATION_ARROW.exists() else None,
        "source_arrow_sha256": sha256_file(C4_VALIDATION_ARROW) if C4_VALIDATION_ARROW.exists() else None,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_snapshot": snapshot,
        "sequence_length": seq_len,
        "num_sequences": count,
        "seed": seed,
        "entries": entries,
        "data_file": str(path.relative_to(ROOT)),
        "data_sha256": sha256_file(path),
    }
    _save_record(MANIFEST_PATH, key, record)
    return record


def prepare_calibration(
    model_id: str,
    *,
    seq_len: int = 128,
    count: int = 256,
    seed: int = 314159,
) -> dict[str, Any]:
    tokenizer, snapshot, revision = load_tokenizer(model_id)
    register_model(model_id, snapshot, revision)
    key = f"{model_id}|calibration_c4|seq{seq_len}|n{count}|seed{seed}"
    path = DATA_DIR / f"{slug(model_id)}_{revision[:12]}_calibration_c4_seq{seq_len}_n{count}_seed{seed}.npz"
    if path.exists():
        return _load_manifest(MANIFEST_PATH)["records"][key]
    dataset = _c4_dataset()
    manifest = _load_manifest(MANIFEST_PATH)
    excluded = {
        entry["dataset_index"]
        for record in manifest.get("records", {}).values()
        if record.get("dataset") == "c4"
        for entry in record.get("entries", [])
    }
    arrays, entries = _sample_c4_tokens(tokenizer, dataset, count=count, seq_len=seq_len, seed=seed, excluded_indices=excluded)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input_ids=arrays)
    record = {
        "key": key,
        "dataset": "c4_calibration",
        "dataset_config": "allenai/c4 validation shard 0/8",
        "split": "validation (indices disjoint from persisted C4 evaluation slices)",
        "dataset_fingerprint": dataset._fingerprint,
        "source_arrow_file": str(C4_VALIDATION_ARROW) if C4_VALIDATION_ARROW.exists() else None,
        "source_arrow_sha256": sha256_file(C4_VALIDATION_ARROW) if C4_VALIDATION_ARROW.exists() else None,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_snapshot": snapshot,
        "sequence_length": seq_len,
        "num_sequences": count,
        "seed": seed,
        "entries": entries,
        "data_file": str(path.relative_to(ROOT)),
        "data_sha256": sha256_file(path),
        "exact_samples_saved": True,
    }
    _save_record(MANIFEST_PATH, key, record)
    return record


def load_sequences(record: dict[str, Any]) -> np.ndarray:
    with np.load(ROOT / record["data_file"]) as value:
        return value["input_ids"].copy()
