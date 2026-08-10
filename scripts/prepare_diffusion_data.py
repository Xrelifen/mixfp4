#!/usr/bin/env python3
"""Pin disjoint SANA calibration, proxy, screening, and finalist prompt sets."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.artifacts import atomic_json, timestamp  # noqa: E402


ENV = ROOT / "artifacts" / "00_environment"
PROMPT_DIR = ENV / "diffusion_prompts"
QDIFF = ROOT / "upstreams" / "deepcompressor" / "examples" / "diffusion" / "prompts" / "qdiff.yaml"
SANA_SNAPSHOT = Path(
    "/share2/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/"
    "snapshots/ac0da2ff55fbe434795be0dce883042e4d49e2fc"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(name: str, rows: list[dict]) -> dict:
    path = PROMPT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "count": len(rows),
        "first_id": rows[0]["prompt_id"],
        "last_id": rows[-1]["prompt_id"],
    }


def component(name: str, relative: str) -> dict:
    path = SANA_SNAPSHOT / relative
    return {
        "name": name,
        "config_path": str(path),
        "config_sha256": sha256(path),
        "config": json.loads(path.read_text()),
    }


def main() -> None:
    if not SANA_SNAPSHOT.exists():
        raise FileNotFoundError(SANA_SNAPSHOT)
    qdiff = yaml.safe_load(QDIFF.read_text())
    qdiff_items = sorted(((str(key), str(value)) for key, value in qdiff.items()), key=lambda item: item[0])
    if len(qdiff_items) != 1024:
        raise RuntimeError(f"expected 1024 qdiff prompts, found {len(qdiff_items)}")

    mjhq_snapshot = Path(
        snapshot_download(
            "playgroundai/MJHQ-30K",
            repo_type="dataset",
            allow_patterns=("meta_data.json",),
            cache_dir="/share2/huggingface/hub",
        )
    )
    mjhq_path = mjhq_snapshot / "meta_data.json"
    mjhq = json.loads(mjhq_path.read_text())
    names = sorted(mjhq)
    random.Random(0).shuffle(names)
    screening_names = sorted(names[:128])
    finalist_names = sorted(names[128 : 128 + 1024])
    if set(screening_names) & set(finalist_names):
        raise RuntimeError("screening and finalist MJHQ sets overlap")

    calibration = [
        {"prompt_id": f"qdiff-{key}", "prompt": prompt, "seed": 314159 + index, "source_index": index}
        for index, (key, prompt) in enumerate(qdiff_items[:256])
    ]
    proxy = [
        {"prompt_id": f"qdiff-{key}", "prompt": prompt, "seed": 271828 + index, "source_index": 256 + index}
        for index, (key, prompt) in enumerate(qdiff_items[256:272])
    ]
    screening = [
        {
            "prompt_id": f"mjhq-{name}",
            "prompt": str(mjhq[name]["prompt"]),
            "category": str(mjhq[name]["category"]),
            "seed": 12345 + index,
        }
        for index, name in enumerate(screening_names)
    ]
    finalist = [
        {
            "prompt_id": f"mjhq-{name}",
            "prompt": str(mjhq[name]["prompt"]),
            "category": str(mjhq[name]["category"]),
            "seed": 54321 + index,
        }
        for index, name in enumerate(finalist_names)
    ]
    sets = {
        "calibration_qdiff_256": write_jsonl("calibration_qdiff_256.jsonl", calibration),
        "proxy_qdiff_16": write_jsonl("proxy_qdiff_16.jsonl", proxy),
        "screening_mjhq_128": write_jsonl("screening_mjhq_128.jsonl", screening),
        "finalist_mjhq_1024": write_jsonl("finalist_mjhq_1024.jsonl", finalist),
    }

    model_index = json.loads((SANA_SNAPSHOT / "model_index.json").read_text())
    manifest = {
        "schema_version": 1,
        "captured_at": timestamp(),
        "model_id": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "model_revision": SANA_SNAPSHOT.name,
        "model_snapshot": str(SANA_SNAPSHOT),
        "pipeline_class": model_index["_class_name"],
        "components": {
            "transformer": component("SanaTransformer2DModel", "transformer/config.json"),
            "scheduler": component("DPMSolverMultistepScheduler", "scheduler/scheduler_config.json"),
            "vae": component("AutoencoderDC", "vae/config.json"),
            "text_encoder": component("Gemma2Model", "text_encoder/config.json"),
            "tokenizer": component("GemmaTokenizerFast", "tokenizer/tokenizer_config.json"),
        },
        "protocol": {
            "dtype": "torch.bfloat16",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
            "pag": None,
            "negative_prompt": "",
            "clean_caption": True,
            "use_resolution_binning": True,
            "max_sequence_length": 300,
            "proxy_timesteps": [999, 500, 1],
            "calibration_timestep_assignment": "round_robin over [999,500,1]",
        },
        "prompt_sets": sets,
        "disjointness": {
            "qdiff_calibration_vs_qdiff_proxy": not ({row["prompt_id"] for row in calibration} & {row["prompt_id"] for row in proxy}),
            "mjhq_screening_vs_mjhq_finalist": not ({row["prompt_id"] for row in screening} & {row["prompt_id"] for row in finalist}),
            "calibration_and_evaluation_sources": "qdiff calibration; MJHQ image evaluation",
        },
        "sources": {
            "qdiff_path": str(QDIFF.relative_to(ROOT)),
            "qdiff_sha256": sha256(QDIFF),
            "qdiff_repository_commit": "69f3473f5e1c1504bae35cc50c7858ef900a9b17",
            "mjhq_repo_id": "playgroundai/MJHQ-30K",
            "mjhq_revision": mjhq_snapshot.name,
            "mjhq_metadata_path": str(mjhq_path),
            "mjhq_metadata_sha256": sha256(mjhq_path),
            "mjhq_selection": "random.Random(0).shuffle(sorted IDs); screening first 128; finalist next 1024; each subset sorted",
        },
    }
    atomic_json(ENV / "diffusion_eval_manifest.json", manifest)
    print(json.dumps({"manifest": str((ENV / "diffusion_eval_manifest.json").relative_to(ROOT)), "sets": sets}, indent=2))


if __name__ == "__main__":
    main()
