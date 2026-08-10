# mixfp4

This repository contains the project code and experiment orchestration for the mixfp4 workflow.

## 1. Clone repository

```bash
git clone https://github.com/Xrelifen/mixfp4.git
cd mixfp4
```

Sub-projects under `upstreams/` are tracked as git submodules, so clone with:

```bash
git clone --recurse-submodules https://github.com/Xrelifen/mixfp4.git
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## 2. Repository layout

- `project_quant/` core Python package for quantization workflow and scripts.
- `scripts/` data/experiment orchestration scripts.
- `tests/` test helpers and validation scripts.
- `research*.md` research notes and phase summaries.
- `upstreams/` external dependencies as git submodules.
- `artifacts/` local experiment outputs (intentional local workspace, not tracked by git).

## 3. Initialize python environment

Use Python 3.10+ and create an isolated env.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```
This `requirements.txt` is a single full dependency list for the codebase, including both LLM and diffusion scripts.

Some subprojects provide their own dependencies in their respective directories; install those when working in that subproject.

## 4. Quick start

1. Ensure submodules are initialized.
2. Explore `research*.md` for the current phase and target scripts.
3. Run the relevant script in `scripts/`.

Example:

```bash
python scripts/build_experiment_queues.py
python scripts/run_llm_experiment.py
```

## 5. Run tests

```bash
pytest -q
```

## 6. Development workflow

- Keep workspace outputs under `artifacts/` locally; they are ignored by git on purpose.
- Commit code changes in `project_quant/`, `scripts/`, and `tests/`.
- Keep third-party code under `upstreams/` as submodule pointers and avoid directly editing tracked history there unless needed.

## 7. Update submodules

```bash
git submodule update --remote --merge
```

Or enter a specific submodule and pull normally:

```bash
cd upstreams/fouroversix
git pull
git checkout main
cd -
git add upstreams/fouroversix
git commit -m "chore: bump upstreams/fouroversix"
```
