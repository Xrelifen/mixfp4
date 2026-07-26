# Quantisation perplexity study — state, and what to do next

Handoff document. Everything below was produced on a machine whose GPU was contended for the whole
session (an unrelated job held 26–30 GB of a 32 GB RTX 5090, leaving 2–5 GB free), which is the
single reason the study stops where it does. **If you have a free GPU, the top priority is
Llama-3.1-8B**, which needs ~16 GB in fp16 and could not be loaded at all here.

Read `docs/quantization_quality.md` for the results and what they mean. This file is only about
what is done, what is not, and how to run it.

---

## 1. What exists

`python/mixfp4/` — quantiser and Triton kernels.

| piece | file | state |
|---|---|---|
| codebooks, UE4M3 codec, nibble packing | `codebook.py` | done, cross-checked against the CUDA path |
| per-group candidate selection, two-level scale | `quant.py` | done |
| fitting methods (`rtn`, `rtn_search`, `hqq`) | `quantizers/` | done, registry — add more with `@register` |
| dynamic activation quantisation | `activation.py` | done (RTN-grade; no outlier mitigation) |
| Triton kernels (gemm / gemm_splitK / gemv) | `triton_kernels/` | correct, **never benchmarked, never used in any result below** |
| `MixFP4Linear`, `patch_model` | `linear.py` | done, survives `torch.compile(fullgraph=True)` |
| perplexity harness | `scripts/eval_perplexity.py` | done; wikitext2 + C4; W4A16 and W4A4 |
| tests | `tests/triton/test_mixfp4_triton.py` | 23 passing |

**The candidate/method grid.** Six format policies × three fitting methods. A *policy* is the set
of encodings a group may choose between; a *method* is how codes and scale are fitted once a
candidate is picked.

```
policies: nvfp4 (E2M1@6)   e2m1-4 (E2M1@4 everywhere)   e0m3 (E0M3 everywhere)
          nvfp4-46 (choose 6 or 4)   mixed (choose E2M1@6 or E0M3@7)   mixed-46 (all three)
methods:  rtn   rtn_search   hqq
```

## 2. What has been measured

Two models, both small, wikitext-2 only. Full 6×3 grids exist for:

| model | W4A16 | W4A4 | windows |
|---|---|---|---|
| facebook/opt-125m | done | done | 140 |
| Qwen/Qwen2.5-0.5B | done | done | 60 |

Headline: `mixed + 4/6` with HQQ is best under W4A4 on both models, recovering 18.7% / 18.5% of the
perplexity 4-bit costs. Every *non-adaptive* format (E2M1@4 everywhere, E0M3 everywhere) is
markedly worse than plain NVFP4, on both models, under all three methods — the value is entirely
in the per-group choice.

## 3. What is missing, in priority order

### 3.1 Llama-3.1-8B, wikitext-2 and C4 — **blocked here, do this first**

The model is cached at `/mnt/disk1/share/huggingface/hub/models--meta-llama--Llama-3.1-8B` (15 GB)
but never fit in the free VRAM. This matters more than any other gap because:

- both models measured are ≤0.5B, and **larger models are consistently more tolerant of 4-bit**,
  so the reported gains are likely upper bounds;
- Four Over Six reports its results on 1B–70B, so an 8B number is the first directly comparable
  point (their Llama-3-8B: BF16 7.54, NVFP4 RTN 8.43, +4/6 MSE 8.30);
- C4 has never been run on anything. It is wired up and smoke-tested, but every published number
  here is wikitext-only.

```bash
for ds in wikitext2 c4; do
  for act in none match; do          # none = W4A16, match = W4A4
    python scripts/eval_perplexity.py --model meta-llama/Llama-3.1-8B \
      --dataset $ds --activations $act --select-p 2.0 --logit-chunk 256 \
      --json results/llama31-8b-$ds-$act.json
  done
done
```

Four runs, 19 configs each. Budget generously: on a free 5090, W4A16 is maybe 1.5–2 h per dataset
and W4A4 considerably more, since activation quantisation runs on every layer of every forward.
Start with `--limit 40` to sanity-check timing before committing to a full run.

### 3.2 Close the loop on the kernels

**No result in this study used the Triton kernels.** Everything is simulated quantisation —
`quantize_mixfp4` → `dequantize_mixfp4` → `weight.data.copy_()`, with the matmul still in fp16.
That is the correct methodology for an accuracy table (GPTQ, AWQ and Four Over Six all do it), but
it means the kernels are validated only against synthetic tensors in the test suite, never
end-to-end.

Add a `--backend triton` flag that swaps in real `MixFP4Linear` layers and confirm it reproduces
the simulated perplexity. Two constraints:

- **W4A16 only.** The kernels are A16W4; they cannot express 4-bit activations, so no W4A4 row is
  reachable this way.
- The autotuner will dominate the first run. `python/mixfp4/configs/` is wired up but **empty** —
  populate it first (see 3.3).

### 3.3 Ship a tuned config cache, then benchmark

`MIXFP4_AUTOTUNE=fast` explores ~200 configs per distinct `(M, N, K)`, which is minutes of
compilation per shape. GemLite ships per-GPU JSON files for exactly this reason. Write a script
that sweeps a target model's shapes and calls `mixfp4.triton_kernels.config.save_config()`.

Then benchmark — **there are no performance numbers anywhere in this work.** Correctness is
measured; speed is not. Until that changes there is no evidence these kernels beat dequantising to
bf16 and calling cuBLAS.

### 3.4 Activation outlier mitigation

`activation.py` is plain dynamic RTN-style quantisation. No SmoothQuant, no random Hadamard
transform, no AWQ scaling. **Every W4A4 number here is therefore a floor, not a state-of-the-art
result**, and the gap to the paper's numbers is partly attributable to this. RHT is the cheapest
meaningful addition and the one Four Over Six itself uses.

### 3.5 Activation-aware selection

The sharpest open question. Candidates are currently chosen by per-group *weight* reconstruction
error, and there is direct evidence that is a loose proxy:

- the best `--select-p` is unstable across models and configurations, spanning up to 0.15 ppl;
- on OPT under `search`, adding a *third* candidate made things worse than the best pair — which
  cannot happen if selection were exact.

AWQ's insight is that error on channels which see large activations matters disproportionately.
Selecting the candidate by activation-weighted error, using a calibration set, addresses this
directly. Expect it to need a calibration pass in the harness.

### 3.6 Smaller items

- `--select-p` is global but should be per-method: `rtn_search` collapses at p<1 (it over-rewards
  clipping under an lp norm) while `hqq` is immune.
- Per-group candidate selection uses the *continuous* scale, before UE4M3 rounding. Small
  approximation, never quantified.
- `granule_n`/`granule_k` coarsening exists so a checkpoint stays legal for the CUDA kernel's
  instruction-level granule, but no perplexity run has used a coarse granule. Worth one run:
  the CUDA path cannot go finer than an MMA atom, so if coarsening is expensive the two backends
  cannot share a checkpoint.

---

## 4. Environment

Nothing here is in `CMakeLists.txt`; the package is pure Python and does not need the CUDA build.

```bash
python -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
./venv/bin/pip install transformers datasets        # only for scripts/eval_perplexity.py

PYTHONPATH=python ./venv/bin/python tests/triton/test_mixfp4_triton.py
MIXFP4_AUTOTUNE=default ... tests/...                # skip the full autotune sweep
```

Verified on torch 2.9.1+cu130 / Triton 3.5.1 / RTX 5090 (sm_120), Python 3.13.

## 5. Gotchas that cost time here

**Large-vocabulary OOM.** Passing `labels=` to a HF model materialises logits for the whole window;
Qwen's 152k vocab needs ~600 MB for 2048 tokens in fp16 and more for the fp32 upcast inside
cross-entropy — far more than a 0.5B model's own weights. `_window_nll` applies the LM head in
slices instead. Lower `--logit-chunk` if you still OOM.

**C4 will not load through `load_dataset`.** Its loader rebuilds the cache key from the exact
`data_files` spec that created it, which is unrecoverable afterwards, so
`load_dataset("allenai/c4", "en", ...)` fails against an existing cache. `_find_c4_arrow()` reads
the validation shard directly as an arrow file. If you are online and starting fresh this is moot.

**Set `PYTORCH_ALLOC_CONF=expandable_segments:True`** when VRAM is tight.

**`--limit N` first.** Always time a short run before a full sweep; a 19-config W4A4 grid on
Qwen-0.5B took ~70 minutes at 60 windows.

**Window counts must match to compare.** Perplexity depends on how many windows you evaluate, so a
140-window run and a 60-window run are not comparable even on the same model — each needs its own
fp16 baseline. The Qwen tables in `quantization_quality.md` carry this caveat.

**Uniform format tagging is correct by accident.** A permutation of a constant is that constant, so
an all-E2M1 or all-E0M3 checkpoint will pass tests that a genuinely mixed one fails. This is how a
real bug survived in the CUDA path. `force_format="random"` exists for exactly this reason — keep
using it in kernel tests.

## 6. Reproducing the existing numbers

```bash
# W4A16 and W4A4 full grids, both models
for m in facebook/opt-125m Qwen/Qwen2.5-0.5B; do
  for act in none match; do
    python scripts/eval_perplexity.py --model $m --activations $act \
      --select-p 2.0 --logit-chunk 256 --json results/$(basename $m)-$act.json
  done
done
```

Qwen used `--limit 60` for both of its runs. Add `--dataset c4` for the C4 variant.
