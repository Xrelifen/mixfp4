# Does mixing E2M1 and E0M3 actually help?

The CUDA work in `docs/mixed_nvfp4_report.md` answered *can a mixed-format GEMM run at full speed*
(yes, 1.0–4.9% over stock NVFP4). It never asked whether mixing is worth doing. This measures that:
wikitext-2 perplexity against ordinary NVFP4, holding everything else fixed.

**Short answer: yes, but modestly, and only if you select the format on the right objective.**
The best configuration on both models tested recovers 18–23% of the perplexity that 4-bit
quantisation costs, relative to pure NVFP4. Selecting on plain squared error — the obvious
choice — can make things *worse* than not mixing at all.

A later section adds *Four Over Six* (arXiv:2512.02010), which attacks the same weakness in E2M1
by a different route. It reaches the same −0.2 to −0.34 ppl band, and — the more interesting
result — **the two do not stack**: offering both choices per group is better than either in only
one of six cells. They are competing fixes for one defect, not complementary ones.

## Method

Simulated quantisation: each weight is round-tripped through the format (quantise → dequantise)
and the matmul still runs in fp16. That isolates what the *format* costs from what a *kernel*
costs, and makes the numbers valid for either backend, since the CUDA kernel computes the same
arithmetic on the tensor core.

Everything is held fixed except the two variables under test:

- **format policy** — which candidate encodings a group may choose between. `nvfp4` pins every
  group to E2M1 with amax on 6 (ordinary NVFP4); `e0m3` pins every group to sign-magnitude INT4;
  `mixed` chooses between those two; `nvfp4-46` chooses between mapping amax onto 6 or onto 4
  (Four Over Six, below); `mixed-46` offers all three.
- **method** — how codes and scale are fitted once a codebook is chosen: `rtn` (round to nearest at
  the amax-derived scale), `search` (sweep 8 scale multipliers, keep the best), `hqq` (HQQ-style
  half-quadratic proximal optimisation, see below).

Group size 16, UE4M3 per-group scales plus one global fp32 factor, `lm_head` left in fp16.
Wikitext-2 test, 2048-token windows, 140 windows. Reproduce with:

```bash
python scripts/eval_perplexity.py --model Qwen/Qwen2.5-0.5B --select-p 0.5
```

### The HQQ-style method

HQQ minimises `||W - dequant(quant(W))||_p` with p < 1 rather than least squares, by half-quadratic
splitting: alternate a shrinkage step on an error slack variable with a closed-form update of the
quantiser's meta-parameters, annealing the coupling upward.

Stock HQQ optimises the *zero-point* of an asymmetric affine quantiser. Both codebooks here are
symmetric sign-magnitude, so there is no zero-point — the only free meta-parameter is the per-group
scale, and the zero-point update is replaced by a least-squares scale update, `s* = <W - W_e, C> /
<C, C>`. The shrinkage operator, the annealing schedule and the defaults (p=0.7, β=10, κ=1.01,
20 iterations) are HQQ's. Calling it "HQQ" outright would overclaim; it is HQQ's machinery applied
to a different meta-parameter.

## Results

`select-p` is the exponent of the norm used to *choose* between codebooks per group.

### Qwen2.5-0.5B (fp16 baseline 13.070)

| config | ppl @ p=2.0 | vs nvfp4-rtn | ppl @ p=0.5 | vs nvfp4-rtn | E0M3 share |
|---|---|---|---|---|---|
| nvfp4-rtn | 14.367 | — | 14.367 | — | 0% |
| e0m3-rtn | 14.597 | +0.230 | 14.597 | +0.230 | 100% |
| mixed-rtn | 14.235 | −0.132 | 14.254 | −0.113 | 59% / 39% |
| nvfp4-search | 14.236 | −0.131 | 15.217 | +0.851 | 0% |
| mixed-search | 14.185 | −0.182 | 15.161 | +0.794 | 55% / 40% |
| nvfp4-hqq | 14.198 | −0.168 | 14.181 | −0.186 | 0% |
| **mixed-hqq** | 14.170 | −0.197 | **14.150** | **−0.217** | 54% / 41% |

### OPT-125m (fp16 baseline 27.656)

| config | ppl @ p=2.0 | vs nvfp4-rtn | ppl @ p=0.5 | vs nvfp4-rtn | E0M3 share |
|---|---|---|---|---|---|
| nvfp4-rtn | 29.151 | — | 29.151 | — | 0% |
| e0m3-rtn | 29.659 | +0.508 | 29.659 | +0.508 | 100% |
| mixed-rtn | 29.274 | **+0.123** | 29.223 | **+0.072** | 58% / 38% |
| nvfp4-search | 29.011 | −0.140 | 29.986 | +0.835 | 0% |
| mixed-search | 29.113 | −0.038 | 29.626 | +0.475 | 53% / 38% |
| nvfp4-hqq | 28.830 | −0.321 | 28.926 | −0.225 | 0% |
| **mixed-hqq** | 29.017 | −0.134 | **28.814** | **−0.337** | 52% / 40% |

## What the numbers say

**1. E0M3 alone is always worse than NVFP4** — by 0.23 (Qwen) and 0.51 (OPT). It is not a better
format. Its value is entirely in being *complementary*: a group whose weights are near-uniformly
distributed is served badly by E2M1's log spacing, and those groups are common enough to matter.

**2. Mixing helps, but it is the smaller of the two levers.** On OPT, the optimiser is worth
−0.32 and mixing at most −0.13. On Qwen they are comparable (−0.19 vs −0.13). They **compose**:
`mixed-hqq` is the best configuration on both models at p=0.5, and the best at p=2.0 on Qwen.

**3. Squared error is the wrong selection objective.** This is the sharpest result here. At p=2.0
the selector picks E0M3 for 52–59% of groups and, on OPT, that makes perplexity *worse than not
mixing* (+0.12 for `mixed-rtn`). Dropping the selection exponent to 0.5 moves the share to 38–41%
and flips the sign. Lower weight-reconstruction error does not imply lower perplexity — the same
observation that motivated GPTQ, AWQ and HQQ in the first place, showing up here in the choice
between codebooks rather than in the choice of codes.

**4. `select-p` must be per-method, not global.** `rtn_search` collapses at p=0.5 (+0.85 on Qwen,
+0.84 on OPT) because a pure scale search under an lp<1 objective over-rewards clipping: the norm
barely penalises a few badly-clipped outliers, so the search drives the scale down until they are
destroyed. HQQ is immune because its slack variable absorbs exactly those outliers. Currently one
`--select-p` applies to everything, which is why the search rows are unusable at 0.5; it should be
a per-method default.

**Fraction of the quantisation gap recovered** by `mixed-hqq @ p=0.5` versus `nvfp4-rtn`:
Qwen 1.296 → 1.080 (**17%**), OPT 1.495 → 1.159 (**23%**).

## Four Over Six (arXiv:2512.02010)

Cook et al. make a closely related observation about E2M1 alone. Its step sizes are 0.5 below 2,
1 up to 4, then **2** up to 6, so the top of the range is where resolution is worst: a value that
should land near 5 has nowhere to go. *Four Over Six* maps a block's amax onto **4** instead of 6,
giving up ±5 and ±6 in exchange for uniform spacing over what remains — and, crucially, chooses
between the two caps **per block** by quantisation error. Capping every block at 4 is worse than
never doing it (their Table 3, reproduced below).

That is the same machinery mixfp4 already uses, with a different pair of candidates, so it drops
into the same framework as a third one. It also needs no format support at all: the per-group scale
is stored explicitly, so a block capped at 4 simply stores `amax/4` and nothing downstream needs to
know. This implementation also sidesteps the paper's §3.1 caveat — it derives the global normaliser
from the per-group scales actually selected, so the largest lands on 448 by construction whatever
cap produced it, and the paper's forced drop of `M_FP8` from 448 to 256 is unnecessary.

Reproduced on synthetic data before evaluating. The paper's Table 1 example `[10, 20, 30, 40]`
is lossless at M=4 and not at M=6 (30 → 26.67), and their Table 3 finding holds: capping *all*
blocks at 4 is worse than standard NVFP4 (0.0983 vs 0.0951 relative Frobenius, gaussian), while
adaptive selection beats both (0.0879).

### Perplexity, both models, all three selection rules

`46` is Four Over Six; `mixed46` offers all three candidates per group (E2M1@6, E2M1@4, E0M3@7).
Best per column in bold.

**Qwen2.5-0.5B** — fp16 13.070, nvfp4-rtn 14.367

| config | p=2.0 (MSE) | p=1.0 (MAE) | p=0.5 |
|---|---|---|---|
| mixed-rtn | 14.235 | 14.216 | 14.254 |
| 46-rtn | 14.329 | 14.278 | 14.273 |
| mixed46-rtn | 14.245 | 14.187 | 14.225 |
| nvfp4-hqq | 14.198 | 14.177 | 14.181 |
| mixed-hqq | 14.170 | **14.135** | 14.150 |
| 46-hqq | 14.250 | 14.178 | **14.137** |
| mixed46-hqq | **14.147** | 14.192 | 14.169 |

**OPT-125m** — fp16 27.656, nvfp4-rtn 29.151

| config | p=2.0 (MSE) | p=1.0 (MAE) | p=0.5 |
|---|---|---|---|
| mixed-rtn | 29.274 | 29.165 | 29.223 |
| 46-rtn | 29.149 | 29.277 | 29.283 |
| mixed46-rtn | 29.248 | 28.995 | 29.215 |
| nvfp4-hqq | **28.830** | 28.829 | 28.926 |
| mixed-hqq | 29.017 | 28.886 | **28.814** |
| 46-hqq | 28.930 | **28.815** | 28.991 |
| mixed46-hqq | 28.995 | 28.942 | 29.033 |

### What this says

**4/6 works, and is worth about as much as E0M3 mixing** — best case −0.34 (OPT, `46-hqq` at p=1.0)
and −0.23 (Qwen, `46-hqq` at p=0.5), against −0.34 and −0.23 for `mixed-hqq`. Two independent
answers to the same question, landing in the same place.

**The two do not stack.** `mixed46`, which offers all three candidates, beats the best two-way
option in exactly one of six model×rule cells. That is the expected outcome once you see what both
are doing: E2M1@4 and E0M3@7 are both fixes for E2M1's coarse top end, so they compete for the same
groups rather than composing. Offering both mostly splits the vote — the E0M3 share drops from
~54% to ~47% when E2M1@4 is available, and accuracy does not improve.

**No selection rule wins consistently.** The paper reports MSE best for its setting; here the best
rule is p=2.0, p=1.0 and p=0.5 depending on model and configuration, spanning up to 0.15 ppl. This
is an unstable hyperparameter, and the instability is itself evidence that per-group reconstruction
error — under any exponent — is a loose proxy for what actually matters.

**These gains are much smaller than the paper's** (they report −0.43 on Llama-3-1B). The likely
reason is regime, not implementation: **the paper evaluates W4A4, this evaluates weight-only
W4A16.** Their Figure 2 attributes NVFP4's degradation to near-maximal values in activations *and*
weights, and quantising activations is where values near 5 are most damaging. Weight-only removes
half of the problem the method was designed for, so the small gains here are not evidence against
it. Their models are also 1B–70B rather than 0.1–0.5B.

## Caveats

- Two small models (125M, 0.5B). The GPU available during this work had ~2 GB free, which set the
  ceiling. Behaviour at 7B+ is unmeasured and the literature suggests larger models are *more*
  tolerant of 4-bit weights, so these gains may shrink.
- One dataset (wikitext-2) and one metric. No downstream task evaluation.
- **Weight-only (W4A16) throughout.** Four Over Six targets W4A4, where its own analysis locates
  most of the damage; measuring it here understates it. Activation quantisation is not implemented.
- No calibration data anywhere. The obvious next step is activation-aware selection — AWQ's
  observation that error on channels which see large activations matters disproportionately.
  Selecting the codebook by activation-weighted error, rather than by weight error, directly
  addresses finding 3 and is the most promising untried lever.
- The format decision is made against the *continuous* scale, before it is rounded to UE4M3. The
  perturbation is small enough not to reorder the choice in practice, but it is an approximation.

## Adding a method

`python/mixfp4/quantizers/` is a registry. A method receives grouped weights and one codebook id,
and returns codes, a per-group scale, a per-group error, and the target the codes should be
re-derived from if the scale later moves:

```python
@register("mine")
def fit_mine(grouped, fmt, cfg) -> FitResult: ...
```

Returning `target` separately is what lets HQQ work: its codes are fitted to a slack-corrected
`W - W_e`, and re-deriving them from `W` after the scale is rounded would silently discard the
optimisation.
