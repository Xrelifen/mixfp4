# Does mixing E2M1 and E0M3 actually help?

The CUDA work in `docs/mixed_nvfp4_report.md` answered *can a mixed-format GEMM run at full speed*
(yes, 1.0–4.9% over stock NVFP4). It never asked whether mixing is worth doing. This measures that:
wikitext-2 perplexity against ordinary NVFP4, holding everything else fixed.

**Short answer: yes — and the answer depends heavily on whether activations are also quantised.**

Under **W4A4**, which is the regime that matters and the one *Four Over Six* (arXiv:2512.02010)
targets, the best configuration is **all three candidates per group plus HQQ-style fitting**
(`mixed46-hqq`): −0.54 ppl on OPT-125m and −0.47 on Qwen2.5-0.5B against ordinary NVFP4, with an
almost identical ranking on both models. Jump to [W4A4](#w4a4--and-why-the-weight-only-conclusions-above-are-the-wrong-ones).

Under **W4A16** (weight-only) every effect is roughly half the size, and two conclusions that
section reaches do not survive the move to W4A4:

- that mixing can be *worse* than not mixing — an artifact of weight-only; both such rows flip
  sign under W4A4;
- that E0M3 mixing and Four Over Six **do not stack** — they do stack under W4A4, in all four
  cells tested.

The weight-only sections are kept below because they are what motivated the W4A4 run, and because
the contrast between the two regimes is the most useful thing measured here.

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
Wikitext-2 test, 2048-token windows (140 for W4A16 and for OPT under W4A4; 60 for Qwen under
W4A4, to fit the time budget — internally consistent either way). Reproduce with:

```bash
python scripts/eval_perplexity.py --model Qwen/Qwen2.5-0.5B --select-p 0.5           # W4A16
python scripts/eval_perplexity.py --model Qwen/Qwen2.5-0.5B --activations match      # W4A4
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

## Results — W4A16 (weight-only)

Kept for the contrast with W4A4 below; see the caveat in the summary above before drawing
conclusions from this section. `select-p` is the exponent of the norm used to *choose* between
candidates per group.

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

## W4A4 — and why the weight-only conclusions above are the wrong ones

Everything to this point is weight-only (W4A16). That turns out to be the wrong regime to judge
any of these methods in, including Four Over Six, whose own Figure 2 locates most of NVFP4's
damage in *activations*.

Activations are quantised dynamically, per token, in groups of 16 along K — structurally identical
to weights grouped per output channel along K, so the same fitting path is reused. `--activations
match` ties the activation policy to the weight policy, which is how 4/6 is meant to be used (its
Figure 3 puts `Q(4/6)` on the activation path too).

Where the error comes from, on OPT-125m:

| | ppl | vs fp16 |
|---|---|---|
| fp16 | 27.656 | — |
| W16A4 (activations only) | 28.785 | +1.130 |
| W4A16 (weights only) | 29.151 | +1.495 |
| W4A4 (both) | 30.553 | +2.897 |

The two sources are close to additive (1.130 + 1.495 = 2.624 against an observed 2.897), so
weight-only was measuring roughly half the problem.

### OPT-125m, W4A4, matched activations, MSE selection

fp16 row is W16A4 (28.785); `vs nvfp4-rtn` is against 30.553.

| config | ppl | vs nvfp4-rtn | same figure under W4A16 |
|---|---|---|---|
| e0m3-rtn | 31.268 | +0.715 | +0.508 |
| mixed-rtn | 30.349 | **−0.204** | **+0.123** ← sign flip |
| 46-rtn | 30.393 | −0.159 | −0.002 |
| mixed46-rtn | 30.315 | **−0.238** | **+0.098** ← sign flip |
| nvfp4-hqq | 30.172 | −0.381 | −0.320 |
| mixed-hqq | 30.071 | −0.481 | −0.134 |
| 46-hqq | 30.069 | −0.483 | −0.221 |
| **mixed46-hqq** | **30.010** | **−0.543** | −0.156 |

### Qwen2.5-0.5B, W4A4, matched activations, MSE selection

60 windows rather than 140, to fit the time budget; internally consistent, so the `vs nvfp4-rtn`
column is comparable across rows. The fp16 row is W16A4 (13.302); `vs nvfp4-rtn` is against 14.837.

| config | ppl | vs nvfp4-rtn | same figure under W4A16 |
|---|---|---|---|
| e0m3-rtn | 15.940 | +1.104 | +0.230 |
| 46-rtn | 14.716 | −0.121 | −0.038 |
| mixed-rtn | 14.496 | −0.341 | −0.132 |
| mixed46-rtn | 14.485 | −0.352 | −0.122 |
| nvfp4-hqq | 14.620 | −0.217 | −0.168 |
| 46-hqq | 14.557 | −0.280 | −0.116 |
| mixed-hqq | 14.403 | −0.434 | −0.197 |
| **mixed46-hqq** | **14.366** | **−0.471** | −0.220 |

Every method is worth roughly twice as much under W4A4 as under W4A16, and the ordering is
identical on both models.

### Two findings above are now corrected

**Correction 1: mixing is not sometimes-harmful. That was an artifact of weight-only.** Under
W4A16, `mixed-rtn` was *worse* than plain NVFP4 on this model (+0.123), which drove the earlier
conclusion that squared-error selection can hurt. Under W4A4 the same configuration, the same
selection rule and the same weights give −0.204. Both mixed rows flip sign.

**Correction 2: 4/6 and E0M3 mixing do stack — under W4A4.** The earlier section concluded they
were competing fixes for one defect, because the three-way `mixed46` beat the best two-way option
in only one of six cells. Under W4A4 it stacks in *both* cells tested:

| | 3-way | best 2-way | |
|---|---|---|---|
| W4A16, rtn | 29.248 | 29.149 | does not stack |
| W4A16, hqq | 28.995 | 28.930 | does not stack |
| W4A4, rtn | 30.315 | 30.349 | **stacks** |
| W4A4, hqq | 30.010 | 30.069 | **stacks** |

and on Qwen too, in both cells (rtn 14.485 vs 14.496; hqq 14.366 vs 14.403).

A plausible reading, offered as a hypothesis rather than a demonstrated mechanism: with
`--activations match`, the per-group choice now also runs over *activation* tensors, whose
distributions differ markedly from weights'. A wider candidate set covers a wider range of block
shapes, so the third candidate earns its place where against weights alone it merely split the vote
(the E0M3 share falls from ~54% to ~47% when E2M1@4 is available, in both regimes).

**Unchanged: E0M3 alone is still the worst option**, and by more than before (+0.715 vs +0.508).
Its value remains entirely in being available as a choice.

### Which is best

`mixed46-hqq` — all three candidates per group, HQQ-style fitting — is first on **both** models,
and the full ranking is nearly identical across them:

| rank | OPT-125m | | Qwen2.5-0.5B | |
|---|---|---|---|---|
| 1 | mixed46-hqq | −0.543 | mixed46-hqq | −0.471 |
| 2 | 46-hqq | −0.483 | mixed-hqq | −0.434 |
| 3 | mixed-hqq | −0.481 | mixed46-rtn | −0.352 |
| 4 | nvfp4-hqq | −0.381 | mixed-rtn | −0.341 |
| 5 | mixed46-rtn | −0.238 | 46-hqq | −0.280 |
| 6 | mixed-rtn | −0.204 | nvfp4-hqq | −0.217 |
| 7 | 46-rtn | −0.159 | 46-rtn | −0.121 |
| — | e0m3-rtn | +0.715 | e0m3-rtn | +1.104 |

Taking one lever at a time on top of `nvfp4-rtn`, the three are not equal:

| lever | OPT | Qwen |
|---|---|---|
| HQQ-style fitting | −0.381 | −0.217 |
| E0M3 mixing | −0.204 | −0.341 |
| Four Over Six | −0.159 | −0.121 |

**E0M3 mixing beats Four Over Six on both models**, and 4/6 is the weakest of the three
individually. But that ignores cost, and on cost the ordering inverts: 4/6 needs *no format
support at all* — no tag bit, no second codebook in the kernel, just a different scale — whereas
E0M3 mixing is what forced the entire dispatch-tree architecture in `mixed_nvfp4_report.md`. For
roughly 40–70% of E0M3's benefit at zero kernel cost, 4/6 is the better deal for anyone not
already committed to a mixed-format kernel. For this repository, which *is* committed, the answer
is to use both: they stack.

## Appendix: the full grid

Every candidate policy against every fitting method, W4A4, matched activations, MSE selection.
`vs 4bit` is against NVFP4 RTN; `recov` is the share of the 4-bit cost undone.

### OPT-125m — 140 windows

| baseline | ppl | vs fp16 |
|---|---|---|
| W16A16 (fp16) | 27.656 | — |
| W16A4 (activations only) | 28.785 | +1.130 |
| W4A4 NVFP4 RTN (reference) | 30.553 | +2.897 |

| candidates per group | RTN ppl | recov | search ppl | recov | HQQ ppl | recov |
|---|---|---|---|---|---|---|
| E2M1@6 — plain NVFP4 | 30.553 | 0.0% | 30.415 | 4.8% | 30.172 | 13.1% |
| E2M1@4 — *all* blocks | 31.146 | −20.5% | 31.077 | −18.1% | 31.106 | −19.1% |
| E0M3 — *all* blocks | 31.268 | −24.7% | 31.200 | −22.3% | 31.013 | −15.9% |
| 4/6 adaptive | 30.393 | 5.5% | 30.074 | 16.5% | 30.069 | 16.7% |
| mixed E2M1/E0M3 | 30.349 | 7.0% | 30.166 | 13.3% | 30.071 | 16.6% |
| **mixed + 4/6** | 30.315 | 8.2% | 30.208 | 11.9% | **30.010** | **18.7%** |

Three things only the full grid shows:

**Neither alternative format is better than NVFP4 on its own.** Capping every block at 4 loses
20%, and E0M3 everywhere loses 25% — under all three fitting methods. All the value is in having a
per-group *choice*; this reproduces Four Over Six's Table 3 independently, and extends the same
conclusion to E0M3.

**Every adaptive row beats NVFP4, at every method level.** Six for six.

**For 4/6, a cheap scale search buys almost all of HQQ's benefit** — 16.5% against 16.7%, for
eight multiplier evaluations rather than twenty annealed iterations. Worth knowing when
quantisation time is a constraint.

### Qwen2.5-0.5B — 60 windows

| baseline | ppl | vs fp16 |
|---|---|---|
| W16A16 (fp16) | 12.293 | — |
| W16A4 (activations only) | 13.302 | +1.009 |
| W4A4 NVFP4 RTN (reference) | 14.837 | +2.544 |

| candidates per group | RTN ppl | recov | search ppl | recov | HQQ ppl | recov |
|---|---|---|---|---|---|---|
| E2M1@6 — plain NVFP4 | 14.837 | 0.0% | 14.676 | 6.3% | 14.620 | 8.5% |
| E2M1@4 — *all* blocks | 15.947 | −43.6% | 15.771 | −36.7% | 16.049 | −47.6% |
| E0M3 — *all* blocks | 15.940 | −43.4% | 16.102 | −49.7% | 16.000 | −45.7% |
| 4/6 adaptive | 14.716 | 4.7% | 14.486 | 13.8% | 14.557 | 11.0% |
| mixed E2M1/E0M3 | 14.496 | 13.4% | 14.409 | 16.8% | 14.403 | 17.1% |
| **mixed + 4/6** | 14.485 | 13.8% | **14.341** | **19.5%** | 14.366 | 18.5% |

### Across both grids

**`mixed + 4/6` is the best policy on both models** — the top cell of each grid is one of its rows.
Offering all three candidates per group wins.

**Neither alternative format is usable on its own, on either model**, and Qwen is far more brutal
about it: −37% to −50% for the non-adaptive rows, against −16% to −25% on OPT.

**Every adaptive row beats NVFP4 at every method level** — twelve for twelve across both grids.

**HQQ is the method to pick, though not always the winner.** On `mixed + 4/6` it gives 18.7% (OPT)
and 18.5% (Qwen) — remarkably consistent. `search` gives 19.5% on Qwen but only 11.9% on OPT, so
it is higher-variance despite occasionally winning.

**A non-monotonicity, on one model only.** Under `search` on OPT, adding the third candidate is
*worse* than the best pair — 11.9% against 16.5% for `4/6` alone. A wider menu cannot hurt if
selection were exact, so this is evidence that per-group reconstruction error is a loose proxy for
end-to-end loss. It does **not** reproduce on Qwen, where the same cell is the best in the grid
(19.5%), so it is one data point rather than a general law — but it is the same underlying problem
as the `select-p` instability noted earlier.

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
