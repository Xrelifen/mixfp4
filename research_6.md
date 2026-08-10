# research_6.md
# Research-6 — Coupled Coarse-Format Assignment + Selective Format-Control Granularity
## Paper-Oriented Program for N8/N16 Control Granularity, Sparse N8 Refinement, and Top-Tier Closure

**SPEC_VERSION:** `research_6_v1.0_2026-08-10`  
**Program status:** `BOUNDED_CONTINUE / SELECTIVE_GRANULARITY_DIAGNOSTIC_PENDING`  
**Primary paper scope:** low-bit LLM weight quantization under hardware-oriented coarse format-control granularity  
**Base format-control point:** `N8×K64`  
**Coarser control point:** `N16×K64`  
**Scale granularity:** `K16`  
**Candidate formats:** `E2M1` vs project-defined `E0`  
**Established optimizer:** `FullLayer-CD2`, exactly two bounded coupled coordinate sweeps  
**Development families:** Llama-3.1-8B, Qwen3-8B, Mistral-7B-v0.3  
**Primary development corpora:** WikiText-2 and fixed C4 evaluation slice  
**Primary selective-granularity environment:** W4A16  
**Secondary deployment environment:** W4A4  
**Cross-domain extension:** SANA-1.6B, optional and gated  
**Hardware status:** accuracy-side work only; independent hardware cost evidence required for strong systems claim  
**Native SM120 status:** `WAIT_FOR_SM120`  
**Research standard:** ICLR/NeurIPS-level falsifiability, generalization, statistics, and reviewer-oriented closure  
**Core rule:** Research-1~5 and matched-completion artifacts are immutable evidence

---

# 0. Executive position

Research-6 is a new bounded research program built on completed Research-1~5.

It does not reopen Research-5 A/B/C.

It does not redefine prior negative results.

The current scientific story is:

```text
fine K16 format preferences exist
↓
hardware-oriented format-control granularity compresses many preferences
into fewer legal format decisions
↓
independent/local selection loses quality
↓
errors from different K64 regions interact after linear projection
↓
coupled N8 assignment (CD2) recovers substantial loss
↓
coarsening N8 → N16 further compresses format control
↓
N16 accuracy becomes model/corpus/quantization-regime dependent
↓
N8-child conflicts are common, but merge regret is highly heterogeneous
```

This motivates two paper tracks.

---

## Track A — Base paper

Working theme:

> **Coupled Coarse-Format Assignment under Hardware Format-Control Granularity**

Track A consists of:

```text
fine-to-coarse decision-compression evidence
+
Raw / OutputAware / CD2 progression
+
cross-K coupled-assignment mechanism
+
strong N8-CD2 accuracy
+
matched N8-vs-N16 granularity boundary
+
W4A4 robustness / Activation-4Over6 compatibility
+
prospective family
+
proper downstream
+
baseline/novelty audit
+
hardware/control-cost accounting
```

Track A does not depend on Selective Granularity succeeding.

---

## Track B — Stronger extension

Working theme:

> **Selective Format-Control Granularity**

Default:

```text
N16×K64:
  one shared E2/E0 format decision
```

Exception:

```text
selected N16 regions:
  split into two N8×K64 child regions
  each child receives its own E2/E0 format decision
```

Goal:

> Recover most of the N8 quality benefit while using N16 control for most regions.

Track B is the highest-upside new contribution.

It must first prove that a useful sparse-refinement Pareto exists.

If not:

```text
SELECTIVE_GRANULARITY_NO_HEADROOM
```

and Track B closes.

Track A is then evaluated independently.

---

# 1. Authority and precedence

Use:

```text
CURRENT research_6.md
    >
CURRENT coding_agent_prompt_6.md
    >
immutable Research-5 final/matched-completion artifacts
    >
immutable Research-4 artifacts
    >
immutable Research-3 artifacts
    >
immutable Research-2 artifacts
    >
immutable Research-1 artifacts
    >
Research-6 suggestion documents
    >
older informal summaries
```

Research-1~5 must not be rewritten.

Expected historical roots may include:

```text
artifacts/research_1/
artifacts/research_2/
artifacts/research_3/
artifacts/research_4/
artifacts/research_5/
```

Research-6 must create a new root:

```text
artifacts/research_6/
```

If existing project layout differs, record the actual paths in the source manifest.

Never infer prior values from narrative text when exact raw artifacts exist.

---

# 2. Mandatory source audit before new experimentation

Before modifying quantization code:

1. locate the final Research-3 artifacts;
2. locate the final Research-4 artifacts;
3. locate the final Research-5 / matched N8-N16 completion artifacts;
4. hash all source manifests and final decision files;
5. regenerate the key tables entering Research-6.

At minimum regenerate:

```text
Research-3:
  N8-CD2 Llama/Qwen Wiki/C4
  Mistral Wiki/C4
  W4A4 N8 rows where applicable

Research-5 matched completion:
  W4A16 N8 vs N16
  W4A4 Fixed-E2 A0 N8 vs N16
  W4A4 Activation-4Over6 A1 N8 factorial
  N8-child conflict statistics
  local merge-regret summaries
  paired NLL/bootstrap results
```

Create:

```text
artifacts/research_6/00_environment/source_artifact_manifest.csv
artifacts/research_6/00_environment/source_hashes.json
artifacts/research_6/00_environment/prior_result_reconstruction.md
```

Do not start Selective Granularity if the historical endpoints cannot be reproduced.

---

# 3. Immutable prior scientific conclusions

## 3.1 Research-5 A/B/C remains closed

Research-5 diagnostic restart concluded:

```text
COARSE_FORMAT_RESTART_NO_GO
confidence = HIGH
```

Closed branches include:

```text
reliability-only rescue
selective module applicability as formulated in Research-5
first-order / Fisher / teacher-KL surrogate rescue
Research-4 adaptive-scale rescue
```

Research-6 is not a retry of those methods.

Do not reopen them without a new mechanistic observation and a new written spec.

---

## 3.2 N8-CD2 is a strong established baseline

Latest matched W4A16 PPL entering Research-6:

| Model | Corpus | Fixed-E2 | 4Over6 | CD2-N8 | CD2-N16 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | Wiki | 6.623027 | 6.603869 | **6.555580** | 6.580924 |
| Llama-3.1-8B | C4 | 9.558380 | 9.494530 | **9.476854** | 9.487519 |
| Qwen3-8B | Wiki | 9.912184 | 9.854119 | **9.831382** | 9.859444 |
| Qwen3-8B | C4 | 13.812383 | 13.769569 | **13.728875** | 13.822827 |
| Mistral-7B-v0.3 | Wiki | 5.665597 | 5.477368 | 5.505075 | **5.440835** |
| Mistral-7B-v0.3 | C4 | 8.029211 | 8.009194 | **7.998210** | 8.007381 |

Interpretation:

```text
N8-CD2 is strong on the main matrix.
N16 does not behave like uniform extra noise.
Mistral Wiki is a known sign-reversal / hard case.
```

Do not hide Mistral Wiki.

---

## 3.3 Global N16 is not accuracy-equivalent to N8

Matched completion conclusion:

```text
N16_MODEL_DEPENDENT
```

W4A16:

```text
CD2-N16 is worse than CD2-N8 in 5/6 point estimates,
with the positive-regression settings supported by paired statistics.
```

Fixed-E2M1 W4A4:

```text
CD2-N16 is worse than CD2-N8 in all six point estimates.
```

Therefore do not claim:

```text
N16 is a safe replacement for N8
```

and do not claim:

```text
N16 always fails
```

Both are false.

---

## 3.4 N8-child format conflicts are common

Frozen N8 child disagreement inside proposed N16 regions:

```text
Llama   ~43.04%
Qwen    ~46.00%
Mistral ~42.71%
```

Thus N16 frequently removes an N8 format degree of freedom.

Conflict frequency alone is not enough to explain quality.

---

## 3.5 Merge regret is highly non-uniform

Latest local merge-regret summaries:

| Model | Corpus | Mean | p90 | p99 |
|---|---|---:|---:|---:|
| Llama | Wiki | 0.001607 | 0.000839 | 0.023012 |
| Llama | C4 | 0.002914 | 0.001322 | 0.041424 |
| Mistral | Wiki | 0.001441 | 0.001770 | 0.027408 |
| Mistral | C4 | 0.002270 | 0.002714 | 0.043528 |
| Qwen | Wiki | 0.085989 | 0.008574 | 0.352228 |
| Qwen | C4 | 0.151786 | 0.013845 | 0.584346 |

This is descriptive evidence for heterogeneity.

It is not proof that a merge-regret selector improves end-to-end NLL.

Research-4/5 already demonstrated that a reconstruction surrogate can improve while NLL
regresses.

Therefore Research-6 must validate every selective-granularity claim with end-to-end NLL.

---

## 3.6 W4A4 evidence is supportive, not the novelty axis

Existing Activation-4Over6 results show:

```text
Activation-4Over6 improves most tested W4A4 point estimates.
N8-CD2 + Activation-4Over6 is strong across the completed matrix.
CD2 × activation interaction is not uniformly positive.
```

Therefore Activation-4Over6 should be positioned as:

```text
strong activation baseline / deployment robustness evidence
```

not:

```text
new Research-6 activation contribution
```

---

# 4. Core problem formulation

Research-6 studies:

```text
format representation
vs
format-control granularity
```

These are different.

Format representation:

```text
E2M1
project E0
```

Scale granularity:

```text
K16
```

Format-control granularity:

```text
N8×K64
N16×K64
or a selective mixture of the two
```

The central paper-level problem is:

> Many fine representation preferences must be compressed into a smaller number of legal
> hardware control decisions. This decision compression introduces heterogeneous quality
> loss. Coupled assignment recovers some loss, and selective control granularity may retain
> fine decisions only where coarse sharing is costly.

Coordinate descent is a solver.

It is not the novelty headline.

---

# 5. Novelty boundary

Do not claim novelty as:

```text
coordinate descent
greedy residual update
layer-wise reconstruction
mixed FP4/INT4 formats
blockwise mixed numeric formats
adaptive scale selection
generic layer sensitivity
generic selective quantization
```

At minimum, submission-time positioning must audit:

```text
CDQuant
BlockDialect
Four Over Six
Adaptive Block-Scaled Data Types / IF4
MixFP4
AdaMX if available/relevant
current NVFP4 PTQ work
```

The intended novelty boundary is:

```text
hardware-oriented format-decision compression
+
format-control granularity as an explicit accuracy axis
+
cross-K coupled assignment of coarse format bits
+
matched N8/N16 boundary
+
selective allocation of N8 control inside an N16-default representation
```

If Track B succeeds, the strongest potential contribution becomes:

> Spatially selective format-control granularity within one quantized model, rather than
> globally choosing one control granularity for the full tensor/model.

Before submission, perform a fresh literature audit.

Do not rely on this 2026-08 planning snapshot as proof of novelty.

---

# 6. Frozen quantization semantics

## 6.1 Weight tensor

\[
W\in\mathbb R^{N\times K}.
\]

## 6.2 Candidate formats

```text
E2M1
project-defined E0
```

Use the exact frozen codebooks and fake/reference semantics from prior work.

Do not imply project E0 equals a verified native Blackwell datatype.

## 6.3 Scale granularity

```text
K16
```

Scale grouping and legal scale semantics remain unchanged.

## 6.4 N8 control

```text
one N8×K64 region
-> one E2/E0 format decision
```

## 6.5 N16 control

```text
one N16×K64 region
-> one E2/E0 format decision
```

One N16 region equals:

```text
two adjacent N8 row regions
```

forced to share the format decision.

This is a quantization/control abstraction, not an MMA instruction definition.

## 6.6 Activation

Primary Selective Granularity:

```text
W4A16
```

Secondary deployment:

```text
W4A4 A0 = Fixed-E2M1 / NVFP4-compatible activation
W4A4 A1 = Activation-4Over6 using the previously frozen project semantics
```

No activation E2/E0 mixing.

No activation CD.

---

# 7. Established N8/N16 coupled objective

For one N8 stripe and K64 coordinate \(k\):

\[
E_{k}^{f}
=
X_{K_k}
\left(
Q_f(W_{N8,K_k})-W_{N8,K_k}
\right)^T.
\]

For format map \(F\):

\[
J(F)
=
\left\|
\sum_kE_k^{f_k}
\right\|_F^2.
\]

For N16, the same definition uses 16 output rows.

Frozen CD2:

```text
matched OutputAware/local initialization
exactly 2 coordinate sweeps
accept candidate iff full stripe objective decreases
deterministic tie behavior
```

Research-6 must not add:

```text
CD3
CD4
until convergence
random order banks
new scale policies
new codebooks
```

---

# 8. Research-6 Track A and Track B are separate decision axes

Track A can reach:

```text
SUBMIT_READY_BASE
```

even if Track B returns:

```text
SELECTIVE_GRANULARITY_NO_HEADROOM
```

Track B can strengthen the paper to:

```text
STRONG_TOP_TIER_CANDIDATE
```

if the selective Pareto, prospective generalization, and hardware motivation are strong.

Do not collapse:

```text
base-method status
selective-granularity status
diffusion status
hardware status
native status
```

into one label.

---

# 9. Phase 0 — Research-6 initialization

Create an isolated Research-6 branch/worktree.

Record:

```text
repo URL
branch
HEAD
submodule SHAs
dirty state
Research-1~5 result hashes
local diff
```

Create:

```text
artifacts/research_6/00_environment/
```

with:

```text
spec_acknowledgement.md
spec_manifest.json
source_artifact_manifest.csv
source_hashes.json
repo_manifest.json
patch_manifest.json
environment.txt
gpu_usage_log.jsonl
literature_snapshot.md
```

No push unless explicitly requested.

---

# 10. Phase 0A — baseline endpoint reproduction

Before implementing mixed granularity:

Reproduce exact frozen anchors for at least:

```text
one Llama N8 row
one Llama N16 row
one Qwen N8 row
one Qwen N16 row
```

Prefer exact format-map hash and per-sequence NLL reproduction.

Required:

```text
CD2-N8
CD2-N16
```

If endpoint reproduction materially disagrees:

```text
RESEARCH6_ENDPOINT_REPRODUCTION_FAIL
```

Stop new selective work until resolved.

---

# 11. Phase 0B — complete missing W4A4 N16 + Activation-4Over6 rows

This is mandatory bounded characterization.

The previous gate did not unlock these rows.

Research-6 explicitly runs them.

Label:

```text
post_hoc_N16_A1_characterization
```

not:

```text
prospective
```

---

## 11.1 Required new rows

Weights:

```text
Raw-N16K64
CD2-N16K64
```

Activation:

```text
A1 = Activation-4Over6
```

Models:

```text
Llama-3.1-8B
Qwen3-8B
Mistral-7B-v0.3
```

Corpora:

```text
WikiText-2
C4
```

Total:

```text
2 × 3 × 2 = 12 new rows
```

Reuse matching N8+A1 anchors if exact hashes/configs match.

Do not retune weight maps.

---

## 11.2 Required complete W4A4 N8/N16 table

Produce:

| Model | Corpus | Raw N8+A0 | CD2 N8+A0 | Raw N16+A0 | CD2 N16+A0 | Raw N8+A1 | CD2 N8+A1 | Raw N16+A1 | CD2 N16+A1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Also report paired NLL/bootstrap CIs.

Questions:

```text
Does A1 shrink or enlarge the N16-vs-N8 gap?
Does A1 rescue Qwen C4 N16?
Does Mistral Wiki retain/reverse its qualitative behavior?
Does CD2-N16 improve Raw-N16 under A1 in all six settings?
Is N8-CD2+A1 still the best W4A4 point?
```

This characterization does not change old preregistered decisions.

---

# 12. Phase 1 — Selective Granularity headroom study

This is the highest-information Track-B experiment.

Do not build a learned selector first.

Primary question:

> Is N16 accuracy loss concentrated enough that sparse N8 refinement can recover it?

Primary environment:

```text
W4A16
```

Development models:

```text
Llama
Qwen
Mistral
```

Corpora:

```text
Wiki
C4
```

---

# 13. Selective representation

For each N16×K64 region \(r\):

\[
s_r\in\{0,1\}.
\]

Meaning:

```text
s_r = 0:
  unsplit
  one shared N16 E2/E0 format decision

s_r = 1:
  split
  top N8 child gets one E2/E0 decision
  bottom N8 child gets one E2/E0 decision
```

No finer-than-N8 control.

No new scale state.

No new payload datatype.

---

# 14. Refinement budget

Let:

```text
R = number of N16 regions
S = number of split N16 regions
```

Define:

\[
B=\frac{S}{R}.
\]

Primary budget grid:

```text
B ∈ {0%, 1%, 2%, 5%, 10%, 20%, 50%, 100%}
```

Endpoints:

```text
0%   = global CD2-N16
100% = global CD2-N8
```

No dense budget sweep unless a later paper figure needs one after the method is frozen.

---

## 14.1 Accuracy-side control-count proxy

Ignoring the unresolved exception-map encoding:

```text
global N16 format decisions = R
global N8 format decisions  = 2R
selective decisions         = R + S
```

Thus an idealized decision-count ratio relative to global N8 is:

\[
C_{\text{decision}}
=
\frac{R+S}{2R}
=
\frac{1+B}{2}.
\]

Examples:

```text
B=0%   -> 50% of N8 decision count
B=10%  -> 55%
B=20%  -> 60%
B=100% -> 100%
```

This is **not hardware cost**.

It ignores:

```text
split-mask/exception encoding
routing
decoder complexity
fanout
area
power
timing
```

Label it only:

```text
idealized format-decision-count proxy
```

---

# 15. Exact mixed N8/N16 objective

For one N16 output stripe, each K64 coordinate \(k\) can be unsplit or split.

Let the top and bottom N8 child candidate errors be:

\[
E_{k,A}^{f_A}
\in\mathbb R^{T\times 8},
\]

\[
E_{k,B}^{f_B}
\in\mathbb R^{T\times 8}.
\]

---

## 15.1 Unsplit coordinate

For shared format \(f\):

\[
E_k^{\text{unsplit},f}
=
\operatorname{concat}_{N}
\left(
E_{k,A}^{f},
E_{k,B}^{f}
\right)
\in\mathbb R^{T\times16}.
\]

Legal states:

```text
E2/E2
E0/E0
```

with one shared format decision.

---

## 15.2 Split coordinate

For independent child formats:

\[
E_k^{\text{split},f_A,f_B}
=
\operatorname{concat}_{N}
\left(
E_{k,A}^{f_A},
E_{k,B}^{f_B}
\right).
\]

Legal states:

```text
E2/E2
E2/E0
E0/E2
E0/E0
```

and the representation uses two child format decisions.

---

## 15.3 Full mixed-stripe objective

For split mask \(S\) and legal state map \(F\):

\[
J(F;S)
=
\left\|
\sum_kE_k(F_k;S_k)
\right\|_F^2.
\]

Important:

The top/bottom N8 outputs occupy disjoint output columns.

The N8 children are coupled by the **shared decision constraint** when unsplit, not by a
cross-child dot-product.

Cross-K interactions remain within the full residual.

This distinction must be preserved in explanations and code.

---

# 16. Mixed CD2 algorithm

The split mask is fixed before quantization-map optimization.

For each mask:

### Initialization

```text
unsplit coordinate:
  matched local/OutputAware shared N16 state

split coordinate:
  matched local/OutputAware child states
```

### Sweep

Process K64 coordinates in the exact frozen coordinate order.

For unsplit coordinate:

```text
evaluate 2 legal shared states
choose objective-minimizing state
```

For split coordinate:

```text
evaluate 4 legal child-state pairs
choose objective-minimizing pair
```

Accept only if full mixed-stripe objective decreases.

Run:

```text
exactly 2 sweeps
```

No extra passes.

---

# 17. Endpoint equivalence is a hard correctness gate

Before any expensive Phase-1 sweep:

## B=0

Must reproduce frozen:

```text
CD2-N16
```

including:

```text
format map
objective
per-sequence NLL
PPL
```

within exact/reference tolerances.

## B=100

Must reproduce frozen:

```text
CD2-N8
```

The mixed implementation may update two N8 child states as one four-state coordinate only
because their output supports are disjoint.

The resulting endpoint must still be bit/map-equivalent to the frozen independent N8 CD2
implementation under the same coordinate/tie semantics.

If B=100 does not reproduce N8 exactly:

```text
STOP
```

and correct ordering/tie/residual semantics before proceeding.

---

# 18. Mandatory mixed-representation correctness tests

Before Phase 1:

```text
[ ] B=0 reproduces CD2-N16.
[ ] B=100 reproduces CD2-N8.
[ ] one split N16 maps exactly to its two N8 children.
[ ] one unsplit N16 remains one shared decision.
[ ] K16 scale grouping is unchanged.
[ ] candidate reconstruction equals frozen semantics.
[ ] T×16 residual embedding is exact.
[ ] unsplit flip delta equals full recomputation.
[ ] split four-state update equals full recomputation.
[ ] every accepted state change is objective-nonincreasing.
[ ] tail N/K handling is deterministic.
[ ] serialization/reload preserves split mask and format map.
[ ] evaluation manifests match frozen anchors.
```

If any fail:

```text
SELECTIVE_GRANULARITY_SEMANTIC_FAIL
```

Do not run the full sweep.

---

# 19. Phase-1 selector/diagnostic families

Compare at minimum:

```text
Random
Conflict-only
Opposite-margin severity
Local merge regret
Conditional full-stripe split gain
Module-level structural selector
Actual-NLL oracle diagnostic if computationally feasible
```

No learned router in Phase 1.

---

# 20. S0 — Random baseline

For each budget \(B\):

```text
randomly split B% of N16 regions
```

Use at least:

```text
5 fixed seeds
```

if computationally feasible.

Report:

```text
mean
std
best
worst
```

of end-to-end paired NLL.

Random is mandatory.

If a proposed score is not materially better than random, its mechanism value is weak.

---

# 21. S1 — Conflict-only baseline

Using frozen N8 child decisions:

```text
conflict = E2/E0 or E0/E2
```

Rank conflicts ahead of agreements.

Within tied groups use deterministic region ID ordering.

This is intentionally crude.

It tests whether binary disagreement alone is sufficient.

---

# 22. S2 — Opposite-margin severity

For each N8 child define local format margin:

\[
m
=
J(E0)-J(E2).
\]

Signs:

```text
m > 0 -> E2 preferred
m < 0 -> E0 preferred
```

Primary severity score:

\[
Score_{\text{margin}}
=
\mathbb 1[\operatorname{sign}(m_A)\ne\operatorname{sign}(m_B)]
\cdot
\min(|m_A|,|m_B|).
\]

This ranks pairs where both children strongly prefer opposite formats.

Secondary normalized version may be reported:

\[
Score_{\text{margin,norm}}
=
\frac{Score_{\text{margin}}}
{J_{shared}+\epsilon}.
\]

Do not create a large score zoo.

---

# 23. S3 — Local merge regret

For the two child regions:

\[
J_{\text{free}}
=
\min_{f_A}J_A(f_A)
+
\min_{f_B}J_B(f_B),
\]

while shared:

\[
J_{\text{shared}}
=
\min_{f\in\{E2,E0\}}
\left[
J_A(f)+J_B(f)
\right].
\]

Define:

\[
R_{\text{merge}}
=
J_{\text{shared}}-J_{\text{free}}.
\]

It should be non-negative up to numerical tolerance.

Rank descending.

Use the exact frozen local output-aware candidate semantics.

Do not call this end-to-end loss.

---

# 24. S4 — Conditional full-stripe split gain

This is the strongest calibration-only mechanistic score to test.

Begin from the frozen global N16 CD2 solution.

For region \(k\), hold all other region states fixed.

Current residual:

\[
R.
\]

Current shared contribution:

\[
E_k^{f_{\text{shared}}}.
\]

For each of the four possible child states \((f_A,f_B)\), construct:

\[
R'
=
R
-
E_k^{f_{\text{shared}}}
+
E_k^{\text{split},f_A,f_B}.
\]

Define:

\[
Gain_{\text{conditional},k}
=
J(R)
-
\min_{f_A,f_B}J(R').
\]

Rank descending.

This differs from local merge regret because it includes the current full-stripe residual and
cross-K interactions.

Use calibration data only.

---

# 25. S5 — Module-level structural granularity baseline

Research-5/N16 artifacts show that merge regret can be concentrated in specific module
families/instances.

Therefore test a hardware-friendlier structural baseline.

For each module \(l\), aggregate a region score:

Primary:

\[
Score_l
=
\sum_{r\in l}R_{\text{merge},r}.
\]

Secondary report:

```text
mean regret
p90 regret
fraction high-conflict
```

Construct module-level refinement:

```text
rank whole modules
mark all N16 regions in selected modules as split
```

Stop when the cumulative split-region count reaches or remains below the target budget.

Do not partially select regions inside a module for this baseline.

Actual achieved \(B\) may differ slightly from target.

Plot it at its actual refinement fraction.

Question:

> Is most selective-granularity value achievable with a much simpler module-level control
> policy?

If yes, this may be more hardware-friendly than region-level exceptions.

---

# 26. Optional S6 — hierarchical module → region selector

Locked initially.

Unlock only if:

```text
module-level score captures substantial headroom
but region-level merge/conditional ranking is meaningfully better
```

Then test exactly one hierarchical scheme:

```text
select high-risk modules using module aggregate score
then
within selected modules rank regions by the winning region score
```

Do not train a learned model.

Do not unlock before Phase-1 basic curves are complete.

---

# 27. Optional actual-NLL oracle diagnostic

The purpose is only:

```text
upper-bound headroom
```

It is not deployable.

A full per-region end-to-end NLL oracle may be computationally prohibitive.

Allowed bounded implementations:

```text
module-level split intervention oracle
or
sampled high-risk region intervention oracle
```

If used:

1. pre-register the sampled units before outcome inspection;
2. compute actual NLL change from split vs unsplit;
3. construct an oracle diagnostic curve;
4. never use oracle labels to choose the final calibration score.

If oracle headroom is weak:

```text
Track B should stop.
```

The actual-NLL oracle is optional if cost is unreasonable.

The end-to-end budget curves of calibration-only selectors remain mandatory.

---

# 28. Phase-1 data discipline

Use the established calibration protocol.

Primary selector feature/calibration source:

```text
the frozen C4 calibration protocol used by CD2
```

Use the same:

```text
sequence IDs
sequence length
number of sequences
model revision
layer-input semantics
```

where possible.

The score must not use final evaluation NLL.

---

## 28.1 Development-diagnostic versus final development evaluation

To avoid repeated PPL tuning, separate:

```text
calibration train/validation
development diagnostic evaluation
full development closure evaluation
```

Recommended:

### Calibration

Use existing frozen calibration shards.

### Development diagnostic set

Use a fixed subset of Wiki/C4 sequences for Phase-1 budget curves.

Record IDs before running the curves.

### Full development closure

After one selector and budget rule are frozen, use the full existing Wiki/C4 evaluation
manifests for Llama/Qwen/Mistral.

Do not repeatedly redesign the selector using full closure results.

---

# 29. Negative-gap / N16-better settings are first-class hard cases

For some settings:

\[
NLL(N16)<NLL(N8).
\]

Example entering Research-6:

```text
Mistral Wiki W4A16
```

For such settings:

```text
Recovery(N16->N8)
```

is not a meaningful success metric.

Instead require:

```text
selective policy does not destroy the N16 advantage
```

Define:

\[
\Delta_{\text{selective-vs-N16}}
=
NLL(Selective)-NLL(N16).
\]

For negative-gap settings, the main gate is non-regression relative to N16.

This is essential.

A selector that improves Qwen by blindly moving every model toward N8 is not general.

---

# 30. Phase-1 metrics

For every:

```text
model
corpus
selector
budget/actual refinement fraction
```

store:

```text
PPL
mean NLL/token
paired ΔNLL vs global N16
paired ΔNLL vs global N8
95% paired bootstrap CI
win fraction
B
number of split regions
extra format decisions
idealized decision-count ratio
E2/E0 ratio
split distribution by layer
split distribution by module type
offline scoring wall time
peak memory
```

---

# 31. N16→N8 gap and recovery

Define:

\[
Gap_{16\rightarrow8}
=
NLL(N16)-NLL(N8).
\]

When:

```text
Gap > practical positive threshold
```

define:

\[
Recovery(B)
=
\frac{
NLL(N16)-NLL(Selective_B)
}{
NLL(N16)-NLL(N8)
}.
\]

Do not clip values.

Always display absolute ΔNLL beside Recovery.

If denominator is tiny or negative:

```text
do not use Recovery as primary metric.
```

---

# 32. Phase-1 primary figures

## Figure 1 — Quality vs refinement budget

x:

```text
fraction of N16 regions split to N8
```

y:

```text
paired ΔNLL vs global N16
```

Show:

```text
Random
Conflict
Margin
MergeRegret
ConditionalGain
Module-level
N8 endpoint
```

for all three models and both corpora.

---

## Figure 2 — Gap recovery vs refinement budget

x:

```text
B
```

y:

\[
Recovery(B)
\]

only for meaningful positive-gap settings.

Include 0 and 100 endpoints.

---

## Figure 3 — Regret concentration

Sort regions by local merge regret.

Plot:

```text
top X% regions
vs
cumulative fraction of total local merge regret
```

Also plot conditional split-gain concentration.

This tests heavy-tail exploitability.

---

## Figure 4 — Selector quality

At fixed budgets:

```text
Random
Conflict
Margin
MergeRegret
ConditionalGain
Module-level
optional oracle
```

Show end-to-end ΔNLL.

---

## Figure 5 — Control proxy

If Track B is positive, plot quality against:

\[
C_{\text{decision}}=\frac{1+B}{2}.
\]

Label it:

```text
idealized decision-count proxy
```

not hardware cost.

If hardware collaborators later provide real cost, replace/add a true hardware x-axis.

---

# 33. Phase-1 stopping and success thresholds

Freeze these before running the final Phase-1 development-diagnostic curves.

Recommended Research-6 v1.0 gate:

---

## 33.1 Immediate Track-B stop

Issue:

```text
SELECTIVE_GRANULARITY_NO_HEADROOM
```

if any of the following strong conditions holds:

```text
A. small-budget actual-NLL oracle/headroom diagnostic is weak, when measured;

OR

B. at <=20% refinement, the best calibration-only selector
   recovers <50% of meaningful N16->N8 gap
   on most positive-gap development settings;

OR

C. merge-regret / conditional-gain is not materially better than random;

OR

D. near-N8 quality generally requires >=50% refinement;

OR

E. gains are dominated by one model/corpus and known hard cases are not improved;

OR

F. negative-gap settings are systematically harmed.
```

Then:

```text
do not build learned router
do not add CD3/CD4
do not add new scores endlessly
```

Track B closes.

Track A continues.

---

## 33.2 Diagnostic GO

Issue:

```text
SELECTIVE_GRANULARITY_DIAGNOSTIC_GO
```

if:

```text
1. <=20% refinement recovers >=70% of the meaningful N16->N8 gap
   on at least 4/5 meaningful positive-gap W4A16 development settings;

2. no negative-gap setting shows a meaningful new regression
   beyond the frozen practical threshold;

3. one calibration-only selector clearly beats Random and Conflict-only;

4. Qwen C4 improves materially;

5. endpoint and mixed-objective correctness are exact.
```

The known W4A16 matrix has five positive-gap N16-worse settings and one N16-better setting.

If future regenerated data differ, derive the count from artifacts rather than hard-coding
the model names.

---

## 33.3 Strong signal

Internal paper-level target:

```text
<=10% refinement
recovers >=80% of meaningful positive N16->N8 gap
on most development settings
```

plus:

```text
negative-gap settings remain safe
```

and later:

```text
same frozen selector/budget generalizes prospectively.
```

This is a target, not a promise.

---

# 34. Phase-1 selector winner rule

If Track B passes:

Choose **one** calibration-only score family.

Preference order is not hard-coded.

Rank by:

```text
1. worst-case end-to-end behavior across development settings
2. recovery at small B
3. margin over Random
4. simplicity
5. calibration cost
6. hardware interpretability
```

Do not choose by average PPL alone.

Candidate final forms:

```text
global ranked budget
threshold policy
module-level policy
one hierarchical policy if unlocked
```

No learned neural router unless a later human-authored spec explicitly authorizes it.

---

# 35. Phase 2 — build one frozen Selective Granularity policy

Only after:

```text
SELECTIVE_GRANULARITY_DIAGNOSTIC_GO
```

Freeze:

```text
score definition
normalization
budget B or threshold tau
module/region granularity
calibration protocol
coordinate order
CD sweep count = 2
tie behavior
```

Do not tune separately for:

```text
Llama
Qwen
Mistral
Wiki
C4
```

---

## 35.1 Budget selection

Prefer a global fixed budget if possible.

Example policy:

```text
score all N16 regions
split top B*
```

Choose \(B^*\) using a predeclared rule such as:

> smallest budget whose calibration/development-diagnostic behavior meets the recovery and
> non-regression target across the development matrix.

Record every candidate considered.

Do not choose \(B^*\) after prospective-family outcomes.

---

## 35.2 Threshold alternative

Use only if one threshold is stable across development families.

Policy:

```text
split iff score > tau
```

Record resulting B per model.

If threshold produces wildly different refinement fractions or unstable quality:

```text
prefer fixed-budget ranking.
```

---

# 36. Phase-2 development closure

After selector/budget freeze, run full development evaluation:

```text
Llama Wiki
Llama C4
Qwen Wiki
Qwen C4
Mistral Wiki
Mistral C4
```

Compare:

```text
High Precision
Fixed-E2
4Over6
Raw-N8
OutputAware-N8
CD2-N8
Raw-N16
CD2-N16
Selective Granularity
```

Primary paired comparisons:

```text
Selective vs CD2-N16
Selective vs CD2-N8
Selective vs 4Over6
```

No retuning after these results.

---

# 37. Phase-2 selector GO gate

Issue:

```text
SELECTOR_DEVELOPMENT_GO
```

only if:

```text
positive-gap settings:
  selective recovers the intended majority of N8 benefit

negative-gap settings:
  selective is non-regressive vs N16

all settings:
  materially better than random at matched B
  competitive with 4Over6
  no model-specific tuning
```

If the Phase-1 curve looked positive but the frozen policy fails full development closure:

```text
SELECTOR_NO_GO
```

Track B closes.

---

# 38. Prospective family must be selected/sealed before final selector freeze

The development families:

```text
Llama
Qwen
Mistral
```

have been repeatedly inspected.

They are not prospective evidence.

Before Phase-2 final selector freeze:

1. choose one genuinely unseen LLM family/architecture/scale case;
2. record exact model ID/revision;
3. verify loading/harness support without running quantized-quality outcomes;
4. seal evaluation outputs.

Create:

```text
artifacts/research_6/02_prospective_seal/primary_family.json
```

If feasible, also preselect a second independent confirmation:

```text
secondary_family_or_scale.json
```

Do not choose the family after looking for a favorable result.

---

# 39. One prospective family can serve both Track A and Track B

Do not consume a prospective family early for Track A if Track B is still under
development.

Preferred order:

```text
Track-B method development completes or closes
↓
Track-A base CD2-N8 is already frozen
Track-B selective method is frozen if it exists
↓
unseal prospective family
↓
evaluate both frozen methods together
```

This preserves a fair prospective test for both paper tracks.

If Track B closed:

```text
evaluate frozen CD2-N8 for Track A.
```

If Track B succeeded:

```text
evaluate CD2-N8 + Selective Granularity.
```

No retuning.

---

# 40. Prospective validation

Use at least:

```text
two predeclared corpora/tasks
```

with the same frozen:

```text
format semantics
score
budget/threshold
calibration size
calibration protocol
two-sweep CD
```

Track A prospective success:

```text
CD2-N8 is non-regressive/competitive with strong baselines.
```

Track B prospective success:

```text
Selective improves the N16 point where needed
while remaining safe where N16 is already strong,
and preserves a useful small-refinement Pareto.
```

If Track B fails prospectively:

```text
PROSPECTIVE_SELECTIVE_FAIL
```

Do not retune.

Track A may remain viable.

---

# 41. Second independent confirmation

For:

```text
STRONG_TOP_TIER_CANDIDATE
```

prefer one additional:

```text
unseen family
OR
larger model / independent architecture-scale point
```

with no retuning.

A single prospective family can support:

```text
SUBMIT_READY_BASE
```

if other evidence is strong.

The stronger Track-B claim should ideally receive a second confirmation.

---

# 42. Track-A closure requirements

Even if Track B fails, coding work must evaluate whether Track A can reach:

```text
SUBMIT_READY_BASE
```

Track A minimum closure:

### Problem/mechanism

```text
fine->coarse decision compression
Raw/OA/CD2 progression
cross-K interaction diagnostics
N8/N16 matched boundary
known failure cases
```

### Accuracy

```text
Llama/Qwen/Mistral × Wiki/C4
paired NLL
strong Fixed-E2 / 4Over6 comparison
```

### W4A4

```text
A0 Fixed-E2M1
A1 Activation-4Over6
complete N8 evidence
Research-6 N16+A1 post-hoc characterization
```

### Generalization

```text
one frozen unseen family or independent scale/architecture case
```

### Tasks

```text
proper downstream after method freeze
```

### Literature

```text
CDQuant
IF4
MixFP4
BlockDialect
4Over6
closest current hardware-aware method
```

### Deployment

```text
static map accounting
K16 scale accounting
offline calibration cost
format-decision/control accounting
hardware collaborator handoff
```

---

# 43. W4A4 Selective Granularity validation

Only if Track B passes W4A16 development.

Do not run the full budget grid again.

Use:

```text
B=0%
B=best frozen small budget
B=one predeclared medium budget
B=100%
```

or the exact threshold policy outputs.

Use the exact same split score/policy.

---

## 43.1 A0 first

Activation:

```text
Fixed-E2M1 / NVFP4-compatible
```

Run:

```text
Llama/Qwen/Mistral × Wiki/C4
```

Question:

> Does the Selective Granularity Pareto survive activation quantization?

---

## 43.2 A1 second

Activation:

```text
Activation-4Over6
```

Use the same frozen weight split masks/policy.

No A1-driven retuning.

Question:

> Is selective weight-control granularity compatible with the stronger activation quantizer?

Activation A1 remains a supporting baseline.

---

# 44. Proper downstream task closure

After the final LLM method is frozen.

Recommended fixed suite if supported:

```text
HellaSwag
PIQA
ARC-Challenge
WinoGrande
BoolQ
MMLU
```

Optional:

```text
GSM8K
```

only if generation protocol is already stable and predeclared.

Compare at minimum:

```text
High Precision
Fixed-E2
4Over6
CD2-N8
CD2-N16
final Selective method if Track B passes
```

Do not use a tiny 100-example sanity set as the headline table.

Primary goal:

```text
no systematic downstream regression
```

not necessarily win every task.

---

# 45. W4A4/Downstream are required before strong paper status

A positive PPL Pareto alone is insufficient for:

```text
STRONG_TOP_TIER_CANDIDATE
```

Need:

```text
W4A4 survival
proper downstream
prospective family
hardware motivation
```

---

# 46. Diffusion extension — optional cross-domain stress test

Diffusion is not required for an LLM-only top-tier submission.

Run only after the LLM selector is frozen.

Primary diffusion model:

```text
SANA-1.6B
```

because prior project infrastructure exists.

Do not use diffusion as a method-development sandbox.

---

# 47. Diffusion D0 — frozen baselines

Use fixed:

```text
model revision
prompts
seeds
scheduler
steps
resolution
```

Compare:

```text
High Precision
Fixed-E2 / NVFP4-compatible
4Over6
Raw-N8
CD2-N8
Raw-N16
CD2-N16
```

Reuse historical rows only if exact protocol hashes match.

No diffusion-specific tuning.

---

# 48. Diffusion D1 — Selective Granularity proxy

Only if Track B has a frozen LLM selector.

Apply the same:

```text
score definition
budget/threshold
mixed N8/N16 semantics
```

to SANA using a fixed SANA calibration prompt set.

Model-specific split masks may be computed from SANA calibration because weights/activations differ.

Do not change:

```text
score formula
budget
threshold
CD sweep count
```

for SANA.

Budgets:

```text
0%
frozen selected small budget
optional frozen medium budget
100%
```

Do not rerun the full LLM budget grid.

Metrics:

```text
denoiser/flow prediction MSE
NMSE
relative L2
cosine error
latent trajectory deviation
per-layer output error
```

If Selective is clearly dominated:

```text
DIFFUSION_PROXY_FAIL
```

Stop cross-domain extension.

LLM paper remains unaffected.

---

# 49. Diffusion D2 — fixed image screen

If D1 is non-dominated/interesting:

```text
128 or 192 fixed prompts
```

Metrics:

```text
LPIPS ↓
PSNR ↑
SSIM ↑
CLIP ↑
ImageReward ↑
```

Prior work already showed:

```text
better reconstruction
can coexist with worse semantic image quality.
```

Therefore do not declare success from LPIPS/PSNR alone.

---

# 50. Diffusion decision

Issue:

```text
DIFFUSION_CROSS_DOMAIN_NO_GO
```

if:

```text
proxy improves but semantic metrics materially regress
or
Selective/CD2 is dominated by 4Over6 on the relevant Pareto
or
SANA-specific retuning is required
```

Issue:

```text
DIFFUSION_CONDITIONAL_GO
```

if:

```text
same frozen policy is non-dominated
and semantic quality is safe.
```

Only run 1024 images or a second diffusion family if the small screen clearly passes.

---

# 51. Hardware/deployment collaboration is a core Track-A/Track-B requirement

Accuracy work must not invent hardware costs.

The hardware team should independently quantify:

## N8

```text
format decision count
format metadata/control bits
decode/control distribution
routing/fanout
area
power
timing
native implementation implications
```

## N16

Same quantities.

## Selective

```text
base N16 control
+
additional child decision for each split region
+
exception/split map encoding
+
exception routing/decoder cost
```

The key systems question is:

> Why should a hardware designer prefer N16 or sparse N8 exceptions instead of N8
> everywhere?

Without a credible answer, Track B is only an accuracy exercise.

---

# 52. Accuracy-side hardware handoff quantities

Research-6 coding agent must provide the hardware collaborator:

```text
number of N16 regions R
split count S
B=S/R
R+S idealized format-decision count
2R N8 decision count
split density per layer/module
split-mask sparsity
split-mask run-length statistics
split-mask entropy statistic
E2/E0 fractions
selected-region spatial distribution
quality/NLL at each B
```

These are characterization inputs.

Do not turn them into area/power/latency claims.

---

## 52.1 Optional information-theoretic mask statistic

It is acceptable to report:

\[
H_2(B)
=
-B\log_2B-(1-B)\log_2(1-B)
\]

and:

\[
R H_2(B)
\]

as an **information-theoretic lower-bound-style mask description statistic**.

Do not call this actual metadata cost.

Actual encoding/decoder implementation belongs to hardware work.

---

# 53. Hardware status axis

Use separately:

```text
HARDWARE_COST_UNRESOLVED
HARDWARE_ANALYTICAL_ACCOUNTING
HARDWARE_PARETO_SUPPORTED
```

`HARDWARE_PARETO_SUPPORTED` requires credible collaborator evidence.

Do not infer it from B alone.

---

# 54. Native status

Use separately:

```text
WAIT_FOR_SM120
NATIVE_VALIDATED
```

A6000 / RTX 6000 Ada reference quantization cannot establish native Blackwell throughput.

---

# 55. External baseline / literature audit

Before submission, create:

```text
baseline_semantics_table.md
literature_novelty_matrix.md
```

At minimum describe:

```text
CDQuant
BlockDialect
Four Over Six
IF4
MixFP4
AdaMX/current hardware-aware microscaling if relevant
```

For each:

```text
weight format
activation format
format decision granularity
scale granularity
metadata
offline/online selection
optimizer
hardware evidence
models/tasks
```

Do not force numerical comparison if semantics differ materially.

Run numerical baselines only when:

```text
official/author code exists
the exact model/eval is supported
bit/scale semantics can be matched fairly
```

Never label an approximate reimplementation "official".

---

# 56. Relation to CDQuant must be explicit

Do not claim:

```text
coordinate descent is novel
```

The paper should say:

> CDQuant and related work demonstrate coordinate-descent-style reconstruction optimization.
> Our contribution is the constrained format-control problem: coarse hardware regions share
> legal format decisions, cross-K errors interact, and the design variable is the spatial
> allocation of format-control granularity.

Evidence must make this distinction concrete:

```text
matched local vs coupled objective
N8/N16 control constraints
selective split representation
hardware-control accounting
```

---

# 57. Calibration/offline cost accounting

For:

```text
CD2-N8
CD2-N16
Selective score generation
Selective mixed CD2
```

record:

```text
calibration sequence count
forward passes
wall time
GPU time
peak VRAM
peak host RAM
activation/error cache bytes
score computation time
sort/ranking time
mixed CD map time
serialized format map bytes
split-mask bytes in the chosen reference serialization
```

The reference split-mask serialization is not proof of hardware cost.

Label separately.

---

# 58. Primary statistics

Model quality:

```text
paired per-sequence/per-token NLL
mean ΔNLL
median ΔNLL
95% paired bootstrap CI
win fraction
PPL secondary
```

Use fixed bootstrap seeds.

For selective curves, compare:

```text
Selective vs N16
Selective vs N8
selector vs Random at matched B
```

Do not rely on point PPL.

---

# 59. Random-seed statistics

For Random-B:

Use multiple fixed masks/seeds.

Report:

```text
mean paired NLL
std across masks
95% range if enough masks
best random
worst random
```

A deterministic selector should beat the **random distribution**, not merely one random seed.

---

# 60. Correlation/mechanism statistics

For region/module diagnostics:

```text
Spearman(score, local merge regret)
Spearman(score, conditional split gain)
```

and, where actual-NLL interventions are measured:

```text
Spearman(score, actual ΔNLL benefit)
```

Do not treat millions of regions as independent model-family samples.

Report:

```text
model-wise
corpus-wise
module-type
layer-depth
```

first.

---

# 61. Practical non-regression margin

Before final development/prospective tests, freeze a meaningful NLL non-regression margin.

Prefer:

```text
reuse a prior frozen project margin
```

if statistically appropriate.

Otherwise derive from:

```text
repeat/no-op numerical variability
+
predeclared practical tolerance
```

Store:

```text
artifacts/research_6/00_spec/noninferiority_margin.json
```

Do not choose it after seeing Selective results.

---

# 62. Prospective leakage rules

Before unseal:

```text
score frozen
normalization frozen
budget/threshold frozen
calibration protocol frozen
mixed CD code frozen
code commit frozen
```

Store timestamps/hashes.

Machine fields:

```text
method_freeze_timestamp
prospective_unseal_timestamp
prospective_was_accessed_pre_freeze
```

If leakage occurs:

```text
prospective claim is invalid
```

and a new unseen case is required.

---

# 63. Existing development matrices are not prospective

Llama/Qwen/Mistral have been repeatedly inspected.

Do not call:

```text
Mistral
```

prospective.

Do not call:

```text
new budget on old Mistral
```

unseen evidence.

---

# 64. Research-6 artifact tree

Everything new:

```text
artifacts/research_6/
```

Recommended:

```text
00_environment/
  spec_acknowledgement.md
  spec_manifest.json
  source_artifact_manifest.csv
  source_hashes.json
  repo_manifest.json
  patch_manifest.json
  environment.txt
  gpu_usage_log.jsonl
  literature_snapshot.md

00_spec/
  research_questions.md
  track_a_gate.json
  track_b_gate.json
  budget_grid.json
  noninferiority_margin.json
  leakage_policy.md
  prohibited_rescues.md

01_reproduction/
  endpoint_reproduction.csv
  n8_map_hashes.json
  n16_map_hashes.json
  paired_nll_reproduction.csv
  reproduction_report.md

02_prospective_seal/
  primary_family.json
  secondary_family_or_scale.json
  seal_log.json

03_w4a4_n16_a1/
  semantics_audit.md
  llama_wiki.csv
  llama_c4.csv
  qwen_wiki.csv
  qwen_c4.csv
  mistral_wiki.csv
  mistral_c4.csv
  paired_nll.csv
  paired_bootstrap.json
  complete_n8_n16_a0_a1_table.csv
  characterization_report.md

04_selective_correctness/
  mixed_n8_n16_grouping.json
  endpoint_b0_equivalence.json
  endpoint_b100_equivalence.json
  residual_embedding_tests.json
  state_delta_equivalence.json
  objective_monotonicity.json
  serialization_equivalence.json
  tail_tests.json

05_region_scores/
  region_scores.parquet
  conflict_scores.parquet
  margin_scores.parquet
  merge_regret_scores.parquet
  conditional_split_gain.parquet
  module_scores.parquet
  score_summary.csv

06_phase1_pareto/
  random/
  conflict/
  margin/
  merge_regret/
  conditional_gain/
  module_level/
  oracle_optional/
  budget_results.csv
  paired_statistics.csv
  recovery.csv
  control_proxy.csv
  phase1_figures/
  phase1_report.md
  phase1_decision.json

07_selective_policy/
  unlock_decision.md
  selected_score.json
  selected_budget_or_threshold.json
  frozen_policy.json
  policy_hashes.json
  calibration_validation.csv
  development_diagnostic.csv

08_development_closure/
  llama_wiki.csv
  llama_c4.csv
  qwen_wiki.csv
  qwen_c4.csv
  mistral_wiki.csv
  mistral_c4.csv
  paired_nll.csv
  bootstrap.json
  selector_gate.json
  development_report.md

09_prospective/
  unseal_log.json
  primary_family_results.csv
  primary_paired_nll.csv
  primary_decision.md
  secondary_confirmation/
  prospective_summary.md

10_w4a4_selective/
  a0/
  a1/
  paired_nll.csv
  recovery.csv
  summary.md

11_downstream/
  task_manifest.json
  results.csv
  summary.md

12_diffusion/
  frozen_protocol.json
  baselines/
  proxy/
  images_128_192/
  images_1024/
  diffusion_decision.md

13_hardware_handoff/
  accuracy_side_requirements.md
  format_decision_counts.csv
  split_mask_statistics.csv
  per_layer_split_density.csv
  per_module_split_density.csv
  quality_control_proxy.csv
  collaborator_questions.md

14_external_baselines/
  baseline_semantics_table.md
  literature_novelty_matrix.md
  configs/
  results.csv

15_cost/
  offline_cost.csv
  reference_serialization.md
  cache_accounting.csv

16_track_a/
  base_paper_checklist.md
  track_a_evidence_matrix.csv
  track_a_decision.json

17_track_b/
  selective_paper_checklist.md
  track_b_evidence_matrix.csv
  track_b_decision.json

18_final/
  experiment_manifest.jsonl
  failed_runs.jsonl
  master_results.csv
  generalization_matrix.csv
  main_tables.md
  final_figures/
  final_decision.json
  final_decision_report.md
  results_summary.md
  limitations.md
  paper_claims_boundary.md
  reproduction_commands.sh
```

Terminal output is not the source of truth.

---

# 65. Machine-readable common fields

```text
experiment_id
phase
track
model_id
model_revision
corpus
weight_policy
activation_policy

format_granularity
scale_granularity
split_policy
split_score
target_budget
actual_budget

calibration_manifest_hash
evaluation_manifest_hash
code_hash
config_hash
git_commit

gpu_physical_id
gpu_uuid
gpu_type
start_time
end_time
status
```

---

# 66. Selective fields

```text
n16_region_id
layer_index
module_name
module_type
k64_index
split

child_a_format
child_b_format
child_conflict
child_a_margin
child_b_margin

local_merge_regret
conditional_split_gain
module_score

initial_state
final_state
state_flip_count

R_total
S_split
B_refinement
idealized_decision_count
idealized_decision_ratio_vs_n8
```

---

# 67. Evaluation fields

```text
sequence_id
token_count
nll
paired_reference
paired_delta_nll

mean_delta_nll
median_delta_nll
ci95_low
ci95_high
win_fraction
ppl

gap_n16_to_n8
recovery_fraction
negative_gap_setting
```

---

# 68. Hardware-handoff fields

```text
num_n16_regions
num_split_regions
split_fraction
idealized_format_decision_count
n8_format_decision_count
split_run_count
mean_split_run_length
split_mask_binary_entropy
per_layer_split_fraction
per_module_split_fraction

hardware_area = UNMEASURED
hardware_power = UNMEASURED
hardware_latency = UNMEASURED
```

Do not fill unavailable hardware quantities with estimates unless provided by the hardware team.

---

# 69. GPU execution policy

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum concurrent Research-6 GPUs:

```text
3
```

Before every GPU process:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi pmon -c 1
```

For every visible PID:

```bash
ps -o user=,pid=,cmd= -p <PID>
```

Rules:

```text
another-user process -> GPU unavailable
unknown occupied GPU -> unavailable
low utilization != free
never share physical GPU with another user
never kill another user's process
never use broad pkill
use project reservation/lock
re-check immediately before launch
log every admission/rejection
```

Reference/fake quantization only.

No native SM120 claim.

---

# 70. Git policy

Before changes:

```text
record HEAD
record branch
record dirty state
record submodules
record prior result commit hashes
```

Create isolated Research-6 branch/worktree.

Do not:

```text
modify immutable Research-1~5 artifacts
force-push
push without explicit user instruction
destructively rewrite another branch
```

Local commits should mark:

```text
Research-6 initialization
W4A4 N16+A1 completion
mixed representation correctness
Phase-1 Pareto
selector freeze
prospective results
final aggregation
```

---

# 71. Failure handling

Every failed/aborted run must store:

```text
command
config
stdout
stderr
exit code
timestamp
GPU state
classification
```

Classes:

```text
code_bug
OOM
dataset_network
model_access
dependency
GPU_became_occupied
endpoint_mismatch
grouping_mismatch
residual_equivalence_failure
serialization_failure
objective_nonmonotone
activation_semantic_mismatch
calibration_leakage
evaluation_leakage
prospective_leakage
hardware_dependency
runtime_excessive
unknown
```

Never silently drop failures.

---

# 72. Prohibited Research-6 rescue work

Do not automatically reopen:

```text
CD3 / CD4
until-convergence CD
random coordinate-order search
new scale grids
Research-4 DualScale
target5
Hadamard
rotation
packing
permutation
Fisher objective
Taylor objective
teacher-KL objective
new E0 codebook
activation E2/E0 mixing
activation CD
model-specific selector thresholds
Mistral-specific rescue
Qwen-specific rescue
```

If Track B is weak:

```text
close it.
```

Do not rescue with unrelated tricks.

---

# 73. Track-A prospective closure

If Track B closes early, do not terminate all Research-6 work.

Continue:

```text
prospective family for frozen N8-CD2
downstream
baseline/novelty audit
hardware handoff
cost accounting
```

Track A final question:

> Is the established N8-CD2 contribution sufficiently novel, general, deployment-relevant,
> and statistically complete for a reasonable ICLR/NeurIPS submission?

---

# 74. Track-A submission-ready gate

Issue:

```text
SUBMIT_READY_BASE
```

only if the majority of the following are satisfied:

```text
1. decision-compression problem is directly demonstrated;
2. cross-K coupling mechanism is directly measured;
3. N8-CD2 remains strong on Llama/Qwen/Mistral core matrix;
4. N8/N16 matched boundary is complete and honestly model-dependent;
5. W4A4 A0 and A1 evidence is complete, including Research-6 N16+A1 characterization;
6. one frozen unseen family/independent case is non-regressive/competitive;
7. meaningful downstream suite has no systematic regression;
8. closest prior work is audited fairly;
9. offline calibration cost is explicit;
10. hardware/control-granularity motivation has at least credible analytical/collaborator
    accounting;
11. negative cases remain in the paper.
```

This makes submission reasonable.

It does not guarantee acceptance.

---

# 75. Track-B strong paper gate

Issue:

```text
STRONG_TOP_TIER_CANDIDATE
```

only if Track-A essentials are satisfied and additionally:

```text
1. Selective Granularity has a clearly curved Pareto;
2. <=10–20% refinement recovers most of the meaningful N16->N8 gap;
3. calibration-only selector clearly beats random/conflict baselines;
4. negative-gap settings remain safe;
5. frozen selector works on an unseen case with no retuning;
6. second confirmation/larger case is positive if feasible;
7. W4A4 preserves the qualitative Pareto;
8. hardware/control-cost evidence shows sparse exceptions can plausibly matter;
9. contribution is explainable without "we tried many heuristics".
```

---

# 76. Hardware risk is a top-tier risk

If Track B succeeds in accuracy but the hardware team finds:

```text
sparse exception-map cost
+
routing/decoder cost
```

eliminates the N16-control benefit, the Track-B systems story weakens substantially.

Do not hide that.

Possible final state:

```text
SELECTIVE_ACCURACY_GO
HARDWARE_COST_UNRESOLVED
```

or:

```text
SELECTIVE_ACCURACY_GO
HARDWARE_PARETO_NOT_SUPPORTED
```

Track A may still remain viable.

---

# 77. Diffusion is optional

A failure on SANA means:

```text
remove cross-domain claim
```

not:

```text
kill the LLM paper.
```

Do not make Research-6 hostage to diffusion.

---

# 78. Main paper-ready tables

## Table A — W4A16 core

| Model | Corpus | Fixed-E2 | 4Over6 | Raw N8 | CD2 N8 | Raw N16 | CD2 N16 | Selective |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | Wiki | | | | | | | |
| | C4 | | | | | | | |
| Qwen | Wiki | | | | | | | |
| | C4 | | | | | | | |
| Mistral | Wiki | | | | | | | |
| | C4 | | | | | | | |

If Track B fails, omit Selective from the final base-paper main table.

---

## Table B — Selective control Pareto

| Model | Corpus | B | ΔNLL vs N16 | ΔNLL vs N8 | Recovery | Random ΔNLL | Decision-count proxy |
|---|---|---:|---:|---:|---:|---:|---:|
| ... | | | | | | | |

---

## Table C — W4A4 N8/N16 A0/A1

| Model | Corpus | CD2 N8+A0 | CD2 N16+A0 | CD2 N8+A1 | CD2 N16+A1 |
|---|---|---:|---:|---:|---:|
| ... | | | | | |

Include Raw matched rows in appendix/full table.

---

## Table D — prospective

| Model | Corpus/Task | 4Over6 | CD2-N8 | CD2-N16 | Selective |
|---|---|---:|---:|---:|---:|
| unseen | ... | | | | |

---

# 79. Main paper-ready figures

### Figure 1

```text
decision-compression ladder:
Fine16 -> Raw N8 -> CD2 N8 -> Raw/CD2 N16
```

### Figure 2

```text
cross-K interaction / coupled recovery
```

### Figure 3

```text
merge-regret concentration
```

### Figure 4

```text
Selective quality vs B
```

### Figure 5

```text
Selective quality vs idealized control-decision count
or real hardware cost if collaborator supplies it
```

### Figure 6

```text
per-layer/module split density
```

---

# 80. Recommended Track-B claim if successful

A safe claim form:

> Coarsening format control from N8 to N16 creates heterogeneous decision-compression regret.
> Most N16 regions tolerate shared control, while a small subset is responsible for a
> disproportionate quality loss. A calibration-only selective refinement policy retains N8
> decisions only in high-value regions and recovers most of the N8 quality benefit at a much
> smaller fine-control fraction.

Do not say:

```text
we reduce hardware cost by X%
```

unless real hardware accounting supports X.

---

# 81. Recommended Track-A claim if Selective fails

A safe base-paper story:

> Low-bit mixed-format quantization is constrained not only by numeric representation but by
> hardware format-control granularity. Fine format preferences compressed into N8 control
> produce cross-K interacting errors; coupled offline assignment recovers a substantial
> fraction of this loss while preserving the same low-bit payload structure. Further
> coarsening to N16 exposes a model-dependent granularity boundary.

This can remain publishable if:

```text
prospective
downstream
hardware motivation
novelty audit
```

are strong.

---

# 82. Research-6 final decision vocabulary

Algorithmic/base:

```text
RESEARCH6_ENDPOINT_REPRODUCTION_FAIL
BASE_METHOD_PAPER_CANDIDATE
SUBMIT_READY_BASE
```

Selective:

```text
SELECTIVE_GRANULARITY_SEMANTIC_FAIL
SELECTIVE_GRANULARITY_NO_HEADROOM
SELECTIVE_GRANULARITY_WEAK_HEADROOM
SELECTIVE_GRANULARITY_DIAGNOSTIC_GO
SELECTOR_NO_GO
SELECTOR_DEVELOPMENT_GO
PROSPECTIVE_SELECTIVE_FAIL
SELECTIVE_ACCURACY_GO
STRONG_TOP_TIER_CANDIDATE
```

Diffusion:

```text
DIFFUSION_NOT_RUN
DIFFUSION_PROXY_FAIL
DIFFUSION_CROSS_DOMAIN_NO_GO
DIFFUSION_CONDITIONAL_GO
CROSS_DOMAIN_STRONG
```

Hardware:

```text
HARDWARE_COST_UNRESOLVED
HARDWARE_ANALYTICAL_ACCOUNTING
HARDWARE_PARETO_NOT_SUPPORTED
HARDWARE_PARETO_SUPPORTED
```

Native:

```text
WAIT_FOR_SM120
NATIVE_VALIDATED
```

Keep the axes separate in final JSON/report.

---

# 83. Exact execution order

```text
0. Read/hash the full Research-6 spec.
1. Create isolated Research-6 worktree.
2. Audit/hash Research-1~5 and matched completion.
3. Regenerate Research-6 immutable input tables.
4. Reproduce frozen CD2-N8 and CD2-N16 endpoints.
5. Freeze non-inferiority threshold and Phase-1 GO/NO-GO thresholds.
6. Select/seal prospective family/families before Selective method freeze.

7. Complete missing W4A4 N16+A1 post-hoc characterization:
      Raw-N16 + A1
      CD2-N16 + A1
      Llama/Qwen/Mistral × Wiki/C4
   Reuse exact N8+A1 anchors.

8. Implement mixed N8/N16 representation.
9. Implement unsplit 2-state and split 4-state coordinate update.
10. Pass all B=0/B=100 correctness gates.
11. Generate calibration-only region scores:
      conflict
      margin
      local merge regret
      conditional full-stripe split gain
      module aggregate

12. Freeze Phase-1 diagnostic sequence manifests.
13. Run Random baseline masks/seeds.
14. Run Phase-1 W4A16 budget curves:
      B=0/1/2/5/10/20/50/100
      Conflict
      Margin
      MergeRegret
      ConditionalGain
      Module-level
      optional actual-NLL oracle diagnostic

15. Compute:
      paired NLL
      Recovery
      negative-gap safety
      selector-vs-random
      regret concentration
      split distributions
      idealized decision-count proxy

16. Issue Track-B Phase-1 decision:
      NO_HEADROOM
      WEAK_HEADROOM
      or DIAGNOSTIC_GO

17A. If NO_HEADROOM:
      close Track B
      do NOT create router
      continue Track-A closure.

17B. If DIAGNOSTIC_GO:
      choose ONE simple selector form.
      freeze score/budget/threshold.
      run full Llama/Qwen/Mistral development closure.
      issue SELECTOR_DEVELOPMENT_GO or SELECTOR_NO_GO.

18. When Track-B development is closed/frozen:
      unseal prospective family.
      evaluate frozen CD2-N8.
      evaluate Selective too if Track B survived.
      no retuning.

19. If strong:
      run second independent/larger confirmation.

20. Track-A/Track-B proper downstream.

21. If Track B survived:
      W4A4 Selective validation:
        A0 selected budgets
        A1 same masks/policy

22. Optional SANA:
      only frozen policy
      proxy
      128/192 images if promising
      larger only if non-dominated

23. External baseline/literature audit.

24. Offline/calibration/reference-serialization cost accounting.

25. Hardware handoff:
      region/split statistics
      control-decision proxy
      exception-map statistics
      collaborator questions
   Do not fabricate area/power/latency.

26. Regenerate final Track-A and Track-B evidence matrices.

27. Issue:
      base paper status
      selective status
      diffusion status
      hardware status
      native status

28. Write final paper-readiness report.
```

---

# 84. Immediate STOP rules

Stop new Selective method development when:

```text
B=0 endpoint mismatch
B=100 endpoint mismatch
mixed residual objective incorrect
small-budget Pareto weak
selector approximately random
near-N8 quality requires broad >=50% refinement
negative-gap hard cases are systematically damaged
```

Do not react by adding unrelated heuristics.

---

# 85. Entire paper STOP / reassessment rules

Do not abandon the whole paper merely because Selective fails.

Reassess Track A.

Recommend stopping the entire paper direction only if fair closure shows a combination of:

```text
prospective generalization fails badly
hardware/control-granularity motivation is weak/nonexistent
closest prior work makes the contribution incremental
downstream/W4A4 do not preserve core gains
```

One failure alone is not automatically fatal.

In particular:

```text
"CDQuant also uses coordinate descent"
```

is not by itself a sufficient stop reason.

The correct comparison is problem/coordinate/control semantics.

---

# 86. What counts as "good enough" for submission?

## Tier 0 — Not ready

```text
only existing PPL characterization
no prospective
no downstream
hardware motivation unresolved
no fair novelty audit
```

---

## Tier 1 — SUBMIT_READY_BASE

Required approximately:

```text
strong N8-CD2 mechanism and core matrix
N8/N16 boundary
complete W4A4 N8/N16 A0/A1 characterization
one frozen unseen case
proper downstream
fair baseline/novelty audit
credible control-granularity accounting/hardware motivation
```

Selective may have failed.

This is a reasonable ICLR/NeurIPS submission state.

---

## Tier 2 — STRONG_TOP_TIER_CANDIDATE

Tier 1 plus:

```text
clear small-budget Selective Pareto
calibration-only selector > random
negative-gap safety
prospective Selective generalization
second confirmation/larger scale
W4A4 Selective survival
hardware evidence that sparse exceptions have real value
```

This is the strongest Research-6 target.

---

## Tier 3 — CROSS_DOMAIN_STRONG

Tier 2 plus:

```text
SANA works without retuning
semantic image metrics remain safe
```

Optional.

---

# 87. Final scientific principle

Research-6 must test one coherent idea:

\[
\boxed{
\text{format-control granularity is itself a quantization design variable}
}
\]

The desired evidence chain is:

```text
fine format preference
→
coarse decision compression
→
cross-K interaction
→
coupled N8 assignment
→
N8/N16 granularity boundary
→
heterogeneous merge regret
→
sparse selective refinement
→
accuracy/control-cost Pareto
```

Every new experiment must support or falsify this chain.

If an experiment does not help answer this chain, do not add it by default.
