# research_3.md
# Research-3 — Coupled Coarse-Format Assignment: Method Closure, Robustness, and Generalization

**SPEC_VERSION:** `research_3_v1.1_2026-08-09`  
**Current scientific status:** `CONDITIONAL_GO / METHOD_CLOSURE_REQUIRED`  
**Research-2 formal decision:** `CONDITIONAL_GO`, confidence `MODERATE`  
**Primary scientific finding entering Research-3:** fixed N8×K64 format control is recoverable by coupled K64 assignment within an N8 output stripe  
**Primary unresolved issue:** close the optimization/method gap, quantify coupled interaction structure, establish calibration robustness/cost, and only then test generalization  
**Native SM120 status:** `WAIT_FOR_SM120`  
**Primary execution environment:** RTX A6000 / RTX 6000 Ada reference/fake quantization  
**Research standard:** top-tier-paper standard; do not lower gates merely to keep the project alive

---

# 0. Why Research-3 exists

Research-1 established that:

```text
fine E2/E0 format heterogeneity is useful
+
coarse N8K64 format sharing destroys part of the gain
```

and diagnosed E0-heavy coarse assignment.

Research-2 then answered a more important question:

> Is N8K64 intrinsically too restrictive, or did Research-1 use an insufficient selector?

Research-2 found:

```text
P0-A foldable N packing              = FAIL
P0-B FullLayer-Coarse-CD-N8K64       = PASS
P1-A ResidualAware-Sequential-N8K64  = PASS
final status                         = CONDITIONAL_GO
```

This changes the scientific story.

The strongest current hypothesis is no longer:

```text
coarse N8K64 is fundamentally unrecoverable
```

Instead:

> **local independent format decisions fail to account for cross-K64 error interaction
> within the same N8 output stripe; coupled offline assignment can recover a substantial
> fraction of the fine-format benefit without changing inference-time format semantics.**

However, Research-2 is not paper-ready because:

1. FullLayer-CD was labeled mainly as a bounded diagnostic even though it has the same inference-time representation as the practical selector.
2. The one-pass ResidualAware approximation leaves a large gap to FullLayer-CD, especially on Qwen.
3. Qwen C4 remains weak/uncertain for ResidualAware.
4. Research-2 calibration uses independent FP layer inputs, so cross-layer quantization drift is not modeled.
5. Calibration robustness across shards/seeds is not established.
6. Only two 8B LLM families have validated Research-2 results.
7. The new selector has not been tested on SANA, meaningful downstream tasks, W4A4, or native SM120.
8. Current 2026 literature substantially raises the novelty bar for generic adaptive formats, heterogeneity, residual compensation, and permutation.

Research-3 therefore has a different purpose:

```text
NOT: invent more quantization tricks
NOT: immediately expand benchmark count

YES:
1. determine the scientifically correct primary algorithm;
2. explain and close the CD-vs-sequential gap;
3. establish calibration robustness;
4. freeze one method;
5. only then test broader generalization.
```

---

# 1. Authority and artifact precedence

Research-3 is a new execution specification.

This v1.1 revision incorporates the useful, bounded additions from the supplementary
`research3_suggestion(1).md`. The supplementary suggestion remains review material,
not an execution spec. Where this v1.1 file differs from Research-3 v1.0, this file wins.

Use:

```text
CURRENT research_3.md
    >
CURRENT coding_agent_prompt_3.md
    >
immutable Research-2 artifacts
    >
immutable Research-1 artifacts
    >
regenerated final tables
    >
older narrative summaries
```

Do not overwrite Research-1 or Research-2.

Expected roots:

```text
artifacts/research_1/
artifacts/research_2/
artifacts/research_3/
```

Research-2 final source bundle supplied to the planning agent had:

```text
05_final.zip SHA256:
9e431bf0b355a8c03f75b99d94ba4806751488205eabc514f0424aa40579344e

Research-2 final_go_no_go.json SHA256:
842092f9ab5eabedadb7ab045a9b5b8588a592394ec5c0ca234f5cfc81b7dc8b

Research-2 final_decision_report.md SHA256:
ebbfcb8c8a38b8efdf106d6140982ac8605b28bcb85bad0c2425dc137af7e5e2

Research-2 master_results.csv SHA256:
40d57535809bb25b9b3c944fc80b9cc06d21095490b4cc0eeb12ae9ba501d3e3

Research-2 experiment_manifest.jsonl SHA256:
568155b5546ad49f94093005694f89950f994b800929b769ca3e9171e08ff335
```

The coding agent must recompute hashes from its local canonical Research-2 artifacts rather than blindly trusting these copied values.

---

# 2. Research-2 final evidence

## 2.1 Research-2 artifact integrity

Research-2 final checks reported:

```text
artifact validation: PASS
inventory files: 472
missing required final files: none
Research-1 immutable: true

final consistency: PASS
manifest lines: 46
completed attempt records: 21
frozen maps verified: 2
formal retry determinism: 4/4
P0-B trace lines: 4,128,768
git status: clean
push performed: false
```

This is a strong provenance base.

---

# 3. Established scientific facts inherited from Research-1

These are not Research-3 questions unless a regression audit invalidates them.

## 3.1 Fine mixed E2/E0 adaptation has real signal

Full W4A16 primary PPL:

| Model | Corpus | NVFP4 | MSE-Oracle16 |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.623339 | **6.559806** |
| Llama-3.1-8B | C4 | 9.557657 | **9.454540** |
| Qwen3-8B | WikiText | 9.911164 | **9.736490** |
| Qwen3-8B | C4 | 13.812140 | **13.657144** |

Thus:

```text
fine heterogeneous representation value = established
```

---

## 3.2 Raw N8K64 coarse format control loses that benefit

| Model | Corpus | MSE-Oracle16 | MSE-N8K64 |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.559806 | 6.638539 |
| Llama-3.1-8B | C4 | 9.454540 | 9.560929 |
| Qwen3-8B | WikiText | 9.736490 | 9.919024 |
| Qwen3-8B | C4 | 13.657144 | 13.765566 |

Thus:

```text
coarse decision-compression loss = established
```

---

## 3.3 E0-heavy coarse exposure is mostly an aggregation/count-bias effect

Research-1 established approximately:

```text
fine E0 ratio:
  Llama ~0.621
  Qwen  ~0.605

coarse N8K64 E0 exposure:
  Llama ~0.914
  Qwen  ~0.871
```

Counterfactual shuffle diagnostics indicate:

```text
count imbalance     = dominant
margin magnitude    = secondary
spatial clustering = small contribution
```

Do not restart broad E0-collapse diagnosis.

Do not assume E0 prevalence itself is causal harm.

---

## 3.4 Simple scale adaptation is not the current core recovery axis

Research-1 B0 was classified:

```text
not_useful
```

Simple 4/6 and 6/7/5-6-7 rules captured only partial sampled legal-scale headroom.

A full legal-scale model oracle is still technically unresolved, but it is not Research-3 Phase A priority.

---

## 3.5 Generic rotation is closed

Fixed H64 worsened full C4 on both LLMs.

Random SANA rotations were seed-fragile.

Research-3 must not reopen generic rotation sweeps.

---

# 4. Research-2 P0-A: packing is scientifically closed

Frozen Qwen foldable greedy packing:

| Corpus | Raw MSE-N8K64 | Packed | MSE-Oracle16 | NLL recovery |
|---|---:|---:|---:|---:|
| WikiText | 9.919024 | 9.89437 | 9.73649 | 13.4% |
| C4 | 13.765566 | 13.8362 | 13.65714 | -64.7% |

The map passed:

```text
equivalence
leakage checks
foldability
```

but C4 regressed and the preregistered 30% recovery target failed on both corpora.

Therefore:

```text
generic/global foldable N packing = CLOSED
```

Research-3 must not:

```text
run learned/Sinkhorn packing
run more greedy packing variants
search more permutation seeds
```

unless a future, independent mechanism provides compelling new evidence.

The active mechanism is K-coupled assignment, not N-layout packing.

---

# 5. Research-2 P0-B: coupled assignment is the strongest current result

## 5.1 FullLayer-Coarse-CD-N8K64 results

| Model | Corpus | PPL | NLL recovery vs coarse→fine |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | **6.565253** | **93.0%** |
| Llama-3.1-8B | C4 | **9.477595** | **78.2%** |
| Qwen3-8B | WikiText | **9.808126** | **60.5%** |
| Qwen3-8B | C4 | **13.725540** | **36.8%** |

All four paired CIs versus raw MSE-N8K64 excluded regression.

Pointwise, FullLayer-CD also beat:

```text
local OutputAware-N8K64 in 4/4
Canonical-4Over6 in 4/4
```

This establishes:

> fixed N8K64 is not intrinsically devoid of good assignments under the project-defined
> fake-quant semantics.

It does **not** establish a mathematical optimum.

---

## 5.2 The correct mathematical mechanism

For a linear layer:

\[
Y=XW^T.
\]

Partition weights into N8×K64 regions.

For N8 output stripe \(n\) and K64 input region \(k\):

\[
E_{n,k}^{f}
=
X_{K_k}
\left(
Q_f(W_{N_n,K_k})-W_{N_n,K_k}
\right)^T.
\]

Different N8 output stripes occupy disjoint output-channel coordinates.

Therefore:

\[
\|\Delta Y\|_F^2
=
\sum_n
\left\|
\sum_k E_{n,k}^{f_{n,k}}
\right\|_F^2.
\]

The important coupling is therefore:

```text
within one N8 output stripe
across its K64 regions
```

The local selector minimizes region terms independently:

\[
\min_f \|E_{n,k}^{f}\|_F^2.
\]

But the stripe objective is:

\[
J_n(F_n)
=
\left\|
\sum_k E_{n,k}^{f_{n,k}}
\right\|_F^2,
\]

which includes cross-K error interaction.

This is the core Research-3 scientific object.

---


## 5.3 Exact binary-quadratic form of one N8-stripe assignment

The same coupled objective can be written as a binary quadratic optimization problem.

For one N8 output stripe, let the two candidate output-error contributions for K64
region \(k\) be:

\[
E_k^{E2}, \qquad E_k^{E0}.
\]

Use E2 as the reference and define:

\[
D_k = E_k^{E0}-E_k^{E2}.
\]

Let:

\[
z_k\in\{0,1\},
\]

where:

```text
z_k = 0 -> E2
z_k = 1 -> E0
```

Define the all-E2 stripe residual:

\[
R_0=\sum_k E_k^{E2}.
\]

Then:

\[
R(z)=R_0+\sum_k z_kD_k.
\]

The exact stripe objective is:

\[
J(z)
=
\left\|
R_0+\sum_k z_kD_k
\right\|_F^2.
\]

Expanding gives:

\[
J(z)
=
c+2g^Tz+z^TQz,
\]

with:

\[
c=\|R_0\|_F^2,
\]

\[
g_k=\langle R_0,D_k\rangle_F,
\]

and:

\[
Q_{ij}=\langle D_i,D_j\rangle_F.
\]

Because \(Q\) is a Gram matrix:

\[
Q\succeq0.
\]

Thus fixed-N8K64 format assignment is a **binary quadratic optimization (BQP)**
over one format bit per K64 region.

This is an analytical characterization, not a novelty claim by itself.

The main scientific implication is that independent local selectors omit the
off-diagonal interaction terms \(Q_{ij}, i\neq j\), which directly encode
cross-K error cancellation and reinforcement.

---

## 5.4 Exact sufficient statistics for offline search

Once candidate error contributions have been constructed, the frozen calibration
objective can be evaluated from:

```text
c
g
Q
z
```

without repeatedly materializing the full calibration-token output-error tensor during
each format-bit search step.

For a stripe with \(r=K/64\) K64 regions:

```text
g: r values
Q: r x r values
z: r bits
```

For a single-bit flip \(z_i \leftarrow 1-z_i\), let:

\[
\delta_i = 1-2z_i\in\{-1,+1\}.
\]

The exact objective change is:

\[
\Delta J_i
=
2\delta_i\left(g_i+(Qz)_i\right)
+
Q_{ii}.
\]

This identity must be unit-tested against direct SSE recomputation.

Potential benefit:

```text
candidate-error construction remains expensive
but
coordinate-search/correction sweeps can become compact quadratic updates
```

Do not implement a new quadratic backend merely because it is possible.

Research-3 must first profile whether search/candidate-evaluation time is a meaningful
fraction of freeze cost.

---

## 5.5 Optimization-bound terminology

CD2 is not a mathematical oracle.

Research-3 may use the continuous box relaxation:

\[
0\le z_k\le1
\]

of:

\[
\min_z c+2g^Tz+z^TQz.
\]

Because \(Q\succeq0\), this is a convex quadratic program. Its **true optimum** satisfies:

\[
J_{\mathrm{relax}}^\*
\le
J_{\mathrm{binary}}^\*
\le
J_{\mathrm{CD2}}.
\]

However, a numerically computed feasible relaxation point is **not automatically a
certified lower bound** on the binary optimum.

Therefore use:

```text
certified_relaxation_lower_bound
```

only if the solver provides trustworthy optimality/convergence evidence for the convex
problem.

Otherwise label the value:

```text
approx_continuous_relaxation_objective
```

and use it only as a diagnostic, not as a formal lower bound.

For a certified relaxation solution, define a normalized remaining-gap diagnostic:

\[
RemainingGap_{\mathrm{relax}}
=
\frac{
J_{\mathrm{CD2}}-J_{\mathrm{relax}}^\*
}{
J_{\mathrm{init}}-J_{\mathrm{relax}}^\*+\epsilon
}.
\]

Lower is better.

A small exact/MIQP/enumeration audit is allowed only on preselected small/reduced
subproblems and with a strict time limit. Do not turn Research-3 into a solver-comparison
project.


# 6. Research-2 P1-A: ResidualAware has signal but should not automatically remain the primary method

Research-2 selected:

```text
ResidualAware-Sequential-N8K64
order = descending_local_sensitivity
```

with one family-neutral calibration-selected order.

Results:

| Model | Corpus | PPL | NLL recovery |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.575067 | 80.5% |
| Llama-3.1-8B | C4 | 9.486475 | 69.9% |
| Qwen3-8B | WikiText | 9.874326 | 24.3% |
| Qwen3-8B | C4 | 13.755790 | 9.0% |

It improves raw N8K64 point-estimate PPL in 4/4.

Paired CI vs raw N8K64 excludes regression in 3/4.

Qwen C4 CI crosses zero.

It beats Canonical-4Over6 by PPL in 3/4; Qwen WikiText is approximately 0.0203 PPL worse.

Thus:

```text
ResidualAware = real positive practical signal
ResidualAware = not yet universally robust
```

---

# 7. Critical Research-3 reinterpretation: FullLayer-CD may itself be the practical primary method

Research-2 called FullLayer-CD primarily a bounded upper-bound diagnostic and ResidualAware the primary deployable candidate.

That distinction is not yet scientifically justified.

Both methods have the same inference-time representation:

```text
format region = N8 x K64
scale group = K16
one static E2/E0 decision per region
no online transform
no runtime permutation
no extra GEMM
no extra online arithmetic
```

The difference is offline calibration/search.

Observed completed Research-2 freeze times on RTX 6000 Ada:

```text
FullLayer-CD Qwen 8B:
  ~20.33 min

FullLayer-CD Llama 8B:
  ~22.24 min

ResidualAware Qwen 8B:
  ~25.10 min

ResidualAware Llama 8B:
  ~24.45 min
```

These are implementation observations, not normalized algorithmic complexity measurements.

But they invalidate the untested assumption:

```text
CD is necessarily too expensive to be a method
while one-pass ResidualAware is necessarily cheaper
```

Therefore Research-3 must treat:

```text
FullLayer-CD2
FullLayer-CD1
ResidualAware
ResidualAware + correction
```

as competing **offline calibration algorithms with identical inference semantics**.

Research-3 must measure calibration cost fairly before declaring one "deployable" and another "diagnostic."

---

# 8. The most important unresolved scientific gap

Define:

```text
diagnostic/practical gap
=
FullLayer-CD quality
-
ResidualAware quality
```

PPL gap:

| Model | Corpus | CD | ResidualAware | RA − CD |
|---|---|---:|---:|---:|
| Llama | WikiText | 6.565253 | 6.575067 | +0.009814 |
| Llama | C4 | 9.477595 | 9.486475 | +0.008880 |
| Qwen | WikiText | 9.808126 | 9.874326 | +0.066200 |
| Qwen | C4 | 13.725540 | 13.755790 | +0.030250 |

The gap is substantially larger on Qwen.

This is now the highest-information clue.

Possible mechanisms:

### H1 — path dependence / inability to revisit early decisions

One-pass sequential assignment greedily commits formats.

Coordinate descent revisits decisions.

If true:

```text
ResidualAware + one correction sweep
```

should close much of the gap.

### H2 — initialization matters

Research-2 FullLayer-CD used:

```text
initialization = OutputAware-N8K64
2 sweeps
```

while ResidualAware constructs assignments sequentially.

If initialization is important, CD1/CD2 from different starts will expose it.

### H3 — calibration objective misses cross-layer propagation

Research-2 P1-A used:

```text
independent FP layer inputs
```

At inference, later layers receive activations from an already-quantized prefix.

This mismatch may matter more for Qwen.

### H4 — calibration distribution/shard sensitivity

Research-2 used one fixed C4 calibration manifest.

The Qwen weakness may reflect map instability across calibration subsets.

### H5 — the remaining Qwen gap is intrinsic to the available static coarse decisions

Even FullLayer-CD only recovers 36.8% on Qwen C4.

A substantial part of the fine-vs-coarse gap may remain irreducible under the fixed format region.

Research-3 must distinguish H1–H5.

---

# 9. Research-3 central hypothesis

The current strongest falsifiable hypothesis is:

> **Coarse-format loss is substantially an assignment-optimization problem rather than
> only a representation-capacity problem. A coupled offline optimizer that explicitly
> models cross-K error interaction within N8 stripes can recover the fine-format value
> while preserving the exact same inference-time N8K64 representation.**

A secondary hypothesis is:

> **The remaining Qwen weakness comes from greedy/path-dependent search and/or
> cross-layer calibration mismatch, not from a need for additional runtime metadata.**

These must be tested before broad benchmark expansion.

---

# 10. Research-3 Phase A — METHOD CLOSURE
# Mandatory before external generalization

Research-3 Phase A is the most important phase.

Do not run SANA, third-model expansion, W4A4, or native work before the Phase-A gate.

---

# 11. A0 — Handoff audit and exact statistical closure

## 11.1 Required audit

Before new experiments:

1. read current `research_3.md`;
2. compute SHA256;
3. create isolated Research-3 worktree/branch;
4. verify Research-2 final artifact hashes;
5. regenerate the Research-2 core result table directly from immutable sequence-level artifacts;
6. confirm the current frozen maps and calibration manifests;
7. audit the exact Research-2 ResidualAware order-selection process;
8. audit P0-B trace integrity and sweep convergence;
9. confirm Research-2 B1/P1 inputs used independent FP layer activations;
10. confirm no evaluation data entered format-map selection.

## 11.2 Missing paired comparisons to compute

Research-2 emphasized CI vs raw coarse.

Research-3 must additionally compute paired per-sequence NLL comparisons:

```text
FullLayer-CD vs ResidualAware
FullLayer-CD vs Canonical-4Over6
ResidualAware vs Canonical-4Over6
FullLayer-CD vs LocalOutputAware
```

for all four original model/corpus pairs.

Do not rely on PPL point estimates alone.

Report:

```text
mean ΔNLL
median ΔNLL
95% paired bootstrap CI
win fraction
effect size per token
```

This establishes whether FullLayer-CD's pointwise 4/4 advantage over 4Over6 is statistically reliable.

---

# 12. A1 — Fair offline-calibration cost comparison

## 12.1 Question

Is FullLayer-CD actually impractical relative to ResidualAware?

Research-2 timing suggests no, but the scripts were not designed as a normalized performance study.

## 12.2 Compare

On the same GPU type, same model, same calibration manifest, same row budget:

```text
OutputAware freeze
ResidualAware freeze
FullLayer-CD1 freeze
FullLayer-CD2 freeze
ResidualAware + Refine1 freeze
```

At minimum:

```text
Llama-3.1-8B
Qwen3-8B
```

Use one fixed RTX 6000 Ada when possible.

## 12.3 Record

```text
wall time
GPU time
peak GPU memory
peak host memory
calibration activation cache size
format map size
number of candidate format evaluations
number of accepted flips
number of sweeps
I/O time
initialization time
search time
evaluation time
```

If scripts have different avoidable Python overhead, profile both:

```text
observed implementation cost
estimated algorithmic kernel cost
```

Do not hide implementation inefficiency.

## 12.4 Interpretation

If FullLayer-CD2 is:

```text
<= 2x ResidualAware calibration wall time
```

and retains clearly better model quality, it is valid to treat CD2 as a practical offline PTQ method.

The 2x criterion is a project design rule, not a universal scientific law.

If CD1 approaches CD2 quality with lower cost, prefer CD1.

### A1 conditional optimization rule

Break freeze time into:

```text
candidate error / activation construction
I/O
initialization
coordinate-search / candidate evaluation
final serialization
```

Only if coordinate-search/candidate evaluation is a meaningful fraction of total cost,
implement the compact \(c,g,Q,z\) sufficient-statistics backend from Section 5.4.

If search is already negligible relative to error construction, do **not** spend Research-3
time optimizing the search backend.

If implemented, require:

```text
quadratic objective == direct stripe SSE
quadratic flip delta == direct recomputation delta
frozen map == reference search map under identical tie-breaking
```

within declared numerical tolerance.

---

# 13. A2 — Explain the CD-vs-ResidualAware gap

This is mandatory.

## 13.1 Assignment-map diagnostics

For each model/module/N8 stripe compare:

```text
MSE-N8K64
OutputAware-N8K64
ResidualAware
FullLayer-CD1
FullLayer-CD2
```

Metrics:

```text
format Hamming disagreement
E0 exposure
number of flips from initialization
sweep-1 flip count
sweep-2 flip count
per-stripe objective J_n
held-out per-stripe J_n
residual norm
cross-term cancellation gain
local candidate-error sum
full stripe error
```

Define:

\[
J_n(F_n)
=
\left\|
\sum_k E_{n,k}^{f_{n,k}}
\right\|_F^2.
\]

Define independent-error sum:

\[
L_n(F_n)
=
\sum_k
\|E_{n,k}^{f_{n,k}}\|_F^2.
\]

Define cross interaction:

\[
C_n(F_n)
=
J_n(F_n)-L_n(F_n).
\]

Interpretation:

```text
C_n < 0:
  errors cancel constructively for total accuracy

C_n > 0:
  region errors reinforce each other
```

Use this only as a decomposition of the chosen objective; do not overclaim causal independence.

## 13.2 Search-gap metric

For method \(m\):

\[
SearchGap_n(m)
=
J_n(F_n^m)-J_n(F_n^{CD2}).
\]

Aggregate by:

```text
module type
layer depth
model
conflict level
E0 exposure
K-region count
```

Question:

> Is Qwen dominated by a small number of hard stripes/modules or by broad mild search error?

This determines whether selective refinement is possible.

## 13.3 Pairwise-interaction diagnostics

From the BQP Gram matrix \(Q\), compute for representative/high-conflict stripes:

```text
num_k64_regions
pairwise_coupling_abs_sum
pairwise_coupling_offdiag_ratio
negative_coupling_fraction
positive_coupling_fraction
largest_abs_pairwise_coupling
interaction_matrix_spectral_norm
```

Suggested definitions:

\[
CouplingRatio
=
\frac{
\sum_{i\neq j}|Q_{ij}|
}{
\sum_iQ_{ii}+\epsilon
}.
\]

\[
NegativeCouplingFraction
=
\frac{
\sum_{i<j,Q_{ij}<0}|Q_{ij}|
}{
\sum_{i<j}|Q_{ij}|+\epsilon
}.
\]

Interpretation:

```text
large off-diagonal coupling:
  independent local assignment is structurally questionable

large negative interaction mass:
  cancellation opportunity exists

large positive interaction mass:
  some candidate switches reinforce error
```

These are calibration-objective diagnostics, not causal decompositions of end-to-end PPL.

Also test whether:

```text
coupling metric
correlates with
MSE/OA/RA search gap to CD2
```

at stripe/module level.

Use rank correlations and binned plots; do not claim a universal predictor from two models.

## 13.4 Bounded CD2 optimality/headroom audit

On a predeclared representative subset of stripes:

1. construct \(c,g,Q\);
2. solve the continuous box relaxation if a suitable convex-QP solver is available;
3. record solver status, residual/KKT or other convergence diagnostics;
4. use the result as a certified lower bound only when solver evidence supports that label;
5. otherwise record only an approximate relaxation diagnostic.

Optional exact/MIQP audit is allowed only for small/reduced subproblems with a strict
time limit.

The purpose is:

```text
measure whether CD2 still has obvious assignment-optimization headroom
```

not:

```text
find a publishable generic optimizer
```

If CD2 is close to a certified relaxation bound on most representative stripes, stop
optimizer fishing.

If the remaining gap is large specifically on Qwen hard stripes, retain one bounded
optimization refinement path.

---

# 14. A3 — Minimal correction-sweep study

This is the first new algorithmic experiment.

Do **not** create a large optimizer bank.

Compare:

```text
OA-CD1:
  OutputAware initialization + 1 coordinate sweep

OA-CD2:
  OutputAware initialization + 2 coordinate sweeps
  = Research-2 FullLayer-CD reference

RA:
  existing ResidualAware sequential map

RA-Refine1:
  ResidualAware map + 1 coordinate correction sweep
```

Optional only if A2 indicates need:

```text
RA-Refine2
```

Do not run more than two correction sweeps.

### A3 conditional best-improvement CD

Do **not** run this by default.

Unlock exactly one deterministic `best-improvement CD` variant only if all are true:

```text
A2/A3 demonstrates real path dependence
AND
cyclic correction needs repeated revisits
AND
search cost is material
AND
RA-Refine1 does not already close the gap
```

At each step, evaluate the exact single-bit flip deltas from the quadratic sufficient
statistics and apply the largest improving flip.

No random-order bank.

### A3 conditional hard-stripe refinement

First use CD2-derived quantities only to diagnose whether the Qwen gap is concentrated.

Important distinction:

```text
diagnostic hard-stripe metrics may use CD2-derived search gap / disagreement
but
a practical cost-saving screen may NOT require CD2 to be run first
```

If concentration exists, a deployable/offline-efficient selective refinement may use
only **pre-CD cheap predictors**, for example:

```text
pairwise coupling ratio
initial/local held-out stripe objective
initial format ambiguity
cheap one-pass residual statistics
```

Thresholds must be chosen on calibration train/validation.

Do not claim a CD-cost saving method if its screening criterion requires already running
full CD2 over the whole model.

## 14.1 Why RA-Refine1 is high-value

If:

```text
RA-Refine1 ≈ CD2
```

then the major problem is simply irreversible greedy decisions.

The practical method can be:

```text
sequential build
+
one offline correction pass
```

with unchanged inference semantics.

If:

```text
OA-CD1 ≈ CD2
```

then the simplest method may just be one coordinate-descent sweep from a strong local initializer.

If:

```text
only CD2 works
```

but calibration cost is still reasonable, CD2 itself may be the method.

## 14.2 Run order

First:

```text
Qwen3-8B C4
Qwen3-8B WikiText
```

because Qwen has the largest diagnostic/practical gap.

Then run Llama as a negative/control validation only for variants that show Qwen signal.

## 14.3 Selection discipline

Choose the algorithm using:

```text
calibration train split
+
held-out calibration validation split
```

not final evaluation sequences.

Freeze before WikiText/C4 final evaluation.

---

# 15. A4 — Sequential teacher-aligned / quantized-prefix calibration

This experiment is mandatory if A3 does not already produce a robust winner, and highly recommended even if it does.

Research-2 explicitly leaves this unresolved.

## 15.1 Current limitation

Current calibration uses:

```text
X_l = FP-prefix layer input
```

for each layer independently.

Real quantized inference uses:

```text
X_l^Q = activation produced by quantized layers < l
```

The selector therefore may optimize the wrong input distribution.

## 15.2 Teacher-aligned objective

For layer \(l\), cache:

```text
X_l^Q
  = input from already-quantized prefix

Y_l^*
  = full-precision teacher output for the same original calibration samples
```

Before current-layer weight quantization:

\[
B_l
=
X_l^Q W_l^T
-
Y_l^*.
\]

This is the upstream/prefix mismatch already present before quantizing \(W_l\).

For region assignment \(F_l\):

\[
\Delta Y_l(F_l)
=
X_l^Q
\left(Q_{F_l}(W_l)-W_l\right)^T.
\]

Optimize:

\[
J_l^{teacher}(F_l)
=
\left\|
B_l+\Delta Y_l(F_l)
\right\|_F^2.
\]

For an N8 stripe:

\[
J_{l,n}^{teacher}
=
\left\|
B_{l,n}
+
\sum_k E_{l,n,k}^{f_{n,k}}
\right\|_F^2.
\]

This preserves the same N8-stripe decomposition.

## 15.3 Why this matters

It separates:

```text
within-layer cross-K interaction
```

from:

```text
cross-layer accumulated quantization drift
```

The latter is a plausible explanation for Qwen C4 weakness.

## 15.4 Bounded variants

Only apply teacher alignment to the best one or two Phase-A search algorithms:

```text
Teacher-CD1 or Teacher-CD2
Teacher-RA-Refine1
```

Do not cross every search algorithm with teacher alignment.

## 15.5 First target

Run first on:

```text
Qwen3-8B C4
Qwen3-8B WikiText
```

Promote to Llama only if Qwen improves on held-out calibration and frozen evaluation.

## 15.6 Novelty caution

Sequential quantized-prefix / teacher-aligned calibration is already adjacent to established residual/error-compensation PTQ literature.

Do not claim teacher alignment itself as novel.

If successful, the project-specific claim is:

> **teacher/residual-aware optimization of discrete E2/E0 format bits under coarse N8K64 format-control constraints.**

---

# 16. A5 — Calibration robustness and overfitting study

A method cannot be promoted based on one C4 calibration shard.

## 16.1 Calibration shard robustness

For the final two candidate algorithms at most:

```text
3 independent C4 calibration shards/seeds
```

Use fixed predeclared sample counts.

Recommended:

```text
128 sequences each
```

if resources permit.

At minimum:

```text
64 sequences each
```

for the initial robustness screen.

## 16.2 Measure

Across calibration shards:

```text
format-map agreement
per-layer map agreement
per-stripe map agreement
E0 exposure variation
held-out layer-output objective
held-out model NLL
final WikiText PPL/NLL
final C4 PPL/NLL
```

## 16.3 Calibration-size scaling

On Qwen only, for the eventual candidate:

```text
32
64
128
```

sequences.

Choose the smallest size whose held-out calibration performance is within a preregistered tolerance of 128.

Do not choose calibration size using final evaluation.

## 16.3.1 If shard instability appears: one conservative robustness mechanism only

Do not start a robust-learning project.

First baseline:

\[
J_{\mathrm{mean}}(F)
=
\frac1S\sum_sJ_s(F),
\]

implemented by pooled/multi-shard calibration when mathematically equivalent.

Alternative conservative flip rule:

```text
accept flip if:
  train ΔJ < -epsilon_train
  AND
  validation ΔJ <= epsilon_val
```

Choose at most one of these after observing genuine shard instability.

Do not run:

```text
learned robust controller
large lambda grid
distributionally robust neural selector
```

## 16.3.2 Interpret map disagreement carefully

Low map Hamming agreement does not automatically mean method instability.

Different maps can have nearly identical:

```text
held-out stripe objective
model NLL
```

Therefore report separately:

```text
map stability
objective stability
model-quality stability
```

If map disagreement is large, optionally report:

\[
EquivalentMapRate(\epsilon)
=
P\left(
\frac{|J(F_a)-J(F_b)|}{|J(F_a)|+\epsilon_0}<\epsilon
\right).
\]

Use this only as a diagnostic.

## 16.4 Cross-distribution calibration

Current C4 calibration already transfers positively to WikiText.

Do not automatically build a large dataset bank.

Only if shard robustness or Qwen C4/Wiki behavior remains inconsistent, compare:

```text
C4 calibration
vs
one Wiki-like calibration source
```

using a held-out calibration selection protocol.

---


# 16B. Optional bounded model-level attribution
# Run only after a near-final Phase-A candidate exists

This is a mechanism experiment, not a routing method.

Start from raw MSE-N8K64.

Rank modules using **calibration-only** coupled-objective improvement from the near-final
Research-3 assignment.

Replace raw assignments by Research-3 assignments in predeclared cumulative groups:

```text
top 10%
top 25%
top 50%
all
```

Measure held-out/model NLL.

Report:

```text
cumulative calibration-objective recovery
cumulative NLL recovery
module family/depth composition
```

Questions:

> Are gains broadly distributed or driven by a small number of sensitive modules?

> Does coupled-objective gain rank model-important recovery opportunities?

Do not choose module subsets using final evaluation.

Do not promote this into a module router inside Research-3 unless a separate future study
is explicitly opened.


# 17. A6 — Freeze the Research-3 primary algorithm

Candidate names are internal and provisional:

```text
FullLayer-CD2-N8K64
FullLayer-CD1-N8K64
ResidualAware-Refine1-N8K64
TeacherAligned-CD1-N8K64
TeacherAligned-CD2-N8K64
TeacherAligned-ResidualRefine1-N8K64
```

Do not expose a paper name yet.

The selected method must be frozen before Phase B.

---


## 17.1 Primary-method preference order

After normalized cost, statistics, and robustness are available:

```text
Case A:
OA-CD1 matches CD2
-> prefer FullLayer-CD1-N8K64

Case B:
RA-Refine1 matches CD2 and is cheaper
-> prefer ResidualAware-Refine1-N8K64

Case C:
only CD2 is robustly strong and offline cost is acceptable
-> use FullLayer-CD2-N8K64

Case D:
teacher alignment materially improves Qwen without harming Llama
-> choose the simplest teacher-aligned variant satisfying all gates

Case E:
no candidate reaches strict 4/4 + robustness gate
-> do not broaden Phase B
```

Do not reject CD2 merely because it is a two-sweep coordinate method.

Offline PTQ calibration cost must be measured, not inferred from method names.


# 18. Phase-A decision gate

A Research-3 method is `METHOD_GO` if, on the original four settings:

```text
1. improves raw MSE-N8K64 in 4/4 by point estimate;
2. paired CI vs raw N8K64 excludes meaningful regression in 4/4;
3. achieves >=30% coarse-to-fine NLL recovery in all 4;
4. beats or statistically ties Canonical-4Over6 in all 4;
5. preferably beats Canonical-4Over6 by point estimate in >=3/4;
6. is robust across calibration shards;
7. uses no evaluation tuning;
8. inference semantics remain unchanged;
9. offline calibration cost is measured and acceptable.
```

The >=30% recovery threshold is an internal design target.

### Important nuance

Existing Research-2 CD2 already has recovery:

```text
93.0%
78.2%
60.5%
36.8%
```

so it already clears the recovery threshold.

Research-3 must determine:

```text
statistical advantage vs strong baselines
calibration robustness
offline cost
cross-layer robustness
```

before declaring it a method.

## Phase-A stop/downgrade rules

If:

```text
CD2 is not robust across calibration shards
AND
no correction/teacher variant fixes Qwen
```

then downgrade to:

```text
CONDITIONAL_GO_WEAK
```

and do not launch a large benchmark campaign.

If only Llama remains strong while Qwen is unstable:

```text
do not claim universal coarse-assignment recovery
```

If all practical methods fail but CD2 remains a stable, modest-cost offline method:

```text
CD2 may itself remain the primary candidate
```

provided its calibration overhead is explicitly reported.

---

# 19. Research-3 Phase B — GENERALIZATION
# Run only after METHOD_GO

Phase B tests whether the mechanism is broader than the two original 8B cases.

---

# 20. B0 — Baseline and literature implementation audit

Before expanding models, audit reproducible baselines.

Mandatory conceptual baselines:

```text
HighPrecision
NVFP4
Canonical-4Over6
raw MSE-N8K64
MSE-Oracle16 reference
Local OutputAware-N8K64
Research-3 frozen method
```

Also audit whether official/reproducible implementations are available for:

```text
IF4 / Adaptive Block-Scaled Data Types
published MixFP4
BlockDialect
AdaMX
```

Only run external baselines if:

```text
official or author code is available
semantics can be matched
model/eval configuration is reproducible
```

Do not invent an approximate external baseline and label it official.

If a baseline differs in:

```text
scale semantics
block size
metadata
format candidates
runtime arithmetic
```

report those differences explicitly.

---

# 21. B1 — Third LLM architecture

Two 8B families are not sufficient for a broad claim.

Pre-register one third model family before viewing results.

Selection criteria:

```text
7B–10B class preferred
existing harness support
weights/model access available
no major custom architecture code
fits available GPU memory
not another Llama derivative if avoidable
```

Suggested candidates, subject to repository support:

```text
Mistral 7B family
or
Gemma 2/3 ~9B family
```

Do not choose whichever gives the better result after testing both.

If both are cheap, pre-register:

```text
primary third family
secondary confirmation family
```

before results.

Run:

```text
WikiText
C4
```

with the same frozen Research-3 algorithm and calibration policy.

---

# 22. B2 — Model-size scaling

If the third-family result is positive, test one larger model.

Target:

```text
~14B class
```

when feasible on the available GPUs.

Selection criteria:

```text
same or compatible harness
memory feasible
calibration feasible
no semantic changes to quantizer
```

Possible examples, subject to access/support:

```text
Qwen ~14B
or another supported 12B–14B model
```

Do not jump to 70B if it requires a qualitatively different infrastructure before the method is stable.

The purpose is:

```text
does the assignment mechanism survive model scale?
```

not benchmark count.

---

# 23. B3 — Proper downstream LLM evaluation

Research-1 downstream work was only a small sanity screen.

After the method is frozen, run meaningful downstream evaluation.

Use the existing supported evaluation harness.

Pre-register a task set before results.

Recommended classes:

```text
commonsense reasoning
reading/knowledge
reasoning/problem solving
```

Example fixed task set if supported:

```text
HellaSwag
PIQA
ARC-Challenge
WinoGrande
BoolQ
MMLU or a fixed MMLU subset/full suite depending cost
```

Do not change tasks after seeing results.

Compare at minimum:

```text
HighPrecision
NVFP4
Canonical-4Over6
raw MSE-N8K64
Research-3 method
```

MSE-Oracle16 is diagnostic and optional for expensive downstream evaluation.

Report:

```text
task accuracy
mean accuracy
per-task delta
confidence/bootstrap where supported
```

---

# 24. B4 — SANA W4A16 cross-domain validation

Research-2 did not test the new coupled selector on SANA.

Run only after the LLM method is frozen.

## Stage 1 — proxy

Compare:

```text
HighPrecision
NVFP4
MSE-Oracle16
raw MSE-N8K64
Canonical-4Over6
Research-3 method
```

Use the exact fixed prompt/seed/calibration manifests from Research-1 where possible.

Metrics:

```text
proxy MSE
proxy NMSE
relative L2
cosine error
latent trajectory NMSE
```

No retuning on diffusion prompts.

## Stage 2 — 128-image screen

Only if the proxy is not clearly negative.

Run fixed 128-image evaluation.

Metrics:

```text
LPIPS
PSNR
SSIM
CLIP
ImageReward
```

Interpret as a Pareto set.

Do not require every metric to move in the same direction, but do not hide semantic regressions behind distortion metrics.

## Stage 3 — 1024 images

Only if the frozen method is clearly competitive/non-dominated on the 128-image screen.

---

# 25. B5 — W4A4 survival test

A weight-only W4A16 paper may be incomplete for a Blackwell-oriented story.

However activation recovery is not the current core method.

Therefore first test whether the **weight-side coupled assignment remains useful when activation quantization noise is present**.

Freeze activation policy to a strong canonical choice:

```text
activation = fixed E2 / NVFP4-compatible / canonical 4Over6
```

depending the existing implementation.

Compare weights:

```text
NVFP4/fixed E2
raw N8K64
Canonical-4Over6
Research-3 coupled method
```

Do not invent a new activation selector in the same experiment.

If the weight-side benefit survives A4 noise, then a later activation-specific project/extension is justified.

If it disappears, report the limitation honestly.

---

# 26. Research-3 Phase C — DEPLOYMENT / HARDWARE
# Gated after algorithmic generalization

## 26.1 Offline calibration cost

Required even without SM120:

```text
calibration wall time
peak memory
calibration samples
number of search sweeps
map-generation throughput
stored metadata size
```

Compare to strong offline PTQ baselines where possible.

## 26.2 Inference representation

The Research-3 main method must preserve:

```text
one legal E2/E0 decision per N8K64 region
K16 scales
no extra GEMM
no online transform
no runtime permutation
no dynamic model-dependent controller
```

If any method violates this, classify it separately.

## 26.3 Native SM120

Native work remains:

```text
WAIT_FOR_SM120
```

until actual compatible hardware exists.

When available, Phase C must verify separately:

```text
native E0 semantics
format-bit encoding
scale semantics
memory layout
decode path
Tensor Core compatibility
latency
throughput
power/overhead if measurable
```

Fake/reference accuracy is not native proof.

---

# 27. Research-3 novelty and literature guardrails
# Current to 2026-08-09 planning audit

The novelty bar is high.

Nearest known/current overlap includes:

## 27.1 Adaptive Block-Scaled Data Types / IF4 — arXiv:2603.28765

Already uses:

```text
FP4 vs INT4 per group of 16
E4M3 block scaling
format metadata reuse
hardware MAC study
```

Therefore do not claim:

```text
adaptive FP4/INT4 at K16
```

as novel.

## 27.2 MixFP4 — arXiv:2605.31035

Already uses adaptive low-bit micro-format selection under NVFP4-like scaling with no additional metadata.

Therefore do not frame novelty as:

```text
adaptive FP4 micro-format selection
```

or use the public name MixFP4 for a semantically different method.

## 27.3 BlockDialect — arXiv:2501.01144

Already studies per-block mixed-format quantization from a formatbook.

Therefore:

```text
different blocks prefer different number formats
```

is not a sufficient contribution.

## 27.4 AdaMX — arXiv:2608.03867

Already studies:

```text
block-level representation heterogeneity
precision-recovery heterogeneity
weight/activation asymmetry
hardware implementation
```

This substantially raises the novelty threshold.

## 27.5 dMX — arXiv:2606.04115

Already studies differentiable hardware-compatible mixed-precision format assignment at layer granularity.

Do not frame module/layer routing generically as novel.

## 27.6 Residual/error compensation literature

GPTQ and newer residual-compensation work already establish:

```text
sequential/error-aware calibration can improve PTQ
```

Recent work also explicitly studies quantized-prefix drift and teacher alignment.

Therefore:

```text
"we use residual compensation"
```

is not the contribution.

---

# 28. Strongest defensible novelty center

If Research-3 succeeds, the strongest remaining conceptual center is:

> **Coarse hardware format-control decision compression.**
>
> Fine K16 blocks prefer different representations, but hardware-oriented execution
> requires one shared format decision over a much coarser N8K64 region. Local selection
> loses quality because the coarse decision must account for coupled cross-K error.
> We formulate and optimize this discrete coarse-format assignment while preserving the
> exact same low-bit inference representation.

Potential components:

```text
1. quantify fine-to-coarse decision-compression loss;
2. expose local-MSE/local-output objective mismatch;
3. formulate N8-stripe coupled assignment;
4. optimize with bounded offline coordinate/residual search;
5. preserve inference-time metadata/arithmetic;
6. show robustness/generalization.
```

Do not overclaim that no prior work considers coupled quantization decisions.

The claim must be specific to:

```text
discrete format-bit assignment
+
coarse hardware format-control regions
+
fixed nominal bitwidth
+
same inference representation
```

A limited current literature search did not reveal an obvious exact duplicate of this formulation, but absence from a bounded search is not proof of novelty.

A full paper-writing literature audit is still required.

---

# 29. Potential paper-method organization if successful

Do not finalize names now.

A coherent method family could be:

```text
CoupledFormat-CD2:
  accuracy-first offline coupled assignment

CoupledFormat-CD1:
  lower calibration-cost one-sweep variant

CoupledFormat-Teacher:
  optional prefix-aware variant if cross-layer drift materially helps
```

ResidualAware sequential may remain:

```text
ablation / fast constructive initializer
```

rather than the primary method if CD is equally or more practical.

The paper should not artificially force the one-pass method to be primary.

---

# 30. Research-3 decision hierarchy

## 30.1 METHOD_GO

Require Phase A robust frozen method.

## 30.2 ALGORITHMIC_STRONG_GO

Require:

```text
METHOD_GO
+
third LLM family positive
+
larger-model or strong scale evidence
+
proper downstream evaluation
+
no major calibration instability
```

## 30.3 CROSS_DOMAIN_GO

Require:

```text
ALGORITHMIC_STRONG_GO
+
SANA W4A16 non-dominated/positive evidence
```

## 30.4 DEPLOYMENT_READY_FOR_NATIVE

Require:

```text
stable algorithm
offline cost characterized
W4A4 survival characterized
native implementation plan complete
```

Native proof itself still requires SM120.

## 30.5 FINAL_NO_GO

Use if the central method fails scientifically, for example:

```text
FullLayer-CD not robust across calibration shards
AND
no bounded correction/teacher method stabilizes Qwen

or

benefit disappears on a pre-registered third architecture

or

strong baseline comparison shows no meaningful advantage

or

method requires evaluation-specific tuning

or

the apparent gain is explained by an unfair semantic/baseline mismatch
```

A single negative SANA result does not necessarily invalidate an LLM-only paper, but it invalidates a cross-domain claim.

---

# 31. Pre-registered success targets

These are project design targets, not universal scientific laws.

## Original 4 settings

Target frozen method:

```text
raw N8K64 improvement: 4/4
CI vs raw regression excluded: 4/4
coarse-to-fine NLL recovery: >=30% in 4/4
Canonical-4Over6: beat/tie in 4/4
no major cross-corpus sign reversal
```

## Calibration robustness

Target:

```text
3 calibration shards
same sign of model-quality gain
no single-shard catastrophic regression
reasonable format-map stability
```

Map agreement need not be near 100% if multiple assignments have equivalent objective.

Therefore also compare held-out objective and PPL stability.

## Third model

Target:

```text
positive raw-coarse recovery
no statistically supported regression vs strong canonical baseline
```

If third model is clearly negative, re-evaluate universality.

---

# 32. Statistics

Primary LLM paired variable:

```text
per-sequence NLL
```

For method \(m\):

\[
\Delta_m=NLL_m-NLL_{baseline}.
\]

Report:

```text
mean
median
95% paired bootstrap CI
win fraction
PPL
```

Recovery:

\[
Recovery_m
=
\frac{
NLL_{coarse}-NLL_m
}{
NLL_{coarse}-NLL_{fine}
}
\]

only if:

\[
NLL_{coarse}>NLL_{fine}.
\]

For comparisons to CD:

\[
CDApproxRecovery_m
=
\frac{
NLL_{coarse}-NLL_m
}{
NLL_{coarse}-NLL_{CD2}
}
\]

when denominator is positive.

This measures how much of CD2's available coarse recovery a cheaper approximation captures.

Do not call it fine-oracle recovery.

---

# 33. Calibration/evaluation separation

Every new map must record:

```text
calibration dataset
calibration sample IDs
calibration manifest hash
train/validation split IDs
evaluation sample IDs
evaluation hash
selector/search configuration
order
initialization
sweeps
seed
```

Forbidden:

```text
choose sweeps from Wiki/C4 eval PPL
choose calibration size from eval
choose third model after trying several
choose SANA policy from final image metrics
report best-of-eval map
```

If multiple candidates are explored, choose using calibration validation and freeze.

---

# 34. GPU policy

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum concurrent project GPUs:

```text
3
```

Before every GPU process launch:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi pmon -c 1
```

For visible PIDs:

```bash
ps -o user=,pid=,cmd= -p <PID>
```

Rules:

```text
occupied by another user => unavailable
unknown owner + occupied => unavailable
low utilization != free
never share another user's GPU
never kill another user's process
never broad pkill
use project reservation lock
immediately re-check before launch
log every admission/rejection
```

Research-3 does not require SM120.

---

# 35. Git/repository policy

Before modifications record:

```text
repo URL
branch
HEAD SHA
Research-2 result commit
Research-2 aggregation commit
submodule SHAs
dirty state
local diff hash
```

Expected Research-2 reported commits:

```text
formal result:
248ae3af03b7a4bfc8fa4f327aacd7e4f1c35b3f

final aggregation:
ce466c9438d60d0435656e498767a3969781bc8a
```

Create an isolated Research-3 branch/worktree.

Do not push unless explicitly requested.

Do not rewrite immutable Research-1/2 artifacts.

---

# 36. Mandatory unit/regression tests

## 36.1 Quantization semantics

Re-run or reuse only with matching fingerprints:

```text
E2 codebook validation
project E0 sign-magnitude validation
K16 scale-group invariant
N8K64 format-region invariant
tail handling
fixed-E2 baseline equivalence
canonical 4Over6 reduction
```

## 36.2 Coupled objective

Test:

```text
direct full layer SSE == stripe-decomposed SSE
cross-N8 stripe Frobenius cross-term == 0
incremental residual update == full recomputation
accepted coordinate flip never increases objective
CD1 objective <= initialization
CD2 objective <= CD1
deterministic tie breaking

BQP c/g/Q objective == direct stripe SSE
BQP single-bit flip delta == direct recomputation
Q symmetry within tolerance
Q PSD sanity within numerical tolerance
quadratic-backend map == reference map under matched tie-breaking
```

If continuous relaxation is used:

```text
record solver status
record convergence/optimality diagnostics
never mark "certified lower bound" without solver evidence
```

## 36.3 Teacher-aligned objective

Synthetic test:

```text
B_l = X_Q W^T - Y_FP
B_l + X_Q(Q(W)-W)^T
==
X_Q Q(W)^T - Y_FP
```

within tolerance.

Test stripe decomposition with nonzero base residual \(B_l\).

## 36.4 Freeze/replay

For every selected map:

```text
freeze -> save -> reload -> bit-identical assignment
evaluation rerun deterministic within declared tolerance
```

---

# 37. Required Research-3 artifact tree

```text
artifacts/research_3/

00_environment/
  spec_acknowledgement.md
  spec_manifest.json
  research2_handoff_audit.md
  research2_source_hashes.json
  repo_manifest.json
  environment.txt
  patch_manifest.json
  gpu_usage_log.jsonl
  literature_guardrail_notes.md

01_tests/
  quant_semantics_regression.json
  stripe_objective_tests.json
  incremental_residual_tests.json
  teacher_aligned_objective_tests.json
  freeze_replay_tests.json
  paired_statistics_tests.json

02_phase_a_method_closure/
  a0_statistical_closure/
  a1_calibration_cost/
  a2_search_gap_diagnostics/
    interaction_matrix_summary.csv
    hard_stripe_summary.csv
    optimization_gap_summary.csv
    representative_interaction_matrices.npz
  a3_correction_sweep/
  a4_teacher_aligned/
  a5_calibration_robustness/
  a6_frozen_method/

03_phase_b_generalization/
  b0_baseline_audit/
  b1_third_model/
  b2_scale_model/
  b3_downstream/
  b4_sana/
  b5_w4a4_survival/

04_phase_c_deployment/
  offline_cost/
  metadata_cost/
  native_handoff/

05_final/
  experiment_manifest.jsonl
  failed_runs.jsonl
  master_results.csv
  method_comparison.csv
  calibration_robustness.csv
  final_decision.json
  final_decision_report.md
  results_summary.md
  limitations.md
  reproduction_commands.sh
  paper_claims_boundary.md
  native_sm120_handoff.md
```

Preserve all failed attempts.

Terminal output is not the source of truth.

---

# 38. Required machine-readable fields

Common:

```text
experiment_id
phase
model
model_revision
corpus
method
repo_sha
config_hash
calibration_manifest
calibration_hash
calibration_train_hash
calibration_val_hash
eval_hash
seed
gpu_physical_id
gpu_uuid
status
start_time
end_time
```

Coupled method:

```text
initialization
order
num_sweeps
num_candidate_evaluations
num_format_flips
sweep1_flips
sweep2_flips
format_map_hash
weight_e0_ratio
```

Objective:

```text
stripe_id
module_name
J_full_stripe
L_independent_sum
C_cross_interaction
residual_norm
heldout_J
search_gap_vs_cd2
format_hamming_vs_cd2

num_k64_regions
pairwise_coupling_abs_sum
pairwise_coupling_offdiag_ratio
negative_coupling_fraction
positive_coupling_fraction
largest_abs_pairwise_coupling
interaction_matrix_spectral_norm

continuous_relaxation_status
continuous_relaxation_objective
continuous_relaxation_certified
cd1_gap_to_relaxation
cd2_gap_to_relaxation
ra_gap_to_relaxation
ra_refine1_gap_to_relaxation

hard_stripe_flag
hard_stripe_reason
```

If a compact quadratic backend is implemented:

```text
interaction_precompute_seconds
quadratic_search_seconds
quadratic_matrix_bytes
objective_reconstruction_error
flip_delta_reconstruction_error
```

Calibration cost:

```text
freeze_wall_seconds
gpu_seconds
peak_gpu_memory_mb
peak_host_memory_mb
activation_cache_bytes
format_map_bytes
```

Teacher aligned:

```text
prefix_mode
teacher_target_mode
base_residual_norm
teacher_objective_before
teacher_objective_after
```

Statistics:

```text
num_eval_sequences
num_eval_tokens
mean_nll_per_token
paired_delta_mean
paired_delta_median
paired_ci_low
paired_ci_high
win_fraction
coarse_loss_nll
fine_recovery_fraction
cd_approx_recovery_fraction
```

Robustness:

```text
calibration_shard_id
map_agreement
heldout_objective
cross_shard_ppl_std
cross_shard_nll_std
```

---

# 39. Failure handling

For every failed/aborted run store:

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

Classifications:

```text
code_bug
OOM
dataset/network
model_access
dependency
GPU_became_occupied
numerical_issue
calibration_leakage
equivalence_failure
objective_mismatch
runtime_excessive
unsupported_semantics
external_baseline_unavailable
unknown
```

Never silently erase failures.

---

# 40. Explicitly prohibited Research-3 behavior

Do not:

```text
reopen generic rotation
reopen generic packing
run arbitrary scale heuristic grids
tune on final WikiText/C4
promote a method from PPL point estimates only
call FullLayer-CD a mathematical oracle
call MSE-Oracle16 a PPL oracle
claim native SM120 properties from A6000/Ada fake quant
claim generic residual compensation as novelty
claim adaptive mixed format itself as novelty
add 10 new models before Phase A closes
run 1024 SANA before a frozen method
```

---

# 41. Recommended exact execution order

```text
0. read research_3.md fully; record SHA256
1. create isolated Research-3 branch/worktree
2. audit/hash Research-2 immutable artifacts
3. regenerate Research-2 key tables from raw sequence-level data
4. compute missing paired CIs:
     CD vs RA
     CD vs 4Over6
     RA vs 4Over6
     CD vs Local OutputAware
5. audit P0-B convergence traces and P1-A order selection
6. profile normalized calibration cost:
     OA
     RA
     CD1
     CD2
7. produce per-stripe CD-vs-RA search-gap diagnostics
8. construct c/g/Q on representative stripes
9. unit-test exact BQP objective/flip reconstruction
10. compute coupling metrics
11. run bounded continuous-relaxation headroom audit on representative stripes
    only with correct certified/approximate terminology
12. if profiling says search cost is material:
      implement/test compact quadratic search backend
13. run Qwen correction-sweep study:
      OA-CD1
      CD2 reference
      RA
      RA-Refine1
14. unlock best-improvement only if path-dependence + cost gates justify it
15. if A2 shows concentrated hard stripes:
      diagnose selective refinement
      practical screening must use pre-CD features
16. select <=2 candidates on calibration validation
17. frozen Qwen Wiki+C4 evaluation
18. run Llama controls only for surviving candidates
19. implement/test teacher-aligned quantized-prefix objective
20. run teacher-aligned first on Qwen
21. promote only if held-out + frozen eval improves
22. run 3-shard calibration robustness for <=2 finalists
23. if instability exists:
      test exactly one conservative multi-shard/validation rule
24. calibration-size scaling on Qwen finalist
25. optional bounded cumulative module-attribution curve
26. freeze one Research-3 primary method
27. issue Phase-A METHOD_GO / downgrade decision
28. if METHOD_GO: audit official external baselines
29. pre-register third model family
30. run third model Wiki+C4
31. if positive: run one ~14B scale model
32. run proper downstream tasks
33. run frozen SANA W4A16 proxy
34. if proxy positive: 128-image screen
35. if LLM generalization strong: W4A4 survival test
36. quantify offline calibration/metadata cost
37. regenerate final aggregates
38. issue Research-3 final scientific decision
39. prepare native SM120 handoff only; do not fabricate native results
```

Do not delay the correction-sweep experiment for a large solver study.

The BQP/relaxation work is explanatory and bounded.


# 42. What would count as a strong scientific result?

The strongest result would not be:

```text
we found a better heuristic
```

It would be:

> **Fine-format value is lost when many local representation preferences are compressed
> into coarse hardware format-control regions. This loss is partly caused by coupled
> output-error interaction that independent local selectors cannot optimize. A bounded
> offline coupled assignment recovers the lost quality while retaining exactly the same
> inference representation and metadata granularity.**

Evidence required:

```text
mechanism diagnostics
strong coupled optimizer
cheap-enough practical variant or practical CD itself
calibration robustness
multiple model families
meaningful downstream quality
cross-domain or explicit domain limitation
hardware/deployment accounting
```

---

# 43. What would falsify the paper direction?

Research-3 should stop/downgrade if:

1. CD2 advantage is calibration-shard fragile;
2. its paired advantage over strong baselines is not reproducible;
3. Qwen weakness cannot be stabilized and a third architecture is negative;
4. the method only works because of a project-specific scale/format mismatch unavailable in real hardware;
5. external official baselines dominate under matched semantics;
6. calibration cost becomes unreasonable at modest model scaling with no cheaper approximation;
7. native semantics later invalidate the assumed E0/E2 execution model.

A negative result here is preferable to a weak top-tier claim.

---

# 44. Final Research-3 research framing

Research-3 is no longer primarily a "mixed FP4 format" project.

Its strongest question is:

> **How should many fine representation preferences be compressed into a much coarser
> hardware format-control decision?**

The current evidence says:

```text
local MSE              -> insufficient
local output SSE       -> insufficient
generic packing        -> fails generalization
coupled stripe output  -> strong
one-pass residual      -> useful but model-dependent
```

Research-3 v1.1 adds the mathematical closure:

```text
coarse stripe assignment
=
binary quadratic format-bit optimization

off-diagonal Gram terms
=
cross-K cancellation / reinforcement
```

The purpose of this formulation is to connect:

```text
mechanism
-> measurable interaction diagnostics
-> bounded optimization-headroom audit
-> cost-aware offline method
-> end-to-end NLL
```

not to turn the project into a generic BQP solver paper.

Research-3 must determine whether:

```text
coupled offline format assignment
```

is robust enough to become the actual method.

Do not force the answer in advance.
