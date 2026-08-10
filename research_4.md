# research_4.md
# Research-4 — Coupled Coarse-Format Assignment × Legal Within-Format Adaptive Scaling
## Closing the Last Unresolved Factorial Before Final Generalization Judgment

**SPEC_VERSION:** `research_4_v1.0_2026-08-09`  
**Operational status:** `BOUNDED_REOPEN / SCALE×COUPLING_INTERACTION_UNRESOLVED`  
**Research-3 broad-method status remains:** `FINAL_NO_GO` for the frozen dual-static CD2 instantiation  
**Purpose of Research-4:** test exactly one unresolved interaction: whether legal K16 within-format scale adaptation changes the error geometry enough to improve the already-validated N8K64 coupled format assignment  
**Primary method entering Research-4:** `FullLayer-CD2-N8K64`  
**Current scale policy entering Research-4:** `E2 static6 / E0 static7`  
**Primary development families:** Llama-3.1-8B, Qwen3-8B  
**Prospective held-out family:** Mistral-7B-v0.3  
**Primary evaluation:** paired per-sequence NLL; PPL secondary  
**Native SM120 status:** `WAIT_FOR_SM120`  
**Execution hardware:** RTX A6000 / RTX 6000 Ada reference/fake-quant experiments  
**Top-tier standard:** do not weaken gates to rescue the project

---

# 0. Executive decision

Research-4 is scientifically justified, but only as a **strictly bounded continuation**.

Research-3 did not falsify the coupled-assignment mechanism.

Research-3 established:

```text
fine K16 E2/E0 heterogeneity has value
coarse N8K64 format sharing loses value
local MSE / local output-aware selectors are insufficient
cross-K interactions inside one N8 stripe are measurable
FullLayer-CD2 recovers substantial coarse loss on Llama/Qwen
W4A4 weight-side survival is positive
CD2 adds no inference arithmetic or metadata beyond the same N8K64 format representation
```

Research-3's `FINAL_NO_GO` was driven by the stronger general-method claim, especially:

```text
Mistral C4     -> positive
Mistral Wiki   -> negative point estimate / generalization gate failure
SANA proxy     -> strong
SANA semantic  -> negative
```

The exact combination:

```text
legal adaptive K16 scaling
×
the final winning coupled CD2 selector
```

was never closed.

That missing factorial cell was already anticipated in Research-2 as:

```text
better scale × best selector
```

but the true best selector was only discovered later as FullLayer-CD2.

Therefore Research-4 asks one narrow question:

> **Does legal within-format scale adaptation reshape candidate error magnitude/direction
> and cross-K interaction so that coupled N8K64 format assignment becomes materially
> better and more general?**

Research-4 is not permission to reopen:

```text
rotation
packing
arbitrary scale grids
learned controllers
activation-selector research
model-specific heuristics
Wiki-specific calibration
native kernel work
```

If the bounded scale×coupling test fails, stop this branch with high confidence.

---

# 1. Authority and precedence

This file is the active Research-4 execution specification.

Use:

```text
CURRENT research_4.md
    >
CURRENT coding_agent_prompt_4.md
    >
immutable Research-3 final artifacts
    >
immutable Research-2 artifacts
    >
immutable Research-1 artifacts
    >
older planning/suggestion documents
```

The supplementary suggestion is review material, not an execution specification.

Research-1/2/3 artifacts must remain immutable.

Expected roots:

```text
artifacts/research_1/
artifacts/research_2/
artifacts/research_3/
artifacts/research_4/
```

---

# 2. Research history that the coding agent must understand

## 2.1 Research-1

Research-1 established:

```text
MSE-Oracle16 > NVFP4 quality in 4/4 primary LLM settings
raw MSE-N8K64 loses the fine benefit in 4/4
E0-heavy coarse exposure is largely aggregation/count driven
simple scale adaptation alone was not a robust recovery method
local OutputAware selector helps some settings but is inconsistent
generic rotation is not robust
generic packing was not established
```

Research-1 B0 scale work supports:

> scale adaptation alone is not sufficient.

It does **not** support:

> scale adaptation cannot interact with a later coupled selector.

That distinction is central to Research-4.

---

## 2.2 Research-2

Research-2 closed two major questions.

### Packing

Frozen Qwen foldable packing failed prospective full validation:

```text
Wiki recovery ~ +13.4%
C4 recovery   ~ -64.7%
```

Packing is closed.

### Coupled assignment

`FullLayer-Coarse-CD-N8K64` succeeded.

Approximate coarse-to-fine NLL recovery:

```text
Llama Wiki  ~93%
Llama C4    ~78%
Qwen Wiki   ~61%
Qwen C4     ~37%
```

Research-2 also showed a one-pass ResidualAware approximation was much weaker on Qwen.

This established that fixed N8K64 still contains good format maps and that local/sequential assignment can miss them.

---

## 2.3 Research-3

Research-3 mathematically characterized the N8-stripe problem and closed the primary method.

The final primary method was:

```text
FullLayer-CD2-N8K64
pooled 3×128 C4 calibration
OutputAware initialization
two coupled coordinate sweeps
E2 static6
E0 static7
```

Original four approximate final recoveries:

```text
Llama Wiki  ~106%
Llama C4    ~79%
Qwen Wiki   ~47%
Qwen C4     ~35%
```

The exact Research-3 artifacts are authoritative; regenerate exact values from them.

Research-3 also found strong association between interaction strength and RA→CD search gap:

```text
Qwen:
  Pearson  ~0.81
  Spearman ~0.91

Llama:
  Pearson  ~0.83
  Spearman ~0.84
```

Again, regenerate exact values rather than copying narrative summaries.

Research-3 prospective third-family result:

```text
Mistral C4:
  CD2-Static improved

Mistral Wiki:
  CD2-Static had a negative point estimate
  and failed the preregistered broad-generalization gate
```

Research-3 therefore ended:

```text
FINAL_NO_GO
```

for the **current dual-static broad method**.

This Research-4 continuation does not erase that result.

---

# 3. What is frozen and must not change

## 3.1 Format granularity

Weights:

```text
W[N,K]
```

Format-control region:

```text
N8 × K64
```

One format decision:

```text
E2 or project-E0
```

per N8×K64 region.

Research-4 must not restore fine K16 format dispatch.

---

## 3.2 Scale granularity

Scale granularity remains:

```text
K16
```

Each K16 scale group continues to store one legal scale using the existing representation.

Research-4 may change the chosen legal scale value.

It may not add an independent runtime "which scale rule" selector bit.

---

## 3.3 Project E0 semantics

Project E0 codebook:

```text
0, ±1, ±2, ..., ±7
```

sign-magnitude project-defined fake/reference semantics.

Until native verification:

```text
native_sm120_verified = false
semantic_status = PROJECT_DEFINED_FAKE_QUANT_SEMANTICS
```

Do not call project E0 a verified Blackwell-native format.

---

## 3.4 Fine reference naming

`MSE-Oracle16` means:

```text
per-K16 E2/E0 choice
objective = local weight reconstruction MSE
```

It is not a PPL oracle.

Under a new scale policy \(s\), name the matched fine reference explicitly:

```text
MSE-Oracle16-ScalePolicy[s]
```

or an equivalent unambiguous identifier.

---

# 4. Current static CD2

Current Research-3 CD2 uses two region candidate families:

\[
E_k^{E2,6}
\]

and:

\[
E_k^{E0,7}.
\]

For one N8 output stripe:

\[
J(F)
=
\left\|
\sum_k E_k^{f_k}
\right\|_F^2.
\]

Current CD2 optimizes:

\[
f_k\in\{E2,E0\}.
\]

It does not vary K16 scale state inside a format branch.

---

# 5. Research-4 scale candidate sets

Research-4 initially permits exactly:

```text
E2 branch:
  static target 6
  or adaptive legal {4,6}

E0 branch:
  static target 7
  or adaptive legal {6,7}
```

The first bounded factorial variants are:

```text
CD2-Static
  E2 = static6
  E0 = static7

CD2-E2Scale46
  E2 = local legal {4,6}
  E0 = static7

CD2-E0Scale67
  E2 = static6
  E0 = local legal {6,7}

CD2-DualScale46x67
  E2 = local legal {4,6}
  E0 = local legal {6,7}
```

Diagnostic-only, locked initially:

```text
E0 {5,6,7}
full legal representable E4M3 scale oracle
arbitrary target sets
per-layer target sets
model-specific scale sets
```

---

# 6. Level-1 local scale preselection semantics

This must be deterministic and frozen before final evaluation.

For each K16 scale group and a fixed format branch:

1. enumerate the permitted legal target candidates;
2. generate the legal stored scale using the existing global/block scale semantics;
3. quantize/dequantize the K16 values;
4. compute local weight reconstruction SSE;
5. choose the legal candidate with minimum local SSE;
6. deterministic tie-breaking:
   - E2: prefer static target6;
   - E0: prefer static target7.

Thus:

```text
E2 {4,6}
-> choose local weight-SSE best legal scale candidate

E0 {6,7}
-> choose local weight-SSE best legal scale candidate
```

Do not use end-to-end Wiki/C4 PPL to choose K16 scale state.

For the E2 branch, forced-E2 behavior must reproduce canonical Four Over Six semantics
under matched global-scale / quantization settings before Research-4 proceeds.

For the E0 branch, target6/target7 semantics are project-defined and must be exhaustively
tested.

---

# 7. Why scale can interact with coupled format assignment

With adaptive scales:

\[
E_k^{E2,s}, \quad s\in\{4,6\}
\]

and:

\[
E_k^{E0,s}, \quad s\in\{6,7\}.
\]

The stripe objective expands as:

\[
J
=
\sum_k\|E_k\|_F^2
+
2\sum_{i<j}\langle E_i,E_j\rangle_F.
\]

A new scale can change:

```text
error norm
error direction
pairwise alignment
cancellation
reinforcement
```

Therefore a locally non-best scale under a coupled residual can still be globally useful.

For example:

\[
\|E^4\|_F > \|E^6\|_F
\]

can coexist with:

\[
\|R+E^4\|_F < \|R+E^6\|_F.
\]

This is the mechanism that makes Research-4 scientifically distinct from Research-1 B0.

---

# 8. Research-4 hypotheses

## H0 — no interaction

Adaptive scale merely gives a stronger local quantizer.

Expected pattern:

```text
fine improves
raw coarse improves
CD2 improves similarly
matched recovery fraction does not increase
CD2-vs-local advantage does not increase
cross-interaction geometry does not become more favorable
```

Interpretation:

```text
STRONGER_QUANTIZER_ONLY
```

This does not rescue the coupled-format paper story.

---

## H1 — scale × coupled-format complementarity

Adaptive scale changes error geometry in a way that CD2 can exploit.

Expected evidence:

```text
matched CD recovery increases
and/or
CD2-vs-local selector gain increases
and
interaction diagnostics move consistently with that gain
```

This is the scientifically interesting outcome.

---

## H2 — restricted candidate-basis caused part of Mistral reversal

Static6/static7 may provide an insufficient candidate basis.

Adaptive legal scale may improve Mistral Wiki while preserving Mistral C4.

This must be tested prospectively after all method development is frozen.

---

## H3 — even adaptive legal scales cannot rescue cross-family/cross-corpus generalization

If the prospective Mistral sign reversal persists after a frozen Research-4 method:

```text
close broad method branch with high confidence
```

Do not continue target-set tuning.

---

# 9. Important novelty boundary

Research-4 must not claim:

```text
adaptive scale selection is novel
4/6 is novel
scale optimization is novel
joint format/scale quantization is novel in the abstract
adaptive mixed FP4/INT4 is novel
```

Current neighboring work already makes these broad claims unsafe.

Planning guardrails include:

```text
Four Over Six
  adaptive NVFP4 scale choice

Adaptive Block-Scaled Data Types / IF4
  adaptive FP4/INT4 representation per small group

MixFP4
  adaptive micro-format under NVFP4-like scaling

SOAR
  explicit NVFP4 scale optimization

FOCUS
  coupled-relaxation + dual-granularity FP4 scaling

AdaMX
  heterogeneity-aware microscaling and hardware support
```

The Research-4 novelty, if successful, must remain:

> **Under a fixed coarse N8K64 format-control constraint, legal K16 scale states reshape
> the cross-K error interactions that determine the coupled format map. Jointly exploiting
> those existing scale degrees of freedom recovers coarse decision-compression loss without
> changing inference-time format granularity or scale storage.**

The paper center remains:

```text
coarse format-decision compression
+
coupled cross-K assignment
+
legal scale-state interaction
```

not:

```text
adaptive scaling
```

---

# 10. Research-4 experimental philosophy

Research-4 is divided into:

```text
Phase 0: handoff + semantic correctness
Phase A: calibration-only scale factorial and mechanism
Phase B: development-family frozen evaluation
Phase C: optional bounded joint scale/format refinement
Phase D: ONE prospective Mistral generalization test
Phase E: broader evidence only if Mistral passes
```

Critical rule:

> **Mistral must not be inspected until every Research-4 algorithmic choice is frozen.**

This is stricter than the supplementary suggestion and is required to preserve a
prospective generalization claim.

---

# 11. Phase 0 / Gate S0 — handoff and semantic correctness

## 11.1 Audit immutable Research-3

Before code changes:

```text
locate Research-3 final decision
locate frozen CD2 map-generation code
locate pooled 3×128 calibration manifests
locate Mistral evaluation IDs
locate Research-3 scale/global-scale implementation
record repo SHA and patch state
hash immutable final artifacts
```

Regenerate the key Research-3 table from raw artifacts.

Do not copy narrative numbers as source of truth.

---

## 11.2 Scale-semantic audit

Create a dedicated note covering:

```text
global scale
K16 block scale
E4M3 stored scale
target value semantics
scale clipping/rounding
zero behavior
tail handling
underflow/overflow
format-specific max code
```

No Research-4 job may launch until this is explicit.

---

## 11.3 Forced-E2 Four Over Six reduction

When:

```text
format = fixed E2
scale policy = {4,6}
```

the implementation must reproduce the canonical Four Over Six quantizer under the same
weight/global-scale configuration.

Test:

```text
selected target
stored scale bits/value
quantized FP4 code
dequantized value
per-block SSE
```

on:

```text
exhaustive synthetic boundary vectors
random K16 vectors
real sampled weight blocks
```

Any semantic mismatch must be explained before proceeding.

---

## 11.4 E0 {6,7} semantic tests

For project E0:

```text
target6
target7
```

test:

```text
legal stored scales
tie cases
saturation boundaries
zero
positive/negative symmetry
all codebook values
random vectors
real blocks
```

---

## 11.5 Metadata-equivalence test

Verify:

```text
one N8K64 format bit remains
one existing legal K16 scale field remains
no extra 4-vs-6 bit
no extra 6-vs-7 bit
no runtime selector table
no online search
```

The chosen stored scale value itself must fully encode the final scale state.

If this is false:

```text
STOP Research-4
```

because the intended hardware/runtime premise changes.

---

# 12. Phase A — bounded scale factorial on development families

Development models only:

```text
Llama-3.1-8B
Qwen3-8B
```

Datasets used for final development evaluation:

```text
WikiText2
C4
```

Do not use Mistral in Phase A.

---

# 13. A0 — calibration split and policy selection discipline

Reuse the exact Research-3 calibration pool when possible:

```text
3 × 128 C4 sequences
```

If the existing artifacts already define train/validation roles, preserve them.

Otherwise use three-fold cross-shard validation:

```text
fold 1:
  train shards 2+3
  validate shard 1

fold 2:
  train shards 1+3
  validate shard 2

fold 3:
  train shards 1+2
  validate shard 3
```

Do this separately for Llama and Qwen.

No WikiText/C4 final evaluation sequence may appear in calibration.

---

# 14. A1 — four scale policies

Evaluate exactly:

```text
A. Static
B. E2Scale46
C. E0Scale67
D. DualScale46x67
```

For each policy and each development model/fold:

1. construct locally selected legal K16 scale candidates;
2. construct E2/E0 candidate error tensors;
3. create the OutputAware initializer under the same scale policy;
4. run exactly two coupled coordinate sweeps;
5. evaluate coupled layer/stripe objective on the held-out calibration shard;
6. store the format map and scale map;
7. never look at final Wiki/C4 PPL during policy selection.

---

# 15. A2 — calibration-only scale-policy selection

For each policy \(s\), define held-out validation objective:

\[
J_{m,f,s}^{val}.
\]

Normalize to static:

\[
RelGain_{m,f,s}
=
\frac{
J_{m,f,static}^{val}
-
J_{m,f,s}^{val}
}{
J_{m,f,static}^{val}+\epsilon
}.
\]

Aggregate over:

```text
model = Llama/Qwen
fold = 1/2/3
```

A scale policy is eligible only if:

```text
mean held-out coupled objective is not worse than Static for Llama
AND
mean held-out coupled objective is not worse than Static for Qwen
```

Choose the eligible policy with the largest cross-model mean normalized held-out gain.

Tie-breaking:

```text
prefer simpler one-branch adaptive policy
over
DualScale
```

if the validation difference is within numerical/statistical equivalence.

If no adaptive policy is eligible:

```text
S1_PRE = FAIL
Research-4 may still run a minimal static reproduction,
but no adaptive method is promoted.
```

Freeze:

```text
selected_scale_policy.json
selection_evidence.json
```

before development Wiki/C4 evaluation.

---

# 16. A3 — matching controls are mandatory

For every scale policy that reaches development evaluation, construct matched:

```text
Fine16 + scale policy
Raw N8K64 + scale policy
Local/OutputAware N8K64 + scale policy
CD2 N8K64 + scale policy
```

The main scientific comparison must never be only:

```text
CD2-adaptive vs CD2-static
```

because that confounds stronger local quantization with coupled-selector complementarity.

---

# 17. Matched fine reference under scale policy

For each K16 block:

1. construct E2 branch using that policy's legal E2 scale rule;
2. construct E0 branch using that policy's legal E0 scale rule;
3. choose E2/E0 by local weight MSE.

This gives:

```text
MSE-Oracle16-[policy]
```

It remains a local weight-MSE fine reference, not a PPL oracle.

---

# 18. Matched raw N8K64 under scale policy

For each N8K64 region:

1. locally choose K16 scales inside the E2 branch;
2. locally choose K16 scales inside the E0 branch;
3. aggregate weight reconstruction MSE across the region for each format;
4. choose the lower-MSE format.

This produces:

```text
MSE-N8K64-[policy]
```

This is the matched coarse baseline.

---

# 19. Matched OutputAware under scale policy

Use the same Research-3 OutputAware calibration semantics but replace the candidate tensors
with those generated under the current scale policy.

Do not change:

```text
input semantics
objective definition
row budget
layer handling
```

unless a bug is found and documented.

---

# 20. Matched CD2 under scale policy

For each policy:

```text
initializer = matched OutputAware map
sweeps = exactly 2
stripe objective = unchanged coupled output SSE
tie-breaking = Research-3 deterministic behavior
```

No new format order/search variants.

---

# 21. Core matched quantities

For scale policy \(s\):

\[
Gap_s
=
NLL_{raw,s}
-
NLL_{fine,s}.
\]

When \(Gap_s>0\):

\[
Recovery_s
=
\frac{
NLL_{raw,s}
-
NLL_{CD2,s}
}{
Gap_s
}.
\]

Also define CD selector gain:

\[
CDGain_s
=
NLL_{raw,s}
-
NLL_{CD2,s}.
\]

Define scale×coupling interaction gain:

\[
InteractionGain_s
=
CDGain_s
-
CDGain_{static}.
\]

Positive `InteractionGain` means the coupled selector gains more over its matched raw
baseline after scale adaptation.

Also report:

\[
\Delta Recovery_s
=
Recovery_s-Recovery_{static}.
\]

Do not compute recovery when the matched coarse gap is non-positive.

---

# 22. Calibration-objective interaction metrics

For each N8 stripe record:

```text
J_static_before_cd
J_static_after_cd
J_adaptive_before_cd
J_adaptive_after_cd

L_independent_static
L_independent_adaptive

C_cross_static
C_cross_adaptive

format_map_hamming_static_vs_adaptive
E0_ratio_static
E0_ratio_adaptive

scale46_selection_rate
scale67_selection_rate
```

where:

\[
L=\sum_k\|E_k\|_F^2
\]

and:

\[
C=J-L.
\]

More negative \(C\) means more favorable net cancellation for the chosen assignment.

This is an objective decomposition, not a direct causal decomposition of PPL.

---

# 23. BQP interaction diagnostics under scale adaptation

For the post-local-scale E2/E0 candidate pair, reconstruct the Research-3 BQP:

\[
J(z)=c+2g^Tz+z^TQz.
\]

For Static and adaptive policy record:

```text
pairwise_coupling_abs_sum
pairwise_coupling_offdiag_ratio
negative_coupling_fraction
positive_coupling_fraction
largest_abs_pairwise_coupling
interaction_matrix_spectral_norm
```

Also compute change:

```text
delta_pairwise_offdiag_ratio
delta_negative_coupling_fraction
delta_cross_interaction
delta_cd_objective_gain
```

Primary mechanism question:

> Does the adaptive scale policy change pairwise geometry in a way associated with larger
> coupled CD gain and better held-out model NLL?

Do not require every stripe to have more negative coupling.

---

# 24. Phase B — development-family frozen evaluation

After scale policy is selected and frozen using calibration only:

Run on:

```text
Llama WikiText2
Llama C4
Qwen WikiText2
Qwen C4
```

Use exact same evaluation IDs as Research-3.

Required methods:

```text
HighPrecision
NVFP4
Canonical-4Over6
Fine16-Static
Raw-N8K64-Static
OutputAware-N8K64-Static
CD2-Static

Fine16-[SelectedScale]
Raw-N8K64-[SelectedScale]
OutputAware-N8K64-[SelectedScale]
CD2-[SelectedScale]
```

Do not choose a new policy after seeing these results.

---

# 25. Statistics

Primary statistic:

```text
per-sequence NLL
```

For every paired comparison:

```text
mean paired ΔNLL
median paired ΔNLL
95% paired bootstrap CI
win fraction
PPL
```

Use fixed bootstrap seed and fixed sequence IDs.

Required direct comparisons:

```text
CD2-selected vs CD2-static
Raw-selected vs Raw-static
Fine-selected vs Fine-static
CD2-selected vs Raw-selected
CD2-selected vs 4Over6
```

Also report:

```text
Recovery_selected
Recovery_static
ΔRecovery
InteractionGain
```

---

# 26. Gate S1 — scale×coupling complementarity

The selected adaptive policy passes S1 only if all of the following hold:

### S1.1 No original-four deterioration

```text
CD2-selected point-estimate NLL <= CD2-static in 4/4
```

and paired confidence intervals do not support a material regression.

### S1.2 Meaningful improvement

At least two of four settings must show a non-trivial improvement.

Use both:

```text
absolute paired ΔNLL
and
matched recovery change
```

Do not promote from a tiny numerical change.

An internal design target is:

```text
ΔRecovery >= +5 percentage points in >=2/4
```

or an equivalently strong increase in matched `CDGain`.

This is a project gate, not a universal law.

### S1.3 Coupled interaction evidence

Require at least one:

```text
A. average matched ΔRecovery > 0 with positive direction in most/all settings

or

B. average InteractionGain > 0 and objective-level CD-vs-local gain increases

or

C. pairwise/cross-interaction diagnostics show a consistent mechanism associated with CD gain
```

### S1.4 Not merely a stronger quantizer

If:

```text
fine improves
raw improves similarly
CD2 improves similarly
Recovery / CDGain interaction is essentially unchanged
```

classify:

```text
STRONGER_QUANTIZER_ONLY
```

and do not reopen the coupled-scale paper claim.

---

# 27. Research-4 development status after S1

Possible statuses:

```text
S1_FAIL
STRONGER_QUANTIZER_ONLY
SCALE_COUPLING_COMPLEMENTARY
```

Only `SCALE_COUPLING_COMPLEMENTARY` unlocks Phase C/D.

---

# 28. Phase C — optional Level-2 joint format/scale refinement
# Locked unless S1 shows real complementarity and calibration diagnostics show scale headroom

Do not run Level 2 merely because Level 1 improved.

Unlock only if both:

```text
S1 = SCALE_COUPLING_COMPLEMENTARY
AND
local_scale_choice_vs_coupled_residual disagreement is non-trivial
```

Evidence for scale headroom should come from calibration-only diagnostics.

---

# 29. Level-2 multi-state problem

For N8×K64 format region \(k\):

\[
f_k\in\{E2,E0\}.
\]

For each K16 scale group \(b\) inside region \(k\):

if:

\[
f_k=E2,
\]

then:

\[
s_{k,b}\in\{4,6\}.
\]

If:

\[
f_k=E0,
\]

then:

\[
s_{k,b}\in\{6,7\}.
\]

The stripe objective remains:

\[
J(F,S)
=
\left\|
\sum_{k,b}E_{k,b}^{f_k,s_{k,b}}
\right\|_F^2.
\]

No global combinatorial solver is allowed.

---

# 30. Bounded alternating coordinate algorithm

Pre-register one algorithm:

```text
initialization:
  Level-1 selected local scales
  +
  matched OutputAware format map

step 1:
  format sweep

step 2:
  scale sweep

step 3:
  format sweep

step 4:
  optional final scale sweep
  only if calibration validation shows step 3 has not converged
```

No random ordering bank.

Tie-breaking must be deterministic.

A scale flip for K16 group \(b\) changes the stripe residual by:

\[
\Delta E_{k,b}^{f,s\rightarrow s'}
=
X_b
\left[
Q_{f,s'}(W_b)
-
Q_{f,s}(W_b)
\right]^T.
\]

Accept when:

\[
\|R+\Delta E\|_F^2<\|R\|_F^2.
\]

---

# 31. Level-2 scale-headroom diagnostics

Before promoting joint refinement, measure:

```text
local_scale_choice_vs_coupled_scale_choice_disagreement
coupled_scale_flip_count
coupled_scale_flip_rate
heldout_J_gain_from_scale_flips
heldout_J_gain_from_format_flips
format_flip_count_after_scale_sweep
```

If disagreement/held-out gain is negligible:

```text
do not run large Level-2 evaluation
keep Level-1 method
```

---

# 32. Selection between Level 1 and Level 2

Mistral must still be unseen.

Select the final Research-4 method using only:

```text
Llama/Qwen calibration train
Llama/Qwen calibration validation
Llama/Qwen development evidence
```

Primary rule:

```text
prefer Level 1 unless Level 2 gives clear held-out calibration improvement
without materially increasing offline calibration cost
```

No extra format/scale states may be introduced.

Freeze exactly one final method:

```text
research4_frozen_method.json
research4_frozen_scale_policy.json
research4_frozen_algorithm.json
```

After this point, no method tuning is allowed.

---

# 33. Why Mistral must be evaluated only once

Mistral is the prospective test of the exact failure that stopped Research-3.

If Mistral is used to:

```text
choose E2/E0 scale set
choose Level 1 vs Level 2
choose sweep count
choose calibration size
choose scale tie-breaking
choose objective
```

then the prospective claim is invalid.

Therefore:

> **All Research-4 method development must be complete before opening Mistral results.**

If a new method is invented after seeing Mistral, Mistral becomes development evidence and
a new unseen family would be required for prospective validation.

Research-4 does not permit that rescue loop.

---

# 34. Phase D / Gate S2 — prospective Mistral repair

Only after final Research-4 method freeze.

Model:

```text
Mistral-7B-v0.3
```

Calibration:

```text
same frozen policy
same calibration protocol
C4 calibration only
no Wiki-specific calibration
no Mistral-specific scale rule
```

Generate the Mistral format/scale map using the frozen algorithm.

Then evaluate exactly:

```text
Mistral WikiText2
Mistral C4
```

using the same evaluation IDs/protocol as Research-3.

---

# 35. Required Mistral controls

Run:

```text
HighPrecision
NVFP4
Canonical-4Over6

Fine16-Static
Raw-N8K64-Static
CD2-Static

Fine16-[Research4 Frozen Scale]
Raw-N8K64-[Research4 Frozen Scale]
Research4 Frozen Method
```

If Level 2 is selected, include the frozen Level-1 adaptive method as an ablation, but do not
select between them after seeing Mistral.

---

# 36. Gate S2 — prospective generalization

S2 passes only if:

### WikiText

The Research-4 method removes the Research-3 failure:

```text
point-estimate NLL <= matched raw N8K64
paired CI does not support meaningful regression
competitive with Canonical-4Over6
```

### C4

The existing positive behavior is preserved:

```text
positive/non-regressive vs matched raw N8K64
competitive with Canonical-4Over6
```

### Cross-corpus

There is no C4-positive / Wiki-negative sign reversal.

Strongest success:

```text
Wiki becomes positive
AND
C4 remains positive
```

If S2 fails:

```text
RESEARCH4_FINAL_STOP_BROAD_METHOD
```

No:

```text
target5
model-specific scale sets
Wiki calibration
more scale grids
Mistral-specific corrections
```

---

# 37. Important interpretation if S2 fails

A failed S2 means:

> Legal {4,6}/{6,7} scale adaptation does not repair the main prospective generalization
> failure under the current coupled-format formulation.

Then the Research-3 `FINAL_NO_GO` becomes substantially more final.

The separate question:

```text
Can we predict when a model×distribution pair should abstain/fallback?
```

may still be scientifically interesting.

But that is a **new project** requiring new prospective families and a separate preregistered
safe-applicability design.

It is not automatically part of Research-4 and must not be started by the coding agent.

---

# 38. Phase E / Gate S4 — broader evidence
# Only if S2 passes

If Mistral passes prospectively, the broad-method claim is reopened.

Then pre-register broader evidence before running it.

---

# 39. E0 — one additional family OR one scale case

Choose before results.

Preferred priorities:

```text
1. one new non-Llama-like model family, 7B–10B
2. then one ~12–14B model-size case if resources permit
```

Potential families depend on harness support and model access, e.g.:

```text
Gemma family
another non-Llama/non-Qwen architecture
```

Do not test several families and report only the favorable one.

---

# 40. E1 — downstream LLM suite

Use the same frozen Research-4 method.

Fixed task suite if supported:

```text
HellaSwag
PIQA
ARC-Challenge
WinoGrande
BoolQ
MMLU
```

Do not alter tasks after seeing results.

Compare:

```text
HighPrecision
NVFP4
Canonical-4Over6
matched raw N8K64
CD2-Static
Research-4 frozen method
```

---

# 41. E2 — W4A4 survival

Research-3 showed weight-side CD2 survived activation quantization noise.

Repeat only with the frozen Research-4 scale policy.

Do not invent a new activation selector.

Use one fixed canonical activation policy.

Question:

> Does scale-coupled weight assignment retain its advantage under W4A4?

---

# 42. E3 — SANA last

SANA is not a method-development environment.

Only after LLM generalization passes:

```text
frozen proxy
-> if positive, 128-image screen
-> 1024 only if non-dominated
```

Never retune:

```text
scale sets
search order
sweep count
calibration objective
```

from image metrics.

A negative SANA result limits cross-domain claims but does not automatically invalidate an
LLM-only result.

---

# 43. Runtime/hardware premise

Research-4 must preserve:

```text
one format decision per N8K64 region
one existing scale per K16 group
no new metadata field
no extra selector bit
no runtime search
no extra GEMM
no runtime permutation
no online transform
```

The scale value changes offline; storage class does not.

This premise must be audited in Phase 0.

---

# 44. Offline calibration cost

For Static, selected Level 1, and any selected Level 2 variant record:

```text
total freeze wall time
GPU time
peak GPU memory
peak host memory
activation/error construction time
local scale-search time
format CD time
scale CD time
I/O
serialization
candidate error cache bytes
format map bytes
scale storage bytes
```

Report incremental offline cost:

```text
adaptive scale vs Static CD2
joint format-scale vs Level 1
```

No inference overhead claim may be inferred from offline cost.

---

# 45. Native SM120 policy

Native remains:

```text
WAIT_FOR_SM120
```

Reference/fake-quant results do not establish:

```text
native E0 decode
Tensor Core support
latency
throughput
power
actual metadata path
```

When SM120 becomes available, native verification is a separate phase.

Research-4 must not fabricate it.

---

# 46. GPU policy

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

For every visible PID:

```bash
ps -o user=,pid=,cmd= -p <PID>
```

Rules:

```text
occupied by another user => unavailable
unknown owner + occupied => unavailable
low utilization != free
never share a physical GPU with another user
never kill another user's process
never broad pkill
use project reservation lock
immediately re-check before exec
log every admission/rejection
```

---

# 47. Git/repository policy

Before modification record:

```text
repo URL
branch
HEAD SHA
submodule SHAs
dirty status
local diff
Research-3 final result commit/hash
```

Create an isolated Research-4 branch/worktree.

Do not:

```text
rewrite Research-3 artifacts
push unless user explicitly requests
modify another user's branch destructively
```

Use local commits at meaningful milestones.

---

# 48. Mandatory tests before GPU factorial

## 48.1 E2 scale tests

```text
static6 exact reproduction
target4 legal scale generation
target6 legal scale generation
4/6 local selection
canonical FourOverSix reduction
tie -> target6
tail K16 behavior
```

## 48.2 E0 scale tests

```text
static7 exact reproduction
target6 legal scale generation
target7 legal scale generation
6/7 local selection
tie -> target7
sign symmetry
zero
saturation
tails
```

## 48.3 Scale granularity invariant

Changing policy must not change:

```text
K16 scale grouping
N8K64 format grouping
tensor shape
weight ordering
```

## 48.4 Runtime metadata invariant

Serialized representation must have the same field counts/widths as Static CD2.

## 48.5 Matched-reference tests

On tiny synthetic tensors:

```text
Fine16-[policy] exact local decision
Raw-N8K64-[policy] exact aggregated MSE decision
OutputAware-[policy] exact candidate use
CD2-[policy] monotone stripe objective
```

---

# 49. Level-2 tests if unlocked

```text
scale flip residual delta == full recomputation
format flip residual delta == full recomputation
accepted scale flip never increases objective
accepted format flip never increases objective
alternating pass objective monotone
save/reload format map bit-identical
save/reload scale map bit-identical
```

---

# 50. Evaluation leakage rules

Never use final evaluation to choose:

```text
scale policy
target set
tie-breaking
Level 1 vs Level 2
sweep count
calibration size
scale objective
model-specific behavior
```

Store:

```text
calibration IDs
calibration hashes
fold roles
final eval IDs
eval hashes
policy-selection timestamp
method-freeze timestamp
Mistral-unseal timestamp
```

Mistral results must not be accessed before method freeze.

If accidental Mistral leakage occurs:

```text
mark Mistral as contaminated development evidence
do not claim prospective S2
require a new held-out family
```

---

# 51. Failure handling

Every failed/aborted run must record:

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
dependency
model_access
GPU_became_occupied
semantic_mismatch
canonical_4over6_mismatch
scale_illegal
metadata_mismatch
calibration_leakage
evaluation_leakage
Mistral_leakage
numerical_issue
runtime_excessive
unknown
```

Never silently drop failures.

---

# 52. Research-4 artifact tree

All new artifacts:

```text
artifacts/research_4/
```

Required:

```text
00_environment/
  spec_acknowledgement.md
  spec_manifest.json
  environment.txt
  repo_manifest.json
  research3_handoff_audit.md
  research3_source_hashes.json
  gpu_usage_log.jsonl
  patch_manifest.json
  literature_guardrail_notes.md

01_semantics_tests/
  scale_semantics.md
  static_cd2_reproduction.json
  four_over_six_reduction.json
  e2_scale46_semantics.json
  e0_scale67_semantics.json
  scale_granularity_invariant.json
  runtime_metadata_equivalence.json
  matched_reference_unit_tests.json

02_phase_a_factorial/
  calibration_manifest.json
  calibration_folds.json
  policy_static/
  policy_e2scale46/
  policy_e0scale67/
  policy_dualscale46x67/
  heldout_objective.csv
  policy_selection_summary.csv
  selected_scale_policy.json
  scale_choice_stats.csv
  interaction_stats.csv
  stripe_metrics.csv
  format_map_comparison.csv

03_phase_b_development_eval/
  llama_wiki.csv
  llama_c4.csv
  qwen_wiki.csv
  qwen_c4.csv
  per_sequence_nll.csv
  paired_bootstrap.json
  matched_recovery.csv
  scale_selector_interaction.csv
  s1_decision.md
  s1_decision.json

04_phase_c_joint_refinement/
  unlock_decision.md
  # if unlocked:
  local_vs_coupled_scale_disagreement.csv
  joint_trace.jsonl
  heldout_validation.csv
  level1_vs_level2.csv
  frozen_method.json
  frozen_scale_policy.json
  frozen_algorithm.json

05_phase_d_mistral_prospective/
  mistral_unseal_log.json
  calibration_manifest.json
  frozen_policy_copy.json
  wiki.csv
  c4.csv
  per_sequence_nll.csv
  paired_bootstrap.json
  matched_recovery.csv
  s2_decision.md
  s2_decision.json

06_phase_e_broader/
  preregistration.md
  third_family/
  scale_model/
  downstream/
  w4a4/
  sana/

07_deployment/
  offline_cost.csv
  metadata_accounting.md
  native_sm120_handoff.md

08_final/
  experiment_manifest.jsonl
  failed_runs.jsonl
  master_results.csv
  method_comparison.csv
  matched_recovery.csv
  interaction_summary.csv
  final_decision.json
  final_decision_report.md
  results_summary.md
  limitations.md
  paper_claims_boundary.md
  reproduction_commands.sh
  future_safe_applicability_handoff.md
```

Terminal output is not the source of truth.

---

# 53. Required machine-readable fields

Common:

```text
experiment_id
phase
model
model_revision
dataset
corpus
method
scale_policy
format_policy
repo_sha
config_hash
calibration_hash
calibration_fold
eval_hash
seed
gpu_physical_id
gpu_uuid
status
start_time
end_time
```

Scale:

```text
format
k16_group_id
scale_candidate_set
selected_scale_target
stored_scale_value
stored_scale_bits_if_available
local_sse_target4
local_sse_target6
local_sse_target7
scale46_selected
scale67_selected
```

Format:

```text
n8k64_region_id
initializer_format
final_format
format_flip_count
format_map_hash
e0_ratio
```

Interaction:

```text
stripe_id
J_before_cd
J_after_cd
L_independent
C_cross
pairwise_coupling_abs_sum
pairwise_coupling_offdiag_ratio
negative_coupling_fraction
positive_coupling_fraction
largest_abs_pairwise_coupling
interaction_matrix_spectral_norm
```

Matched evaluation:

```text
sequence_id
nll
paired_delta_nll
fine_gap_nll
recovery_fraction
cd_gain
interaction_gain
bootstrap_ci_low
bootstrap_ci_high
win_fraction
```

Level 2:

```text
scale_flip_count
scale_flip_rate
local_vs_coupled_scale_disagreement
heldout_gain_from_scale_flips
alternating_pass_index
objective_before
objective_after
```

Leakage/prospective:

```text
method_freeze_timestamp
mistral_unseal_timestamp
mistral_was_accessed_pre_freeze
prospective_status
```

---

# 54. Final main tables if successful

Main method table:

| Model / Corpus | NVFP4 | 4Over6 | Fine | Raw N8K64 | CD2-Static | Research-4 |
|---|---:|---:|---:|---:|---:|---:|
| Llama Wiki | | | | | | |
| Llama C4 | | | | | | |
| Qwen Wiki | | | | | | |
| Qwen C4 | | | | | | |
| Mistral Wiki | | | | | | |
| Mistral C4 | | | | | | |

Factorial ablation:

| Model / Corpus | Static | E2 {4,6} | E0 {6,7} | Dual {4,6}×{6,7} |
|---|---:|---:|---:|---:|
| Llama Wiki | | | | |
| Llama C4 | | | | |
| Qwen Wiki | | | | |
| Qwen C4 | | | | |

Mechanism table:

| Setting | ΔLocal term | ΔCross term | ΔCDGain | ΔRecovery |
|---|---:|---:|---:|---:|
| ... | | | | |

Do not put diagnostic scale oracles in the main table.

---

# 55. Research-4 decision tree

```text
START
  |
  |-- S0 semantics correct?
  |      |
  |      +-- NO -> STOP
  |      |
  |      +-- YES
  |
  |-- Calibration-only A/B/C/D scale factorial
  |      |
  |      +-- no adaptive policy improves held-out objective
  |             -> S1_PRE_FAIL -> STOP adaptive branch
  |
  |-- Freeze one Level-1 policy
  |
  |-- Development Llama/Qwen evaluation + matched controls
  |      |
  |      +-- generic quantizer gain only
  |      |      -> STRONGER_QUANTIZER_ONLY -> close coupled-scale claim
  |      |
  |      +-- no meaningful gain
  |      |      -> S1_FAIL -> STOP
  |      |
  |      +-- true scale×coupling complementarity
  |             -> S1 PASS
  |
  |-- Is local-vs-coupled scale disagreement/headroom meaningful?
  |      |
  |      +-- NO -> keep Level 1
  |      |
  |      +-- YES -> bounded Level-2 alternating refinement
  |                    |
  |                    +-- validation gain weak -> keep Level 1
  |                    |
  |                    +-- validation gain strong -> choose Level 2
  |
  |-- FREEZE EXACTLY ONE RESEARCH-4 METHOD
  |
  |-- UNSEAL MISTRAL ONCE
  |
  |-- Mistral Wiki + C4 prospective S2
         |
         +-- FAIL
         |     -> RESEARCH4_FINAL_STOP_BROAD_METHOD
         |     -> no Mistral rescue tuning
         |
         +-- PASS
               -> broad method claim reopened
               -> pre-register Phase E
               -> new family / scale / downstream / W4A4 / SANA
```

---

# 56. Final decision vocabulary

Use exactly one:

```text
RESEARCH4_SEMANTIC_FAIL
SCALE_BRANCH_NO_SIGNAL
STRONGER_QUANTIZER_ONLY
SCALE_COUPLING_COMPLEMENTARY_BUT_NOT_GENERAL
RESEARCH4_FINAL_STOP_BROAD_METHOD
CONDITIONAL_GO_SCALE_COUPLED
ALGORITHMIC_STRONG_GO
CROSS_DOMAIN_GO
WAIT_FOR_SM120
```

The final report may pair an algorithmic status with `WAIT_FOR_SM120`.

---

# 57. What qualifies as top-tier progress?

A strong Research-4 result is not:

```text
4/6 gives lower PPL
```

It is:

1. the scale policy is selected without evaluation tuning;
2. matched Fine/Raw/CD controls show a true interaction rather than generic quantizer gain;
3. interaction diagnostics show that legal scale choices reshape cross-K error geometry;
4. the frozen method improves or preserves all original four settings;
5. Mistral is repaired prospectively without Mistral-specific tuning;
6. broader evidence survives on at least one additional family/scale setting;
7. downstream/W4A4 remain positive;
8. inference representation remains unchanged;
9. novelty is framed around coarse decision compression × error-geometry coupling, not adaptive scaling.

If Mistral is not prospectively repaired, do not call Research-4 a universal method success.

---

# 58. What would falsify this continuation?

Stop if any:

```text
FourOverSix semantic reduction fails
metadata equivalence fails
no adaptive policy improves calibration-held-out coupled objective
adaptive scale only improves fine/raw/CD equally
CD2 matched recovery does not improve
development 4/4 becomes less stable
Mistral cross-corpus reversal persists
method requires Mistral-specific policy
method requires evaluation-based scale selection
method requires extra runtime metadata
```

A clean negative result is preferable to a weak top-tier claim.

---

# 59. Recommended exact execution order

```text
0. read/hash Research-4 spec
1. create isolated worktree
2. audit/hash Research-3 final artifacts
3. reproduce Research-3 CD2-Static exactly
4. audit global/K16 scale semantics
5. verify forced-E2 canonical FourOverSix reduction
6. verify E0 target6/7 semantics
7. verify metadata/storage equivalence
8. build matched Fine/Raw/OutputAware/CD2 policy abstraction
9. create/reuse 3×128 calibration folds
10. run calibration-only Static/E2Scale46/E0Scale67/DualScale factorial
11. compute held-out objective + scale-choice + interaction diagnostics
12. select/freeze one Level-1 scale policy without final eval
13. run Llama/Qwen Wiki+C4 matched evaluation
14. compute paired NLL, matched recovery, InteractionGain
15. issue S1 decision
16. if S1 not complementary -> STOP branch
17. if S1 complementary:
      compute local-vs-coupled scale disagreement/headroom
18. if headroom meaningful:
      run bounded joint format/scale refinement on Llama/Qwen only
19. choose Level1 vs Level2 from calibration/development evidence
20. freeze exactly one Research-4 method
21. write freeze hashes/timestamps
22. unseal Mistral
23. generate Mistral map using frozen C4 calibration protocol
24. run Mistral Wiki+C4 exactly once for S2
25. issue S2 decision
26. if S2 fails:
      FINAL STOP broad method; no Mistral rescue tuning
27. if S2 passes:
      pre-register one new family / one scale case
28. run broader LLM evidence
29. run downstream
30. run W4A4 survival
31. run SANA proxy last
32. if SANA proxy positive: 128 images
33. quantify offline calibration + metadata cost
34. regenerate all final aggregates
35. issue strict final decision
36. prepare native SM120 handoff only
```

---

# 60. Final framing

Research-4 exists because one logically important cell is still empty:

| | Static K16 scale | Legal adaptive K16 scale |
|---|---:|---:|
| Local/MSE selector | tested | tested / weak |
| Coupled CD2 selector | tested / strong | **unresolved** |

Research-4 closes that cell.

The central scientific test is:

> **Does legal K16 scale adaptation merely reduce local quantization error, or does it
> reshape cross-K error directions so that the coupled N8K64 format optimizer can recover
> more coarse decision-compression loss and generalize prospectively?**

If the answer is no, stop.

If the answer is yes on Llama/Qwen but Mistral still fails, stop the broad-method claim.

If the answer is yes and the frozen method prospectively repairs Mistral, then—and only
then—the paper direction deserves a broader top-tier validation campaign.
