# research_5.md
# Research-5 — Reliability-Aware Coarse-Format Quantization
## From Reconstruction Gain to Model-Quality Gain under Coarse N-Side × K64 Format-Control Constraints

**SPEC_VERSION:** `research_5_v1.2_2026-08-09`  
**Previous active spec:** `research_5_v1.1_2026-08-09`  
**Previous v1.1 research_5.md SHA256:** `f4169a2d591dae28422e5e75f8649801834f5c0bb10010f2baee541ce363019e`  
**Previous v1.1 coding_agent_prompt_5.md SHA256:** `ba4a1e784aa800db98bf9c1f5ba81dc9f9b699c90b7e10a295995a0cc824ec1a`  
**Earlier v1.0 research_5.md SHA256:** `25f196963519a16c2657fe5cb640141d0543a9dbf1ef2fdd1b45a294761623bb`  
**Earlier v1.0 coding_agent_prompt_5.md SHA256:** `5fa5aa109dba00fbffceff1202ffdccfe6bd87daf2c23ae4bb2bb43e3caab64b`  
**Program status:** `RESTART_DIAGNOSTIC_PHASE / SURROGATE_ALIGNMENT_UNRESOLVED`  
**Research-1~4 status:** completed first research program; immutable evidence  
**Research-4 scale×coupling status:** `SCALE_BRANCH_NO_SIGNAL / BOUNDED_REOPEN_CLOSED`  
**Primary algorithmic premise:** N8×K64 format control, K16 scale groups  
**Bounded paper-completeness granularity extension:** N8×K64 → N16×K64 accuracy/PPL only; hardware overhead out of scope  
**Strong existing baseline:** `FullLayer-CD2-N8K64 / CD2-Static`  
**Primary development families:** Llama-3.1-8B, Qwen3-8B, Mistral-7B-v0.3  
**Prospective families:** must be selected and sealed before Phase-2 method development  
**Primary statistical metric:** paired per-sequence/per-token NLL  
**PPL:** secondary interpretable metric  
**Native SM120:** `WAIT_FOR_SM120`  
**Research standard:** top-tier conference/journal standard; falsifiability over continuation  
**Core workflow change:** evidence-first, not method-first

---

# 0. Executive recommendation

Research-5 is a conceptual restart, not Research-4 with a fifth heuristic.

Research-1~4 established a coherent sequence:

```text
Research-1:
  fine K16 representation preferences exist
  ↓
  forcing format control to N8×K64 creates decision-compression loss

Research-2/3:
  local independent format objectives are incomplete
  ↓
  K64-region errors interact within each N8 output stripe
  ↓
  coupled CD2 substantially recovers the coarse loss on Llama/Qwen

Research-3:
  CD2 can be calibration-robust and deployment-static
  ↓
  but broad generalization fails on Mistral Wiki

Research-4:
  legal adaptive K16 scales improve held-out coupled reconstruction
  and change interaction geometry
  ↓
  yet frozen end-to-end NLL regresses in 3/4 development settings
```

The repeated unresolved pattern is:

\[
\boxed{
\text{reconstruction-surrogate improvement}
\not\Rightarrow
\text{deployment/model-quality improvement}.
}
\]

Therefore Research-5 must **not** ask:

> How can we lower the CD stripe SSE even more?

It must ask:

> **Which coarse-format quantization changes actually improve end-to-end model quality,
> can those useful changes be predicted before final evaluation, and can a static PTQ
> method safely exploit them under the same N8×K64 deployment representation?**

Research-5 will screen three deliberately different hypothesis families:

```text
A. Safe Acceptance / Reliability
   The existing coupled surrogate contains useful signal,
   but some apparently good changes are unreliable.

B. Selective Applicability
   Coupled assignment is useful only in some modules/regions;
   safe static fallback should be used elsewhere.

C. Model-Aware Objective
   Reconstruction/output SSE itself is the wrong or incomplete surrogate;
   model-loss-aware direction/sensitivity is needed.
```

The first major deliverable is a **shared intervention-based diagnostic dataset**.

A/B/C must initially consume the same intervention ground truth.

Only 0–2 directions may be promoted.

If no direction has meaningful cross-model predictive/mechanistic signal:

```text
COARSE_FORMAT_RESTART_NO_GO
```

and the research direction stops.

If one or two directions survive, Research-5 proceeds through:

```text
diagnostic screening
→ bounded method development
→ development-family closure
→ frozen method
→ truly unseen prospective family A
→ truly unseen confirmation family B / independent scale case
→ downstream
→ W4A4
→ optional SANA cross-domain gate
→ deployment accounting
→ final top-tier readiness decision
```

No new Direction D is allowed merely to avoid a negative result.

---

# 1. Authority and precedence

Use:

```text
CURRENT research_5.md
    >
CURRENT coding_agent_prompt_5.md
    >
immutable Research-4 final artifacts
    >
immutable Research-3 artifacts
    >
immutable Research-2 artifacts
    >
immutable Research-1 artifacts
    >
supplementary suggestion/review documents
    >
older prose summaries
```

Research-1~4 must not be rewritten.

---

## 1.1 In-flight spec migration from Research-5 v1.0/v1.1 to v1.2

Research-5 v1.2 is an **in-place execution-spec update** issued while the Coding Agent may
already be running valid Research-5 v1.0 or v1.1 work.

The update does **not** invalidate completed work merely because it was generated under an
older Research-5 spec.

At the first v1.2 checkpoint, the Coding Agent must re-read the full current `research_5.md`
and classify every Research-5 job/artifact as:

```text
COMPLETED_COMPATIBLE
  result already satisfies v1.2 semantics and provenance
  -> reuse; do not rerun

COMPLETED_NEEDS_AUDIT
  result is reusable but a v1.2 field/test/hash is missing
  -> audit/augment metadata; rerun only if scientific semantics differ

RUNNING_COMPATIBLE
  current project job remains valid under v1.2
  -> allow it to finish; annotate launch spec + v1.2 compatibility

RUNNING_NOW_INVALID
  current project job conflicts with a new v1.2 semantic/leakage/freeze rule
  -> preserve logs; safely stop only this project's job if appropriate;
     never kill another user's process; relaunch later if the experiment remains required

PLANNED_UPDATED
  not launched yet
  -> use v1.2 only
```

Mandatory migration artifacts:

```text
artifacts/research_5/00_environment/spec_update_v1_2_acknowledgement.md
artifacts/research_5/00_environment/spec_update_v1_2_manifest.json
artifacts/research_5/00_environment/inflight_reconciliation_v1_2.csv
```

`inflight_reconciliation_v1_2.csv` must include:

```text
experiment_id
launch_spec_version
current_status
v1_2_compatibility_class
reason
reuse_or_rerun
new_required_fields
action
```

Do not discard good v1.0/v1.1 results.

Do not relaunch expensive experiments solely because the version string changed.

From the v1.2 acknowledgement timestamp onward:

```text
all NEW launches must follow v1.2
```

The main A/B/C surrogate-alignment program remains unchanged in scientific priority.

Research-5 v1.2 keeps the v1.1 bounded W4A4 paper-completeness extension and adds one new,
strictly bounded accuracy-side granularity track:

```text
N8×K64
vs
N16×K64
```

The N8/N16 track is:

```text
accuracy / PPL / coupled-assignment mechanism only
```

and explicitly excludes:

```text
area
power
native latency
throughput
Tensor-Core synthesis
format-control overhead estimates
```

Those hardware-side quantities belong to a separate hardware effort.

The Coding Agent must **not interrupt valid A/B/C jobs merely to start N16**.

Queue the N8/N16 track at the v1.2 execution point defined below, unless the project has
already legitimately reached/passed that point.


Expected roots:

```text
artifacts/research_1/
artifacts/research_2/
artifacts/research_3/
artifacts/research_4/
artifacts/research_5/
```

Every Research-5 analysis must cite the exact immutable source hashes used.

---

# 2. Immutable scientific facts from Research-1~4

## 2.1 Hardware/control-granularity phenomenon

Weights:

```text
W[N,K]
```

Format-control region:

```text
N8 × K64
```

Scale group:

```text
K16
```

Many fine K16 representation preferences must be compressed into one coarser N8×K64
format decision.

This decision-compression phenomenon is established and must not be re-proven.

---

## 2.2 Fine heterogeneous representation has value

Research-1 showed fine K16 E2/E0 choice improves over fixed NVFP4 in the original four
Llama/Qwen Wiki/C4 settings.

Do not reopen:

```text
"does fine heterogeneity exist?"
```

It does.

---

## 2.3 Local selectors are incomplete

For an N8 stripe:

\[
E_{n,k}^{f}
=
X_{K_k}
\left(
Q_f(W_{N_n,K_k})-W_{N_n,K_k}
\right)^T.
\]

True stripe output error:

\[
J_n(F_n)
=
\left\|
\sum_k E_{n,k}^{f_{n,k}}
\right\|_F^2.
\]

Expansion:

\[
J_n
=
\sum_k\|E_{n,k}\|_F^2
+
2\sum_{i<j}
\langle E_{n,i},E_{n,j}\rangle_F.
\]

Cross-K cancellation/reinforcement is real.

Do not return to pure independent local MSE as the primary scientific answer.

---

## 2.4 BQP characterization remains valid

Using E2 as reference:

\[
D_k=E_k^{E0}-E_k^{E2},
\qquad
z_k\in\{0,1\},
\]

and:

\[
R_0=\sum_kE_k^{E2},
\]

the stripe objective is:

\[
J(z)
=
\left\|
R_0+\sum_kz_kD_k
\right\|_F^2
=
c+2g^Tz+z^TQz,
\]

where:

\[
Q_{ij}=\langle D_i,D_j\rangle_F.
\]

\(Q\) is a Gram matrix.

This is a useful analytical object, not the generic novelty claim.

---

## 2.5 CD2 is the strongest existing coarse-map generator/baseline

Frozen Research-3 method:

```text
FullLayer-CD2-N8K64
OutputAware initialization
2 coordinate sweeps
pooled 3×128 C4 calibration
E2 static target6
E0 static target7
```

Research-3 original four settings passed its strict method-closure gate.

Research-5 must preserve CD2 as:

```text
strong baseline
candidate-map generator
diagnostic intervention policy
```

CD2 does not have to remain the final method.

---

## 2.6 Research-3 broad failure remains real

Research-3's Mistral prospective gate showed:

```text
Mistral C4:
  positive CD2 behavior

Mistral Wiki:
  negative point estimate / broad-generalization gate failure
```

Therefore:

```text
same architecture
+
different distribution/corpus
can produce different applicability
```

Research-5 must not reduce the problem to a model-only classifier.

The relevant unit of generalization may be:

```text
model × distribution × module/context
```

---

## 2.7 Research-4 is a critical contrastive failure

Research-4 selected DualScale legitimately using calibration-only held-out coupled objective.

Calibration-level improvement was consistent:

```text
Llama: positive on all 3 folds
Qwen:  positive on all 3 folds
```

Yet frozen CD2-selected versus CD2-static model quality:

```text
Llama Wiki: regression
Llama C4:   improvement
Qwen Wiki:  regression
Qwen C4:    regression
```

The three regressions had paired NLL CIs entirely in the regression direction.

Research-4 also showed:

```text
candidate geometry changed
coupled objective gain increased
interaction terms changed
```

yet NLL did not reliably improve.

Thus:

\[
\boxed{
\text{stable held-out reconstruction gain is not sufficient evidence of model-quality gain}.
}
\]

This is one of the strongest reasons for Research-5.

---

## 2.8 Research-4 causal caution

Research-4 limitations state that the E2 canonical policy also used the pinned FourOverSix
256 tensor-normalization convention.

Therefore Research-5 may claim:

> the complete frozen adaptive policy improved the calibration surrogate but failed NLL.

It must **not** simplify that into:

> K16 scale target alone causally produced the NLL regression.

Research-4 is a contrastive policy failure, not a pure single-variable causal study.

---

# 3. Branches that remain closed

Research-5 must not reopen during diagnostic/method phases:

```text
generic rotation / Hadamard banks
generic packing / permutation
target-3/target-5 scale fishing
arbitrary scale multipliers
more CD sweeps merely to lower stripe SSE
model-specific format candidate sets
Wiki-specific rescue calibration
new activation quantizer, EXCEPT the explicitly bounded Phase-6 Activation-4Over6 paper-completeness policy
new runtime selector metadata, except algorithmic accounting of that bounded dynamic A4 selector
dynamic online controller
full Hessian before cheap C-surrogates
native SM120 claims without hardware
```

Research-4 DualScale should generally appear as:

```text
known contrastive negative/objective-mismatch reference
```

not as a primary competitive method.

---

# 4. New umbrella research question

Use:

> **Under a fixed coarse N8×K64 format-control constraint, which quantization-error
> improvements are causally or predictively relevant to end-to-end model quality,
> and how can PTQ safely exploit only those improvements without changing deployment
> semantics?**

Possible internal theme:

```text
Reliable Coarse-Format Quantization:
From Reconstruction Gain to Model-Quality Gain
```

Do not finalize a paper title before Phase-2 evidence.

---

# 5. Top-tier standard

Research-5 is not considered successful because:

```text
a new surrogate correlates on one model
a selective rule fixes one known failure
a learned classifier gets high random-split accuracy
a method improves average PPL but regresses one family/corpus
```

Top-tier readiness requires, at minimum:

### Mechanism

```text
shared intervention ground truth
clear evidence explaining why reconstruction gain can misrank model benefit
winner hypothesis supported across development families
Research-4 failure explained or flagged by at least one promoted mechanism
```

### Method

```text
one frozen deployable/static method
same N8K64 format-control representation
no final-eval tuning
no systematic regression on Llama/Qwen/Mistral development matrix
competitive with strong canonical baselines
```

### Prospective generalization

```text
at least one truly unseen family positive on >=2 predeclared distributions
paired CI supports non-regression
no cross-corpus sign reversal
```

For a strong top-tier claim, additionally require:

```text
one second independent confirmation:
  another unseen family
  OR a predeclared larger-model/architecture-scale case
```

### Task quality

```text
meaningful downstream suite
not a 100-example sanity check
```

### Deployment

```text
offline cost measured
metadata/storage explicit
runtime semantics unchanged
native claims kept separate until real SM120
```

### Granularity boundary completeness

The N8→N16 accuracy extension must be completed and honestly reported.

Research-5 does **not** require N16 to win.

Valid paper-complete outcomes include:

```text
PREFER_N16_ACCURACY_SAFE
N16_VIABLE_WITH_SMALL_ACCURACY_TRADEOFF
KEEP_N8_FOR_ACCURACY
N16_MODEL_DEPENDENT
```

The scientific requirement is to establish the accuracy boundary, not to force a favorable
N16 conclusion.

Hardware overhead is explicitly outside this Research-5 gate.

---

# 6. Current literature/novelty guardrails
# Verified planning snapshot: 2026-08-09

This section is positioning guidance, not a claim of exhaustive novelty.

Important neighboring work includes:

```text
BlockDialect (ICML 2025)
  per-block mixed numeric formats / formatbook

Adaptive Block-Scaled Data Types / IF4 (2026)
  FP4-vs-INT4 per small block with block scaling and hardware study

MixFP4 (2026)
  adaptive low-bit micro-formats under NVFP4-like scaling

AdaMX (2026)
  representation / precision-recovery heterogeneity and hardware realization

Rethinking Residual Errors in Compensation-based LLM Quantization (2026)
  teacher-aligned / residual-aware calibration

MixQuant (2026)
  layer sensitivity depends on upstream quantization configurations

SliderQuant (2026)
  layer-dependent PTQ sensitivity

DAQ (2026)
  reconstruction metrics can miss a task/parameter-delta-relevant direction

Understanding Quantization-Aware Training (2026)
  local reconstruction/Hessian-style PTQ objectives can select high-loss quantized points

FAQ / calibration-regeneration work (2026)
  calibration-distribution representativeness matters
```

Therefore Research-5 must not claim generic novelty as:

```text
mixed formats
layer sensitivity
selective quantization
teacher alignment
residual compensation
reconstruction error is imperfect
calibration data matters
gradient sensitivity
Fisher weighting
```

Potential novelty must remain specific to:

```text
coarse format-decision compression
+
coupled cross-K format assignment
+
predictive reliability/applicability/model-aware surrogate
+
same N8K64 deployment representation
```

Fresh literature audit is mandatory again before paper writing.

---

# 7. Research-5 hypothesis families

## Direction A — Safe Acceptance / Reliability

Hypothesis:

> The existing coupled reconstruction signal is useful, but apparently good format-map
> changes differ in reliability. Calibration-derived confidence can identify harmful
> or non-transferable changes before deployment evaluation.

Important Research-4 challenge:

> A simple same-surrogate three-shard consensus is unlikely to be sufficient by itself,
> because Research-4 DualScale improved the coupled objective on all six Llama/Qwen folds
> and still regressed end-to-end in 3/4 settings.

Therefore A must test both:

```text
within-distribution calibration reliability
and
bounded cross-distribution surrogate stability
```

without using final prospective-family outcomes.

---

## Direction B — Selective Applicability

Hypothesis:

> CD2 is conditionally useful. Some modules benefit from coupled assignment; others should
> retain a safe fallback. A static offline applicability policy can preserve useful CD gain
> while avoiding harmful modules.

Before building any predictor, prove applicability headroom with intervention ground truth.

---

## Direction C — Model-Aware Objective

Hypothesis:

> Coupled output SSE is incomplete because equal or lower reconstruction error can point in
> directions with very different effects on language-model loss.

Candidate predictors:

```text
first-order loss sensitivity
diagonal Fisher / Gauss-Newton proxy
first+Fisher Taylor score
teacher-logit divergence
```

First test prediction of intervention ΔNLL.

Only then use a winning score as an optimization/selection objective.

---

# 8. A/B/C independence rule

Phase-1 A/B/C must remain independent.

Do not initially combine:

```text
shard confidence
gradient score
applicability router
teacher KL
```

into one model.

Each direction must produce its own scorecard.

At Phase-1 gate:

```text
promote 0, 1, or at most 2 directions
```

Never all three.

If three have weak positive metrics, choose at most two by:

```text
cross-model evidence
effect magnitude
simplicity
calibration cost
deployment compatibility
```

---

# 9. Truly unseen prospective families must be sealed early

Before Phase-2 method development:

1. choose a primary unseen LLM family not used in Research-1~4;
2. choose a secondary unseen confirmation family if available;
3. record exact model ID/revision/license/access;
4. verify only that loading/harness support works **without running quantization outcome evaluation**;
5. seal evaluation outputs.

Preferred primary characteristics:

```text
non-Llama
non-Qwen
non-Mistral
dense transformer
~7B–10B preferred
fits available GPUs
existing evaluation harness support
```

A reasonable preordered candidate policy may be:

```text
1. Gemma-family ~9B model if supported/access permitted
2. another non-Llama/Qwen/Mistral dense 7B–14B family
```

Secondary confirmation should be architecturally distinct if feasible.

If a candidate cannot load for access/dependency reasons before any result inspection,
the next predeclared candidate may be selected and the reason must be logged.

Machine field:

```text
prospective_family_was_accessed_pre_freeze = false
```

must remain false for quality outputs.

Known development families are:

```text
Llama-3.1-8B
Qwen3-8B
Mistral-7B-v0.3
```

Mistral is no longer prospective evidence.

---

# 10. Phase 0 — environment, handoff, and baseline reconstruction

## 10.1 Create isolated project/worktree

Create:

```text
research_5_restart
```

branch/worktree.

Record:

```text
repo URL
HEAD
branch
submodules
dirty state
Research-1~4 source hashes
local patch hash
```

No push unless explicitly requested.

---

## 10.2 Audit immutable artifacts

Hash and inventory:

```text
Research-1
Research-2
Research-3
Research-4
```

Regenerate from raw artifacts:

```text
Research-3 CD2-Static original-four table
Research-3 Mistral result
Research-4 selected policy table
Research-4 3/4 CD2-selected regression table
Research-4 calibration fold gains
Research-4 interaction geometry summary
```

Do not use copied prose as source of truth.

---

## 10.3 Frozen baseline set

Research-5 baseline policies:

```text
HighPrecision
NVFP4
Canonical-4Over6
Raw-N8K64-Static
OutputAware-N8K64-Static
CD2-Static
Fine16-Static  # diagnostic reference
Research4-DualScale  # negative contrastive reference, not primary
```

Do not change their semantics.

---

## 10.4 Baseline equivalence tests

Require:

```text
Raw static bit/map reproduction
CD2 static map reproduction
4Over6 semantic reduction
Research4 DualScale reproduction on one Llama + one Qwen checkpoint
per-sequence NLL evaluator determinism
intervention runner no-op equivalence
```

---

# 11. Phase 0B — prospective-family sealing

Before feature/threshold method development:

```text
prospective_family_A.json
prospective_family_B.json  # if feasible
```

Each contains:

```text
model ID
revision
selection timestamp
selection rule
availability check only
corpora to be evaluated later
output seal status
```

Do not evaluate quantization quality.

---

# 12. Phase 1 — build the shared intervention ground-truth dataset

This is the most important Research-5 experiment.

Do **not** begin by implementing a new final quantizer.

The dataset must answer:

> Which module-level coarse-format changes actually help or hurt paired model NLL?

---

# 13. Unit of intervention

Primary:

```text
linear module
```

Examples:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Secondary:

```text
selected N8 output stripes
```

Stripes are for mechanism validation and local surrogate analysis.

Do not exhaustively measure true end-to-end ΔNLL for every stripe.

---

# 14. Primary module sampling

For each known model, pre-register approximately:

```text
48 modules
```

initially.

Minimum:

```text
30
```

Maximum initial:

```text
60
```

Stratify before seeing intervention labels across:

```text
module type
normalized layer depth
CD2 objective-gain quantile
cross-interaction quantile
CD-vs-OutputAware format-disagreement quantile
calibration-instability quantile
format-flip-rate quantile
```

Ensure attention and MLP modules are both represented.

Preserve exact sample manifest.

If class/effect diversity is insufficient after Phase-1 pilot, an expansion is allowed only
using a predeclared stratum-balancing rule, not outcome cherry-picking.

---

# 15. Diagnostic corpora

Known families are development families.

Use two diagnostic distributions:

```text
C4-like diagnostic set
Wiki-like diagnostic set
```

with fixed disjoint sequence manifests.

Recommended initial size:

```text
128 sequences per diagnostic distribution
```

If a module label remains uncertain, expand that module's evaluation deterministically to:

```text
256 sequences
```

using predeclared additional IDs.

Do not expand based on whether the sign is favorable; expand only based on CI width/uncertainty.

Full development Wiki/C4 evaluation remains separate for Phase-2 method closure.

---

# 16. Calibration feature distributions

For A and C calibration features, use:

```text
C4 calibration shards
```

from the established protocol.

Additionally construct one disjoint:

```text
Wiki-like calibration microset
```

for **reliability diagnostics only**.

This is not final evaluation data.

Its role is to test whether surrogate/coupling statistics are distribution-stable.

Do not add many calibration domains in Phase 1.

Optional third code-like domain is locked unless the C4-vs-Wiki reliability signal is
clearly useful and top-tier generalization later requires it.

---

# 17. Mandatory intervention contexts

For each sampled module \(l\), measure both directions.

## 17.1 Raw → CD2 insertion

Base:

```text
Raw-N8K64-Static everywhere
```

Intervention:

```text
replace only module l with CD2-Static module l
```

Measure:

\[
\Delta NLL_l^{insert}
=
NLL(Raw + CD_l)
-
NLL(Raw).
\]

Negative is beneficial.

---

## 17.2 CD2 → Raw fallback

Base:

```text
CD2-Static everywhere
```

Intervention:

```text
replace only module l with Raw-N8K64-Static
```

Measure:

\[
\Delta NLL_l^{fallback}
=
NLL(CD2 \setminus l + Raw_l)
-
NLL(CD2).
\]

Positive means CD2 at module \(l\) is useful in the full-CD context.

---

## 17.3 Why both contexts are mandatory

Module effects need not be additive.

A module can be:

```text
beneficial when inserted into Raw
but redundant/harmful in full CD context
```

or vice versa.

Using both insertion and fallback prevents a single-context label from being mistaken for
universal ground truth.

---

# 18. Context-consistent labels

Primary continuous labels remain:

```text
delta_nll_insert
delta_nll_fallback
```

Also define context-consistent categories.

For a chosen numerical/practical threshold \(\epsilon\):

### CD_BENEFICIAL

```text
insert CI upper < -epsilon
AND
fallback CI lower > +epsilon
```

### CD_HARMFUL

```text
insert CI lower > +epsilon
AND
fallback CI upper < -epsilon
```

### CONTEXT_DEPENDENT

```text
one context significant
or
contexts disagree
```

### UNCERTAIN

```text
both CIs overlap the epsilon zone
```

Do not force every module into binary labels.

---

# 19. Noise-floor / epsilon protocol

Do not tune \(\epsilon\) to maximize classifier metrics.

Before large intervention analysis:

1. rerun no-op/baseline and a small fixed intervention subset;
2. estimate numerical/reproducibility NLL noise;
3. freeze a primary \(\epsilon\) based on the measured noise floor;
4. report sensitivity at a small predeclared set around it.

Example sensitivity bank:

```text
epsilon_primary
0.5 × epsilon_primary
2.0 × epsilon_primary
```

Primary conclusions must not depend on one arbitrary epsilon.

---

# 20. Paired intervention statistics

For each intervention:

```text
per-sequence NLL
per-token NLL
paired delta
token count
mean
median
95% paired bootstrap CI
win fraction
bootstrap seed
```

Report model/corpus-specific statistics before pooled statistics.

Do not treat modules from the same model as independent model-family evidence.

---

# 21. Research-4 contrastive intervention subset

Research-4 must be used as a known failure case, not hidden.

For a predeclared subset of approximately:

```text
12–20 modules per Llama/Qwen model
```

stratified by:

```text
largest DualScale-vs-Static surrogate gain
small surrogate gain
different module types/depths
```

measure:

### Static → DualScale insertion

Base:

```text
CD2-Static
```

replace only module \(l\) with the frozen Research-4 DualScale module.

\[
\Delta NLL_l^{R4-insert}
=
NLL(StaticCD + DualScale_l)
-
NLL(StaticCD).
\]

### DualScale → Static fallback

Base:

```text
Research4-DualScale
```

fallback only module \(l\) to Static CD2.

This creates direct ground truth for:

> Which locally/surrogate-improving Research-4 changes were actually harmful?

A/B/C should be explicitly scored on this contrastive subset.

---

# 22. Shared feature extraction protocol

Critical leakage rule:

```text
extract and freeze calibration-only features
BEFORE merging with intervention outcome labels
```

Artifact workflow:

```text
features.parquet
feature_hash
then
labels.parquet
then
merged_dataset.parquet
```

No feature may depend on final intervention NLL.

---

# 23. Shared reconstruction features

Per module:

```text
raw_local_output_sse
oa_local_output_sse
cd2_output_sse
cd2_objective_gain
normalized_cd2_gain
fine_to_coarse_gap_proxy
relative_output_error
output_nmse
```

For Research-4 contrastive policy:

```text
dualscale_output_sse
dualscale_vs_static_surrogate_gain
```

---

# 24. Shared coupling features

Aggregate stripe-level quantities into module summaries:

```text
cross_interaction_C_mean
cross_interaction_C_std
pairwise_coupling_abs_sum_mean
offdiag_coupling_ratio_mean
offdiag_coupling_ratio_p90
negative_coupling_fraction_mean
positive_coupling_fraction_mean
interaction_spectral_norm_mean
largest_abs_pairwise_coupling
```

Also:

```text
hard_stripe_fraction
```

using a predeclared coupling/conflict quantile.

---

# 25. Shared map/search features

```text
format_flip_count
format_flip_rate
sweep1_flip_count
sweep2_flip_count
second_sweep_fraction
E0_ratio_before
E0_ratio_after
format_hamming_OA_vs_CD2
format_margin_mean
format_margin_p10
format_margin_near_tie_fraction
```

---

# 26. Shared calibration-robustness features

From established independent C4 shards:

```text
objective_gain_shard_mean
objective_gain_shard_std
objective_gain_shard_min
objective_gain_shard_max
objective_gain_sign_consensus
map_agreement_across_shards
heldout_calibration_gain
heldout_calibration_gain_std
```

Important:

> Research-4 shows these features may not be sufficient, so they are tested rather than
> assumed.

---

# 27. Cross-distribution reliability features

Using disjoint C4-like and Wiki-like calibration microsets:

```text
objective_gain_c4
objective_gain_wiki_like
objective_gain_cross_domain_min
objective_gain_cross_domain_mean
objective_gain_cross_domain_sign_consensus

format_map_hamming_c4_vs_wiki_like
E0_ratio_shift
```

For BQP/coupling summaries:

```text
coupling_offdiag_shift
negative_coupling_fraction_shift
interaction_spectral_norm_shift
```

If representative interaction matrices \(Q\) are stored:

\[
QShift
=
\frac{
\|Q_{C4}-Q_{WikiLike}\|_F
}{
\|Q_{C4}\|_F+\epsilon
}.
\]

This is a diagnostic of surrogate-geometry shift, not a guarantee of NLL transfer.

---

# 28. Activation/context features

At minimum:

```text
activation_norm
activation_second_moment
activation_max
activation_kurtosis_or_robust_tail_proxy
activation_condition_proxy
output_norm
normalized_output_error
```

Do not create a huge activation feature zoo.

---

# 29. Direction A — Safe Acceptance screening

A tests whether calibration-only reliability can predict harmful CD/module changes.

A must not use gradient/Fisher/teacher-logit features in Phase 1.

---

# 30. A1 — same-distribution shard consensus

Candidate reliability variables:

```text
mean objective improvement
std
worst-shard improvement
sign consensus
map agreement
train-vs-heldout objective gap
```

Test whether harmful interventions concentrate at:

```text
high variance
weak worst-shard gain
low sign consensus
large train/heldout gap
```

---

# 31. A2 — conservative confidence score

If \(\Delta J\) denotes candidate minus baseline, lower is better.

Predeclare a tiny bank:

\[
Score_A(\lambda)
=
E[\Delta J]+\lambda Std[\Delta J],
\]

with:

```text
lambda ∈ {0, 0.5, 1.0}
```

or equivalent bootstrap upper confidence bound.

Do not tune a large lambda grid.

---

# 32. A3 — train/validation agreement

A change is high confidence only if:

```text
calibration-train objective improves
AND
calibration-validation objective does not regress
```

Quantify margin, not just a boolean.

---

# 33. A4 — cross-distribution consistency

Because Research-4 defeated simple same-surrogate shard consensus, A must also test:

```text
C4-like calibration gain
vs
Wiki-like calibration gain
```

Candidates:

```text
minimum cross-domain gain
cross-domain sign consensus
QShift / coupling-geometry shift
map disagreement
```

Hypothesis:

> harmful/non-transferable changes are enriched among changes whose apparent surrogate
> improvement is distribution-sensitive.

This is still A because it uses reconstruction/coupling reliability only, not task gradients.

---

# 34. A5 — risk–coverage / abstention analysis

Rank module changes by confidence.

For coverage:

```text
25%
50%
75%
100%
```

simulate:

```text
accept CD on highest-confidence modules
fallback elsewhere
```

Use measured intervention labels first.

Report:

```text
harmful intervention rate
retained beneficial NLL gain
risk
coverage
```

Do not cherry-pick one coverage.

---

# 35. Direction A metrics

Required:

```text
harmful-detection AUROC
harmful-detection AUPRC
Spearman(confidence, -ΔNLL)
risk@25/50/75/100%
beneficial gain retained
harmful rate
cross-model held-out metrics
Research-4 contrastive-case detection
```

Primary validation:

```text
leave-one-model-out
```

across Llama/Qwen/Mistral.

Random module split is secondary only.

---

# 36. Direction A promotion gate

Promote A only if at least one predeclared score:

```text
harmful AUROC >= 0.70
```

on cross-model held-out diagnosis for at least two of three held-out-family tests,

and near 50% coverage:

```text
harmful intervention rate reduced >= 50%
while retaining >= 70% of beneficial CD gain
```

relative to unconditional CD application.

Additionally, if Research-4 contrastive subset contains confirmed harmful changes:

```text
harmful R4 changes must be enriched in the low-confidence / abstain region
```

If only same-distribution shard consensus appears positive but fails the R4 contrastive
case:

```text
A_SIMPLE_CONSENSUS_REJECTED
```

Do not promote that score.

If no A score passes:

```text
A_NO_SIGNAL
```

---

# 37. Direction B — Selective Applicability screening

B asks:

> Does module-selective use of CD2 have real headroom over any global policy?

Do not train a router before proving headroom.

---

# 38. B0 — one-module applicability oracle proxy

Using measured intervention outcomes, construct a diagnostic:

```text
one_module_applicability_oracle_proxy
```

For each module, identify whether CD2 or Raw is beneficial under measured contexts.

Important:

```text
not a deployable method
not a formal global oracle
```

because module effects interact.

Report oracle-proxy headroom separately by:

```text
model
corpus
module type
depth
```

---

# 39. B1 — cumulative composition test

Rank modules by **measured development intervention benefit** only for oracle-headroom
diagnosis.

Construct cumulative static models with CD enabled for:

```text
top 10%
top 25%
top 50%
top 75%
all
```

and Raw elsewhere.

Evaluate actual NLL.

This tests whether isolated module effects compose.

If the cumulative curve does not improve over both:

```text
Raw-all
CD-all
```

meaningfully:

```text
B_NO_HEADROOM
```

Stop B.

---

# 40. B2 — fallback choice

If Raw/CD selective headroom exists, compare two fixed fallbacks:

```text
Raw-N8K64-Static
Canonical-4Over6
```

Do not add more policies.

Audit representation compatibility.

Question:

> Is the useful choice primarily CD-vs-Raw, or does 4Over6 provide a safer static fallback?

---

# 41. B3 — calibration-only applicability predictor

Only if B0/B1 pass.

Use interpretable low-capacity models:

```text
single threshold
regularized logistic regression
depth-limited shallow decision tree
```

Feature inputs may include:

```text
reconstruction
coupling
map/search
robustness
activation
cross-distribution reliability
```

but not model ID and not final-eval NLL.

Use nested:

```text
leave-one-model-out
```

for model selection and evaluation.

No random module split as primary evidence.

---

# 42. Direction B metrics

```text
oracle-proxy headroom
cumulative coverage–NLL
predictor retained CD benefit
regression avoidance
fraction fallback
policy disagreement
cross-model held-out AUROC/ranking
fraction of oracle headroom captured
offline predictor cost
```

---

# 43. Direction B promotion gate

Promote B only if:

1. oracle/cumulative selective policy shows substantial headroom;
2. at least one known regression/weak setting is materially improved;
3. positive CD settings retain at least:

```text
>=70% of their positive CD gain
```

4. calibration-only predictor captures approximately:

```text
>=50% of measured selective headroom
```

in leave-one-model-out testing.

If oracle selective routing itself gives little improvement:

```text
B_NO_HEADROOM
```

Stop B before predictor development.

---

# 44. Direction C — Model-Aware Objective screening

C asks:

> Which inexpensive model-aware score predicts real intervention ΔNLL better than coupled
> reconstruction SSE?

Do not immediately optimize every new surrogate.

First test predictiveness on the shared ground truth.

---

# 45. C0 — coupled SSE baseline predictor

For each intervention, compute the predicted change in the existing reconstruction/coupled
objective.

Name:

```text
S_SSE
```

All C scores must beat this baseline meaningfully.

---

# 46. C1 — first-order loss sensitivity

For module output perturbation:

\[
\Delta Y_l
=
Y_l^{candidate}
-
Y_l^{base},
\]

compute calibration language-model loss gradient at the base-policy module output:

\[
G_l
=
\nabla_{Y_l}\mathcal L.
\]

First-order predicted loss change:

\[
S_{1,l}
=
\langle G_l,\Delta Y_l\rangle.
\]

Use the same diagnostic sequence IDs.

For insertion:

```text
base = Raw model
candidate = Raw with CD module
```

For fallback:

```text
base = CD model
candidate = CD with Raw module
```

This context-matched design is preferred over one universal gradient.

---

# 47. C2 — diagonal Fisher / Gauss–Newton proxy

Estimate:

\[
D_{l,i}
=
E[g_{l,i}^2].
\]

Score:

\[
S_{F,l}
=
\frac12
\sum_i
D_{l,i}
(\Delta Y_{l,i})^2.
\]

Also compute the Taylor score:

\[
S_{1+F,l}
=
\langle G_l,\Delta Y_l\rangle
+
\frac12\sum_iD_{l,i}(\Delta Y_{l,i})^2.
\]

No arbitrary lambda is needed for the primary Taylor form.

If an alternative scaling is tested, it must be predeclared and bounded.

No full Hessian in Phase 1.

---

# 48. C3 — teacher-logit divergence

Use a fixed HighPrecision teacher on the same diagnostic inputs.

For base and intervention logits:

\[
p_{HP},\quad p_{base},\quad p_{cand},
\]

define the primary teacher score as:

\[
\Delta KL_l
=
D_{KL}(p_{HP}\|p_{cand})
-
D_{KL}(p_{HP}\|p_{base}).
\]

Negative predicts movement toward the teacher.

This difference is more informative than reporting candidate KL alone.

Secondary diagnostics:

```text
delta logit cosine
delta top-k overlap
```

but predeclare `delta_teacher_KL` as primary.

Teacher KL is initially a predictor, not a final optimizer.

---

# 49. C4 — Research-4 contrastive prediction

For Static→DualScale module interventions, compute:

```text
S_SSE
S1
SF
S1+F
delta_teacher_KL
```

A strong C result should explain at least part of:

```text
surrogate says better
but module/end-to-end NLL says worse
```

If all model-aware scores simply agree with the misleading SSE on confirmed harmful R4
interventions, C is weakened substantially.

---

# 50. Optional C5 — quantized-prefix sensitivity

Locked initially.

Unlock only if:

```text
C1–C3 show real predictive signal
but remaining failures cluster by depth/prefix context
```

Then compute candidate perturbations and gradients under:

```text
actual quantized prefix
```

rather than isolated FP layer input.

Do not repeat Research-3 teacher-alignment blindly.

This is a bounded diagnostic extension.

---

# 51. Direction C metrics

For every score:

```text
Pearson(score, ΔNLL)
Spearman(score, ΔNLL)
harmful AUROC
beneficial AUROC
sign accuracy excluding uncertain cases
model-wise results
corpus-wise results
module-type results
depth-quartile results
calibration compute/memory cost
```

Orientation must be normalized so:

```text
larger predicted harm = larger actual ΔNLL
```

before pooled ranking.

---

# 52. Direction C promotion gate

Promote C if at least one model-aware score:

```text
Spearman >= 0.4–0.5
```

on cross-model/module intervention ranking,

and improves over coupled SSE by approximately:

```text
ΔSpearman >= 0.10
OR
harmful AUROC improvement >= 0.10
```

on at least two development-family/corpus settings,

and does not derive all gain from one module type.

At least two of three leave-one-model-out held-out-family tests must retain positive
predictive advantage over SSE.

If no model-aware score beats SSE:

```text
C_NO_SIGNAL
```

Stop C.

---

# 53. Phase-1 measurement gate before A/B/C conclusions

Before any promotion decision, verify intervention ground truth is usable.

Require:

```text
no-op rerun numerical noise small relative to measured effects
paired evaluation deterministic
at least a non-trivial set of BENEFICIAL/HARMFUL/CONTEXT_DEPENDENT modules
not >~80–90% UNCERTAIN after deterministic sequence expansion
```

If almost all interventions are uncertain:

```text
INTERVENTION_MEASUREMENT_INSUFFICIENT
```

Do not invent methods.

Improve measurement/sample size within the predeclared maximum and re-evaluate.

If ground truth remains unusable:

```text
COARSE_FORMAT_RESTART_NO_GO
```

---

# 54. Phase-1 scorecard

Produce exactly:

| Direction | Core hypothesis | Primary evidence | Status |
|---|---|---|---|
| A | reliability predicts harmful change | LO-model-out AUROC, risk–coverage, R4 contrastive | PASS/FAIL |
| B | selective CD has usable headroom | cumulative oracle headroom + predictor | PASS/FAIL |
| C | model-aware surrogate predicts ΔNLL | Spearman/AUROC vs SSE + R4 contrastive | PASS/FAIL |

Promote:

```text
0
1
or at most 2
```

directions.

If all fail:

```text
COARSE_FORMAT_RESTART_NO_GO
```

This is a terminal algorithmic decision.

---

# 55. Phase 2 — bounded method development
# Only promoted directions may create methods

No new hypothesis family may be introduced in Phase 2.

Method development uses:

```text
Llama
Qwen
Mistral
```

as development families.

Prospective families remain sealed.

---

# 56. Phase-2 development/evaluation discipline

Create three roles:

```text
module-diagnostic data
method-calibration/validation data
development full evaluation
```

Thresholds/predictors must be chosen using calibration/diagnostic development data.

Full development Wiki/C4 PPL/NLL is used for method comparison **after each candidate is
frozen**, not for repeated threshold fishing.

Use minimax/non-regression preference over average gain.

---

# 57. Method path A — Reliability-Gated CD

If A passes:

1. generate standard CD2 module maps;
2. compute the winning A confidence score per module;
3. accept CD2 only for modules above a confidence threshold;
4. fallback elsewhere to the B2-selected safe baseline if B2 exists, otherwise Raw;
5. compile one final static format/scale map.

No runtime controller.

Threshold selection:

```text
predeclared coverage/risk candidates
nested leave-one-model-out
choose the simplest threshold meeting a target non-regression/risk constraint
```

Do not train a large classifier for A.

Primary method name:

```text
Reliable-CD
```

internal only.

---

# 58. Method path B — Selective-CD

If B passes:

Use the simplest B predictor that captured meaningful oracle headroom.

Candidate hierarchy:

```text
single threshold
logistic regression
small shallow tree
```

Train on all three development families only after leave-one-model-out validation.

Choose:

```text
CD2 module
or
safe fallback module
```

offline.

Compile into one static final quantization map.

No runtime routing metadata beyond the resulting weights/scales/format bits.

Internal name:

```text
Selective-CD
```

---

# 59. Method path C — Loss-Aware coarse-format optimization

If C passes, the preferred final method depends on which C score wins.

## 59.1 If first-order/Fisher/Taylor wins

Construct a direct loss-aware coarse-format objective.

Use a base quantized policy, initially:

```text
Raw-N8K64-Static
```

on calibration sequences.

For one stripe, candidate perturbation from the base map:

\[
\Delta Y(z)
=
R_0+\sum_kz_kD_k.
\]

With calibration loss gradient \(G\) and diagonal Fisher \(D_F\):

\[
S_{LA}(z)
=
\langle G,\Delta Y(z)\rangle
+
\frac12
\sum_i
D_{F,i}
\Delta Y_i(z)^2.
\]

This remains a binary quadratic objective.

Define Fisher-weighted inner product:

\[
\langle A,B\rangle_F
=
\sum_iD_{F,i}A_iB_i.
\]

Then pairwise terms are:

\[
Q_{ij}^{F}
=
\langle D_i,D_j\rangle_F.
\]

The gradient contributes a linear term.

Use bounded CD:

```text
deterministic initialization
<=2 sweeps initially
```

Do not add global solvers.

This path directly replaces the SSE surrogate with an approximate model-loss objective
while preserving N8K64 format control.

Internal method:

```text
LossAware-CD
```

---

## 59.2 If teacher-KL wins but Taylor scores do not

Do not force teacher KL into an expensive per-bit global optimizer immediately.

First use:

```text
module-level choice among Raw / CD2
```

based on calibration `delta_teacher_KL`.

If that static selector is strong on development-family held-out evaluation, it may be the
C-method.

Only pursue finer teacher-KL format-bit search if module-level evidence leaves clear
headroom and cost is acceptable.

---

# 60. C-method objective tests

For Taylor LossAware-CD:

```text
first-order term reconstruction exact
Fisher-weighted quadratic term exact
single-bit flip delta == direct recomputation
accepted flip decreases approximate loss objective
deterministic replay
```

Also store:

```text
approx objective change
actual module intervention ΔNLL
```

to verify the optimized objective remains aligned.

---

# 61. If two directions pass — combination rules

At most one predeclared combination may be tested.

Do not exhaustively combine everything.

### A + C

Preferred combination:

```text
LossAware-CD proposals
+
A reliability abstention
```

Use if A clearly detects failures of the C objective.

### B + C

Preferred:

```text
C score/objective
+
B static applicability/fallback
```

only if B has strong oracle headroom.

### A + B

Preferred:

```text
B applicability predictor
+
A confidence abstention
```

No model-aware objective.

Combination is allowed only after both branches independently validate.

---

# 62. Phase-2 candidate cap

Maximum candidate methods entering full development evaluation:

```text
3
```

For example:

```text
best single branch A/B/C
second independently promoted branch
one justified combination
```

Do not enter five methods into full evaluation.

---

# 63. Development-family method closure

Evaluate frozen candidates on:

```text
Llama Wiki + C4
Qwen Wiki + C4
Mistral Wiki + C4
```

Required baselines:

```text
HighPrecision
NVFP4
Canonical-4Over6
Raw-N8K64-Static
OutputAware-N8K64-Static
CD2-Static
Fine16-Static diagnostic
Research4-DualScale negative reference where useful
```

Primary paired comparisons:

```text
candidate vs Raw
candidate vs CD2
candidate vs 4Over6
```

---

# 64. Development-family METHOD_GO gate

A candidate can be frozen for prospective testing only if:

```text
1. no systematic regression across Llama/Qwen/Mistral × Wiki/C4;
2. point estimate improves Raw in >=5/6 settings;
3. remaining setting(s) are statistically non-inferior / CI does not support meaningful regression;
4. competitive with Canonical-4Over6 in all 6;
5. fixes or avoids the known Mistral Wiki weakness;
6. no model-specific/corpus-specific tuning;
7. same runtime N8K64 representation;
8. offline cost measured;
9. mechanism branch evidence remains consistent with method behavior.
```

Internal design target:

```text
positive paired ΔNLL improvement in >=4/6 with CI excluding 0
```

is desirable but not an absolute universal law.

If no candidate passes:

```text
METHOD_DEVELOPMENT_NO_GO
```

Stop.

---

# 65. Final method selection rule

If multiple candidates pass:

Rank by:

```text
1. worst-case paired ΔNLL / non-regression
2. average gain
3. performance vs 4Over6
4. simplicity
5. calibration cost
6. no extra metadata
```

Do not pick by average PPL alone.

Freeze exactly one:

```text
frozen_method.json
frozen_method_hashes.json
freeze_report.md
```

No changes after prospective family A is unsealed.

---

# 66. Phase 3A — prospective family A

Only after final method freeze.

Unseal the primary prospective family selected in Phase 0.

Use the same frozen:

```text
quantizer semantics
feature set
thresholds
predictor
objective
calibration size
calibration distributions
sweep count
fallback policy
```

No family-specific changes.

Evaluate at minimum:

```text
WikiText2-like setting
C4-like setting
```

using fixed manifests.

If a third standard corpus is already supported and was preregistered in Phase 0, run it too.

---

# 67. Prospective family A gate

Require:

```text
no cross-corpus sign reversal
candidate <= Raw NLL point estimate on all predeclared corpora
paired CI does not support meaningful regression
competitive with Canonical-4Over6 on all corpora
mechanism/reliability score behaves in the expected direction
```

Strong success:

```text
positive paired improvement with CI excluding zero on >=1 corpus
and safe/non-inferior on the others
```

If family A fails:

```text
PROSPECTIVE_GENERALIZATION_FAIL
```

Final stop.

No rescue tuning.

---

# 68. Phase 3B — second independent confirmation

For a strong top-tier method claim, one prospective family is not enough.

If family A passes, unseal:

```text
prospective family B
```

if preselected and available.

If a second family is unavailable, use one predeclared independent architecture/scale case,
but final status must record lower generalization strength.

Evaluate the same >=2 corpora with no retuning.

---

# 69. Confirmation gate

Require:

```text
no systematic regression
competitive with 4Over6
no cross-corpus reversal
```

If family B fails:

```text
CONDITIONAL_PAPER_GO
```

may remain for a narrower claim, but do not call the method broadly top-tier-ready without
an explicit limitation.

If A and B both pass:

```text
PROSPECTIVE_STRONG_GO
```

---

# 70. Phase 4 — model-size scaling

After prospective strong evidence:

Run one larger model:

```text
~12B–14B preferred
```

chosen/predeclared before results.

Purpose:

```text
algorithmic/calibration scaling
not another favorable-family search
```

Record:

```text
quality
calibration wall time
VRAM/RAM
map generation time
feature extraction cost
predictor/objective cost
```

---

# 71. Phase 5 — proper downstream evaluation

Use a fixed full/meaningful task suite supported by the harness.

Recommended if available:

```text
HellaSwag
PIQA
ARC-Challenge
WinoGrande
BoolQ
MMLU
```

Optionally include:

```text
GSM8K
```

only if generation/evaluation protocol is stable and predeclared.

Do not use 100-example sanity numbers as main evidence.

Compare at minimum:

```text
HighPrecision
NVFP4
Canonical-4Over6
Raw-N8K64
CD2-Static
Research-5 final method
```

Report:

```text
per-task metric
macro average
confidence/bootstrap when supported
```

---


# 71B. Phase 5B — N8×K64 → N16×K64 weight-granularity accuracy extension
## Accuracy/PPL and coupled-assignment mechanism only — hardware overhead out of scope

This is a bounded paper-completeness / hardware-handoff accuracy experiment.

It does **not** replace the N8-based Research-5 surrogate-alignment program.

It asks:

> **Can one weight format decision be shared across N16×K64 instead of N8×K64 without
> materially degrading end-to-end model quality, and can the same frozen CD2 principle
> recover the additional decision-compression loss?**

The experiment must not redesign the algorithm.

---

## 71B.1 Exact scope

Primary comparison:

```text
N8×K64:
  Raw-N8K64
  CD2-N8K64

vs

N16×K64:
  Raw-N16K64
  CD2-N16K64
```

Keep identical:

```text
E2M1 / project-E0 candidate semantics
K16 scale granularity
scale policy
calibration source/IDs
sequence length
calibration size
layer-input semantics
OutputAware/local initialization semantics
two coordinate sweeps
evaluation corpora
model revisions
random seeds
```

The only intended algorithmic change is:

```text
N8 output stripe
→
N16 output stripe
```

No:

```text
CD3
extra sweeps
new search order
new objective
new scale policy
new calibration rule
model-specific N16 rescue
```

---

## 71B.2 Hardware overhead is explicitly out of scope

Do not attempt to validate or reproduce claims such as:

```text
N8K64 hardware overhead ≈ some value
N16K64 hardware overhead ≈ some value
```

Do not spend Research-5 effort on:

```text
area
power
native Tensor-Core synthesis
native latency
throughput
decoder/control overhead
format-bit routing overhead
```

The accuracy-side handoff may provide only:

```text
paired NLL/PPL cost of N16
CD2 recovery at N16
format statistics
N8-pair conflict statistics
merge-regret statistics
accuracy-side recommendation
```

A separate hardware team may combine these with its own hardware estimates later.

---

## 71B.3 Frozen format/scale semantics

Weight tensor:

\[
W\in\mathbb R^{N\times K}.
\]

Candidate formats:

```text
E2M1
project E0
```

using the exact frozen Research-3/5 candidate codebooks and legal quantization semantics.

Scale granularity remains:

```text
K16
```

for both N8 and N16.

Do not change scale storage or legal scale semantics.

---

## 71B.4 Exact meaning of N16×K64

Research abstraction:

```text
N8×K64:
  one E2/E0 format decision controls 8×64 weights

N16×K64:
  one E2/E0 format decision controls 16×64 weights
```

Equivalently:

```text
one N16×K64 region
=
two adjacent N8×K64 row regions
forced to share one format decision
```

The K16 scale states remain independent exactly as in the frozen method.

Under the current row-wise abstraction, one N16×K64 region contains:

```text
16 rows × 4 K16 groups per row
= 64 K16 scale groups
```

but only:

```text
1 shared E2/E0 format decision
```

for that N16×K64 region.

Do not reinterpret this as an MMA-instruction definition.

It is the quantization/control-granularity abstraction being evaluated.

---

## 71B.5 Frozen CD2-N16 algorithm

For one N16 output stripe and K64 region \(k\):

\[
E_k^f
=
X_{K_k}
\left(
Q_f(W_{N16,K_k})-W_{N16,K_k}
\right)^T,
\]

for:

\[
f\in\{E2,E0\}.
\]

Objective:

\[
J(F)
=
\left\|
\sum_kE_k^{f_k}
\right\|_F^2.
\]

Use exactly:

```text
1. matched OutputAware/local initialization
2. coordinate sweep
3. accept a format flip iff full N16-stripe objective decreases
4. complete exactly 2 sweeps
5. serialize only final legal format map + existing K16 scales
```

Do not optimize N16 with any Research-5 A/B/C method during the core granularity study.

The purpose is a controlled N8-vs-N16 test of the existing coupled principle.

---

## 71B.6 W4A16 is the primary N8-vs-N16 study

Primary activation:

```text
A16 / high-precision activation
```

This isolates:

```text
weight format-control granularity
```

from activation-quantization noise.

Do not begin the N16 decision from W4A4.

---

## 71B.7 Models and corpora

Development-family set:

```text
Llama-3.1-8B
Qwen3-8B
Mistral-7B-v0.3
```

Evaluation:

```text
WikiText-2
C4 fixed evaluation slice
```

Use the exact same model revisions and evaluation manifests as the matched existing N8 rows
where possible.

Mistral is mandatory because it already defines an important known cross-corpus boundary.

No Mistral-specific N16 tuning.

---

## 71B.8 Required W4A16 matrix

At minimum:

| Method | Format granularity | Role |
|---|---|---|
| High Precision | — | reference |
| Fixed-E2 / NVFP4-style reference | fixed E2 | standard reference |
| Canonical-4Over6 | fixed E2 | strong scale baseline |
| Fine16 | 1×K16 format decision | fine diagnostic |
| Raw-N8K64 | N8×K64 | existing coarse baseline |
| CD2-N8K64 | N8×K64 | existing coupled baseline |
| **Raw-N16K64** | **N16×K64** | new coarse baseline |
| **CD2-N16K64** | **N16×K64** | new target |

Optional if already cheap/supported:

```text
LocalOutputAware-N8K64
LocalOutputAware-N16K64
```

Do not rerun valid N8 rows if exact hashes/manifests match.

---

## 71B.9 Primary matched NLL quantities

### Raw granularity cost

\[
\Delta_{\mathrm{RawGran}}
=
NLL(Raw_{N16})-NLL(Raw_{N8}).
\]

### CD2 granularity cost

\[
\Delta_{\mathrm{CD2Gran}}
=
NLL(CD2_{N16})-NLL(CD2_{N8}).
\]

### CD2 gain at N8

\[
G_8
=
NLL(Raw_{N8})-NLL(CD2_{N8}).
\]

### CD2 gain at N16

\[
G_{16}
=
NLL(Raw_{N16})-NLL(CD2_{N16}).
\]

### Change in coupled recoverability

\[
\Delta G=G_{16}-G_8.
\]

Interpretation:

```text
ΔG > 0:
  CD2 becomes more valuable under coarser N control

ΔG ≈ 0:
  CD2 benefit survives at similar magnitude

ΔG < 0:
  N16 introduces decision conflicts that the frozen CD2 cannot recover as effectively
```

Always report absolute paired ΔNLL; do not summarize this only with PPL.

---

## 71B.10 Fine-to-coarse recovery

For \(g\in\{N8,N16\}\):

\[
Gap_g
=
NLL(Raw_g)-NLL(Fine16),
\]

and when \(Gap_g>0\):

\[
Recovery_g
=
\frac{
NLL(Raw_g)-NLL(CD2_g)
}{
NLL(Raw_g)-NLL(Fine16)
}.
\]

Rules:

```text
Fine16 is not a PPL oracle
tiny denominator -> recovery ratio unstable
absolute paired NLL is primary
```

Question:

> Does CD2 recover the extra decision-compression loss introduced by N16?

---

## 71B.11 Pre-freeze N16 non-inferiority margin

Before inspecting final N16 evaluation outcomes, freeze a practical NLL non-inferiority margin.

Preferred rule:

```text
reuse an already-frozen Research-5 meaningful-regression/non-inferiority threshold
```

if one exists and is appropriate.

Otherwise derive a margin from:

```text
repeat/no-op numerical variability
+
a predeclared practical tolerance
```

and save:

```text
13b_n16_granularity/00_spec/n16_noninferiority_margin.json
```

The Coding Agent must not choose the margin after seeing N16 PPL/NLL.

---

## 71B.12 N8-pair format-conflict diagnostic

Every N16 region contains two adjacent N8 child regions:

```text
N8-A
N8-B
```

Using the frozen N8 solution, classify each paired K64 region:

```text
agreement:
  E2/E2
  E0/E0

conflict:
  E2/E0
  E0/E2
```

Record:

```text
pair_format_agreement_rate
pair_format_conflict_rate
```

by:

```text
model
layer
module type
```

This directly measures how often N16 removes an N8 format degree of freedom.

---

## 71B.13 Merge-regret diagnostics

Use two explicitly named diagnostics.

### A. Local pair merge regret

For each paired N8 K64 region, under the same local output-error diagnostic:

\[
J_{\mathrm{pair,free}}
=
\min_{f_A}J_A(f_A)+\min_{f_B}J_B(f_B),
\]

while with a shared format:

\[
J_{\mathrm{pair,shared}}
=
\min_{f\in\{E2,E0\}}
\left[J_A(f)+J_B(f)\right].
\]

Define:

\[
R_{\mathrm{merge,local}}
=
J_{\mathrm{pair,shared}}-J_{\mathrm{pair,free}}
\ge0
\]

up to numerical tolerance.

Report:

```text
mean
median
p90
p99
module/depth summaries
```

This is a local constraint-regret diagnostic, not an end-to-end oracle.

### B. Frozen-CD stripe merge penalty

For two adjacent frozen N8 CD2 stripes \(A,B\):

\[
J_{8,\mathrm{CD2}}
=
J_A(F_A^{CD2})+J_B(F_B^{CD2}).
\]

For the frozen two-sweep N16 CD2 solution:

\[
J_{16,\mathrm{CD2}}
=
J_{A+B}(F_{shared}^{CD2}).
\]

Define:

\[
R_{\mathrm{merge,CD2}}
=
J_{16,\mathrm{CD2}}-J_{8,\mathrm{CD2}}.
\]

Because both maps are bounded CD solutions, this is a **heuristic-solution penalty**, not an
exact optimality gap.

Do not call CD2 a mathematical oracle.

---

## 71B.14 Conflict severity

For each N8 child define a format-preference margin under the relevant local diagnostic:

\[
m=J(E0)-J(E2).
\]

For an N8 pair:

```text
agreement:
  child signs agree

low-conflict:
  child signs disagree, but preference magnitudes are small

high-conflict:
  child signs disagree and both strongly prefer opposite formats
```

Freeze the low/high margin threshold using calibration statistics before model-quality
correlation analysis.

Report:

```text
conflict rate
strong-conflict rate
local merge regret by conflict class
CD2 merge penalty by conflict class
```

Test:

> Is N16 accuracy loss concentrated in high-conflict N8 pairs?

Do not assume the answer.

---

## 71B.15 Module-wise analysis

For each:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
other Linear
```

report N8 vs N16:

```text
E0 fraction
format flip count/rate
CD objective improvement
N8-pair conflict rate
strong-conflict rate
local merge regret
frozen-CD merge penalty
layer-output error
```

Question:

> Is N16 globally safe but concentrated-risk in specific module families/depths?

Do not introduce module-specific N8/N16 routing in this experiment.

That would be a new method and is out of scope.

---

## 71B.16 Calibration protocol

Use the frozen CD2 calibration protocol.

Do not retune calibration size for N16.

Match:

```text
data source
sequence IDs
sequence length
number of sequences
random seeds
layer-input semantics
```

to N8.

Any unavoidable mismatch must be logged and prevents a strict matched claim.

---

## 71B.17 Paired NLL statistics

For every N8-vs-N16 comparison store:

```text
sequence_id
token_count
nll_raw_n8
nll_cd2_n8
nll_raw_n16
nll_cd2_n16
```

Paired bootstrap CIs:

```text
Raw-N16 - Raw-N8
CD2-N16 - CD2-N8
CD2-N8 - Raw-N8
CD2-N16 - Raw-N16
```

Report:

```text
mean ΔNLL
median ΔNLL
95% paired bootstrap CI
win fraction
PPL secondary
```

PPL point estimates alone are insufficient.

---

## 71B.18 Accuracy-side result categories

Use both diagnostic and final handoff labels.

Diagnostic interpretation:

```text
ACCURACY_SAFE
ACCURACY_TRADEOFF_BUT_RECOVERABLE
CD2_RECOVERY_WEAKENS
MODEL_DEPENDENT
UNACCEPTABLE_ACCURACY_LOSS
```

Final hardware-handoff accuracy recommendation:

```text
PREFER_N16_ACCURACY_SAFE
N16_VIABLE_WITH_SMALL_ACCURACY_TRADEOFF
KEEP_N8_FOR_ACCURACY
N16_MODEL_DEPENDENT
INSUFFICIENT_EVIDENCE
```

### ACCURACY_SAFE

Use only if:

```text
CD2-N16 is practically close to CD2-N8
paired CI satisfies the frozen non-inferiority criterion
no new cross-corpus sign pathology
```

### ACCURACY_TRADEOFF_BUT_RECOVERABLE

Use if:

```text
Raw-N16 materially degrades vs Raw-N8
CD2-N16 recovers most of the added loss
final CD2-N16 remains practically close to CD2-N8
```

### CD2_RECOVERY_WEAKENS

Use if:

```text
G16 is materially below G8
```

or merge conflicts visibly defeat the frozen two-sweep recovery.

### MODEL_DEPENDENT

Use if safety differs materially across model/corpus settings.

Do not average this away.

### UNACCEPTABLE_ACCURACY_LOSS

Use if N16 creates consistent meaningful degradation that frozen CD2 cannot recover.

---

## 71B.19 Secondary Fixed-E2M1 W4A4 survival

After W4A16 is complete, run:

```text
A4 = Fixed-E2M1 / NVFP4-compatible
```

with:

```text
Raw-N8K64
CD2-N8K64
Raw-N16K64
CD2-N16K64
```

on:

```text
Llama
Qwen
Mistral

Wiki
C4
```

Strong fixed-weight baselines may be reused if already supported.

Question:

> Does the N8→N16 accuracy conclusion survive activation quantization noise?

This is secondary evidence.

Do not use it to redesign N16.

---

## 71B.20 Interaction with the separate Activation-4Over6 W4A4 extension

Do **not** create the full Cartesian product:

```text
N8/N16
×
Fixed-E2/Activation-4Over6
×
all weight baselines
```

by default.

Order:

```text
First:
  W4A16 N8-vs-N16 viability

Then:
  Fixed-E2M1 A4 N8-vs-N16 survival

Only if N16 remains accuracy-viable:
  optionally add limited Activation-4Over6 N16 rows
```

If unlocked, the minimum useful A1 rows are:

```text
Raw-N16K64 + Activation-4Over6
CD2-N16K64 + Activation-4Over6
```

with matching N8 rows already provided by the v1.1 W4A4 factorial.

Do not add every historical weight baseline at N16 under A1 unless a later paper-layout need
is explicitly justified.

---

## 71B.21 Final Research-5 method portability to N16

The core N16 experiment is about frozen CD2.

Do not automatically port the final Research-5 method to N16.

A final Research-5 method may receive an N16 portability test only if:

```text
its semantics can be changed from N8 to N16 by the same single granularity substitution
AND
no threshold/predictor/objective is retuned for N16
AND
the N16 CD2 core study is accuracy-viable or scientifically compelling
```

If Research-5 uses N8-specific applicability/conflict features that do not transfer without
new training, do not force an N16 version into this extension.

Record:

```text
r5_n16_portability_status
```

as:

```text
NOT_APPLICABLE
LOCKED
TESTED
```

---

## 71B.22 N16 correctness tests

Before full evaluation:

```text
1. N16 grouping covers exact expected W[N,K] coordinates.
2. Two adjacent N8 regions map exactly to one N16 region.
3. Tail N dimensions are deterministic.
4. K16 scale grouping is unchanged.
5. E2/E0 candidate reconstruction matches frozen semantics.
6. Raw-N16 selector matches exact aggregated local objective.
7. CD2-N16 flip delta equals full objective recomputation.
8. Every accepted CD2-N16 flip is objective-nonincreasing.
9. Save/reload N16 map is bit-identical.
10. N8 implementation reproduces frozen prior artifacts first.
11. N8/N16 evaluation manifests are identical for matched comparisons.
```

Do not run the full matrix while these tests fail.

---

## 71B.23 Paper-ready tables

### W4A16 — format granularity

| Model | Corpus | Fine16† | Raw N8 | CD2 N8 | Raw N16 | CD2 N16 |
|---|---|---:|---:|---:|---:|---:|
| Llama | Wiki | | | | | |
| | C4 | | | | | |
| Qwen | Wiki | | | | | |
| | C4 | | | | | |
| Mistral | Wiki | | | | | |
| | C4 | | | | | |

Footnote:

```text
† Fine16 is a fine-grained local format-selection reference, not a PPL oracle.
```

### Granularity analysis

| Model | Corpus | Raw ΔNLL N16−N8 | CD2 ΔNLL N16−N8 | G8 | G16 | ΔG |
|---|---|---:|---:|---:|---:|---:|
| Llama | Wiki | | | | | |
| | C4 | | | | | |
| Qwen | Wiki | | | | | |
| | C4 | | | | | |
| Mistral | Wiki | | | | | |
| | C4 | | | | | |

---

## 71B.24 Recommended figures

Figure A — accuracy/granularity ladder:

```text
Fine16
  ↓
Raw N8
  ↓ CD2 recovery
CD2 N8

Fine16
  ↓
Raw N16
  ↓ CD2 recovery
CD2 N16
```

Plot paired ΔNLL, not only PPL.

Figure B — mechanism:

```text
N8-pair conflict / merge regret
vs
N16 degradation
```

Use module/model aggregation and do not treat every region as an independent model-level
sample.

---

## 71B.25 N16 decision does not redefine Research-5 paper success

The N16 extension is complete when the accuracy boundary is measured.

A result of:

```text
KEEP_N8_FOR_ACCURACY
```

does **not** invalidate an otherwise successful N8 Research-5 method.

A result of:

```text
PREFER_N16_ACCURACY_SAFE
```

does **not** establish hardware superiority because hardware overhead is out of scope.

The N16 result is handed to the hardware team as an accuracy-side Pareto input.


# 72. Phase 6 — W4A4 completion and activation-policy factorial

Research-5 remains primarily a **weight-side coarse-format** project.

However, paper completeness now requires a bounded W4A4 extension because the existing
evidence has two reviewer-visible gaps:

```text
1. Mistral-7B-v0.3 is missing from the existing W4A4 table.
2. Existing W4A4 uses only one Fixed-E2M1 / NVFP4-compatible activation policy.
```

The extension must answer:

> **Does weight-side CD2 / Research-5 remain useful under a stronger activation quantizer,
> and is Activation-4Over6 merely a generic A4 improvement or specifically complementary
> to the weight-side coupled policy?**

This phase is a controlled:

```text
Weight policy
×
Activation policy
```

factorial.

It is **not** permission to start a new activation-quantization research branch.

---

## 72.1 Preserve the historical W4A4 interpretation

Existing Research-3 W4A4 rows mean:

```text
Weight:
  method under test

Activation:
  Fixed-E2M1
  NVFP4-compatible
  no E2/E0 activation format selector
  no activation CD2
```

Thus:

```text
W = CD2-Static
A = Fixed-E2M1 A4
```

must not be rewritten as:

```text
W = CD2
A = CD2
```

or:

```text
W = CD2
A = Activation-4Over6
```

The original question was:

> Does the **weight-side** CD2 benefit survive activation quantization noise?

Keep that history explicit.

---

## 72.2 W4A4 execution/freeze position

W4A4 must **not** influence:

```text
A/B/C direction promotion
Phase-2 method selection
prospective-family method freeze
```

The clean default order is:

```text
final Research-5 method frozen
prospective family A passes
independent confirmation passes
then
full W4A4 factorial
```

A historical-baseline-only W4A4 completion job may be run earlier if spare resources are
available, but:

```text
Mistral W4A4 results
Activation-4Over6 results
```

must not be used to alter the Research-5 method.

If early W4A4 results are inspected, label them:

```text
paper_completeness_diagnostic_only
```

and preserve the previously frozen Research-5 method unchanged.

---

## 72.3 Activation policy A0 — Fixed-E2M1

Create an exact semantics note containing:

```text
activation payload format
activation codebook
scale datatype
scale granularity
tensor/global scale if any
dynamic/static scale computation
rounding
clipping
zero
tensor axes
excluded modules
```

Paper-safe name:

```text
Fixed-E2M1 / NVFP4-compatible A4
```

Do not call it canonical NVFP4 unless exact equivalence is validated.

---

## 72.4 Activation policy A1 — Activation-4Over6

Add exactly one bounded second activation policy:

```text
Activation-4Over6 A4
```

For each activation K16 block:

```text
payload format = E2M1
candidate scale target = 4 or 6
selection objective = local activation reconstruction SSE
tie -> target6
```

Required invariants:

```text
activation remains 4-bit
payload remains E2M1
scale grouping remains K16
no E0 activation branch
no activation CD
no weight-policy-dependent activation rule
same activation rule across model families
```

If a reference/canonical activation-side FourOverSix implementation exists and semantics
match, validate against it.

Otherwise use the paper-safe label:

```text
Activation-4Over6 using project 4/6 semantics
```

Do not assume a weight-side FourOverSix path is automatically a canonical activation path.

---

## 72.5 Mandatory A4-4Over6 correctness tests

Before PPL/NLL evaluation, verify on synthetic and sampled real activations:

```text
candidate target4 reconstruction
candidate target6 reconstruction
argmin selection
deterministic ties
exact K16 grouping
forced target6 reduces to the intended Fixed-E2 target6 path
save/reload or deterministic replay reproduces output
tail behavior
zero/clipping behavior
```

If semantics fail:

```text
W4A4_EXTENSION_SEMANTIC_FAIL
```

Stop this extension.

Do not affect the already-frozen W4A16 Research-5 conclusion.

---

## 72.6 Core historical weight-policy set

The core paper-completeness factorial uses:

```text
W0 = Fixed-E2 weight
W1 = Raw-N8K64-Static
W2 = Canonical-4Over6 weight
W3 = CD2-Static
```

Activation:

```text
A0 = Fixed-E2M1 A4
A1 = Activation-4Over6 A4
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

Core matrix:

| Weight policy | A0 Fixed-E2 A4 | A1 4Over6 A4 |
|---|---:|---:|
| Fixed-E2 W | ✓ | ✓ |
| Raw-N8K64 W | ✓ | ✓ |
| 4Over6-W | ✓ | ✓ |
| CD2-Static W | ✓ | ✓ |

Total core historical matrix:

```text
4 × 2 × 3 × 2 = 48 rows
```

Existing Llama/Qwen A0 rows may be reused **only if**:

```text
code hash
config hash
activation semantics hash
weight map hash
sequence manifest hash
```

match exactly.

---

## 72.7 Mistral W4A4 completion is mandatory

Under the existing A0 Fixed-E2 activation policy, add all Mistral rows:

```text
Mistral Wiki:
  Fixed-E2 W
  Raw-N8K64 W
  4Over6-W
  CD2-Static W

Mistral C4:
  same four weights
```

No:

```text
Mistral-specific weight rule
Mistral-specific A4 rule
Wiki-specific rescue
new CD sweeps
new scale tuning
```

Label this:

```text
post-hoc third-family W4A4 completion
```

not new prospective evidence.

---

## 72.8 Research-5 final weight method under both A4 policies

If the final Research-5 weight method is distinct from W0–W3, add:

```text
W4 = Research-5 frozen final weight policy
```

under:

```text
A0 Fixed-E2 A4
A1 Activation-4Over6 A4
```

for all three development families and both corpora.

This adds up to:

```text
1 × 2 × 3 × 2 = 12 rows
```

and is mandatory for `TOP_TIER_LLM_CANDIDATE_GO`.

Do not change the Research-5 method from these results.

---

## 72.9 Activation-only diagnostic

Strongly recommended and required for the complete W4A4 paper package unless the harness
cannot support it.

Run:

```text
W16 (or closest supported high-precision weight)
×
A0 Fixed-E2 A4

W16
×
A1 Activation-4Over6 A4
```

on all three development families and both corpora when feasible.

Purpose:

> measure the pure activation-side quality gain without weight quantization error.

If full-model W16/A4 is technically unsupported, use a controlled layer/module proxy and
record the limitation.

Do not use this diagnostic to tune A1.

---

## 72.10 CD2 × activation interaction

For each model/corpus compute the clean 2×2:

| | A0 Fixed-E2 | A1 4Over6 |
|---|---:|---:|
| W Raw | \(NLL(Raw,A0)\) | \(NLL(Raw,A1)\) |
| W CD2 | \(NLL(CD2,A0)\) | \(NLL(CD2,A1)\) |

Define:

\[
G_{CD2}^{A0}
=
NLL(Raw,A0)-NLL(CD2,A0)
\]

\[
G_{CD2}^{A1}
=
NLL(Raw,A1)-NLL(CD2,A1)
\]

and difference-in-differences:

\[
I_{CD2\times A4}
=
G_{CD2}^{A1}-G_{CD2}^{A0}.
\]

Interpretation:

```text
I > 0:
  stronger A4 increases CD2's relative benefit

I ≈ 0:
  A4-4Over6 is mainly a generic/independent activation gain

I < 0:
  stronger A4 weakens CD2's relative benefit
```

Do not claim complementarity from absolute PPL alone.

---

## 72.11 Research-5 final-method × activation interaction

If a distinct Research-5 final method exists, compute the same:

\[
I_{R5\times A4}
=
[
NLL(Raw,A1)-NLL(R5,A1)
]
-
[
NLL(Raw,A0)-NLL(R5,A0)
].
\]

This determines whether the final reliability/applicability/loss-aware policy is robust to
activation-error regime.

This result is characterization, not a method-selection signal.

---

## 72.12 4Over6-W × 4Over6-A control

Retain the conventional factorial:

```text
Fixed-E2 W / A0
4Over6-W / A0
Fixed-E2 W / A1
4Over6-W / A1
```

Question:

> Are weight- and activation-side FourOverSix gains approximately independent/additive?

If:

```text
4Over6-W + Activation-4Over6
```

dominates CD2/Research-5 combinations, report it directly.

Do not hide a stronger conventional baseline.

---

## 72.13 Mandatory paired NLL statistics

For every important matched comparison store:

```text
sequence_id
token_count
nll_a
nll_b
delta_nll
```

Required paired comparisons:

```text
A1 vs A0 at fixed weight policy
CD2 vs Raw at fixed activation policy
CD2 vs 4Over6-W at fixed activation policy
Research-5 vs Raw at fixed activation policy
Research-5 vs CD2 at fixed activation policy
```

Report:

```text
mean ΔNLL
median ΔNLL
95% paired bootstrap CI
win fraction
PPL secondary
```

---

## 72.14 Activation-4Over6 selector statistics

Record by:

```text
model
corpus
layer
module
```

at minimum:

```text
target4_fraction
target6_fraction
selection_entropy
mean_candidate_SSE_gap
p50_margin
p90_margin
p99_margin
activation_max
activation_RMS
sequence_to_sequence_selection_stability
selector_calls
```

Question:

> Is A1 meaningfully adaptive, or does it nearly collapse to one target?

If target4 usage is negligible, report that the online selector provides little effective
adaptation.

---

## 72.15 Online/deployment work accounting

Activation selection is dynamic and must not be treated as a free offline improvement.

Compare A0 vs A1:

```text
fake-quant wall time
selector wall time
candidate quantization count
extra reductions
temporary memory
selector calls per token/block
stored metadata
runtime metadata
```

Without native supported hardware, report only:

```text
algorithmic online work
fake/reference timing
operation-count proxy
```

Do not claim native latency/throughput.

A1 is not automatically the preferred deployment policy merely because it lowers NLL.

---

## 72.16 Optional activation robustness

Unlock only if A1 shows meaningful quality signal.

Use fixed:

```text
different sequence batches
different C4 shards
different sequence lengths if cheap
```

Do not tune the 4/6 rule.

Purpose:

> characterize activation selector stability under input variation.

---

## 72.17 W4A4 interpretation labels

Use bounded labels:

```text
W4A4_CD2_ROBUST
W4A4_CD2_MODEL_DEPENDENT
RESEARCH5_W4A4_ROBUST
RESEARCH5_W4A4_MODEL_DEPENDENT
A4_4OVER6_GENERIC_GAIN
A4_4OVER6_COMPLEMENTARY_TO_CD2
A4_4OVER6_COMPLEMENTARY_TO_RESEARCH5
A4_4OVER6_NO_SIGNAL
A4_4OVER6_TOO_COSTLY_FOR_GAIN
MISTRAL_W4A4_BOUNDARY_PERSISTS
```

Multiple labels may apply.

---

## 72.18 What not to add

Do not add:

```text
activation E2/E0 mixed-format CD
activation N8K64/M16K64 coupled selector
target {3,4,5,6} grids
activation Fisher selector
learned activation controller
activation Hadamard/rotation
new weight scale policy
Research-4 DualScale rescue
Mistral-specific A4 tuning
```

This phase exists for controlled paper completeness.


# 73. Phase 7 — SANA cross-domain gate
# Not mandatory for an LLM-only top-tier paper, but required for any cross-domain claim

Run SANA last.

Do not use diffusion results to tune the LLM method.

Stage:

```text
frozen proxy
→ if non-negative/interesting
128-image fixed screen
→ 1024 only if non-dominated and a cross-domain claim is desired
```

Metrics:

```text
proxy MSE/NMSE
LPIPS
PSNR
SSIM
CLIP
ImageReward
```

Research-3 already showed distortion/semantic divergence is possible.

If SANA fails:

```text
remove cross-domain claim
```

Do not automatically kill an otherwise strong LLM paper.

---

# 74. Phase 8 — external baseline audit

For paper readiness, audit reproducible official/author implementations for the closest
current methods.

Priority conceptual baselines:

```text
IF4 / Adaptive Block-Scaled Data Types
MixFP4
BlockDialect
AdaMX
```

and current strong FP4/NVFP4 PTQ baselines where semantics fit.

Only run an external method if:

```text
official/author code or exact reproducible implementation exists
model/eval is supported
bit/scale/metadata semantics are understood
comparison can be labeled fairly
```

If semantics differ, report:

```text
format granularity
scale granularity
metadata
weight/activation scope
native/runtime assumptions
```

Do not invent an approximate external baseline and label it official.

---

# 75. Phase 9 — deployment/offline cost accounting

For:

```text
Raw
CD2
Research-5
```

record:

```text
calibration sequences
forward passes
backward passes
teacher passes
wall time
GPU time
peak VRAM
peak host RAM
activation cache
gradient/Fisher cache
teacher-logit cache
map-generation time
stored map bytes
stored scale bytes
additional metadata bits
```

For A/B final static policies:

```text
predictor/routing is offline
compiled map must require no runtime controller
```

For C:

```text
model-aware calibration may cost backward/teacher passes
must be reported
```

---

# 76. Runtime representation gate

Research-5 main method must preserve:

```text
one format decision per N8K64 region
K16 scales
no dynamic online policy
no extra GEMM
no runtime permutation
no online transform
```

A/B module policy decisions must compile into the existing static map.

If an approach requires runtime module-policy bits beyond existing quantized representation:

```text
classify separately
```

and it is not the main Research-5 method unless hardware cost is explicitly justified.

---

# 77. Native SM120

Native status remains:

```text
WAIT_FOR_SM120
```

A6000/6000 Ada fake/reference results cannot establish:

```text
native E0 semantics
native decode
Tensor Core support
latency
throughput
power
```

Top-tier algorithmic LLM paper readiness does not strictly require native SM120 if the paper
claim is clearly algorithmic/reference and runtime representation is honestly characterized.

Any Blackwell-native performance claim requires real supported hardware.

---

# 78. Statistics — model quality

Primary:

```text
paired per-sequence/per-token NLL
```

For every important comparison:

```text
mean ΔNLL
median ΔNLL
95% paired bootstrap CI
win fraction
PPL
```

CIs after web/tool-independent fixed seed.

Do not use PPL point estimates alone for gates.

---

# 79. Statistics — intervention prediction

For A/C:

```text
AUROC
AUPRC
Spearman
Pearson
risk–coverage
calibration curve where applicable
```

Use:

```text
leave-one-model-out
```

as primary generalization evaluation.

For small sample uncertainty:

```text
bootstrap across modules within held-out model
```

but never claim module count equals independent model-family count.

Report model-wise metrics first.

---

# 80. Multiple-testing / robustness reporting

Module intervention labels are numerous.

Primary labels use paired CIs/effect threshold.

Additionally report a conservative sensitivity analysis using:

```text
Benjamini-Hochberg FDR on module-level paired tests
```

where a valid paired test is available.

Do not make the main method depend on one multiple-testing correction.

The purpose is to show conclusions are not driven by many false-positive module labels.

---

# 81. Phase-1 sample-size expansion rules

Initial:

```text
48 modules/model
128 diagnostic sequences/distribution
```

Expansion allowed if:

```text
>60% modules remain UNCERTAIN
or
held-out AUROC CI is too wide to distinguish signal from chance
```

Maximum before Plan-level redesign:

```text
~96 modules/model
256 sequences/distribution
```

Expansion must preserve original strata and be determined mechanically.

Do not expand only favorable modules.

---

# 82. Caching/reproducibility

Cache with hashes:

```text
FP calibration activations
Raw quantized weights
CD2 maps
module outputs
per-sequence baseline logits
HighPrecision teacher logits
gradients
Fisher statistics
interaction summaries
```

Record:

```text
dtype
shape
precision
source model hash
sequence manifest hash
code hash
```

Do not independently recompute A/B/C tensors if one shared artifact suffices.

---

# 83. Generic intervention runner

Implement one reusable interface conceptually like:

```python
run_module_intervention(
    model,
    base_policy,
    module_name,
    intervention_policy,
    sequence_manifest,
)
```

It must support:

```text
Raw -> CD2
CD2 -> Raw
Static -> Research4 DualScale
DualScale -> Static
later frozen Research-5 policies
```

No three separate evaluation pipelines.

---

# 84. Artifact tree

Use:

```text
artifacts/research_5/
```

Required:

```text
00_environment/
  spec_acknowledgement.md
  spec_manifest.json
  prior_artifact_manifest.csv
  prior_hashes.json
  repo_manifest.json
  patch_manifest.json
  environment.txt
  gpu_usage_log.jsonl
  literature_snapshot.md

01_reproduction/
  research3_reproduction.csv
  research4_reproduction.csv
  baseline_equivalence.json
  intervention_noop_tests.json

02_prospective_seal/
  primary_family.json
  secondary_family.json
  seal_log.json

03_diagnostic_dataset/
  module_sample_manifest.csv
  stripe_sample_manifest.csv
  calibration_split_manifest.json
  diagnostic_c4_manifest.json
  diagnostic_wiki_manifest.json

  raw_to_cd_insert.jsonl
  cd_to_raw_fallback.jsonl
  r4_static_to_dual_insert.jsonl
  r4_dual_to_static_fallback.jsonl

  intervention_statistics.csv
  intervention_labels.csv
  intervention_fdr_sensitivity.csv

  reconstruction_features.parquet
  coupling_features.parquet
  map_search_features.parquet
  robustness_features.parquet
  cross_distribution_features.parquet
  activation_features.parquet
  model_aware_features.parquet
  merged_module_dataset.parquet

04_direction_a/
  shard_consensus.csv
  cross_distribution_consistency.csv
  confidence_scores.csv
  harmful_detection.csv
  risk_coverage.csv
  leave_one_model_out.csv
  r4_contrastive_analysis.csv
  direction_a_decision.md
  direction_a_decision.json

05_direction_b/
  oracle_proxy.csv
  cumulative_coverage.csv
  fallback_ablation.csv
  predictor_configs/
  leave_one_model_out.csv
  headroom_capture.csv
  direction_b_decision.md
  direction_b_decision.json

06_direction_c/
  coupled_sse_scores.csv
  first_order_scores.csv
  fisher_scores.csv
  taylor_scores.csv
  teacher_kl_scores.csv
  surrogate_correlations.csv
  harmful_detection.csv
  leave_one_model_out.csv
  r4_contrastive_analysis.csv
  cost.csv
  direction_c_decision.md
  direction_c_decision.json

07_phase1_gate/
  abc_scorecard.csv
  promoted_directions.json
  intervention_measurement_report.md
  diagnostic_phase_report.md
  phase1_decision.json

08_phase2_methods/
  method_configs/
  calibration_results/
  method_costs.csv
  development_eval/
  paired_nll.csv
  method_gate.csv

09_method_freeze/
  frozen_method.json
  frozen_method_hashes.json
  freeze_report.md

10_prospective_family_a/
  unseal_log.json
  eval_results.csv
  paired_nll.csv
  decision.md
  decision.json

11_prospective_family_b/
  unseal_log.json
  eval_results.csv
  paired_nll.csv
  decision.md
  decision.json

12_scale_model/
  preregistration.md
  eval_results.csv
  calibration_cost.csv

13_downstream/
  task_manifest.json
  results.csv
  summary.md

13b_n16_granularity/
  00_spec/
    experiment_spec.md
    frozen_semantics.json
    source_artifact_hashes.json
    n16_noninferiority_margin.json

  01_correctness/
    n16_grouping_tests.json
    candidate_reconstruction_tests.json
    raw_n16_selector_tests.json
    cd2_objective_tests.json
    n8_reproduction_check.json

  02_w4a16/
    llama_wiki.csv
    llama_c4.csv
    qwen_wiki.csv
    qwen_c4.csv
    mistral_wiki.csv
    mistral_c4.csv
    per_sequence_nll.csv
    paired_bootstrap.json

  03_merge_diagnostics/
    n8_pair_agreement.csv
    local_merge_regret.csv
    frozen_cd_merge_penalty.csv
    conflict_severity.csv
    module_summary.csv

  04_w4a4_fixed_e2/
    llama_wiki.csv
    llama_c4.csv
    qwen_wiki.csv
    qwen_c4.csv
    mistral_wiki.csv
    mistral_c4.csv
    per_sequence_nll.csv
    paired_bootstrap.json

  05_optional_a4_4over6/
    unlock_decision.md
    raw_n16.csv
    cd2_n16.csv
    paired_nll.csv

  06_optional_research5_portability/
    portability_decision.md
    results.csv
    paired_nll.csv

  07_final/
    master_table.csv
    granularity_comparison.csv
    recovery_analysis.csv
    n16_accuracy_decision.json
    n16_accuracy_decision.md
    limitations.md
    hardware_accuracy_handoff.md

14_w4a4/
  00_semantics/
    a4_fixed_e2_semantics.md
    a4_4over6_semantics.md
    a4_4over6_unit_tests.json
    reduction_tests.json
    config_hashes.json

  01_existing_reproduction/
    llama_fixed_e2_a4.csv
    qwen_fixed_e2_a4.csv
    reproduction_report.md

  02_mistral_fixed_e2_a4/
    wiki.csv
    c4.csv
    paired_nll.csv
    bootstrap.json

  03_all_models_a4_4over6/
    llama_wiki.csv
    llama_c4.csv
    qwen_wiki.csv
    qwen_c4.csv
    mistral_wiki.csv
    mistral_c4.csv
    paired_nll.csv

  04_research5_final_method/
    fixed_e2_a4.csv
    a4_4over6.csv
    paired_nll.csv

  05_activation_only/
    w16_a4_fixed_e2.csv
    w16_a4_4over6.csv
    paired_nll.csv

  06_factorial_analysis/
    weight_x_activation_matrix.csv
    cd2_activation_interaction.csv
    research5_activation_interaction.csv
    four_over_six_interaction.csv
    paired_bootstrap.json

  07_activation_stats/
    selection_rate_by_layer.csv
    margin_stats.csv
    activation_distribution_stats.csv
    stability.csv

  08_cost/
    online_work_accounting.md
    fake_quant_timing.csv
    memory.csv

  09_final/
    w4a4_master_table.csv
    final_w4a4_report.md
    limitations.md
    paper_table_fixed_e2.csv
    paper_table_a4_4over6.csv
    interaction_table.csv

15_sana/
  proxy/
  images_128/
  images_1024/
  decision.md

16_external_baselines/
  baseline_audit.md
  configs/
  results.csv

17_deployment/
  offline_cost.csv
  metadata_accounting.md
  runtime_semantics.md
  native_sm120_handoff.md

18_final/
  experiment_manifest.jsonl
  failed_runs.jsonl
  master_results.csv
  intervention_master.csv
  abc_scorecard.csv
  generalization_matrix.csv
  final_decision.json
  final_decision_report.md
  results_summary.md
  limitations.md
  paper_claims_boundary.md
  reproduction_commands.sh
```

---

# 85. Required machine-readable experiment fields

Common:

```text
experiment_id
research_phase
direction
model_id
model_revision
corpus
sequence_manifest_hash
calibration_manifest_hash
calibration_role
module_name
module_type
layer_index
normalized_layer_depth

base_policy
intervention_policy
format_granularity
scale_granularity

objective_type
surrogate_type
confidence_rule
applicability_rule

git_commit
code_hash
config_hash
artifact_hash

gpu_physical_id
gpu_uuid
gpu_type
start_time
end_time
status
```

---

# 86. Intervention fields

```text
delta_nll_insert
delta_nll_insert_ci_low
delta_nll_insert_ci_high
delta_nll_fallback
delta_nll_fallback_ci_low
delta_nll_fallback_ci_high

insert_win_fraction
fallback_win_fraction
num_sequences
num_tokens

beneficial_insert
harmful_insert
beneficial_fallback
harmful_fallback
context_consistent_label
epsilon_used
```

---

# 87. Feature fields

Reconstruction/coupling:

```text
cd2_objective_gain
normalized_cd2_gain
cross_interaction_C
offdiag_coupling_ratio
negative_coupling_fraction
interaction_spectral_norm
```

Reliability:

```text
shard_mean
shard_std
shard_min
sign_consensus
train_val_gap
cross_domain_min_gain
cross_domain_sign_consensus
Q_shift
map_hamming_domain_shift
```

Model-aware:

```text
first_order_score
fisher_score
taylor_score
delta_teacher_kl
```

---

# 88. Predictor/method fields

```text
feature_set
threshold
coverage_target
risk_target
model_class
regularization
tree_depth
training_models
validation_model
fallback_policy
```

Prospective:

```text
method_freeze_timestamp
prospective_unseal_timestamp
prospective_family_was_accessed_pre_freeze
```

---


## 88.0 N8/N16 granularity machine-readable fields

For every N8/N16 row:

```text
weight_format_granularity
n_rows_per_format_region
k_values_per_format_region
scale_granularity

raw_or_cd2
cd_sweep_count
calibration_manifest_hash
eval_manifest_hash

nll
ppl
paired_reference
```

Matched granularity quantities:

```text
raw_n16_minus_n8_delta_nll
cd2_n16_minus_n8_delta_nll
g8
g16
delta_g
fine_gap_n8
fine_gap_n16
recovery_n8
recovery_n16
```

Pair-conflict/merge fields:

```text
n16_region_id
n8_child_a_format
n8_child_b_format
pair_format_conflict
child_a_margin
child_b_margin
conflict_class

local_free_objective
local_shared_objective
local_merge_regret

n8_cd2_objective_sum
n16_cd2_objective
frozen_cd_merge_penalty
```

Final status:

```text
n16_diagnostic_status
n16_accuracy_recommendation
n16_noninferiority_margin
r5_n16_portability_status
hardware_overhead_evaluated = false
```


## 88.1 W4A4 machine-readable fields

For every W4A4 row:

```text
weight_policy
weight_format_policy
weight_scale_policy

activation_policy
activation_format
activation_scale_policy
activation_scale_granularity
activation_selector_rule

a4_semantics_hash
weight_map_hash
eval_manifest_hash

ppl
mean_nll
paired_reference
```

For Activation-4Over6:

```text
target4_fraction
target6_fraction
selection_entropy
mean_selection_margin
p50_selection_margin
p90_selection_margin
p99_selection_margin
selector_calls
selector_wall_seconds
candidate_quantization_count
extra_reduction_count
temporary_memory_bytes
```

For factorial analysis:

```text
cd2_gain_fixed_a4
cd2_gain_4over6_a4
cd2_activation_interaction

research5_gain_fixed_a4
research5_gain_4over6_a4
research5_activation_interaction

four_over_six_weight_activation_interaction
```


# 89. GPU policy

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum project GPUs:

```text
3
```

Before every GPU launch:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi pmon -c 1
```

For visible PID:

```bash
ps -o user=,pid=,cmd= -p <PID>
```

Rules:

```text
occupied by another user => unavailable
unknown occupied => unavailable
low utilization != free
never share another user's GPU
never kill another user's process
never broad pkill
use reservation lock
recheck before launch
log every admission/rejection
```

Because intervention experiments create many jobs:

```text
use deterministic queue/scheduler
```

rather than uncontrolled launch loops.

---

# 90. Git policy

Create isolated Research-5 worktree.

Do not:

```text
overwrite Research-1~4
push unless explicitly requested
rewrite another user's branch
```

Use local commits at:

```text
baseline reproduction
intervention infrastructure
Phase-1 dataset
A/B/C scorecard
Phase-2 method freeze
prospective results
final aggregation
```

---

# 91. Failure handling

Every failure/aborted run:

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
dataset/network
model_access
dependency
GPU_became_occupied
intervention_equivalence_failure
numerical_instability
gradient_failure
teacher_logit_failure
calibration_leakage
prospective_leakage
evaluation_leakage
artifact_mismatch
runtime_excessive
unknown
```

Never silently remove failures.

---

# 92. Mandatory tests

## Baseline

```text
Raw equivalence
CD2 equivalence
4Over6 equivalence
DualScale contrastive reproduction
```

## Intervention

```text
no-op intervention exactly matches base
swap one module only
restore module returns base
module state hash correct
per-sequence IDs preserved
```

## Gradient/Fisher

```text
gradient hook tensor matches module output shape
first-order synthetic finite-difference sanity
Fisher non-negative
Taylor score deterministic
```

## Teacher KL

```text
HP teacher logits fixed
KL zero/equivalence sanity
base-to-base delta KL == 0
```

## LossAware-CD if unlocked

```text
quadratic reconstruction exact
single flip score exact
monotone approximate objective
save/reload map exact
```

---

# 93. Research-4 contrastive sanity gate

Before Phase-2 promotion, the scorecard must explicitly answer:

```text
Which A/B/C evidence would have warned us that Research-4 DualScale was misleading?
```

Possible acceptable outcomes:

```text
A:
  cross-domain / confidence signal flags harmful DualScale modules

B:
  harmful DualScale/CD changes are concentrated and fallback has headroom

C:
  model-aware score predicts harm where SSE predicts gain
```

If none of A/B/C explains any meaningful fraction of the confirmed Research-4 contrastive
failures:

```text
SURROGATE_RESTART_WEAK_GROUNDING
```

and the bar for continuing Phase 2 becomes substantially higher.

---

# 94. Final decision vocabulary

Use one main algorithmic status:

```text
INTERVENTION_MEASUREMENT_INSUFFICIENT
COARSE_FORMAT_RESTART_NO_GO
PHASE1_DIAGNOSTIC_GO
METHOD_DEVELOPMENT_NO_GO
METHOD_GO
PROSPECTIVE_GENERALIZATION_FAIL
CONDITIONAL_PAPER_GO
PROSPECTIVE_STRONG_GO
TOP_TIER_LLM_CANDIDATE_GO
CROSS_DOMAIN_GO
```

Native side status separately:

```text
WAIT_FOR_SM120
NATIVE_VALIDATED
```

N8/N16 accuracy-side status separately:

```text
PREFER_N16_ACCURACY_SAFE
N16_VIABLE_WITH_SMALL_ACCURACY_TRADEOFF
KEEP_N8_FOR_ACCURACY
N16_MODEL_DEPENDENT
INSUFFICIENT_EVIDENCE
```

Do not mix this accuracy recommendation with unverified hardware-overhead conclusions.

---

# 95. TOP_TIER_LLM_CANDIDATE_GO gate

Do not issue this unless all are true:

```text
1. at least one A/B/C mechanism has strong intervention evidence;
2. final method passes Llama/Qwen/Mistral development matrix;
3. prospective family A passes;
4. second independent confirmation passes:
     prospective family B
     or strong predeclared independent scale/architecture case;
5. proper downstream suite is non-regressive/competitive;
6. W4A4 paper-completeness factorial is complete:
     - Mistral included under existing Fixed-E2 A4;
     - Activation-4Over6 semantics validated;
     - CD2 × A4 interaction characterized;
     - final Research-5 method tested under both A4 policies;
     - online activation-selector work/cost reported;
     - final method remains non-regressive under at least the fixed A4 deployment baseline;
7. N8→N16 accuracy boundary is complete:
     - matched W4A16 Raw/CD2 N8-vs-N16 table exists;
     - paired NLL + non-inferiority decision exists;
     - N8-pair conflict / merge-regret mechanism analysis exists;
     - Fixed-E2M1 W4A4 survival is characterized;
     - hardware overhead remains explicitly out of scope;
     - N16 is NOT required to win;
8. closest reproducible baselines are audited and compared fairly;
9. offline cost and static runtime representation are explicit;
10. no final-eval/model-family-specific tuning;
11. paper claim is distinct from generic mixed format / layer sensitivity / residual compensation.
```

SANA is not mandatory for `TOP_TIER_LLM_CANDIDATE_GO`.

SANA is required only for:

```text
CROSS_DOMAIN_GO
```

---

# 96. Potential paper stories after evidence

## If A wins

> Reconstruction-improving coarse-format changes have heterogeneous reliability;
> calibration-derived stability can identify harmful changes, enabling static safe
> acceptance with unchanged runtime representation.

Need real harmful-intervention prediction.

---

## If B wins

> Coupled assignment is conditionally useful; static offline applicability identifies
> where to use CD and where to fallback while retaining the same N8K64 representation.

Must distinguish from generic layer sensitivity/mixed precision.

---

## If C wins

> Coupled reconstruction SSE misranks coarse-format effects because error direction and
> model-loss sensitivity matter. A lightweight model-aware surrogate better predicts and
> optimizes the discrete coarse-format map.

Must show:

```text
predictiveness
→ method
→ prospective generalization
```

---

# 97. Exact execution order

```text
0. RE-READ/hash CURRENT Research-5 v1.2 + CURRENT prompt
1. write v1.2 acknowledgement + in-flight reconciliation
2. reuse completed-compatible v1.0/v1.1 artifacts; do not blindly rerun
3. create/verify isolated worktree
4. audit/hash Research-1~4
5. regenerate Research-3/4 key tables
6. reproduce Raw/CD2/4Over6/DualScale checks
7. implement and validate generic intervention runner

8. select + seal prospective family A
9. select + seal prospective family B if feasible

10. pre-register development module sample
11. create diagnostic C4/Wiki manifests
12. create calibration feature manifests
13. run no-op/noise-floor checks

14. run Raw→CD insertion interventions
15. run CD→Raw fallback interventions
16. deterministically expand uncertain labels if needed

17. run Research-4 Static→Dual module contrastive interventions
18. run Dual→Static fallbacks

19. freeze shared reconstruction/coupling/map/robustness features
20. compute cross-distribution reliability features
21. compute intervention labels/CIs/FDR sensitivity

22. Direction A screening
23. Direction B screening
24. Direction C screening

25. generate diagnostic_phase_report.md
26. issue A/B/C scorecard
27. if 0 directions pass -> FINAL NO_GO
28. promote <=2 directions

29. develop one bounded method per promoted direction
30. independently validate each method
31. if two pass, optionally test one predeclared combination
32. cap full candidates <=3

33. run full development Llama/Qwen/Mistral Wiki+C4
34. issue METHOD_GO
35. if no candidate passes -> STOP

36. freeze exactly one final Research-5 method
37. hash/freeze/timestamp

38. unseal prospective family A
39. run >=2 predeclared corpora
40. issue prospective-A gate
41. if fail -> FINAL STOP

42. unseal prospective family B / confirmation case
43. run same frozen method
44. issue confirmation gate

45. if strong:
      run ~12–14B scale case

46. run proper downstream suite

47. N8→N16 accuracy/granularity extension:
      re-read/freeze N16 sub-spec + non-inferiority margin
      reproduce exact matched N8 row(s)
      implement N16 grouping ONLY
      run N16 grouping/candidate/raw/CD2 correctness tests
      implement Raw-N16
      adapt frozen CD2 to N16 with exactly two sweeps
      run Llama W4A16 Wiki+C4
      run Qwen W4A16 Wiki+C4
      run Mistral W4A16 Wiki+C4
      compute paired NLL / G8 / G16 / ΔG / matched recovery
      compute N8-pair conflict
      compute local merge regret
      compute frozen-CD merge penalty
      compute conflict severity + module summaries
      issue N16 W4A16 accuracy decision
      run Fixed-E2M1 W4A4 N8-vs-N16 survival
      if N16 remains accuracy-viable:
        optionally run limited Raw-N16/CD2-N16 + Activation-4Over6
      do NOT estimate hardware overhead
      write hardware_accuracy_handoff.md

48. W4A4 paper-completeness extension:
      audit A0 Fixed-E2 semantics
      implement/audit A1 Activation-4Over6
      run A1 unit/reduction tests
      freeze A1 before final W4A4 outcomes
      reproduce/reuse exact Llama/Qwen A0 rows
      complete/reuse Mistral A0 rows
      run 4×2×3×2 historical factorial
      run frozen Research-5 method under A0/A1
      run W16/high-precision activation-only diagnostic
      compute CD2×A4 and R5×A4 interactions
      compute 4Over6-W×4Over6-A control
      collect selector statistics
      measure online/fake-quant cost
      no method retuning from W4A4

49. audit/run closest official external baselines
50. quantify calibration/deployment cost

51. if cross-domain claim desired:
      run frozen SANA proxy
      if justified, 128 images
      1024 only if non-dominated

52. regenerate all final artifacts
53. issue strict paper-readiness decision
54. prepare SM120 handoff only
```

Important in-flight rule:

```text
if the Coding Agent has already completed valid v1.0/v1.1 steps,
continue from the first missing/incompatible v1.2 requirement;
do not restart experimentally.
```

The only mandatory immediate action after receiving v1.2 is:

```text
read the NEW research_5.md fully
read the NEW coding_agent_prompt_5.md fully
hash both
reconcile current work
update manifests/plan
```

Do not interrupt a valid running A/B/C job solely because N16 was added.


# 98. Explicit STOP rules

Stop Research-5 if:

```text
intervention effects cannot be measured reliably
A/B/C all fail
B oracle has no headroom
C surrogates do not beat SSE
A confidence only memorizes model/corpus
Phase-2 candidates regress known development families
prospective family A fails
final method requires family-specific tuning
runtime representation premise changes
external baseline audit reveals the contribution is already dominated under matched semantics
```

Do not invent a new direction after a STOP rule.

---

# 99. Final scientific framing

Research-1~4 should be viewed as a completed first program that discovered:

```text
coarse decision compression
cross-K interaction
coupled assignment headroom
and a repeated surrogate-to-model mismatch
```

Research-5 asks a deeper and more falsifiable question:

\[
\boxed{
\text{Can we identify which coarse-format quantization changes actually matter to model loss?}
}
\]

The most important Research-5 product is not necessarily a new quantizer.

It is a scientifically grounded bridge:

```text
local/coupled quantization change
→ intervention ground truth
→ reliability/applicability/model-aware prediction
→ static deployment-safe method
→ prospective generalization
```

If that bridge cannot be demonstrated, the coarse-format direction should be closed.

If it can, and the frozen method survives unseen-family, downstream, and deployment gates,
the work has a plausible path to a top-tier LLM systems/ML quantization paper.
