# research_1.md
# Research Plan v1.2 — AUTHORITATIVE REWRITE
## Coarse-Granularity E2M1/E0M3 Quantization with Scale-, Selector-, and Layout-Aware Recovery

**SPEC_VERSION:** `research_1_v1.2_2026-08-08`  
**Status:** authoritative replacement of all earlier `research_1.md` / `research_1_revised.md` variants  
**Primary domains:** LLM + text-to-image diffusion  
**Primary GPUs:** NVIDIA RTX A6000 (GPU 0–3), NVIDIA RTX 6000 Ada (GPU 4–6)  
**SM120/Blackwell required for this phase:** **No**  
**Future native validation:** RTX 5090 / SM120 only after algorithmic Go/No-Go

Primary implementation anchors:

- NVFP4-RaZeR
- FourOverSix
- DeepCompressor
- SANA
- Nunchaku only as deployment/reference context

---

# 0. Document authority and continuation contract

This document **replaces and overwrites the scientific specification of the previous Research-1 plan**.

For the next coding/research iteration:

```text
CURRENT research_1.md
    >
all earlier research_1*.md variants
    >
older coding-agent prompts
```

Earlier files remain provenance only. Their instructions must not be merged back unless this file explicitly preserves them.

The coding agent must:

1. re-read this whole file from disk;
2. compute and record its SHA256;
3. audit prior artifacts under the definitions in this version;
4. reuse valid immutable results where semantics are unchanged;
5. rerun results whose semantics changed;
6. never rely on cached understanding of the previous Research-1 file.

Use:

```text
PRIOR_ARTIFACT_ROOT = artifacts/
ARTIFACT_ROOT       = artifacts/research_1/
```

Do not overwrite immutable prior experiment attempts.

Each prior item must be classified as one of:

```text
REUSE_CONTEXT_ONLY
REUSE_NUMERIC_RESULT
REGRESSION_CHECK_THEN_REUSE
SUPERSEDED_BY_RESEARCH_1_V12
LEGACY_ABLATION_ONLY
RERUN_REQUIRED
```

---

# 1. Research objective

We want to determine whether useful fine-grained adaptive E2M1/E0M3 format choices can be compressed into coarse hardware-compatible format decisions without losing most model quality.

Core invariants:

```text
weight format proxy     = N8 x K64
activation format proxy = M16 x K64
scale granularity       = K16
```

The format-selection granularity may become coarse, but the scale granularity must remain K16 in the primary study.

The study must answer:

1. Is fine K16 E2/E0 selection useful under a declared objective?
2. How much benefit is lost when format decisions become coarse?
3. Why does coarse N8K64 become strongly E0-heavy in the observed 8B results?
4. How much loss comes from format granularity itself?
5. How much loss comes from using local weight MSE as the selector?
6. Can K16 scale freedom recover part of the loss?
7. Can selector, packing, or rotation recover the remaining loss?
8. Do the conclusions hold across Llama, Qwen, and SANA?
9. Is there enough evidence to justify later SM120 native work?

---

# 2. Existing evidence that should be preserved

The prior artifact bundle contains valid preliminary W4A16 WikiText-2 evidence.

## Llama-3.1-8B

```text
high precision = 5.651335
NVFP4          = 6.027484
MSE-Oracle16   = 5.965058
K64-row        = 6.027838
N8K16          = 6.008820
N8K64          = 6.064335
canonical 4/6  = 6.013704
```

## Qwen3-8B

```text
high precision = 8.529555
NVFP4          = 8.688240
MSE-Oracle16   = 8.548895
K64-row        = 8.654694
N8K16          = 8.807764
N8K64          = 8.749336
canonical 4/6  = 8.615497
```

Preliminary interpretation:

```text
fine E2/E0 selection has signal
but raw N8K64 loses or reverses much of the benefit
```

Observed E0 ratios:

```text
MSE-Oracle16:
  Llama ~0.621
  Qwen  ~0.605

MSE-N8K64:
  Llama ~0.914
  Qwen  ~0.871
```

Therefore the observed coarse collapse is currently **E0-heavy**.

This is a phenomenon to explain, not evidence that E0M3 itself is inherently bad.

---

# 3. Prior artifact integrity rule

Some old narrative summaries were generated before later immutable experiment rows existed.

Evidence precedence:

```text
immutable attempt artifacts/configs/logs
    >
regenerated experiment manifest/aggregation
    >
prior master-results table
    >
old decision_report / go_no_go / limitations prose
```

Before reusing prior results, regenerate a consistency table and record:

```text
prior_completed_attempt_count
prior_master_row_count
prior_manifest_completed_count
stale_summary_files_detected
```

Never inherit an old Go/No-Go field without regeneration.

---

# 4. Terminology

## 4.1 NVFP4

Canonical all-E2M1 reference using the pinned repository semantics.

Positive E2M1 magnitudes should match:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

with sign.

## 4.2 Project E0M3 / INT4-like

Use the project sign-magnitude codebook:

```text
0, +/-1, +/-2, ..., +/-7
```

Do not use two's-complement `-8`.

Until native SM120 evidence exists, this is:

```text
PROJECT_DEFINED_FAKE_QUANT_SEMANTICS
```

not verified native Blackwell E0M3 semantics.

## 4.3 `oracle16` naming correction

Keep the internal machine-readable id:

```text
oracle16
```

for backward compatibility.

For research-facing text use:

```text
display_mode_name   = MSE-Oracle16
selector_objective  = weight_mse
```

Definition:

> MSE-Oracle16 is the finest K16 format-selection reference under local weight-reconstruction MSE.

It is **not**:

- PPL-optimal;
- downstream-accuracy-optimal;
- layer-output-optimal;
- a true model-quality oracle.

---

# 5. Controlled format-granularity experiment

For K16 block `b`:

\[
e_b^{E2}=\|W_b-Q_{E2}(W_b)\|_F^2,
\]

\[
e_b^{E0}=\|W_b-Q_{E0}(W_b)\|_F^2.
\]

Define:

\[
D_b=e_b^{E2}-e_b^{E0}.
\]

Interpretation:

```text
D_b > 0 : E0 lower local MSE
D_b < 0 : E2 lower local MSE
|D_b|   : preference strength under local MSE
```

MSE-Oracle16 uses:

\[
f_b^{MSE}=\arg\min_f e_b^f.
\]

For one N8K64 region `G`:

\[
E_G^{E2}=\sum_{b\in G} e_b^{E2},
\]

\[
E_G^{E0}=\sum_{b\in G} e_b^{E0},
\]

\[
F_G^{MSE}=\arg\min_f E_G^f.
\]

There are:

\[
8\times(64/16)=32
\]

K16 decisions inside one N8K64 weight region.

The clean comparison:

```text
MSE-Oracle16 -> MSE-N8K64
```

changes only format-selection granularity.

Do not change:

- codebooks;
- K16 scale grouping;
- candidate-specific K16 scales;
- selector objective.

---

# 6. Granularity regret

Define:

\[
P_G=\sum_b\max(D_b,0),
\]

\[
N_G=\sum_b\max(-D_b,0).
\]

Then:

\[
R_G
=
E_{coarse}(G)-E_{fine}(G)
=
\min(P_G,N_G).
\]

This identity must remain unit-tested.

Report:

```text
granularity_regret
normalized_regret
margin_conflict
count_homogeneity
```

Margin conflict:

\[
C_G^{margin}
=
\frac{\min(P_G,N_G)}
{P_G+N_G+\epsilon}.
\]

---

# 7. E0-heavy coarse-format collapse

The current aggregate observation is:

```text
fine MSE selection: ~60% E0 blocks
coarse N8K64:       ~87-91% E0 regions
```

Do not call this simply “E0M3 is bad.”

The coarse selector is not majority voting. It uses:

\[
D_G=\sum_bD_b.
\]

The region selects E0 if:

\[
D_G>0.
\]

A minority of E0-preferring blocks may dominate if their positive margins are large.

Current unproven hypotheses:

```text
count bias
margin asymmetry
spatial correlation / clustering
```

The revised study must separate them.

---

# 8. Required E0-collapse diagnostics

## 8.1 Signed-margin distributions

Record:

```text
mean_positive_margin
median_positive_margin
p90_positive_margin
p99_positive_margin
max_positive_margin

mean_negative_margin_abs
median_negative_margin_abs
p90_negative_margin_abs
p99_negative_margin_abs
max_negative_margin_abs
```

## 8.2 Region-level statistics

For every relevant region:

```text
region_sum_D
region_mean_D
oracle_e0_count
oracle_e2_count
oracle_e0_fraction
positive_margin_sum
negative_margin_abs_sum
e0_margin_share
coarse_selected_format
```

where:

\[
E0MarginShare_G
=
\frac{P_G}{P_G+N_G+\epsilon}.
\]

## 8.3 Forced-format spillover

Report:

```text
coarse_e0_block_exposure_ratio
coarse_e2_block_exposure_ratio
forced_E2_to_E0_block_rate
forced_E0_to_E2_block_rate
```

Margin-weighted costs:

\[
C_{E2\rightarrow E0}
=
\sum_{\substack{b:f_b^{MSE}=E2\\F_G=E0}}(-D_b),
\]

\[
C_{E0\rightarrow E2}
=
\sum_{\substack{b:f_b^{MSE}=E0\\F_G=E2}}D_b.
\]

These are more informative than region E0 ratio alone.

## 8.4 Conditional E0 probability

Produce:

```text
P(coarse_E0 | oracle_E0_count = k)
P(coarse_E0 | oracle_E0_fraction bin)
P(coarse_E0 | e0_margin_share bin)
```

## 8.5 Spatial correlation

At minimum:

```text
same_n8_preference_agreement
same_k64_preference_agreement
adjacent_k16_preference_agreement
```

If practical, add a simple spatial autocorrelation statistic over signed margins.

## 8.6 Counterfactual shuffle diagnostics

Run bounded deterministic diagnostics:

```text
actual_D
sign_only_actual_layout
shuffled_D_within_layer_or_module
shuffled_sign_only_within_layer_or_module
```

Interpretation:

```text
actual_D:
  count + margin magnitude + spatial structure

sign_only_actual_layout:
  count + spatial structure
  margin magnitude removed

shuffled_D:
  count + signed-margin distribution
  spatial structure removed

shuffled_sign_only:
  approximate count-bias-only baseline
```

Do not claim an exact additive causal decomposition.

---

# 9. Format entropy/diversity is diagnostic only

Measure:

```text
E0 ratio
dominant-format fraction
format entropy
entropy drop
format flip rate
```

Do not optimize for high entropy.

A low-entropy assignment may be correct if the dominant format is truly better.

Use the phrase:

```text
harmful format collapse
```

only when collapse correlates with worse PPL, layer-output error, or matched granularity regret.

---

# 10. Within-format adaptive scaling — Phase B0

The coarse tile shares one format decision, but each K16 block keeps its own scale.

This gives:

> coarse format selection + fine K16 within-format scale adaptation.

## 10.1 E2 branch

Mandatory:

```text
E2-static6
E2-4over6-MSE
E2-representable-scale-oracle   # diagnostic upper bound
```

Canonical 4Over6 must reduce exactly/numerically to the pinned FourOverSix behavior when format is forced to E2.

The E2 representable-scale oracle must search only legal NV-style E4M3 block scales.

Do not use arbitrary FP32 per-block scales.

## 10.2 E0 branch

Mandatory:

```text
E0-static7
E0-scale67
E0-scale567
E0-representable-scale-oracle
```

Implement target 5/6/7 explicitly.

Do not assume a library enum called `static_6` means target maximum 6 for INT4-like E0.

The representable-scale oracle must obey the same NV-style global-scale and legal E4M3 block-scale semantics.

## 10.3 Mandatory B0 2x2 factorial

For:

```text
MSE-Oracle16
K64-row
N8K16
N4K64
N8K64
```

run:

| E2 branch | E0 branch | name |
|---|---|---|
| static6 | static7 | `dual_static` |
| 4/6 | static7 | `e2_4over6_only` |
| static6 | 6/7 | `e0_scale67_only` |
| 4/6 | 6/7 | `dual_adaptive_scale` |

## 10.4 Scale-oracle coverage

For E0:

\[
Coverage_{67}
=
\frac{E_{static7}-E_{67}}
{E_{static7}-E_{E0oracle}}
\]

when the denominator is positive.

Also report `{5,6,7}` coverage.

For E2:

\[
Coverage_{4/6}
=
\frac{E_{static6}-E_{4/6}}
{E_{static6}-E_{E2oracle}}
\]

when the denominator is positive.

## 10.5 Matching-regret rule

For scale policy `s`:

\[
R_s
=
E_{coarse,s}-E_{fine,s}.
\]

Do not compare `coarse_adaptive` against `fine_static` when claiming granularity recovery.

Define:

\[
\Delta R_{scale}
=
R_{static}-R_{adaptive}.
\]

Interpretation:

```text
coarse error improves, Delta_R_scale ~ 0:
  stronger quantizer only

Delta_R_scale > 0:
  true granularity recovery signal

Delta_R_scale < 0:
  candidate quality improved but coarse conflict worsened
```

The last case is scientifically important.

Scale adaptation may lower E0 local MSE, make more regions select E0, and worsen PPL.

## 10.6 B0 decision

Classify:

```text
true_granularity_recovery
stronger_quantizer_only
not_useful
```

Freeze one selected scale policy after the initial B0 evidence, but always retain `dual_static` as the controlled baseline.

---

# 11. Selector objective as a second axis

The existing Llama result suggests local MSE conflict does not fully predict PPL sensitivity.

Therefore selector objective must be studied separately from format granularity.

Use this conceptual matrix:

| objective | fine K16 diagnostic | coarse N8K64 |
|---|---|---|
| local weight MSE | `MSE-Oracle16` | `MSE-N8K64` |
| local/block-diagonal output-aware | `LocalOutputAware-Fine16` | `OutputAware-N8K64` |

---

# 12. Important mathematical restriction on fine output-aware selection

For a partitioned layer:

\[
\left\|
\sum_b X_b\Delta W_b^T
\right\|_F^2
\]

contains cross-block terms.

Therefore independently minimizing:

\[
\|X_b\Delta W_b^T\|_F^2
\]

for every K16 block is only a local/block-diagonal approximation.

Do not call it:

```text
Sensitivity-Oracle16
```

Use:

```text
LocalOutputAware-Fine16
```

This is a diagnostic reference, not a global output/PPL oracle.

Optional sampled upper bound:

```text
full-layer greedy/coordinate-descent format assignment
```

using full calibration layer-output loss.

Tag:

```text
upper_bound_only = true
```

---

# 13. Phase B1 — early output-aware selector diagnostic

This is promoted earlier than in the previous plan.

Run immediately after the initial B0 WikiText pilot.

## 13.1 Required first matrix

On Llama-3.1-8B and Qwen3-8B WikiText:

```text
MSE-Oracle16
LocalOutputAware-Fine16
MSE-N8K64
OutputAware-N8K64
```

Then, if B0 has signal:

```text
MSE-N8K64 + selected_B0_scale
OutputAware-N8K64 + selected_B0_scale
```

## 13.2 Coarse output-aware objective

For region `G`:

\[
F_G^{out}
=
\arg\min_f
\|X_G(W_G-Q_f(W_G))^T\|_F^2.
\]

A numerically validated Hessian/quadratic equivalent is allowed.

## 13.3 Selector-objective gap

Define:

\[
G_{selector}
=
L_{out}(F_{MSE})
-
L_{out}(F_{out}).
\]

Machine-readable name:

```text
output_aware_selector_gain
```

Do not call this strict regret unless `F_out` is proven globally optimal.

Also report:

```text
mse_granularity_regret
local_outputaware_granularity_gap
selector_disagreement
selector_objective_mismatch
```

If local MSE N/K decomposition and end-to-end/output-aware evidence disagree:

```text
selector_objective_mismatch = true
```

## 13.4 Calibration sizes

Evaluate when promoted:

```text
32
128
256
```

Use 128 as the default pilot if feasible.

Save exact calibration sample IDs.

---

# 14. Confidence-aware selector

Do not launch a large confidence sweep unless plain `OutputAware-N8K64` first shows meaningful signal.

Variants:

```text
outputaware_argmin
outputaware_threshold
outputaware_confidence
```

Define:

\[
\Delta_G^{out}
=
L_G^{out}(E2)-L_G^{out}(E0).
\]

Default uncertainty behavior:

```text
if output-aware evidence is confident:
    use output-aware choice
else:
    fall back to the MSE-N8K64 choice
```

Do **not** hardcode:

```text
uncertain -> E2
```

That may be evaluated only as a secondary ablation.

Do not tune thresholds on final evaluation data.

---

# 15. N/K decomposition

Continue measuring:

```text
MSE-Oracle16 -> K64-row
MSE-Oracle16 -> N8K16
MSE-Oracle16 -> N8K64
```

But do not infer the recovery mechanism from local MSE regret alone.

Report separately:

```text
local_regret_axis
end_to_end_axis
output_aware_axis
```

Current working hypothesis to verify, not assume:

```text
Qwen3-8B:
  likely strong N/channel-packing sensitivity

Llama-3.1-8B:
  likely stronger K and/or selector-objective sensitivity
```

Permutation vs rotation must be routed using the combined evidence.

---

# 16. Phase B2 — foldable sensitivity-aware permutation / packing

Prioritize if N-direction/output-aware conflict remains substantial.

Candidate diagnostics:

```text
no_permutation
sort_by_e0_ratio
margin_vector_clustering
greedy_min_regret_n8
sensitivity_weighted_greedy_n8
```

Preferred representation:

```text
signed-margin signature per output channel over K64 regions
```

Primary objective should emphasize:

```text
sensitivity-weighted signed-margin compatibility
```

not simple E0-ratio sorting.

For Llama/Qwen MLPs, only use foldable transformations after high-precision equivalence is proven.

Example shared intermediate permutation:

```text
gate_proj rows
up_proj rows
down_proj columns
```

Explicit inverse-output variants are upper-bound-only unless folding is proven.

Do not modify attention channel ordering without a proof of functional equivalence.

---

# 17. Phase B3 — rotation / block-aligned transform

Use only if K-direction or residual sensitivity conflict remains important after B0/B1 and, where relevant, permutation.

Candidate bank:

```text
identity
H16
H32
H64
H128
random signed H64 seeds
```

Verify high-precision equivalence first.

Use objective families such as:

\[
L(R)
=
\hat L_{output}
+
\lambda\hat L_{granularity}.
\]

Generic Hadamard wins are not deployment evidence unless cost/foldability is addressed.

Rotation is a conditional residual-recovery mechanism, not the default first solution.

---

# 18. Weight and activation policies may be asymmetric

Do not force the same policy onto both operands.

## Weight

Weights are offline and may use:

- representable-scale search;
- sensitivity calibration;
- precomputed format metadata;
- foldable permutation;
- more expensive one-time preprocessing.

## Activation

Activations are online and have different constraints:

- dynamic token/timestep distribution;
- runtime format/scale selector cost;
- calibration drift;
- selector metadata overhead;
- timestep dependence.

Final Phase C recommendation must compare:

```text
weight adaptive / activation fixed E2M1
activation adaptive / weight fixed E2M1
both adaptive
neither adaptive
```

Working hypothesis only:

```text
weights:
  coarse E2/E0 + stronger scale + output-aware selector + foldable packing

activations:
  initially NVFP4/E2M1 + 4Over6
```

W4A4 evidence must be allowed to reject this hypothesis.

---

# 19. Phase A / LLM requirements

Mandatory models:

```text
Llama-3.1-8B
Qwen3-8B
```

Smoke model may be used only for correctness.

Mandatory datasets:

```text
WikiText-2
C4 fixed slice
```

Finalist downstream tasks where feasible:

```text
ARC-Easy
HellaSwag
PIQA
WinoGrande
```

## W4A16 static granularity matrix

At minimum:

```text
high_precision
nvfp4
all_e0m3
MSE-Oracle16
k32_row
k64_row
n2k64
n4k64
n8k16
n8k64
n16k64
n32k64
n64k64
layer
canonical_nvfp4_4over6
```

Do not repeat already-valid prior rows unnecessarily.

## W4A4

Required one-side/two-side logic:

```text
W fine / A fine
W fine / A E2
W E2   / A fine
W N8K64 / A E2
W E2 / A M16K64
W N8K64 / A M16K64
```

Promote only selected B0/B1 policies to activation-side combinations.

Avoid uncontrolled Cartesian sweeps.

---

# 20. Diffusion / SANA requirements

Mandatory primary diffusion target:

```text
SANA-1.6B
```

Use fixed prompts, seeds, scheduler, steps, CFG/PAG, resolution, model revision, VAE revision, and text-encoder revision.

## W4A16

Run:

```text
high precision
NVFP4 reference
MSE-Oracle16-like weight reference
raw N8K64
B0 selected scaling
B1 selected output-aware selector if promoted
```

Collect proxy metrics first:

```text
denoiser/flow output MSE
NMSE
relative L2
cosine
latent trajectory error
per-layer output error
```

Then image screening:

```text
LPIPS
PSNR
ImageReward
optional SSIM / CLIPScore
```

Finalists should use a larger fixed evaluation set, e.g. MJHQ/1024 images where feasible.

## Timestep stability

Track:

```text
E0 ratio
signed-margin correlation
selector agreement
granularity regret
output-aware loss
B0 scale-choice ratio
```

across representative early/mid/late timesteps.

Only add timestep-bucketed selectors if instability is material.

---

# 21. Canonical 4Over6 and legacy composition

Canonical baseline:

```text
NVFP4 + canonical FourOverSix
```

must remain separate.

The old mixed-format composition used a symmetric `1.5x` E0 extension.

Preserve old rows only under:

```text
legacy_symmetric_1p5x_e0_extension
```

Do not reinterpret them as:

```text
E0-scale67
dual_adaptive_scale
```

For any new composition, forcing format to E2 must reproduce canonical FourOverSix within tolerance.

Never claim published FourOverSix supports the project E0 branch unless the source explicitly does.

---

# 22. GPU operating rules

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum project GPUs concurrently:

```text
3
```

Never share a physical GPU with another user.

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

- unknown owner + occupied => unavailable;
- low utilization != free;
- never kill another user's process;
- never use broad `pkill`;
- re-check immediately before exec;
- use project-side reservation locks;
- log every selection/rejection.

Critical continuation rule:

> Do not reuse an SM120-only architecture gate as the admission gate for Research-1.

A6000/SM86 and RTX 6000 Ada/SM89 are the intended Phase A/B devices.

If SM120 is unavailable:

```text
native_status = WAIT_FOR_SM120
```

and continue Research-1.

---

# 23. Unit and semantic tests

Before large runs require:

1. E2M1 codebook.
2. E0M3 codebook.
3. K16 scale grouping.
4. true `[N,K]` N8K64 mapping.
5. true `[N,K]` N8K16 mapping.
6. true `[M,K]` M16K64 mapping.
7. tail handling.
8. deterministic seeds.
9. `R_G = min(P_G,N_G)`.
10. MSE fine <= matched MSE coarse.
11. E0 target7 reproduces static E0.
12. target5/6/7 are numerically distinct.
13. `{6,7}` equals blockwise min of its two candidates.
14. `{5,6,7}` equals blockwise min of three candidates.
15. E0 scale oracle is no worse than `{5,6,7}`.
16. E2 scale oracle is no worse than 4/6.
17. representable-scale oracles use only legal scale values.
18. canonical forced-E2 4/6 reduction.
19. spillover accounting.
20. counterfactual shuffle determinism.
21. output-aware selector objective.
22. `LocalOutputAware-Fine16` is labeled non-oracle.
23. confidence fallback returns to MSE choice by default.
24. high-precision permutation equivalence.
25. high-precision rotation equivalence.
26. diffusion fixed-prompt/seed determinism.

Do not launch the full matrix while core invariants fail.

---

# 24. Required machine-readable fields

Where appropriate include:

```text
experiment_id
model
dataset_or_promptset
quant_mode
display_mode_name
selector_objective
selector_confidence_rule
selector_threshold
scale_policy
format_region
scale_group_size
operand_role
gpu_physical_id
gpu_logical_id
repo_sha
config_hash

MSE
NMSE
relative_l2
cosine
PPL

e0_ratio
dominant_format_fraction
format_entropy
entropy_drop

mean_positive_margin
median_positive_margin
p90_positive_margin
p99_positive_margin
mean_negative_margin_abs
median_negative_margin_abs
p90_negative_margin_abs
p99_negative_margin_abs

region_sum_D
region_mean_D
positive_margin_sum
negative_margin_abs_sum
e0_margin_share
oracle_e0_fraction
coarse_selected_format

coarse_e0_block_exposure_ratio
forced_E2_to_E0_block_rate
forced_E0_to_E2_block_rate
spillover_cost_E2_to_E0
spillover_cost_E0_to_E2

mse_granularity_regret
local_outputaware_granularity_gap
output_aware_selector_gain
selector_objective_mismatch

same_n8_preference_agreement
same_k64_preference_agreement
adjacent_k16_preference_agreement
counterfactual_mode

Delta_R_scale
relative_regret_recovery
scale67_oracle_coverage
scale567_oracle_coverage
scale46_oracle_coverage
```

Use region-level CSV/Parquet for large region quantities and aggregate summaries for final reporting.

---

# 25. Oracle-gain retention validity

For lower-is-better metric `M`:

\[
oracle\_gain=M_{NVFP4}-M_{MSEOracle16}.
\]

Only report:

\[
Retention(G)
=
\frac{M_{NVFP4}-M_G}
{M_{NVFP4}-M_{MSEOracle16}}
\]

when:

```text
oracle_gain > eps
```

Otherwise:

```text
oracle_gain_retention = NA
retention_reason = ORACLE_NOT_BETTER_THAN_NVFP4
```

Do not divide by zero or a negative oracle-gain denominator.

---

# 26. Artifact structure

All new results:

```text
artifacts/research_1/
```

Required:

```text
00_environment/
  spec_acknowledgement.md
  spec_manifest.json
  environment.txt
  repo_manifest.json
  patch_manifest.json
  gpu_usage_log.jsonl
  prior_evidence_manifest.csv
  prior_handoff_audit.md

01_repo_audit/
  nvfp4_razer.md
  fouroversix.md
  deepcompressor.md
  sana.md
  four_over_six_composition.md

02_tests/
  unit_tests.txt
  codebook_tests.json
  granularity_tests.json
  scale_oracle_tests.json
  four_over_six_reduction.json
  outputaware_tests.json
  spillover_tests.json
  counterfactual_tests.json

03_phase_a/
  llm/
  diffusion/

04_phase_b/
  scale_adaptation/
  collapse_diagnostics/
  outputaware_selector/
  permutation/
  rotation/
  combined/

05_cross_gpu/

06_final/
  master_results.csv
  llm_results.csv
  diffusion_results.csv
  per_layer_metrics.csv
  format_region_metrics.csv
  timestep_metrics.csv
  experiment_manifest.jsonl
  failed_runs.jsonl
  results_summary.md
  decision_report.md
  go_no_go.json
  phase_c_handoff.md
  limitations.md
  reproduction_commands.sh
```

Terminal output is not the source of truth.

---

# 27. Required plots

At minimum:

```text
ppl_vs_granularity
oracle_gain_retention_vs_granularity
regret_vs_granularity
N_vs_K_conflict
local_regret_axis_vs_PPL_axis

positive_vs_negative_margin_distribution
region_sum_D_histogram
coarse_E0_probability_vs_oracle_E0_count
coarse_E0_probability_vs_E0_margin_share
forced_E2_to_E0_rate_by_layer
E0_preference_spatial_heatmap
actual_vs_signonly_vs_shuffle_collapse

e0_ratio_vs_granularity_by_scale_policy
format_entropy_vs_granularity
delta_R_scale_vs_granularity
E0_scale67_vs_scale_oracle
E2_4over6_vs_scale_oracle

MSE_vs_outputaware_selector
selector_flip_rate
outputaware_gain_vs_PPL
confidence_stability_if_promoted

permutation_variants
rotation_variants
combined_finalists

SANA_proxy_vs_granularity
SANA_image_metrics
SANA_timestep_stability

A6000_vs_6000Ada
```

Every plot must have a machine-readable source table.

---

# 28. Diagnostic-first execution order

Do not start with a huge sweep.

Follow:

```text
0. re-read the current overwritten research_1.md
1. record research_1.md SHA256 and write spec acknowledgement
2. create isolated continuation worktree/branch
3. audit/hash prior artifacts
4. use A6000/Ada-compatible GPU guard
5. reuse old invariants only when code/config fingerprints match
6. implement/test E0 target5/6/7 + E0 representable-scale oracle
7. implement/test E2 representable-scale oracle + canonical E2 4/6 reduction
8. regression-check NVFP4 / MSE-Oracle16 / canonical 4/6
9. run B0 2x2 WikiText pilot on both 8B models
10. run E0-collapse spillover + margin + conditional + shuffle diagnostics
11. run early B1:
      MSE-Oracle16
      LocalOutputAware-Fine16
      MSE-N8K64
      OutputAware-N8K64
12. repeat coarse B1 pair with selected B0 scale if B0 has signal
13. only if output-aware has signal, evaluate confidence variants
14. complete C4 validation using frozen pilot policies
15. complete SANA W4A16 screening using frozen pilot policies
16. freeze scale + selector policy
17. finish necessary W4A16 granularity matrix
18. run LLM W4A4 asymmetric weight/activation study
19. run SANA W4A4 selected study
20. run permutation if N-direction/output-aware evidence remains
21. run rotation only if K-direction/residual evidence remains
22. run SANA timestep study
23. combine justified finalists
24. run downstream LLM + larger SANA finalist evaluation
25. optional secondary diffusion model only after mandatory work
26. cross-repo and matched A6000/Ada validation
27. regenerate all aggregate tables from immutable attempts
28. strict final Go/No-Go
29. Phase C handoff
```

---

# 29. Go/No-Go logic

## Gate A — fine MSE adaptation

Does MSE-Oracle16 meaningfully improve NVFP4 on multiple meaningful settings?

If not, the E2/E0 mixed-format direction is weak.

## Gate B — raw coarse loss

Does N8K64/M16K64 lose a meaningful fraction of the fine benefit?

If not, complex recovery is unnecessary.

## Gate C — collapse mechanism

Can the E0-heavy collapse be explained by measured:

```text
count bias
margin asymmetry
spatial correlation
```

without unsupported speculation?

## Gate D — B0 value

Classify B0:

```text
true_granularity_recovery
stronger_quantizer_only
not_useful
```

## Gate E — selector objective

Does OutputAware-N8K64 materially outperform MSE-N8K64 under held-out evaluation?

If yes, selector objective is a major recovery axis.

If no, do not keep expanding selector complexity.

## Gate F — layout/transform recovery

Do foldable permutation and/or credible rotation recover remaining loss?

Upper-bound-only wins are not sufficient for a deployable claim.

## Gate G — generalization

Does the phenomenon and selected recovery hold across:

```text
Llama-3.1-8B
Qwen3-8B
SANA-1.6B
```

## Strong Go

Strong Go requires:

- useful fine adaptation in multiple settings;
- a real coarse-granularity problem or a clear coarse-deployment advantage;
- a recovery mechanism that is not merely evaluation cherry-picking;
- evidence that the final method remains useful against canonical 4Over6 and stronger scale baselines;
- deployable/foldable logic for the claimed components;
- cross-model/domain stability.

## Conditional Go

Use when the phenomenon is real but the deployable recovery mechanism or cross-domain evidence is incomplete.

## No-Go

Use when:

- fine adaptation is weak/inconsistent;
- coarse loss destroys the benefit and cannot be recovered;
- only non-deployable upper bounds win;
- stronger scale/selector baselines remove the need for the mixed-format method;
- results are fragile or model-specific.

Do not lower the paper standard merely to continue the project.

---

# 30. Phase C handoff

Phase C is future SM120/RTX5090 work.

Do not claim native speedup from A6000/Ada fake/reference quantization.

The handoff must specify:

```text
best weight format rule
best activation format rule
scale semantics
format granularity
selector objective
confidence rule
permutation/rotation
metadata
reference PPL/image metrics
numerical tolerances
native experiments still required
```

Native status may remain:

```text
WAIT_FOR_SM120
```

without blocking Research-1 completion.

---

# 31. Final research framing

Do not frame the work as:

> “We propose 6Over7.”

Do not frame MSE-Oracle16 as a model-quality oracle.

Do not assume E0-heavy collapse is inherently harmful.

The preferred research question is:

> When realistic hardware granularity forces many K16 scale blocks to share one E2M1/E0M3 format decision, can scale optimization, better selection objectives, and foldable layout/representation shaping compress the fine decisions into a coarse hardware decision while preserving most of the model-quality benefit?

That question should remain falsifiable throughout the study.
