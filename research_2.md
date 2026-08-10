# research_2.md
# Research-2 — Bounded Final Recovery Round for Coarse-Granularity E2M1/E0M3 Quantization

**SPEC_VERSION:** `research_2_v1.0_2026-08-09`  
**Status:** `CONDITIONAL_CONTINUE / INSUFFICIENT_EVIDENCE_FOR_FINAL_NO_GO`  
**Purpose:** close the two remaining high-information gaps from Research-1, then make a final stop/continue decision  
**Scope:** primarily weight-side W4A16 on Llama-3.1-8B and Qwen3-8B; SANA/activation/native work is gated  
**Primary GPUs:** RTX A6000 (GPU 0–3), RTX 6000 Ada (GPU 4–6)  
**SM120/Blackwell required:** **No**  
**Native status:** `WAIT_FOR_SM120`

---

# 0. Authority and relationship to Research-1

This file defines **Research-2**, a new, strictly bounded continuation after Research-1.

Research-1 is not discarded. Its immutable artifacts are the evidence base for this plan.

Use this precedence:

```text
CURRENT research_2.md
    >
CURRENT coding_agent_prompt_2.md
    >
immutable Research-1 attempts/configs/logs
    >
regenerated Research-1 final aggregation
    >
older Research-1 narrative summaries
```

Research-1's former final decision was:

```text
NO_GO
confidence = high
```

but its own Gate F recorded:

```text
full_validation_required = true
best_paired_gains = []
```

while a strong bounded foldable Qwen packing signal remained unvalidated, and the
full-layer coupled coarse-format assignment upper-bound diagnostic had not been run.

Therefore Research-2 does **not** promote the project to GO. It changes the operational status to:

```text
CONDITIONAL_CONTINUE
```

for **one bounded final recovery round only**.

If the mandatory P0 experiments fail, Research-2 must end in:

```text
FINAL_NO_GO
```

without reopening a broad method search.

---

# 1. Research-1 evidence that is already established

Research-1 is substantially complete.

Final artifact-level facts:

```text
completed experiments: 328
preserved failed attempts: 20
artifact validation: PASS
artifact validation issues: 0
incomplete gates: none
native SM120 tested: false
native status: WAIT_FOR_SM120
```

Research-1 completed substantial LLM, diffusion, W4A4, scaling, selector, permutation,
rotation, collapse-diagnostic, downstream-sanity, and cross-GPU work.

Do not repeat already-valid experiments unless required for a matched Research-2 comparison.

---

# 2. Core positive and negative findings from Research-1

## 2.1 Fine E2/E0 adaptation is real

Full primary W4A16 PPL:

| Model | Corpus | NVFP4 | MSE-Oracle16 |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.623339 | **6.559806** |
| Llama-3.1-8B | C4 | 9.557657 | **9.454540** |
| Qwen3-8B | WikiText | 9.911164 | **9.736490** |
| Qwen3-8B | C4 | 13.812140 | **13.657144** |

Thus:

```text
fine K16 E2/E0 selection has genuine model-quality signal
```

Research-2 must not restart by questioning whether the phenomenon exists.

---

## 2.2 Raw N8K64 loses the fine-format benefit

Full PPL:

| Model | Corpus | MSE-Oracle16 | MSE-N8K64 |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.559806 | 6.638539 |
| Llama-3.1-8B | C4 | 9.454540 | 9.560929 |
| Qwen3-8B | WikiText | 9.736490 | 9.919024 |
| Qwen3-8B | C4 | 13.657144 | 13.765566 |

Therefore:

```text
fine adaptation useful
+
coarse N8K64 format sharing harmful
```

is established.

The core scientific question is no longer whether a coarse problem exists.

It is:

> Is fixed N8K64 intrinsically too restrictive, or did Research-1 fail because its
> selectors/layout assignments were too local or insufficiently coupled?

---

## 2.3 E0-heavy collapse is largely diagnosed

Observed approximate exposure:

```text
MSE-Oracle16 fine E0 ratio:
  Llama ~0.621
  Qwen  ~0.605

MSE-N8K64 coarse E0 exposure:
  Llama ~0.914
  Qwen  ~0.871
```

Research-1 counterfactuals show:

- count imbalance is the dominant driver;
- signed-margin magnitude adds a smaller shift;
- actual-vs-shuffled differences are small;
- spatial clustering is not the primary explanation.

Forced E2→E0 block rates were approximately:

```text
Llama ~0.336
Qwen  ~0.329
```

Do not spend more broad effort re-diagnosing E0 collapse unless a new method explicitly depends on it.

Do not assume E0 prevalence itself is harmful.

---

## 2.4 B0 simple scale adaptation is not a validated global recovery method

Research-1 B0 classification:

```text
not_useful
```

It showed some pilot gains, especially on Llama, but no robust cross-model recovery.

Sampled legal-scale coverage:

| Model | E2 4/6 coverage | E0 6/7 coverage | E0 5/6/7 coverage |
|---|---:|---:|---:|
| Llama-3.1-8B | ~0.603 | ~0.308 | ~0.340 |
| Qwen3-8B | ~0.606 | ~0.322 | ~0.353 |

Important limitation:

```text
representable-scale oracle was sampled SSE diagnostics,
not a full-model PPL upper bound
```

Therefore:

```text
simple 4/6 + 6/7 as standalone recovery: CLOSED/FAILED
legal scale headroom: NOT FULLY CLOSED
```

---

## 2.5 Local OutputAware-N8K64 exposes selector mismatch but is not universal

Full PPL:

| Model | Corpus | MSE-N8K64 | OutputAware-N8K64 |
|---|---|---:|---:|
| Llama-3.1-8B | WikiText | 6.638539 | **6.583551** |
| Llama-3.1-8B | C4 | 9.560929 | **9.505624** |
| Qwen3-8B | WikiText | 9.919024 | **9.896762** |
| Qwen3-8B | C4 | 13.765566 | **13.817555** |

Approximate coarse-vs-fine NLL-gap recovery:

```text
Llama WikiText ~70%
Llama C4       ~52%
Qwen WikiText  ~12%
Qwen C4        negative
```

Conclusion:

```text
selector objective matters
but independent local output SSE is not a universal selector
```

This does **not** prove no better coupled selector exists.

---

## 2.6 Qwen has a strong bounded foldable N-packing signal

Research-1 bounded 8-sequence WikiText pilot:

```text
Qwen MSE-N8K64, no permutation:
  8.758292

Qwen foldable_mlp_greedy_min_regret_n8:
  8.648944
```

Approximate pilot gain:

```text
ΔPPL ~ -0.10935
```

This recovered roughly half of the pilot coarse-vs-fine NLL gap.

Mechanism-consistent local evidence:

```text
Qwen total weight modules: 252
Qwen MLP projection modules: 108
MLP modules with lower local regret after greedy packing: 104 / 108
attention modules modified: 0 / 144
```

Approximate MLP local regret sum:

```text
before ~5470.8
after  ~5322.9
```

However, this is **not full evaluation**.

Llama bounded pilot moved in the wrong direction:

```text
MSE-N8K64 baseline ~6.0633
greedy packing     ~6.0942
```

Therefore:

```text
global/universal packing = disfavored
selective/mechanism-routed packing = still unresolved
```

---

## 2.7 Generic rotation is closed for this branch

Research-1 fixed H64 full C4 check worsened both LLMs.

SANA random signed H64 was seed-fragile; only one seed jointly improved proxy and trajectory.

Therefore Research-2 must **not** run:

```text
more generic Hadamard seeds
larger generic rotation banks
random transform fishing
```

Rotation may only re-enter a future project if a new transform explicitly optimizes coarse-format conflict and has a credible foldable/deployment path.

---

## 2.8 SANA is not a clean representation failure

SANA W4A16 proxy approximately:

```text
NVFP4 proxy NMSE          ~0.1954
MSE-Oracle16              ~0.1728
MSE-N8K64                 ~0.1847
OutputAware-N8K64         ~0.1795
```

Thus the coarse mixed format still beats NVFP4 on the proxy.

On the 128-image screen, OutputAware improved LPIPS/PSNR but worsened ImageReward/CLIP.

Therefore:

```text
SANA shows a Pareto conflict, not a clean all-metric failure
```

Do not launch 1024-image SANA in Research-2 unless a weight-side Research-2 winner is frozen first.

---

# 3. Frozen semantics and terminology

## 3.1 MSE-Oracle16

Keep internal id if needed:

```text
oracle16
```

Research-facing name:

```text
MSE-Oracle16
selector_objective = weight_mse
```

Definition:

> per-K16 E2/E0 choice minimizing local weight reconstruction MSE.

It is not a PPL oracle.

---

## 3.2 Project E0 semantics

Project E0 codebook:

```text
0, +/-1, +/-2, ..., +/-7
```

sign-magnitude.

Until native evidence:

```text
semantic_status = PROJECT_DEFINED_FAKE_QUANT_SEMANTICS
native_sm120_verified = false
```

Do not claim native Blackwell E0 semantics.

---

## 3.3 Primary hardware-oriented proxy

Weights:

```text
W[N,K]
format region = N8 x K64
scale group   = K16
```

Changing format granularity must never silently change K16 scale granularity.

---

# 4. A key Research-2 decomposition: N-layout vs K-coupled assignment

Research-2 has two unresolved high-information mechanisms.

## 4.1 P0-A attacks the N direction

Foldable MLP packing changes which output/intermediate channels share N8 groups.

This tests:

> Can better channel grouping make each N8 stripe more format-compatible?

## 4.2 P0-B attacks coupled K decisions inside an N8 stripe

For a linear layer:

\[
Y=XW^T.
\]

Partition weight into N8×K64 regions \(G=(n,k)\).

A region contributes output error only to its N8 output-channel stripe:

\[
E_{n,k}^{f}
=
X_{K_k}
\left(Q_f(W_{N_n,K_k})-W_{N_n,K_k}\right)^T.
\]

### Important mathematical simplification

If two regions have different N8 output stripes \(n_1\neq n_2\), their output-error
supports are disjoint in output-channel coordinates.

Therefore their Frobenius cross term is zero.

The full layer-output objective decomposes across N8 output stripes:

\[
\|\Delta Y\|_F^2
=
\sum_n
\left\|
\sum_k E_{n,k}^{f_{n,k}}
\right\|_F^2.
\]

Thus the meaningful full-layer coupling for fixed N8 grouping is primarily:

```text
within one N8 stripe, across its K64 regions
```

This is extremely useful:

- P0-A changes the N8 stripe composition;
- P0-B jointly assigns E2/E0 across K64 regions within each stripe;
- P0-B does not need a combinatorial search across independent N8 output stripes.

This decomposition must be used in the implementation.

---

# 5. Research-2 central decision question

Research-2 asks only:

> Can the strongest existing foldable N-direction signal and/or a coupled full-layer
> coarse-format assignment recover enough of the fine-format benefit to invalidate a
> final NO-GO?

No other broad exploration is allowed before this is answered.

---

# 6. P0-A — MUST DO
# Full validation of frozen Qwen foldable greedy N8 packing

## 6.1 Question

Does the existing strong Qwen packing pilot survive matched full WikiText and C4 evaluation?

## 6.2 Freeze the method

Use the exact existing:

```text
foldable_mlp_greedy_min_regret_n8
```

implementation and calibration logic.

Do not retune based on full evaluation results.

Before running, audit:

```text
packing calibration source
packing calibration sample IDs
packing calibration hash
whether any eval sequence influenced calibration
exact permutation maps
foldability proof
high-precision equivalence
eligible MLP layer list
attention untouched
```

If evaluation leakage cannot be ruled out, regenerate the frozen packing maps using a clean,
predeclared calibration split before evaluating.

Do not choose the better of old/new maps after seeing evaluation.

## 6.3 Required main evaluations

Qwen3-8B:

```text
full WikiText evaluation
fixed C4 evaluation
```

Matched baselines on the exact same evaluation sequences:

```text
HighPrecision
NVFP4
Canonical-4Over6
MSE-Oracle16
MSE-N8K64 dual_static
foldable_mlp_greedy_min_regret_n8
```

Llama full evaluation is a useful negative/generalization control if cheap, but Qwen full Wiki+C4 is mandatory.

## 6.4 Required metrics

Use per-sequence NLL as the primary comparison variable.

Record:

```text
PPL
mean per-sequence NLL
median paired ΔNLL
paired bootstrap 95% CI
win fraction per sequence
coarse-loss NLL
packing recovery fraction in NLL
per-layer/module local regret change
packing calibration predicted gain
held-out layer-output gain
full PPL gain
high-precision equivalence
exact permutation maps
```

Define:

\[
L_{\mathrm{coarse}}
=
NLL(\mathrm{MSE\mbox{-}N8K64})
-
NLL(\mathrm{MSE\mbox{-}Oracle16}).
\]

When \(L_{\mathrm{coarse}}>0\):

\[
Recovery_P
=
\frac{
NLL(\mathrm{MSE\mbox{-}N8K64})
-
NLL(\mathrm{Packed})
}{
L_{\mathrm{coarse}}
}.
\]

## 6.5 P0-A success rule

A strong positive Qwen result should satisfy all:

```text
1. packed improves over raw MSE-N8K64 on full WikiText
2. packed improves over raw MSE-N8K64 on fixed C4
3. paired CI does not support a substantial regression
4. high-precision equivalence passes
5. no evaluation leakage
6. >= 30% coarse-loss NLL recovery on both corpora is the pre-registered design target
```

The 30% value is an internal continuation threshold, not a universal law.

Also require that absolute paired ΔNLL is nontrivial; do not promote a method only because a tiny denominator produces a large recovery percentage.

## 6.6 P0-A failure rule

If the full signal disappears, reverses on one corpus, or requires retuning on evaluation:

```text
P0_A_PACKING = FAIL
packing branch = CLOSED
```

Do not immediately replace it with learned/Sinkhorn packing.

---

# 7. P0-B — MUST DO
# Full-layer coupled coarse-format recoverability diagnostic

## 7.1 Question

Is N8K64 intrinsically unrecoverable, or did the local independent selector miss useful
cross-K64 residual cancellation within each N8 stripe?

## 7.2 Name

Use:

```text
FullLayer-Coarse-CD-N8K64
```

or an equivalent explicit name.

Do not call it a true oracle.

Metadata:

```text
upper_bound_only = true
selector_scope = full_linear_module_output
coupling_scope = same_N8_stripe_across_K64_regions
```

## 7.3 First-stage layer selection

Before a full-model run, use representative modules.

For each model:

Llama:

```text
high coarse-loss MLP module
high coarse-loss attention projection
middle-conflict control
low-conflict control
```

Qwen:

```text
high N-conflict MLP module
high coarse-loss MLP control
attention projection control
middle/low-conflict control
```

Use Research-1 per-layer metrics to choose these **before** looking at P0-B outcomes.

Persist the selected module list and reason.

## 7.4 Calibration semantics

First P0-B must use the same calibration-input semantics as the existing Research-1
OutputAware selector if possible, so the only changed factor is:

```text
independent local region decision
vs
coupled stripe/full-module output decision
```

Research-2 must audit whether prior B1 used:

```text
FP independent-layer inputs
or
quantized-prefix inputs
```

Record the answer.

Do not mix that question into the first P0-B comparison.

Sequential quantized-prefix calibration is P1, not P0.

## 7.5 Efficient residual formulation

For each N8 stripe \(n\), cache full-precision teacher output for the module/stripe.

Maintain stripe residual:

\[
R_n
=
\sum_k E_{n,k}^{f_{n,k}}.
\]

When changing one region \(G=(n,k)\) from current format \(a\) to candidate \(b\):

\[
\Delta R_G
=
X_{K_k}
\left(
Q_b(W_G)-Q_a(W_G)
\right)^T.
\]

Compare:

\[
\|R_n+\Delta R_G\|_F^2
\]

against the current stripe residual norm.

Update only if the candidate is better.

Different N8 stripes can be optimized independently.

## 7.6 Bounded search variants

Initializations:

```text
init_mse_n8k64
init_outputaware_n8k64
```

Orders:

```text
natural_K_order
descending_local_sensitivity
```

Sweeps:

```text
1 sweep
2 sweeps maximum
```

Do not create a large order/search bank.

For reproducibility, tie-breaking must be deterministic.

## 7.7 Required layer-level metrics

For each representative module/stripe:

```text
full_layer_output_sse_before
full_layer_output_sse_local_outputaware
full_layer_output_sse_cd
cd_gain_vs_mse
cd_gain_vs_local_outputaware
full_layer_cd_sweep
full_layer_cd_order
full_layer_format_flip_rate
num_regions
num_flips
convergence_after_sweep
residual_norm_before
residual_norm_after
```

Also report the fine-reference full-layer SSE under:

```text
MSE-Oracle16
LocalOutputAware-Fine16
```

with explicit non-oracle labeling.

## 7.8 Promotion from representative modules to full model

Promote P0-B to a full-model static assignment only if representative modules show a clear
and repeatable gain over local OutputAware.

Suggested promotion signal:

```text
multiple high-conflict modules show >=25% additional recoverable output-SSE gap
over local OutputAware, with no systematic control-module regressions
```

This is a design threshold, not a universal law.

If promoted:

- generate all module assignments from calibration only;
- freeze them;
- run full WikiText and C4 PPL/NLL;
- do not change assignments after evaluation.

## 7.9 P0-B full-model metrics

Same paired NLL protocol as P0-A.

Define coarse-to-fine NLL gap using MSE-N8K64 vs MSE-Oracle16 when positive.

Report:

```text
PPL
paired ΔNLL
95% paired bootstrap CI
win fraction
coarse-loss recovery fraction
gain vs MSE-N8K64
gain vs OutputAware-N8K64
gain/loss vs Canonical-4Over6
```

## 7.10 P0-B interpretation

Strong evidence of recoverability:

```text
>50% coarse-loss NLL recovery
AND meaningful absolute paired ΔNLL
AND no major held-out corpus regression
```

Evidence for intrinsic/representation bottleneck:

```text
<10% recovery
or unstable sign
or CD barely improves local OutputAware even on representative high-conflict modules
```

Coordinate descent is still a search heuristic, not a mathematical global oracle.

Therefore failure supports but does not literally prove an information-theoretic impossibility.

---

# 8. P0 final decision logic

After P0-A and P0-B, assign:

```text
P0_A_PACKING = PASS / FAIL
P0_B_COUPLED_ASSIGNMENT = PASS / FAIL / NOT_PROMOTED
```

## Case 1 — both fail

```text
P0-A FAIL
AND
P0-B FAIL or strong no-signal
```

Then:

```text
FINAL_NO_GO
```

Stop Research-2 algorithmic recovery.

Do not run P1.

## Case 2 — packing passes, coupled assignment fails

Interpretation:

```text
N-layout/channel grouping is recoverable
but coupled K assignment adds little
```

Unlock only packing/selective-routing P1 paths.

## Case 3 — packing fails, coupled assignment passes

Interpretation:

```text
layout pilot was fragile
but N8K64 representation is recoverable by coupled assignment
```

Unlock residual-aware/global-selector P1 paths.

## Case 4 — both pass

Unlock both:

```text
residual-aware coarse assignment
selective foldable packing
bounded combination study
```

Do not immediately combine them before each is independently validated.

---

# 9. P1-A — CONDITIONAL
# Residual-aware sequential coarse-format assignment

Unlock only if P0-B has meaningful signal.

## 9.1 Motivation

Local selector asks:

```text
which format gives this region smaller standalone error?
```

Residual-aware selector asks:

```text
given the residual already accumulated in this N8 stripe,
which format minimizes the updated residual?
```

## 9.2 Method

For each N8 stripe and ordered K64 region \(g\):

\[
R_{g-1}
=
\sum_{j<g}E_j^{f_j}.
\]

Choose:

\[
f_g
=
\arg\min_f
\|R_{g-1}+E_g^f\|_F^2.
\]

Then update:

\[
R_g=R_{g-1}+E_g^{f_g}.
\]

This is one-pass/offline.

Runtime representation remains:

```text
one E2/E0 format decision per N8K64 region
same K16 scales
no extra inference arithmetic
```

## 9.3 Bounded ordering bank

Only:

```text
natural_K_order
descending_local_sensitivity
descending_abs_format_margin
descending_spillover_cost
```

Do not expand beyond this unless one ordering clearly wins for a mechanistic reason.

## 9.4 Required evaluation

Use calibration-only assignment, freeze, then evaluate matched full WikiText + C4.

Stop rule:

```text
If no ordering beats local OutputAware-N8K64 on held-out PPL/NLL
on both corpora for at least one 8B model, close this branch.
```

---

# 10. P1-B — CONDITIONAL
# Sequential teacher-aligned calibration audit/variant

Unlock only if:

- P0-B or P1-A shows coupled/global selector headroom, and
- Research-1 B1 used independent FP layer inputs.

Audit first:

```text
Did Research-1 selector calibrate each layer independently from FP inputs?
or
Did it use activations from the already-quantized prefix?
```

If independent:

For layer \(l\):

```text
input  = activation produced by quantized layers < l
target = FP teacher output at layer l
```

Choose/freeze current layer format assignment against:

\[
\|Y_l^{Q-prefix,Q-layer}-Y_l^{FP-teacher}\|^2.
\]

Purpose:

```text
incorporate cross-layer error propagation
```

Novelty must not be claimed as generic residual compensation.

Project-specific claim, if successful:

> error-propagation-aware discrete coarse-format assignment under fixed N8K64 hardware format regions.

---

# 11. P1-C — CONDITIONAL
# Full-model legal representable-scale upper bound

This branch is not mandatory before P0.

Unlock if:

- P0 is positive, or
- the implementation is cheap enough that it does not delay P0.

For every K16 candidate, search legal representable E4M3 block scales under the same global-scale semantics.

Run:

```text
Fine16 + legal scale oracle
N8K64 + legal scale oracle
```

for weights.

Do not use arbitrary FP32 per-block scales.

Interpretation:

```text
little PPL gain:
  close scale branch

fine improves, coarse does not:
  scale improves quantizer quality but does not solve granularity

N8K64 materially improves:
  simple 4/6 and 6/7 were too restrictive
```

Remember expensive offline weight scale search does not imply runtime search cost if only the selected legal scale is stored.

---

# 12. P1-D — CONDITIONAL
# Scale × improved-selector interaction

Only after one selector beyond local OutputAware is validated.

Run exactly:

```text
dual_static + MSE
better_scale + MSE
dual_static + best_improved_selector
better_scale + best_improved_selector
```

on:

```text
Llama WikiText + C4
Qwen WikiText + C4
```

Do not run a Cartesian scale grid.

If scale helps only with a better selector, reinterpret old B0 as:

```text
B0-alone not useful
```

not:

```text
scale adaptation irrelevant
```

---

# 13. P1-E — CONDITIONAL
# Selective foldable packing

Unlock only if P0-A validates the Qwen full packing signal.

Do not apply packing to every eligible layer/model.

For each MLP motif, choose:

```text
identity
or
frozen foldable packing
```

Calibration predictor should use a bounded combination of:

```text
local N8 granularity-regret reduction
held-out layer-output reduction
signed-margin compatibility improvement
stability across calibration shards
```

Use calibration train/validation splits.

Do not use final WikiText/C4 evaluation to decide layer routing.

Foldability invariant:

```text
gate_proj rows
up_proj rows
down_proj columns
```

must share a functionally equivalent intermediate permutation.

No explicit runtime inverse permutation for the deployable path.

A method that manually says:

```text
if model == Qwen: pack
if model == Llama: identity
```

is not sufficient.

The routing/predictor must make the decision from calibration evidence.

---

# 14. P1-F — CONDITIONAL
# Robust calibration-consensus selector

Unlock only if:

- a selector path is promising, but
- cross-corpus instability remains, especially Qwen WikiText vs C4.

Use multiple calibration shards/distributions.

For region \(G\), obtain:

\[
f_G^{(1)},\dots,f_G^{(K)}.
\]

Stability:

\[
Stability_G
=
\max_f
\frac{1}{K}
\sum_k \mathbf{1}[f_G^{(k)}=f].
\]

Possible rule:

```text
if stable:
    use robust output-aware/residual-aware consensus
else:
    fallback to MSE or module default
```

Alternative robust objective:

\[
L_G^{robust}(f)
=
mean_k L_{G,k}(f)
+
\lambda std_k L_{G,k}(f).
\]

A worst-shard loss variant is allowed.

This is different from the old confidence rule:

```text
old: is local margin large?
new: is the decision stable across calibration distributions?
```

---

# 15. P2 — LOCKED BY DEFAULT
# Cheap diagnostics / secondary ideas

Do not run these unless P1 evidence specifically motivates them.

## 15.1 MSE bias-corrected selector

Cheap baseline:

\[
E0 \quad \text{if}\quad D_G>\tau_l.
\]

Possible scores:

```text
raw sum D
normalized sum D
sensitivity-weighted sum D
sum D - layer_bias
```

Fit on calibration only.

Do not justify by forcing E0 ratio toward 60% or maximizing entropy.

Close if threshold does not generalize across held-out text slices.

## 15.2 Format-histogram-preserving counterfactual

Diagnostic only.

Constrain coarse E0 exposure to a pre-registered budget and evaluate PPL.

Purpose:

```text
is E0 spillover causal or mostly a symptom?
```

Do not tune the budget on evaluation.

## 15.3 Task-loss / Fisher-aware selector

Only if P0-B proves better coupled objectives can recover N8K64.

A gradient-weighted candidate score is allowed.

Risks:

```text
backward calibration cost
overfitting
weak generic sensitivity novelty
```

Do not use before P0-B.

## 15.4 Model/module policy routing

Only if selective heterogeneity is strongly supported.

Candidate module policies:

```text
all E2 + canonical 4Over6
all E0
MSE-N8K64
improved-selector N8K64
packed N8K64
```

Use calibration-only policy search and held-out acceptance.

If almost all modules choose 4Over6/all-E2, the mixed-format contribution is effectively gone and must not be hidden.

## 15.5 Learned/Sinkhorn packing

Locked unless frozen greedy packing first passes full validation.

Its role would only be to estimate whether greedy packing is far from a foldable packing optimum.

Do not pursue merely because it is learnable.

---

# 16. Branches explicitly stopped/deferred in Research-2

Do not expand:

```text
more random Hadamard seeds
larger generic rotation banks
arbitrary extra 5/6/7 scale heuristics
large confidence-threshold sweeps
secondary diffusion model
1024-image SANA without a frozen Research-2 winner
native SM120 kernel implementation
large downstream benchmark expansion
activation-side recovery before weight-side recoverability
```

No method fishing.

---

# 17. Activation and diffusion policy

Research-2 is weight-first.

Research-1 W4A4 showed fine activation adaptation has signal, but no convincing coarse activation recovery exists.

Therefore:

```text
activation-side new recovery = DEFERRED
```

unless a weight-side Research-2 method passes final validation.

If a weight winner exists, re-open only a deliberately asymmetric comparison:

```text
weight adaptive / activation fixed E2+4Over6
activation adaptive / weight fixed E2
both adaptive
neither adaptive
```

For SANA:

- first use W4A16 proxy only with a frozen Research-2 winner;
- run 128-image matched screen only if proxy is positive;
- run 1024 images only if the frozen candidate has no clear Pareto disqualification.

---

# 18. Future hardware co-design — NOT RUN IN RESEARCH-2

If Research-2 ends `FINAL_NO_GO` under exact fixed N8K64 one-format-per-tile constraints, a separate future project may study:

## 18.1 Sparse format exceptions

```text
one base format per N8K64
+
0 / 1 / 2 / 4 opposite-format K16 exceptions
```

Study:

\[
Recovery(k)
=
\frac{E_{coarse}-E_{exception-k}}
{E_{coarse}-E_{fine}}.
\]

This creates metadata-vs-recovery Pareto curves.

## 18.2 Two-subtile decisions

Examples:

```text
N4K64 + N4K64
N8K32 + N8K32
```

This is a hardware co-design branch, not evidence for the exact original N8K64 constraint.

Research-2 may write a handoff plan for these but must not implement native hardware work.

---

# 19. Statistical protocol

Primary LLM paired statistic:

```text
per-sequence NLL
```

Do not use raw PPL differences alone for paired inference.

For method \(m\):

\[
\Delta_m = NLL_m-NLL_{baseline}.
\]

Report:

```text
mean paired ΔNLL
median paired ΔNLL
95% paired bootstrap CI
win fraction
PPL for interpretability
```

Use fixed identical evaluation sequences across paired methods.

For recovery:

\[
Recovery_m
=
\frac{
NLL_{coarse}-NLL_m
}{
NLL_{coarse}-NLL_{fine}
},
\]

only if:

\[
NLL_{coarse}>NLL_{fine}.
\]

Internal design targets:

```text
P0 coupled upper-bound:
  >50% recovery = strong evidence representation is recoverable

deployable candidate:
  >30% recovery consistently across multiple model/corpus settings = worth continuation

<10% recovery or unstable sign:
  close branch
```

These are pre-registration thresholds, not universal laws.

Always combine recovery fraction with:

```text
absolute paired ΔNLL
CI
sign consistency
```

A large percentage from a tiny denominator is not sufficient.

---

# 20. Evaluation leakage and calibration discipline

Every new method must store:

```text
calibration dataset
calibration sample IDs
calibration hash
calibration split role
evaluation sample IDs
evaluation hash
```

No evaluation sequence may influence:

```text
packing maps
selector thresholds
coordinate-descent assignment
module routing
scale policy
ordering choice
```

If multiple candidate methods/orders are compared, choose using calibration/validation data and freeze before final WikiText/C4 evaluation.

Do not report "best of evaluation" results.

---

# 21. Repository and Git rules

Before editing:

```text
record repo URL
branch
commit SHA
submodule SHAs
dirty status
local diff hash
```

Create an isolated Research-2 continuation branch/worktree.

Do not destructively modify Research-1 immutable artifacts.

Do not push unless explicitly instructed by the user.

Local commits are allowed and encouraged at meaningful milestones.

Do not kill or overwrite unrelated user processes/files.

---

# 22. GPU rules

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum project GPUs concurrently:

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
unknown owner + occupied => unavailable
low utilization != free
never share another user's physical GPU
never kill another user's process
never broad pkill
project reservation lock required
immediate re-check before exec
log every selection/rejection
```

SM120 absence is not a Research-2 blocker.

---

# 23. Mandatory tests before P0

## 23.1 Packing tests

```text
high-precision equivalence
permutation invertibility
shared MLP permutation consistency
attention unchanged
calibration/evaluation separation
deterministic map generation
```

## 23.2 FullLayer-CD tests

On small synthetic modules:

```text
stripe decomposition matches direct full-layer SSE
different N8 stripes have zero cross-term in Frobenius output error
incremental residual update matches full recomputation
E2/E0 flip delta is exact within tolerance
1-sweep objective never increases
2-sweep objective never increases
MSE initialization reproduced
OutputAware initialization reproduced
deterministic tie-breaking
```

## 23.3 Statistical tests

```text
identical sequence IDs across paired methods
NLL aggregation reproduces PPL
bootstrap deterministic under fixed seed
recovery undefined when coarse<=fine
```

Do not launch P0 full jobs until these pass.

---

# 24. Required Research-2 artifact tree

All new artifacts go under:

```text
artifacts/research_2/
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
  research1_handoff_audit.md
  research1_source_hashes.json

01_tests/
  packing_equivalence.json
  packing_leakage_audit.md
  full_layer_cd_unit_tests.json
  stripe_decomposition_test.json
  incremental_residual_test.json
  paired_nll_test.json

02_p0/
  p0_a_packing/
    config.json
    calibration_manifest.json
    permutation_maps/
    qwen_wikitext_results.csv
    qwen_c4_results.csv
    llama_control_results.csv
    paired_nll.csv
    bootstrap_summary.json
    module_regret.csv
    p0_a_decision.md

  p0_b_full_layer_cd/
    selected_modules.json
    calibration_manifest.json
    layer_level_results.csv
    stripe_level_results.csv
    cd_trace.jsonl
    promotion_decision.md
    full_model_results.csv
    paired_nll.csv
    bootstrap_summary.json
    p0_b_decision.md

  p0_final_decision.md
  p0_final_decision.json

03_p1/
  residual_aware/
  sequential_teacher_aligned/
  legal_scale_oracle/
  scale_selector_interaction/
  selective_packing/
  robust_consensus/

04_optional_diagnostics/
  mse_bias/
  histogram_counterfactual/
  task_loss_selector/
  module_routing/

05_final/
  master_results.csv
  experiment_manifest.jsonl
  failed_runs.jsonl
  results_summary.md
  final_decision_report.md
  final_go_no_go.json
  limitations.md
  reproduction_commands.sh
  future_hardware_codesign_handoff.md
```

All failed/aborted attempts must be preserved.

Terminal output is not the source of truth.

---

# 25. Required machine-readable fields

Common:

```text
experiment_id
phase
method
model
corpus
model_revision
dataset_hash
calibration_hash
eval_hash
repo_sha
config_hash
gpu_physical_id
gpu_logical_id
seed
status
```

Paired evaluation:

```text
sequence_id
nll
paired_nll_delta
coarse_loss_nll
method_recovery_fraction_nll
paired_bootstrap_ci_low
paired_bootstrap_ci_high
win_fraction
```

Packing:

```text
packing_method
packing_selected_layers
packing_identity_layers
packing_calibration_file
packing_calibration_predicted_gain
packing_heldout_output_gain
packing_full_ppl_gain
permutation_map_hash
foldability_verified
high_precision_equivalence_passed
```

FullLayer-CD:

```text
cd_initialization
full_layer_cd_sweep
full_layer_cd_order
stripe_id
region_id
format_before
format_after
full_layer_format_flip_rate
residual_norm_before
residual_norm_after
residual_cancellation_gain
full_layer_output_sse
full_layer_vs_local_outputaware_gain
convergence_status
```

Residual-aware:

```text
residual_aware_order
residual_norm_before
residual_norm_after
residual_cancellation_gain
```

Robust selector:

```text
selector_shard_agreement
selector_cross_corpus_agreement
selector_robust_loss
```

Routing:

```text
module_policy
module_policy_switch_count
module_policy_metadata_bits
```

---

# 26. Failure handling

For every failure store:

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

Classification examples:

```text
code_bug
OOM
dataset_or_network
model_access
dependency
GPU_became_occupied
numerical_issue
calibration_leakage
equivalence_failure
unsupported_semantics
runtime_excessive
unknown
```

Fix and rerun only when scientifically valid.

Never silently drop failed cases.

---

# 27. Novelty guardrails inherited from the current suggestion/literature audit

These are planning guardrails, not a substitute for a fresh paper-submission literature review.

Do not claim as novel:

```text
K16 FP4/INT4 adaptive format selection itself
generic adaptive FP4 micro-formats
generic block heterogeneity
generic weight/activation asymmetry
Hadamard rotation for quantization
generic residual compensation
generic learnable channel permutation
generic global mixed-precision control
```

Current overlap noted in the suggestion includes:

```text
Adaptive Block-Scaled Data Types / IF4
published MixFP4
AdaMX / heterogeneity-aware microscaling
QuaRot / SpinQuant
GPTQ / residual-compensation PTQ
PermLLM-like learnable permutation
WINDQuant/global controller-style work
```

The most defensible remaining novelty center is:

> **coarse hardware format-control granularity and decision compression**:
> compress many fine K16 representation preferences into one legal coarse N8K64 decision,
> using coupled/global assignment and/or foldable channel packing with little/no runtime overhead.

If residual-aware assignment succeeds, novelty must be framed as:

```text
coupled residual-aware discrete format assignment
under fixed coarse hardware format regions
```

not "we use residual compensation."

If packing succeeds, novelty must be:

```text
foldable channel packing for coarse-format compatibility
```

not "we learn/permute channels."

Avoid using `MixFP4` as the final public method name if the semantics differ from the published method.

---

# 28. Final Research-2 decision tree

```text
START
  |
  |-- P0-A: full Qwen foldable packing on WikiText + C4?
  |       |
  |       +-- PASS --> N-layout recovery alive
  |       |
  |       +-- FAIL --> packing branch closed
  |
  |-- P0-B: coupled FullLayer-CD recover substantial coarse loss?
          |
          +-- FAIL
          |    |
          |    +-- if P0-A also FAIL:
          |           FINAL_NO_GO
          |
          +-- PASS
               |
               +-- fixed N8K64 representation is recoverable
               |
               +-- unlock residual-aware practical selector
               |
               +-- optionally legal scale / robust selector
               |
               +-- cross-model validation
                        |
                        +-- stable --> CONDITIONAL_GO / STRONG_GO
                        |
                        +-- unstable --> FINAL_NO_GO
```

If P0-A passes while P0-B fails:

```text
unlock selective packing only
```

If P0-B passes while P0-A fails:

```text
unlock residual/global selector only
```

---

# 29. Research-2 final status vocabulary

Use one of:

```text
FINAL_NO_GO
CONDITIONAL_GO
STRONG_GO
INSUFFICIENT_EVIDENCE
```

`INSUFFICIENT_EVIDENCE` is only allowed for genuine external blockers, not because experiments were skipped.

---

# 30. Execution order

```text
0. read current research_2.md completely and record SHA256
1. create isolated Research-2 branch/worktree
2. audit Research-1 final artifacts and source hashes
3. confirm P0-A calibration provenance / no eval leakage
4. confirm Research-1 B1 calibration-input semantics
5. confirm FullLayer-CD truly NOT_RUN previously
6. run packing equivalence + leakage tests
7. implement/test N8-stripe FullLayer-CD residual mathematics
8. run P0-A Qwen full WikiText + C4
9. run Llama packing negative control if cheap
10. run P0-B representative modules
11. decide whether P0-B promotes to full model
12. if promoted, run full-model coupled assignments and matched WikiText/C4
13. write P0 final decision
14. if both P0 fail -> FINAL_NO_GO and STOP
15. if P0-B positive -> P1 residual-aware sequential selector
16. if P0-A positive -> P1 selective packing/routing
17. only if motivated -> legal scale oracle / sequential teacher alignment / robust consensus
18. run bounded scale x improved-selector interaction only after selector winner
19. cross-model/corpus validation
20. only after frozen LLM winner -> SANA W4A16 proxy
21. only after SANA proxy passes -> bounded 128-image screen
22. only after robust cross-domain winner -> reconsider W4A4 / 1024 images / native
23. regenerate all final aggregates from immutable attempts
24. issue final strict Research-2 decision
25. if FINAL_NO_GO, write hardware co-design handoff but do not implement native work
```

---

# 31. Final framing

Research-2 is **not** a search for one more trick.

It exists to answer one unresolved question:

> Is N8K64 fundamentally too restrictive, or did Research-1 use selectors/layout
> assignments that were too local to exploit cross-region error cancellation and
> model-specific N-channel structure?

If the two P0 tests are negative, stop.

If either is positive, continue only along the mechanism identified by that positive result.

The project must remain falsifiable.
