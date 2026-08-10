# Research Plan: Coarse-Granularity MixFP4 with Hardware-Aware Recovery on A6000 / RTX 6000 Ada

**Scope:** Phase A + Phase B algorithmic / accuracy study only  
**Primary domains:** LLM + text-to-image diffusion  
**Target GPUs for this document:** NVIDIA RTX A6000 (GPU 0–3) and NVIDIA RTX 6000 Ada (GPU 4–6)  
**Blackwell / SM120 required for completion:** **No**  
**Future hardware-validation target:** RTX 5090 / SM120, only after Phase A/B Go/No-Go  
**Primary implementation anchors:**
- NVFP4-RaZeR: https://github.com/abdelfattah-lab/NVFP4-RaZeR
- FourOverSix / IF data types: https://github.com/mit-han-lab/fouroversix

**Secondary diffusion anchors:**
- DeepCompressor: https://github.com/nunchaku-ai/deepcompressor
- SANA: https://github.com/NVlabs/Sana
- Nunchaku (deployment/reference only in Phase A/B): https://github.com/nunchaku-ai/nunchaku

---

# 0. Executive summary

This project asks whether a fine-grained adaptive FP4 idea can remain useful when its **format-selection granularity is made coarse enough to resemble a realistic hardware operand tile**.

The core setup is:

- Every K16 block keeps its own block scale.
- For each K16 block, construct two numerical candidates:
  - E2M1 / NVFP4-like candidate.
  - E0M3 / INT4-like sign-magnitude candidate.
- Fine-grained `Oracle16` chooses the better format independently for every K16 block.
- Coarse `MixFP4` forces many K16 blocks to share one format decision while retaining their own K16 scales.
- The main weight hardware proxy is `N8×K64`.
- The main activation hardware proxy is `M16×K64`.

The research has five goals:

1. Verify that fine-grained E2M1/E0M3 adaptation is meaningfully better than standard NVFP4.
2. Measure how much benefit is lost when format selection is coarsened.
3. Decompose the loss into N-direction conflict, K-direction conflict, and selector-objective mismatch.
4. Recover the loss without restoring per-K16 runtime format dispatch.
5. Decide, using evidence from both LLMs and diffusion models, whether a later RTX 5090 / SM120 native-kernel phase is justified.

The key scientific story is **not** “make hardware granularity finer.” It is:

> Reshape selection, layout, and representation so that a coarse hardware selector becomes sufficient.

All results, including failed runs, must be written to `artifacts/`. Terminal output is never the source of truth.

---

# 1. Research thesis

## 1.1 The granularity mismatch

Let a K16 scale block be indexed by `b`. For each block we compute candidate errors:

\[
e_b^{E2}, \qquad e_b^{E0}.
\]

The ideal fine-grained selector is:

\[
f_b^* = \arg\min_{f\in\{E2,E0\}} e_b^f.
\]

This gives the `Oracle16` error inside a coarse region `G`:

\[
E_{oracle}(G)=\sum_{b\in G}\min(e_b^{E2},e_b^{E0}).
\]

A coarse hardware region must choose one format:

\[
E_{coarse}(G)=
\min\left(\sum_{b\in G}e_b^{E2},\sum_{b\in G}e_b^{E0}\right).
\]

Define **format-granularity regret**:

\[
R_G = E_{coarse}(G)-E_{oracle}(G) \ge 0.
\]

The research target is to reduce the end-to-end effect of this regret while keeping the coarse format region fixed.

## 1.2 Format margin

For each K16 block define:

\[
D_b=e_b^{E2}-e_b^{E0}.
\]

Interpretation:

- `D_b < 0`: E2M1 is preferred.
- `D_b > 0`: E0M3 is preferred.
- `|D_b|`: cost of choosing the wrong format under the local objective.

For a region `G`, define:

\[
P_G=\sum_{b\in G}\max(D_b,0),
\]

\[
N_G=\sum_{b\in G}\max(-D_b,0).
\]

Then:

\[
R_G=\min(P_G,N_G).
\]

This identity should be unit-tested and used to interpret coarse-format conflict.

## 1.3 Margin-weighted conflict

Count-based homogeneity is useful but insufficient. A tile with one very high-margin minority block can be more harmful than a 50/50 tile whose margins are near zero.

Define:

\[
C_G^{margin}=
\frac{\min(P_G,N_G)}{P_G+N_G+\epsilon}.
\]

Also report:

\[
H_G^{margin}=1-C_G^{margin}.
\]

These metrics are diagnostic; they do not replace direct PPL/image-quality evaluation.

---

# 2. Terminology and baseline semantics

The experiment names must be exact and machine-readable. Do not use ambiguous labels such as “mixed FP4” without recording the exact scale rule, format rule, and region shape.

## 2.1 Canonical baseline: NVFP4

`NVFP4` means:

- payload/codebook: E2M1.
- K16 scale granularity unless the pinned reference implementation requires an explicitly documented equivalent.
- no E0M3 candidate.
- reference behavior must be validated against NVFP4-RaZeR and/or FourOverSix.

## 2.2 Canonical baseline: 4Over6

`NVFP4+4Over6` means the canonical FourOverSix 4/6 method as implemented by the pinned FourOverSix repository.

Do not rewrite or reinterpret 4Over6 merely to make it compose with MixFP4.

Record exact scale rules, rounding, saturation, transform behavior, and repository commit.

## 2.3 Project method family: MixFP4

In this project, `MixFP4` is the shorthand for **adaptive E2M1/E0M3 selection with K16 scale blocks**.

Required modes:

- `MixFP4-Oracle16`: one format choice per 1×K16 block.
- `MixFP4-K64`: one format choice per 1×K64 region.
- `MixFP4-N8K16`: one format choice per N8×K16 region; diagnostic only.
- `MixFP4-N8K64`: one format choice per N8×K64 weight region; primary weight hardware proxy.
- `MixFP4-M16K16`: activation diagnostic.
- `MixFP4-M16K64`: primary activation hardware proxy.

All constituent K16 blocks retain their own candidate scales.

## 2.4 Proposed compositional baseline: MixFP4 + 4Over6

The user's proposed comparison is scientifically useful **if and only if composition semantics are well-defined**.

The desired comparison ladder is:

1. `NVFP4`
2. `NVFP4+4Over6`
3. `NVFP4+MixFP4` (operationally: coarse MixFP4 relative to standard NVFP4)
4. `NVFP4+MixFP4+4Over6`
5. `NVFP4+MixFP4+Ours`
6. `NVFP4+MixFP4+4Over6+Ours`

For implementation and reporting, use less ambiguous names:

- `nvfp4`
- `nvfp4_4over6`
- `mixfp4_oracle16`
- `mixfp4_n8k64`
- `mixfp4_n8k64_4over6`
- `mixfp4_n8k64_ours`
- `mixfp4_n8k64_4over6_ours`

### Composition validity requirement

Before using `mixfp4_*_4over6` as a baseline:

1. Audit canonical 4Over6 semantics in the pinned repo.
2. Define exactly how its scale-selection rule applies to E2 and E0 candidates.
3. Prefer a symmetric candidate-level definition if mathematically valid.
4. Verify that fixing the format to E2M1 reduces to canonical `NVFP4+4Over6` within numerical tolerance.
5. If exact reduction cannot be established, rename the method `mixfp4_*_4over6_style` and treat it as a project ablation, **not** a canonical 4Over6 baseline.
6. Never claim published 4Over6 supports E0M3 unless the repository/paper explicitly does.

This protection is mandatory.

## 2.5 “Ours” definition

`Ours` is not one fixed trick at the beginning. It is a Phase B family selected by calibration evidence:

- `Ours-S`: sensitivity/activation-aware coarse selector.
- `Ours-P`: margin-aware foldable permutation/packing.
- `Ours-R`: granularity-aware block-aligned rotation/transform.
- `Ours-SP`, `Ours-SR`, `Ours-PR`, `Ours-SPR`: valid combinations.
- `Ours-Best`: best deployable combination selected without evaluation-set cherry-picking.

The final report must state exactly which components are included.

---

# 3. Core hypotheses

## H1 — Fine-grained adaptation is real

`MixFP4-Oracle16` should outperform standard NVFP4 on a meaningful subset of model×dataset settings.

If this is not true, stop the direction before hardware work.

## H2 — Hardware-format granularity causes measurable loss

Moving from `Oracle16` to `N8K64` / `M16K64` should produce measurable granularity regret in at least some layers/models.

If coarse granularity already retains nearly all benefit, this is good for deployment but reduces the need for complex Phase B recovery.

## H3 — Regret has structure

The loss should be explainable by one or more of:

- N-direction format conflict.
- K-direction format conflict.
- low-margin noisy decisions.
- mismatch between local weight MSE and downstream sensitivity.
- diffusion timestep-dependent activation distribution.

## H4 — Coarse-format loss is recoverable

At least one Phase B method should materially improve coarse MixFP4 without restoring per-K16 runtime dispatch.

## H5 — The phenomenon generalizes

The direction should appear in:

- Llama-3.1-8B.
- Qwen3-8B.
- at least one diffusion transformer, mandatory primary target SANA-1.6B.

Cross-domain success is stronger evidence than a single-model win.

---

# 4. Scope boundaries

## Included

- Reference/fake E2M1 and E0M3 quantization.
- K16 candidate scales.
- Fine and coarse format selection.
- W4A16 diagnostics.
- W4A4 fake/reference quantization.
- LLM PPL and selected downstream tasks.
- Diffusion denoiser/flow-output error and image-quality evaluation.
- N/K conflict decomposition.
- Margin-aware conflict metrics.
- Activation/Hessian-like coarse selector.
- Foldable channel permutation for valid graph motifs.
- Block-aligned Hadamard/structured rotation experiments.
- Granularity-aware transform selection.
- Diffusion timestep stability analysis.
- Canonical 4Over6 baseline.
- Conditional MixFP4+4Over6 composition.
- A6000 vs RTX 6000 Ada numerical reproducibility.
- Artifacted Go/No-Go recommendation.

## Excluded from completion criteria

- Native SM120 E0M3 instruction execution.
- SASS patching.
- CUTLASS SM120 implementation.
- RTX 5090 kernel throughput.
- claims about hidden Blackwell instruction throughput.
- claims that A6000/6000 Ada execute native NVFP4 tensor-core arithmetic.

Any runtime measured from fake quantization on A6000 / RTX 6000 Ada is engineering information only, not evidence of Blackwell FP4 speed.

---

# 5. Repository strategy

## 5.1 NVFP4-RaZeR — primary LLM harness

Use for:

- Llama/Qwen model wrappers.
- PPL evaluation.
- weight and activation quantization hooks.
- NVFP4 baseline.
- repository-provided NVIF4/adaptive reference where useful.
- 4Over6 contextual support where present.

Do not overwrite original quantizers. Add a separate granularity-aware path.

Suggested API:

```python
quant_mixfp4_granularity(
    x,
    *,
    scale_group_size=16,
    format_region="oracle16",
    operand_role="weight_b",
    selector="mse",
    scale_rule="standard",
    calibration_stats=None,
    transform=None,
    permutation=None,
    return_stats=False,
)
```

## 5.2 FourOverSix — semantic and independent reference

Use for:

- canonical NVFP4.
- canonical 4Over6.
- IF/adaptive datatype reference behavior.
- PyTorch reference backend on non-Blackwell GPUs.
- Hadamard/transform reference.
- Diffusers integration where useful.

On A6000 / 6000 Ada, use the repository's reference path when native kernels require Blackwell, e.g. `SKIP_CUDA_BUILD=1` if required by the pinned version.

## 5.3 DeepCompressor — primary diffusion PTQ/evaluation harness

Use for:

- SANA-1.6B configuration.
- diffusion calibration collection.
- W4A4 PTQ scaffolding.
- MJHQ/image evaluation.
- SVDQuant/NVFP4 comparison context.
- paired reference-image protocol.

Integrate the same project MixFP4 core logic rather than implementing unrelated E2/E0 semantics inside diffusion code.

## 5.4 NVlabs/Sana — canonical diffusion model code

Use official SANA weights/model definitions when possible.

## 5.5 Nunchaku — later deployment/context reference

Use in Phase A/B only for:

- understanding existing 4-bit diffusion deployment.
- optional reference checkpoint conversion.

Do not make Nunchaku native performance a Phase A/B completion requirement.

## 5.6 Repository pinning

Before experiments save:

- URL.
- branch.
- exact commit SHA.
- submodule SHAs.
- dirty status.
- local patch diff/hash.
- dependency versions.

Write:

```text
artifacts/00_environment/repo_manifest.json
artifacts/00_environment/patch_manifest.json
```

Never report only `main`.

---

# 6. Mandatory GPU operating policy

This machine is shared.

Physical mapping:

```text
GPU 0,1,2,3 = NVIDIA RTX A6000
GPU 4,5,6   = NVIDIA RTX 6000 Ada
```

Maximum GPUs used concurrently by this project:

```text
3
```

## 6.1 Before every GPU process launch

Immediately before **every** GPU process launch, inspect GPU state.

At minimum run:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi pmon -c 1
```

For each visible compute PID:

```bash
ps -o user=,pid=,cmd= -p <PID>
```

A GPU is eligible only if:

- no process from another user is actively using it; and
- it is not already reserved by another active job from this project in a conflicting way.

If ownership is unknown and the GPU looks occupied, mark it unavailable.

Low utilization does **not** imply the GPU is free.

## 6.2 No sharing

Never:

- share a GPU with another user.
- launch first and inspect later.
- kill another user's process.
- use broad `pkill`/destructive cleanup.
- enable a policy that allows accidental GPU sharing.

If all compliant GPUs are occupied, continue CPU-side analysis/artifact work and re-check before the next GPU launch.

## 6.3 Launch guard

Implement:

```text
scripts/gpu_guard.py
```

It must:

- inventory GPUs.
- map UUIDs to physical indices.
- map PIDs to users where possible.
- enforce the A6000/6000Ada mapping.
- enforce max 3 active project GPUs.
- prevent use of GPUs occupied by other users.
- create project-side reservation/lock records to reduce race conditions.
- re-check immediately before exec.
- log every decision.

Log:

```text
artifacts/00_environment/gpu_usage_log.jsonl
```

For single-GPU jobs launch with:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_id> ...
```

Record both physical GPU id and the process-visible logical id.

Most Phase A/B jobs should be independent single-GPU processes.

---

# 7. Quantization invariants

## 7.1 Scale granularity is not format granularity

Critical invariant:

```text
scale_group_size = 16
```

must remain fixed unless an experiment explicitly studies a different published scale rule such as canonical 4Over6.

For the primary MixFP4 granularity study, changing format region must **not** silently change the number of values sharing a scale.

Never simulate `N8K64` by flattening 512 values and assigning one scale.

## 7.2 Preserve matrix layout

Weights:

```text
W.shape = [N, K]
```

Primary weight region:

```text
W[n:n+8, k:k+64]
```

Activations:

```text
X.shape = [M, K]
```

Primary activation region:

```text
X[m:m+16, k:k+64]
```

Do not use `reshape(-1, groupsize)` as a substitute for real 2-D hardware-region grouping.

## 7.3 E2M1 candidate

Use repository-consistent NVFP4 E2M1 semantics.

Reference positive magnitudes should be consistent with:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

with sign.

## 7.4 E0M3 candidate

For this project reference simulation, use sign-magnitude uniform levels:

```text
0, ±1, ±2, ±3, ±4, ±5, ±6, ±7
```

Do not use two's-complement `-8`.

Saturate magnitude at 7.

If the pinned FourOverSix IF semantics differ, record the difference and run an explicit semantic-cross-check; do not silently replace this project definition.

## 7.5 Candidate scales

For every K16 block:

- build E2 candidate and its scale.
- build E0 candidate and its scale.
- retain candidate-specific K16 scales.

After a coarse region picks one format, each K16 block dequantizes using the scale associated with that chosen candidate.

---

# 8. Required metrics

## 8.1 Universal local metrics

Record:

- MSE.
- NMSE.
- relative L2.
- cosine error.
- mean absolute error.
- max absolute error.
- E0/E2 counts.
- signed/absolute format margin.
- count homogeneity.
- margin-weighted conflict/homogeneity.
- Oracle error.
- constrained error.
- granularity regret.
- normalized regret.
- selector disagreement.
- per-layer/module statistics.

## 8.2 Oracle gain retention

For an end-to-end metric where lower is better, e.g. PPL:

\[
Retention(G)=
\frac{M_{NVFP4}-M_G}
{M_{NVFP4}-M_{Oracle16}}.
\]

Only report when the denominator indicates a meaningful Oracle improvement.

Otherwise report `N/A` and raw deltas.

## 8.3 Sensitivity-weighted error

For a weight region `G`, collect calibration activations and compare:

\[
L_f^{out}=\|X_G(W_G-Q_f(W_G))^T\|_F^2.
\]

Equivalent quadratic form:

\[
L_f^{out}=\mathrm{tr}(\Delta W_G H_G \Delta W_G^T),
\qquad H_G=X_G^T X_G.
\]

Define sensitivity-aware margins/regret analogously.

---

# 9. Phase 0 — environment, semantic audit, and unit tests

Phase A must not start until core invariants pass.

## 9.1 Environment capture

Save:

- hostname/OS.
- Python.
- PyTorch.
- Transformers.
- Diffusers.
- CUDA runtime/toolkit.
- NVIDIA driver.
- GPU inventory.
- package freeze.
- repo manifests.
- disk/cache locations.

## 9.2 Baseline reproduction

LLM smoke model first:

- high precision.
- NVFP4.
- repository adaptive/IF reference if available.
- canonical 4Over6.

Diffusion smoke run:

- high precision SANA path.
- repository NVFP4/reference fake quantization path.
- verify deterministic prompt/seed reproduction.

## 9.3 Mandatory unit tests

1. E2M1 codebook mapping.
2. E0M3 mapping.
3. K16 scale grouping invariant.
4. Oracle error <= all-E2 error.
5. Oracle error <= all-E0 error.
6. Coarse constrained error >= Oracle under the same objective.
7. `R_G = min(P_G,N_G)` identity.
8. nested MSE granularity monotonicity where mathematically applicable.
9. correct `[N,K]` N8K64 mapping.
10. correct `[N,K]` N8K16 mapping.
11. correct `[M,K]` M16K64 mapping.
12. correct `[M,K]` M16K16 mapping.
13. tail dimensions.
14. deterministic seeds.
15. no baseline regression.
16. high-precision permutation equivalence.
17. high-precision rotation equivalence.
18. canonical 4Over6 reduction test for any MixFP4+4Over6 composition.
19. diffusion fixed-prompt/fixed-seed reproducibility.

Write all test output under:

```text
artifacts/02_tests/
```

---

# 10. Phase A0 — fast feasibility and N/K conflict decomposition

This is the first scientific gate.

## 10.1 Models

Mandatory first-pass:

- `meta-llama/Meta-Llama-3.1-8B` or exact supported ID/revision.
- `Qwen/Qwen3-8B`.
- SANA-1.6B using pinned official/DeepCompressor configuration.

Use a smaller smoke model before 8B runs if required for correctness.

## 10.2 Weight-only W4A16 modes

For each model, evaluate:

```text
high_precision
nvfp4
all_e0m3
mixfp4_oracle16
mixfp4_k64_row
mixfp4_n8k16
mixfp4_n8k64
nvfp4_4over6
```

If composition is validated:

```text
mixfp4_oracle16_4over6
mixfp4_n8k64_4over6
```

## 10.3 Why N8K16 is mandatory

`N8K16` is a diagnostic counterfactual.

Compare:

- Oracle16 → K64 isolates K-direction coarsening.
- Oracle16 → N8K16 isolates N-direction coarsening.
- K64/N8K16 → N8K64 measures interaction.

For every layer/module report:

- `regret_k_only`.
- `regret_n_only`.
- `regret_nk`.
- interaction residual.

The purpose is to decide whether Phase B should prioritize permutation or K-aligned transforms.

## 10.4 Gate

Proceed to full Phase A/B if:

- Oracle16 has measurable benefit in at least a meaningful subset of settings; and
- coarse regions create measurable loss or a scientifically interesting near-zero-loss result.

If Oracle16 is consistently useless, produce an early No-Go report but still complete required reproducibility/audit artifacts.

---

# 11. Phase A1 — full LLM W4A16 granularity study

## 11.1 Main models

Mandatory:

- Llama-3.1-8B.
- Qwen3-8B.

## 11.2 Datasets

Mandatory:

- WikiText-2.
- C4 fixed slices.

Use identical tokenization/evaluation slices across compared methods.

## 11.3 Granularity sweep

Required weight regions:

```text
all_e2m1
all_e0m3
oracle16      = 1×K16
k32_row       = 1×K32
k64_row       = 1×K64
n2k64
n4k64
n8k16         = diagnostic
n8k64         = primary hardware proxy
n16k64
n32k64
n64k64
layer
```

Also run:

- canonical 4Over6.
- repository contextual baselines (RaZeR, IF/reference) where semantically relevant.
- validated MixFP4+4Over6 endpoints.

## 11.4 Per-module analysis

At minimum:

- q_proj.
- k_proj.
- v_proj.
- o_proj.
- gate_proj.
- up_proj.
- down_proj.
- other linear layers.

Report E0 ratio, margins, count homogeneity, margin conflict, regret, sensitivity-weighted regret, and PPL-sensitive layer ablations where practical.

---

# 12. Phase A2 — diffusion W4A16 feasibility

Weight-only diffusion is a diagnostic, even if final deployment is W4A4.

## 12.1 Primary diffusion model

Mandatory:

- SANA-1.6B.

Secondary if environment/time permits after mandatory work:

- PixArt-Sigma, or
- FLUX.1-schnell.

Do not replace SANA silently.

## 12.2 Fixed evaluation protocol

Persist:

- prompts.
- random seeds.
- scheduler.
- number of steps.
- CFG/PAG settings.
- resolution.
- model revision.
- VAE/text-encoder revisions.

Generate a BF16/high-precision reference set once.

## 12.3 Fast proxy metrics for every configuration

For fixed prompts/seeds and selected denoising steps record:

- model/flow/denoiser output MSE.
- NMSE.
- relative L2.
- cosine error.
- latent trajectory deviation.
- layer-output error.

## 12.4 Image metrics

Screening set for all important Phase A configurations:

```text
128–256 fixed prompts
```

Measure when supported:

- LPIPS vs reference.
- PSNR vs reference.
- SSIM optional.
- ImageReward.
- CLIPScore optional/contextual.

For finalists later, use a larger set (e.g. 1024 fixed MJHQ prompts) and report FID/ImageReward plus paired metrics.

Do not use tiny-sample FID as strong evidence.

---

# 13. Phase A3 — W4A4 and one-side/two-side adaptation

Only start after W4A16 semantics and diagnostics are stable.

## 13.1 Activation format regions

Required:

```text
oracle16_a = 1×K16 per activation row
k64_row_a = 1×K64
m16k16    = M16×K16 diagnostic
m4k64
m8k64
m16k64    = primary activation hardware proxy
m32k64
```

Scale granularity remains K16 for the primary MixFP4 study.

## 13.2 Mandatory LLM comparisons

```text
W oracle16 / A oracle16
W oracle16 / A E2
W E2       / A oracle16

W n8k64    / A E2
W E2       / A m16k64
W n8k64    / A m16k64

W n8k64    / A E0
W E0       / A m16k64
```

Add 4Over6-composed counterparts only if validated and computationally reasonable.

Answer:

- Which operand is more sensitive?
- Is weight-only adaptation sufficient?
- Is activation adaptation worth runtime complexity?
- Does both-side adaptive accuracy justify a future native implementation?

## 13.3 Mandatory SANA W4A4 subset

At minimum compare:

```text
BF16/high precision
NVFP4 W4A4 reference
MixFP4 Oracle16 W / Oracle16 A
MixFP4 N8K64 W / E2 A
E2 W / MixFP4 M16K64 A
MixFP4 N8K64 W / M16K64 A
```

Use proxy evaluation first, then image generation for selected variants.

---

# 14. Phase A4 — canonical and compositional baseline matrix

The final comparison matrix must make the user's proposed ladder explicit.

## 14.1 Canonical baselines

```text
HP/BF16
NVFP4
NVFP4 + canonical 4Over6
All-E0M3
MixFP4 Oracle16
MixFP4 coarse (N8K64 / M16K64)
```

## 14.2 Advanced baseline if composition validates

```text
MixFP4 Oracle16 + 4Over6
MixFP4 coarse + 4Over6
```

If composition does not exactly inherit canonical behavior when fixed to E2M1, report as:

```text
MixFP4 + 4Over6-style scaling
```

and keep canonical 4Over6 separate.

## 14.3 Ours comparisons

Mandatory endpoint comparisons after Phase B:

```text
MixFP4 coarse
MixFP4 coarse + Ours

MixFP4 coarse + 4Over6              # if valid
MixFP4 coarse + 4Over6 + Ours       # if valid
```

This directly tests whether Ours is complementary to better scaling rather than merely compensating for a weak baseline.

---

# 15. Phase B1 — sensitivity/activation-aware coarse selector (Ours-S)

For each N8K64 weight region compare:

## Selector MSE

\[
\arg\min_f \|W_G-Q_f(W_G)\|_F^2.
\]

## Selector output-aware

\[
\arg\min_f \|X_G(W_G-Q_f(W_G))^T\|_F^2.
\]

Calibration sizes:

```text
32
128
256
```

sequences/prompts where feasible.

For diffusion, sample activations across denoising timesteps rather than only one timestep.

Record:

- selector disagreement.
- margin of changed decisions.
- local weight regret.
- sensitivity-weighted regret.
- layer-output error.
- PPL or diffusion proxy/image effects.
- stability vs calibration size.

Do not select on evaluation data.

---

# 16. Phase B2 — margin-aware permutation and packing (Ours-P)

This is the primary method if N-direction conflict is substantial.

## 16.1 Upper-bound all-linear permutation

For diagnostic purposes:

1. derive per-output-channel format-preference signatures.
2. cluster/sort channels.
3. pack into N8 groups.
4. quantize with N8K64.
5. explicitly inverse-permute outputs.

Methods:

```text
no_permutation
sort_by_e0_ratio
margin_vector_clustering
greedy_min_regret_n8
sensitivity_weighted_greedy_n8
```

Explicit inverse-permutation variants are `upper_bound_only` unless folding is proven.

## 16.2 Signature definition

Do not use only E0 ratio.

Construct per-channel signatures from signed margins across K64 locations, e.g.:

\[
s_n=[D_{n,0},D_{n,1},\dots].
\]

Also test sensitivity-weighted signatures.

## 16.3 Foldable LLM MLP permutation

For gated Llama/Qwen MLP, apply a shared intermediate-channel permutation across:

```text
gate_proj rows
up_proj rows
down_proj columns
```

Verify high-precision equivalence before quantization.

## 16.4 Foldable diffusion FFN permutation

For a graph motif equivalent to:

\[
y=W_2\,\sigma(W_1x),
\]

apply a common intermediate permutation only when exact graph equivalence is established.

Do not modify attention ordering without an explicit proof/test.

## 16.5 Optimization objective

Preferred objective:

\[
P^*=\arg\min_P \sum_G R_G(PW)
\]

or sensitivity-weighted equivalent.

This is stronger than sorting by E0 ratio.

---

# 17. Phase B3 — rotation / block-aligned transform (Ours-R)

Use primarily when K-direction conflict is substantial or permutation leaves large residual regret.

## 17.1 Baseline transform bank

```text
identity
H16
H32
H64
H128
random_signed_H64_seed0
random_signed_H64_seed1
random_signed_H64_seed2
random_signed_H64_seed3
```

Use normalized orthogonal transforms.

Verify high precision:

\[
(XH)(WH)^T \approx XW^T.
\]

## 17.2 Granularity-aware transform objective

Do not choose transforms only by ordinary MSE.

Evaluate:

\[
L(R)=
\hat L_{output}(R)
+\lambda \hat L_{granularity}(R),
\]

where terms are normalized and `L_granularity` includes coarse regret or sensitivity-weighted coarse regret.

Required `lambda` values:

```text
0
0.1
1.0
10.0
```

Candidate bank selection must use calibration data only.

## 17.3 Hardware-aligned structured transforms

If generic Hadamard shows positive signal, add a bounded set of K32/K64-local structured transforms/permutations rather than immediately attempting expensive continuous learned rotation.

Every transform must be tagged:

```text
exact_foldable
potentially_foldable
upper_bound_only
```

Never call a transform “free” unless folding is demonstrated.

---

# 18. Phase B4 — combined Ours

Select combinations based on Phase B individual results, not an uncontrolled Cartesian sweep.

Required candidates:

```text
coarse MixFP4 + Ours-S
coarse MixFP4 + Ours-P
coarse MixFP4 + Ours-R
coarse MixFP4 + Ours-S+P
coarse MixFP4 + Ours-S+R
coarse MixFP4 + Ours-P+R
best valid coarse MixFP4 + Ours
```

If 4Over6 composition validates, run matched endpoints:

```text
coarse MixFP4 + 4Over6
coarse MixFP4 + 4Over6 + best Ours
```

The comparison must answer whether Ours remains useful on top of a stronger scale baseline.

---

# 19. Phase B5 — diffusion timestep stability and optional timestep-aware selection

Diffusion activations vary across denoising time. This must be measured rather than assumed away.

## 19.1 Timestep sampling

For a 20-step SANA configuration, analyze at least representative early/mid/late positions, e.g. six approximately spaced steps, and all steps when inexpensive.

Persist exact timestep indices/noise levels.

## 19.2 Metrics over timestep

For each selected layer/region track:

- E0/E2 preference ratio vs timestep.
- signed-margin correlation across timesteps.
- selector agreement across timesteps.
- N8K64/M16K64 homogeneity vs timestep.
- margin conflict vs timestep.
- regret vs timestep.
- sensitivity-weighted regret vs timestep.

## 19.3 Decision

Classify each layer as:

```text
stable_format_preference
stable_coarse_locality
unstable_timestep_dependent
```

Only if instability is material, test a small timestep-bucketed selector/transform as an ablation.

Do not introduce dynamic timestep metadata unless it produces a clear benefit.

---

# 20. Models, datasets, and evaluation sets

## 20.1 LLM smoke model

Preferred:

```text
meta-llama/Llama-3.2-1B
```

If inaccessible, use the smallest supported Qwen model and record the substitution reason.

## 20.2 LLM main models

Mandatory:

```text
Llama-3.1-8B
Qwen3-8B
```

Record exact Hugging Face IDs and revisions.

## 20.3 LLM datasets

Mandatory:

```text
WikiText-2
C4
```

Persist fixed C4 sample indices/offsets.

Finalists should also attempt:

```text
ARC-Easy
HellaSwag
PIQA
WinoGrande
```

PPL remains mandatory even if downstream integration fails.

## 20.4 Diffusion primary model

Mandatory:

```text
SANA-1.6B
```

Use a pinned official/DeepCompressor-supported configuration.

## 20.5 Diffusion secondary model

After mandatory SANA work, attempt one of:

```text
PixArt-Sigma
FLUX.1-schnell
```

Choose based on environment/model access/runtime practicality and document the choice.

## 20.6 Diffusion calibration/evaluation

Calibration:

- fixed 128 prompt default.
- timestep-stratified activation collection.

Screening evaluation:

- 128–256 fixed prompts/seeds.

Finalist evaluation:

- 1024 fixed prompts (prefer MJHQ where supported).

Large FID evaluation beyond 1024 is optional unless compute budget clearly permits.

---

# 21. Cross-GPU numerical reproducibility

On one LLM smoke model and one small SANA screening set, run matched configurations on:

- one free RTX A6000.
- one free RTX 6000 Ada.

At minimum:

```text
high_precision
nvfp4
mixfp4_oracle16
mixfp4_n8k64
best_ours
```

Compare:

- PPL or diffusion output metrics.
- aggregate MSE.
- selector counts.
- margin distributions.
- homogeneity.
- regret.
- chosen permutations/transforms where deterministic.

Do not require bitwise identity, but investigate decision differences concentrated around low-margin blocks.

---

# 22. Required ablations

At minimum:

1. format granularity sweep.
2. N-only vs K-only diagnostic.
3. count homogeneity vs margin-weighted conflict.
4. MSE selector vs output-aware selector.
5. calibration size.
6. E0-ratio sort vs margin clustering vs greedy regret packing.
7. ordinary rotation objective vs granularity-aware objective.
8. weight-only adaptive vs activation-only adaptive vs both.
9. standard scaling vs canonical 4Over6.
10. MixFP4 vs validated MixFP4+4Over6 composition.
11. Ours without 4Over6 vs Ours on top of 4Over6 composition.
12. diffusion timestep stability.
13. cross-model family.
14. cross-domain LLM vs diffusion.
15. cross-GPU numerical stability.

---

# 23. Artifact policy

Everything must live under:

```text
artifacts/
```

Recommended structure:

```text
artifacts/
├── 00_environment/
│   ├── environment.txt
│   ├── pip_freeze.txt
│   ├── gpu_inventory.json
│   ├── gpu_usage_log.jsonl
│   ├── repo_manifest.json
│   ├── patch_manifest.json
│   ├── dataset_manifest.json
│   ├── model_manifest.json
│   └── diffusion_eval_manifest.json
│
├── 01_repo_audit/
│   ├── nvfp4_razer_audit.md
│   ├── fouroversix_audit.md
│   ├── deepcompressor_audit.md
│   ├── sana_audit.md
│   ├── quantization_semantics.md
│   ├── four_over_six_composition.md
│   └── implementation_mapping.md
│
├── 02_tests/
│   ├── unit_test_report.txt
│   ├── codebook_validation.json
│   ├── granularity_invariants.json
│   ├── permutation_equivalence.json
│   ├── rotation_equivalence.json
│   ├── four_over_six_reduction.json
│   └── cross_repo_validation.md
│
├── 03_phase_a/
│   ├── llm/
│   │   ├── raw/
│   │   ├── per_layer/
│   │   ├── ppl/
│   │   ├── downstream/
│   │   └── plots/
│   ├── diffusion/
│   │   ├── raw/
│   │   ├── proxy/
│   │   ├── images/
│   │   ├── metrics/
│   │   └── plots/
│   └── summaries/
│
├── 04_phase_b/
│   ├── selector/
│   ├── permutation/
│   ├── rotation/
│   ├── timestep/
│   ├── combined/
│   └── plots/
│
├── 05_cross_gpu/
│   ├── llm_a6000_vs_6000ada.csv
│   ├── diffusion_a6000_vs_6000ada.csv
│   └── reproducibility_report.md
│
└── 06_final/
    ├── master_results.csv
    ├── llm_results.csv
    ├── diffusion_results.csv
    ├── per_layer_metrics.csv
    ├── format_region_metrics.csv
    ├── timestep_metrics.csv
    ├── experiment_manifest.jsonl
    ├── failed_runs.jsonl
    ├── results_summary.md
    ├── decision_report.md
    ├── go_no_go.json
    ├── phase_c_handoff.md
    ├── limitations.md
    └── reproduction_commands.sh
```

Never overwrite a prior run without preserving provenance.

Every experiment must have:

- stable `experiment_id`.
- config snapshot.
- exact command.
- status.
- stdout/stderr log.
- raw metrics.
- summary row.
- GPU info.
- start/end time.
- code commit/patch hash.
- model revision.
- dataset/prompt manifest.

---

# 24. Mandatory machine-readable tables

## 24.1 `master_results.csv`

At minimum:

```text
experiment_id
phase
domain
model
model_revision
dataset_or_promptset
quantization_mode
weight_format_mode
activation_format_mode
weight_format_granularity
activation_format_granularity
scale_rule
scale_group_size
selector
permutation
rotation
calibration_size
four_over_six_mode
gpu_index
gpu_type
ppl
ppl_delta_vs_nvfp4
oracle_gain_retention
image_reward
lpips
psnr
fid
proxy_nmse
status
```

Use empty/NA fields where a metric does not apply.

## 24.2 `per_layer_metrics.csv`

At minimum:

```text
experiment_id
domain
model
layer_idx
module_name
module_type
N
K
format_granularity
num_scale_blocks
num_format_regions
e0_ratio
e2_ratio
mean_homogeneity
mean_margin_conflict
mean_format_margin
oracle_mse
constrained_mse
granularity_regret
sensitivity_regret
nmse
relative_l2
cosine_error
```

## 24.3 `format_region_metrics.csv`

One row per coarse region when storage permits:

```text
experiment_id
layer
module
region_n_start
region_k_start
region_n_size
region_k_size
num_k16_blocks
oracle_e0_count
oracle_e2_count
homogeneity
P_G
N_G
margin_conflict
oracle_error
constrained_error
regret
selected_format_mse
selected_format_sensitivity
mean_abs_margin
max_abs_margin
```

## 24.4 `timestep_metrics.csv`

For diffusion:

```text
experiment_id
model
promptset
timestep
layer
module
e0_ratio
selector_agreement_vs_reference_step
mean_margin
mean_homogeneity
mean_margin_conflict
granularity_regret
sensitivity_regret
proxy_nmse
```

---

# 25. Required plots

Generate at minimum:

```text
ppl_vs_granularity
relative_ppl_delta_vs_nvfp4
oracle_gain_retention_vs_granularity
regret_vs_granularity
n_conflict_vs_k_conflict
homogeneity_vs_regret
margin_conflict_vs_regret
regret_vs_ppl_delta
sensitivity_regret_vs_ppl_delta
e0_ratio_by_layer
margin_conflict_by_layer
n8k64_regret_layer_heatmap
selector_comparison
permutation_comparison
rotation_ppl_vs_regret
rotation_homogeneity_vs_regret
combined_ours_comparison
four_over_six_composition_comparison
llm_cross_model_comparison
sana_proxy_vs_granularity
sana_image_metrics_vs_granularity
sana_timestep_preference_stability
sana_timestep_regret
a6000_vs_6000ada_numerical_comparison
```

Every plot must have a CSV/JSON source table.

---

# 26. Required result summary tables

`artifacts/06_final/results_summary.md` must include actual numbers for:

1. HP vs NVFP4 vs canonical 4Over6 vs E0 vs Oracle16.
2. LLM PPL across granularity.
3. N8K16/K64/N8K64 conflict decomposition.
4. N8K64 Oracle gain retention.
5. W4A4 one-side/two-side results.
6. SANA W4A16 proxy/image results.
7. SANA W4A4 selected results.
8. diffusion timestep stability.
9. MSE vs sensitivity selector.
10. permutation variants.
11. rotation variants.
12. best Ours.
13. MixFP4+4Over6 composition validity and results.
14. MixFP4+Ours vs MixFP4+4Over6+Ours.
15. cross-model/domain generalization.
16. A6000 vs 6000 Ada numerical consistency.

Do not provide narrative only.

---

# 27. Go/No-Go logic

The final report must not force a positive story.

## Gate A — Oracle value

Does `MixFP4-Oracle16` meaningfully improve NVFP4?

If no, recommend stopping the E2/E0 adaptive direction.

## Gate B — coarse retention

How much Oracle gain survives at N8K64 and M16K64?

Report exact retention plus raw metrics.

## Gate C — source of loss

Is regret primarily:

- N-direction?
- K-direction?
- selector-objective mismatch?
- timestep instability?
- unstructured/noisy?

## Gate D — recovery

Does `Ours` recover a large and reproducible fraction of the lost end-to-end benefit?

## Gate E — stronger-scale complementarity

If MixFP4+4Over6 composition is valid, does Ours remain useful on top of it?

This is important: a method that only fixes a weak scale baseline is less compelling.

## Gate F — generalization

Does the direction hold across:

- Llama-3.1-8B.
- Qwen3-8B.
- SANA-1.6B.
- WikiText/C4 and fixed diffusion promptsets.

## Strong GO to Phase C

Recommend RTX 5090 / SM120 native work if:

- Oracle16 is meaningfully better than NVFP4 in multiple settings; and
- coarse granularity retains substantial gain or Phase B reliably recovers it; and
- the result is not isolated to one layer/model; and
- at least one deployable method (`Ours-S`, foldable `Ours-P`, or a foldable/low-overhead transform) is responsible; and
- cross-repo/cross-GPU semantics are stable.

## Conditional GO

Oracle benefit is real, raw coarse granularity loses much of it, but one Phase B method consistently recovers a large fraction. This may be the strongest research story, but native feasibility still needs Blackwell.

## No-Go

Recommend stopping/redirecting if:

- Oracle16 gain is negligible/inconsistent; or
- coarse granularity destroys nearly all gain and Phase B cannot recover it; or
- recovery only appears with upper-bound-only operations; or
- results are fragile to model/dataset/timestep; or
- validated canonical 4Over6 already dominates the method without meaningful complementarity.

---

# 28. `go_no_go.json` schema

Populate from measurements only:

```json
{
  "oracle16_has_meaningful_gain": null,
  "llm_weight_n8k64_gain_retention": null,
  "llm_activation_m16k64_gain_retention": null,
  "diffusion_weight_n8k64_gain_retention": null,
  "diffusion_activation_m16k64_gain_retention": null,
  "dominant_conflict_axis_llm": null,
  "dominant_conflict_axis_diffusion": null,
  "sensitivity_selector_recovers_gap": null,
  "permutation_recovers_gap": null,
  "rotation_recovers_gap": null,
  "best_ours_method": null,
  "four_over_six_composition_valid": null,
  "ours_complements_four_over_six": null,
  "preferred_adaptive_operand_llm": null,
  "preferred_adaptive_operand_diffusion": null,
  "diffusion_timestep_stability": null,
  "cross_model_generalization": null,
  "cross_domain_generalization": null,
  "cross_repo_validation": null,
  "cross_gpu_numerical_stability": null,
  "recommended_phase_c": null,
  "recommended_design": null,
  "confidence": null,
  "blocking_issues": []
}
```

---

# 29. Phase C handoff (do not execute on A6000/Ada)

Create:

```text
artifacts/06_final/phase_c_handoff.md
```

It must specify:

- exact recommended format semantics.
- scale rule/granularity.
- weight and activation format regions.
- whether only one operand should be adaptive.
- selector.
- permutation/transform.
- metadata requirements.
- expected fake/reference accuracy metrics.
- exact LLM and diffusion test cases.
- numerical tolerances.
- what must be validated on RTX 5090 / SM120.
- likely integration anchor in FourOverSix/CUTLASS/Nunchaku-style infrastructure.

Do not claim native speedup before Phase C.

---

# 30. Execution order

Follow this order unless a documented blocker requires reordering:

```text
0. repo/environment audit
1. GPU guard
2. baseline reproduction
3. unit/semantic tests
4. smoke Oracle16/K64/N8K16/N8K64
5. Phase A0 N/K decomposition
6. full LLM W4A16 granularity sweep
7. SANA W4A16 granularity/proxy/image screening
8. canonical 4Over6 + composition validation
9. W4A4 one-side/two-side LLM study
10. SANA W4A4 selected study
11. Ours-S sensitivity selector
12. Ours-P margin-aware permutation
13. Ours-R rotation/transform only where justified
14. diffusion timestep stability
15. combined Ours finalists
16. MixFP4+4Over6+Ours matched endpoints if valid
17. finalist LLM downstream tasks
18. finalist diffusion 1024-image evaluation
19. secondary diffusion model if feasible
20. cross-repo validation
21. cross-GPU validation
22. aggregation/plots
23. strict decision report
24. Phase C handoff
```

Do not jump directly to rotation before identifying N-vs-K conflict.

---

# 31. Failure handling

Every failed job must be preserved.

Record:

- command.
- config.
- stdout/stderr.
- exit code.
- GPU state.
- timestamp.
- classification.

Classifications:

```text
code_bug
OOM
model_access
dataset_or_network
dependency
numerical_issue
GPU_became_occupied
unsupported_semantics
runtime_excessive
unknown
```

Fix and rerun when safe, but keep the failed attempt in:

```text
artifacts/06_final/failed_runs.jsonl
```

Never take over an occupied GPU.

---

# 32. Completion criteria

The task is not complete because code exists.

Completion requires:

- repos audited and pinned.
- GPU guard implemented and used.
- baselines reproduced.
- semantic/unit tests passed or blockers documented.
- Llama-3.1-8B Phase A/B completed to the extent supported.
- Qwen3-8B Phase A/B completed to the extent supported.
- SANA-1.6B Phase A/B completed to the extent supported.
- N/K conflict decomposition complete.
- W4A16 granularity sweep complete.
- W4A4 one-side/two-side study complete for required configs.
- Ours-S tested.
- Ours-P tested.
- Ours-R tested when diagnostic evidence justifies it.
- diffusion timestep stability analyzed.
- canonical 4Over6 baseline included.
- MixFP4+4Over6 composition either validated and tested or explicitly rejected with evidence.
- best Ours compared both without and, when valid, with 4Over6.
- cross-GPU checks complete.
- all artifacts populated.
- actual plots/tables generated.
- strict Go/No-Go decision written.
- Phase C handoff written.

If an experiment is impossible due to a real external blocker, finish all independent work and document the blocker precisely. Partial silent skipping is not allowed.

---

# 33. Expected scientific outcomes

## Outcome 1 — Granularity is already cheap

N8K64/M16K64 retain most Oracle16 benefit.

Then the main contribution becomes demonstrating that coarse hardware selection is sufficient; Phase C can focus on native execution.

## Outcome 2 — Granularity is a barrier, but co-design recovers it

Raw coarse MixFP4 loses substantial accuracy, but selector/permutation/transform recovers most of the loss.

This is likely the strongest story:

> Fine-grained format adaptation can be compressed into coarse hardware-format regions by granularity-aware selection and representation shaping.

## Outcome 3 — Better scaling plus Ours is complementary

Canonical 4Over6 improves the base scale behavior, while Ours independently reduces format-conflict regret.

This is especially strong because the proposed method remains useful on top of a stronger quantization baseline.

## Outcome 4 — No robust recovery

Oracle16 is good, but coarse hardware regions eliminate the benefit and deployable Phase B methods cannot recover it.

Recommend against spending RTX 5090 kernel-development resources on this exact direction.

---

# 34. References / implementation anchors

1. NVFP4-RaZeR  
   https://github.com/abdelfattah-lab/NVFP4-RaZeR

2. FourOverSix / Adaptive Block-Scaled Data Types  
   https://github.com/mit-han-lab/fouroversix

3. DeepCompressor / SVDQuant  
   https://github.com/nunchaku-ai/deepcompressor

4. SANA  
   https://github.com/NVlabs/Sana

5. Nunchaku  
   https://github.com/nunchaku-ai/nunchaku

Use exact pinned commits in all experimental artifacts.
