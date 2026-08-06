# Research 3 repository audit

Audit time: 2026-08-07 04:49:03 +08:00
Repository path: `/home/JaaaaaA_l/mixfp4_2`
Research specification SHA256: `1c633d5f9828d520971a711d4176f02f2cb67b81a66223fa259139dcb55917f1`

## Gate result first

The host has four SM86 RTX A6000 GPUs and three SM89 RTX 6000 Ada GPUs. It has **zero SM120
devices**. Per `research_3.md`, the formal decision is therefore `INCOMPLETE` with blocking issue
`SM120 GPU unavailable`. No A6000/Ada QDQ or simulation result is used as native evidence.

## Git state

- Audit input commit: `ce62564a8d271343fbec4f7ed8452f1748faed1d`.
- Branch on entry: `exp/ccdt-fp4-feasibility`.
- Local-only Research 3 branch: `exp/native-mix-sm120-research-3`.
- `git remote -v`: empty; no remote is configured or modified.
- The working tree was already heavily dirty: every tracked kernel, test, script, doc, and both
  submodule paths appeared deleted. Untracked `NVFP4-RaZeR/`, `AngelSlim/`, and research documents
  were present. These pre-existing changes were not reset, restored, cleaned, staged, or overwritten.
- Requested `main` branch was not present locally. The exact HEAD object was audited with
  `git show HEAD:<path>` so the deleted working-tree files did not need restoration.
- Submodules are uninitialized/deleted in the worktree:
  - CUTLASS gitlink `e64a9136dd929639e5f7c969fe5af3bf7415cd4f`.
  - `sm120-e0m3-mma` gitlink `8b755d9ef43b963150b6da453d10d359ebc14b1d`.
- The root repository has no tracked license file and no configured remote. No source was copied
  from either unavailable submodule in this round.

## Required-input availability

| Input | Status | Audit treatment |
|---|---|---|
| `research_3.md` | read completely | working-tree file, SHA256 pinned above |
| `repository_audit.md` | missing | searched filesystem, HEAD tree, and visible history; recorded missing |
| `related_work_positioning.md` | missing | searched filesystem, HEAD tree, and visible history; recorded missing |
| `docs/mixed_nvfp4_report.md` | read completely | read from exact HEAD object |
| main-branch kernel code | branch unavailable | audited exact HEAD kernel objects |
| cubin patcher | read completely | `scripts/patch_mixed_nvfp4_gemm.py` at HEAD |
| correctness tests | read completely | all tracked `tests/mma_intrinsics/*` at HEAD |
| Research 0–2 reports | read completely | nested official NVFP4-RaZeR artifact reports |

Missing requested documents are a documentation gap, not silently reconstructed inputs.

## Kernel and patch path

- `src/mixed_nvfp4_gemm.cu` builds the standard CUTLASS SM120 block-scaled collective and replaces
  the dispatch policy and MMA atom with local mixed-format variants.
- `src/collective/sm120_blockscaled_mma_tma_mixed.hpp` is the adapted collective mainloop;
  `mma_sm120_mixed.hpp` provides the mixed atom.
- CMake targets `sm_120a`. `scripts/build_mixed.sh` additionally hard-codes CUDA 13.1, which is not
  installed on this host.
- PTX spells only E2M1. The current patcher rewrites bits 14:15 of the second OMMA encoding word.
  Existing site mapping is: `00` E2M1×E2M1, `01` E0M3×E2M1, `10` E2M1×E0M3, and `11`
  E0M3×E0M3. This mapping is legacy evidence and remains unverified in Research 3.
- Sites are associated with PRMT selectors `0x3210`, `0x3214`, `0x3254`, and `0x3654`.
- The existing patcher checks opcode presence, baseline zero format bits, site counts, and a
  post-patch census. It does **not** yet provide all Phase 2 requirements: version allowlist,
  before/after disassembly artifacts, original/patched SHA manifest, runtime canary, and restore flow.
- The format tag is intended to occupy bit 7 of each UE4M3 scale byte. The scale magnitude uses
  bits 0–6. Research 3 did not revalidate that bit as hardware-ignored because SM120 is absent.

## Current tests and granularity claims

- Legacy probes require device major 12 and therefore fail closed on this host.
- Existing tests cover selected codebook values, formats, scales, boundaries, and random tagging;
  they do not satisfy the new exhaustive 16×16 operand truth table, 100 random cases per mode,
  accumulator variants, negative signature matrix, timing characterization, or two-device gate.
- Legacy report claims an `m16n8k64` instruction footprint and software control at several scopes.
  The actual minimum independently controllable type scope is `INCONCLUSIVE` until the Phase 1
  granularity probe runs on SM120.

## Accuracy and model path

- `python/mixfp4/` contains E2M1/E0M3 codebooks, UE4M3 scale/tag handling, weight quantization,
  activation quantization, linear replacement, and Triton software kernels.
- `scripts/eval_perplexity.py` provides a fake/software evaluation path. It is not native SM120
  evidence and was not run as a substitute.
- Neither `Qwen/Qwen3-8B` nor `meta-llama/Llama-3.1-8B` was found in the visible Hugging Face cache.
  Model download and selection were not started after the mandatory ISA gate became unavailable.

## Environment and resource use

- Host: `gpuserv4`, Linux 5.15.0-156-generic.
- Driver: 565.57.01. `nvidia-smi` advertises CUDA 12.7.
- PATH toolkit: CUDA 12.8, nvcc 12.8.93. `/usr/local/cuda` points to CUDA 11.8, creating an
  additional disassembler/toolchain ambiguity. CUDA 13.1 is absent.
- Python 3.12.9; PyTorch 2.8.0+cu128; Transformers 4.57.1; Datasets 4.4.1; Triton 3.4.0.
- SM120 GPU-hours used: 0. GPU jobs launched: 0.
- GPU 2 had a process owned by `sts_l` during the final inventory. It was not used or disturbed.

## Files added in this round

- `experiments/native_mix_research_3/`: independent CPU reference, tests, artifact generator.
- `artifacts/native_mix_research_3/`: audit, CPU expected tables, manifests, NOT_RUN ledgers, and
  the required `INCOMPLETE` decision package.

No legacy baseline source was modified.
