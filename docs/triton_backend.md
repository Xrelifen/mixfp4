# The Triton backend, and what Triton can and cannot do here

`python/mixfp4/` is a PyTorch-facing quantisation and inference package: a pluggable weight
quantiser, a packing format shared with the CUDA path, and three Triton kernels. It exists to make
the format usable from a model, which the CUDA executables were never able to be.

Read this first, because it bounds everything below.

## Triton cannot reach the E0M3 tensor core

E0M3 is not a data format you can request. It is **bits 14:15 of the compiled `OMMA` SASS
instruction**, which is why this repository reaches it by disassembling a cubin and patching bytes
(`scripts/patch_mixed_nvfp4_gemm.py`). Three consequences:

- PTX has no `e0m3` token, so `tl.inline_asm_elementwise` cannot express it either.
- Triton exposes no hook between its cubin and the driver, so there is nowhere to patch.
- No arithmetic identity maps one codebook onto the other. E0M3's `{0..7}` and E2M1's
  `{0, .5, 1, 1.5, 2, 3, 4, 6}` agree only up to index 4 under any single scale factor, so you
  cannot emulate one as the other times a constant.

**So no Triton kernel can use the mixed-format tensor core.** `tl.dot_scaled` handles E2M1 and is
the right kernel for pure NVFP4, but it has no E0M3 to offer.

What these kernels do instead is decode both codebooks in registers and run an ordinary fp16/bf16
`tl.dot`. That makes them **A16W4** (16-bit activations, 4-bit weights) — a *different operator*
from the CUDA path's **A4W4**, serving a different regime. They will never reproduce the
1165 TFLOP/s in `docs/mixed_nvfp4_report.md`.

The regime they do serve is the one that matters most for serving: at batch 1–4, decode is bound
by reading K·N/2 weight bytes, and the FP4 tensor core buys nothing at all. The CUDA kernel's
advantage is real only at prefill and large batch.

## What software decode buys: granularity

Both codebooks live in one 32-entry table, `[E2M1_CODEBOOK, E0M3_CODEBOOK]`, and a group's format
flag becomes an offset into it:

```python
value = tl.gather(lut, nibble + 16 * fmt, axis=1)
```

The per-granule format selection that cost the CUDA path an entire dispatch-tree architecture — a
specialised copy of the whole mainloop per flag pattern, an 8-arm budget, and a hard codegen cliff
past 16 arms — is, here, **one integer add**.

The consequence is granularity. The CUDA kernel's format granule is bounded below by one MMA
atom's footprint; the Triton path selects per *scale group*, 16 elements of K by one output column,
which is the finest the format can express. `quantize_mixfp4(..., granule_n=, granule_k=)` coarsens
the decision by pooling error over a block so one checkpoint can serve both backends.

The format tag itself is bit 7 of the UE4M3 scale byte — the same channel as `MIXFP4_FLAG_MASK`,
free because UE4M3 is unsigned and the hardware masks the byte to 7 bits. Masking that bit off and
bitcasting to `float8_e4m3fn` *is* the scale decode, since the fields and bias are identical.

**This is specific to NVFP4.** MXFP4's UE8M0 scale uses all eight bits for the exponent, so there
is no spare bit and this tagging scheme does not carry over.

## Kernels

Dispatch by batch size follows GemLite's thresholds, for the same reasons.

| M | kernel | shape |
|---|---|---|
| ≤ 4 | `gemv` | no `tl.dot`; one row, one N slice, one K slab per program, atomic accumulate |
| 5–63 | `gemm_splitK` | too few output tiles to fill the machine, so split the K reduction |
| ≥ 64 | `gemm` | enough tiles; plain tiled matmul |

Two deliberate departures from GemLite:

**No `gemv_revsplitK`.** GemLite's fastest bs=1 kernel unrolls two K stages sharing one scale load,
which needs `BLOCK_K * 2 <= group_size`. NVFP4's group size is 16, so that caps `BLOCK_K` at 8 and
costs more in launch count than it saves in metadata traffic. A real technique this group size
does not admit.

**Atomic reductions accumulate in fp32**, always, then cast once. Rounding each partial sum into a
bf16 output loses ~5× accuracy (measured: 7.1e-3 relative error versus 1.7e-3) and fp16 ~6×.
At the M these kernels serve the fp32 buffer is tens of KB against megabytes of weight traffic.
After this change all three kernels produce *identical* error — 2.05e-04 (fp16), 1.65e-03 (bf16) —
independent of M, N, K and kernel, confirming the residual is dequant rounding alone.

One consequence: **results are not bitwise reproducible across launches**, because atomic
accumulation order varies. Tests compare with tolerance, not equality.

## Verification

`tests/triton/test_mixfp4_triton.py` — 17 tests, all passing on an RTX 5090 (sm_120), torch 2.9.1,
Triton 3.5.1.

The load-bearing ones cross-check the *CUDA path's* conventions by importing
`tests/mma_intrinsics/expected_value.py`, the independent reference used to validate the patched
hardware instruction. Codebooks match; UE4M3 decode matches for every finite code. If these ever
diverge, a checkpoint quantised for one backend is being misread by the other.

Kernel accuracy against a dequantised reference, across `(M, N, K)` from (1, 256, 512) to
(128, 256, 1024), fp16 and bf16, and four tagging patterns (all-E2M1, all-E0M3, random, and
error-driven):

| dtype | relative error |
|---|---|
| fp16 | 2.05e-04 |
| bf16 | 1.65e-03 |

Random tagging matters: uniform tagging is correct by accident under a permuted layout — a
permutation of a constant is that constant — which is how the CUDA path's granule-mapping bug
survived undetected.

### A bug found in the repo's own reference

`tests/mma_intrinsics/expected_value.py` decodes UE4M3 code `0x7f` as 480.0. CUTLASS's
`float_ue4m3_t` documents the range as `[0:448]` with `has_NaN: true` and defines
`isnan(x) { return x.storage == 0x7f; }`; torch's `float8_e4m3fn` agrees. **0x7f is NaN, not 480.**
Unreachable in practice — the quantiser clamps scales to 448 so `0x7f` is never emitted — but the
reference is wrong, and it is the file the hardware tests are validated against.

## Gap analysis: is this enough for quantised model inference?

Present, working:

- weight quantiser with pluggable methods, per-group format selection, coarsenable granule
- packing and scale layout shared with the CUDA path
- three kernels covering decode through prefill, with autotune, a config pruner, and a JSON cache
- `MixFP4Linear`, a `torch.library.custom_op` forward that survives `torch.compile(fullgraph=True)`,
  state-dict round-trip, and `patch_model` to swap an `nn.Linear` model in place
- perplexity harness (`scripts/eval_perplexity.py`)

Missing, in rough order of how much it blocks real serving:

1. **No tuned config cache is shipped.** The first call to each shape autotunes. In `fast` mode
   that is ~200 configs and minutes of compilation. GemLite ships per-GPU JSON files for exactly
   this reason; `mixfp4/configs/` is wired up but empty. A tuning script that populates it for the
   shapes of a target model is the single highest-value missing piece.
2. **No performance numbers.** The GPU was 99% busy with another job for this entire session, so
   nothing here is benchmarked. Correctness is measured; speed is not. Until that is done there is
   no evidence these kernels beat, say, dequantising to bf16 and calling cuBLAS.
3. **No fused MoE.** All three kernels are strictly 2-D. Any MoE model needs a grouped/batched
   GEMM with an expert-index argument. GemLite has the same gap.
4. **No activation quantisation.** Weight-only (A16W4) only. A4W4 or A8W4 needs dynamic per-token
   activation quantisation, and for the mixed format specifically it is unclear it is even
   meaningful — see below.
5. **No CUDA backend binding.** The fast kernel is still only reachable from a standalone
   executable. Getting it into PyTorch means solving the post-compile SASS patch inside a build
   system: ship a pre-patched fatbin, or run `scripts/patch_mixed_nvfp4_gemm.py` as a build step.
   Until then the repo's performance thesis cannot be exercised from a model.
6. No tensor-parallel sharding, no vLLM/SGLang integration, no GGUF or compressed-tensors loading.

### The A4W4 question

For A4W4 both operands would be FP4, and if either carries E0M3 the Triton path must dequantise it
anyway — at which point the format has bought nothing and cost a decode. Mixed-format A4W4 is only
worth doing on the CUDA path, where the tensor core reads the nibbles directly. That is an argument
for item 5 above being the real priority if prefill throughput is the goal, and for the Triton path
staying deliberately scoped to decode.

## Environment

Nothing here is wired into `CMakeLists.txt`; the package is pure Python.

```bash
python -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
./venv/bin/pip install transformers datasets      # only for scripts/eval_perplexity.py

PYTHONPATH=python python tests/triton/test_mixfp4_triton.py
MIXFP4_AUTOTUNE=default python tests/triton/test_mixfp4_triton.py   # skip the full config sweep
```

Environment variables: `MIXFP4_AUTOTUNE` (`default`/`fast`/`max`), `MIXFP4_FORCE_MATMUL`
(`gemv`/`gemm_splitK`/`gemm`, to pin one kernel for benchmarking).
