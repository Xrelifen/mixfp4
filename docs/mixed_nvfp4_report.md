# Mixed E0M3/E2M1 block-scaled NVFP4 GEMM on SM120

A single GEMM kernel in which each operand block is decoded as either **E2M1** (standard NVFP4)
or **E0M3** (sign-magnitude INT4), chosen per block at runtime, on a GeForce RTX 5090 (sm_120a).

The kernel had been working but running at **504 TFLOP/s against stock NVFP4's 1207** — a 2.4x
regression. This report covers what was actually causing that, why six previous attempts to fix
it all failed, the fix, and the correctness and throughput results.

**Result: 1.0%–4.9% overhead versus stock NVFP4 across shapes, and 2.3x–3.6x FP8 throughput.**

---

## 1. The problem

The format is not data. It is two bits (14:15) of the **compiled SASS instruction encoding**, so a
single `mma.sync` has one A-format and one B-format for its entire footprint. Selecting a format
at runtime therefore means selecting among distinct *instructions*, which means a branch — and
PTX has no `e0m3` token at all, so the four variants are produced by patching the compiled cubin
(`scripts/patch_mixed_nvfp4_gemm.py`).

The natural implementation puts a four-way branch inside the MMA atom's `fma()`. That is what cost
2.4x, and the reason is one hardware fact:

> **A predicated-off OMMA still consumes a tensor-pipe issue slot.**

The stock SM120 NVFP4 mainloop is tensor-pipe bound — ncu reports `sm__inst_executed_pipe_tensor`
at **84% of peak** with `math_pipe_throttle` as the dominant warp stall. Throughput is therefore
close to a linear function of how many OMMA instructions are *issued*, useful or not.

ptxas runs its own if-conversion pass over whatever PTX it receives, and a branch arm containing a
lone OMMA is a textbook if-conversion candidate. So it folded the choice into `@P0 OMMA` /
`@!P0 OMMA` pairs no matter how the branch was written. Measured directly:

| | stock NVFP4 | mixed (per-mma branch) |
|---|---|---|
| OMMAs in SASS | 64, all unpredicated | 256, **128 `@P0` + 128 `@!P0`, zero unpredicated** |
| `sm__inst_executed_pipe_tensor` | 8,388,608 | **16,777,216 — exactly 2x** |
| dominant warp stall | `math_pipe_throttle` 4.42 | `wait` 3.96, `branch_resolving` 1.04 |

One wasted tensor issue for every useful one: half the machine, before counting the branch itself.

### Why this was hard to see

The investigation before this one had chased *synchronization*, which looked compelling: the
mixed kernel had 264 `WARPSYNC` instructions against the baseline's 3. Annotating the branches
`bra.uni` (the PTX ISA's explicit non-divergence assertion) dropped WARPSYNC from 264 to exactly 3
— a complete fix of the thing being measured — and throughput moved from 508 to 504 TFLOP/s.
WARPSYNC was a symptom, never the cost.

The tensor-issue model explains every prior data point retroactively, including ones that looked
anomalous at the time:

| encoding | TFLOP/s | explanation under the model |
|---|---|---|
| flat 4-way compound predicates | 357 | no outer real branch leaves **four** predicated OMMAs; 1207/357 ≈ 4x |
| C++ if/else (4 asm blocks) | 504 | outer branch real, inner if-converted → 2x |
| hand-written PTX `bra.uni` | 504 | same 2x; WARPSYNC removal is irrelevant |
| trivially-uniform `blockIdx` | 604 | still 2x, minus the flag-extraction cost |
| `__shfl_sync`-proven uniform | 412 | 2x plus two `SHFL.IDX` per site |
| `brx.idx` indirect jump table | 296 | **1x tensor issues** — if-conversion fully defeated — but a constant-bank `LDC` + `BRX` + `WARPSYNC.ALL` per MMA |
| `ptxas --allow-expensive-optimizations=false` | 470 | predicates all four instead of two |

The `brx.idx` row is the decisive one. It is the only per-MMA encoding that restored
`sm__inst_executed_pipe_tensor` to **exactly 8,388,608**, confirming the model — and it is also
the most expensive kind of branch the hardware has. That combination is what rules out the entire
per-MMA approach.

### Arm size is not the lever

The obvious next move is to make the branch arms big enough that if-conversion becomes
unprofitable. It does not work. ptxas if-converts the innermost diamond regardless of size:

| arms per branch | result |
|---|---|
| 1 OMMA per arm | if-converted |
| 16 OMMAs per arm (one whole `cute::gemm`) | if-converted — all 16 `@!P0`, all 16 `@P0` |
| 32 OMMAs per arm | if-converted — 256 of 512 OMMAs predicated |

Hoisting the branch around a single `cute::gemm` therefore only reached 596–600 TFLOP/s.

---

## 2. The fix

What ptxas will *not* if-convert is a region containing the pipeline's barriers and shared-memory
stage bookkeeping. So the dispatch wraps **an entire k_tile iteration**: each arm is a full copy of
the mainloop body — both k_blocks' smem→rmem copies, the named barrier, the consumer
release/acquire, and 32 OMMAs. The branch is paid once per k_tile, and CUTLASS's register-level
software pipeline (copy k_block+1 while multiplying k_block) stays intact *inside* each arm rather
than being flattened.

Each arm is specialized at compile time on the whole **pattern** of format flags across the warp's
footprint, so no branch remains inside an arm — every MMA's atom is statically known.

Resulting codegen: **512 OMMAs, zero predicated**, `sm__inst_executed_pipe_tensor` back to
8,388,608 (1x), registers unchanged at 168, no spills, `math_pipe_throttle` restored as the
dominant stall.

---

## 3. Correctness

Nothing had previously verified a genuinely mixed result — correctness was only ever checked
against CUTLASS's stock block-scaled reference on untagged data, which exercises one of the four
sites and nothing else. Making that test real surfaced three separate bugs.

**A. No E0M3-aware reference existed.** CUTLASS's `Gemm3x` decodes every operand as E2M1 and reads
the whole scale byte as a UE4M3 magnitude; here bit 7 is the format tag and a tagged granule
decodes under E0M3. The two formats index the same nibble — E2M1 as `{0, .5, 1, 1.5, 2, 3, 4, 6}`
with a sign bit, E0M3 as the equal-spaced signed integers `0..7` (confirmed on this hardware in
`3rdparty/sm120-e0m3-mma/RESULTS.md`) — so recovering the nibble from an E2M1-decoded value and
re-reading it under E0M3 reproduces the patched instruction exactly.

**B. The host did not know where a format granule lives.** It assumed atom *j* covers rows/columns
`[8j, 8j+8)`. That is false: the TiledMma carries a `PermTileN` that permutes N, and a warp's atoms
are strided, not contiguous. **The real granules are not contiguous blocks** —

- an **A** granule is two 16-row blocks **64 rows apart** (e.g. rows 0–15 together with 64–79)
- a **B** granule is two 16-column blocks **32 columns apart** (e.g. cols 0–15 with 32–47)

This produced correct results for every *uniform* tagging (a permutation of a constant is that
constant) while corrupting every genuinely mixed one — which is precisely why it had gone
unnoticed. The map is now derived by partitioning an identity tensor with the same TiledMma the
kernel uses, sized from the CTA tile.

**C. The SASS patcher mis-attributed sites.** Its PRMT regex rejected `.reuse` operands, and a
declined match did not stop the backward scan — it ran on to an older PRMT writing the same
register and silently assigned those OMMAs to the wrong format (a 128/132/128/124 census where all
four must be 128). It now finds the nearest writer of the SFA register, whatever it is, and
insists that be a tagged PRMT, failing loudly otherwise.

> **Tagging finer than a granule is not merely inaccurate.** Below one atom, lanes of a warp
> disagree, the warp splits across arms, and `mma.sync.aligned` runs partially converged. That
> hung the GPU. Building with `-DMIXFP4_DEBUG_UNIFORMITY=1` catches it (it reported agreement mask
> `0x55555555` for an 8-row tagging, matching SFALayout's thread→row mapping exactly).

### Verified

All against the E0M3-aware reference, on the patched binary, compared by relative Frobenius norm
(the GPU sums K in a different order and rounds to bfloat16, so near-cancelling elements show
large element-wise relative error while being correct; a mis-decoded granule moves the norm by
order 1, not 1e-3).

| tagging | sites exercised | 1024³ rel. error | |
|---|---|---|---|
| none | 0 (E2M1×E2M1) | 0.00178 | PASSED |
| all-A | 1 (E0M3×E2M1) | 0.00178 | PASSED |
| all-B | 2 (E2M1×E0M3) | 0.00178 | PASSED |
| all | 3 (E0M3×E0M3) | 0.00178 | PASSED |
| random per granule | **all four simultaneously** | 0.00172 | PASSED |

Random tagging across shapes: 256³, 512³, 1024³, 2048³, and non-square 1024×2048×512 — all PASSED
(rel. error 0.0017–0.0017).

Negative control: the *unpatched* binary with E0M3 tags fails at 0.75 relative error, confirming
the test actually discriminates.

---

## 4. Throughput

RTX 5090, verified-idle GPU, best of 3 runs, mixed kernel randomly tagged so all four format sites
are live. TFLOP/s.

| M × N × K | FP8 | NVFP4 | **mixed** | overhead vs NVFP4 | vs FP8 |
|---|---|---|---|---|---|
| 1024³ | 119.4 | 282.6 | 274.5 | 2.9% | 2.30× |
| 2048³ | 271.4 | 797.4 | 777.4 | 2.5% | 2.86× |
| 4096³ | 340.0 | 1206.7 | 1165.9 | 3.4% | 3.43× |
| 8192³ | 392.8 | 1401.8 | 1387.6 | **1.0%** | 3.53× |
| 4096×4096×16384 | 351.6 | 1289.5 | 1254.0 | 2.7% | 3.57× |
| 8192×8192×2048 | 375.0 | 1258.1 | 1195.9 | 4.9% | 3.19× |
| 16384×16384×2048 | 388.7 | 1313.8 | 1249.0 | 4.9% | 3.21× |

Overhead is largest on short-K shapes (K=2048), where the per-k_tile dispatch amortizes over fewer
iterations — the expected shape of the cost.

> A shared GPU makes these numbers fragile: with another process resident, stock `nvfp4_gemm`
> itself read 852 instead of 1208. Both `scripts/sweep.sh` and `scripts/bench_all.sh` refuse to run
> unless the card is idle.

---

## 5. Granularity, and what it costs

Arms are specialized on the full flag pattern across a warp's footprint, so the arm count grows as
`2^(A granules) × 2^(B granules)`. Measured at 4096³:

| granule (A rows × B cols × K) | arms | OMMAs | codegen | TFLOP/s |
|---|---|---|---|---|
| 32 × 64 × 128 | 4 | 256 | clean | 1184.8 |
| **32 × 32 × 128 (default)** | **8** | **512** | **clean** | **1165.9** |
| 16 × 32 × 128 | 16 | 1024 | `CALL.REL.NOINC`, `STACK:912` | 36.7 |
| 16 × 16 × 128 | 64 | 4096 | `CALL.REL.NOINC`, `STACK:912` | 32.3 |

**There is a hard cliff between 8 and 16 arms.** Past it, cicc stops inlining the specialized body
and outlines it into a real ABI function call, spilling the accumulators to a 912-byte stack frame
— the same failure mode as the original `noinline` experiment. `-Xcicc -inline-threshold=2000000`
does not move it.

The 16×16 configuration is **numerically correct** (it passes the full random-tagging test) but
36× slower, so it is not usable.

**So 32 rows × 32 columns × 128 K is the finest symmetric granule that survives**, and it is the
default. Note also that 16×16 is the hardware floor regardless of the compiler, because the atom
is `m16n8k64` — A can never go below 16 rows.

Configurable via `-DMIXFP4_A_ATOMS_PER_GRANULE` / `-DMIXFP4_B_ATOMS_PER_GRANULE` (in atoms; A
atoms are 16 rows, B atoms 8 columns).

---

## 6. Using it

```bash
# build (needs a configured CUTLASS build tree for its generated headers)
./scripts/build_mixed.sh build/mixed_nvfp4_gemm

# install the real E0M3 formats -- WITHOUT THIS the kernel is plain NVFP4,
# since PTX cannot spell e0m3 and all four sites compile as E2M1 x E2M1
python3 scripts/patch_mixed_nvfp4_gemm.py build/mixed_nvfp4_gemm build/mixed_patched

./build/mixed_patched 4096 4096 4096          # MIXFP4_TAG=random by default
MIXFP4_TAG=none|a|b|all ./build/mixed_patched 1024 1024 1024
MIXFP4_SKIP_REF=1 ./build/mixed_patched 8192 8192 8192   # skip the O(M*N*K) host reference

./scripts/bench_all.sh                        # the table in section 4
```

Build-time options: `-DMIXFP4_DEBUG_UNIFORMITY=1` (catch tagging finer than a granule),
`-DMIXFP4_NO_DISPATCH=1` (compile the dispatch out — the in-source performance ceiling).

---

## 7. Limitations

- **The binary must be patched.** Unpatched it computes ordinary NVFP4, silently and correctly.
- **The format tag consumes bit 7 of every UE4M3 scale byte.** Architecturally ignored by the
  tensor core (verified on hardware in `tests/mma_intrinsics`), so it costs no storage or
  bandwidth — but it is not free if some other consumer of those scale factors reads that bit.
- **The granule is not a contiguous tile** (section 3B). Host-side quantization must respect the
  strided shape, and violating it below atom granularity hangs the GPU rather than returning wrong
  numbers.
- **The granule is tied to the tile shape and warp layout.** Change `ThreadBlockShape` or the
  AtomLayout and the granule changes with it; the map is derived automatically, but the *arm
  count* — and therefore the cliff in section 5 — must be rechecked.
- Only `ue4m3` scale factors are exercised. `ue8m0` and `scale_vec::2X` are untested here (the
  latter has a known pre-existing E0M3 hardware limitation, unrelated to this work).
- E0M3 semantics rest on an undocumented, patched instruction encoding. It is validated
  numerically on one GPU and could change on other silicon or with a different disassembler.
