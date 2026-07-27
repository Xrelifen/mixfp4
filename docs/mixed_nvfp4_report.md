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

### Against cuBLAS

The comparison above is against CUTLASS kernels, which answers "what does the same library cost
without mixed formats". The more practical question is what the vendor library gives you.
cuBLAS 13.2 exposes the **same** block-scaled NVFP4 format this kernel uses
(`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` — 16-element blocks, UE4M3 scales) and does have
kernels for it on sm_120, so this is like-for-like rather than an approximation. Every algorithm
the heuristic returns is timed and the best kept, so cuBLAS is shown at its best.

| M × N × K | cuBLAS bf16 | cuBLAS fp8 | cuBLAS nvfp4 | CUTLASS nvfp4 | **mixed** | vs cuBLAS nvfp4 |
|---|---|---|---|---|---|---|
| 1024³ | 127.6 | 264.7 | 281.7 | 281.1 | 275.2 | −2.3% |
| 2048³ | 173.4 | 496.4 | 804.9 | 800.4 | 777.5 | −3.4% |
| 4096³ | 197.7 | 617.6 | 1200.5 | 1204.9 | 1164.3 | −3.0% |
| 8192³ | 205.8 | 743.4 | 1401.4 | 1400.1 | 1386.6 | **−1.1%** |
| 4096×4096×16384 | 200.1 | 654.5 | 1332.6 | 1289.1 | 1254.5 | −5.9% |
| 8192×8192×2048 | 202.5 | 670.7 | 1278.1 | 1259.0 | 1196.1 | −6.4% |
| 16384×16384×2048 | 204.5 | 696.7 | 1330.5 | 1313.7 | 1249.0 | −6.1% |

Two things worth reading off this:

- **cuBLAS-nvfp4 and CUTLASS-nvfp4 agree to within 0.4%.** Two independent implementations landing
  on the same number is good evidence the baseline really is the hardware ceiling, not an artifact
  of one library's tuning.
- The mixed kernel costs **1–6% against the best available NVFP4**, while being ~1.9× cuBLAS fp8
  and ~6.8× cuBLAS bf16. Note also that cuBLAS's fp8 is roughly 1.8× the CUTLASS fp8 example
  kernel, so the earlier "3.5× FP8" figure was flattering — against a properly tuned fp8 the honest
  multiplier is ~1.9×.

> A shared GPU makes these numbers fragile: with another process resident, stock `nvfp4_gemm`
> itself read 852 instead of 1208, and cuBLAS-nvfp4 read 983 instead of 1200. All three benchmark
> scripts (`sweep.sh`, `bench_all.sh`, `bench_vs_cublas.sh`) refuse to run unless the card is idle.

---

## 5. Granularity, and what it costs

The hardware floor is the footprint of one `mma.sync`: because the atom is `m16n8k64`, a format
granule can never be finer than **16 rows of A × 8 columns of B × 64 elements of K**. The kernel
now reaches that floor and is numerically correct there. It is not free, and the rest of this
section is about what it costs and why.

### The measured curve

4096³, best of 5 on a verified-idle RTX 5090, randomly tagged so all four format sites are live.
"No dispatch" is the same source built with `-DMIXFP4_NO_DISPATCH=1`. Each percentage is against
the stock `nvfp4_gemm` measured in *its own* run — 1204.5 and 1205.8 TFLOP/s across the two sweeps
that produced this table, which is also a fair reading of the run-to-run noise floor.

| granule (A rows × B cols × K) | dispatch | arms | OMMAs | codegen | TFLOP/s | vs stock |
|---|---|---|---|---|---|---|
| — (no dispatch) | none | 1 | 64 | clean | 1187.6 | +1.4% |
| **32 × 32 × 128 (default)** | C++, per k_tile | 8 | 512 | clean | **1164.1** | **+3.4%** |
| **16 × 64 × 128** | C++, per k_tile | 8 | 512 | clean | **1164.7** | **+3.6%** |
| 16 × 32 × 128 | C++ + blob | 16 | 1024 | clean | 1113.8 | +7.5% |
| 32 × 16 × 128 | C++ + blob | 32 | 2048 | clean | 1082.5 | +10.1% |
| 32 × 32 × 64 | PTX, 1 per k_block | 8 | 512 | clean | 922.0 | +23.5% |
| 16 × 16 × 64 | PTX, 1 per k_block | 64 | 4096 | clean | 887.2 | +26.3% |
| **16 × 8 × 64 (the floor)** | PTX, 2 per k_block | 2 × 64 | 4096 | clean | **807.9** | **+32.9%** |
| 16 × 8 × 64 | PTX, 4 per k_block | 4 × 16 | 1024 | clean | 677.4 | +43.8% |
| 16 × 8 × 64 | PTX, 8 per k_block | 8 × 8 | 512 | clean | 469.4 | +61.0% |

The three 16×8×64 rows are the same granule reached with different group shapes, and their
ordering is the whole argument: throughput falls **monotonically with the number of dispatches**,
even though code size falls 8× going the other way (4096 → 512 OMMAs). Fewer, fatter tables win.
Solving for the marginal cost of one dispatch per k_tile at 4096³ gives ~11–13 µs, i.e. **~70–120
warp-scheduler cycles each** — which is the same number section "where the dispatch sits" arrives
at from the other direction.

Repeated at 8192×8192×2048 (short-K, where dispatch amortizes over fewer k_tiles), the ordering is
identical and the costs slightly worse: +25.5% / +34.3% / +42.5% / +59.3%.

Two separate mechanisms are at work, and separating them is what the rest of this section does.
Among the K=128 rows, cost grows with **arm count** — that is instruction-fetch pressure from the
code footprint. Among the K=64 rows, it is dominated by **how many dispatches sit inside the
k_tile body**, and the footprint barely matters (the 512-OMMA and 4096-OMMA floor configs differ
by 17 points in the *wrong* direction).

### It is the dispatch's *position*, not the K axis — and that is fixable

Rows 2 and 5 of that table are a controlled experiment. **32×32×128 and 32×32×64 have the same
spatial granule, the same 8 arms, and the same 512 OMMAs.** The only difference is that the
dispatch moved from outside the k_tile body to inside it, once per k_block. That alone costs
**20.2 points**, +3.3% → +23.5%, and it is the single largest term anywhere in this section.

It is tempting to read that as "the K axis is expensive". It is not. A K granule of 64 only
*implies* an in-body dispatch if you insist on choosing the format after entering the body. The
alternative is to specialize the k_tile on **both k_blocks' patterns at once** — `2^(2·bits)` arms
instead of `2^bits`, with the branch left where it is cheap. Two variants, both on the blob path:

| mode | A granule | B granule | arms | TFLOP/s | vs stock |
|---|---|---|---|---|---|
| shipped default | 32 rows × 128 K | 32 cols × 128 K | 8 | 1167.3 | +3.4% |
| C++, fine A | **16 rows** × 128 K | 64 cols × 128 K | 8 | 1164.7 | +3.6% |
| **`MIXFP4_JOINT_KB`** | 32 rows × 128 K | **64 cols × 64 K** | 8 | **1147.6** | **+5.0%** |
| `MIXFP4_JOINT_KB`, fine A | **16 rows** × 128 K | **64 cols × 64 K** | 16 | 1098.8 | +9.0% |
| `MIXFP4_JOINT_K` | 32 rows × **64 K** | 64 cols × **64 K** | 16 | 1093.2 | +9.5% |
| `MIXFP4_JOINT_KB`, finer B | 32 rows × 128 K | 32 cols × **64 K** | 32 | 1057.7 | +12.5% |
| in-body dispatch | 32 rows × 64 K | 32 cols × 64 K | 8 | 923.5 | +23.6% |

Row 2 is worth calling out on its own: **A reaches its 16-row hardware floor for +3.6%**, within
noise of the shipped default and identical to it at 8192×8192×2048 (both 1196.3). Sixteen rows
costs two A-granule bits, but coarsening B to 64 columns buys them straight back, so it is still
3 bits and 8 arms. The budget is spendable on either operand — just not both.

Row 4 prices the combination: fine A *and* a 64-element weight-K granule needs
`kAGran + 2·kBGran` = 4 bits, so 16 arms and +9.0%. The 4-point gap to row 3 is the arm doubling,
not the granularity.

So a 64-element K granule costs **1.6 points** on the weight operand, or ~6 on both — not 20. The
cleanest reading is the pair of 16-arm rows in the two tables: 16×32×128 (+7.7%) versus joint-K
32×64×64 (+8.9%). Same arm count, same dispatch placement, differing only in K. **K=64 is worth
about 1.2 points.** Everything else that looked like a K cost was the branch moving indoors.

`JOINT_KB` is the asymmetric one and usually the right default: in a linear layer B is the weight
operand, and weights are what a quantizer groups along K (16 elements per scale). A is
activations. Spending the arm budget only on B costs `kAGran + 2·kBGran` bits rather than
`2·(kAGran + kBGran)`, which is how it fits a 64-element K granule into the *same 8 arms* the
shipped default already uses. The only added work is two shared-memory flag reads per k_tile,
outside the body.

Two implementation notes, both of which cost a debugging round:

- At the top of a k_tile **only k_block 0's operands are resident** — k_block 1 is copied inside
  the body by `copy_kblock(k_block_next)`. Reading its flags from the register fragment at
  dispatch time silently picks up the *previous* k_tile (0.29 relative error, not 0.66, because
  most granules still happen to match). Both joint paths read from the smem stage instead, which
  the TMA producer filled before `consumer_wait` released us.
- `tCsSF*_stage` is the copy **source** view, shaped `(CPY, CPY_MN, CPY_K)`, and `CPY_MN` is the
  copy atom's tiling, *not* the MMA atom index — for SFB the copy moves several atoms at once, so
  `(0, 4, k)` is not atom 4. Index it linearly: `copy()` guarantees logical element *i* of the
  source lands in element *i* of the retiled register view, so atom *a*'s byte 0 is flat index
  `V·a`. This is invisible in any configuration with one granule per operand (index 0 is index 0
  under every layout) and is 0.37 relative error the moment a second granule exists.

What the joint trick cannot do is reach the floor. 16×8×64 needs 10 bits per k_block, so 20 jointly
— far past the 32-arm cliff. That is why the floor still pays the in-body dispatch, twice.

### Shrinking the CTA tile: works structurally, does not pay

A 16×16×128 granule needs `kAGran=2 + kBGran=4` = 6 bits, so 64 arms, and 64 arms outlines
(`STACK:912` plain, `STACK:864` with the blob — the blob shaves the frame but does not prevent the
spill). Measured, that build runs at **41.8 TFLOP/s, +96.5%**, consistent with the 32.3 this report
recorded originally.

But the arm count is not a property of the granule — it is a property of the granule *relative to
the warp tile*. A warp owns 8 n-atoms only because the CTA tile is 128 wide in N. At
`-DMIXFP4_TILE_N=64` it owns 4, so a 16-column granule costs 2 bits instead of 4 and the whole
thing fits in **16 arms**, clean: `REG:168`, `STACK:0`, 512 OMMAs, no `CALL`, no `LDL`/`STL`.

That part works. It just does not pay:

| route to 16 × 16 × 128 | tile N | arms | TFLOP/s | vs stock |
|---|---|---|---|---|
| ceiling, no dispatch | 128 | 1 | 1187.3 | +1.4% |
| **ceiling, no dispatch** | **64** | **1** | **903.2** | **+25.0%** |
| 16×16×128, half-width tile | 64 | 16 | 893.5 | +25.8% |
| 16×16×128, 64 arms, outlined | 128 | 64 | 41.8 | +96.5% |
| 16×16×**64**, PTX in-body dispatch | 128 | 64 | 886.7 | +26.4% |

**The half-width tile costs 25% before any dispatch exists.** Halving N halves the reuse of each
A-fragment load, and this kernel is close enough to its roofline that the tile shape dominates
everything the dispatch does. The 16-arm dispatch on top of it is nearly free — 903.2 → 893.5, a
0.8-point cost, which is a clean independent confirmation that an out-of-body dispatch at 16 arms
is cheap — but it is 0.8 points on top of a 25-point loss.

So for a 16×16 spatial granule the in-body PTX path wins outright: same price (+26.4% vs +25.8%,
inside run-to-run noise) and it delivers K=64 rather than K=128. **Shrinking the tile to buy
dispatch bits is a dead end**, and the reason is worth remembering: the tile shape is a throughput
parameter first and a granularity parameter only incidentally.

The practical consequence for a quantization scheme: **a 64-element K group on the weights is
nearly free** (+5.0%), fine channel granularity at K=128 is cheap (16×32×128 at +7.5%), and only
the combination of both, or the true floor, runs into the 20-point wall.

### The warp tile sets the arm count, and it is the cheapest thing to change

Everything above treats the arm count as a property of the granule. It is not: it is a property of
the granule **relative to the warp tile**, and the warp tile is a free parameter.

The builder gives a 128×128 CTA tile to 8 MMA warps as `Layout<Shape<_4,_2,_1>>`, so a warp owns
32 rows × 64 columns — *two* 16-row m-atoms. That two is the whole problem for a 16-row granule: it
costs 2 dispatch bits per k_block, so 4 jointly, and with B's cheapest single bit that is 5 bits
and 32 arms. Measured, **1065.8 TFLOP/s, +13.3%**.

`Layout<Shape<_8,_1,_1>>` gives each warp ONE m-atom and all 16 n-atoms. The identical granule now
costs 2·1 + 1 = 3 bits, i.e. **8 arms**. Unlike shrinking the CTA tile — which loses operand reuse
and cost 25% before any dispatch existed — the tile stays 128×128, so the CTA computes the same
product and moves the same global traffic. Only intra-CTA shared-memory reads grow (every warp
reads all 128 columns of B rather than 64), and this kernel had headroom there. It is also not an
exotic layout: it is what CUTLASS's own sm120 blockscaled builder selects for tiles narrower than
16, so the smem layouts and copy atoms already support it.

One trap, and it is worth more than anything else in this section. 8×1 makes `MMA_TILE_M` 128, so
`EPI_TILE_M % MMA_TILE_M == 0` fails and the epilogue tile must be named explicitly. Which one you
name dominates the result:

| epilogue tile | no-dispatch ceiling | vs stock |
|---|---|---|
| `Shape<_128,_16>` | 1169.4 | +3.3% |
| `Shape<_128,_64>` | 1161.3 | +4.0% |
| **`Shape<_128,_32>`** | **1201.8** | **+0.5%** |

At 128×32 the 8×1 arrangement's ceiling is *above* the 4×2 arrangement's own 1187 — the warp
rearrangement is free, and the 3.3% the first guess cost was entirely the epilogue tile.

With that, `MIXFP4_JOINT_KA` delivers a **16 row × 64 K** format granule — the A footprint of one
`mma.sync` — at 8 arms:

| shape | stock | mixed | overhead |
|---|---|---|---|
| 4096³ | 1206.0 | 1149.9 | **4.88%** |
| 4096×4096×8192 | 1274.1 | 1207.3 | 5.53% |
| 8192×8192×2048 | 1257.8 | 1208.4 | 4.09% |
| 8192³ | 1402.5 | 1330.5 | 5.41% |
| 2048³ | 798.2 | 785.4 | 1.62% |

Two smaller things were worth ~2 points each. Only k_block 1's flags need the smem round trip —
k_block 0's operands are resident in registers at dispatch time, so reading them back out of
shared memory put an LDS latency on the branch's critical path for nothing. And the remaining smem
read can be **software-pipelined**: the next k_tile's operands become readable right after
`copy_kblock(0)` refills the register fragment, so computing the next arm index there gives the
load 16 MMAs to hide behind (1130.4 → 1150.7).

### Both operands at 16×64 is blocked, and by how much

The obvious next ask is a 16×16×64 granule — both operands at one `mma.sync`'s footprint. It does
not fit, and the reason is arithmetic rather than tuning.

A warp's footprint is `CTA_M·CTA_N/8` = 2048 elements. For a 16×16 granule the flag count is
`warp_rows/16 + warp_cols/16`, which is minimised by a square-ish warp tile and equals **6 for
every arrangement of 8 warps** (32×64 → 2+4; 64×32 → 4+2; 16×128 → 1+8). So 64 arms per k_block,
and 12 bits / 4096 arms if specialised jointly to keep the branch out of the loop body. Three
escapes were tried and all are closed:

| escape | result |
|---|---|
| 16 warps (4×4), halving the warp tile to 32×32 → 4 bits | the cooperative kernel `static_assert`s "TiledMMA operating using 256 threads" |
| CTA tile K=64, so one k_tile *is* one k_block and 6 bits suffice out-of-body | no-dispatch ceiling **720.8 TFLOP/s, −40%** — a k_tile amortises half as much over each TMA load and barrier |
| 64 arms out-of-body at K=128 (16×16×128) | outlines: `STACK:864`, `CALL:1` — 64 arms × 32 MMAs is past the cliff, though 64 × 16 stays clean |

That leaves a per-k_block dispatch, and there is a hard empirical bound on it. At
`MIXFP4_TAG=none` — one arm, index effectively free, no i-cache pressure, and only *3* flags
rather than 6 — two in-body dispatches per k_tile already cost **1174 → 994 TFLOP/s (+21%)**. The
real target needs strictly more than that, and measures:

| granule | mechanism | TFLOP/s | vs stock |
|---|---|---|---|
| 16×16×64 | `brx.idx`, 64 arms/k_block | 887.6 | +35.9% |
| 16×16×64 | balanced `bra` tree, 64 arms/k_block | 790.2 | +52.8% |
| 16×64 on A only | `JOINT_KA`, 8 arms, out-of-body | **1150.7** | **+4.9%** |

Why the jump is slow is worth recording, because the earlier reading of it was wrong. `brx.idx`
compiles to `IMAD → LDC c[0x2][...] → BRX`: **the branch target is a constant-memory load**, so
instruction fetch cannot resolve until it returns, and a `WARPSYNC.ALL` follows at the target. But
that is not the dominant term either — replacing it with a balanced `bra` tree removes the `LDC`
entirely and is *worse* (790 vs 888), and moving the pipeline barrier inside the first arm to stop
it stranding between two dispatch regions buys only +4 (935 → 939).

What the profiler says instead is that it is not a stall at all. Across the pair, `smsp__issue_active`
is unchanged (21.9% vs 22.4%) and cycles track instruction count almost exactly (+24% instructions
→ +21% cycles), while `math_pipe_throttle` *falls* (2.94 → 2.20) — the tensor pipe is going idle.
Each dispatch is ~23 instructions of which only ~6 are the branch; the rest is reading and
assembling the flags. Splitting a k_tile into two separately-dispatched regions doubles that and
halves the straight-line run each one amortises over.

So the exchange rate stands: **one operand at the `mma.sync` floor is ~5%; both is ~36%.**

### Where the dispatch sits is worth ~90 cycles

A dispatch placed *inside* the k_tile body costs about 90 cycles; one that encloses the whole
k_tile costs about 2.4 cycles per branch level. That is a 20–40× difference for the same decision,
and it is **not** the branch opcode. Three measurements pin it down, all at 4096³:

| variant | TFLOP/s | what it isolates |
|---|---|---|
| 16 MMAs in one opaque `asm volatile`, **no dispatch** | 1189.9 | the opaque blob itself is **free** — identical to the no-dispatch ceiling |
| `brx.idx` with the index computed **inside** the asm | 944.6 | ~90 cycles per dispatch |
| `brx.idx` with the index **hoisted** into C++ | 962.2 | +18 only, so it is not the `bfe`/`mad` → `LDC` dependency chain |
| balanced `bra` tree instead of `brx.idx` | 915.5 | direct branches are *worse*, so it is not the indirect jump |

(All four at `MIXFP4_TAG=none`, which pins every dispatch to arm 0 and so removes the code-footprint
term.) What is left is that a branch inside the loop body is a **scheduling barrier**: ptxas can no
longer interleave the next k_block's `LDSM` shared-memory loads with this k_block's MMAs, so the
shared-memory latency stops being hidden. That is why the per-k_tile C++ dispatch costs 3.3% for
the same decision that costs 28% per k_block.

The consequence for granularity is structural: **a K granule of 64 requires a dispatch per k_block
and therefore costs ≥20%, however the branch is spelled.** K=128 keeps the dispatch outside the
body, and then the binding constraint is arm count instead.

One aside worth recording, because it contradicts an earlier note here: the `bra` tree only stays
un-if-converted if each arm contains something ptxas will not speculate. A single
`bar.warp.sync -1` per arm is enough — without it, 512 of 4096 OMMAs come back predicated; with
it, zero do, and the branch is a plain `BRA`. The poison is nearly free; the tree is still slower
than `brx.idx`.

### The 8-arm cliff moves — it was statement count, not code size

This report previously recorded a hard cliff between 8 and 16 arms that "resists the obvious
levers" (`-inline-threshold`, `always_inline`, halving the code). That was right about those
levers and wrong about the cause. The trigger is not how much code cicc sees but **how many
inline-asm statements** it sees: emitting a k_block's 16 MMAs as one opaque blob instead of 16
separate `cute::gemm` statements takes a k_tile body from 32 statements to 2, and the cliff moves
from 8 arms to somewhere between 32 and 64.

| arms | granule | C++ + `cute::gemm` | C++ + blob (`scripts/gen_mixed_mma_blob.py`) |
|---|---|---|---|
| 8 | 32×32×128 | clean, 1166.5 | clean |
| 16 | 16×32×128 | `STACK:912`, **36.7** | clean, `STACK:0`, **1116.3** |
| 32 | 32×16×128 | (not reached) | clean, `STACK:0`, **1084.2** |
| 64 | 16×16×128 | (not reached) | `STACK:864`, outlined |

So the arm budget for the *cheap* dispatch is 4× larger than recorded. Throughput still falls with
arm count — that is the code-footprint term — so 16 and 32 arms cost 7.5% and 10.2% rather than
3.3%. `gen_mixed_mma_blob.py` refuses to emit past 32 arms rather than silently producing the
outlined build.

### Reaching the floor: split jump tables

At the floor a warp's k_block carries 2 A flags + 8 B flags = 10 bits, so a single table indexed
by the whole pattern would be 1024 arms × 16 MMAs = 16,384 `mma.sync` in one asm statement. Instead
`scripts/gen_mixed_mma_ptx.py` partitions the warp's `MMA_M × MMA_N` atom grid into groups and
gives each group its own `brx.idx` over only the flags its own MMAs need — trading one extra
indirect branch per group against an exponential reduction in code:

| group (atoms) | groups | bits | arms/group | OMMAs in kernel |
|---|---|---|---|---|
| 2×8 | 1 | 10 | 1024 | 65536 — not buildable |
| 2×4 | 2 | 6 | 64 | 4096 |
| 2×2 | 4 | 4 | 16 | 1024 |
| 1×2 | 8 | 3 | 8 | 512 |

Every configuration above is numerically correct (all four sites, random per-granule tagging, at
256³/512³/1024³/2048³ and 1024×2048×512) with `REG:168`, `STACK:0`, zero predicated OMMAs and an
exact 4-way per-site OMMA census. Because each group's A-flag and B-flag occupy disjoint,
independently varying bits of its pattern, every MMA sees each of the four sites in exactly
2^(bits−2) of the arms, so the patcher's equal-count invariant holds by construction.

### What this means for the 5%-of-stock budget

Stock NVFP4 runs a 4096³ k_tile in about 720 warp-scheduler cycles, so a 5% budget is ~36 cycles
per k_tile — roughly 1.6 OMMA issue slots. The shipped 8-arm default spends about 14 of those on 3
bits of format selection. That is the real currency: **at 5% you can afford about 3 bits of format
choice per k_tile**, and every extra bit doubles the code. The floor needs 10 bits at K=128, or 20
at K=64.

So the granule ladder, by budget:

- **≤5%:** 32×32×128 (+3.4%) or **16×64×128 (+3.6%, A at its row floor)** — 3 bits, 8 arms — **and**
  a 64-element K granule on the weight operand via `MIXFP4_JOINT_KB` at +5.0%, also 8 arms.
- **7–13%:** 16×32×128 (+7.5%) or 32×16×128 (+10.1%) via the blob path; fine A *plus* weight-K=64
  at +9.0%; K=64 on both operands via `MIXFP4_JOINT_K` at +9.5%; 32-column B at K=64 at +12.5%.
- **24–33%:** anything needing a dispatch *inside* the k_tile body — which now means only the
  configurations too fine to fit the joint scheme's arm budget, including the 16×8×64 floor.

The floor is therefore available and correct, but it is a granularity-first option, not a
throughput-competitive one. What *did* move is the K axis: it used to cost 20 points and now costs
1.6 on the weights, because the joint schemes keep the branch outside the loop body. What has not
moved is the arm budget, and that is what still rules out the floor.

Configurable via `-DMIXFP4_A_ATOMS_PER_GRANULE` / `-DMIXFP4_B_ATOMS_PER_GRANULE` (in atoms; A
atoms are 16 rows, B atoms 8 columns) for the C++ paths, or by regenerating the header for the
blob and PTX paths.

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

MIX=build/mixed_patched ./scripts/bench_all.sh   # the table in section 4
./scripts/sweep.sh                              # the granularity curve in section 5
```

Finer granules than the 8-arm default need a generated header. Both generators print the
granule, the arm count and the per-site OMMA count the patcher should report, so a mismatch is
caught before the GPU is involved:

```bash
# K=128, cheap per-k_tile dispatch, up to 32 arms  (section 5, "the cliff moves")
A_ATOMS=1 B_ATOMS=4 python3 scripts/gen_mixed_mma_blob.py       # 16 x 32 x 128
EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_A_ATOMS_PER_GRANULE=1 -DMIXFP4_B_ATOMS_PER_GRANULE=4" \
  ./scripts/build_mixed.sh build/blob16

# a 64-element K granule on the WEIGHT operand, in the same 8 arms as the default: +5.0%
A_ATOMS=2 B_ATOMS=8 python3 scripts/gen_mixed_mma_blob.py
EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ATOMS_PER_GRANULE=2 -DMIXFP4_B_ATOMS_PER_GRANULE=8" \
  ./scripts/build_mixed.sh build/jkb          # A 32 rows x 128 K, B 64 cols x 64 K

# K=64 on both operands (-DMIXFP4_JOINT_K=1 instead) costs 16 arms and +9.5%

# K=64, down to the hardware floor, at the per-k_block dispatch cost
python3 scripts/gen_mixed_mma_ptx.py --a-atoms 1 --b-atoms 1 --m-per-group 2 --n-per-group 2
EXTRA="-DMIXFP4_PTX=1" ./scripts/build_mixed.sh build/floor          # 16 x 8 x 64
```

The granule macros are taken *from* the generated header on the PTX path, so the host-side tagging
follows automatically — including the K granule, which is 64 there and 128 on the C++ paths.

Build-time options: `-DMIXFP4_DEBUG_UNIFORMITY=1` (catch tagging finer than a granule — it now
runs on the PTX path too, per k_block), `-DMIXFP4_NO_DISPATCH=1` (compile the dispatch out — the
in-source performance ceiling), `-DMIXFP4_PTX=1`, `-DMIXFP4_BLOB=1`. `-DMIXFP4_PTX_16X16=1` is a
deprecated alias for `-DMIXFP4_PTX=1`; what it builds is now whatever the generated header holds.

---

## 7. Limitations

- **The binary must be patched.** Unpatched it computes ordinary NVFP4, silently and correctly.
- **The format tag consumes bit 7 of every UE4M3 scale byte.** Architecturally ignored by the
  tensor core (verified on hardware in `tests/mma_intrinsics`), so it costs no storage or
  bandwidth — but it is not free if some other consumer of those scale factors reads that bit.
- **The granule is generally not a contiguous tile** (section 3B). Host-side quantization must
  respect the strided shape, and violating it below atom granularity hangs the GPU rather than
  returning wrong numbers. At the hardware floor it happens to *become* contiguous — one n-atom is
  8 adjacent columns and one m-atom is 16 adjacent rows — but that is a property of that one
  configuration, not something to rely on.
- **The granule is tied to the tile shape and warp layout.** Change `ThreadBlockShape` or the
  AtomLayout and the granule changes with it; the map is derived automatically, but the *arm
  count* — and therefore the cliff in section 5 — must be rechecked. The PTX path additionally
  hardcodes the warp's atom counts in generated code, and `static_assert`s them against the tile.
- **Fine granularity and throughput are in direct conflict, and the exchange rate is steep**
  (section 5): about 3 bits of format choice per k_tile fit in a 5% budget, and each extra bit
  doubles the code. The 16×8×64 floor is correct and available but costs ~28% or more, because
  a K granule of 64 forces a dispatch inside the k_tile body.
- Only `ue4m3` scale factors are exercised. `ue8m0` and `scale_vec::2X` are untested here (the
  latter has a known pre-existing E0M3 hardware limitation, unrelated to this work).
- E0M3 semantics rest on an undocumented, patched instruction encoding. It is validated
  numerically on one GPU and could change on other silicon or with a different disassembler.
