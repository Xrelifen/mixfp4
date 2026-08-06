Decision: INCOMPLETE
Confidence: high
Paper positioning: none
Primary candidate: OneSided-WeightAdaptive-NativeMix
Primary reason: The audited host has zero SM120 GPUs, so the mandatory native ISA gate and every downstream native gate could not run.
SM120 devices tested: 0
Qwen3-8B native completed: No
Llama-3.1-8B native completed: No
Median kernel overhead: NOT_RUN
P95 kernel overhead: NOT_RUN
Native kernel coverage: NOT_RUN
Vs 4Over6: NOT_RUN
Vs software mixed: NOT_RUN
Public API supported: No
No push performed: Yes

# Scope and completed work

Phase 0 audit was completed subject to two missing requested documents and absent legacy raw logs.
The old headline performance arithmetic was recomputed but remains `UNVERIFIED`. Phase 1's
independent CPU reference was implemented and passed 8 tests; it produced both 16-nibble decode
tables and a 1,024-row four-combination CPU truth table. No native observation is present in those
tables. Phases 2–9 are `NOT_RUN` because the ISA gate could not be attempted.

# ISA gate

The gate is `INCOMPLETE`, not failed numerically. None of these mandatory claims is established in
Research 3: patched/candidate agreement, stock negative-control disagreement, 100% fallback canary,
one-sided mixed operation, or absence of unexplained nibbles.

# Required mechanism answers

## 1. Is E0M3 bit-exact, stable, and not fallback?

`INCONCLUSIVE`. Legacy source proposes it, but Research 3 has no SM120 output, raw legacy cubin, or
runtime canary. The public PTX surface exposes E2M1, not E0M3.

## 2. What are the 16 nibble values?

The independent **candidate** CPU mapping is:

`0x0..0x7 → +0,+1,+2,+3,+4,+5,+6,+7`;
`0x8..0xf → -0,-1,-2,-3,-4,-5,-6,-7`.

This is a sign-magnitude integer lattice, scale-equivalent to an E1M2/INT-like half-step lattice
multiplied by two. It is a hypothesis until every nibble matches an SM120 observation.

## 3. Can A and B select formats independently?

`INCONCLUSIVE`. The legacy two-bit site map suggests four A/B combinations, but the exhaustive
four-mode Research 3 test was not run.

## 4. What is the minimum format-control granularity?

`INCONCLUSIVE`. An MMA footprint and a software dispatch tile are different concepts. The required
adjacent-16-value probe did not run, so K2 must not assume a 16×64 type block.

## 5. Why do predication/dispatch lose performance?

Legacy report analysis attributes the per-MMA loss to added issued instructions, branch/control
footprint, and disruption of cross-iteration scheduling, leaving Tensor Core issue capacity idle.
This causal account is `UNVERIFIED` in Research 3 because raw profiler data is absent.

## 6. Which dispatch strategy is best?

`NOT_RUN`. Legacy aggregate tables favor one-sided, low-bit-count k-tile/CTA control and show a
bad per-MMA baseline, but K0/K2/K3/K6/K7 were not interleaved and remeasured.

## 7. Did one-sided adaptive reach ≤3% median overhead?

`NOT_RUN`. Legacy arithmetic includes individual one-sided observations near 0.6% and 2.6%, but
there is no model-shape-weighted median/P95 raw dataset. No gate is passed.

## 8. Does native kernel coverage reach the main GEMMs of both 8B models?

`NOT_RUN`; both revisions, shape traces, adapters, fallbacks, and coverage are absent.

## 9. Does NativeMix beat Four Over Six?

`INCONCLUSIVE`; neither quality nor native throughput comparison was run.

## 10. What is the ideal-per-16 versus hardware-feasible accuracy gap?

`NOT_RUN`; the hardware type-block scope is unknown and no selector split was created.

## 11. How much faster is native than software decode?

`NOT_RUN`; K2 and K7 were not built or timed.

## 12. Are Qwen3-8B and Llama-3.1-8B consistent?

`NOT_RUN`; WikiText-2, C4, and downstream values are all absent for both models.

## 13. Was it reproduced on a second SM120 and different driver?

No. Zero SM120 devices were available; cross-device and cross-version statuses are `NOT_RUN`.

## 14. Systems co-design, architecture-only, or NO_GO?

None can be selected from current evidence. The required protocol dictates `INCOMPLETE`. Even an
architecture-only decision requires at least one complete SM120 ISA characterization; full systems
co-design requires two devices, two 8B models, and native end-to-end results.

## 15. The single highest-priority next task

Provision one exclusive SM120 GPU and run Phase 1 exhaustive ISA characterization through the
launch guard, including all four operand combinations and fail-loud negative controls. Do not
start patcher refactoring, kernels, selectors, or model work until that ISA gate passes.

# Negative and missing evidence

- `repository_audit.md` and `related_work_positioning.md` were not present.
- No legacy raw logs, cubins, disassemblies, clocks, power, or manifests were recoverable.
- CUTLASS and the reference SM120 submodule are unavailable in the dirty worktree.
- The legacy build helper expects CUDA 13.1; the host has CUDA 12.8 nvcc.
- Neither required 8B checkpoint was cached.
- GPU 2 was occupied by another user's process; it was only observed and not disturbed.

These secondary issues did not cause the formal stop—the decisive blocker is the absence of SM120.
