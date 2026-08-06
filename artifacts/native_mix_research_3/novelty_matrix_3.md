# Novelty matrix

| Candidate contribution | Prior/public overlap | Evidence required | Current status |
|---|---|---|---|
| E2M1/E0M3 adaptive codebooks | IF4 and MixFP4 already establish adaptive micro-formats | none can make generic dual codebooks new | **not claimed** |
| scale sign/tag bit | IF4 and MixFP4 already use unused scale sign bit | none can make the tag itself new | **not claimed** |
| generic scale search / 4-over-6 cap | Four Over Six and Research 2 are strong baselines | must beat Four Over Six | NOT_RUN |
| latent SM120 E0M3 execution semantics | absent from public PTX E0M3 surface | exhaustive bit-exact multi-device ISA characterization | INCONCLUSIVE |
| safe binary exposure | binary rewriting is only useful if signature/version bound and fail closed | hashes, disassembly, patch-count tests, runtime canary, restore | NOT_RUN |
| one-sided format-polymorphic GEMM | distinct only if native and efficient | K2 median ≤3%, P95 ≤5%, zero failures | NOT_RUN |
| hardware-aware format assignment | distinct only if measured control/cost alters the Pareto | S4 vs error-only selectors on two models | NOT_RUN |
| two-model native quality–throughput Pareto | systems-level end-to-end contribution | two pinned 8B models, native coverage ≥90%, full workloads | NOT_RUN |
| portability/safe delivery | necessary for undocumented mode | two SM120 devices and two driver/CUDA environments | NOT_RUN |

Paper readiness: `false`. Current paper positioning: `none`. At most an architecture-only result
could follow from a single SM120; full ASPLOS/MICRO positioning requires every systems gate.
