# Cost analysis

Status: `INCOMPLETE`; no native timing was performed.

| Item | Intended NativeMix design | Research 3 measured value |
|---|---|---|
| FP4 payload | 4 bits/weight | not exported |
| format metadata | proposed reuse of UE4M3 bit 7 | not hardware-revalidated |
| additional logical bits/weight | intended 0 | `null` until type-block semantics pass |
| physical checkpoint overhead | intended 0% for payload/tag | `null`; no checkpoint built |
| block scale | UE4M3 magnitude in bits 0–6 | CPU representation checked; native not checked |
| tensor scale | existing global scale | not evaluated |
| online activation path | dynamic E2M1 | not integrated |
| online reorder | intended none for K2 | not tested |
| extra inference GEMM | intended none for K2 | not tested |
| extra kernel control | offline type map + native dispatch | cost not measured |
| software decode | required K7 comparator | not run |
| build/patch time | signature-bound offline operation | not run |
| load canary | required, fail closed | not implemented/run in Research 3 |
| native coverage | target ≥90% eligible FLOPs | not run |

Legacy Python/fake-quant timing is not reported as inference latency. Legacy aggregate TFLOP/s is
kept only as `UNVERIFIED` arithmetic in the prior-results report. Energy/token, memory, load time,
TTFT, TPOT, and tokens/s have no values because the native end-to-end phase was not entered.

The current deployment hypothesis cannot be accepted or rejected on this host. Its public API
support is explicitly `false`.
