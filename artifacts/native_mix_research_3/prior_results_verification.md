# Prior results verification

## Formal status

`UNVERIFIED`. The exact HEAD report was found, but no raw timing CSV, benchmark log, cubin,
disassembly, clock/power record, or run manifest exists in the tracked tree or visible history.
Arithmetic below was independently recomputed from aggregate report tables; it is not a Research 3
remeasurement. Remeasurement was prohibited by the absence of SM120.

## Recomputed legacy arithmetic

The headline k-tile table defines throughput loss as
`(stock_TFLOPs - mixed_TFLOPs) / stock_TFLOPs`. Recalculation gives:

| Shape | Stock | Mixed | Recomputed throughput loss |
|---|---:|---:|---:|
| 1024³ | 282.6 | 274.5 | 2.8662% |
| 2048³ | 797.4 | 777.4 | 2.5082% |
| 4096³ | 1206.7 | 1165.9 | 3.3811% |
| 8192³ | 1401.8 | 1387.6 | 1.0130% |
| 4096×4096×16384 | 1289.5 | 1254.0 | 2.7530% |
| 8192×8192×2048 | 1258.1 | 1195.9 | 4.9440% |
| 16384×16384×2048 | 1313.8 | 1249.0 | 4.9323% |

Thus the report's rounded `1.0–4.9%` range is arithmetically reproducible.

The dispatch sections use latency-equivalent overhead, `stock_TFLOPs / mixed_TFLOPs - 1`:

- One-sided A-pure / B-adaptive: `1207 / 1176.5 - 1 = 2.5924%`, reproducing about 2.6%.
- One-sided B-pure / A-adaptive at 4096³: `1208.0 / 1200.8 - 1 = 0.5996%`, reproducing about 0.6%.
- Two-sided fine: `1205.8 / 887.6 - 1 = 35.8495%`, reproducing about 36%.
- Bad per-MMA observation: 1207 → 504 TFLOP/s, a 58.2436% throughput loss or 2.3948×
  stock-to-bad ratio.

The two percent conventions are both retained to avoid silently changing the old report's meaning.

## Legacy correctness claims

The prior report states relative error about 0.0017 for all four patched format combinations and
0.75 for an unpatched E0M3-tag negative control. Because raw output and the original/patched cubins
are absent, both are `UNVERIFIED`; they do not pass the Research 3 ISA gate.

## Research 0–2 boundary

- Research 0: high-confidence `NO_GO` for codebook-conditioned dual transforms; specialization did
  not beat parameter-matched controls or PPL.
- Research 1: high-confidence `NO_GO`; optimizer choice did not rescue transformation, and the
  bounded scale path was diagnostic rather than a new method.
- Research 2: high-confidence `NO_GO`; Qwen scale result reproduced but E1/E2 conditioning was
  only about 0.314% versus Shared16, was tied with faithful FOCUS on average, and did not establish
  downstream/generalization novelty.

Research 3 therefore cannot claim rotation, scale search, dual-codebook selection, or format-tag
metadata as its novelty. Only native latent-mode characterization, safe exposure, dispatch/kernel
co-design, and native two-model integration remain potentially distinct—and none is newly verified
on this host.

## Verification disposition

All prior performance rows are preserved in `tables/legacy_arithmetic_recompute.csv` with source
and formula. They must be remeasured from raw interleaved trials on an exclusive SM120 before use
in a formal result.
