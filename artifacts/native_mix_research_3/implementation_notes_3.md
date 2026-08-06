# Implementation notes

- Added only CPU-side Research 3 support code under `experiments/native_mix_research_3/`.
- `cpu_reference.py` is an independent implementation written for this audit. It does not import,
  copy, or adapt the legacy GPU probe, CUTLASS, `sm120-e0m3-mma`, IF4, MixFP4, or Four Over Six.
- `generate_cpu_artifacts.py` mechanically generates exhaustive CPU expected-value tables. It
  deliberately writes no synthetic GPU observation and labels every native field `NOT_RUN`.
- No kernel, patcher, baseline quantizer, evaluator, remote, or submodule was modified.
- No external source component was copied or adapted, so there is no added third-party license in
  this round. The root repository itself has no tracked license file; that packaging issue should
  be resolved before external distribution.
- A launch guard was not invoked because zero GPU subprocesses were launched. A future SM120 run
  must first implement/use the required unified guard and capture its preflight inventory.
