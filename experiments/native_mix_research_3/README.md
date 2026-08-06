# NativeMix Research 3 support code

This directory contains CPU-only, independent reference code used by the
Research 3 availability-gated audit.  It deliberately does not import the
legacy CUDA probe or its Python expected-value helper.

No SM120 device was available in the audited host, so none of these helpers
claim native E0M3 validation.  Generated GPU-observation fields are explicitly
marked `NOT_RUN`.

Run the CPU checks with:

```bash
python -m unittest discover -s experiments/native_mix_research_3 -p 'test_*.py'
python experiments/native_mix_research_3/generate_cpu_artifacts.py \
  --output artifacts/native_mix_research_3
```
