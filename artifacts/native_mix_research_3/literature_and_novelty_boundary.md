# Literature and novelty boundary

Sources were checked on 2026-08-07. This is a scope boundary, not a claim that NativeMix has been
validated.

| Source | Established public/related-work surface | NativeMix claim that is not available yet |
|---|---|---|
| NVIDIA TensorRT quantization schemes | NVFP4 uses E2M1, range ±6, per-16 block scaling, ties-to-even | No public E0M3 interface is described |
| NVIDIA Transformer Engine NVFP4 | E2M1 payload × E4M3 block scale × FP32 global scale | No public E0M3 mode is described |
| NVIDIA PTX ISA 9.3 | `mxf4nvf4` valid element type is E2M1; SM120a is listed | Text search finds no `e0m3` token; binary behavior still needs hardware proof |
| IF4, arXiv:2603.28765 | adaptive block-scaled formats and format tag in unused scale sign bit | Dual-codebook/tag mechanism cannot be claimed as new |
| MixFP4, arXiv:2605.31035 | per-block E2M1/E1M2 choice and zero-extra-metadata scale sign tag | Dual-codebook and format tag cannot be claimed as new |
| Four Over Six, arXiv:2512.02010 | adaptive scaling is a strong NVFP4 baseline | NativeMix utility must beat it, not only stock NVFP4 |
| CUTLASS | public block-scaled SM120 building blocks | Undocumented encoding semantics and safe rewriting are outside the public API |

Primary links:

- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html
- https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://github.com/NVIDIA/cutlass
- https://arxiv.org/abs/2603.28765
- https://arxiv.org/abs/2605.31035
- https://arxiv.org/abs/2512.02010
- https://github.com/mit-han-lab/fouroversix

## Permissible positioning if future gates pass

The narrow candidate is a systems/architecture contribution:

1. bit-exact characterization of a latent format-polymorphic SM120 execution mode;
2. signature- and version-bound binary rewriting with fail-closed runtime canaries;
3. a one-sided weight-adaptive native GEMM respecting measured hardware control granularity;
4. hardware-cost-aware assignment tied to measured dispatch cost; and
5. reproducible quality–throughput Pareto on two pinned 8B dense models and two SM120 devices.

## Claims prohibited now

- NVIDIA does not officially support E0M3 through the public interface.
- The candidate mapping `±{0,1,…,7}` is not hardware-confirmed in Research 3.
- The old single-device report is not sufficient for shipping-silicon stability.
- The scale sign bit, adaptive dual codebooks, and generic MixFP4 accuracy are prior art.
- No ASPLOS/MICRO positioning is warranted until ISA, safety, kernel, two-model, end-to-end, and
  portability gates pass.

Current positioning: `none` (`INCOMPLETE`, hardware unavailable).
