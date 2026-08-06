# Driver/CUDA compatibility

Status: `NOT_RUN`.

No SM120 device was available, so no cubin signature, patch offset, canary, correctness, or
throughput was tested under any driver/CUDA combination. The observed host combination is driver
565.57.01 with CUDA 12.8 nvcc, while the legacy build helper expects CUDA 13.1. This observed
non-SM120 environment is not an allowlisted native configuration.

Cross-device reproduction: `NOT_RUN`.
Cross-version reproduction: `NOT_RUN`.
