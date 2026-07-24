#!/usr/bin/env python3
"""Benchmark PyTorch's NVFP4 GEMM (torch._scaled_mm, backed by cuBLASLt) at the same
problem size as src/nvfp4_gemm.cu, for an apples-to-apples throughput comparison.

NVFP4 packs 4-bit (e2m1) elements two per byte (torch.float4_e2m1fn_x2), with one e4m3
scale factor per block of 16 elements, laid out in cuBLASLt's blocked+swizzled scale
format. Since this is a raw throughput comparison (not a cross-library correctness
check against the CUTLASS kernel's own reference), the operand/scale contents are
random garbage of the right shape/dtype -- performance for a dense GEMM doesn't depend
on the data itself.

A: MxK (packed MxK/2), row-major, float4_e2m1fn_x2
B: KxN (packed K/2xN), column-major, float4_e2m1fn_x2
D: MxN, bf16
"""

import argparse

import torch


def ceil_div(a, b):
    return (a + b - 1) // b


def make_operands(m, n, k):
    kp = k // 2

    a = torch.randint(0, 256, (m, kp), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b_row = torch.randint(0, 256, (n, kp), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = b_row.t()  # (kp, n), column-major

    # NVFP4 recipe: block size 16 (unpacked), scale dtype e4m3, blocks padded to a
    # multiple of 4 along K and tiled 128-wide along M/N (cuBLASLt's blocked scale format).
    num_k_blocks = ceil_div(k, 16)
    padded_k_blocks = ceil_div(num_k_blocks, 4) * 4
    scale_a_size = 128 * ceil_div(m, 128) * padded_k_blocks
    scale_b_size = 128 * ceil_div(n, 128) * padded_k_blocks

    scale_a = torch.randint(1, 4, (scale_a_size,), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    scale_b = torch.randint(1, 4, (scale_b_size,), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)

    return a, b, scale_a, scale_b


def benchmark(m, n, k, warmup_iters, bench_iters):
    torch.manual_seed(0)
    a, b, scale_a, scale_b = make_operands(m, n, k)

    for _ in range(warmup_iters):
        torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(bench_iters):
        torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / bench_iters
    flops = 2.0 * m * n * k
    tflops = flops / (avg_ms / 1.0e3) / 1.0e12
    return avg_ms, tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, nargs="?", default=4096)
    parser.add_argument("n", type=int, nargs="?", default=4096)
    parser.add_argument("k", type=int, nargs="?", default=4096)
    parser.add_argument("bench_iters", type=int, nargs="?", default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__} (cuda {torch.version.cuda})")

    avg_ms, tflops = benchmark(args.m, args.n, args.k, args.warmup_iters, args.bench_iters)

    print(f"Problem size: {args.m}x{args.n}x{args.k}")
    print(f"Avg latency: {avg_ms:.6f} ms")
    print(f"Throughput: {tflops:.3f} TFLOP/s (nvfp4)")


if __name__ == "__main__":
    main()
