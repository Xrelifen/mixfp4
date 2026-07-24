#!/usr/bin/env python3
"""Benchmark PyTorch's int8 GEMM (torch._int_mm, backed by cuBLASLt) at the same
problem size / operand layout as src/int8_gemm.cu, for an apples-to-apples comparison.

A: MxK, row-major, int8
B: KxN, column-major, int8
D: MxN, int32 (no epilogue scale/bias, matching alpha=1, beta=0 in the CUTLASS kernel)
"""

import argparse
import time

import torch


def benchmark(m, n, k, warmup_iters, bench_iters):
    torch.manual_seed(0)

    a = torch.randint(-8, 9, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 9, (k, n), dtype=torch.int8, device="cuda").t().contiguous().t()
    assert a.is_contiguous()
    assert not b.is_contiguous() and b.stride() == (1, k)  # column-major KxN

    for _ in range(warmup_iters):
        torch._int_mm(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(bench_iters):
        torch._int_mm(a, b)
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
    print(f"Throughput: {tflops:.3f} TFLOP/s (int8)")


if __name__ == "__main__":
    main()
