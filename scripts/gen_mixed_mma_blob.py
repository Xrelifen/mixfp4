#!/usr/bin/env python3
"""Emit one k_block of mixed-format MMAs as a SINGLE opaque inline-PTX blob per pattern.

WHAT THIS IS FOR
----------------
The cheap dispatch is the C++ one: a branch that encloses a whole k_tile costs ~2.4 cycles per
level, because it sits outside the loop body and so does not stop ptxas interleaving the next
k_block's LDSM loads with this k_block's MMAs. (A dispatch *inside* the body costs ~90 cycles --
see scripts/gen_mixed_mma_ptx.py.) Its limit was arm count: past 8 arms cicc stopped inlining the
specialized k_tile body, outlined it into an ABI call and spilled the accumulators to a 912-byte
frame, taking 1166 -> 33 TFLOP/s. docs/mixed_nvfp4_report.md recorded that cliff as resisting
every lever tried: -inline-threshold, __attribute__((always_inline)), and halving the code.

This moves it. The lever it uses is not code size but *statement count*: emitting a k_block's 16
MMAs as one opaque asm blob rather than 16 separate cute::gemm statements shrinks what cicc sees
for a k_tile body from 32 inline-asm statements to 2. Measured on an RTX 5090:

    arms   granule       C++ + cute::gemm        C++ + blob
      8    32x32x128     clean, 1167 TFLOP/s     clean
     16    16x32x128     STACK:912, 36.7         clean, STACK:0, 1116
     32    32x16x128     (not reached)           clean, STACK:0, 1084
     64    16x16x128     (not reached)           STACK:864, outlined

So the cliff moves from 8 arms to between 32 and 64 -- a 4x larger arm budget for the cheap
dispatch. Note the throughput still falls with arm count (code footprint), so 16 and 32 arms cost
7.5% and 10.2% against stock NVFP4 rather than the 3.3% that 8 arms costs.

K granularity stays at 128 here: the dispatch encloses the whole k_tile, which is exactly why it
is cheap. For K=64 use gen_mixed_mma_ptx.py and pay the in-body dispatch.

    python3 scripts/gen_mixed_mma_blob.py                 # 64 arms, 16x16x128
    A_ATOMS=1 B_ATOMS=4 python3 scripts/gen_mixed_mma_blob.py   # 16 arms, 16x32x128

Then build with -DMIXFP4_BLOB=1 and matching -DMIXFP4_A/B_ATOMS_PER_GRANULE.
"""
from __future__ import annotations
import os, pathlib

MMA_M, MMA_N = 2, 8
# Default to 16 arms (16 rows x 32 cols x 128 K): the cheapest point that is finer than the
# shipped 32x32x128 default and still well inside the moved cliff.
A_ATOMS = int(os.environ.get("A_ATOMS", 1))
B_ATOMS = int(os.environ.get("B_ATOMS", 4))
A_GRAN = MMA_M // A_ATOMS
B_GRAN = MMA_N // B_ATOMS
NPAT = 1 << (A_GRAN + B_GRAN)

# Measured on an RTX 5090 / CUDA 13.1: 32 arms still inlines cleanly, 64 outlines the k_tile body
# and spills the accumulators to a 864-byte frame. Emitting past that produces a correct but
# ~30x slower kernel, which is worth refusing to do silently.
if NPAT > 32:
    raise SystemExit(
        "error: %d arms (a_atoms=%d, b_atoms=%d). Past 32 arms cicc outlines the k_tile body and\n"
        "       spills the accumulators -- verified at 64 arms (STACK:864). Use fewer granules,\n"
        "       or scripts/gen_mixed_mma_ptx.py if you need K=64 granularity."
        % (NPAT, A_ATOMS, B_ATOMS))

SEL = ["0x3210", "0x3214", "0x3254", "0x3654"]
A_BASE, B_BASE, SFA_BASE, SFB_BASE = 64, 72, 88, 90
MMA = ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
       "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3")


def acc(m, n, v):
    return (m * MMA_N + n) * 4 + v


def serpentine(i):
    m = i // MMA_N
    j = i % MMA_N
    return m, (MMA_N - 1 - j) if (m & 1) else j


def emit_pattern(p):
    out = []
    for i in range(MMA_M * MMA_N):
        m, ns = serpentine(i)
        a_flag = (p >> (m // A_ATOMS)) & 1
        b_flag = (p >> (A_GRAN + ns // B_ATOMS)) & 1
        site = a_flag | (b_flag << 1)
        d = [acc(m, ns, v) for v in range(4)]
        a = [A_BASE + m * 4 + v for v in range(4)]
        b = [B_BASE + ns * 2 + v for v in range(2)]
        out.append("    prmt.b32 %%sf%d, %%%d, %%%d, %s;"
                   % (i, SFA_BASE + m, SFA_BASE + m, SEL[site]))
        out.append(
            "    %s {%%%d,%%%d,%%%d,%%%d}, {%%%d,%%%d,%%%d,%%%d}, {%%%d,%%%d}, "
            "{%%%d,%%%d,%%%d,%%%d}, {%%sf%d}, {0, 0}, {%%%d}, {0, 0};"
            % (MMA, d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1],
               d[0], d[1], d[2], d[3], i, SFB_BASE + ns))
    return out


def main():
    outs = ",\n".join(
        '        "+f"(acc(cute::Int<%d>{}, cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m, n)
        for m in range(MMA_M) for n in range(MMA_N) for v in range(4))
    ins = []
    for m in range(MMA_M):
        for v in range(4):
            ins.append('        "r"(a(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m))
    for n in range(MMA_N):
        for v in range(2):
            ins.append('        "r"(b(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, n))
    for m in range(MMA_M):
        ins.append('        "r"(sfa(cute::Int<0>{}, cute::Int<%d>{}))' % m)
    for n in range(MMA_N):
        ins.append('        "r"(sfb(cute::Int<0>{}, cute::Int<%d>{}))' % n)
    ins_s = ",\n".join(ins)

    arms = []
    for pat in range(NPAT):
        asm = ["{", "  .reg .b32 %%sf<%d>;" % (MMA_M * MMA_N)] + emit_pattern(pat) + ["}"]
        body = "\n".join('      "%s\\n"' % ln for ln in asm)
        kw = "if" if pat == 0 else "else if"
        arms.append("  %s constexpr (P == %d) {\n    asm volatile(\n%s\n      :\n%s\n      :\n%s);\n  }"
                    % (kw, pat, body, outs, ins_s))

    header = (
        "// GENERATED by gen_blob.py -- do not edit.\n"
        "// One k_block of mixed-format MMAs as a SINGLE opaque inline-PTX blob whose formats are\n"
        "// all fixed at compile time by P. No dispatch inside; the caller's per-k_tile C++\n"
        "// pattern specialization picks P.\n"
        "//\n"
        "// Granule: %d rows of A x %d columns of B x 128 K (one k_tile), %d arms.\n"
        "#pragma once\n"
        "#include \"cute/tensor.hpp\"\n"
        "namespace mixfp4 {\n"
        "template <uint32_t P, class TAcc, class TA, class TB, class TSFA, class TSFB>\n"
        "CUTLASS_DEVICE void\n"
        "mma_kblock_blob(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb) {\n"
        "%s\n"
        "}\n"
        "} // namespace mixfp4\n" % (A_ATOMS * 16, B_ATOMS * 8, NPAT, "\n".join(arms)))

    dst = pathlib.Path("/home/brian/mixfp4/.claude/worktrees/mixfp4-16x8x64/"
                       "src/collective/mixed_mma_blob_generated.hpp")
    dst.write_text(header)
    print("wrote %s: %d arms x %d MMAs, granule %d x %d x 128"
          % (dst.name, NPAT, MMA_M * MMA_N, A_ATOMS * 16, B_ATOMS * 8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
