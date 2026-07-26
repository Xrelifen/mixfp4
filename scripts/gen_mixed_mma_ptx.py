#!/usr/bin/env python3
"""Generate the 16x16-granularity mixed-format MMA block as a single inline-PTX statement.

WHY THIS EXISTS
---------------
The format (E0M3 vs E2M1) is two bits of the mma.sync *instruction encoding*, so choosing it at
runtime means choosing among instructions. Three ways to do that, all measured on an RTX 5090:

  - branch per MMA          -> ptxas if-converts it; a predicated-off OMMA still consumes a
                               tensor-pipe issue slot, so tensor issues double. 504 TFLOP/s.
  - brx.idx per MMA         -> defeats if-conversion (tensor issues back to exactly 1x) but each
                               dispatch is a constant-bank LDC + BRX + WARPSYNC.ALL. 296 TFLOP/s.
  - specialize on the whole -> no runtime dispatch at all, but 2^n copies of the loop body. cicc
    flag pattern in C++        stops inlining past 8 arms and outlines it into an ABI call that
                               spills the accumulators to a 912-byte frame. 1166 -> 37 TFLOP/s.

This generator takes the third approach's zero-runtime-cost property and the second's immunity to
if-conversion, while dodging the outlining cliff:

  ONE brx.idx, indexed by the ENTIRE flag pattern, dispatching to 2^n blocks of straight-line
  mma.sync -- all inside a SINGLE asm statement.

The load-bearing observation is that a brx.idx costs the same with 4 targets as with 64: it is a
jump-table lookup. Indexing by the whole pattern instead of by one sub-block's format makes the
dispatch cost O(1) in granularity -- one indirect branch per 16 MMAs rather than one per 2 -- so
granularity is bounded only by code size. And because the whole thing is one asm statement, cicc
sees an opaque blob with no function to outline, so there is no ABI and no spill.

16 rows x 16 columns is the hardware floor regardless: the atom is m16n8k64, so A can never be
finer than 16 rows.

Emits src/collective/mixed_mma_16x16_generated.hpp.
"""

from __future__ import annotations

import pathlib

# Warp tile for the default Shape<_128,_128,_128> / 4x2 warp arrangement: 2 m-atoms x 8 n-atoms.
MMA_M = 2
MMA_N = 8

# Atoms per format granule. 1 A-atom = 16 rows, 2 B-atoms = 16 columns.
A_ATOMS = 1
B_ATOMS = 2

A_GRAN = MMA_M // A_ATOMS          # 2 independently-flagged A granules
B_GRAN = MMA_N // B_ATOMS          # 4 independently-flagged B granules
NPAT = 1 << (A_GRAN + B_GRAN)      # 64 patterns

# Per-site identity-prmt selectors. prmt.b32 d,a,a,sel is the identity for any selector whose
# nibbles are (0|4),(1|5),(2|6),(3|7); four distinct spellings keep the four sites textually
# distinct so ptxas cannot tail-merge them, and give the SASS patcher its per-site tag.
SEL = ["0x3210", "0x3214", "0x3254", "0x3654"]

# Inline-asm operand numbering.
#   %0..%63   accumulators, (m,n,v) -> (m*MMA_N + n)*4 + v      ["+f"]
#   %64..%71  A fragments,  (m,v)   -> 64 + m*4 + v             ["r"]
#   %72..%87  B fragments,  (n,v)   -> 72 + n*2 + v             ["r"]
#   %88..%89  SFA words,    m       -> 88 + m                   ["r"]
#   %90..%97  SFB words,    n       -> 90 + n                   ["r"]
A_BASE, B_BASE, SFA_BASE, SFB_BASE = 64, 72, 88, 90


def acc(m: int, n: int, v: int) -> int:
    return (m * MMA_N + n) * 4 + v


def serpentine(i: int) -> tuple[int, int]:
    """CuTe's dispatch-[4] traversal order (cute/algorithm/gemm.hpp): row-major serpentine, which
    exists to maximise operand register reuse between consecutive MMAs. Reproducing it keeps the
    .reuse flags ptxas emits the same as the stock kernel's."""
    m = i // MMA_N
    j = i % MMA_N
    ns = (MMA_N - 1 - j) if (m & 1) else j
    return m, ns


MMA = ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
       "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3")


def emit_pattern(p: int) -> list[str]:
    """One dispatch target: 16 mma.sync whose formats are all fixed by pattern p."""
    out = []
    for i in range(MMA_M * MMA_N):
        m, ns = serpentine(i)
        a_flag = (p >> m) & 1                       # A granule = this m-atom (16 rows)
        b_flag = (p >> (A_GRAN + ns // B_ATOMS)) & 1  # B granule = 2 n-atoms (16 columns)
        site = a_flag | (b_flag << 1)
        d = [acc(m, ns, v) for v in range(4)]
        a = [A_BASE + m * 4 + v for v in range(4)]
        b = [B_BASE + ns * 2 + v for v in range(2)]
        out.append(f"    prmt.b32 %sf{i}, %{SFA_BASE + m}, %{SFA_BASE + m}, {SEL[site]};")
        out.append(
            f"    {MMA} "
            f"{{%{d[0]},%{d[1]},%{d[2]},%{d[3]}}}, "
            f"{{%{a[0]},%{a[1]},%{a[2]},%{a[3]}}}, "
            f"{{%{b[0]},%{b[1]}}}, "
            f"{{%{d[0]},%{d[1]},%{d[2]},%{d[3]}}}, "
            f"{{%sf{i}}}, {{0, 0}}, {{%{SFB_BASE + ns}}}, {{0, 0}};"
        )
    return out


def build_asm() -> list[str]:
    L = []
    L.append("{")
    L.append(f"  .reg .b32 %sf<{MMA_M * MMA_N}>;")
    L.append("  .reg .b32 %fa0,%fa1,%fb0,%fb1,%fb2,%fb3,%ix;")
    # Extract one flag bit per granule from bit 7 of that granule's first scale-factor word.
    for g in range(A_GRAN):
        L.append(f"  bfe.u32 %fa{g}, %{SFA_BASE + g * A_ATOMS}, 7, 1;")
    for g in range(B_GRAN):
        L.append(f"  bfe.u32 %fb{g}, %{SFB_BASE + g * B_ATOMS}, 7, 1;")
    # Pack into the pattern index: A granules in the low bits, B granules above them.
    L.append("  mov.b32 %ix, %fa0;")
    for g in range(1, A_GRAN):
        L.append(f"  mad.lo.u32 %ix, %fa{g}, {1 << g}, %ix;")
    for g in range(B_GRAN):
        L.append(f"  mad.lo.u32 %ix, %fb{g}, {1 << (A_GRAN + g)}, %ix;")
    targets = ", ".join(f"P{p}" for p in range(NPAT))
    L.append(f"  TGT: .branchtargets {targets};")
    L.append("  brx.idx.uni %ix, TGT;")
    for p in range(NPAT):
        L.append(f"  P{p}:")
        L.extend(emit_pattern(p))
        if p != NPAT - 1:
            L.append("    bra.uni DONE;")
    L.append("  DONE:")
    L.append("}")
    return L


def main() -> int:
    asm = build_asm()
    body = "\n".join(f'      "{line}\\n"' for line in asm)

    outs = ",\n".join(
        f'        "+f"(acc(cute::Int<{v}>{{}}, cute::Int<{m}>{{}}, cute::Int<{n}>{{}}))'
        for m in range(MMA_M) for n in range(MMA_N) for v in range(4)
    )
    ins = []
    for m in range(MMA_M):
        for v in range(4):
            ins.append(f'        "r"(a(cute::Int<{v}>{{}}, cute::Int<{m}>{{}}))')
    for n in range(MMA_N):
        for v in range(2):
            ins.append(f'        "r"(b(cute::Int<{v}>{{}}, cute::Int<{n}>{{}}))')
    for m in range(MMA_M):
        ins.append(f'        "r"(sfa(cute::Int<0>{{}}, cute::Int<{m}>{{}}))')
    for n in range(MMA_N):
        ins.append(f'        "r"(sfb(cute::Int<0>{{}}, cute::Int<{n}>{{}}))')
    ins_s = ",\n".join(ins)

    header = f'''// GENERATED by scripts/gen_mixed_mma_ptx.py -- do not edit. Regenerate with:
//   python3 scripts/gen_mixed_mma_ptx.py
//
// One k_block of mixed-format MMAs ({MMA_M}x{MMA_N} atoms = {MMA_M * MMA_N} MMAs) as a single
// inline-PTX statement: a {NPAT}-target brx.idx indexed by the whole format flag pattern, then
// straight-line mma.sync with every format fixed at assembly time.
//
// Granule: {A_ATOMS * 16} rows of A x {B_ATOMS * 8} columns of B -- the hardware floor, since the
// atom is m16n8k64.
//
// Why one big indirect branch rather than a branch per MMA, or C++ specialization: see the
// generator's docstring. The short version is that a brx.idx costs the same with 64 targets as
// with 4, so indexing by the whole pattern makes dispatch cost O(1) in granularity; and keeping
// it inside one asm statement leaves cicc no function to outline, which is what spilled the
// accumulators when this was written as 64 C++ arms.
#pragma once

#include "cute/tensor.hpp"

namespace mixfp4 {{

// acc: (MMA,MMA_M,MMA_N) accumulator fragment.  a: (V,MMA_M) uint32.  b: (V,MMA_N) uint32.
// sfa: (1,MMA_M) uint32.  sfb: (1,MMA_N) uint32.  All indices are compile-time constants.
template <class TAcc, class TA, class TB, class TSFA, class TSFB>
CUTLASS_DEVICE void
mma_kblock_16x16(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb) {{
  asm volatile(
{body}
      :
{outs}
      :
{ins_s});
}}

}} // namespace mixfp4
'''
    dst = pathlib.Path(__file__).resolve().parent.parent / "src/collective/mixed_mma_16x16_generated.hpp"
    dst.write_text(header)
    n_mma = NPAT * MMA_M * MMA_N
    print(f"wrote {dst}")
    print(f"  {NPAT} patterns x {MMA_M * MMA_N} MMAs = {n_mma} mma.sync per asm statement")
    print(f"  {len(asm)} PTX lines, granule {A_ATOMS * 16} rows x {B_ATOMS * 8} cols")
    print(f"  operands: 64 accumulators (+f) + 34 inputs (r) = 98")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
