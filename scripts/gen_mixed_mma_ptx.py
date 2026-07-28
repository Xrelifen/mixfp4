#!/usr/bin/env python3
"""Generate the mixed-format MMA block for one k_block as inline PTX.

WHY THIS EXISTS
---------------
The format (E0M3 vs E2M1) is two bits of the mma.sync *instruction encoding*, so choosing it at
runtime means choosing among instructions. Four ways to do that, all measured on an RTX 5090 at
4096 cubed against stock NVFP4's 1207 TFLOP/s:

  - branch per MMA          -> ptxas if-converts it; a predicated-off OMMA still consumes a
                               tensor-pipe issue slot, so tensor issues double. 504 TFLOP/s.
  - brx.idx per MMA         -> defeats if-conversion (tensor issues back to exactly 1x) but each
                               dispatch is a constant-bank LDC + BRX + WARPSYNC.ALL. 296 TFLOP/s.
  - specialize the whole    -> no runtime dispatch at all, but 2^n copies of the k_tile body.
    k_tile in C++              1167 at 8 arms; cicc outlines past that (see gen_blob.py, which
                               pushes the same idea to 32 arms by hiding each k_block in a blob).
  - ONE brx.idx per k_block -> what this file emits. Dispatch cost is O(1) in granularity, so the
    indexed by the whole       granule can go to the hardware floor. 865 TFLOP/s at 64 targets.
    flag pattern

THE COST, MEASURED
------------------
Any dispatch placed *inside* the k_tile body costs about 90 cycles, and that is NOT the branch
opcode. Three measurements pin it down:

  - the same 16 MMAs in one opaque asm blob with NO dispatch at all runs at 1189.9, exactly the
    dispatch-free ceiling -- so the opaque blob itself is free;
  - hoisting the index computation out of the asm buys only +18 TFLOP/s, so it is not the
    bfe/mad -> LDC dependency chain either;
  - replacing brx.idx with a balanced bra tree (plus a bar.warp.sync per arm to stop ptxas
    if-converting it) is *worse*, 751 vs 865.

What is left is that a branch inside the loop body is a scheduling barrier: ptxas can no longer
interleave the next k_block's LDSM shared-memory loads with this k_block's MMAs, so the shared
memory latency stops being hidden. That is why the per-k_tile C++ dispatch costs 3.3% for the
same work while a per-k_block dispatch costs 28%.

Consequence for granularity: a K granule of 64 requires a dispatch per k_block and therefore
costs >=20%, no matter how the branch is spelled. K=128 keeps the dispatch outside the body, and
then the budget is set by arm count instead -- see the table in the report.

SPLITTING THE TABLE
-------------------
At the hardware floor (A_ATOMS=1, B_ATOMS=1) a warp's k_block carries 2 A flags + 8 B flags = 10
bits, so one table indexed by the whole pattern would be 1024 arms x 16 MMAs = 16384 mma.sync per
asm statement. Instead, partition the warp's MMA_M x MMA_N atom grid into groups of
(m_per_group x n_per_group) atoms and give each group its own brx.idx over only the flags its own
MMAs need. Code size becomes sum over groups of 2^bits_g x MMAs_g, at the price of one extra
indirect branch per group.

Emits src/collective/mixed_mma_generated.hpp.
"""

from __future__ import annotations

import argparse
import pathlib

# Per-site identity-prmt selectors. prmt.b32 d,a,a,sel is the identity for any selector whose
# nibbles are (0|4),(1|5),(2|6),(3|7); four distinct spellings keep the four sites textually
# distinct so ptxas cannot tail-merge them, and give the SASS patcher its per-site tag.
SEL = ["0x3210", "0x3214", "0x3254", "0x3654"]

MMA = ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
       "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3")


class Config:
    def __init__(self, mma_m, mma_n, a_atoms, b_atoms, mpg, npg):
        self.mma_m, self.mma_n = mma_m, mma_n
        self.a_atoms, self.b_atoms = a_atoms, b_atoms
        self.mpg, self.npg = mpg, npg
        for name, big, small in (("m_per_group", mma_m, mpg), ("n_per_group", mma_n, npg)):
            if big % small:
                raise SystemExit(f"error: {name}={small} must divide MMA_{name[0].upper()}={big}")
        if mpg % a_atoms:
            raise SystemExit(f"error: m_per_group={mpg} must be a multiple of a_atoms={a_atoms}")
        if npg % b_atoms:
            raise SystemExit(f"error: n_per_group={npg} must be a multiple of b_atoms={b_atoms}")
        self.groups_m = mma_m // mpg
        self.groups_n = mma_n // npg
        self.groups = self.groups_m * self.groups_n
        # Flags this group's own MMAs need: its A granules, then its B granules.
        self.a_gran = mpg // a_atoms
        self.b_gran = npg // b_atoms
        self.bits = self.a_gran + self.b_gran
        self.npat = 1 << self.bits
        self.mmas_per_arm = mpg * npg
        self.per_stmt = self.npat * self.mmas_per_arm
        self.per_kblock = self.per_stmt * self.groups
        # 2 k_blocks x {main loop body, tail body} = 4 instantiations of a k_block in the kernel.
        self.in_kernel = self.per_kblock * 4
        if self.per_stmt > 4096:
            raise SystemExit(
                f"error: {self.npat} arms x {self.mmas_per_arm} MMAs = {self.per_stmt} mma.sync in "
                f"one asm statement. That is far past what is buildable; raise n_per_group.")

        # Inline-asm operand numbering, per group (each statement takes only its own fragments).
        n_out = 4 * mpg * npg
        self.acc_base = 0
        self.a_base = n_out
        self.b_base = self.a_base + 4 * mpg
        self.sfa_base = self.b_base + 2 * npg
        self.sfb_base = self.sfa_base + mpg
        self.ix_operand = self.sfb_base + npg
        self.n_operands = self.ix_operand + 1

    def acc(self, mi, ni, v):
        return (mi * self.npg + ni) * 4 + v


def emit_arm(cfg: Config, gm: int, gn: int, pattern: int) -> list[str]:
    """One dispatch target: every MMA in this group, formats fixed by `pattern`.

    Traversal follows CuTe's row-major serpentine (cute/algorithm/gemm.hpp) restricted to the
    group, using the GLOBAL m index for parity so the m=0 half ascends and the m=1 half descends
    over the group's n-range -- that is what keeps the B operand held across the mid-arm turn.
    """
    out = []
    i = 0
    for mi in range(cfg.mpg):
        m_global = gm * cfg.mpg + mi
        n_range = range(cfg.npg) if (m_global & 1) == 0 else range(cfg.npg - 1, -1, -1)
        for ni in n_range:
            a_flag = (pattern >> (mi // cfg.a_atoms)) & 1
            b_flag = (pattern >> (cfg.a_gran + ni // cfg.b_atoms)) & 1
            site = a_flag | (b_flag << 1)
            d = [cfg.acc(mi, ni, v) for v in range(4)]
            a = [cfg.a_base + mi * 4 + v for v in range(4)]
            b = [cfg.b_base + ni * 2 + v for v in range(2)]
            sfa = cfg.sfa_base + mi
            sfb = cfg.sfb_base + ni
            out.append(f"    prmt.b32 %sf{i}, %{sfa}, %{sfa}, {SEL[site]};")
            out.append(
                f"    {MMA} "
                f"{{%{d[0]},%{d[1]},%{d[2]},%{d[3]}}}, "
                f"{{%{a[0]},%{a[1]},%{a[2]},%{a[3]}}}, "
                f"{{%{b[0]},%{b[1]}}}, "
                f"{{%{d[0]},%{d[1]},%{d[2]},%{d[3]}}}, "
                f"{{%sf{i}}}, {{0, 0}}, {{%{sfb}}}, {{0, 0}};")
            i += 1
    return out


def build_group_asm(cfg: Config, gm: int, gn: int) -> list[str]:
    g = gm * cfg.groups_n + gn
    L = ["{", f"  .reg .b32 %sf<{cfg.mmas_per_arm}>;"]
    targets = ", ".join(f"G{g}_P{p}" for p in range(cfg.npat))
    L.append(f"  G{g}_TGT: .branchtargets {targets};")
    L.append(f"  brx.idx.uni %{cfg.ix_operand}, G{g}_TGT;")
    for p in range(cfg.npat):
        L.append(f"  G{g}_P{p}:")
        L.extend(emit_arm(cfg, gm, gn, p))
        if p != cfg.npat - 1:
            L.append(f"    bra.uni G{g}_DONE;")
    L.append(f"  G{g}_DONE:")
    L.append("}")
    return L


def emit_statement(cfg: Config, gm: int, gn: int) -> str:
    """One asm volatile for one group, with its own operand list."""
    asm = build_group_asm(cfg, gm, gn)
    body = "\n".join(f'      "{line}\\n"' for line in asm)

    m0, n0 = gm * cfg.mpg, gn * cfg.npg
    outs = ",\n".join(
        f'        "+f"(acc(cute::Int<{v}>{{}}, cute::Int<{m0 + mi}>{{}}, cute::Int<{n0 + ni}>{{}}))'
        for mi in range(cfg.mpg) for ni in range(cfg.npg) for v in range(4))
    ins = []
    for mi in range(cfg.mpg):
        for v in range(4):
            ins.append(f'        "r"(a(cute::Int<{v}>{{}}, cute::Int<{m0 + mi}>{{}}))')
    for ni in range(cfg.npg):
        for v in range(2):
            ins.append(f'        "r"(b(cute::Int<{v}>{{}}, cute::Int<{n0 + ni}>{{}}))')
    for mi in range(cfg.mpg):
        ins.append(f'        "r"(sfa(cute::Int<0>{{}}, cute::Int<{m0 + mi}>{{}}))')
    for ni in range(cfg.npg):
        ins.append(f'        "r"(sfb(cute::Int<0>{{}}, cute::Int<{n0 + ni}>{{}}))')
    ins.append(f'        "r"(ix[{gm * cfg.groups_n + gn}])')
    ins_s = ",\n".join(ins)

    return f"""  asm volatile(
{body}
      :
{outs}
      :
{ins_s});"""


def emit_index_helper(cfg: Config) -> str:
    """Per-group jump indices, computed in C++ so ptxas can hoist them clear of the branches.

    Bit layout per group: its A granule flags in the low bits, its B granule flags above them --
    the same order emit_arm() decodes.
    """
    lines = []
    fast = []
    for gm in range(cfg.groups_m):
        for gn in range(cfg.groups_n):
            g = gm * cfg.groups_n + gn
            terms = []
            srcs = []          # the flag-carrying words, in ascending index-bit order
            for ga in range(cfg.a_gran):
                m = gm * cfg.mpg + ga * cfg.a_atoms
                terms.append(f"(((sfa(cute::Int<0>{{}}, cute::Int<{m}>{{}}) >> 7) & 1u) << {ga})")
                srcs.append(f"sfa(cute::Int<0>{{}}, cute::Int<{m}>{{}})")
            for gb in range(cfg.b_gran):
                n = gn * cfg.npg + gb * cfg.b_atoms
                terms.append(f"(((sfb(cute::Int<0>{{}}, cute::Int<{n}>{{}}) >> 7) & 1u) "
                             f"<< {cfg.a_gran + gb})")
                srcs.append(f"sfb(cute::Int<0>{{}}, cute::Int<{n}>{{}})")
            lines.append(f"  ix[{g}] = {' | '.join(terms)};")
            fast.append(emit_fast_index(g, srcs))

    # TIMING PROBE ONLY -- produces WRONG NUMBERS by construction. Isolates the cost of *reading*
    # the flags from the cost of the branch. The index MUST be warp-uniform: brx.idx.uni with a
    # divergent index is undefined behaviour, and in practice wedges the GPU (an operand register
    # was tried here first -- it is per-lane, and it hung the card).
    probe = [f"  ix[{g}] = blockIdx.x & {cfg.npat - 1}u;" for g in range(cfg.groups)]
    # The naive form is the DEFAULT and it is the faster one, which is not what instruction
    # counting predicts. Its N extractions are mutually independent, so they overlap; the PRMT
    # gather below replaces them with fewer instructions on a single serial chain
    # (PRMT -> PRMT -> AND -> SHR -> IMAD -> SHR -> AND). Measured at 16x16x64, 4096^3:
    # naive 887.6 TFLOP/s, PRMT-gathered 878.1. ILP beats instruction count here.
    return ("#if defined(MIXFP4_FAKE_INDEX) && MIXFP4_FAKE_INDEX\n"
            + "\n".join(probe)
            + "\n#elif defined(MIXFP4_PRMT_INDEX) && MIXFP4_PRMT_INDEX\n"
            + "\n".join(fast)
            + "\n#else\n"
            + "\n".join(lines)
            + "\n#endif")


def emit_fast_index(g: int, srcs: list[str]) -> str:
    """Same index, built with PRMT byte-gathers and a multiply instead of one shift/mask/or per flag.

    Every flag is bit 7 of byte 0 of its own scale word, so the naive form touches N registers with
    a shift+mask+or each. ncu puts the whole computation at 39.2 instructions per k_tile across the
    two dispatches -- 51 of the 176 instructions a k_tile executes, against a 125-instruction
    no-dispatch ceiling, and this kernel is instruction-bound there (its cycles-per-instruction is
    *below* the ceiling's). So the index is the single biggest term, not the branch.

    PRMT gathers four flag-carrying bytes into one word for free, and one multiply compacts four
    set MSBs into four adjacent bits: with y = (w & 0x80808080) >> 7 holding 0/1 in bytes 0..3,
    y * 0x01020408 lands those bits at 24..27, because the multiplier's 2^24/2^17/2^10/2^3 terms
    carry byte i up to bit 24+i and nothing below 2^24 can carry into them.
    """
    n = len(srcs)
    out = []
    # Gather in groups of four: two PRMTs to pair the byte-0s, one to merge the pairs.
    words = []
    for base in range(0, n, 4):
        chunk = srcs[base:base + 4]
        if len(chunk) == 1:
            words.append((f"({chunk[0]})", 1))
        elif len(chunk) == 2:
            words.append((f"__byte_perm({chunk[0]}, {chunk[1]}, 0x0040)", 2))
        elif len(chunk) == 3:
            words.append((f"__byte_perm(__byte_perm({chunk[0]}, {chunk[1]}, 0x0040), "
                          f"{chunk[2]}, 0x4010)", 3))
        else:
            words.append((f"__byte_perm(__byte_perm({chunk[0]}, {chunk[1]}, 0x0040), "
                          f"__byte_perm({chunk[2]}, {chunk[3]}, 0x0040), 0x5410)", 4))
    MASK = {1: "0x00000080u", 2: "0x00008080u", 3: "0x00808080u", 4: "0x80808080u"}
    MUL  = {1: "0x00000001u", 2: "0x00000102u", 3: "0x00010204u", 4: "0x01020408u"}
    SHR  = {1: 0, 2: 8, 3: 16, 4: 24}
    # The multiply's partial products spill above the field being extracted, so the shifted result
    # MUST be masked. Without this the two-flag case yields 0x103 rather than 0x3, ix runs past the
    # end of the .branchtargets table, and brx.idx jumps to garbage -- which hangs the GPU rather
    # than returning wrong numbers.
    KEEP = {1: "1u", 2: "0x3u", 3: "0x7u", 4: "0xFu"}
    for i, (expr, cnt) in enumerate(words):
        piece = (f"((((({expr}) & {MASK[cnt]}) >> 7) * {MUL[cnt]} >> {SHR[cnt]}) & {KEEP[cnt]})"
                 if cnt > 1 else f"((({expr}) >> 7) & 1u)")
        out.append(piece if i == 0 else f"({piece} << {4 * i})")
    return f"  ix[{g}] = {' | '.join(out)};"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mma-m", type=int, default=2, help="m-atoms in a warp's tile (default 2)")
    ap.add_argument("--mma-n", type=int, default=8, help="n-atoms in a warp's tile (default 8)")
    ap.add_argument("--a-atoms", type=int, default=1, help="m-atoms per A granule; 1 = 16 rows")
    ap.add_argument("--b-atoms", type=int, default=1, help="n-atoms per B granule; 1 = 8 columns")
    ap.add_argument("--m-per-group", type=int, default=None,
                    help="m-atoms per dispatch group (default: all of MMA_M)")
    ap.add_argument("--n-per-group", type=int, default=2,
                    help="n-atoms per dispatch group (default 2)")
    ap.add_argument("--out", default=None, help="output header path")
    args = ap.parse_args()

    cfg = Config(args.mma_m, args.mma_n, args.a_atoms, args.b_atoms,
                 args.m_per_group if args.m_per_group is not None else args.mma_m,
                 args.n_per_group)

    statements = "\n\n".join(emit_statement(cfg, gm, gn)
                             for gm in range(cfg.groups_m) for gn in range(cfg.groups_n))

    header = f'''// GENERATED by scripts/gen_mixed_mma_ptx.py -- do not edit. Regenerate with:
//   python3 scripts/gen_mixed_mma_ptx.py --a-atoms {cfg.a_atoms} --b-atoms {cfg.b_atoms} \\
//       --m-per-group {cfg.mpg} --n-per-group {cfg.npg}
//
// One k_block of mixed-format MMAs ({cfg.mma_m}x{cfg.mma_n} atoms = {cfg.mma_m * cfg.mma_n} MMAs)
// as {cfg.groups} inline-PTX statement(s). Each covers a {cfg.mpg}x{cfg.npg}-atom group and opens
// with its own {cfg.npat}-target brx.idx over just that group's {cfg.bits} format flags, followed
// by straight-line mma.sync with every format fixed at assembly time.
//
// Granule: {cfg.a_atoms * 16} rows of A x {cfg.b_atoms * 8} columns of B x 64 K.
//   (16 rows / 8 columns / 64 K is the hardware floor -- the atom is m16n8k64.)
//
// Dispatch cost: {cfg.groups} indirect branch(es) per k_block. Each one inside the k_tile body
// costs ~90 cycles, because it stops ptxas interleaving the next k_block's LDSM loads with this
// k_block's MMAs -- see the generator docstring. Code: {cfg.per_stmt} mma.sync per statement,
// {cfg.in_kernel} in the kernel.
#pragma once

#include "cute/tensor.hpp"

#define MIXFP4_GEN_MMA_M        {cfg.mma_m}
#define MIXFP4_GEN_MMA_N        {cfg.mma_n}
#define MIXFP4_GEN_A_ATOMS      {cfg.a_atoms}
#define MIXFP4_GEN_B_ATOMS      {cfg.b_atoms}
#define MIXFP4_GEN_M_PER_GROUP  {cfg.mpg}
#define MIXFP4_GEN_N_PER_GROUP  {cfg.npg}
#define MIXFP4_GEN_GROUPS       {cfg.groups}
#define MIXFP4_GEN_K_GRANULE    64
#define MIXFP4_GEN_OMMAS_PER_KBLOCK {cfg.per_kblock}

namespace mixfp4 {{

// The jump index, split out so the caller can compute it a k_block EARLY.
//
// It reads only the scale-factor fragments, which copy_kblock() has already loaded for k_block+1
// by the time k_block's MMAs run. Computing it there puts the whole extract/pack chain, and the
// IMAD -> LDC -> BRX that consumes it, off the critical path into the branch. See
// MIXFP4_PIPE_IX in the mainloop.
template <class TSFA, class TSFB>
CUTLASS_DEVICE void
mma_index(TSFA const& sfa, TSFB const& sfb, uint32_t (&ix)[{cfg.groups}]) {{
{emit_index_helper(cfg)}
}}

// acc: (MMA,MMA_M,MMA_N) accumulator fragment.  a: (V,MMA_M) uint32.  b: (V,MMA_N) uint32.
// sfa: (1,MMA_M) uint32.  sfb: (1,MMA_N) uint32.  All indices are compile-time constants.
template <class TAcc, class TA, class TB, class TSFA, class TSFB>
CUTLASS_DEVICE void
mma_kblock_ix(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb,
              uint32_t const (&ix)[{cfg.groups}]) {{
{statements}
}}

template <class TAcc, class TA, class TB, class TSFA, class TSFB>
CUTLASS_DEVICE void
mma_kblock(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb) {{
  uint32_t ix[{cfg.groups}];
  mma_index(sfa, sfb, ix);
  mma_kblock_ix(acc, a, b, sfa, sfb, ix);
}}

}} // namespace mixfp4
'''

    dst = pathlib.Path(args.out) if args.out else (
        pathlib.Path(__file__).resolve().parent.parent / "src/collective/mixed_mma_generated.hpp")
    dst.write_text(header)
    print(f"wrote {dst}")
    print(f"  granule           {cfg.a_atoms * 16} rows x {cfg.b_atoms * 8} cols x 64 K")
    print(f"  groups            {cfg.groups} ({cfg.mpg}x{cfg.npg} atoms each) "
          f"-> {cfg.groups} brx.idx per k_block")
    print(f"  arms per group    {cfg.npat} ({cfg.bits} flag bits)")
    print(f"  mma.sync          {cfg.per_stmt} per statement, {cfg.per_kblock} per k_block, "
          f"{cfg.in_kernel} in the kernel")
    print(f"  expected per-site OMMA count (patcher census): {cfg.in_kernel // 4}")
    print(f"  operands          {cfg.n_operands} per statement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
