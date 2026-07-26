"""Shared autotune config space and pruner for the three mixfp4 kernels.

The pruner follows GemLite's design, where it is a config *rewriter* rather than a filter: it
clamps each block shape to the actual problem, snaps BLOCK_K to whole scale groups, walks
``num_stages`` down until the tile fits in shared memory, then dedupes.  A filter would throw away
most of the space on small problems; rewriting folds it onto the handful of shapes that make sense.
"""

from __future__ import annotations

import triton

from .config import cached_config
from .utils import estimate_shared_memory_per_block, get_gpu_shared_memory, next_power_of_2

#: Names of the tunable fields, in the order they are stored in the JSON cache.
BASE_FIELDS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_STAGES",
               "A_load_order")


def build_configs(mode: str, *, split_k=(1,), block_m=None, block_n=None, block_k=None,
                  group_size_m=(8,), zero_output: bool = False):
    """Cross-product config space.  ``mode`` selects how much of it to explore."""
    if mode == "max":
        bm = block_m or [16, 32, 64, 128, 256]
        bn = block_n or [32, 64, 128, 256]
        bk = block_k or [32, 64, 128, 256]
        stages, warps, loads = [1, 2, 3, 4], [4, 8], [0, 2]
    elif mode == "fast":
        bm = block_m or [16, 32, 64, 128]
        bn = block_n or [64, 128, 256]
        bk = block_k or [32, 64, 128]
        stages, warps, loads = [2, 3, 4], [4, 8], [0, 2]
    else:
        bm = block_m or [16, 64, 128]
        bn = block_n or [128, 256]
        bk = block_k or [64]
        stages, warps, loads = [3], [4], [0]

    out = []
    for m in bm:
        for n in bn:
            for k in bk:
                for g in group_size_m:
                    for s in stages:
                        for w in warps:
                            for lo in loads:
                                for sk in split_k:
                                    fields = {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n,
                                              "BLOCK_SIZE_K": k, "GROUP_SIZE_M": g,
                                              "NUM_STAGES": s, "A_load_order": lo}
                                    if split_k != (1,) or sk > 1:
                                        fields["SPLIT_K"] = sk
                                    pre_hook = _zero_output if (zero_output or sk > 1) else None
                                    out.append(triton.Config(fields, num_warps=w, num_stages=s,
                                                             pre_hook=pre_hook))
    return out


def _zero_output(nargs):
    """Split-K accumulates with atomics, so the output must start at zero.

    Triton calls this immediately before every launch, including the real launch that follows a
    tuning sweep -- which matters, because the sweep leaves partial sums in the same buffer.
    """
    nargs["c_ptr"].zero_()


def make_pruner(matmul_type: str, *, with_split_k: bool = False, force_block_m: int | None = None,
                min_block_k: int = 32, zero_output: bool = False):
    """Build the ``early_config_prune`` callable for one kernel.

    Args:
        matmul_type: cache bucket name, e.g. ``"GEMM"``.
        with_split_k: keep and clamp a ``SPLIT_K`` field.
        force_block_m: pin BLOCK_SIZE_M (the GEMV kernel processes one row per program).
        zero_output: attach the zeroing pre-hook to every config, not only split ones.  The GEMV
            kernel always accumulates atomically because its K loop lives in the grid.
        min_block_k: floor for BLOCK_SIZE_K.  ``tl.dot`` wants at least 32; the GEMV kernel,
            which reduces elementwise, is happy with 16.
    """
    fields = BASE_FIELDS + (("SPLIT_K",) if with_split_k else ())

    def prune(configs, nargs, **kwargs):
        m, n, k = nargs["M"], nargs["N"], nargs["K"]
        # Triton passes runtime args in ``nargs`` and constexprs in ``**kwargs``.
        a_sizeof = kwargs.get("a_sizeof", nargs.get("a_sizeof", 2))
        max_smem = get_gpu_shared_memory()

        hit = cached_config(matmul_type, m, n, k)
        if hit is not None and all(f in hit for f in fields):
            yield triton.Config({f: hit[f] for f in fields},
                                num_warps=hit.get("num_warps", 4),
                                num_stages=hit.get("num_stages", 3),
                                pre_hook=(_zero_output
                                          if (zero_output or hit.get("SPLIT_K", 1) > 1) else None))
            return

        seen = set()
        for cfg in configs:
            kw = dict(cfg.kwargs)
            block_m = force_block_m or max(16, min(next_power_of_2(m), kw["BLOCK_SIZE_M"]))
            block_n = max(16, min(next_power_of_2(n), kw["BLOCK_SIZE_N"]))
            # BLOCK_K must cover whole 16-element scale groups, so that the group index is
            # loop-invariant and only the base pointer advances.
            block_k = max(min_block_k, min(next_power_of_2(k), kw["BLOCK_SIZE_K"]))
            block_k = max(16, (block_k // 16) * 16)

            split_k = kw.get("SPLIT_K", 1)
            if with_split_k:
                # Never split further than there is K to split, and cap the split on larger M
                # where there are already enough tiles to fill the machine.
                split_k = max(1, min(split_k, k // max(block_k, 1)))
                if m >= 32:
                    split_k = min(split_k, 8)

            num_stages, num_warps = cfg.num_stages, cfg.num_warps
            while num_stages > 1 and estimate_shared_memory_per_block(
                    block_m, block_n, block_k, a_sizeof, num_stages) > max_smem:
                num_stages -= 1
            if estimate_shared_memory_per_block(block_m, block_n, block_k, a_sizeof,
                                                num_stages) > max_smem:
                continue

            out = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                   "GROUP_SIZE_M": kw["GROUP_SIZE_M"],
                   "NUM_STAGES": min(kw["NUM_STAGES"], num_stages),
                   "A_load_order": kw["A_load_order"]}
            if with_split_k:
                out["SPLIT_K"] = split_k

            key = tuple(out.values()) + (num_warps, num_stages)
            if key in seen:
                continue
            seen.add(key)
            yield triton.Config(out, num_warps=num_warps, num_stages=num_stages,
                                pre_hook=(_zero_output
                                          if (zero_output or split_k > 1) else None))

    return lambda configs, nargs, **kwargs: list(prune(configs, nargs, **kwargs))
