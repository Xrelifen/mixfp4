#!/usr/bin/env python3
"""Wikitext-2 perplexity for mixfp4 against NVFP4 and an fp16 baseline.

    python scripts/eval_perplexity.py --model facebook/opt-125m
    python scripts/eval_perplexity.py --model Qwen/Qwen2.5-0.5B --seqlen 2048 \
        --configs fp16 nvfp4-rtn mixed-rtn mixed-hqq

The default backend is ``simulated``: each weight is round-tripped through the format
(quantise then dequantise) and the matmul still runs in fp16.  That isolates *what the format
costs in accuracy* from *which kernel runs the matmul*, which is what a format comparison should
measure -- and it means these numbers are equally valid for the CUDA backend, which computes the
same arithmetic on the tensor core.

``--backend triton`` instead swaps in real :class:`MixFP4Linear` layers, so the same table also
serves as an end-to-end check that the kernels reproduce the simulated result.

The comparison that matters is ``nvfp4-rtn`` (ordinary NVFP4: every group E2M1, round to nearest)
against the ``mixed-*`` rows.  Same group size, same scale recipe, same everything except the
per-group codebook choice, so any difference is attributable to that choice alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from mixfp4.activation import (  # noqa: E402
    activation_config,
    install_activation_quantizers,
    remove_activation_quantizers,
)
from mixfp4.codebook import GROUP_SIZE  # noqa: E402
from mixfp4.quant import quantize_dequantize, quantize_mixfp4  # noqa: E402
from mixfp4.quantizers import QuantConfig, available_methods  # noqa: E402

#: Short label -> format policy.  The candidate encodings a group may choose between.
POLICIES = {
    "nvfp4": "nvfp4",        # E2M1, amax onto 6 -- standard NVFP4, no per-group choice
    "e2m1@4": "e2m1-4",      # E2M1, amax onto 4 everywhere -- Four Over Six's Table 3 control
    "e0m3": "e0m3",          # sign-magnitude INT4 everywhere
    "46": "nvfp4-46",        # Four Over Six: choose 6 or 4 per group
    "mixed": "mixed",        # mixfp4: choose E2M1@6 or E0M3@7 per group
    "mixed46": "mixed-46",   # all three candidates per group
}

#: Short label -> fitting method, applied once a candidate is chosen.
METHODS = {
    "rtn": "rtn",            # round to nearest at the amax-derived scale
    "search": "rtn_search",  # sweep 8 scale multipliers, keep the best
    "hqq": "hqq",            # HQQ-style half-quadratic proximal optimisation
}

#: The full grid, plus the unquantised control.
CONFIGS = {"fp16": None}
CONFIGS.update({f"{pol}-{meth}": (POLICIES[pol], METHODS[meth])
                for pol in POLICIES for meth in METHODS})

DEFAULT_CONFIGS = ["fp16"] + [f"{pol}-{meth}" for pol in POLICIES for meth in METHODS]


def quantizable_layers(model, skip: tuple[str, ...]):
    """Yield ``(name, module, transposed)`` for every 2-D weight worth quantising.

    ``transposed`` marks HF's ``Conv1D`` (used by GPT-2), whose weight is stored ``[in, out]``
    rather than ``nn.Linear``'s ``[out, in]``.  The quantiser groups along the *input* dimension,
    so getting this wrong silently groups across output channels instead and wrecks the result.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # pragma: no cover
        Conv1D = ()

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        if not isinstance(module, (torch.nn.Linear,) + ((Conv1D,) if Conv1D else ())):
            continue
        if any(s in name for s in skip):
            continue
        transposed = bool(Conv1D) and isinstance(module, Conv1D)
        in_features = weight.shape[0] if transposed else weight.shape[1]
        if in_features % GROUP_SIZE != 0:
            continue
        yield name, module, transposed


@torch.no_grad()
def apply_quantization(model, originals, policy, method, cfg_kwargs, device, verbose=False):
    """Overwrite every eligible weight with its quantise/dequantise round trip."""
    e0m3_groups = total_groups = 0
    for name, module, transposed in quantizable_layers(model, SKIP):
        original = originals[name]
        # Quantise on the accelerator a layer at a time, then put it straight back: on a memory
        # constrained GPU the model may be resident but there is no room for a second copy.
        w = original.to(device=device, dtype=torch.float32)
        if transposed:
            w = w.t().contiguous()
        q = quantize_mixfp4(w, format_policy=policy, method=method, **cfg_kwargs)
        from mixfp4.quant import dequantize_mixfp4
        recon = dequantize_mixfp4(q, dtype=torch.float32)
        e0m3_groups += q.e0m3_fraction * (q.N * q.K // GROUP_SIZE)
        total_groups += q.N * q.K // GROUP_SIZE
        if transposed:
            recon = recon.t().contiguous()
        module.weight.data.copy_(recon.to(module.weight.dtype))
        del w, q, recon
    if verbose:
        print(f"    e0m3 share: {e0m3_groups / max(total_groups, 1):.1%}")
    return e0m3_groups / max(total_groups, 1)


@torch.no_grad()
def restore(model, originals):
    for name, module, _ in quantizable_layers(model, SKIP):
        module.weight.data.copy_(originals[name].to(module.weight.dtype))


@torch.no_grad()
def _window_nll(model, batch, chunk: int) -> tuple[float, int]:
    """Total NLL over one window, materialising logits only ``chunk`` positions at a time.

    Passing ``labels=`` to a HF model builds logits for the whole window at once.  With a large
    vocabulary that dominates memory -- Qwen2.5's 152k vocab needs ~600 MB for a 2048-token window
    in fp16, and more again for the fp32 upcast inside cross-entropy, which is far more than the
    0.5B model's own weights.  Running the base model once and applying the head in slices keeps
    the peak proportional to ``chunk`` instead of to ``seqlen``.
    """
    import torch.nn.functional as F

    hidden = model.base_model(batch).last_hidden_state
    head = model.get_output_embeddings()
    length = batch.shape[1]
    total, count = 0.0, 0
    for start in range(0, length - 1, chunk):
        end = min(start + chunk, length - 1)
        logits = head(hidden[:, start:end]).float()
        targets = batch[:, start + 1:end + 1]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 targets.reshape(-1), reduction="sum").item()
        count += targets.numel()
        del logits
    return total, count


@torch.no_grad()
def perplexity(model, input_ids, seqlen, device, limit=None, chunk=512):
    """Standard fixed-window wikitext perplexity, as used by the GPTQ/AWQ papers."""
    n = input_ids.numel() // seqlen
    if limit:
        n = min(n, limit)
    total_nll, total_tokens = 0.0, 0
    for i in range(n):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        nll, count = _window_nll(model, batch, chunk)
        total_nll += nll
        total_tokens += count
    return float(torch.exp(torch.tensor(total_nll / total_tokens))), n


SKIP: tuple[str, ...] = ()


def main() -> int:
    global SKIP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate at most this many windows (for a quick smoke run)")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                    help=f"any of: {' '.join(CONFIGS)}")
    ap.add_argument("--skip-modules", nargs="*", default=["lm_head"],
                    help="substrings of layer names to leave in fp16")
    ap.add_argument("--granule-n", type=int, default=1)
    ap.add_argument("--granule-k", type=int, default=1)
    ap.add_argument("--hqq-iters", type=int, default=20)
    ap.add_argument("--select-p", type=float, default=2.0,
                    help="exponent of the candidate-selection error norm; 2=MSE, 1=MAE, inf=worst")
    ap.add_argument("--activations", default="none",
                    help="activation format policy: 'none' for W4A16; 'match' to use each row's "
                         "own weight policy; or a fixed policy (nvfp4 / nvfp4-46 / mixed / "
                         "mixed-46) held constant so the weight format is isolated")
    ap.add_argument("--act-method", default="rtn",
                    help="fitting method for activations; anything beyond rtn is a research "
                         "probe, since activations are quantised per forward pass")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--logit-chunk", type=int, default=512,
                    help="positions per logit slice; lower it if the vocabulary is large")
    ap.add_argument("--json", type=Path, default=None, help="also write results here")
    args = ap.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        ap.error(f"unknown config(s) {unknown}; choose from {sorted(CONFIGS)}")
    SKIP = tuple(args.skip_modules)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"model    : {args.model}")
    print(f"device   : {device}  dtype: {args.dtype}  seqlen: {args.seqlen}")
    print(f"methods  : {available_methods()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    input_ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    print(f"tokens   : {input_ids.numel()}  windows: {input_ids.numel() // args.seqlen}")

    layers = list(quantizable_layers(model, SKIP))
    print(f"layers   : {len(layers)} quantisable (skipping {list(SKIP)})\n")
    # Keep pristine weights on the host; the GPU may not have room for a second copy.
    originals = {name: m.weight.data.detach().to("cpu", torch.float32).clone()
                 for name, m, _ in layers}

    cfg_kwargs = dict(granule_n=args.granule_n, granule_k=args.granule_k,
                      iters=args.hqq_iters, p=args.select_p)

    if args.activations == "none":
        print("activations: none -> W4A16 (weight-only)\n")
    elif args.activations == "match":
        print(f"activations: same policy as the weights, method {args.act_method} -> W4A4\n")
    else:
        print(f"activations: {args.activations} / {args.act_method} -> W4A4\n")

    results = {}
    modules = [m for _, m, _ in layers]
    for name in args.configs:
        spec = CONFIGS[name]
        start = time.time()
        if spec is None:
            restore(model, originals)
            share = None
        else:
            policy, method = spec
            share = apply_quantization(model, originals, policy, method, cfg_kwargs, device)
        quant_s = time.time() - start

        # "match" ties the activation policy to the weight policy, which is how Four Over Six is
        # actually meant to be used -- its Figure 3 applies Q(4/6) to activations as well.  A fixed
        # policy instead holds activations constant, isolating the weight format's contribution.
        act_policy = args.activations
        if act_policy == "match":
            act_policy = spec[0] if spec is not None else "nvfp4"
        act_handles = []
        if args.activations != "none":
            act_handles = install_activation_quantizers(modules, activation_config(
                QuantConfig(granule_n=1, granule_k=1, iters=args.hqq_iters, p=args.select_p),
                format_policy=act_policy, method=args.act_method))

        start = time.time()
        ppl, windows = perplexity(model, input_ids, args.seqlen, device, args.limit,
                                  args.logit_chunk)
        remove_activation_quantizers(act_handles)
        results[name] = {"ppl": ppl, "e0m3_share": share,
                         "quantise_s": quant_s, "eval_s": time.time() - start}
        extra = f"  e0m3 {share:.1%}" if share is not None else ""
        if spec is None and args.activations != "none":
            extra += "  (activations still quantised: this row is W16A4)"
        print(f"{name:14s} ppl {ppl:8.4f}   ({quant_s:5.1f}s quant, "
              f"{results[name]['eval_s']:5.1f}s eval, {windows} windows){extra}")

    restore(model, originals)

    base = results.get("fp16", {}).get("ppl")
    ref = results.get("nvfp4-rtn", {}).get("ppl")
    print("\n" + "=" * 76)
    print(f"{'config':14s} {'ppl':>9s} {'vs fp16':>10s} {'vs nvfp4-rtn':>14s}  e0m3")
    print("-" * 76)
    for name, r in results.items():
        d_base = f"{r['ppl'] - base:+.4f}" if base else "-"
        d_ref = f"{r['ppl'] - ref:+.4f}" if ref and name != "fp16" else "-"
        share = f"{r['e0m3_share']:.1%}" if r["e0m3_share"] is not None else "-"
        print(f"{name:14s} {r['ppl']:9.4f} {d_base:>10s} {d_ref:>14s}  {share}")
    print("=" * 76)
    print("lower is better; 'vs nvfp4-rtn' isolates the effect of the mixed format")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"model": args.model, "seqlen": args.seqlen, "dtype": args.dtype,
             "granule": [args.granule_n, args.granule_k], "select_p": args.select_p,
             "activations": args.activations, "act_method": args.act_method,
             "results": results}, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
