"""Dynamic activation quantisation, for W4A4 evaluation.

Weights are quantised once, offline, per output channel along K.  Activations are quantised on
every forward pass, per *token* along K -- but with the same group size, the same UE4M3 per-group
scale and the same global normaliser, so the grouping is structurally identical and the whole
weight path is reused via :func:`mixfp4.quant.simulate_quantization`.

Two things make activations harder than weights, and both matter for reading W4A4 results:

* They contain far larger outliers, concentrated in a few channels. Methods that exist mainly to
  handle this -- SmoothQuant's per-channel rebalancing, the random Hadamard transform, AWQ's
  activation-aware scaling -- are *not* implemented here. These numbers are plain RTN-style
  dynamic quantisation, which is the honest floor, not the state of the art.
* There is no opportunity to spend compute. A weight can be fitted with 20 HQQ iterations once;
  an activation tensor is quantised and discarded within a single matmul, so anything beyond a
  couple of passes over the data is not viable in a real kernel. Expensive methods are permitted
  here for measurement, but ``--act-method hqq`` is a research probe, not a deployable recipe.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from .quantizers import QuantConfig
from .quant import simulate_quantization


class ActivationQuantizer:
    """Quantise/dequantise the input of a linear layer, per token, in groups along K."""

    def __init__(self, config: QuantConfig):
        self.config = config

    def __call__(self, x: Tensor) -> Tensor:
        k = x.shape[-1]
        if k % self.config.group_size != 0:
            return x
        flat = x.reshape(-1, k)
        # Quantise in the input dtype's compute precision but return the original dtype, so the
        # layer downstream sees exactly what a real A4 kernel would hand it.
        return simulate_quantization(flat, self.config).reshape(x.shape).to(x.dtype)


def _pre_hook(quantizer: ActivationQuantizer):
    def hook(module, args, kwargs=None):
        if not args:
            return None
        return (quantizer(args[0]),) + tuple(args[1:])
    return hook


def install_activation_quantizers(modules, config: QuantConfig) -> list:
    """Attach a forward pre-hook to each module so its input is quantised before the matmul.

    Returns the handles, so the caller can restore the model exactly.  Hooking rather than
    wrapping keeps the module tree untouched, which matters because the perplexity harness
    swaps weights in and out of the *same* model across configurations.
    """
    quantizer = ActivationQuantizer(config)
    return [m.register_forward_pre_hook(_pre_hook(quantizer)) for m in modules]


def remove_activation_quantizers(handles) -> None:
    for handle in handles:
        handle.remove()


def activation_config(base: QuantConfig, *, format_policy: str, method: str) -> QuantConfig:
    """A config for activations, derived from the weight config.

    Granule coarsening is dropped: it exists so a checkpoint stays legal for the CUDA kernel's
    instruction-level format granule, and activations are never stored in a checkpoint.
    """
    return replace(base, format_policy=format_policy, method=method, granule_n=1, granule_k=1)


__all__ = [
    "ActivationQuantizer",
    "activation_config",
    "install_activation_quantizers",
    "remove_activation_quantizers",
]
