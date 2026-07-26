"""``MixFP4Linear`` -- a drop-in replacement for ``nn.Linear`` backed by the mixfp4 Triton kernels.

Dispatch by batch size follows GemLite's thresholds, for the same reasons:

    M <= 4        GEMV          no ``tl.dot``; one row, one N slice, one K slab per program
    4 < M < 64    GEMM_SPLITK   too few output tiles to fill the machine, so split the reduction
    M >= 64       GEMM          enough tiles; a plain tiled matmul wins

The forward is a ``torch.library.custom_op`` so it survives ``torch.compile`` as an opaque call
rather than being traced into (and defeated by) inductor.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn

from .codebook import GROUP_SIZE
from .quant import MixFP4Tensor, quantize_mixfp4
from .triton_kernels.gemm import gemm_forward
from .triton_kernels.gemm_splitK import gemm_splitK_forward
from .triton_kernels.gemv import gemv_forward

GEMV, GEMM_SPLITK, GEMM = 0, 1, 2
_KERNELS = {GEMV: gemv_forward, GEMM_SPLITK: gemm_splitK_forward, GEMM: gemm_forward}
_NAMES = {GEMV: "GEMV", GEMM_SPLITK: "GEMM_SPLITK", GEMM: "GEMM"}

#: Set to ``gemv`` / ``gemm_splitK`` / ``gemm`` to pin one kernel, for benchmarking.
_FORCE = os.environ.get("MIXFP4_FORCE_MATMUL", "").lower()


def get_matmul_type(batch_size: int) -> int:
    """Pick a kernel for this M."""
    if _FORCE:
        return {"gemv": GEMV, "gemm_splitk": GEMM_SPLITK, "gemm": GEMM}[_FORCE]
    if batch_size >= 64:
        return GEMM
    if batch_size > 4:
        return GEMM_SPLITK
    return GEMV


@torch.library.custom_op("mixfp4::matmul", mutates_args=())
def mixfp4_matmul(x: Tensor, W_q: Tensor, scales: Tensor, meta_scale: float,
                  N: int, K: int, matmul_type: int) -> Tensor:
    """``x @ dequantize(W_q, scales).T`` for 2-D ``x``.  Opaque to torch.compile."""
    if matmul_type < 0:
        matmul_type = get_matmul_type(x.shape[0])
    return _KERNELS[matmul_type](x.contiguous(), W_q, scales, meta_scale, N, K)


@mixfp4_matmul.register_fake
def _(x: Tensor, W_q: Tensor, scales: Tensor, meta_scale: float,
      N: int, K: int, matmul_type: int) -> Tensor:
    return x.new_empty((x.shape[0], N))


class MixFP4Linear(nn.Module):
    """``nn.Linear`` with the weight held in mixed E2M1/E0M3 FP4.

    The weight is stored exactly as the kernels read it -- ``[K // 2, N]`` packed nibbles and
    ``[K // 16, N]`` tagged scale bytes -- so nothing is transposed or repacked at call time.
    Bias is applied outside the kernel, as in GemLite: it costs one fused add on a tensor that is
    already in registers, and keeping it out of the kernel avoids an extra pointer and mask in
    every config.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        assert in_features % GROUP_SIZE == 0, (
            f"in_features={in_features} must be a multiple of the scale group size {GROUP_SIZE}")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("W_q", torch.empty((in_features // 2, out_features),
                                                dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.empty((in_features // GROUP_SIZE, out_features),
                                                   dtype=torch.uint8, device=device))
        self.register_buffer("meta_scale", torch.ones((), dtype=torch.float32, device=device))
        # Mirrored as a plain float.  Reading the buffer in forward would mean a ``.item()`` call,
        # which dynamo cannot trace under ``fullgraph=True``; the buffer stays for serialisation.
        self._meta_scale = 1.0
        if bias:
            self.register_parameter("bias", nn.Parameter(
                torch.zeros(out_features, dtype=dtype or torch.float16, device=device)))
        else:
            self.bias = None
        self.e0m3_fraction = 0.0

    # --- construction ----------------------------------------------------------------------

    @classmethod
    def from_tensor(cls, q: MixFP4Tensor, bias: Tensor | None = None) -> "MixFP4Linear":
        layer = cls(q.K, q.N, bias=bias is not None, device=q.W_q.device)
        layer.W_q = q.W_q.contiguous()
        layer.scales = q.scales.contiguous()
        layer.meta_scale = torch.tensor(q.meta_scale, dtype=torch.float32, device=q.W_q.device)
        layer._meta_scale = float(q.meta_scale)
        layer.e0m3_fraction = q.e0m3_fraction
        if bias is not None:
            layer.bias = nn.Parameter(bias.detach().clone())
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear, **quant_kwargs) -> "MixFP4Linear":
        """Quantise an existing ``nn.Linear``.  ``quant_kwargs`` go to :func:`quantize_mixfp4`."""
        q = quantize_mixfp4(linear.weight.data, **quant_kwargs)
        return cls.from_tensor(q, linear.bias.data if linear.bias is not None else None)

    # --- forward ---------------------------------------------------------------------------

    def forward(self, x: Tensor, matmul_type: int = -1) -> Tensor:
        shape = (*x.shape[:-1], self.out_features)
        out = mixfp4_matmul(x.reshape(-1, self.in_features), self.W_q, self.scales,
                            self._meta_scale, self.out_features, self.in_features,
                            matmul_type).reshape(shape)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._meta_scale = float(self.meta_scale)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, e0m3_fraction={self.e0m3_fraction:.3f}")


def patch_model(model: nn.Module, *, skip_modules=("lm_head",), min_features: int = 256,
                **quant_kwargs) -> nn.Module:
    """Replace every eligible ``nn.Linear`` in ``model`` with a :class:`MixFP4Linear`, in place.

    Args:
        skip_modules: substrings of qualified names to leave alone.  ``lm_head`` is skipped by
            default -- output projections are disproportionately sensitive to weight quantisation.
        min_features: leave small projections in 16-bit; they are latency-bound on launch overhead
            rather than on weight bandwidth, so quantising them costs more than it saves.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qualified = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if any(s in qualified for s in skip_modules):
                continue
            if min(child.in_features, child.out_features) < min_features:
                continue
            if child.in_features % GROUP_SIZE != 0:
                continue
            setattr(module, child_name, MixFP4Linear.from_linear(child, **quant_kwargs))
            replaced.append(qualified)
    model._mixfp4_replaced = replaced
    return model
