"""Mixed E2M1/E0M3 4-bit quantisation and Triton inference kernels.

Two 4-bit codebooks index the same nibble -- E2M1 (standard NVFP4, log-spaced) and E0M3
(sign-magnitude INT4, uniformly spaced) -- and each group of 16 weights picks whichever fits its
distribution better.  The choice rides in bit 7 of the group's UE4M3 scale byte, which is
architecturally unused, so it costs no extra storage or bandwidth.

    from mixfp4 import MixFP4Linear, patch_model, quantize_mixfp4

    layer = MixFP4Linear.from_linear(linear, method="hqq", format_policy="mixed")
    patch_model(model, method="hqq")           # swap every eligible nn.Linear in place

See ``docs/quantization_quality.md`` for whether mixing is worth it (yes, modestly) and
``docs/triton_backend.md`` for what these kernels can and cannot do -- in particular, that Triton
cannot reach the E0M3 tensor core and decodes in registers instead.
"""

from .activation import (
    ActivationQuantizer,
    activation_config,
    install_activation_quantizers,
    remove_activation_quantizers,
)
from .codebook import (
    E0M3_CODEBOOK,
    E2M1_CODEBOOK,
    FLAG_MASK,
    GROUP_SIZE,
    combined_lut,
    decode_ue4m3,
    encode_ue4m3,
)
from .linear import MixFP4Linear, get_matmul_type, mixfp4_matmul, patch_model
from .quant import (
    MixFP4Tensor,
    dequantize_mixfp4,
    quantization_error,
    quantize_dequantize,
    quantize_mixfp4,
    simulate_quantization,
)
from .quantizers import QuantConfig, available_methods, register

__all__ = [
    "ActivationQuantizer",
    "E0M3_CODEBOOK",
    "E2M1_CODEBOOK",
    "FLAG_MASK",
    "GROUP_SIZE",
    "MixFP4Linear",
    "MixFP4Tensor",
    "QuantConfig",
    "available_methods",
    "combined_lut",
    "decode_ue4m3",
    "dequantize_mixfp4",
    "encode_ue4m3",
    "get_matmul_type",
    "mixfp4_matmul",
    "patch_model",
    "quantization_error",
    "quantize_dequantize",
    "quantize_mixfp4",
    "register",
    "simulate_quantization",
    "activation_config",
    "install_activation_quantizers",
    "remove_activation_quantizers",
]
