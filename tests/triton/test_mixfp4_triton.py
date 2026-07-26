#!/usr/bin/env python3
"""Correctness tests for the mixfp4 Triton backend.

Runs standalone (``python tests/triton/test_mixfp4_triton.py``) or under pytest.

The load-bearing tests are the ones that cross-check against the *CUDA* path's conventions:
``test_codebooks_match_cuda_reference`` and ``test_ue4m3_matches_cuda_reference`` import
``tests/mma_intrinsics/expected_value.py``, which is itself a deliberately independent
reimplementation of the codebooks used to validate the patched hardware instruction.  If the Triton
decode and that reference ever disagree, the two backends have silently forked and a checkpoint
quantised for one will be misread by the other.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from mixfp4.codebook import (  # noqa: E402
    E0M3_CODEBOOK,
    E2M1_CODEBOOK,
    FLAG_MASK,
    decode_ue4m3,
    encode_ue4m3,
    pack_nibbles_over_k,
    unpack_nibbles_over_k,
)
from mixfp4.linear import GEMM, GEMM_SPLITK, GEMV, MixFP4Linear, get_matmul_type, patch_model  # noqa: E402
from mixfp4.quant import dequantize_mixfp4, quantization_error, quantize_mixfp4  # noqa: E402
from mixfp4.triton_kernels.gemm import gemm_forward  # noqa: E402
from mixfp4.triton_kernels.gemm_splitK import gemm_splitK_forward  # noqa: E402
from mixfp4.triton_kernels.gemv import gemv_forward  # noqa: E402

DEVICE = "cuda"
# The kernels decode weights into the activation dtype before the dot, so the floor on relative
# error is that dtype's epsilon, not anything about the quantisation.
TOL = {torch.float16: 5e-4, torch.bfloat16: 3e-3}


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / a.float().norm().clamp(min=1e-30)).item()


def _load_cuda_reference():
    """Import ``tests/mma_intrinsics/expected_value.py`` by path -- it is not a package."""
    path = REPO / "tests" / "mma_intrinsics" / "expected_value.py"
    spec = importlib.util.spec_from_file_location("mma_expected_value", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- format-level agreement with the CUDA path ---------------------------------------------------


def test_codebooks_match_cuda_reference():
    ref = _load_cuda_reference()
    assert E2M1_CODEBOOK == ref.E2M1_CODEBOOK, "E2M1 codebook diverged from the CUDA reference"
    assert E0M3_CODEBOOK == [float(v) for v in ref.E0M3_CODEBOOK], \
        "E0M3 codebook diverged from the CUDA reference"


def test_ue4m3_matches_cuda_reference():
    """Every finite UE4M3 code must decode identically to the CUDA path's reference.

    Code 0x7f is deliberately excluded.  CUTLASS's ``float_ue4m3_t`` documents the range as
    [0:448] with ``has_NaN: true`` and defines ``isnan(x) { return x.storage == 0x7f; }``, and
    torch's ``float8_e4m3fn`` agrees -- 0x7f is NaN, not a finite 480.  The repo's own
    ``expected_value.py`` has no NaN case and computes (1 + 7/8) * 2^8 = 480.0 for it, so it
    disagrees with both the library and the hardware there.  Unreachable in practice (the
    quantiser clamps scales to 448, so 0x7f is never emitted), but the reference is wrong.
    """
    ref = _load_cuda_reference()
    codes = torch.arange(0x7f, dtype=torch.uint8)
    got = decode_ue4m3(codes)
    for code in range(0x7f):
        expected = ref.decode_ue4m3(code)
        assert abs(got[code].item() - expected) <= 1e-12 * max(expected, 1.0), \
            f"UE4M3 code {code:#04x}: got {got[code].item()}, reference {expected}"
    assert decode_ue4m3(torch.tensor([0x7f], dtype=torch.uint8)).isnan().all(), \
        "0x7f should decode to NaN, matching CUTLASS float_ue4m3_t"


def test_bit7_is_free():
    """Setting the format tag must not perturb the decoded magnitude.

    This is the invariant the whole scheme rests on, and the CUDA path verifies the hardware side
    of it in ``tests/mma_intrinsics/run_tests.sh``.  Here we check the software side.
    """
    codes = torch.arange(0x7f, dtype=torch.uint8)   # 0x7f is the NaN encoding; NaN != NaN
    tagged = codes | FLAG_MASK
    assert torch.equal(decode_ue4m3(codes), decode_ue4m3(tagged))


def test_ue4m3_roundtrip():
    codes = torch.arange(0x7f, dtype=torch.uint8)
    assert torch.equal(encode_ue4m3(decode_ue4m3(codes)), codes)


# --- packing -------------------------------------------------------------------------------------


def test_pack_roundtrip():
    nibbles = torch.randint(0, 16, (64, 256), dtype=torch.uint8, device=DEVICE)
    assert torch.equal(unpack_nibbles_over_k(pack_nibbles_over_k(nibbles)), nibbles)


# --- quantiser -----------------------------------------------------------------------------------


def test_mixed_beats_both_pure_formats():
    """Per-group selection minimises squared error per group, so it cannot lose globally."""
    torch.manual_seed(0)
    for name, w in (
        ("gaussian", torch.randn(256, 512, device=DEVICE)),
        ("uniform", torch.rand(256, 512, device=DEVICE) * 2 - 1),
        ("heavy_tail", torch.randn(256, 512, device=DEVICE)
         * torch.exp(torch.randn(256, 512, device=DEVICE) * 2)),
    ):
        err = quantization_error(w)
        assert err["mixed"] <= min(err["e2m1"], err["e0m3"]) * 1.001, \
            f"{name}: mixed {err['mixed']:.4g} worse than pure {err['e2m1']:.4g}/{err['e0m3']:.4g}"


def test_forced_formats_set_the_tag():
    w = torch.randn(64, 256, device=DEVICE)
    assert quantize_mixfp4(w, force_format="e2m1").e0m3_fraction == 0.0
    assert quantize_mixfp4(w, force_format="e0m3").e0m3_fraction == 1.0
    assert (quantize_mixfp4(w, force_format="e2m1").scales & FLAG_MASK).any() == False  # noqa: E712
    assert (quantize_mixfp4(w, force_format="e0m3").scales & FLAG_MASK).all()


def test_all_zero_weight():
    q = quantize_mixfp4(torch.zeros(32, 64, device=DEVICE))
    assert dequantize_mixfp4(q).abs().max().item() == 0.0


def test_coarse_granule_is_uniform_within_a_block():
    """A coarsened granule must tag every scale byte in the block identically.

    This is what makes a checkpoint readable by the CUDA path, whose granule cannot go below one
    MMA atom's footprint.
    """
    w = torch.randn(128, 512, device=DEVICE)
    q = quantize_mixfp4(w, granule_n=64, granule_k=2)
    flags = (q.scales >> 7).t().reshape(128 // 64, 64, (512 // 16) // 2, 2)
    assert (flags == flags[:, :1, :, :1]).all(), "granule is not uniform within its block"


def test_four_over_six_reproduces_paper_table1():
    """arXiv:2512.02010 Table 1: [10, 20, 30, 40] is lossless capped at 4, lossy capped at 6.

    At M=6 the block scale is 40/6 = 6.67 and 30 normalises to 4.5, which E2M1 cannot represent
    (its entries jump 4 -> 6), so it rounds to 4 and dequantises to 26.7.  At M=4 the scale is 10
    and the block becomes exactly [1, 2, 3, 4].
    """
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0] * 4], device=DEVICE)
    capped = dequantize_mixfp4(quantize_mixfp4(x, format_policy="e2m1-4", method="rtn")).to(DEVICE)
    assert torch.allclose(capped, x, rtol=1e-5), f"M=4 should be lossless here, got {capped[0, :4]}"

    standard = dequantize_mixfp4(quantize_mixfp4(x, format_policy="nvfp4", method="rtn")).to(DEVICE)
    assert not torch.allclose(standard, x, rtol=1e-3), "M=6 should lose the third value"
    assert abs(standard[0, 2].item() - 80.0 / 3.0) < 0.1, \
        f"30 should dequantise to ~26.7 at M=6, got {standard[0, 2].item()}"

    # The adaptive policy must pick the lossless option.
    adaptive = dequantize_mixfp4(quantize_mixfp4(x, format_policy="nvfp4-46",
                                                 method="rtn")).to(DEVICE)
    assert torch.allclose(adaptive, x, rtol=1e-5), "4/6 should select the M=4 fit here"


def test_capping_every_block_is_worse_than_adaptive():
    """The paper's Table 3: capping *all* blocks at 4 is worse than standard NVFP4.

    Giving up +-5 and +-6 costs a representable level; it only pays off on blocks that actually
    have values near 5/6 of their peak.  Selection has to be per block, which is the whole point.
    """
    torch.manual_seed(0)
    w = torch.randn(256, 512, device=DEVICE)

    def rel(policy):
        recon = dequantize_mixfp4(quantize_mixfp4(w, format_policy=policy, method="rtn"))
        return ((w - recon.to(DEVICE)).norm() / w.norm()).item()

    standard, all_four, adaptive = rel("nvfp4"), rel("e2m1-4"), rel("nvfp4-46")
    assert all_four > standard, \
        f"capping every block should be worse: {all_four:.5f} vs {standard:.5f}"
    assert adaptive < min(standard, all_four) * 1.001, \
        f"adaptive should beat both: {adaptive:.5f} vs {standard:.5f}/{all_four:.5f}"


def test_all_policies_round_trip():
    """Every registered policy must produce a decodable tensor under every method."""
    from mixfp4.quantizers import FORMAT_POLICIES, available_methods
    torch.manual_seed(0)
    w = torch.randn(64, 256, device=DEVICE)
    for policy in FORMAT_POLICIES:
        for method in available_methods():
            q = quantize_mixfp4(w, format_policy=policy, method=method)
            recon = dequantize_mixfp4(q).to(DEVICE)
            rel = ((w - recon).norm() / w.norm()).item()
            assert recon.isfinite().all(), f"{policy}/{method} produced non-finite values"
            assert rel < 0.5, f"{policy}/{method}: implausible relative error {rel:.3f}"


def test_activation_path_matches_weight_path():
    """Quantising without packing must be bit-identical to quantising, packing and unpacking.

    The activation path skips the nibble packing because activations are discarded within one
    matmul, but it must not skip anything else -- notably the UE4M3 rounding of the per-group
    scale.  Simulating NVFP4 without that rounding would flatter it.
    """
    from mixfp4.quant import simulate_quantization
    from mixfp4.quantizers import QuantConfig

    torch.manual_seed(0)
    x = torch.randn(128, 512, device=DEVICE)
    for policy in ("nvfp4", "mixed", "nvfp4-46", "mixed-46"):
        cfg = QuantConfig(method="rtn", format_policy=policy)
        direct = simulate_quantization(x, cfg)
        packed = dequantize_mixfp4(quantize_mixfp4(x, format_policy=policy, method="rtn"))
        assert torch.equal(direct, packed.to(DEVICE)), \
            f"{policy}: activation path diverged from the weight path"


def test_activation_quantizer_preserves_shape_and_dtype():
    from mixfp4.activation import ActivationQuantizer
    from mixfp4.quantizers import QuantConfig

    quantizer = ActivationQuantizer(QuantConfig(method="rtn", format_policy="nvfp4"))
    for shape in [(8, 512), (2, 7, 512), (1, 1, 256)]:
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(*shape, device=DEVICE, dtype=dtype)
            out = quantizer(x)
            assert out.shape == x.shape and out.dtype == x.dtype
            assert out.isfinite().all()
            # Quantisation must actually do something, but stay in the right neighbourhood.
            assert not torch.equal(out, x), "activation quantiser was a no-op"
            assert _relative_error(x, out) < 0.25, "activation quantisation destroyed the tensor"

    # A K that is not a multiple of the group size is passed through untouched, not corrupted.
    odd = torch.randn(4, 100, device=DEVICE, dtype=torch.float16)
    assert torch.equal(quantizer(odd), odd)


def test_activation_hooks_install_and_remove():
    from mixfp4.activation import install_activation_quantizers, remove_activation_quantizers
    from mixfp4.quantizers import QuantConfig

    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 256, bias=False).to(DEVICE).half()
    x = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
    clean = layer(x).clone()

    handles = install_activation_quantizers([layer], QuantConfig(method="rtn",
                                                                format_policy="nvfp4"))
    quantized = layer(x).clone()
    assert not torch.equal(clean, quantized), "hook did not quantise the input"

    remove_activation_quantizers(handles)
    assert torch.equal(layer(x), clean), "removing the hook did not restore the layer"


# --- kernels -------------------------------------------------------------------------------------


def _check_kernel(fn, name, M, N, K, dtype, force_format):
    torch.manual_seed(0)
    w = torch.randn(N, K, device=DEVICE)
    q = quantize_mixfp4(w, force_format=force_format)
    x = torch.randn(M, K, device=DEVICE, dtype=dtype)
    ref = x.float() @ dequantize_mixfp4(q).float().t()
    got = fn(x, q.W_q, q.scales, q.meta_scale, N, K).float()
    rel = (got - ref).norm().item() / ref.norm().item()
    assert rel < TOL[dtype], (
        f"{name} M={M} N={N} K={K} {dtype} fmt={force_format}: rel error {rel:.3e}")
    return rel


def test_kernels_match_reference():
    shapes = [(1, 256, 512), (2, 512, 1024), (4, 1024, 2048),
              (8, 256, 512), (32, 512, 512), (96, 384, 768), (128, 256, 1024)]
    for M, N, K in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            # Uniform tagging is correct by accident under a permuted layout; only genuinely mixed
            # tagging exercises the per-group path.  Test both, plus the error-driven choice.
            for fmt in ("e2m1", "e0m3", "random", None):
                _check_kernel(gemm_forward, "gemm", M, N, K, dtype, fmt)
                _check_kernel(gemm_splitK_forward, "gemm_splitK", M, N, K, dtype, fmt)
                if M <= 8:
                    _check_kernel(gemv_forward, "gemv", M, N, K, dtype, fmt)


def test_kernels_agree_with_each_other():
    """All three reduction strategies must be numerically equivalent, not merely each acceptable."""
    torch.manual_seed(0)
    N, K = 512, 1024
    q = quantize_mixfp4(torch.randn(N, K, device=DEVICE), force_format="random")
    x = torch.randn(4, K, device=DEVICE, dtype=torch.float16)
    a = gemm_forward(x, q.W_q, q.scales, q.meta_scale, N, K).float()
    b = gemm_splitK_forward(x, q.W_q, q.scales, q.meta_scale, N, K).float()
    c = gemv_forward(x, q.W_q, q.scales, q.meta_scale, N, K).float()
    for name, other in (("splitK", b), ("gemv", c)):
        rel = (a - other).norm().item() / a.norm().item()
        assert rel < 1e-5, f"gemm and {name} disagree by {rel:.3e}"


def test_non_multiple_shapes():
    """Masking paths: M, N and K that are not multiples of any block size."""
    for M, N, K in [(3, 129, 512), (17, 200, 1024), (65, 33, 528)]:
        _check_kernel(gemm_forward, "gemm", M, N, K, torch.float16, None)
        _check_kernel(gemm_splitK_forward, "gemm_splitK", M, N, K, torch.float16, None)


# --- module --------------------------------------------------------------------------------------


def test_dispatch_thresholds():
    assert get_matmul_type(1) == GEMV
    assert get_matmul_type(4) == GEMV
    assert get_matmul_type(5) == GEMM_SPLITK
    assert get_matmul_type(63) == GEMM_SPLITK
    assert get_matmul_type(64) == GEMM


def test_linear_module():
    torch.manual_seed(0)
    ref = torch.nn.Linear(512, 256, bias=True).to(DEVICE).half()
    layer = MixFP4Linear.from_linear(ref)
    for shape in [(1, 512), (8, 512), (128, 512), (2, 7, 512)]:
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float16)
        out = layer(x)
        assert out.shape == (*shape[:-1], 256)
        # Against the dequantised weight, not the original: the kernel's job is to match its own
        # weight representation.  Quantisation error itself is measured in the quantiser tests.
        w = dequantize_mixfp4(quantize_mixfp4(ref.weight.data)).float()
        expect = x.reshape(-1, 512).float() @ w.t() + ref.bias.float()
        rel = ((out.reshape(-1, 256).float() - expect).norm() / expect.norm()).item()
        assert rel < 5e-3, f"shape {shape}: rel error {rel:.3e}"


def test_torch_compile():
    torch.manual_seed(0)
    layer = MixFP4Linear.from_linear(torch.nn.Linear(512, 256, bias=False).to(DEVICE).half())
    x = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
    eager = layer(x)
    compiled = torch.compile(layer, fullgraph=True)(x)
    # Not bitwise: these kernels accumulate atomically, so summation order varies between
    # launches.  Compared relatively -- an absolute tolerance tight enough to be meaningful here
    # would be below one fp16 ulp at these magnitudes.
    assert _relative_error(eager, compiled) < 1e-3, "torch.compile changed the result"


def test_patch_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512),
    ).to(DEVICE).half()
    patch_model(model, min_features=128)
    assert len(model._mixfp4_replaced) == 2
    assert all(isinstance(m, MixFP4Linear) for m in model if isinstance(m, MixFP4Linear))
    out = model(torch.randn(4, 512, device=DEVICE, dtype=torch.float16))
    assert out.shape == (4, 512) and out.isfinite().all()


def test_state_dict_roundtrip():
    layer = MixFP4Linear.from_linear(torch.nn.Linear(512, 256, bias=True).to(DEVICE).half())
    x = torch.randn(4, 512, device=DEVICE, dtype=torch.float16)
    before = layer(x)
    clone = MixFP4Linear(512, 256, bias=True, device=DEVICE)
    clone.load_state_dict(layer.state_dict())
    # Relative, not bitwise: atomic accumulation is not reproducible across launches.
    assert _relative_error(before, clone(x)) < 1e-3


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
