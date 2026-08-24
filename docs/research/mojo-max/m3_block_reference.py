"""M3-4 for issue #57: the FP64 NumPy reference implementation of the WHOLE
`_BosonDecoderBlock` (Snake1d -> wn_conv_transpose1d -> 3x _BosonResidualUnit), cross-checked
against a real PyTorch FP32 forward of the actual vendored `_BosonDecoderBlock` class
(`.research-scratch/vllm-omni/vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_decoder.py`,
HF `DacModel` decoder layout per `m1-responsibility-map.md` S8).

Extends `numpy_residual_unit` (already validated in `m2_residual_unit_prototype.py`, reused
here verbatim via import, NOT reimplemented) with a new FP64 `numpy_conv_transpose1d` and the
full block wiring.

Weight source for this run: SYNTHETIC. At the time this script was written,
`m3_block_weights.py` (M3-3) existed on disk but `m3-block-results.md` (its results doc) did
not, i.e. M3-3 had not yet been confirmed to have actually run/passed against the real
checkpoint. Per the M3-4 task spec this is an explicitly acceptable interim state: this script
cross-checks the FP64 NumPy path against a PyTorch FP32 forward using the real `_BosonDecoderBlock`
*class* (not a hand-rolled model) with PyTorch's own default-initialized weights and a random
input, both extracted from the SAME `torch.nn.Module` instance. This validates the arithmetic
(Snake1d, conv_transpose1d, residual units, weight_norm fold, cropping) end-to-end. The
real-checkpoint-weight cross-check is a follow-up once M3-3's results land (do not fabricate
that number here).

Usage: /Users/vedmalex/work/higgs-local-test/.venv-tts/bin/python m3_block_reference.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _stub_max_module_for_import() -> None:
    """`m2_residual_unit_prototype.py` imports the `max` (Modular MAX) package at module
    scope purely to define its MAX-graph-building functions, which this script does not use
    (only its FP64 NumPy reference functions -- `numpy_snake`, `numpy_conv1d`,
    `numpy_residual_unit`, `fold_weight_norm` -- are reused here). This venv (`.venv-tts`,
    which has torch+numpy for the PyTorch cross-check) intentionally does not have the `max`
    package installed (that lives in the separate pixi/MAX toolchain env used by the M2/M3-1/
    M3-5+ MAX-graph scripts). Stub out just enough of `max.*` so the import succeeds without
    ever executing any MAX-graph code path."""
    import types

    if "max" in sys.modules:
        return
    max_mod = types.ModuleType("max")
    driver_mod = types.ModuleType("max.driver")
    dtype_mod = types.ModuleType("max.dtype")
    engine_mod = types.ModuleType("max.engine")
    graph_mod = types.ModuleType("max.graph")
    for name in ("CPU", "Accelerator", "Buffer", "accelerator_count"):
        setattr(driver_mod, name, None)
    dtype_mod.DType = None
    engine_mod.InferenceSession = None
    for name in ("DeviceRef", "Graph", "TensorType", "ops"):
        setattr(graph_mod, name, None)
    max_mod.driver = driver_mod
    max_mod.dtype = dtype_mod
    max_mod.engine = engine_mod
    max_mod.graph = graph_mod
    sys.modules["max"] = max_mod
    sys.modules["max.driver"] = driver_mod
    sys.modules["max.dtype"] = dtype_mod
    sys.modules["max.engine"] = engine_mod
    sys.modules["max.graph"] = graph_mod


_stub_max_module_for_import()

from m2_residual_unit_prototype import (  # noqa: E402  (import after sys.path mutation)
    EPS,
    fold_weight_norm,
    numpy_conv1d,
    numpy_residual_unit,
    numpy_snake,
)

DECODER_MODULE_PATH = (
    HERE
    / ".."
    / ".."
    / ".."
    / ".research-scratch"
    / "vllm-omni"
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "higgs_audio_v2"
    / "higgs_audio_decoder.py"
).resolve()


def _load_decoder_module():
    spec = importlib.util.spec_from_file_location("higgs_audio_decoder_m3_4", DECODER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# New: FP64 transposed-conv1d reference.
# --------------------------------------------------------------------------------------


def numpy_conv_transpose1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int = 1,
) -> np.ndarray:
    """Reference 1D transposed convolution, FP64, matching `torch.nn.ConvTranspose1d`
    semantics (groups=1). x: [B, C_in, L_in], weight: [C_in, C_out, K] (PyTorch
    ConvTranspose1d layout -- note C_in first, unlike plain Conv1d's [C_out, C_in, K]),
    bias: [C_out].

    Direct-definition implementation (not the zero-insertion-then-conv equivalence): for
    each kernel offset k, every input sample x[..., i] scatters into
    output position  t = i*stride + k*dilation - padding.  This is exactly what
    PyTorch's ConvTranspose1d computes; L_out follows PyTorch's documented formula:
      L_out = (L_in - 1)*stride - 2*padding + dilation*(K - 1) + output_padding + 1
    """
    b, c_in, l_in = x.shape
    c_in_w, c_out, k = weight.shape
    assert c_in_w == c_in, f"weight C_in {c_in_w} != input C_in {c_in}"
    l_out = (l_in - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1

    x64 = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    out = np.zeros((b, c_out, l_out), dtype=np.float64)

    i_idx = np.arange(l_in)
    for kk in range(k):
        offset = kk * dilation - padding
        t_idx = i_idx * stride + offset  # output position each input sample lands on, for this k
        valid = (t_idx >= 0) & (t_idx < l_out)
        if not np.any(valid):
            continue
        # contract over c_in: contrib[b, c_out, i] = sum_ci x[b, ci, i] * w[ci, c_out, kk]
        contrib = np.einsum("bci,cd->bdi", x64[:, :, valid], w64[:, :, kk])
        np.add.at(out, (slice(None), slice(None), t_idx[valid]), contrib)

    return out + bias.astype(np.float64)[None, :, None]


def _validate_conv_transpose_hand_case() -> None:
    """Small hand/PyTorch-checked case before trusting the implementation above on the
    real block. c_in=1, c_out=1, kernel=2, stride=2, padding=0, output_padding=0.
    x = [1, 2], weight[0,0,:] = [10, 20]  (PyTorch ConvTranspose1d layout [C_in,C_out,K]).

    By definition, output[t] = sum over (i,k) with i*stride + k == t of x[i] * w[k]:
      t=0: i=0,k=0 -> 1*10 = 10
      t=1: i=0,k=1 -> 1*20 = 20
      t=2: i=1,k=0 -> 2*10 = 20
      t=3: i=1,k=1 -> 2*20 = 40
    Expected output (no bias): [10, 20, 20, 40], length 4.
    """
    x = np.array([[[1.0, 2.0]]], dtype=np.float64)
    weight = np.array([[[10.0, 20.0]]], dtype=np.float64)  # [C_in=1, C_out=1, K=2]
    bias = np.array([0.0], dtype=np.float64)
    got = numpy_conv_transpose1d(x, weight, bias, stride=2, padding=0, output_padding=0)
    expected = np.array([[[10.0, 20.0, 20.0, 40.0]]], dtype=np.float64)
    assert got.shape == expected.shape, f"hand case shape mismatch: {got.shape} vs {expected.shape}"
    assert np.allclose(got, expected), f"hand case mismatch: got {got} expected {expected}"

    # Cross-check the same tiny case against torch.nn.functional.conv_transpose1d directly,
    # including a nonzero padding/output_padding combo (stride=3, padding=1, output_padding=1,
    # the real Higgs stride=3 block's shape) with random weights.
    rng = np.random.default_rng(42)
    c_in, c_out, k, stride, padding, output_padding = 3, 2, 6, 3, 2, 1
    l_in = 5
    x2 = rng.normal(size=(1, c_in, l_in)).astype(np.float64)
    w2 = rng.normal(size=(c_in, c_out, k)).astype(np.float64)
    b2 = rng.normal(size=(c_out,)).astype(np.float64)

    got2 = numpy_conv_transpose1d(x2, w2, b2, stride=stride, padding=padding, output_padding=output_padding)

    with torch.no_grad():
        t_out = torch.nn.functional.conv_transpose1d(
            torch.from_numpy(x2),
            torch.from_numpy(w2),
            bias=torch.from_numpy(b2),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ).numpy()

    assert got2.shape == t_out.shape, f"random case shape mismatch: {got2.shape} vs {t_out.shape}"
    abs_err = np.abs(got2 - t_out).max()
    assert abs_err < 1e-9, f"random case max|err|={abs_err:.6g} too large"
    print(
        f"[hand/torch-check] tiny case OK; random (c_in={c_in},c_out={c_out},k={k},"
        f"stride={stride},pad={padding},output_padding={output_padding}) max|err|={abs_err:.6g}"
    )


# --------------------------------------------------------------------------------------
# Full block wiring, FP64.
# --------------------------------------------------------------------------------------


def numpy_boson_decoder_block(
    x: np.ndarray,
    alpha0: np.ndarray,
    ct_weight: np.ndarray,
    ct_bias: np.ndarray,
    stride: int,
    res_params: list[tuple],
) -> np.ndarray:
    """Mirrors `_BosonDecoderBlock.forward` exactly:
      Snake1d(input_dim) -> wn_conv_transpose1d(k=2*stride, stride, padding=ceil(stride/2),
      output_padding=stride%2) -> ResidualUnit(dilation=1) -> ResidualUnit(dilation=3) ->
      ResidualUnit(dilation=9).

    res_params: list of 3 tuples (alpha1, w1, b1, alpha2, w2, b2, dilation, padding), one per
    residual unit, in the order they are applied (dilation=1, 3, 9).
    """
    y = numpy_snake(x, alpha0)
    ct_padding = math.ceil(stride / 2)
    ct_output_padding = stride % 2
    y = numpy_conv_transpose1d(y, ct_weight, ct_bias, stride=stride, padding=ct_padding, output_padding=ct_output_padding)
    for alpha1, w1, b1, alpha2, w2, b2, dilation, padding in res_params:
        y = numpy_residual_unit(y, alpha1, w1, b1, alpha2, w2, b2, dilation, padding)
    return y


# --------------------------------------------------------------------------------------
# Cross-check against the real PyTorch `_BosonDecoderBlock` class.
# --------------------------------------------------------------------------------------


def _extract_residual_unit_params(ru_module) -> tuple:
    """ru_module is a `_BosonResidualUnit`. Returns
    (alpha1, w1, b1, alpha2, w2, b2, dilation, padding) as FP64 numpy arrays / ints,
    folding weight_norm on the host exactly as `fold_weight_norm` does (FP64 fold, matches
    the M3-3 precision-sequencing policy in spirit even though this run uses synthetic
    FP32-native weights, not BF16 checkpoint tensors)."""
    snake1, conv1, snake2, conv2 = ru_module.block
    alpha1 = snake1.alpha.detach().numpy()
    alpha2 = snake2.alpha.detach().numpy()
    w1 = fold_weight_norm(conv1.weight_g.detach().numpy(), conv1.weight_v.detach().numpy())
    b1 = conv1.bias.detach().numpy()
    w2 = fold_weight_norm(conv2.weight_g.detach().numpy(), conv2.weight_v.detach().numpy())
    b2 = conv2.bias.detach().numpy()
    dilation = conv1.dilation[0]
    padding = conv1.padding[0]
    return alpha1, w1, b1, alpha2, w2, b2, dilation, padding


def run_cross_check(input_dim: int, output_dim: int, stride: int, seq_len: int, seed: int) -> float:
    decoder_mod = _load_decoder_module()
    torch.manual_seed(seed)
    block = decoder_mod._BosonDecoderBlock(input_dim, output_dim, stride)
    block.eval()

    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-2.0, 2.0, size=(1, input_dim, seq_len)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    with torch.no_grad():
        torch_out = block(x_t).numpy()

    snake0, conv_t1, ru1, ru2, ru3 = block.block
    alpha0 = snake0.alpha.detach().numpy()
    ct_weight = fold_weight_norm(conv_t1.weight_g.detach().numpy(), conv_t1.weight_v.detach().numpy())
    ct_bias = conv_t1.bias.detach().numpy()

    res_params = [_extract_residual_unit_params(ru) for ru in (ru1, ru2, ru3)]
    dilations = [p[6] for p in res_params]
    assert dilations == [1, 3, 9], f"unexpected dilation order: {dilations}"

    ref = numpy_boson_decoder_block(x_np, alpha0, ct_weight, ct_bias, stride, res_params)

    print(f"input shape={x_np.shape}, torch output shape={torch_out.shape}, numpy-FP64 ref shape={ref.shape}")
    if torch_out.shape != ref.shape:
        print(f"SHAPE MISMATCH: torch {torch_out.shape} vs FP64 ref {ref.shape}")
        return float("inf")

    abs_err = np.abs(torch_out.astype(np.float64) - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-8)
    max_abs_err = abs_err.max()
    print(
        f"stride={stride} input_dim={input_dim} output_dim={output_dim} seq_len={seq_len}: "
        f"max|err|={max_abs_err:.6g} max_rel_err={rel_err.max():.6g} "
        f"nan/inf(torch)={int(np.sum(~np.isfinite(torch_out)))}"
    )
    return float(max_abs_err)


# --------------------------------------------------------------------------------------
# Real-checkpoint cross-check, reusing M3-3's extraction code (m3_block_weights.py), which
# landed (with a results doc) while this task was in progress. Per M3-3's finding, the real
# checkpoint's acoustic_decoder.block.N.* tensors are PLAIN, ALREADY-FOLDED weight/bias pairs
# (no weight_g/weight_v split survives) -- so there is no fold step here; the extracted
# tensors are used directly as effective conv weights, upcast BF16 -> FP32 for the PyTorch
# forward and -> FP64 for the NumPy reference.
# --------------------------------------------------------------------------------------


def run_real_weight_cross_check(seq_len: int, seed: int) -> float:
    import m3_block_weights as m33  # noqa: PLC0415 (only imported when this path runs)
    import torch.nn.functional as F

    decoder_mod = _load_decoder_module()

    raw = {name: m33.load_raw(f"{name}.weight") for name in m33.CONV_KERNEL_NAMES}
    raw_bias = {name: m33.load_raw(f"{name}.bias") for name in m33.CONV_KERNEL_NAMES}
    raw_alpha = {name: m33.load_raw(f"{name}.alpha") for name in m33.ALPHA_NAMES}

    w = {name: t.to(torch.float32) for name, t in raw.items()}
    bias = {name: t.to(torch.float32) for name, t in raw_bias.items()}
    alpha = {name: t.to(torch.float32) for name, t in raw_alpha.items()}

    print(f"real checkpoint: block index {m33.BLOCK_INDEX} (stride=5, 512->256)")
    for name, t in w.items():
        print(f"  {name}: weight shape={tuple(t.shape)} bias shape={tuple(bias[name].shape)}")

    stride = 5
    dilations = {"res_unit1": 1, "res_unit2": 3, "res_unit3": 9}
    paddings = {name: ((7 - 1) * d) // 2 for name, d in dilations.items()}

    rng = np.random.default_rng(seed)
    input_dim = w["conv_t1"].shape[0]  # ConvTranspose1d layout: [C_in, C_out, K]
    x_np = rng.uniform(-2.0, 2.0, size=(1, input_dim, seq_len)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    # --- real PyTorch FP32 forward, using the real extracted weights directly (matches
    # the checkpoint's actual inference path per M3-3: plain conv, no weight_norm active). ---
    with torch.no_grad():
        y = decoder_mod._snake(x_t, alpha["snake1"])
        y = F.conv_transpose1d(
            y, w["conv_t1"], bias["conv_t1"], stride=stride,
            padding=math.ceil(stride / 2), output_padding=stride % 2,
        )
        for ru in ("res_unit1", "res_unit2", "res_unit3"):
            z = decoder_mod._snake(y, alpha[f"{ru}.snake1"])
            z = F.conv1d(z, w[f"{ru}.conv1"], bias[f"{ru}.conv1"], dilation=dilations[ru], padding=paddings[ru])
            z = decoder_mod._snake(z, alpha[f"{ru}.snake2"])
            z = F.conv1d(z, w[f"{ru}.conv2"], bias[f"{ru}.conv2"])
            pad = (y.shape[-1] - z.shape[-1]) // 2
            yc = y[..., pad:-pad] if pad > 0 else y
            y = yc + z
        torch_out = y.numpy()

    # --- FP64 NumPy reference, same real weights (upcast BF16 -> FP64), same input. ---
    res_params = [
        (
            alpha[f"{ru}.snake1"].numpy(),
            w[f"{ru}.conv1"].numpy(),
            bias[f"{ru}.conv1"].numpy(),
            alpha[f"{ru}.snake2"].numpy(),
            w[f"{ru}.conv2"].numpy(),
            bias[f"{ru}.conv2"].numpy(),
            dilations[ru],
            paddings[ru],
        )
        for ru in ("res_unit1", "res_unit2", "res_unit3")
    ]
    ref = numpy_boson_decoder_block(
        x_np, alpha["snake1"].numpy(), w["conv_t1"].numpy(), bias["conv_t1"].numpy(), stride, res_params
    )

    print(f"input shape={x_np.shape}, torch (real weights) output shape={torch_out.shape}, numpy-FP64 ref shape={ref.shape}")
    if torch_out.shape != ref.shape:
        print(f"SHAPE MISMATCH: torch {torch_out.shape} vs FP64 ref {ref.shape}")
        return float("inf")

    abs_err = np.abs(torch_out.astype(np.float64) - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-8)
    max_abs_err = abs_err.max()
    print(
        f"REAL WEIGHTS stride=5 block{m33.BLOCK_INDEX} seq_len={seq_len}: "
        f"max|err|={max_abs_err:.6g} max_rel_err={rel_err.max():.6g} "
        f"nan/inf(torch)={int(np.sum(~np.isfinite(torch_out)))}"
    )

    # --- diagnostic control: is a gap here FP32-rounding noise (expected, harmless) or a
    # genuine reference bug? Re-run the SAME chain in torch FP64 (same real weights, same
    # input, upcast) and compare against (a) torch's own FP32 forward and (b) this script's
    # NumPy-FP64 reference. If (a) and the NumPy-vs-torch-FP32 gap above are the same order
    # of magnitude, and (b) NumPy-vs-torch-FP64 is near machine epsilon, that proves the
    # NumPy FP64 path is correct and any gap above threshold is pure FP32 rounding-order
    # noise from chaining 5 real conv/convT layers at real checkpoint weight magnitude --
    # not a bug in this reference implementation.
    w64 = {name: t.to(torch.float64) for name, t in raw.items()}
    bias64 = {name: t.to(torch.float64) for name, t in raw_bias.items()}
    alpha64 = {name: t.to(torch.float64) for name, t in raw_alpha.items()}
    x_t64 = torch.from_numpy(x_np.astype(np.float64))
    with torch.no_grad():
        y = decoder_mod._snake(x_t64, alpha64["snake1"])
        y = F.conv_transpose1d(
            y, w64["conv_t1"], bias64["conv_t1"], stride=stride,
            padding=math.ceil(stride / 2), output_padding=stride % 2,
        )
        for ru in ("res_unit1", "res_unit2", "res_unit3"):
            z = decoder_mod._snake(y, alpha64[f"{ru}.snake1"])
            z = F.conv1d(z, w64[f"{ru}.conv1"], bias64[f"{ru}.conv1"], dilation=dilations[ru], padding=paddings[ru])
            z = decoder_mod._snake(z, alpha64[f"{ru}.snake2"])
            z = F.conv1d(z, w64[f"{ru}.conv2"], bias64[f"{ru}.conv2"])
            pad = (y.shape[-1] - z.shape[-1]) // 2
            yc = y[..., pad:-pad] if pad > 0 else y
            y = yc + z
        torch_fp64_out = y.numpy()
    torch32_vs_torch64 = float(np.abs(torch_out.astype(np.float64) - torch_fp64_out).max())
    numpy_vs_torch64 = float(np.abs(ref - torch_fp64_out).max())
    print(
        f"  [diagnostic] torch-FP32-forward vs torch-FP64-forward (same real weights/input): "
        f"max|err|={torch32_vs_torch64:.6g}  |  numpy-FP64-ref vs torch-FP64-forward: "
        f"max|err|={numpy_vs_torch64:.6g}"
    )

    return float(max_abs_err)


if __name__ == "__main__":
    print("=== hand/torch-checked numpy_conv_transpose1d validation ===")
    _validate_conv_transpose_hand_case()

    print("\n=== M3-4 cross-check (synthetic weights): FP64 NumPy _BosonDecoderBlock vs real PyTorch FP32 forward ===")
    print("Weight source: SYNTHETIC (PyTorch default nn.Module init) -- see module docstring.")

    results = {}
    results["stride5_512x256_synthetic"] = run_cross_check(input_dim=512, output_dim=256, stride=5, seq_len=20, seed=3141)
    results["stride8_1024x512_synthetic"] = run_cross_check(input_dim=1024, output_dim=512, stride=8, seq_len=12, seed=2718)

    print("\n=== M3-4 cross-check (REAL checkpoint weights, block 1, stride=5, via M3-3's extraction) ===")
    try:
        results["stride5_512x256_real_weights"] = run_real_weight_cross_check(seq_len=20, seed=99)
    except Exception as e:  # noqa: BLE001
        print(f"REAL-WEIGHT cross-check unavailable/failed: {type(e).__name__}: {e}")

    print("\n=== summary ===")
    threshold = 1e-5
    all_pass = True
    for name, max_abs_err in results.items():
        status = "PASS" if max_abs_err < threshold else "FAIL"
        all_pass = all_pass and (max_abs_err < threshold)
        print(f"{name}: max|err|={max_abs_err:.6g} threshold={threshold:.0e} -> {status}")
    print(f"\nOVERALL (literal <1e-5 gate): {'PASS' if all_pass else 'FAIL'}")
    print(
        "Note: if the real-weight case shows FAIL, see the '[diagnostic]' line above -- if "
        "torch-FP32-forward vs torch-FP64-forward differs by the SAME order of magnitude as "
        "numpy-FP64-ref vs torch-FP32-forward, and numpy-FP64-ref vs torch-FP64-forward is near "
        "machine epsilon, the gap is pure FP32 rounding-order noise at real-checkpoint weight "
        "magnitude/depth, not a bug in the FP64 reference -- report this explicitly rather than "
        "silently loosening the threshold or claiming an unqualified PASS."
    )
