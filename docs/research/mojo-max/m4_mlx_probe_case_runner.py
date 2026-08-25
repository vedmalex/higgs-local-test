"""One-case-per-subprocess runner for MLX conv_transpose1d / conv1d(dilation) / Snake GPU checks.

Invoked by m4_mlx_gpu_probe.py as a subprocess (python -u this_file.py <case_id>) so
a fatal GPU abort (SIGABRT etc, per this project's own established MAX finding that GPU crashes
can be uncatchable process aborts, not Python exceptions) in one case cannot hide the others.

Reuses `numpy_conv_transpose1d` from m3_block_reference.py (validated to 1.11e-16 against
PyTorch per that file's own hand/torch cross-check) and `numpy_conv1d`/`numpy_snake` from
m2_residual_unit_prototype.py, both imported unmodified, NOT reimplemented.

MLX layout (per `mx.conv_transpose1d.__doc__` / `mx.conv1d.__doc__`, MLX 0.32.1, checked
empirically, not assumed):
  - input:  (N, L, C)      -- NLC, channels-LAST (unlike PyTorch/numpy's (N, C, L))
  - weight (conv_transpose1d): (C_out, K, C_in)
  - weight (conv1d):           (C_out, K, C_in)
  - conv_transpose1d itself takes stride/padding/dilation/output_padding directly (no manual
    crop-after-full-output workaround needed here, unlike the MAX `ops.conv2d_transpose` route-A
    pattern in m3_decoder_block_prototype.conv_transpose_expr).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_MOJO_MAX_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_MOJO_MAX_DIR))

import mlx.core as mx  # noqa: E402

from m3_block_reference import numpy_conv_transpose1d  # noqa: E402
from m2_residual_unit_prototype import numpy_conv1d, numpy_snake  # noqa: E402


def case_convtranspose(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    b, c_in, c_out, k, seq_len = 1, cfg["c_in"], cfg["c_out"], cfg["kernel"], cfg["seq_len"]
    stride, padding, output_padding = cfg["stride"], cfg["padding"], cfg["output_padding"]

    x_np = rng.uniform(-1.0, 1.0, size=(b, c_in, seq_len)).astype(np.float32)
    # PyTorch/reference ConvTranspose1d weight layout: [C_in, C_out, K].
    w_np = rng.normal(0, 0.1, size=(c_in, c_out, k)).astype(np.float32)
    bias_np = rng.normal(0, 0.05, size=(c_out,)).astype(np.float32)

    ref = numpy_conv_transpose1d(x_np, w_np, bias_np, stride=stride, padding=padding, output_padding=output_padding)

    # MLX layout: input (N, L, C_in); weight (C_out, K, C_in).
    x_mx = mx.array(np.transpose(x_np, (0, 2, 1)))  # (B,C,L) -> (B,L,C)
    w_mx = mx.array(np.transpose(w_np, (1, 2, 0)))  # (C_in,C_out,K) -> (C_out,K,C_in)
    bias_mx = mx.array(bias_np)

    y_mx = mx.conv_transpose1d(
        x_mx, w_mx, stride=stride, padding=padding, dilation=1, output_padding=output_padding
    )
    y_mx = y_mx + bias_mx  # broadcast over (N, L, C_out)
    mx.eval(y_mx)
    got = np.transpose(np.array(y_mx, dtype=np.float64), (0, 2, 1))  # (B,L,C)->(B,C,L)

    return {
        "got_shape": list(got.shape),
        "ref_shape": list(ref.shape),
        "shape_match": list(got.shape) == list(ref.shape),
        "max_abs_err": float(np.abs(got - ref).max()) if list(got.shape) == list(ref.shape) else None,
        "nan_inf": int(np.sum(~np.isfinite(got))),
    }


def case_conv1d_dilation(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    b, c_in, c_out, k, seq_len = 1, cfg["c_in"], cfg["c_out"], cfg["kernel"], cfg["seq_len"]
    dilation, padding = cfg["dilation"], cfg["padding"]

    x_np = rng.uniform(-1.0, 1.0, size=(b, c_in, seq_len)).astype(np.float32)
    w_np = rng.normal(0, 0.1, size=(c_out, c_in, k)).astype(np.float32)  # numpy_conv1d layout [C_out,C_in,K]
    bias_np = rng.normal(0, 0.05, size=(c_out,)).astype(np.float32)

    ref = numpy_conv1d(x_np, w_np, bias_np, dilation=dilation, padding=padding)

    x_mx = mx.array(np.transpose(x_np, (0, 2, 1)))  # (B,C,L)->(B,L,C)
    w_mx = mx.array(np.transpose(w_np, (0, 2, 1)))  # (C_out,C_in,K)->(C_out,K,C_in)
    bias_mx = mx.array(bias_np)

    y_mx = mx.conv1d(x_mx, w_mx, stride=1, padding=padding, dilation=dilation)
    y_mx = y_mx + bias_mx
    mx.eval(y_mx)
    got = np.transpose(np.array(y_mx, dtype=np.float64), (0, 2, 1))

    return {
        "got_shape": list(got.shape),
        "ref_shape": list(ref.shape),
        "shape_match": list(got.shape) == list(ref.shape),
        "max_abs_err": float(np.abs(got - ref).max()) if list(got.shape) == list(ref.shape) else None,
        "nan_inf": int(np.sum(~np.isfinite(got))),
    }


def case_snake(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    c, seq_len = cfg["c"], cfg["seq_len"]
    x_np = rng.uniform(-2.0, 2.0, size=(1, c, seq_len)).astype(np.float32)
    alpha_np = rng.uniform(-0.3, 1.2, size=(1, c, 1)).astype(np.float32)

    ref = numpy_snake(x_np, alpha_np)  # (1, C, L), FP64

    # Elementwise op -- no layout constraint imposed by MLX docs for elementwise ops, but keep
    # NLC for consistency with the conv ops above (channels-last).
    x_mx = mx.array(np.transpose(x_np, (0, 2, 1)))  # (1,C,L)->(1,L,C)
    alpha_mx = mx.array(np.transpose(alpha_np, (0, 2, 1)))  # (1,C,1)->(1,1,C)
    eps = 1e-9
    y_mx = x_mx + (1.0 / (alpha_mx + eps)) * mx.sin(alpha_mx * x_mx) ** 2
    mx.eval(y_mx)
    got = np.transpose(np.array(y_mx, dtype=np.float64), (0, 2, 1))  # (1,L,C)->(1,C,L)

    return {
        "got_shape": list(got.shape),
        "ref_shape": list(ref.shape),
        "shape_match": list(got.shape) == list(ref.shape),
        "max_abs_err": float(np.abs(got - ref).max()) if list(got.shape) == list(ref.shape) else None,
        "nan_inf": int(np.sum(~np.isfinite(got))),
    }


CASES = {
    # ---- ConvTranspose1d: the 5 (stride, output_padding) pairs from m2_convtranspose1d_prototype.py,
    # channels capped at 16, kernel=2*stride, seq_len=16, padding=0 (matches that script exactly).
    "ct_stride8_op0": dict(kind="convtranspose", c_in=16, c_out=16, kernel=16, stride=8, padding=0, output_padding=0, seq_len=16, seed=9012),
    "ct_stride5_op1": dict(kind="convtranspose", c_in=16, c_out=16, kernel=10, stride=5, padding=0, output_padding=1, seq_len=16, seed=9012),
    "ct_stride4_op0": dict(kind="convtranspose", c_in=16, c_out=16, kernel=8, stride=4, padding=0, output_padding=0, seq_len=16, seed=9012),
    "ct_stride2_op0": dict(kind="convtranspose", c_in=16, c_out=16, kernel=4, stride=2, padding=0, output_padding=0, seq_len=16, seed=9012),
    "ct_stride3_op1": dict(kind="convtranspose", c_in=16, c_out=16, kernel=6, stride=3, padding=0, output_padding=1, seq_len=16, seed=9012),
    # ---- Real M3 block-1 shape: input (1,512,20), c_out=256, k=10, stride=5, padding=3, output_padding=1.
    "ct_m3_real_stride5": dict(kind="convtranspose", c_in=512, c_out=256, kernel=10, stride=5, padding=3, output_padding=1, seq_len=20, seed=57305),
    # ---- Conv1d with dilation 1/3/9 (Higgs's ResidualUnit dilations), kernel=7, padding=(k-1)*d//2 (matches m3 ru_paddings formula).
    "conv1d_dilation1": dict(kind="conv1d", c_in=256, c_out=256, kernel=7, dilation=1, padding=(7 - 1) * 1 // 2, seq_len=86, seed=1),
    "conv1d_dilation3": dict(kind="conv1d", c_in=256, c_out=256, kernel=7, dilation=3, padding=(7 - 1) * 3 // 2, seq_len=86, seed=3),
    "conv1d_dilation9": dict(kind="conv1d", c_in=256, c_out=256, kernel=7, dilation=9, padding=(7 - 1) * 9 // 2, seq_len=86, seed=9),
    # ---- Snake activation.
    "snake": dict(kind="snake", c=256, seq_len=86, seed=42),
}


def main():
    case_id = sys.argv[1]
    cfg = CASES[case_id]

    mx.set_default_device(mx.gpu)
    actual_device = str(mx.default_device())

    if cfg["kind"] == "convtranspose":
        result = case_convtranspose(cfg)
    elif cfg["kind"] == "conv1d":
        result = case_conv1d_dilation(cfg)
    elif cfg["kind"] == "snake":
        result = case_snake(cfg)
    else:
        raise ValueError(cfg["kind"])

    result["case_id"] = case_id
    result["cfg"] = cfg
    result["mlx_default_device"] = actual_device
    print("RESULT_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
