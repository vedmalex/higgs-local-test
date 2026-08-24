"""M2 prototype #2 for issue #57: one weight-normed dilated Conv1d (Higgs's
`_BosonResidualUnit` shape) expressed in MAX via `ops.conv2d` with a degenerate height axis
(route A from docs/research/mojo-max/m1-responsibility-map.md S5/S11).

Answers the single largest structural question in the responsibility map: can Higgs's Conv1d
shape (full/non-causal/dilated, weight-normalized) be expressed in MAX without a custom Mojo
kernel? A pass here means M3 is a graph-assembly task; a failure means M3 needs a hand-written
Mojo conv1d kernel.

Usage: pixi run python m2_conv1d_prototype.py  (same pixi env as m2_snake1d_prototype.py)
"""

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """W = g * v / ||v||, norm taken over all dims except out_channels (dim 0) -- matches
    torch.nn.utils.weight_norm's default dim=0 for Conv1d. Folded in FP64 on the host, per
    m1-responsibility-map.md S5's recommendation (never fold in FP16 -- it's a sum-of-squares
    over C_in*k elements, the same overflow-prone reduction M0 flagged for RMSNorm)."""
    v64 = v.astype(np.float64)
    norm = np.sqrt(np.sum(v64 ** 2, axis=(1, 2), keepdims=True))
    return (g.astype(np.float64) * v64 / norm).astype(np.float32)


def numpy_conv1d_reference(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int, padding: int
) -> np.ndarray:
    """Reference conv1d: x [B,C_in,T], weight [C_out,C_in,K], bias [C_out] -> [B,C_out,T_out].
    Symmetric zero padding, PyTorch-matching semantics, computed in FP64."""
    b, c_in, t = x.shape
    c_out, _, k = weight.shape
    x64 = x.astype(np.float64)
    xp = np.pad(x64, ((0, 0), (0, 0), (padding, padding)))
    t_padded = xp.shape[-1]
    t_out = t_padded - (k - 1) * dilation - 1 + 1
    out = np.zeros((b, c_out, t_out), dtype=np.float64)
    w64 = weight.astype(np.float64)
    for kk in range(k):
        offset = kk * dilation
        # xp[:, :, offset:offset+t_out] has shape [B, C_in, T_out]
        window = xp[:, :, offset : offset + t_out]
        # w64[:, :, kk] has shape [C_out, C_in] -- contract over C_in
        out += np.einsum("oc,bct->bot", w64[:, :, kk], window)
    out += bias.astype(np.float64)[None, :, None]
    return out


def build_conv1d_via_conv2d_graph(
    c_in: int, c_out: int, kernel: int, dilation: int, padding: int, device: DeviceRef, dtype: DType
) -> Graph:
    def forward(x, filter_rscf, bias):
        # x: [B, C_in, T] -> NHWC [B, 1, T, C_in]
        x_nhwc = ops.unsqueeze(ops.permute(x, [0, 2, 1]), 1)
        y_nhwc = ops.conv2d(
            x_nhwc,
            filter_rscf,
            stride=(1, 1),
            dilation=(1, dilation),
            padding=(0, 0, padding, padding),
            bias=bias,
        )  # defaults: input_layout=NHWC, filter_layout=RSCF -- exactly what we built above
        # y_nhwc: [B, 1, T_out, C_out] -> [B, C_out, T_out]
        return ops.permute(ops.squeeze(y_nhwc, 1), [0, 2, 1])

    return Graph(
        "conv1d_via_conv2d",
        forward=forward,
        input_types=[
            TensorType(dtype, shape=(batch, c_in, seq_len), device=device),
            TensorType(dtype, shape=(1, kernel, c_in, c_out), device=device),
            TensorType(dtype, shape=(c_out,), device=device),
        ],
    )


if __name__ == "__main__":
    rng = np.random.default_rng(5678)

    batch, c_in, c_out, seq_len = 1, 32, 32, 64
    kernel, dilation = 7, 3
    padding = (kernel - 1) * dilation // 2  # matches _BosonResidualUnit's own formula

    x_np = rng.uniform(-2.0, 2.0, size=(batch, c_in, seq_len)).astype(np.float32)
    # weight_norm parametrization: g [C_out,1,1], v [C_out,C_in,K]
    g_np = rng.uniform(0.5, 2.0, size=(c_out, 1, 1)).astype(np.float32)
    v_np = rng.normal(0, 0.3, size=(c_out, c_in, kernel)).astype(np.float32)
    bias_np = rng.normal(0, 0.1, size=(c_out,)).astype(np.float32)

    weight_np = fold_weight_norm(g_np, v_np)  # [C_out, C_in, K], the "PyTorch layout" weight

    ref = numpy_conv1d_reference(x_np, weight_np, bias_np, dilation, padding)
    print(f"padding={padding}, input T={seq_len}, reference output shape={ref.shape}")

    device_obj = CPU() if accelerator_count() == 0 else Accelerator()
    device = DeviceRef.from_device(device_obj)
    print(f"device: {device_obj}, accelerator_count={accelerator_count()}")

    graph = build_conv1d_via_conv2d_graph(c_in, c_out, kernel, dilation, padding, device, DType.float32)
    session = InferenceSession(devices=[device_obj])
    model = session.load(graph)

    # weight_np is [C_out, C_in, K] (PyTorch layout) -> RSCF [1, K, C_in, C_out]
    filter_rscf = np.transpose(weight_np, (2, 1, 0))[np.newaxis, ...].copy()

    x_buf = Buffer.from_numpy(x_np).to(device_obj)
    filter_buf = Buffer.from_numpy(filter_rscf).to(device_obj)
    bias_buf = Buffer.from_numpy(bias_np).to(device_obj)

    result = model.execute(x_buf, filter_buf, bias_buf)[0]
    got = result.to(CPU()).to_numpy().astype(np.float64)

    print(f"MAX output shape={got.shape}")
    if got.shape != ref.shape:
        print(f"SHAPE MISMATCH: MAX {got.shape} vs reference {ref.shape} -- this alone is a fail")
    else:
        abs_err = np.abs(got - ref)
        rel_err = abs_err / (np.abs(ref) + 1e-8)
        print(
            f"max|err|={abs_err.max():.6g} max_rel_err={rel_err.max():.6g} "
            f"nan/inf={int(np.sum(~np.isfinite(got)))}"
        )
