"""M2 prototype #4 for issue #57: the full `_BosonResidualUnit` composite
(Snake1d -> weight-normed dilated Conv1d -> Snake1d -> weight-normed pointwise Conv1d ->
symmetric-crop residual add) as ONE MAX graph, combining the two already-validated primitives
(m2_snake1d_prototype.py, m2_conv1d_prototype.py) into the actual composite unit used 3x per
decoder block in Higgs's real architecture (higgs_audio_decoder.py:114-130).

Per m1-responsibility-map.md S7: "37 convolutions is enough depth for error to compound... compare
layer-by-layer, not just end-to-end." This tests one full residual unit end-to-end -- a smaller
composite than the whole decoder (which additionally needs ConvTranspose1d, currently blocked on
Metal GPU per m2-convtranspose1d-results.md) but a real step up from isolated single-op tests.

Usage: pixi run python m2_residual_unit_prototype.py
"""

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

EPS = 1e-9


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    v64 = v.astype(np.float64)
    norm = np.sqrt(np.sum(v64 ** 2, axis=(1, 2), keepdims=True))
    return (g.astype(np.float64) * v64 / norm).astype(np.float32)


def numpy_conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int, padding: int) -> np.ndarray:
    b, c_in, t = x.shape
    c_out, _, k = weight.shape
    x64 = x.astype(np.float64)
    xp = np.pad(x64, ((0, 0), (0, 0), (padding, padding)))
    t_out = xp.shape[-1] - (k - 1) * dilation - 1 + 1
    out = np.zeros((b, c_out, t_out), dtype=np.float64)
    w64 = weight.astype(np.float64)
    for kk in range(k):
        offset = kk * dilation
        out += np.einsum("oc,bct->bot", w64[:, :, kk], xp[:, :, offset : offset + t_out])
    return out + bias.astype(np.float64)[None, :, None]


def numpy_snake(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    x64, a64 = x.astype(np.float64), alpha.astype(np.float64)
    return x64 + (1.0 / (a64 + EPS)) * np.sin(a64 * x64) ** 2


def numpy_residual_unit(x, alpha1, w1, b1, alpha2, w2, b2, dilation, padding):
    """Mirrors _BosonResidualUnit.forward exactly: block(x) then symmetric-crop residual add."""
    y = numpy_snake(x, alpha1)
    y = numpy_conv1d(y, w1, b1, dilation=dilation, padding=padding)
    y = numpy_snake(y, alpha2)
    y = numpy_conv1d(y, w2, b2, dilation=1, padding=0)  # pointwise, k=1
    pad = (x.shape[-1] - y.shape[-1]) // 2
    xc = x.astype(np.float64)
    if pad > 0:
        xc = xc[..., pad:-pad]
    return xc + y


def snake_expr(x, alpha, device):
    eps = ops.constant(EPS, DType.float32, device=device)
    recip = ops.div(ops.constant(1.0, DType.float32, device=device), ops.add(alpha, eps))
    return ops.add(x, ops.mul(recip, ops.pow(ops.sin(ops.mul(alpha, x)), ops.constant(2.0, DType.float32, device=device))))


def conv1d_expr(x, filter_rscf, bias, dilation, padding):
    x_nhwc = ops.unsqueeze(ops.permute(x, [0, 2, 1]), 1)
    y_nhwc = ops.conv2d(
        x_nhwc, filter_rscf, stride=(1, 1), dilation=(1, dilation),
        padding=(0, 0, padding, padding), bias=bias,
    )
    return ops.permute(ops.squeeze(y_nhwc, 1), [0, 2, 1])


def build_graph(dim, kernel, dilation, padding, seq_len, batch, device):
    def forward(x, alpha1, filter1, bias1, alpha2, filter2, bias2):
        y = snake_expr(x, alpha1, device)
        y = conv1d_expr(y, filter1, bias1, dilation, padding)
        y = snake_expr(y, alpha2, device)
        y = conv1d_expr(y, filter2, bias2, 1, 0)  # pointwise
        # symmetric crop -- guarded, matching _BosonResidualUnit's own guard (only crop if shorter)
        x_len = int(x.shape[-1])
        y_len = int(y.shape[-1])
        pad = (x_len - y_len) // 2
        xc = x[:, :, pad : x_len - pad] if pad > 0 else x
        return ops.add(xc, y)

    return Graph(
        "residual_unit",
        forward=forward,
        input_types=[
            TensorType(DType.float32, shape=(batch, dim, seq_len), device=device),
            TensorType(DType.float32, shape=(1, dim, 1), device=device),
            TensorType(DType.float32, shape=(1, kernel, dim, dim), device=device),
            TensorType(DType.float32, shape=(dim,), device=device),
            TensorType(DType.float32, shape=(1, dim, 1), device=device),
            TensorType(DType.float32, shape=(1, 1, dim, dim), device=device),
            TensorType(DType.float32, shape=(dim,), device=device),
        ],
    )


if __name__ == "__main__":
    rng = np.random.default_rng(3141)
    batch, dim, seq_len = 1, 32, 64
    kernel, dilation = 7, 3
    padding = (kernel - 1) * dilation // 2

    x_np = rng.uniform(-2.0, 2.0, size=(batch, dim, seq_len)).astype(np.float32)
    alpha1_np = rng.uniform(0.3, 1.2, size=(1, dim, 1)).astype(np.float32)
    alpha2_np = rng.uniform(0.3, 1.2, size=(1, dim, 1)).astype(np.float32)

    g1 = rng.uniform(0.5, 2.0, size=(dim, 1, 1)).astype(np.float32)
    v1 = rng.normal(0, 0.3, size=(dim, dim, kernel)).astype(np.float32)
    w1 = fold_weight_norm(g1, v1)
    b1 = rng.normal(0, 0.1, size=(dim,)).astype(np.float32)

    g2 = rng.uniform(0.5, 2.0, size=(dim, 1, 1)).astype(np.float32)
    v2 = rng.normal(0, 0.3, size=(dim, dim, 1)).astype(np.float32)
    w2 = fold_weight_norm(g2, v2)
    b2 = rng.normal(0, 0.1, size=(dim,)).astype(np.float32)

    ref = numpy_residual_unit(x_np, alpha1_np, w1, b1, alpha2_np, w2, b2, dilation, padding)
    print(f"reference output shape={ref.shape}")

    device_obj = CPU() if accelerator_count() == 0 else Accelerator()
    device = DeviceRef.from_device(device_obj)
    print(f"device: {device_obj}, accelerator_count={accelerator_count()}")

    graph = build_graph(dim, kernel, dilation, padding, seq_len, batch, device)
    session = InferenceSession(devices=[device_obj])
    model = session.load(graph)

    filter1_rscf = np.transpose(w1, (2, 1, 0))[np.newaxis, ...].copy()
    filter2_rscf = np.transpose(w2, (2, 1, 0))[np.newaxis, ...].copy()

    bufs = [
        Buffer.from_numpy(x_np).to(device_obj),
        Buffer.from_numpy(alpha1_np).to(device_obj),
        Buffer.from_numpy(filter1_rscf).to(device_obj),
        Buffer.from_numpy(b1).to(device_obj),
        Buffer.from_numpy(alpha2_np).to(device_obj),
        Buffer.from_numpy(filter2_rscf).to(device_obj),
        Buffer.from_numpy(b2).to(device_obj),
    ]
    result = model.execute(*bufs)[0]
    got = result.to(CPU()).to_numpy().astype(np.float64)

    print(f"MAX output shape={got.shape}")
    if got.shape != ref.shape:
        print(f"SHAPE MISMATCH: MAX {got.shape} vs reference {ref.shape}")
    else:
        abs_err = np.abs(got - ref)
        rel_err = abs_err / (np.abs(ref) + 1e-8)
        print(
            f"max|err|={abs_err.max():.6g} max_rel_err={rel_err.max():.6g} "
            f"nan/inf={int(np.sum(~np.isfinite(got)))}"
        )
