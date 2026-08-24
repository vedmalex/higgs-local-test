"""M2 prototype for issue #57: Snake1d (Higgs Code2Wav's activation) in MAX.

snake(x) = x + (alpha + 1e-9)^-1 * sin(alpha * x)^2

Per docs/research/mojo-max/m1-responsibility-map.md's recommended first M2 prototype: this
needs no missing MAX op (ops.add/mul/div/sin/pow all exist), so a failure here is unambiguously
about precision or broadcast semantics, not a missing primitive. Run locally on M1 first per
project convention; the same script is meant to run unchanged on a T4.

Usage: pixi run python m2_snake1d_prototype.py   (from within a pixi env with `modular` added,
e.g. .mojo-probe-stable — see docs/research/mojo-max/m0-results.md's Reproducing section)
"""

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

EPS = 1e-9


def torch_reference(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """FP64 NumPy reference -- stands in for a PyTorch fp32 forward pass without requiring
    torch as a dependency for this standalone probe."""
    x64 = x.astype(np.float64)
    a64 = alpha.astype(np.float64)
    return x64 + (1.0 / (a64 + EPS)) * np.sin(a64 * x64) ** 2


def build_snake_graph(dtype: DType, device: DeviceRef, compute_fp32: bool) -> Graph:
    """Builds snake(x, alpha). If compute_fp32 and dtype != float32, casts both inputs to
    float32 before the expression and casts the result back -- the explicit-cast strategy
    the M1 responsibility map's precision-policy section (S9) requires MAX to do by hand."""

    def forward(x, alpha):
        xc, ac = x, alpha
        if compute_fp32 and dtype != DType.float32:
            xc = ops.cast(x, DType.float32)
            ac = ops.cast(alpha, DType.float32)
        eps = ops.constant(EPS, DType.float32 if compute_fp32 else dtype, device=device)
        recip = ops.div(ops.constant(1.0, ac.dtype, device=device), ops.add(ac, eps))
        y = ops.add(xc, ops.mul(recip, ops.pow(ops.sin(ops.mul(ac, xc)), ops.constant(2.0, ac.dtype, device=device))))
        if compute_fp32 and dtype != DType.float32:
            y = ops.cast(y, dtype)
        return y

    return Graph(
        "snake1d",
        forward=forward,
        input_types=[
            TensorType(dtype, shape=x_shape, device=device),
            TensorType(dtype, shape=alpha_shape, device=device),
        ],
    )


def run_case(label: str, x_np, alpha_np, dtype: DType, compute_fp32: bool, device_obj) -> None:
    device = DeviceRef.from_device(device_obj)
    graph = build_snake_graph(dtype, device, compute_fp32)
    session = InferenceSession(devices=[device_obj])
    model = session.load(graph)

    np_dtype = {DType.float32: np.float32, DType.float16: np.float16}[dtype]
    x_buf = Buffer.from_numpy(x_np.astype(np_dtype)).to(device_obj)
    alpha_buf = Buffer.from_numpy(alpha_np.astype(np_dtype)).to(device_obj)

    result = model.execute(x_buf, alpha_buf)[0]
    got = result.to(CPU()).to_numpy().astype(np.float64)

    ref = torch_reference(x_np, alpha_np)
    abs_err = np.abs(got - ref)
    finite_ref = np.isfinite(ref)
    rel_err = np.zeros_like(abs_err)
    rel_err[finite_ref] = abs_err[finite_ref] / (np.abs(ref[finite_ref]) + 1e-12)

    n_nan_inf = int(np.sum(~np.isfinite(got)))
    n_zero = int(np.sum(got == 0))

    print(
        f"[{label}] max|err|={abs_err.max():.6g} max_rel_err={rel_err.max():.6g} "
        f"nan/inf={n_nan_inf} exact_zeros={n_zero}"
    )


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    x_shape = (1, 32, 256)
    alpha_shape = (1, 32, 1)

    x_np = rng.uniform(-3.0, 3.0, size=x_shape).astype(np.float32)

    # Deliberately include small alpha values -- Snake's most dangerous regime, per
    # m1-responsibility-map.md S4/S9: 1/(alpha+eps) approaches the storage dtype's ceiling
    # as alpha shrinks.
    alpha_np = rng.uniform(0.05, 1.5, size=alpha_shape).astype(np.float32)
    # 1e-4 gives 1/alpha ~= 1e4, comfortably inside FP16's 65504 ceiling -- not a real test
    # of the overflow hypothesis. Push further: 1e-6/1e-7 give 1/alpha ~= 1e6-1e7, which
    # DOES exceed FP16's finite range and is the actual regime M0's finding was about.
    alpha_np[0, :6, 0] = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 5e-7], dtype=np.float32)

    device_obj = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"device: {device_obj}, accelerator_count={accelerator_count()}")

    run_case("fp32", x_np, alpha_np, DType.float32, compute_fp32=False, device_obj=device_obj)
    run_case(
        "fp16 storage, fp32 compute (explicit cast, per S9)",
        x_np,
        alpha_np,
        DType.float16,
        compute_fp32=True,
        device_obj=device_obj,
    )
    run_case(
        "fp16 storage, fp16 compute (no cast -- expected to break)",
        x_np,
        alpha_np,
        DType.float16,
        compute_fp32=False,
        device_obj=device_obj,
    )
