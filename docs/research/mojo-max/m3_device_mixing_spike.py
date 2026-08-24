"""M3-1 for issue #57 (SPIKE, BLOCKING): prove or disprove mixed CPU/GPU placement inside
ONE MAX graph.

Minimal graph under test (per docs/research/mojo-max/m3-plan.md M3-1 / S3):

    GPU Snake1d -> ops.transfer_to(CPU) -> ops.conv2d_transpose on CPU-placed operands
                (activation AND filter/bias) -> ops.transfer_to(GPU) -> GPU Snake1d

Session is constructed with InferenceSession(devices=[Accelerator()]) -- the host CPU is
appended automatically by max/python/max/engine/api.py (InferenceSession.__init__: "host_cpu =
CPU(); if host_cpu not in seen_devices: final_devices.append(host_cpu)", confirmed by reading
that source under this repo's pinned pixi env, see m3-device-mixing-results.md "Setup").

Per m2-convtranspose1d-results.md, the Metal GPU dispatch path for `ops.conv2d_transpose` is a
FATAL PROCESS ABORT ("symbol not found: cudnnCreate"), not a catchable Python exception -- so if
placement is somehow ignored and the op dispatches to GPU anyway, this script (run in-process)
would simply die with no traceback. That is why the actual "did it really run on CPU" check is
done by running the graph-execution step in an ISOLATED SUBPROCESS (exactly as
m2_convtranspose1d_prototype.py's follow-up runner needed): a clean subprocess exit with a
correct finite result is the positive signal (CPU dispatch, on this pinned toolchain where the
GPU path is known-fatal); a subprocess abort is the negative signal.

Usage:
    pixi run python m3_device_mixing_spike.py            # driver: spawns the isolated subprocess
    pixi run python m3_device_mixing_spike.py --run-graph  # (internal) actually builds+runs the graph
"""

import subprocess
import sys

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

EPS = 1e-9

# ---- shapes for the minimal spike -----------------------------------------------------------
BATCH = 1
C_IN = 8
SEQ_LEN = 12
STRIDE = 5
OUTPUT_PADDING = 1  # stride=5 -> output_padding = 5 % 2 = 1, the risky real Higgs case
KERNEL = 2 * STRIDE
C_OUT = 6


# ---- FP64 NumPy reference (mirrors m2_residual_unit_prototype.py's numpy_snake /
# m2_convtranspose1d_prototype.py's transposed-conv semantics; written fresh here since no
# existing helper covers this exact snake->convT->snake pipeline) --------------------------
def numpy_snake(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    x64, a64 = x.astype(np.float64), alpha.astype(np.float64)
    return x64 + (1.0 / (a64 + EPS)) * np.sin(a64 * x64) ** 2


def numpy_conv_transpose1d(
    x: np.ndarray, weight_io_k: np.ndarray, bias: np.ndarray, stride: int, output_padding: int
) -> np.ndarray:
    """weight_io_k: PyTorch ConvTranspose1d layout [in_channels, out_channels, kernel]."""
    b, c_in, t_in = x.shape
    _, c_out, k = weight_io_k.shape
    x64 = x.astype(np.float64)
    w64 = weight_io_k.astype(np.float64)
    t_out = (t_in - 1) * stride + k + output_padding
    out = np.zeros((b, c_out, t_out), dtype=np.float64)
    for t in range(t_in):
        for kk in range(k):
            pos = t * stride + kk
            out[:, :, pos] += np.einsum("bi,io->bo", x64[:, :, t], w64[:, :, kk])
    return out + bias.astype(np.float64)[None, :, None]


def numpy_reference(x, alpha1, weight_io_k, bias, alpha2, stride, output_padding):
    y = numpy_snake(x, alpha1)
    y = numpy_conv_transpose1d(y, weight_io_k, bias, stride, output_padding)
    y = numpy_snake(y, alpha2)
    return y


# ---- MAX graph expressions -------------------------------------------------------------------
def snake_expr(x, alpha, device):
    eps = ops.constant(EPS, DType.float32, device=device)
    recip = ops.div(ops.constant(1.0, DType.float32, device=device), ops.add(alpha, eps))
    return ops.add(
        x, ops.mul(recip, ops.pow(ops.sin(ops.mul(alpha, x)), ops.constant(2.0, DType.float32, device=device)))
    )


def build_mixed_device_graph(gpu_device: DeviceRef, cpu_device: DeviceRef) -> Graph:
    """GPU Snake1d -> transfer_to(CPU) -> CPU conv2d_transpose (filter+bias also CPU-placed)
    -> transfer_to(GPU) -> GPU Snake1d."""

    def forward(x, alpha1, filter_rscf, bias, alpha2):
        # -- GPU stage 1 --
        y = snake_expr(x, alpha1, gpu_device)

        # -- cross to CPU: activation only (filter/bias are declared CPU-native inputs below,
        # never touching GPU at all -- this is the M3-1 requirement that the CPU dispatch is
        # tested/achieved for real, not just a CPU-placed activation next to a GPU weight) --
        y_cpu = ops.transfer_to(y, cpu_device)

        # NHWC route-A layout, identical to m2_convtranspose1d_prototype.py's build_graph.
        # NOTE: passing `bias=` directly into ops.conv2d_transpose was tried first and produces
        # a malformed shape ((1, C_out, T_out, C_out) instead of (1, 1, T_out, C_out)) for H=1
        # inputs on this MAX version -- confirmed by tracing the op's symbolic output shape
        # standalone before writing this script. m2_convtranspose1d_prototype.py's working route
        # never exercised the bias path either. Route around it exactly like that prototype:
        # call conv2d_transpose with bias=None, then add the (CPU-placed) bias manually after
        # squeeze/permute back to [B, C, T].
        y_nhwc = ops.unsqueeze(ops.permute(y_cpu, [0, 2, 1]), 1)
        z_nhwc = ops.conv2d_transpose(
            y_nhwc,
            filter_rscf,
            stride=(1, STRIDE),
            padding=(0, 0, 0, 0),
            output_paddings=(0, OUTPUT_PADDING),
        )
        z_cpu = ops.permute(ops.squeeze(z_nhwc, 1), [0, 2, 1])
        z_cpu = ops.add(z_cpu, ops.reshape(bias, (1, C_OUT, 1)))

        # -- cross back to GPU --
        z_gpu = ops.transfer_to(z_cpu, gpu_device)

        # -- GPU stage 2 --
        return snake_expr(z_gpu, alpha2, gpu_device)

    return Graph(
        "m3_device_mixing_spike",
        forward=forward,
        input_types=[
            TensorType(DType.float32, shape=(BATCH, C_IN, SEQ_LEN), device=gpu_device),
            TensorType(DType.float32, shape=(1, C_IN, 1), device=gpu_device),
            TensorType(DType.float32, shape=(1, KERNEL, C_OUT, C_IN), device=cpu_device),
            TensorType(DType.float32, shape=(C_OUT,), device=cpu_device),
            TensorType(DType.float32, shape=(1, C_OUT, 1), device=gpu_device),
        ],
    )


def run_graph() -> int:
    """Actually builds and executes the mixed-device graph. Run this ONLY inside an isolated
    subprocess (see main() below) -- a Metal `conv2d_transpose` GPU-dispatch bug would abort the
    whole process with no Python exception, per m2-convtranspose1d-results.md."""
    from max.driver import CPU, Accelerator, Buffer, accelerator_count
    from max.engine import InferenceSession

    print(f"accelerator_count={accelerator_count()}", flush=True)
    if accelerator_count() == 0:
        print("No accelerator on this host -- cannot test mixed CPU/GPU placement. Exiting.", flush=True)
        return 0

    rng = np.random.default_rng(4257)
    x_np = rng.uniform(-1.0, 1.0, size=(BATCH, C_IN, SEQ_LEN)).astype(np.float32)
    alpha1_np = rng.uniform(0.2, 1.3, size=(1, C_IN, 1)).astype(np.float32)
    alpha2_np = rng.uniform(0.2, 1.3, size=(1, C_OUT, 1)).astype(np.float32)
    # PyTorch ConvTranspose1d weight layout: [in_channels, out_channels, kernel].
    weight_io_k = rng.normal(0, 0.15, size=(C_IN, C_OUT, KERNEL)).astype(np.float32)
    bias_np = rng.normal(0, 0.05, size=(C_OUT,)).astype(np.float32)
    # RSCF for conv2d_transpose: (1, kernel, out_channels, in_channels).
    filter_rscf = np.transpose(weight_io_k, (2, 1, 0))[np.newaxis, ...].copy()

    ref = numpy_reference(x_np, alpha1_np, weight_io_k, bias_np, alpha2_np, STRIDE, OUTPUT_PADDING)
    print(f"FP64 reference output shape={ref.shape}", flush=True)

    accel = Accelerator()
    cpu = CPU()
    gpu_device = DeviceRef.from_device(accel)
    cpu_device = DeviceRef.CPU()

    print(
        f"Session devices requested=[Accelerator()] "
        f"(host CPU auto-appended per InferenceSession.__init__); "
        f"activation device={gpu_device}, conv_transpose device={cpu_device}",
        flush=True,
    )

    graph = build_mixed_device_graph(gpu_device, cpu_device)
    session = InferenceSession(devices=[accel])
    model = session.load(graph)

    bufs = [
        Buffer.from_numpy(x_np).to(accel),
        Buffer.from_numpy(alpha1_np).to(accel),
        Buffer.from_numpy(filter_rscf).to(cpu),
        Buffer.from_numpy(bias_np).to(cpu),
        Buffer.from_numpy(alpha2_np).to(accel),
    ]
    result = model.execute(*bufs)[0]
    got = result.to(cpu).to_numpy().astype(np.float64)

    print(f"MAX mixed-device output shape={got.shape}", flush=True)
    if got.shape != ref.shape:
        print(f"SHAPE MISMATCH: MAX {got.shape} vs reference {ref.shape}", flush=True)
        return 2

    abs_err = np.abs(got - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-8)
    n_nan_inf = int(np.sum(~np.isfinite(got)))
    print(
        f"max|err|={abs_err.max():.6g} max_rel_err={rel_err.max():.6g} nan/inf={n_nan_inf}",
        flush=True,
    )
    if n_nan_inf > 0 or abs_err.max() > 1e-3:
        print("RESULT: FAIL -- non-finite or excessive error", flush=True)
        return 3

    print("RESULT: PASS -- mixed CPU/GPU single-graph execution produced a finite, correct output", flush=True)
    return 0


def main() -> None:
    if "--run-graph" in sys.argv:
        raise SystemExit(run_graph())

    # Driver: spawn the actual graph-build-and-execute step in its own subprocess so a fatal
    # Metal abort (per m2-convtranspose1d-results.md) cannot kill this process or hide the
    # outcome. Mirrors the isolated per-case subprocess runner m2_convtranspose1d_prototype.py
    # needed.
    proc = subprocess.run(
        [sys.executable, "-u", __file__, "--run-graph"],
        capture_output=True,
        text=True,
    )
    print("--- subprocess stdout ---")
    print(proc.stdout)
    if proc.stderr:
        print("--- subprocess stderr ---")
        print(proc.stderr)
    print(f"--- subprocess exit code: {proc.returncode} ---")

    if proc.returncode == 0:
        print("VERDICT: mixed CPU/GPU placement inside one MAX graph WORKS on this toolchain.")
    elif proc.returncode < 0 or proc.returncode > 128:
        print(
            "VERDICT: subprocess was killed/aborted (fatal process abort) -- mixed-device "
            "single-graph placement does NOT work cleanly on this toolchain. M3-1 disproved; "
            "fall back to the two-stitched-graphs design for Stage C."
        )
    else:
        print(
            f"VERDICT: subprocess exited non-zero ({proc.returncode}) without a fatal abort -- "
            "treat as a correctness failure, inspect stdout/stderr above."
        )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
