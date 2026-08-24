"""M2 prototype #3 for issue #57: does max.nn.ConvTranspose1d / ops.conv2d_transpose actually
execute on GPU, per docs/research/mojo-max/m1-responsibility-map.md S6/S11 (E4) -- the highest-
risk item in the map, since the entire 960x upsample in Higgs's Code2Wav is built from this op,
and upstream's own source carries a `# TODO(GEX-2043): Add support for GPU kernel for
conv_transpose` comment.

Method: build the IDENTICAL graph twice (once with CPU as the device, once with an accelerator),
run both with the same inputs, and compare. This avoids needing an independent hand-rolled
transposed-conv1d reference -- if GPU silently falls back to CPU or is unimplemented, either the
GPU run errors outright, or (if it runs) matches the CPU run bit-for-bit-ish; if GPU actually
executes its own kernel and that kernel is broken, results would diverge from the CPU reference
we trust as correct (MAX's own CPU path, exercised without issue by both earlier prototypes).

Also tests the specific stride/output_padding combinations Higgs's five decoder blocks actually
use: rates=(8,5,4,2,3) -> output_padding = stride % 2 -> (0, 1, 0, 0, 1). The docstring for
conv2d_transpose claims "Only 0 is supported" for output_paddings, which would make strides 5
and 3 (both giving output_padding=1) a real blocker distinct from the GPU-kernel question --
tested explicitly here, not assumed from the docstring text alone.

Usage: pixi run python m2_convtranspose1d_prototype.py
"""

import traceback

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_graph(c_in: int, c_out: int, kernel: int, stride: int, output_padding: int, device: DeviceRef):
    def forward(x, filter_rscf):
        # x: [B, C_in, T] -> NHWC [B, 1, T, C_in]
        x_nhwc = ops.unsqueeze(ops.permute(x, [0, 2, 1]), 1)
        y = ops.conv2d_transpose(
            x_nhwc,
            filter_rscf,
            stride=(1, stride),
            padding=(0, 0, 0, 0),
            output_paddings=(0, output_padding),
        )
        # conv2d_transpose's docstring claims NCHW output, but the ACTUAL runtime shape
        # observed here is [B, 1, T_out, C_out] -- i.e. still NHWC, contradicting the
        # docstring. Squeeze the real H=1 axis (1), then permute channel-last -> channel-first.
        return ops.permute(ops.squeeze(y, 1), [0, 2, 1])

    return Graph(
        f"convT1d_stride{stride}_outpad{output_padding}",
        forward=forward,
        input_types=[
            TensorType(DType.float32, shape=(batch, c_in, seq_len), device=device),
            TensorType(DType.float32, shape=(1, kernel, c_out, c_in), device=device),
        ],
    )


def run_on(device_obj, c_in, c_out, kernel, stride, output_padding, x_np, filter_np):
    device = DeviceRef.from_device(device_obj)
    graph = build_graph(c_in, c_out, kernel, stride, output_padding, device)
    session = InferenceSession(devices=[device_obj])
    model = session.load(graph)
    x_buf = Buffer.from_numpy(x_np).to(device_obj)
    filter_buf = Buffer.from_numpy(filter_np).to(device_obj)
    result = model.execute(x_buf, filter_buf)[0]
    return result.to(CPU()).to_numpy()


if __name__ == "__main__":
    rng = np.random.default_rng(9012)
    batch, seq_len = 1, 16

    # Higgs's five decoder-block (channels, stride) pairs and the output_padding each implies.
    cases = [
        (1024, 512, 8, 0),  # block 0: stride 8, output_padding = 8 % 2 = 0
        (512, 256, 5, 1),  # block 1: stride 5, output_padding = 5 % 2 = 1  <- the risky one
        (256, 128, 4, 0),  # block 2
        (128, 64, 2, 0),  # block 3
        (64, 32, 3, 1),  # block 4: stride 3, output_padding = 3 % 2 = 1  <- the risky one
    ]
    # Shrink channel counts for a fast prototype -- structure (stride, output_padding), not
    # raw channel count, is what's under test.
    cases = [(min(c_in, 16), min(c_out, 16), s, op) for c_in, c_out, s, op in cases]

    print(f"accelerator_count={accelerator_count()}")
    if accelerator_count() == 0:
        print("No accelerator on this host -- cannot test the GPU-vs-CPU question. Exiting.")
        raise SystemExit(0)

    for c_in, c_out, stride, output_padding in cases:
        kernel = 2 * stride
        x_np = rng.uniform(-1.0, 1.0, size=(batch, c_in, seq_len)).astype(np.float32)
        # PyTorch ConvTranspose1d weight layout: [in_channels, out_channels, kernel].
        # RSCF for conv2d_transpose per its docstring: (height, width, out_channels, in_channels)
        # -> [1, kernel, out_channels, in_channels].
        weight_pt = rng.normal(0, 0.1, size=(c_in, c_out, kernel)).astype(np.float32)
        filter_rscf = np.transpose(weight_pt, (2, 1, 0))[np.newaxis, ...].copy()

        label = f"stride={stride} output_padding={output_padding} (c_in={c_in},c_out={c_out},k={kernel})"
        print(f"\n=== {label} ===")

        try:
            cpu_out = run_on(CPU(), c_in, c_out, kernel, stride, output_padding, x_np, filter_rscf)
            print(f"CPU: shape={cpu_out.shape} ok")
        except Exception as e:  # noqa: BLE001
            print(f"CPU: FAILED -- {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        try:
            gpu_out = run_on(Accelerator(), c_in, c_out, kernel, stride, output_padding, x_np, filter_rscf)
            print(f"GPU: shape={gpu_out.shape} ok")
        except Exception as e:  # noqa: BLE001
            print(f"GPU: FAILED -- {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        if cpu_out.shape != gpu_out.shape:
            print(f"SHAPE MISMATCH: CPU {cpu_out.shape} vs GPU {gpu_out.shape}")
        else:
            abs_err = np.abs(cpu_out.astype(np.float64) - gpu_out.astype(np.float64))
            print(
                f"CPU vs GPU: max|err|={abs_err.max():.6g} "
                f"nan_in_gpu={int(np.sum(~np.isfinite(gpu_out)))}"
            )
