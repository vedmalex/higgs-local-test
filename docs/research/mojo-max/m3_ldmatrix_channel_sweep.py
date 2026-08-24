"""Diagnostic for issue #57's M3-10 T4 finding: the full `_BosonDecoderBlock` graph aborts
fatally on Colab Tesla T4 with `LLVM ERROR: Cannot select: intrinsic
%llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16`, while M2's isolated Conv1d/residual-unit
prototypes (32 channels) passed cleanly on the same T4. Leading hypothesis (see the issue #57
comment recording this finding, and https://forum.modular.com/t/having-issues-with-max-matmul-on-
default-google-colab-gpu-t4/1658): MAX's `ops.conv2d` kernel-selection heuristic switches to a
tensor-core/GEMM-based kernel above some channel-count threshold, and that specific kernel path
has a Turing (sm_75) codegen gap that the forum thread confirms exists for MAX's matmul path too.

This script reuses `m2_conv1d_prototype.py`'s exact graph-construction logic UNCHANGED except for
c_in/c_out, sweeping channel counts to find the threshold where the abort first appears.

Each channel count is run as an ISOLATED SUBPROCESS -- per m2_convtranspose1d_prototype.py's and
m3_device_mixing_spike.py's own established pattern -- because the failure mode observed on T4 is
a fatal, uncatchable process abort (SIGABRT / LLVM ERROR), not a Python exception. Running
multiple channel counts in one process would let the first abort hide every later result.

Usage (single case, called by the sweep driver below via subprocess):
    pixi run python m3_ldmatrix_channel_sweep.py --case <channels>

Usage (the actual sweep, run this):
    pixi run python m3_ldmatrix_channel_sweep.py
"""

import argparse
import subprocess
import sys

import numpy as np

# The exact channel counts to test. 32 is M2's already-passing baseline; 512/256 are the real
# Higgs stride-5 block's sizes (the ones that abort in M3-5); the values in between narrow down
# the threshold. 1024 covers the stride-8 block's larger side too, for completeness.
CHANNEL_COUNTS = [32, 64, 128, 256, 512, 1024]


def run_one_case(channels: int) -> None:
    """Build and execute the exact conv1d-via-conv2d graph from m2_conv1d_prototype.py, with
    c_in=c_out=channels (matching that prototype's own square-channel convention), on whatever
    GPU accelerator is present. Prints a machine-parseable result line. Meant to be invoked as
    its own subprocess -- if this aborts, the parent sees a non-zero/signal exit code and moves
    on to the next channel count."""
    from max.driver import CPU, Accelerator, Buffer, accelerator_count
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef

    # Import the actual M2 prototype's own graph-building function, unmodified -- this is the
    # exact code path that already passed at 32 channels on this same T4 in M2, so any behavior
    # difference at higher channel counts is isolated to the channel-count variable alone, not to
    # any reimplementation drift.
    import m2_conv1d_prototype as m2c
    from m2_conv1d_prototype import build_conv1d_via_conv2d_graph, fold_weight_norm, numpy_conv1d_reference

    rng = np.random.default_rng(5678)
    batch, seq_len = 1, 64
    kernel, dilation = 7, 3
    padding = (kernel - 1) * dilation // 2
    c_in = c_out = channels
    # build_conv1d_via_conv2d_graph's forward() closure reads `batch`/`seq_len` as module-level
    # globals of m2_conv1d_prototype (set inside that file's own __main__ block when run
    # standalone) -- set them here since we're calling the function from a different module.
    m2c.batch = batch
    m2c.seq_len = seq_len

    x_np = rng.uniform(-2.0, 2.0, size=(batch, c_in, seq_len)).astype(np.float32)
    g_np = rng.uniform(0.5, 2.0, size=(c_out, 1, 1)).astype(np.float32)
    v_np = rng.normal(0, 0.3, size=(c_out, c_in, kernel)).astype(np.float32)
    bias_np = rng.normal(0, 0.1, size=(c_out,)).astype(np.float32)
    weight_np = fold_weight_norm(g_np, v_np)

    ref = numpy_conv1d_reference(x_np, weight_np, bias_np, dilation, padding)

    accel_count = accelerator_count()
    device_obj = CPU() if accel_count == 0 else Accelerator()
    device = DeviceRef.from_device(device_obj)
    print(f"channels={channels} device={device_obj} accelerator_count={accel_count}")

    graph = build_conv1d_via_conv2d_graph(c_in, c_out, kernel, dilation, padding, device, DType.float32)
    session = InferenceSession(devices=[device_obj])
    model = session.load(graph)

    filter_rscf = np.transpose(weight_np, (2, 1, 0))[np.newaxis, ...].copy()
    x_buf = Buffer.from_numpy(x_np).to(device_obj)
    filter_buf = Buffer.from_numpy(filter_rscf).to(device_obj)
    bias_buf = Buffer.from_numpy(bias_np).to(device_obj)

    result = model.execute(x_buf, filter_buf, bias_buf)[0]
    got = result.to(CPU()).to_numpy().astype(np.float64)

    if got.shape != ref.shape:
        print(f"channels={channels} SHAPE MISMATCH: MAX {got.shape} vs reference {ref.shape}")
        return

    abs_err = np.abs(got - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-8)
    print(
        f"channels={channels} RESULT: max|err|={abs_err.max():.6g} "
        f"max_rel_err={rel_err.max():.6g} nan/inf={int(np.sum(~np.isfinite(got)))} -- PASSED"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=None, help="internal: run a single channel count")
    args = parser.parse_args()

    if args.case is not None:
        run_one_case(args.case)
        return

    # Sweep driver: one isolated subprocess per channel count.
    print(f"Sweeping channel counts {CHANNEL_COUNTS} to find the ldmatrix/T4 abort threshold.")
    print("Each case runs as its own subprocess -- a fatal abort in one must not hide the rest.\n")
    results = []
    for channels in CHANNEL_COUNTS:
        print(f"=== case channels={channels} ===")
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--case", str(channels)],
            capture_output=True,
            text=True,
        )
        print("--- stdout ---")
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print("--- stderr ---")
            print(proc.stderr.strip())
        if proc.returncode < 0:
            verdict = f"ABORTED (fatal process abort, signal {-proc.returncode})"
        elif proc.returncode != 0:
            verdict = f"FAILED (exit code {proc.returncode})"
        elif "PASSED" in proc.stdout:
            verdict = "PASSED"
        else:
            verdict = "UNKNOWN (no PASSED marker, but exited 0 -- inspect stdout above)"
        print(f"--- verdict: {verdict} ---\n")
        results.append((channels, verdict))

    print("=== Summary ===")
    for channels, verdict in results:
        print(f"channels={channels}: {verdict}")


if __name__ == "__main__":
    main()
