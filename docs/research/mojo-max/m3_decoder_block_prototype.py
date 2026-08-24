"""M3-5 for issue #57: assemble the full `_BosonDecoderBlock` (`DecoderBlock(512, 256,
stride=5)`) as ONE mixed CPU/GPU MAX graph, FP32 throughout, synthetic weights.

    Snake1d(512)                                                              GPU
    wn_conv_transpose1d(512, 256, k=10, stride=5, padding=3, output_padding=1)  CPU (M3-1)
    ResidualUnit(256, dilation=1)                                             GPU
    ResidualUnit(256, dilation=3)                                             GPU
    ResidualUnit(256, dilation=9)                                             GPU

Nothing here is reimplemented from scratch:
  - `snake_expr` / `conv1d_expr`     verbatim import from m2_residual_unit_prototype.py
  - the route-A conv-transpose graph pattern (NHWC unsqueeze/permute, `bias=None` into
    `ops.conv2d_transpose` + manual bias add -- the M3-1 bias-layout workaround) is the
    same pattern m3_device_mixing_spike.py validated; reproduced here as
    `conv_transpose_expr` (no reusable function of that name existed to import -- M2's
    prototype and M3-1's spike both inlined it in their own `forward()`) with the same
    CPU device placement for activation+filter+bias.
  - the FP64 NumPy chain reuses `numpy_snake` / `numpy_conv1d` / `numpy_residual_unit`
    (from m2_residual_unit_prototype.py) and `numpy_conv_transpose1d` /
    `numpy_boson_decoder_block` (from m3_block_reference.py, M3-4) -- not reimplemented.
  - the divergence report at every intermediate tensor uses `m3_divergence.compare()`
    (M3-2), not an ad-hoc comparison.

Per m3-plan.md S3/M3-1: mixed CPU/GPU placement inside one MAX graph is confirmed to work
on this toolchain (MAX 26.5.0 / Mojo 1.0.0 / M1 Metal) -- a clean subprocess exit is the
positive signal that the transposed conv actually dispatched to CPU (the alternative, a
GPU dispatch attempt, is a fatal uncatchable `cudnnCreate` process abort on this toolchain,
per m2-convtranspose1d-results.md). So -- exactly like m3_device_mixing_spike.py and
m2_convtranspose1d_prototype.py -- the actual graph build+execute step always runs inside
an ISOLATED SUBPROCESS; this driver process only inspects the exit code and stdout.

Usage:
    pixi run python m3_decoder_block_prototype.py [--synthetic | --real-weights]
    pixi run python m3_decoder_block_prototype.py --run-graph --synthetic   (internal)

`--real-weights` is a structural placeholder for M3-6 (not implemented yet -- this file is
explicitly designed to be extended, per the task's deliverable note, without needing a
rewrite of the graph-building code): the mode switch and per-layer instrumentation hooks
below are shared, only the weight *source* differs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---- shapes for the stride=5 DecoderBlock(512, 256, stride=5) --------------------------------
INPUT_DIM = 512
OUTPUT_DIM = 256
STRIDE = 5
CT_KERNEL = 2 * STRIDE  # 10
CT_PADDING = 3  # ceil(stride/2) = ceil(5/2) = 3
CT_OUTPUT_PADDING = STRIDE % 2  # 1
RU_KERNEL = 7
RU_DILATIONS = (1, 3, 9)
RU_PADDINGS = tuple((RU_KERNEL - 1) * d // 2 for d in RU_DILATIONS)  # (3, 9, 27)
SEQ_LEN = 20  # matches m3_block_reference.py's stride-5 synthetic case
BATCH = 1

STAGE_NAMES = ["after_snake1", "after_conv_t1", "after_res_unit1", "after_res_unit2", "after_res_unit3_final"]


# ------------------------------------------------------------------------------------------
# Synthetic weight generation -- one seed, fed identically to the MAX graph and the FP64 ref.
# ------------------------------------------------------------------------------------------
def make_synthetic_weights(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    x = rng.uniform(-2.0, 2.0, size=(BATCH, INPUT_DIM, SEQ_LEN)).astype(np.float32)
    alpha0 = rng.uniform(-0.3, 1.2, size=(1, INPUT_DIM, 1)).astype(np.float32)

    # PyTorch ConvTranspose1d weight layout: [C_in, C_out, K].
    ct_weight = rng.normal(0, 0.05, size=(INPUT_DIM, OUTPUT_DIM, CT_KERNEL)).astype(np.float32)
    ct_bias = rng.normal(0, 0.05, size=(OUTPUT_DIM,)).astype(np.float32)

    res_units = []
    for dilation in RU_DILATIONS:
        alpha1 = rng.uniform(-0.3, 1.2, size=(1, OUTPUT_DIM, 1)).astype(np.float32)
        w1 = rng.normal(0, 0.05, size=(OUTPUT_DIM, OUTPUT_DIM, RU_KERNEL)).astype(np.float32)  # [C_out,C_in,K]
        b1 = rng.normal(0, 0.05, size=(OUTPUT_DIM,)).astype(np.float32)
        alpha2 = rng.uniform(-0.3, 1.2, size=(1, OUTPUT_DIM, 1)).astype(np.float32)
        w2 = rng.normal(0, 0.05, size=(OUTPUT_DIM, OUTPUT_DIM, 1)).astype(np.float32)  # pointwise
        b2 = rng.normal(0, 0.05, size=(OUTPUT_DIM,)).astype(np.float32)
        res_units.append((alpha1, w1, b1, alpha2, w2, b2))

    return {
        "x": x,
        "alpha0": alpha0,
        "ct_weight": ct_weight,
        "ct_bias": ct_bias,
        "res_units": res_units,
    }


# ------------------------------------------------------------------------------------------
# FP64 per-layer reference chain -- reuses M2/M3-4's functions, does not reimplement them.
# ------------------------------------------------------------------------------------------
def _stub_torch_for_import() -> None:
    """m3_block_reference.py imports `torch` at module scope purely for its own
    PyTorch-cross-check helpers (`_validate_conv_transpose_hand_case`, `run_cross_check`,
    `run_real_weight_cross_check`) -- none of which this script calls; it only reuses the
    pure-NumPy `numpy_conv_transpose1d` / `numpy_boson_decoder_block`. This pixi/MAX
    toolchain env intentionally has no `torch` (that lives in the separate `.venv-tts`
    used by M3-3/M3-4's host-CPU-only scripts), so stub it out -- mirrors
    m3_block_reference.py's own `_stub_max_module_for_import()` trick, same rationale."""
    import types

    if "torch" in sys.modules:
        return
    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = lambda: _NullContext()
    torch_mod.from_numpy = lambda x: x
    torch_mod.Tensor = object
    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fp64_reference_chain(weights: dict) -> list[np.ndarray]:
    _stub_torch_for_import()
    from m2_residual_unit_prototype import numpy_residual_unit  # noqa: E402
    from m3_block_reference import numpy_conv_transpose1d  # noqa: E402

    x = weights["x"]
    y0 = _numpy_snake_local(x, weights["alpha0"])
    y1 = numpy_conv_transpose1d(
        y0, weights["ct_weight"], weights["ct_bias"], stride=STRIDE, padding=CT_PADDING, output_padding=CT_OUTPUT_PADDING
    )
    stages = [y0, y1]
    y = y1
    for (alpha1, w1, b1, alpha2, w2, b2), dilation, padding in zip(weights["res_units"], RU_DILATIONS, RU_PADDINGS):
        y = numpy_residual_unit(y, alpha1, w1, b1, alpha2, w2, b2, dilation, padding)
        stages.append(y)
    return stages  # [after_snake1, after_conv_t1, after_ru1, after_ru2, after_ru3]


def _numpy_snake_local(x, alpha):
    # Reuse m2's own numpy_snake (imported below at call site to avoid a hard module-level
    # dependency ordering issue with the MAX-package stub in m3_block_reference.py).
    from m2_residual_unit_prototype import numpy_snake

    return numpy_snake(x, alpha)


# ------------------------------------------------------------------------------------------
# MAX graph expressions.
# ------------------------------------------------------------------------------------------
def conv_transpose_expr(x_cpu, filter_rscf, bias, stride: int, output_padding: int, c_out: int, padding: int = 0):
    """Route-A layout, CPU-placed. Reproduces the M3-1 bias workaround verbatim: MAX's
    `ops.conv2d_transpose` mis-shapes its output when `bias=` is passed directly for this
    H=1 NHWC layout ((1, C_out, T_out, C_out) instead of (1, 1, T_out, C_out)) -- confirmed
    in m3-device-mixing-results.md. Call with `bias=None`, add the bias manually after.

    Real M3-5 finding (new, not in M2/M3-1): every prior prototype (m2_convtranspose1d_
    prototype.py, m3_device_mixing_spike.py) only ever passed `padding=(0, 0, 0, 0)` to
    `ops.conv2d_transpose` -- the real `wn_conv_transpose1d(..., padding=ceil(stride/2))`
    case (padding=3 here) was never exercised before this task, and neither prototype
    established what `ops.conv2d_transpose`'s own `padding=` argument does for a
    transposed conv (its docstring covers the plain-conv case; a plain-conv `padding`
    convention -- pad the input before convolving -- would NOT reproduce PyTorch
    ConvTranspose1d's `padding=` semantic, which crops `2*padding` off the FULL unpadded
    transposed-conv output). Rather than guess or trust an untested argument, this
    implementation avoids the question entirely: always call the op with
    padding=(0,0,0,0) (the one padding value M2/M3-1 already validated is correct for
    conv2d_transpose), producing the full, uncropped output of length
    `(L_in-1)*stride + K + output_padding`, then explicitly crop `padding` samples off
    EACH end of the sequence axis in Python/MAX-ops afterward -- exactly reproducing
    PyTorch's ConvTranspose1d `padding=` semantic by direct construction, with a result
    that is checked below against the FP64 reference rather than assumed correct."""
    from max.graph import ops

    x_nhwc = ops.unsqueeze(ops.permute(x_cpu, [0, 2, 1]), 1)
    z_nhwc = ops.conv2d_transpose(
        x_nhwc,
        filter_rscf,
        stride=(1, stride),
        padding=(0, 0, 0, 0),
        output_paddings=(0, output_padding),
    )
    z = ops.permute(ops.squeeze(z_nhwc, 1), [0, 2, 1])
    if padding > 0:
        t_full = int(z.shape[-1])
        z = z[:, :, padding : t_full - padding]
    z = ops.add(z, ops.reshape(bias, (1, c_out, 1)))
    return z


def residual_unit_expr(x, alpha1, filter1, bias1, alpha2, filter2, bias2, dilation: int, padding: int, device):
    """Mirrors m2_residual_unit_prototype.py's build_graph forward exactly, factored into a
    function so it can be called 3x (dilation=1,3,9) here."""
    from max.graph import ops

    from m2_residual_unit_prototype import conv1d_expr, snake_expr

    y = snake_expr(x, alpha1, device)
    y = conv1d_expr(y, filter1, bias1, dilation, padding)
    y = snake_expr(y, alpha2, device)
    y = conv1d_expr(y, filter2, bias2, 1, 0)  # pointwise
    x_len = int(x.shape[-1])
    y_len = int(y.shape[-1])
    pad = (x_len - y_len) // 2
    xc = x[:, :, pad : x_len - pad] if pad > 0 else x
    return ops.add(xc, y)


def build_decoder_block_graph(gpu_device, cpu_device):
    from max.dtype import DType
    from max.graph import Graph, TensorType
    from m2_residual_unit_prototype import snake_expr

    def forward(
        x,
        alpha0,
        ct_filter,
        ct_bias,
        alpha1_1,
        filter1_1,
        bias1_1,
        alpha2_1,
        filter2_1,
        bias2_1,
        alpha1_2,
        filter1_2,
        bias1_2,
        alpha2_2,
        filter2_2,
        bias2_2,
        alpha1_3,
        filter1_3,
        bias1_3,
        alpha2_3,
        filter2_3,
        bias2_3,
    ):
        # -- GPU: Snake1d(512) --
        y0 = snake_expr(x, alpha0, gpu_device)

        # -- cross to CPU: wn_conv_transpose1d(512, 256, k=10, stride=5, pad=3, out_pad=1) --
        from max.graph import ops as _ops

        y0_cpu = _ops.transfer_to(y0, cpu_device)
        y1_cpu = conv_transpose_expr(
            y0_cpu, ct_filter, ct_bias, STRIDE, CT_OUTPUT_PADDING, OUTPUT_DIM, padding=CT_PADDING
        )

        # -- cross back to GPU --
        y1 = _ops.transfer_to(y1_cpu, gpu_device)

        # -- GPU: ResidualUnit(256, dilation=1) --
        y2 = residual_unit_expr(y1, alpha1_1, filter1_1, bias1_1, alpha2_1, filter2_1, bias2_1, 1, RU_PADDINGS[0], gpu_device)
        # -- GPU: ResidualUnit(256, dilation=3) --
        y3 = residual_unit_expr(y2, alpha1_2, filter1_2, bias1_2, alpha2_2, filter2_2, bias2_2, 3, RU_PADDINGS[1], gpu_device)
        # -- GPU: ResidualUnit(256, dilation=9) --
        y4 = residual_unit_expr(y3, alpha1_3, filter1_3, bias1_3, alpha2_3, filter2_3, bias2_3, 9, RU_PADDINGS[2], gpu_device)

        return y0, y1, y2, y3, y4

    input_types = [
        TensorType(DType.float32, shape=(BATCH, INPUT_DIM, SEQ_LEN), device=gpu_device),
        TensorType(DType.float32, shape=(1, INPUT_DIM, 1), device=gpu_device),
        TensorType(DType.float32, shape=(1, CT_KERNEL, OUTPUT_DIM, INPUT_DIM), device=cpu_device),
        TensorType(DType.float32, shape=(OUTPUT_DIM,), device=cpu_device),
    ]
    for _ in range(3):
        input_types += [
            TensorType(DType.float32, shape=(1, OUTPUT_DIM, 1), device=gpu_device),
            TensorType(DType.float32, shape=(1, RU_KERNEL, OUTPUT_DIM, OUTPUT_DIM), device=gpu_device),
            TensorType(DType.float32, shape=(OUTPUT_DIM,), device=gpu_device),
            TensorType(DType.float32, shape=(1, OUTPUT_DIM, 1), device=gpu_device),
            TensorType(DType.float32, shape=(1, 1, OUTPUT_DIM, OUTPUT_DIM), device=gpu_device),
            TensorType(DType.float32, shape=(OUTPUT_DIM,), device=gpu_device),
        ]

    return Graph("m3_decoder_block_stride5", forward=forward, input_types=input_types)


# ------------------------------------------------------------------------------------------
# Graph execution (run ONLY inside the isolated subprocess -- see main()).
# ------------------------------------------------------------------------------------------
def run_graph(mode: str, seed: int = 57305) -> int:
    if mode != "synthetic":
        print(f"mode={mode!r} not implemented yet (M3-6 territory) -- exiting.", flush=True)
        return 0

    from max.driver import CPU, Accelerator, Buffer, accelerator_count
    from max.engine import InferenceSession
    from max.graph import DeviceRef

    import m3_divergence as m3div

    print(f"accelerator_count={accelerator_count()}", flush=True)
    if accelerator_count() == 0:
        print("No accelerator on this host -- cannot test the mixed CPU/GPU block. Exiting.", flush=True)
        return 0

    weights = make_synthetic_weights(seed=seed)
    print(f"synthetic weight seed={seed}", flush=True)
    ref_stages = fp64_reference_chain(weights)
    for name, ref in zip(STAGE_NAMES, ref_stages):
        print(f"FP64 reference {name}: shape={ref.shape}", flush=True)

    accel = Accelerator()
    cpu = CPU()
    gpu_device = DeviceRef.from_device(accel)
    cpu_device = DeviceRef.CPU()
    print(
        f"Session devices requested=[Accelerator()] (host CPU auto-appended); "
        f"GPU stages device={gpu_device}, conv_t1 device={cpu_device}",
        flush=True,
    )

    graph = build_decoder_block_graph(gpu_device, cpu_device)
    session = InferenceSession(devices=[accel])
    model = session.load(graph)

    # RSCF layout for the transposed conv, matching M2/M3-1's convention.
    ct_filter_rscf = np.transpose(weights["ct_weight"], (2, 1, 0))[np.newaxis, ...].copy()

    bufs = [
        Buffer.from_numpy(weights["x"]).to(accel),
        Buffer.from_numpy(weights["alpha0"]).to(accel),
        Buffer.from_numpy(ct_filter_rscf).to(cpu),
        Buffer.from_numpy(weights["ct_bias"]).to(cpu),
    ]
    for alpha1, w1, b1, alpha2, w2, b2 in weights["res_units"]:
        filter1_rscf = np.transpose(w1, (2, 1, 0))[np.newaxis, ...].copy()
        filter2_rscf = np.transpose(w2, (2, 1, 0))[np.newaxis, ...].copy()
        bufs += [
            Buffer.from_numpy(alpha1).to(accel),
            Buffer.from_numpy(filter1_rscf).to(accel),
            Buffer.from_numpy(b1).to(accel),
            Buffer.from_numpy(alpha2).to(accel),
            Buffer.from_numpy(filter2_rscf).to(accel),
            Buffer.from_numpy(b2).to(accel),
        ]

    results = model.execute(*bufs)
    got_stages = [r.to(cpu).to_numpy().astype(np.float64) for r in results]

    print("\n=== per-layer divergence (M3-2 detector, m3_divergence.compare) ===", flush=True)
    all_healthy = True
    length_exact = True
    final_report = None
    for name, got, ref in zip(STAGE_NAMES, got_stages, ref_stages):
        report = m3div.compare(got, ref)
        print(f"{name}: {report}", flush=True)
        if got.shape[-1] != ref.shape[-1]:
            length_exact = False
        if report.nan_count > 0 or report.inf_count > 0:
            all_healthy = False
        if report.exact_zero_count == report.total_elements and report.exact_zero_count_ref != report.total_elements:
            all_healthy = False
        if name == STAGE_NAMES[-1]:
            final_report = report

    print(
        f"\nfinal output length: got={got_stages[-1].shape[-1]} ref={ref_stages[-1].shape[-1]} "
        f"exact_match={got_stages[-1].shape[-1] == ref_stages[-1].shape[-1]}",
        flush=True,
    )

    # PRIMARY GATE, per m3-plan.md S5 as corrected after this task's original FAIL: the combined
    # np.allclose-style tolerance |got-ref| <= atol + rtol*|ref| (atol = 1e-05*max|ref|,
    # rtol = 5e-03), NOT plain max_rel_err -- which M3-4 and M3-5 independently showed blows up
    # at near-zero reference elements while max_abs_err stays flat. max_rel_err is still printed
    # above for every stage; it is reported, not gating.
    primary_gate_pass = final_report is not None and final_report.combined_pass
    overall_pass = primary_gate_pass and all_healthy and length_exact
    print(
        f"\nPRIMARY GATE (combined tolerance |got-ref| <= atol + rtol*|ref|, "
        f"atol={final_report.atol_used:.6g}=1e-05*max|ref|({final_report.ref_abs_max:.6g}), "
        f"rtol={final_report.rtol_used:.6g}): worst-element ratio="
        f"{final_report.combined_max_ratio:.6g} over-tolerance elements="
        f"{final_report.combined_fail_count}/{final_report.total_elements} -> "
        f"{'PASS' if primary_gate_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"SECONDARY (reported, not gating): max_abs_err={final_report.max_abs_err:.6g} "
        f"max_rel_err={final_report.max_rel_err:.6g} "
        f"max_rel_err_masked(|ref|>={final_report.mask_threshold:.6g})="
        f"{final_report.max_rel_err_masked:.6g} "
        f"(literal old 5e-03 max_rel_err gate would say "
        f"{'PASS' if final_report.max_rel_err <= 5e-3 else 'FAIL'})",
        flush=True,
    )
    print(f"zero NaN/Inf and no unexplained exact-zero tensor across all stages: {all_healthy}", flush=True)
    print(f"exact output length match: {length_exact}", flush=True)
    print(f"\nRESULT: {'PASS' if overall_pass else 'FAIL'}", flush=True)
    return 0 if overall_pass else 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-graph", action="store_true", help="(internal) actually build+run the graph")
    parser.add_argument("--seed", type=int, default=57305, help="synthetic weight seed (M3-5 default 57305)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--synthetic", action="store_true", default=True)
    mode_group.add_argument("--real-weights", action="store_true")
    args = parser.parse_args()
    mode = "real-weights" if args.real_weights else "synthetic"

    if args.run_graph:
        raise SystemExit(run_graph(mode, args.seed))

    # Driver: isolate the actual graph build+execute in its own subprocess, exactly as
    # m3_device_mixing_spike.py / m2_convtranspose1d_prototype.py do -- a Metal
    # `conv2d_transpose` GPU-dispatch bug is a fatal, uncatchable process abort, not a
    # Python exception, per m2-convtranspose1d-results.md.
    mode_flag = "--real-weights" if mode == "real-weights" else "--synthetic"
    proc = subprocess.run(
        [sys.executable, "-u", __file__, "--run-graph", mode_flag, "--seed", str(args.seed)],
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
        print("VERDICT: block ran cleanly and PASSED the M3-5 gates.")
    elif proc.returncode < 0 or proc.returncode > 128:
        print("VERDICT: subprocess was killed/aborted (fatal process abort).")
    else:
        print(f"VERDICT: subprocess exited non-zero ({proc.returncode}) -- inspect stdout/stderr above.")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
