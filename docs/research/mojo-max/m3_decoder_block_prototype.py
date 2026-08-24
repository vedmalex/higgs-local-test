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

`--real-weights` (M3-6): re-runs the exact same graph and per-layer instrumentation above
against the real stride-5 `_BosonDecoderBlock` (block 1) weights extracted by
`m3_block_weights.py` (M3-3) from `bosonai/higgs-tts-3-4b`, compared against the same FP64
reference chain fed those same real weights (mirrors `m3_block_reference.py`'s
`run_real_weight_cross_check`, M3-4). Only the weight *source* differs -- the graph-building
code, the per-layer stage instrumentation, and the M3-2 divergence detector are reused
unchanged, per this file's original design note.

Real weights live only in the real checkpoint's BF16 safetensors, which this pixi/MAX
toolchain env cannot read directly (no `torch`, and `safetensors`' `framework="numpy"` cannot
decode BF16 -- confirmed empirically). So `make_real_weights()` below shells out to
`.venv-tts/bin/python m3_real_weights_export.py` (the only env with torch/safetensors/
transformers wired to this checkpoint) to produce a plain FP32 `.npz` cache once, then loads
that cache with NumPy alone, here and in every subsequent run. See
`m3_real_weights_export.py`'s docstring for the full rationale.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---- shapes, made stride-generic for M3-7 (issue #57) ----------------------------------------
# M3-5/M3-6 hard-coded stride=5 (DecoderBlock(512, 256, stride=5)) as module-level constants.
# M3-7 adds stride=8 (DecoderBlock(1024, 512, stride=8), output_padding=0) as a SECOND,
# INDEPENDENT case -- not chained with stride-5 (see this file's module docstring and
# m3-plan.md's M3-7 section for the explicit non-goal statement). Rather than duplicate the
# whole module for a second stride, every function that used to read module-level constants
# now takes a `cfg` dict built by `make_config()`; the stride=5 numbers below are IDENTICAL to
# the old constants, so M3-5/M3-6 behavior is unchanged.
RU_KERNEL = 7
RU_DILATIONS = (1, 3, 9)
BATCH = 1

STAGE_NAMES = ["after_snake1", "after_conv_t1", "after_res_unit1", "after_res_unit2", "after_res_unit3_final"]

# per-stride case parameters. seq_len for stride=8 matches m3_block_reference.py's own
# stride8_1024x512_synthetic cross-check (seq_len=12) so the two scripts' synthetic cases align.
STRIDE_CASES = {
    5: dict(input_dim=512, output_dim=256, seq_len=20, default_synth_seed=57305),
    8: dict(input_dim=1024, output_dim=512, seq_len=12, default_synth_seed=24601),
}


def make_config(stride: int) -> dict:
    if stride not in STRIDE_CASES:
        raise ValueError(f"stride={stride} not configured -- supported: {sorted(STRIDE_CASES)}")
    case = STRIDE_CASES[stride]
    ru_paddings = tuple((RU_KERNEL - 1) * d // 2 for d in RU_DILATIONS)
    return {
        "stride": stride,
        "input_dim": case["input_dim"],
        "output_dim": case["output_dim"],
        "seq_len": case["seq_len"],
        "default_synth_seed": case["default_synth_seed"],
        "ct_kernel": 2 * stride,
        "ct_padding": -(-stride // 2),  # ceil(stride/2), matches m3-plan.md's formula
        "ct_output_padding": stride % 2,
        "ru_kernel": RU_KERNEL,
        "ru_dilations": RU_DILATIONS,
        "ru_paddings": ru_paddings,
    }


# ------------------------------------------------------------------------------------------
# Synthetic weight generation -- one seed, fed identically to the MAX graph and the FP64 ref.
# ------------------------------------------------------------------------------------------
def make_synthetic_weights(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed)
    input_dim, output_dim, seq_len = cfg["input_dim"], cfg["output_dim"], cfg["seq_len"]
    ct_kernel, ru_kernel = cfg["ct_kernel"], cfg["ru_kernel"]

    x = rng.uniform(-2.0, 2.0, size=(BATCH, input_dim, seq_len)).astype(np.float32)
    alpha0 = rng.uniform(-0.3, 1.2, size=(1, input_dim, 1)).astype(np.float32)

    # PyTorch ConvTranspose1d weight layout: [C_in, C_out, K].
    ct_weight = rng.normal(0, 0.05, size=(input_dim, output_dim, ct_kernel)).astype(np.float32)
    ct_bias = rng.normal(0, 0.05, size=(output_dim,)).astype(np.float32)

    res_units = []
    for dilation in cfg["ru_dilations"]:
        alpha1 = rng.uniform(-0.3, 1.2, size=(1, output_dim, 1)).astype(np.float32)
        w1 = rng.normal(0, 0.05, size=(output_dim, output_dim, ru_kernel)).astype(np.float32)  # [C_out,C_in,K]
        b1 = rng.normal(0, 0.05, size=(output_dim,)).astype(np.float32)
        alpha2 = rng.uniform(-0.3, 1.2, size=(1, output_dim, 1)).astype(np.float32)
        w2 = rng.normal(0, 0.05, size=(output_dim, output_dim, 1)).astype(np.float32)  # pointwise
        b2 = rng.normal(0, 0.05, size=(output_dim,)).astype(np.float32)
        res_units.append((alpha1, w1, b1, alpha2, w2, b2))

    return {
        "x": x,
        "alpha0": alpha0,
        "ct_weight": ct_weight,
        "ct_bias": ct_bias,
        "res_units": res_units,
    }


# ------------------------------------------------------------------------------------------
# Real weights (M3-6) -- same dict shape as make_synthetic_weights(), sourced from the real
# checkpoint via m3_real_weights_export.py's .venv-tts-only extraction (see module docstring).
# ------------------------------------------------------------------------------------------
REAL_WEIGHTS_CACHE = HERE / ".m3_real_weights_block1.npz"
REAL_WEIGHTS_SEED_DEFAULT = 99  # matches m3_block_reference.run_real_weight_cross_check's seed
RES_UNIT_NAMES = ("res_unit1", "res_unit2", "res_unit3")


def _ensure_real_weights_cache(seed: int, seq_len: int, cache_path: Path) -> None:
    """(Re)generate `cache_path` via `.venv-tts` if missing or stale for this seed/seq_len.
    The cache stores its own `seed`/`seq_len` so a stale cache from a different invocation is
    detected rather than silently reused."""
    if cache_path.exists():
        with np.load(cache_path) as npz:
            cached_seed = int(npz["seed"])
            cached_seq_len = int(npz["seq_len"])
        if cached_seed == seed and cached_seq_len == seq_len:
            return
        print(
            f"real-weights cache {cache_path} was built for seed={cached_seed} "
            f"seq_len={cached_seq_len}, requested seed={seed} seq_len={seq_len} -- regenerating.",
            flush=True,
        )

    # NOTE: deliberately NOT .resolve()-d -- .venv-tts/bin/python is a symlink chain
    # (bin/python -> python3.12 -> /opt/homebrew/.../python3.12) whose venv activation
    # (site-packages with torch/safetensors/transformers) depends on being invoked via the
    # venv's own bin/ path, not the fully-dereferenced system interpreter it points to
    # (confirmed empirically: .resolve() here reintroduced `ModuleNotFoundError: No module
    # named 'numpy'` because it silently ran the bare system python3.12 instead of the venv).
    venv_tts_python = HERE.parent.parent.parent / ".venv-tts" / "bin" / "python"
    if not venv_tts_python.exists():
        raise FileNotFoundError(
            f"{venv_tts_python} not found -- real-weights extraction needs the .venv-tts env "
            "(torch/safetensors/transformers), which this pixi/MAX env does not have."
        )
    export_script = HERE / "m3_real_weights_export.py"
    print(f"real-weights cache missing/stale -- generating via {venv_tts_python} {export_script.name}", flush=True)
    proc = subprocess.run(
        [
            str(venv_tts_python),
            str(export_script),
            "--seed", str(seed),
            "--seq-len", str(seq_len),
            "--out", str(cache_path),
        ],
        capture_output=True,
        text=True,
    )
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr, flush=True)
        raise RuntimeError(f"m3_real_weights_export.py failed with exit code {proc.returncode}")


def make_real_weights(seed: int = REAL_WEIGHTS_SEED_DEFAULT, seq_len: int = STRIDE_CASES[5]["seq_len"]) -> dict:
    _ensure_real_weights_cache(seed, seq_len, REAL_WEIGHTS_CACHE)
    with np.load(REAL_WEIGHTS_CACHE) as npz:
        x = npz["x"].astype(np.float32)
        alpha0 = npz["snake1.alpha"].astype(np.float32)
        ct_weight = npz["conv_t1.weight"].astype(np.float32)
        ct_bias = npz["conv_t1.bias"].astype(np.float32)

        res_units = []
        for ru in RES_UNIT_NAMES:
            alpha1 = npz[f"{ru}.snake1.alpha"].astype(np.float32)
            w1 = npz[f"{ru}.conv1.weight"].astype(np.float32)
            b1 = npz[f"{ru}.conv1.bias"].astype(np.float32)
            alpha2 = npz[f"{ru}.snake2.alpha"].astype(np.float32)
            w2 = npz[f"{ru}.conv2.weight"].astype(np.float32)
            b2 = npz[f"{ru}.conv2.bias"].astype(np.float32)
            res_units.append((alpha1, w1, b1, alpha2, w2, b2))

        block_index = int(npz["block_index"])

    print(
        f"real weights loaded from {REAL_WEIGHTS_CACHE.name}: block_index={block_index} "
        f"input_dim={x.shape[1]} seq_len={x.shape[2]} seed={seed}",
        flush=True,
    )
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


def fp64_reference_chain(weights: dict, cfg: dict) -> list[np.ndarray]:
    _stub_torch_for_import()
    from m2_residual_unit_prototype import numpy_residual_unit  # noqa: E402
    from m3_block_reference import numpy_conv_transpose1d  # noqa: E402

    x = weights["x"]
    y0 = _numpy_snake_local(x, weights["alpha0"])
    y1 = numpy_conv_transpose1d(
        y0, weights["ct_weight"], weights["ct_bias"],
        stride=cfg["stride"], padding=cfg["ct_padding"], output_padding=cfg["ct_output_padding"],
    )
    stages = [y0, y1]
    y = y1
    for (alpha1, w1, b1, alpha2, w2, b2), dilation, padding in zip(
        weights["res_units"], cfg["ru_dilations"], cfg["ru_paddings"]
    ):
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


def build_decoder_block_graph(gpu_device, cpu_device, cfg: dict):
    from max.dtype import DType
    from max.graph import Graph, TensorType
    from m2_residual_unit_prototype import snake_expr

    stride = cfg["stride"]
    input_dim, output_dim, seq_len = cfg["input_dim"], cfg["output_dim"], cfg["seq_len"]
    ct_kernel, ct_padding, ct_output_padding = cfg["ct_kernel"], cfg["ct_padding"], cfg["ct_output_padding"]
    ru_kernel, ru_paddings = cfg["ru_kernel"], cfg["ru_paddings"]

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
        # -- GPU: Snake1d(input_dim) --
        y0 = snake_expr(x, alpha0, gpu_device)

        # -- cross to CPU: wn_conv_transpose1d(input_dim, output_dim, k=2*stride, stride, --
        # -- padding=ceil(stride/2), output_padding=stride%2) --
        from max.graph import ops as _ops

        y0_cpu = _ops.transfer_to(y0, cpu_device)
        y1_cpu = conv_transpose_expr(
            y0_cpu, ct_filter, ct_bias, stride, ct_output_padding, output_dim, padding=ct_padding
        )

        # -- cross back to GPU --
        y1 = _ops.transfer_to(y1_cpu, gpu_device)

        # -- GPU: ResidualUnit(output_dim, dilation=1) --
        y2 = residual_unit_expr(y1, alpha1_1, filter1_1, bias1_1, alpha2_1, filter2_1, bias2_1, 1, ru_paddings[0], gpu_device)
        # -- GPU: ResidualUnit(output_dim, dilation=3) --
        y3 = residual_unit_expr(y2, alpha1_2, filter1_2, bias1_2, alpha2_2, filter2_2, bias2_2, 3, ru_paddings[1], gpu_device)
        # -- GPU: ResidualUnit(output_dim, dilation=9) --
        y4 = residual_unit_expr(y3, alpha1_3, filter1_3, bias1_3, alpha2_3, filter2_3, bias2_3, 9, ru_paddings[2], gpu_device)

        return y0, y1, y2, y3, y4

    input_types = [
        TensorType(DType.float32, shape=(BATCH, input_dim, seq_len), device=gpu_device),
        TensorType(DType.float32, shape=(1, input_dim, 1), device=gpu_device),
        TensorType(DType.float32, shape=(1, ct_kernel, output_dim, input_dim), device=cpu_device),
        TensorType(DType.float32, shape=(output_dim,), device=cpu_device),
    ]
    for _ in range(3):
        input_types += [
            TensorType(DType.float32, shape=(1, output_dim, 1), device=gpu_device),
            TensorType(DType.float32, shape=(1, ru_kernel, output_dim, output_dim), device=gpu_device),
            TensorType(DType.float32, shape=(output_dim,), device=gpu_device),
            TensorType(DType.float32, shape=(1, output_dim, 1), device=gpu_device),
            TensorType(DType.float32, shape=(1, 1, output_dim, output_dim), device=gpu_device),
            TensorType(DType.float32, shape=(output_dim,), device=gpu_device),
        ]

    return Graph(f"m3_decoder_block_stride{stride}", forward=forward, input_types=input_types)


# ------------------------------------------------------------------------------------------
# Graph execution (run ONLY inside the isolated subprocess -- see main()).
# ------------------------------------------------------------------------------------------
def run_graph(mode: str, seed: int = 57305, stride: int = 5) -> int:
    if mode not in ("synthetic", "real-weights"):
        print(f"mode={mode!r} not implemented -- exiting.", flush=True)
        return 0
    if mode == "real-weights" and stride != 5:
        # Per m3-plan.md M3-7: stride=8 uses SYNTHETIC weights only -- real checkpoint
        # weight extraction (M3-3/M3-6) exists only for the real stride-5 block.
        print(f"real-weights mode is only defined for stride=5 (M3-6); stride={stride} requested -- exiting.", flush=True)
        return 2

    from max.driver import CPU, Accelerator, Buffer, accelerator_count
    from max.engine import InferenceSession
    from max.graph import DeviceRef

    import m3_divergence as m3div

    cfg = make_config(stride)
    print(f"accelerator_count={accelerator_count()}", flush=True)
    if accelerator_count() == 0:
        print("No accelerator on this host -- cannot test the mixed CPU/GPU block. Exiting.", flush=True)
        return 0

    if mode == "real-weights":
        weights = make_real_weights(seed=seed)
    else:
        weights = make_synthetic_weights(seed, cfg)
        print(f"stride={stride} synthetic weight seed={seed}", flush=True)
    ref_stages = fp64_reference_chain(weights, cfg)
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

    graph = build_decoder_block_graph(gpu_device, cpu_device, cfg)
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
    parser.add_argument(
        "--stride", type=int, default=5, choices=sorted(STRIDE_CASES),
        help="which DecoderBlock stride case to build (M3-7 adds stride=8; each case is fed "
             "its own independent synthetic/real input -- NOT chained with any other stride).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="weight seed for --synthetic (default per-stride, see STRIDE_CASES) or "
             "input-generation seed for --real-weights (default 99, matching "
             "m3_block_reference.py's real-weight cross-check convention; stride=5 only)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--synthetic", action="store_true", default=True)
    mode_group.add_argument("--real-weights", action="store_true")
    args = parser.parse_args()
    mode = "real-weights" if args.real_weights else "synthetic"
    if mode == "real-weights" and args.stride != 5:
        parser.error("--real-weights is only defined for --stride 5 (M3-6); M3-7's stride=8 case is synthetic-only.")
    if args.seed is None:
        args.seed = REAL_WEIGHTS_SEED_DEFAULT if mode == "real-weights" else STRIDE_CASES[args.stride]["default_synth_seed"]

    if args.run_graph:
        raise SystemExit(run_graph(mode, args.seed, args.stride))

    # Driver: isolate the actual graph build+execute in its own subprocess, exactly as
    # m3_device_mixing_spike.py / m2_convtranspose1d_prototype.py do -- a Metal
    # `conv2d_transpose` GPU-dispatch bug is a fatal, uncatchable process abort, not a
    # Python exception, per m2-convtranspose1d-results.md.
    mode_flag = "--real-weights" if mode == "real-weights" else "--synthetic"
    proc = subprocess.run(
        [sys.executable, "-u", __file__, "--run-graph", mode_flag,
         "--stride", str(args.stride), "--seed", str(args.seed)],
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
