"""M3-8 for issue #57: exercise the `pad > 0` crop branch of `_BosonResidualUnit.forward`
(`higgs_audio_decoder.py:114-130`, mirrored in `m2_residual_unit_prototype.py`'s
`numpy_residual_unit` / `build_graph`), which `m2-residual-unit-results.md` records has ONLY
ever run its `pad == 0` no-op branch.

Standalone by design (see the M3-8 task note in `m3-plan.md`): kept separate from
`m3_decoder_block_prototype.py` so this does not collide with concurrent M3-7 work on that file.

Step 1 below answers reachability FIRST, per the plan: does any real `_BosonDecoderBlock`
residual-unit config (kernel=7, dilation in {1,3,9}) ever produce `pad > 0`? If not (spoiler:
it does not -- shown with explicit arithmetic below), this task is defensive/edge-case-only,
and Step 2 constructs one synthetic even-diff config to exercise the crop branch's *arithmetic*
plus one synthetic ODD-diff config to exercise the asymmetric-crop hazard
`m1-responsibility-map.md` S7 warns about.

Usage: (inside `.mojo-probe-stable`) `arch -arm64 pixi run python
../docs/research/mojo-max/m3_padding_branch_check.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from m2_residual_unit_prototype import (  # noqa: E402
    conv1d_expr,
    fold_weight_norm,
    numpy_conv1d,
    numpy_snake,
    snake_expr,
)
from m3_divergence import compare  # noqa: E402

from max.driver import CPU, Buffer  # noqa: E402
from max.dtype import DType  # noqa: E402
from max.engine import InferenceSession  # noqa: E402
from max.graph import DeviceRef, Graph, TensorType, ops  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1 -- reachability check (real _BosonDecoderBlock residual-unit configs)
# ---------------------------------------------------------------------------

def reachability_check() -> bool:
    """`_BosonResidualUnit.__init__` (higgs_audio_decoder.py): kernel is hardcoded to 7,
    `pad = ((7 - 1) * dilation) // 2`, dilation in {1, 3, 9} for the three residual units in
    every real `_BosonDecoderBlock`. For a stride-1, dilation-`d` conv with kernel `k` and
    symmetric padding `P` on a length-`T` input:

        len(y) = T + 2*P - (k - 1)*d

    so `diff = len(x) - len(y) = (k - 1)*d - 2*P`. With the real formula `P = (k-1)*d // 2`:
    since `k=7` makes `(k-1)=6` EVEN, `6*d` is even for every integer `d`, so `(6*d)//2 = 3*d`
    EXACTLY (no floor-division truncation ever occurs) and `2*P = 6*d = (k-1)*d` exactly.
    Therefore `diff = 0` for every real dilation -- not just the three actually used.
    """
    k = 7
    print("Step 1 -- reachability of pad > 0 for real _BosonDecoderBlock residual units")
    print(f"  kernel={k} (hardcoded in _BosonResidualUnit.__init__), (k-1)={k - 1} (EVEN)")
    any_reachable = False
    for d in (1, 3, 9):
        p_formula = ((k - 1) * d) // 2  # exactly what the real source computes
        p_exact = (k - 1) * d / 2       # no floor, for comparison
        two_p = 2 * p_formula
        diff = (k - 1) * d - two_p
        reachable = diff > 0
        any_reachable = any_reachable or reachable
        print(
            f"  dilation={d}: (k-1)*d={(k - 1) * d}, P=((k-1)*d)//2={p_formula} "
            f"(exact P would be {p_exact:g}, no truncation since (k-1)*d is even), "
            f"2*P={two_p}, diff=(k-1)*d-2*P={diff}  => pad_crop={'>' if diff > 0 else '=='} 0"
        )
    print(
        f"  VERDICT: any real dilation in {{1,3,9}} reaches pad>0? {any_reachable}. "
        "For every real dilation, (k-1) is even, so (k-1)*d is always even and the // in the "
        "padding FORMULA never truncates -- pad_crop is EXACTLY 0 for all three, always. This "
        "confirms m2-residual-unit-results.md's finding and extends it: it is not merely that "
        "the run so far happened to hit pad==0, it is that NO real Higgs decoder-block dilation "
        "can ever reach pad>0 with kernel=7."
    )
    return any_reachable


# ---------------------------------------------------------------------------
# Step 2a -- synthetic EVEN-diff case: pad>0 branch actually executes and produces a number
# ---------------------------------------------------------------------------

def build_crop_graph(dim, seq_len, batch, device, x_len_hint, y_len_hint):
    """Minimal graph: takes x (len=x_len_hint) and a precomputed y (len=y_len_hint, already the
    result of the block's snake->conv->snake->conv chain) and applies exactly the guarded
    symmetric-crop residual add from `_BosonResidualUnit.forward` / `build_graph` in
    m2_residual_unit_prototype.py."""

    def forward(x, y):
        x_len = int(x.shape[-1])
        y_len = int(y.shape[-1])
        pad = (x_len - y_len) // 2
        xc = x[:, :, pad : x_len - pad] if pad > 0 else x
        return ops.add(xc, y)

    return Graph(
        "crop_branch",
        forward=forward,
        input_types=[
            TensorType(DType.float32, shape=(batch, dim, x_len_hint), device=device),
            TensorType(DType.float32, shape=(batch, dim, y_len_hint), device=device),
        ],
    )


def even_diff_case() -> bool:
    """Kernel=7, dilation=3 (a real dilation value, reused for realism), but the padding
    DELIBERATELY set 2 less than the real formula's 9 -> P=7. diff = (k-1)*d - 2*P = 18-14 = 4
    (EVEN), pad_crop = 4//2 = 2 > 0: the crop guard branch actually executes."""
    print("\nStep 2a -- synthetic EVEN-diff case (pad>0, symmetric crop)")
    rng = np.random.default_rng(48)
    batch, dim, seq_len = 1, 16, 40
    kernel, dilation = 7, 3
    real_padding = (kernel - 1) * dilation // 2  # 9, the real formula's value (diff would be 0)
    forced_padding = real_padding - 2            # 7 -> forces diff=4, even, pad_crop=2
    assert forced_padding >= 0

    x_np = rng.uniform(-2.0, 2.0, size=(batch, dim, seq_len)).astype(np.float32)
    alpha1 = rng.uniform(0.3, 1.2, size=(1, dim, 1)).astype(np.float32)
    alpha2 = rng.uniform(0.3, 1.2, size=(1, dim, 1)).astype(np.float32)
    g1 = rng.uniform(0.5, 2.0, size=(dim, 1, 1)).astype(np.float32)
    v1 = rng.normal(0, 0.3, size=(dim, dim, kernel)).astype(np.float32)
    w1 = fold_weight_norm(g1, v1)
    b1 = rng.normal(0, 0.1, size=(dim,)).astype(np.float32)
    g2 = rng.uniform(0.5, 2.0, size=(dim, 1, 1)).astype(np.float32)
    v2 = rng.normal(0, 0.3, size=(dim, dim, 1)).astype(np.float32)
    w2 = fold_weight_norm(g2, v2)
    b2 = rng.normal(0, 0.1, size=(dim,)).astype(np.float32)

    # FP64 reference block computation (mirrors numpy_residual_unit exactly, but with the
    # deliberately-reduced padding so len(x) != len(y)).
    y64 = numpy_snake(x_np, alpha1)
    y64 = numpy_conv1d(y64, w1, b1, dilation=dilation, padding=forced_padding)
    y64 = numpy_snake(y64, alpha2)
    y64 = numpy_conv1d(y64, w2, b2, dilation=1, padding=0)
    x_len, y_len = x_np.shape[-1], y64.shape[-1]
    diff = x_len - y_len
    pad_ref = diff // 2
    print(f"  x_len={x_len} y_len={y_len} diff={diff} (even) pad_crop={pad_ref}")
    assert diff > 0 and diff % 2 == 0 and pad_ref > 0, "case setup must force an even, positive diff"
    xc64 = x_np.astype(np.float64)[..., pad_ref : x_len - pad_ref]
    ref = xc64 + y64
    print(f"  FP64 reference shape={ref.shape} (matches y_len={y_len}: {ref.shape[-1] == y_len})")

    # MAX graph: run the same block via snake_expr/conv1d_expr (CPU is sufficient -- no GPU
    # placement needed for this branch, per the plan's "Devices: CPU is sufficient" note), then
    # the crop-guard graph above for the residual add.
    device_obj = CPU()
    device = DeviceRef.from_device(device_obj)
    session = InferenceSession(devices=[device_obj])

    def block_forward(x, a1, f1, bb1, a2, f2, bb2):
        yy = snake_expr(x, a1, device)
        yy = conv1d_expr(yy, f1, bb1, dilation, forced_padding)
        yy = snake_expr(yy, a2, device)
        yy = conv1d_expr(yy, f2, bb2, 1, 0)
        return yy

    block_graph = Graph(
        "block_only",
        forward=block_forward,
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
    block_model = session.load(block_graph)
    filter1_rscf = np.transpose(w1, (2, 1, 0))[np.newaxis, ...].copy()
    filter2_rscf = np.transpose(w2, (2, 1, 0))[np.newaxis, ...].copy()
    block_bufs = [
        Buffer.from_numpy(x_np).to(device_obj),
        Buffer.from_numpy(alpha1).to(device_obj),
        Buffer.from_numpy(filter1_rscf).to(device_obj),
        Buffer.from_numpy(b1).to(device_obj),
        Buffer.from_numpy(alpha2).to(device_obj),
        Buffer.from_numpy(filter2_rscf).to(device_obj),
        Buffer.from_numpy(b2).to(device_obj),
    ]
    y_max = block_model.execute(*block_bufs)[0]
    y_max_np = y_max.to(CPU()).to_numpy()
    assert y_max_np.shape[-1] == y_len, "MAX block output length must match the NumPy chain"

    crop_graph = build_crop_graph(dim, seq_len, batch, device, x_len, y_len)
    crop_model = session.load(crop_graph)
    crop_bufs = [
        Buffer.from_numpy(x_np).to(device_obj),
        Buffer.from_numpy(y_max_np.astype(np.float32)).to(device_obj),
    ]
    got = crop_model.execute(*crop_bufs)[0].to(CPU()).to_numpy().astype(np.float64)

    report = compare(got, ref)
    print(f"  MAX crop-branch result vs FP64 reference: {report}")
    ok = report.is_healthy_combined() and got.shape == ref.shape
    print(f"  Step 2a PASS={ok}")
    return ok


# ---------------------------------------------------------------------------
# Step 2b -- synthetic ODD-diff case: the asymmetric-crop hazard from m1-responsibility-map.md S7
# ---------------------------------------------------------------------------

def odd_diff_case() -> bool:
    """Per m1-responsibility-map.md S7: 'when len_x - len_y is odd, integer division makes the
    crop asymmetric-by-one in PyTorch's favour.' Concretely: pad = diff // 2 removes 2*pad
    elements total; for an odd diff, 2*pad = diff - 1, so the cropped x ends up
    len(y) + 1 -- ONE element longer than y, not equal to it. This is checked directly against
    REAL PyTorch (not assumed) below: the residual add x_cropped + y raises a genuine
    RuntimeError in PyTorch for any odd diff once pad>0, and the FP64 NumPy mirror of the exact
    same guard must raise the equivalent (ValueError) error -- i.e. matching PyTorch's actual
    behaviour means reproducing the SAME FAILURE, not computing a spurious number.
    """
    print("\nStep 2b -- synthetic ODD-diff case (the S7 asymmetric-crop hazard)")

    def crop_guard(len_x, len_y):
        pad = (len_x - len_y) // 2
        return pad

    # Sweep small diffs to show the pattern generally (arithmetic only, no torch needed for this
    # table -- it is the same integer arithmetic PyTorch's `//` does on positive ints).
    print("  diff  pad_crop  xc_len  y_len  xc_len==y_len")
    for diff in range(0, 6):
        len_x, len_y = 40, 40 - diff
        pad = crop_guard(len_x, len_y)
        xc_len = len_x - 2 * pad if pad > 0 else len_x
        print(f"  {diff:>4}  {pad:>8}  {xc_len:>6}  {len_y:>5}  {xc_len == len_y}")

    # Concrete odd case with pad>0: diff=3 (odd), forced via kernel=8 (EVEN -> (k-1)=7 ODD),
    # dilation=1 (ODD), so (k-1)*d=7 is ODD -- any padding choice on this config yields an odd
    # diff (2*P is always even, so diff = 7 - 2*P is always odd, regardless of P). Choose P=2
    # (< the floor-halved value 3) to also make pad_crop >= 1, i.e. to actually enter the guard.
    rng = np.random.default_rng(777)
    batch, dim, seq_len = 1, 8, 30
    kernel, dilation, padding = 8, 1, 2
    x_np = rng.uniform(-1.0, 1.0, size=(batch, dim, seq_len)).astype(np.float32)
    alpha1 = rng.uniform(0.3, 1.2, size=(1, dim, 1)).astype(np.float32)
    g1 = rng.uniform(0.5, 2.0, size=(dim, 1, 1)).astype(np.float32)
    v1 = rng.normal(0, 0.3, size=(dim, dim, kernel)).astype(np.float32)
    w1 = fold_weight_norm(g1, v1)
    b1 = rng.normal(0, 0.1, size=(dim,)).astype(np.float32)

    y64 = numpy_snake(x_np, alpha1)
    y64 = numpy_conv1d(y64, w1, b1, dilation=dilation, padding=padding)
    x_len, y_len = x_np.shape[-1], y64.shape[-1]
    diff = x_len - y_len
    pad = crop_guard(x_len, y_len)
    xc_len = x_len - 2 * pad if pad > 0 else x_len
    print(
        f"  concrete case: kernel={kernel} dilation={dilation} padding={padding} "
        f"x_len={x_len} y_len={y_len} diff={diff} (odd={diff % 2 == 1}) pad_crop={pad} "
        f"xc_len={xc_len} (!= y_len: {xc_len != y_len})"
    )
    assert diff > 0 and diff % 2 == 1 and pad > 0, "case setup must force an odd, positive diff with pad>0"
    assert xc_len != y_len, "the S7 hazard: cropped-x length must NOT equal y length for odd diff"

    # NumPy mirror of the exact guard: attempting the add must raise, matching real PyTorch.
    xc64 = x_np.astype(np.float64)[..., pad : x_len - pad]
    numpy_raised = False
    try:
        _ = xc64 + y64
    except ValueError as e:
        numpy_raised = True
        print(f"  NumPy add raised (expected): {e!r}")

    # Cross-check against REAL PyTorch, if available, so "matches PyTorch's actual behaviour" is
    # verified against PyTorch itself, not assumed from the map's prose description.
    torch_raised = None
    torch_error_repr = None
    try:
        import torch

        xt = torch.from_numpy(x_np.astype(np.float64))
        yt = torch.from_numpy(y64)
        padt = (xt.shape[-1] - yt.shape[-1]) // 2
        xct = xt[..., padt : xt.shape[-1] - padt] if padt > 0 else xt
        try:
            _ = xct + yt
            torch_raised = False
        except RuntimeError as e:
            torch_raised = True
            torch_error_repr = repr(e)
            print(f"  PyTorch add raised (expected): {e!r}")
    except ImportError:
        print("  torch not importable in this env; skipping the live PyTorch cross-check "
              "(NumPy-vs-guard-arithmetic check above still stands).")

    ok = numpy_raised and (torch_raised is None or torch_raised)
    print(
        f"  Step 2b VERDICT: NumPy mirror raises on odd diff = {numpy_raised}; "
        f"real PyTorch raises = {torch_raised} ({torch_error_repr}). "
        f"MATCH (both fail the same way, confirming S7's hazard is a genuine crash, not a "
        f"silent asymmetric numeric answer) = {ok}"
    )
    return ok


if __name__ == "__main__":
    reachable = reachability_check()
    even_ok = even_diff_case()
    odd_ok = odd_diff_case()

    print("\n=== M3-8 summary ===")
    print(f"reachable_by_real_config={reachable}")
    print(f"even_diff_case_pass={even_ok}")
    print(f"odd_diff_case_pass={odd_ok}")
    overall = even_ok and odd_ok
    print(f"OVERALL PASS={overall}")
    sys.exit(0 if overall else 1)
