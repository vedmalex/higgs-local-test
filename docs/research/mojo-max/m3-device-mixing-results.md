# M3-1 — mixed CPU/GPU placement inside one MAX graph: PASSED on M1/Metal

Date: 2026-08-24. Run: `docs/research/mojo-max/m3_device_mixing_spike.py`, same M1 host and
same pixi env (`.mojo-probe-stable`) as the M2 prototypes. This is the M3-1 blocking spike from
[`m3-plan.md`](m3-plan.md) §2/§3 — the highest-risk open question for M3: can one MAX graph
legally mix a CPU-placed op into an otherwise-GPU graph, specifically routing
`ops.conv2d_transpose` (GPU-broken per `m2-convtranspose1d-results.md`) through the CPU while
everything else in the same graph stays GPU-placed.

## Setup

Minimal graph, exactly per the plan's spec:

```text
GPU Snake1d -> ops.transfer_to(CPU) -> ops.conv2d_transpose on CPU-placed operands
            (activation AND filter/bias, all CPU) -> ops.transfer_to(GPU) -> GPU Snake1d
```

`InferenceSession(devices=[Accelerator()])` — host CPU is appended automatically. Confirmed by
reading the actual installed source under this repo's pinned pixi env
(`.mojo-probe-stable/.pixi/envs/default/lib/python3.14/site-packages/max/engine/api.py`,
`InferenceSession.__init__`):

```python
host_cpu = CPU()
if host_cpu not in seen_devices:
    final_devices.append(host_cpu)
    seen_devices.add(host_cpu)
```

`DeviceRef.CPU()` (from `max/graph/type.py`) was used to place the transposed-conv's activation,
filter, and bias — not just the activation — as the plan requires ("a CPU-placed activation with
a GPU-placed weight/bias would not actually test CPU dispatch of the op").

**Toolchain actually used** (checked live, not assumed from the plan):

```text
$ arch -arm64 pixi run python --version   -> Python 3.14.7
$ arch -arm64 pixi run mojo --version     -> Mojo 1.0.0 (ed45d567)
$ arch -arm64 pixi run max --version      -> MAX 26.5.0
$ pixi list | grep -i max/mojo/modular:
  max            26.5.0  conda  https://conda.modular.com/max
  mojo           1.0.0   conda  https://conda.modular.com/max
  modular        26.5.0  conda  https://conda.modular.com/max
$ sw_vers -> macOS 26.6.2 (25G83)
$ xcodebuild -version -> Xcode 26.6 (Build 17F113)
```

**This exactly matches the plan's pinned toolchain** (MAX 26.5.0, Mojo 1.0.0 `ed45d567`, macOS
26.6.2 (25G83), Xcode 26.6 (17F113)) — no discrepancy to report.

Subprocess isolation, per `m2-convtranspose1d-results.md`'s "Why this needed subprocess
isolation" section: the Metal GPU failure mode for `ops.conv2d_transpose` is a fatal, uncatchable
process abort, not a Python exception. `m3_device_mixing_spike.py` therefore never builds/runs
the graph in its own process — `main()` always spawns `python -u m3_device_mixing_spike.py
--run-graph` as an isolated `subprocess.run(...)` and inspects the exit code: `0` = clean pass,
a negative/large exit code = fatal abort (the negative signal), any other non-zero = a catchable
correctness failure.

## An upstream layout bug found along the way (not the M3-1 question, but real and worth recording)

Passing `bias=` directly into `ops.conv2d_transpose` produced a malformed output shape for this
graph's H=1 NHWC route-A layout: `(1, C_out, T_out, C_out)` instead of the expected
`(1, 1, T_out, C_out)`. Confirmed by tracing the op's symbolic shape standalone:

```text
x_nhwc shape: [Dim(1), Dim(1), Dim(12), Dim(8)]
filter shape: [Dim(1), Dim(10), Dim(6), Dim(8)]
no-bias out shape:   [Dim(1), Dim(1), Dim(66), Dim(6)]   -- correct, matches m2's NHWC finding
with-bias out shape: [Dim(1), Dim(6), Dim(66), Dim(6)]   -- wrong, C_out leaks into two axes
```

`m2_convtranspose1d_prototype.py`'s working route never exercised the bias path either (it never
passed `bias=`), so this wasn't previously caught. Worked around the same way M2 already
validated: call `conv2d_transpose(bias=None)`, then add the CPU-placed bias manually after the
squeeze/permute back to `[B, C, T]`. This is a separate, minor upstream bug (candidate addition to
M3-11's Modular bug report) — it is not the mixed-device question and does not change the M3-1
verdict below.

## Result

Three consecutive runs, byte-identical numeric output each time:

```text
--- subprocess stdout ---
accelerator_count=1
FP64 reference output shape=(1, 6, 66)
Session devices requested=[Accelerator()] (host CPU auto-appended per InferenceSession.__init__); activation device=gpu:0, conv_transpose device=cpu:0
MAX mixed-device output shape=(1, 6, 66)
max|err|=3.47909e-07 max_rel_err=1.68195e-05 nan/inf=0
RESULT: PASS -- mixed CPU/GPU single-graph execution produced a finite, correct output

--- subprocess exit code: 0 ---
VERDICT: mixed CPU/GPU placement inside one MAX graph WORKS on this toolchain.
```

No abort. No traceback. Exit code `0` from the isolated subprocess on all 3 runs. Output shape
matches the FP64 NumPy reference exactly (`(1, 6, 66)`), max abs err `3.48e-07`, max rel err
`1.68e-05`, zero NaN/Inf — well inside FP32-vs-FP64 rounding noise, consistent with M2's own
composite numbers (e.g. residual-unit max|err| 4.10e-06).

## Verdict: PASSED — mixed-device single-graph placement works on this toolchain

Per the plan's own done-criterion: "a clean run is proof of CPU dispatch on this pinned
toolchain" because the alternative (the transposed conv actually dispatching to GPU) is a fatal,
uncatchable `cudnnCreate` process abort on Metal, per `m2-convtranspose1d-results.md` — and that
abort did **not** occur, across 3 runs. A clean, numerically-correct, finite result from the
isolated subprocess **is** the positive signal this spike was designed to produce.

**M3-1 is answered: YES, one MAX graph can mix CPU- and GPU-placed operands, and a
CPU-placed `ops.conv2d_transpose` really does dispatch to the CPU kernel inside an
otherwise-GPU-hosting `InferenceSession`, on MAX 26.5.0 / Mojo 1.0.0 / M1 Metal.** The two-graph
fallback design (§3 of the plan) is **not required** for Stage C on this toolchain. M3-5 onward
can proceed with the single mixed-device graph shape the plan describes as its "intended M3
shape."

**Caveat, restated from the plan (§2 M3-1, "Caveat on re-use of this criterion")**: this verdict
is valid only as long as `ops.conv2d_transpose`'s GPU path keeps failing the way
`m2-convtranspose1d-results.md` documents. If Modular ever fixes the Metal GPU dispatch for this
op, a clean run would stop distinguishing "ran on CPU as intended" from "ran on GPU because the
bug got fixed" — at that point a positive placement check (e.g. compiled-graph per-op device
inspection) would be needed instead of this absence-of-crash proxy.

**Not yet answered — deferred to M3-10**: whether this same mixed-device single-graph placement
behaves identically under CUDA on a Tesla T4. Per the plan, this is expected to work there too
(Tesla T4 CPU dispatch of this op was already 5/5 in M2), but is unverified until M3-10 actually
runs it.

## Files

- `docs/research/mojo-max/m3_device_mixing_spike.py` — the graph + subprocess-isolated runner
  (this doc's source of truth for the exact commands/output above).
