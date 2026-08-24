# M2 prototype #3 — ConvTranspose1d GPU-executability: CPU works everywhere, GPU fails on both M1 and T4

Date: 2026-08-24. Run: `docs/research/mojo-max/m2_convtranspose1d_prototype.py` (plus an
isolated per-case subprocess runner, needed because the GPU failure mode is a fatal process
abort, not a catchable exception), on this project's M1, same pixi env as the other M2
prototypes. This is the E4 check from
[`m1-responsibility-map.md`](m1-responsibility-map.md) §6/§11 — the map's single highest-risk
item, since the entire 960× upsample in Higgs's Code2Wav is built from `ConvTranspose1d`, and
upstream's own source carries a `# TODO(GEX-2043): Add support for GPU kernel for
conv_transpose` comment.

## Reported upstream (M3-11)

Reported as comments on two **pre-existing** `modular/modular` issues rather than as a new issue
(a duplicate-search found both already open, so a third report would have fragmented triage):

- [modular/modular#6563 (comment)](https://github.com/modular/modular/issues/6563#issuecomment-5400477726)
  — the `max.graph`-only forward reproducer that maintainer `mdanatg` explicitly asked for on that
  thread; shows the `symbol not found: cudnnCreate` abort needs no autodiff/backward pass.
- [modular/modular#6726 (comment)](https://github.com/modular/modular/issues/6726#issuecomment-5400479775)
  — the Metal + Tesla T4 + MAX 26.5.0 data points (that thread was A10 / MAX 26.2.0), plus the
  separate `bias=` NHWC double-permute layout bug from
  [`m3-device-mixing-results.md`](m3-device-mixing-results.md), offered for splitting out if
  upstream prefers.

## Setup

Tested all five `(stride, output_padding)` pairs Higgs's five decoder blocks actually use
(`rates=(8,5,4,2,3)`, `output_padding = stride % 2`): `(8,0), (5,1), (4,0), (2,0), (3,1)`.
Channel counts shrunk to 16 for a fast prototype — the (stride, output_padding) combination is
what's under test, not raw channel count. Same route-A layout mapping as the Conv1d prototype
(NHWC input via permute+unsqueeze, RSCF-ish filter), built via `ops.conv2d_transpose`.

Method: build the identical graph for each case, execute once on `CPU()` and once on
`Accelerator()` (this M1's Metal GPU), each in its own subprocess (required — see below).

## Result

```text
case=0 device=cpu stride=8 output_padding=0 -> shape=(1, 16, 136) nan_inf=0
case=1 device=cpu stride=5 output_padding=1 -> shape=(1, 16,  86) nan_inf=0
case=2 device=cpu stride=4 output_padding=0 -> shape=(1, 16,  68) nan_inf=0
case=3 device=cpu stride=2 output_padding=0 -> shape=(1, 16,  34) nan_inf=0
case=4 device=cpu stride=3 output_padding=1 -> shape=(1, 16,  52) nan_inf=0

case=0..4 device=gpu (Metal, Accelerator()) -> ALL FIVE:
  ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:762:18: symbol not found: cudnnCreate
```

**CPU: all five cases, including both `output_padding=1` cases, execute cleanly** — no NaN/Inf,
correct output shapes matching the expected `(L_in-1)*stride - 2*0 + dilation*(k-1) +
output_padding + 1` transposed-conv length formula. **The map's docstring-derived concern that
"output_paddings: Only 0 is supported" would block strides 5 and 3 did not materialize on CPU**
— worth correcting explicitly, since the docstring text alone would have wrongly flagged this as
a second blocker.

**GPU (Metal): every single case is a fatal, unrecoverable process abort**, not a Python
exception — `try`/`except` around the call does not catch it; the whole interpreter dies. The
error is `symbol not found: cudnnCreate` — an attempt to dynamically load **NVIDIA's cuDNN
library**, on an **Apple Silicon Metal** device. This is a materially different and more severe
finding than upstream's own `# TODO(GEX-2043)` comment implies: "GPU kernel not yet implemented"
would normally mean a clean `NotImplementedError` or a CPU fallback; this is MAX's
`conv2d_transpose` GPU dispatch **unconditionally attempting a CUDA/cuDNN code path regardless of
the actual accelerator backend**, and crashing the whole process when that backend is Metal, not
CUDA.

## Why this needed subprocess isolation

The first version of this prototype ran all five cases in one process with a `try`/`except`
around each device call, expecting a normal `Exception` on GPU failure per the "TODO" comment.
Instead the process aborted outright on the very first GPU case with no Python traceback and no
further output (stdout buffering made even the earlier successful CPU-case print vanish until
`python -u` was used). Confirmed via `python -u` that the abort happens exactly at the GPU
`conv2d_transpose` call, after the CPU case for the same stride had already printed `ok`. Each
case was then re-run as an isolated subprocess so one crash couldn't hide the other four results.

## What this means for the port

1. **Apple Silicon (M1/Metal) cannot currently run Higgs's upsample stage on GPU via
   `ops.conv2d_transpose`/`max.nn.ConvTranspose1d` in this MAX version at all** — not "slow", not
   "falls back to CPU gracefully", but a hard crash. Any M2/M3 work targeting Apple GPU for the
   full Code2Wav pipeline must either avoid this op on Metal (CPU-only for this stage, which is
   fine for **correctness** prototyping — M2's own stated goal — but blocks any Apple-GPU
   **performance** story until Modular fixes this), or wait for/report the fix.
2. **Resolved below (T4 section): the crash mechanism is Metal-specific, but the outcome is
   not.** On T4, cuDNN does load and the kernel dispatches — but every single one of the five
   tested cases still fails, with `CUDNN_STATUS_ALLOC_FAILED` rather than a missing-symbol
   abort. GPU execution of this op is currently unusable on both platforms, for different
   reasons.
3. **The docstring's "output_paddings: Only 0 is supported" claim is not enforced/true on the
   CPU path** in this MAX version — both `output_padding=1` cases ran and produced finite,
   correctly-shaped output. Do not treat the docstring text as authoritative without an empirical
   check, as this session already had to correct once for the E1 decoder-branch question.

## Next

- **Done below**: the T4 run (all 5 cases) — see "T4 result" and "Combined conclusion" sections.
- Report both findings upstream to Modular: the Metal-specific `cudnnCreate` symbol-load crash,
  and the T4 `CUDNN_STATUS_ALLOC_FAILED` on every tested shape for tensors too small to
  plausibly need it.
- For M3: plan the Code2Wav port's upsample stage to run on CPU, everything else (already
  GPU-correct on both M1 and T4: Snake1d, Conv1d, the residual-unit composite) on GPU, rather
  than blocking on a MAX fix.

## T4 result (2026-08-24, complete — all 5 cases via `notebooks/mojo_max_m2_t4.ipynb`)

Full raw output: [`m2-convtranspose1d-output-t4.txt`](m2-convtranspose1d-output-t4.txt).

```text
case=0 device=cpu stride=8 output_padding=0 -> shape=(1, 16, 136) nan_inf=0    -- PASSED
case=0 device=gpu -> cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED  -- FAILED
case=1 device=cpu stride=5 output_padding=1 -> shape=(1, 16,  86) nan_inf=0    -- PASSED
case=1 device=gpu -> cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED  -- FAILED
case=2 device=cpu stride=4 output_padding=0 -> shape=(1, 16,  68) nan_inf=0    -- PASSED
case=2 device=gpu -> cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED  -- FAILED
case=3 device=cpu stride=2 output_padding=0 -> shape=(1, 16,  34) nan_inf=0    -- PASSED
case=3 device=gpu -> cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED  -- FAILED
case=4 device=cpu stride=3 output_padding=1 -> shape=(1, 16,  52) nan_inf=0    -- PASSED
case=4 device=gpu -> cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED  -- FAILED
```

**CPU: 5/5 PASSED, exact match with the M1 CPU results** (same shapes, same clean run,
including both `output_padding=1` cases). **GPU: 5/5 FAILED, identically, for every single
`(stride, output_padding, kernel)` combination** — `kernel` ranges over 16, 10, 8, 4, 6 across
the five cases, and every one hits the exact same `CUDNN_STATUS_ALLOC_FAILED` error, at exactly
the same op (`conv_transpose_mogg`).

**This resolves the ambiguity the single-case partial result left open.** A shape-specific
workspace-sizing bug (candidate 2 from the partial write-up) would be expected to affect some
kernel sizes and not others — it did not; every kernel size from 4 to 16 failed identically.
That leaves two candidates, and the uniformity across all five cases favors the first:

1. **A systematic MAX bug in the `conv_transpose_mogg` cuDNN dispatch path**, requesting or
   sizing a workspace allocation that fails regardless of the (tiny, ~16-channel) input shape.
   The inputs here are minuscule — a real "allocation failed" for tensors this small, on a T4
   with 15 GB VRAM and nothing else running in a fresh Colab session, would be surprising if the
   requested workspace size were actually proportional to the op's real needs.
2. A T4-wide (`sm_75`) incompatibility in whichever cuDNN algorithm MAX's heuristic selects for
   transposed convolution, not caught earlier because it never got tested on real Turing
   hardware before this session.

Not distinguished by this pass (would need `MODULAR_DEBUG=source-tracebacks` and/or a direct
cuDNN workspace-size query outside of MAX to isolate further); either way, the practical
conclusion is the same.

## Combined conclusion — CPU-only on both targets for now

**`ConvTranspose1d`/`ops.conv2d_transpose` on GPU does not currently work on Apple Silicon
(Metal) or on Tesla T4 (CUDA) in this MAX version** — via two different failure mechanisms
(a fatal missing-symbol process abort on Metal; a catchable but 100%-reproducible cuDNN
allocation failure on T4), but with the same practical outcome: **CPU is the only GPU-backend
that currently executes this op correctly, on either platform.** CPU itself is fully correct
on both M1 and T4 (5/5 cases, identical shapes, zero NaN/Inf) — this is a GPU-execution gap in
MAX 26.5 stable specifically, not a Higgs-port design problem, and not resolved by choosing a
different platform.

Consequences for the Higgs Code2Wav port:

- A **correctness** parity experiment (M2's actual goal) can still proceed — it just needs to
  run the upsample stage on CPU while everything else (Snake1d, Conv1d — both confirmed
  GPU-correct on M1 *and* T4 above) can run on GPU. Mixed CPU/GPU execution within one MAX graph
  is a reasonable near-term plan, not a blocker.
- A **performance** story for the full pipeline cannot be told yet for the op that the entire
  960× upsample depends on — CPU execution of `ConvTranspose1d` inside an otherwise-GPU pipeline
  will be a real bottleneck, and this needs to be reported upstream rather than worked around
  silently.
- **Worth reporting to Modular** (`modular/modular`) as a GPU-execution bug in
  `conv2d_transpose`'s cuDNN dispatch: fails on every tested `(stride, output_padding, kernel)`
  combination on a real T4, for tensors far too small to plausibly exhaust 15 GB of VRAM, and
  separately crashes the whole process on Metal by attempting to load a CUDA-only symbol.
