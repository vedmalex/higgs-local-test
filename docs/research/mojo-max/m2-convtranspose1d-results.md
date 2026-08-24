# M2 prototype #3 — ConvTranspose1d GPU-executability: CPU works, Metal GPU hard-crashes

Date: 2026-08-24. Run: `docs/research/mojo-max/m2_convtranspose1d_prototype.py` (plus an
isolated per-case subprocess runner, needed because the GPU failure mode is a fatal process
abort, not a catchable exception), on this project's M1, same pixi env as the other M2
prototypes. This is the E4 check from
[`m1-responsibility-map.md`](m1-responsibility-map.md) §6/§11 — the map's single highest-risk
item, since the entire 960× upsample in Higgs's Code2Wav is built from `ConvTranspose1d`, and
upstream's own source carries a `# TODO(GEX-2043): Add support for GPU kernel for
conv_transpose` comment.

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
2. **This crash is specifically an attempt to load a CUDA-only library.** On a genuine T4 (which
   *does* have CUDA and, presumably, cuDNN available in a properly configured environment), this
   exact failure mode may not occur — the crash is plausibly Metal-specific, not a general
   "ConvTranspose1d GPU is broken everywhere" fact. **This must be tested on T4, not assumed
   either way** — the T4 result could be "works via real cuDNN," "same abort if cuDNN isn't on
   the runtime's library path," or something else entirely.
3. **The docstring's "output_paddings: Only 0 is supported" claim is not enforced/true on the
   CPU path** in this MAX version — both `output_padding=1` cases ran and produced finite,
   correctly-shaped output. Do not treat the docstring text as authoritative without an empirical
   check, as this session already had to correct once for the E1 decoder-branch question.

## Next

- Run this exact script on Colab T4 — the critical missing data point. If GPU also aborts
  there, the port's Apple-GPU and T4-GPU upsample story both currently require CPU-only
  `ConvTranspose1d`, which changes the performance conversation substantially (not the
  correctness one). If it works on T4, the gap is confirmed Metal-specific and worth reporting
  upstream to Modular as a Metal-backend bug (attempting a CUDA-only symbol load on a non-CUDA
  accelerator).
- Report this finding upstream if T4 confirms it's Metal-specific: `modular/modular`'s
  `conv2d_transpose` implementation should not attempt to load `cudnnCreate` when the target
  device is not a CUDA device.
