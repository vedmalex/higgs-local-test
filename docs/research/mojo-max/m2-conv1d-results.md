# M2 prototype #2 — weight-normed dilated Conv1d via `ops.conv2d`: PASSED on M1 and T4 GPU

Date: 2026-08-24. Run: `docs/research/mojo-max/m2_conv1d_prototype.py`, same M1/Metal GPU and
pixi env as the Snake1d prototype. Second M2 prototype per
[`m1-responsibility-map.md`](m1-responsibility-map.md) §5/§11 — route A: express Higgs's Conv1d
(full, non-causal, dilated, weight-normalized) as `ops.conv2d` with a degenerate height axis,
rather than writing a custom Mojo conv1d kernel.

This answers the map's stated "single largest structural question": can Higgs's conv shape be
expressed in MAX without a custom Mojo kernel?

## Setup

Reproduces one `_BosonResidualUnit` conv exactly: `C_in=C_out=32`, `kernel=7`, `dilation=3`,
`padding=(kernel-1)*dilation//2=9` — the same formula `higgs_audio_decoder.py:117` uses. Random
weight-norm parametrization (`g`, `v`), folded to a plain `[C_out,C_in,K]` weight in FP64 on the
host per §5's recommendation (fold in FP32/FP64, never FP16 — it's a sum-of-squares reduction
over `C_in*K=224` elements, the same overflow-prone shape M0 flagged for RMSNorm).

**Layout mapping** (`route A`): input `[B,C_in,T]` → NHWC `[B,1,T,C_in]` via
`permute`+`unsqueeze`; PyTorch weight `[C_out,C_in,K]` → RSCF `[1,K,C_in,C_out]` via
`transpose(2,1,0)` + a leading axis; `padding=(0,0,pad,pad)` (H gets no padding, W gets the
conv1d's symmetric padding); `dilation=(1, d)`. Output comes back NHWC `[B,1,T_out,C_out]`,
squeezed and permuted back to `[B,C_out,T_out]`.

Reference: a from-scratch FP64 NumPy conv1d (explicit sliding-window sum via `einsum`, no
scipy/torch dependency), matching PyTorch's `Conv1d` semantics for symmetric zero padding and
dilation.

## Result

```text
padding=9, input T=64, reference output shape=(1, 32, 64)
device: Device(type=gpu,id=0), accelerator_count=1
MAX output shape=(1, 32, 64)
max|err|=2.37e-06  max_rel_err=6.45e-04  nan/inf=0
```

- **Shapes match exactly** — the padding/dilation arithmetic translated correctly between
  PyTorch's convention and `ops.conv2d`'s `(before,after)`-pair convention. This was flagged in
  the map (§7) as a plausible off-by-one source; it was not one here, but this is one padding/
  dilation combination, not exhaustive coverage.
- **Numerics match to FP32-rounding level** (2.4e-6 absolute, on values order ~1) — no
  meaningful divergence from the reference.
- Ran on the Apple M1 **GPU** (`Device(type=gpu,id=0)`), not CPU-only.

## What this resolves

**Route A works for correctness, on this platform, for this one conv configuration.** No custom
Mojo kernel is needed to get a numerically correct dilated Conv1d in MAX — `ops.conv2d` with a
degenerate height axis, plus a host-side weight_norm fold, is sufficient. Per the map's own
framing, this makes **M3 a graph-assembly task for Conv1d specifically**, not a kernel-authoring
task — the opposite finding would have forced writing a hand-rolled Mojo conv1d.

## What this does not resolve

- **This is Conv1d, not ConvTranspose1d.** The map's highest-risk item (§6) — whether
  `max.nn.ConvTranspose1d`'s GPU path actually executes, given `conv2d_transpose`'s `# TODO`
  marking it unimplemented on GPU upstream — is untested by this prototype and remains open.
  `_BosonDecoderBlock`'s upsampling step needs `ConvTranspose1d`, not `Conv1d`; this result does
  not extend to it.
- **Only one (kernel, dilation) combination tested** — `k=7, d=3`. The stack also uses `d=1`,
  `d=9`, and `k=1` (the residual unit's pointwise conv). Different dilation/kernel combinations
  are cheap to add but were not run here.
- **Correctness only, not performance.** This says nothing about whether `conv2d` with `H=1` is
  efficient on GPU — a real perf comparison against a native conv1d path is M4's job, not M2's.
- **Done below (T4 result)**: ran on Apple M1 (Metal) first; now also re-run unchanged on Colab
  T4 (CUDA), with matching results — the T4 re-run discipline flagged here is complete for this
  prototype.

## T4 result (2026-08-24, via Colab T4 run)

Full raw output: [`m2-conv1d-output-t4.txt`](m2-conv1d-output-t4.txt).

```text
padding=9, input T=64, reference output shape=(1, 32, 64)
device: Device(type=gpu,id=0), accelerator_count=1
MAX output shape=(1, 32, 64)
max|err|=2.37133e-06 max_rel_err=0.000645421 nan/inf=0
```

**This matches the M1 result to within normal FP rounding, essentially exactly:**

| | M1 | T4 |
| --- | --- | --- |
| shape | (1, 32, 64) | (1, 32, 64) |
| max abs err | 2.37e-06 | 2.37133e-06 |
| max rel err | 6.45e-04 | 6.45421e-04 |
| nan/inf | 0 | 0 |

The absolute and relative errors agree to 3+ significant figures (2.37e-06 vs 2.37133e-06 abs;
6.45e-04 vs 6.45421e-04 rel), and shapes and nan/inf counts are identical. **Route A
(`ops.conv2d` with a degenerate height axis for Higgs's weight-normed dilated Conv1d) is now
confirmed numerically correct on both Apple M1 GPU (Metal) and Tesla T4 (CUDA)**, not just on M1.

## Next

- The `ConvTranspose1d` GPU-executability question (map §6/E4) is now the most valuable
  remaining unknown — it's the op the entire 960× upsample depends on, and it's exactly the one
  flagged upstream as GPU-unimplemented. That should be the next prototype, not further Conv1d
  variations.
