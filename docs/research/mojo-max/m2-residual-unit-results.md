# M2 prototype #4 — full `_BosonResidualUnit` composite: PASSED on M1 GPU

Date: 2026-08-24. Run: `docs/research/mojo-max/m2_residual_unit_prototype.py`, same M1/Metal
GPU and pixi env as the other M2 prototypes. Combines the two already-validated primitives
(Snake1d, weight-normed Conv1d) into the actual composite unit Higgs's real architecture uses
three times per decoder block (`_BosonResidualUnit`, `higgs_audio_decoder.py:114-130`):

```text
y = Snake1d(alpha1)(x)
y = wn_conv1d(dim, dim, k=7, dilation=3, padding=9)(y)
y = Snake1d(alpha2)(y)
y = wn_conv1d(dim, dim, k=1)(y)                      # pointwise
pad = (len(x) - len(y)) // 2
if pad > 0: x = x[..., pad:-pad]
return x + y
```

Per [`m1-responsibility-map.md`](m1-responsibility-map.md) §7's own warning ("37 convolutions is
enough depth for error to compound... compare layer-by-layer, not just end-to-end"), this is a
deliberate step between isolated single-op tests and the full 37-conv decoder — a real multi-op
composite, still small enough to fully control and inspect.

## Result

```text
reference output shape=(1, 32, 64)
device: Device(type=gpu,id=0), accelerator_count=1
MAX output shape=(1, 32, 64)
max|err|=4.10e-06  max_rel_err=9.91e-04  nan/inf=0
```

Shapes match exactly (the symmetric-crop guard — `if pad > 0` — matches
`_BosonResidualUnit`'s own guard, and in this configuration `pad == 0` since the dilated conv's
`(k-1)*dilation//2` padding formula already preserves length exactly, so the guard is exercised
as a no-op branch, not tested for `pad > 0` here). Numerics match an FP64 NumPy reference to
FP32-rounding level, executed on the Apple M1 GPU, ran on the first attempt after one shape-
handling fix (an initial static-shape crop implementation needed adjusting to use
`TensorValue.__getitem__` slicing directly rather than a nonexistent `ops.slice_tensor` helper).

## What this confirms

MAX graph-composition of already-validated primitives works as expected — no new numerical
surprises appear purely from chaining Snake1d and Conv1d together with a residual add. This is
consistent with (not a new finding beyond) the individual Snake1d and Conv1d prototype results;
its value is confirming composability, not discovering a new precision hazard.

## What this does not cover

- Still all-FP32, all-CPU-reference-matched. The FP16/BF16 precision questions already explored
  in isolation (Snake1d prototype) were not re-tested in this composite — a reasonable next
  addition if time permits, but not done here.
- Still missing `ConvTranspose1d` — a full `_BosonDecoderBlock` (this residual-unit ×3, plus one
  `ConvTranspose1d` upsample step) cannot be assembled end-to-end on Apple GPU until the
  Metal/`cudnnCreate` crash from `m2-convtranspose1d-results.md` is understood (T4-specific or
  general).
- Only one `pad == 0` case exercised; the crop guard's `pad > 0` branch is untested by this run.

## Next

Per the map: the T4 re-run of the ConvTranspose1d prototype remains the critical open item before
a full `_BosonDecoderBlock` (let alone the full `BosonDacDecoder`) can be attempted end-to-end on
either platform.
