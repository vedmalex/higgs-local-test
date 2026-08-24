# M3-2 — reusable per-layer divergence detector: verified against the M2 residual-unit number

Date: 2026-08-24. Closes E5 from `m1-responsibility-map.md`.

## What was built

`docs/research/mojo-max/m3_divergence.py` — one importable module (`compare(got, ref,
orig_dtype=None)` -> `DivergenceReport`) that generalizes the ad-hoc three-line comparison each
M2 prototype hand-rolled at the bottom of its `__main__` block (see
`m2_residual_unit_prototype.py`'s final lines: inline `abs_err`/`rel_err`/NaN-count only). The
report always includes:

- `max_abs_err`, `max_rel_err`
- `nan_count`, `inf_count`
- `exact_zero_count` (on `got`) and `exact_zero_count_ref` (on `ref`) — first-class, not derived
  from the NaN/Inf scan, per §11/E5's warning that a NaN/Inf-only scan reports a fully-zeroed
  tensor as healthy (the #48 failure shape)
- `saturation_count` — values at/beyond the *original* dtype's representable ceiling (e.g.
  FP16's 65504), only populated when `orig_dtype` is passed in (otherwise `None`, distinct from
  "checked and found zero")
- `is_healthy()` convenience gate that explicitly fails the all-zero-got-vs-nonzero-ref case even
  when NaN/Inf/abs-err would otherwise look fine

## Reproduction of the published number

Command run (from `.mojo-probe-stable`, the pixi env pinned to `modular >=26.5.0,<27`):

```
arch -arm64 pixi run python /private/tmp/.../scratchpad/m3_2_repro.py
```

This script imports `m2_residual_unit_prototype.py`'s own functions unmodified
(`fold_weight_norm`, `numpy_residual_unit`, `build_graph`) with the same seed (3141) and shapes,
runs the same MAX graph on the same M1 GPU device, and replaces only the final ad-hoc comparison
snippet with a call to `m3_divergence.compare()`.

Actual output:

```text
reference output shape=(1, 32, 64)
device: Device(type=gpu,id=0), accelerator_count=1
MAX output shape=(1, 32, 64)
m3_divergence report: max|err|=4.09571e-06 max_rel_err=0.000991201 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=2048
PUBLISHED (m2-residual-unit-results.md): max|err|=4.10e-06 max_rel_err=9.91e-04 nan/inf=0
```

`4.09571e-06` rounds to `4.10e-06` at the 3-significant-figure precision the published doc
reports it at; `9.91201e-04` likewise rounds to the published `9.91e-04`. NaN/Inf/exact-zero
counts are all 0, consistent with "PASSED" in the original doc. **Reproduced, no discrepancy.**

## Self-check

`m3_divergence.py` also runs standalone (`arch -arm64 pixi run python m3_divergence.py`) with
three synthetic cases: a normal near-match, an all-zero-`got`-vs-nonzero-`ref` case (asserts
`is_healthy()` is `False` — the #48 shape), and an FP16-saturation case (asserts
`saturation_count == 2`). All three assertions pass.
