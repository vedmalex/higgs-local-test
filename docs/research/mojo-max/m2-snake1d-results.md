# M2 prototype #1 — Snake1d in MAX: PASSED on M1 and T4, confirms the explicit-cast precision policy

Date: 2026-08-24. Run: `docs/research/mojo-max/m2_snake1d_prototype.py`, on this project's M1
(Apple Silicon, Metal GPU — `Device(type=gpu,id=0)`, `accelerator_count=1`), via the
`.mojo-probe-stable` pixi env already used for M0. Per the issue's "run locally first" guidance,
this validates on M1 before a T4 run; the script is unmodified between targets.

This is the first M2 prototype recommended by
[`m1-responsibility-map.md`](m1-responsibility-map.md) §11: `snake(x) = x + (alpha+1e-9)^-1 *
sin(alpha*x)^2` needs no missing MAX op, so a failure is unambiguously about precision, not a
missing primitive.

## Setup

`x` shape `[1,32,256]`, uniform(-3,3), fixed seed (`np.random.default_rng(1234)`). `alpha` shape
`[1,32,1]`, mostly uniform(0.05,1.5), with 6 channels deliberately set to `1e-3, 1e-4, 1e-5, 1e-6,
1e-7, 5e-7` — the last three exceed FP16's `1/x` finite range once `x` is that small
(`1/1e-6 = 1e6 > 65504`). Reference: FP64 NumPy (stands in for an FP32 PyTorch forward pass
without adding torch as a dependency to this standalone probe).

**Methodological note, recorded honestly**: the first run of this prototype used only
`1e-3/1e-4`-scale alpha values and showed *no* FP16 failure (max abs err 0.0044, no NaN/Inf) —
because `1/1e-4 = 1e4` is comfortably inside FP16's range. That result would have been a false
"Snake1d is fine in FP16" conclusion. Re-checked against the actual regime M0's finding was about
(`1/alpha` genuinely exceeding 65504) before drawing any conclusion — see numbers below.

## Results

```text
device: Device(type=gpu,id=0), accelerator_count=1
[fp32]                                                  max|err|=5.82e-07  max_rel_err=5.35e-07  nan/inf=0     exact_zeros=0
[fp16 storage, fp32 compute (explicit cast, per S9)]    max|err|=3.08e-03  max_rel_err=1.99e-03  nan/inf=0     exact_zeros=0
[fp16 storage, fp16 compute (no cast)]                  max|err|=nan       max_rel_err=nan       nan/inf=1024  exact_zeros=0
```

## What this confirms

1. **The MAX graph toolchain works end-to-end on this build**: graph construction (`ops.add`,
   `ops.mul`, `ops.div`, `ops.pow`, `ops.sin`, `ops.cast`, `ops.constant`), compilation, and
   execution on the Apple M1 GPU via `InferenceSession`/`Buffer` all succeeded on the first
   correctly-scoped attempt. No custom Mojo kernel was needed, matching the responsibility map's
   prediction.
2. **FP32 end-to-end matches the reference to ~6e-7** — normal FP32 rounding, no surprises. This
   is the recommended M2 baseline per §9 ("build in FP32 first").
3. **FP16 storage with an explicit FP32 compute cast (the §9 policy) works**: 0.003 max abs error
   is unremarkable FP16-rounding-on-store error, zero NaN/Inf, zero silent zeros.
4. **FP16 storage with FP16 compute (no cast) fails completely, not partially**: 1024 of 8192
   output elements are NaN or Inf — not a small precision degradation, a hard break. This is the
   exact failure class M0 predicted (`1/(alpha+eps)` overflowing FP16's 65504 ceiling before the
   `sin(...)^2` term is even applied) and directly supports the #48 hypothesis that an unguarded
   FP16 Snake activation is a plausible root cause, *if* the real T4 run is shown (via E2) to
   actually run this op in FP16 rather than the checkpoint's native BF16.

## What this does not show

- This is a synthetic input/alpha distribution, not the real Higgs checkpoint's actual `alpha`
  values. Whether any real trained `alpha` is small enough to hit this regime is unverified —
  E2 (capturing real per-layer activations/parameters from the actual checkpoint) is the next
  step to check, not assumed here.
- **Done below (T4 result)**: this ran on Apple M1 GPU (Metal) first; it has now also been run
  unchanged on a Colab T4 (CUDA) and the results match, closing this gap.
- This tests Snake1d in isolation, not inside the real 37-instance conv stack — compounding
  effects across layers are unmeasured.

## T4 result (2026-08-24, via `notebooks/mojo_max_m0_t4.ipynb`-style Colab T4 run)

Full raw output: [`m2-snake1d-output-t4.txt`](m2-snake1d-output-t4.txt).

```text
device: Device(type=gpu,id=0), accelerator_count=1
[fp32] max|err|=1.62231e-06 max_rel_err=1.38224e-06 nan/inf=0 exact_zeros=0
[fp16 storage, fp32 compute (explicit cast, per S9)] max|err|=0.00307625 max_rel_err=0.00199181 nan/inf=0 exact_zeros=0
[fp16 storage, fp16 compute (no cast -- expected to break)] max|err|=nan max_rel_err=nan nan/inf=1024 exact_zeros=0
```

**This matches the M1 results exactly, case by case:**

| Case | M1 max abs err | T4 max abs err | M1 max rel err | T4 max rel err | M1 nan/inf | T4 nan/inf |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5.82e-07 | 1.62e-06 | 5.35e-07 | 1.38e-06 | 0 | 0 |
| fp16 storage, fp32 compute | 3.08e-03 | 3.08e-03 | 1.99e-03 | 1.99e-03 | 0 | 0 |
| fp16 storage, fp16 compute (no cast) | nan | nan | nan | nan | 1024 | 1024 |

The fp32 case differs only within normal run-to-run FP rounding variation (both ~1e-6, same
order of magnitude — consistent with the bit-for-bit-except-rounding-order comparability M0
already established between M1 and T4). The fp16-storage/fp32-compute case matches to the
displayed precision (3.08e-03 / 1.99e-03 on both). The no-cast fp16 failure is identical on both
platforms: exactly 1024 of 8192 elements are NaN/Inf on both M1 and T4. **Conclusion: the
explicit-cast precision policy (fp16 storage + fp32 compute) and the no-cast overflow failure
both hold identically on T4, not just on M1** — this closes the "re-run on T4 before treating as
portable evidence" caveat above.

## Next

- Per `m1-responsibility-map.md` §11, the second prototype is one weight-normed dilated Conv1d
  via `ops.conv2d` with a degenerate height axis (route A) — the single largest structural
  question (can Higgs's conv shape be expressed in MAX without a custom Mojo kernel). Also now
  confirmed on T4 — see [`m2-conv1d-results.md`](m2-conv1d-results.md).
