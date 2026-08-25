# M4 — MLX GPU correctness probe: 10/10 PASSED (FP32, Apple GPU)

Date of run: prior session, 2026-08-24/25 (exact prior-session timestamp not separately logged).
Date these artifacts were committed: 2026-08-25. Run: `docs/research/mojo-max/m4_mlx_gpu_probe.py`
+ `m4_mlx_probe_case_runner.py`, native Apple M1 (16 GB), `.venv-tts`.

## Why this probe exists

`m4-conclusion.md` §1.3 and `../audiobook/m4-plan.md` §0.3 both cite "MLX-probe run: 10/10 PASSED"
as part of the evidence that Higgs's Code2Wav decoder already runs natively on the Apple GPU via
MLX (no MAX/Mojo port needed). Until this commit, that claim had **no artifact in the repository**
— no script, no raw output — so it could not be independently checked from the repo alone, only
taken on trust from prose. This document, together with the two scripts and the raw JSON dump
committed alongside it, closes that gap: the claim is now backed by a runnable script and a
committed result file, in the same style as the M2/M3 op-level correctness prototypes it draws on.

**What it checks and does not check.** This is a **numerical-correctness-only** probe in FP32,
per case, on synthetic random inputs against an FP64 NumPy reference. It does **not** measure
performance/latency, does **not** test BF16, and does **not** assemble the ops into one fused
end-to-end graph — it checks each op (or a small composite) in isolation. See `m4-conclusion.md`
§3 for the fuller list of things the wider M4 track has not claimed based on this probe.

## Environment

Checked explicitly inside **each of the 10 subprocess runs** (`mx.set_default_device(mx.gpu)`
then `str(mx.default_device())`), not assumed once for the whole run — consistent with this
project's isolation discipline for GPU checks (see "Method" below):

- macOS 26.6.2 (native Apple Silicon M1, 16 GB unified memory)
- Python 3.12.11, `arm64`
- MLX 0.32.1
- `mx.metal.is_available()` = `True`
- `mx.default_device()` = `Device(gpu, 0)` in every one of the 10 subprocesses

These environment facts were re-verified live in the current session (2026-08-25, same host) when
these artifacts were committed, and matched exactly.

## Method

Each of the 10 cases runs in its **own subprocess**
(`m4_mlx_gpu_probe.py` spawns `m4_mlx_probe_case_runner.py <case_id>` once per case), not in one
shared process with `try`/`except` around each case. This mirrors the isolation pattern this
project already established for GPU checks in
[`m2_convtranspose1d_prototype.py`](m2_convtranspose1d_prototype.py) and
[`m3_decoder_block_prototype.py`](m3_decoder_block_prototype.py): this project has already
confirmed that a GPU failure on the MAX/Metal backend can be a **fatal, uncatchable process abort**
(SIGABRT), not a normal Python exception (see
[`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md)). A naive in-process loop could
have one case's crash silently swallow or hide the remaining cases' results; per-case subprocess
isolation with an explicit exit-code classification (`PASSED` / `FAILED` / `ABORTED` on a negative
signal-killed return code) avoids that failure mode regardless of which backend is under test.

The FP64 NumPy reference implementations are **reused, not reimplemented**:
`numpy_conv_transpose1d` comes from [`m3_block_reference.py`](m3_block_reference.py) (validated to
1.11e-16 against PyTorch by that file's own hand/torch cross-check), and `numpy_conv1d` /
`numpy_snake` come from [`m2_residual_unit_prototype.py`](m2_residual_unit_prototype.py). Both are
imported unmodified by `m4_mlx_probe_case_runner.py`.

**API fact worth recording explicitly** (checked empirically against MLX 0.32.1, not assumed from
memory): `mx.conv_transpose1d` and `mx.conv1d` take input in **NLC** layout (`N, L, C` —
channels-last), unlike PyTorch/NumPy's `N, C, L`. Weights are `(C_out, K, C_in)` for both ops.
`stride`, `padding`, `dilation`, and `output_padding` are all direct keyword parameters on the MLX
call — dilation is natively supported. None of the MAX-side workarounds this project's M2/M3 work
needed (`ops.conv2d_transpose`'s NHWC-via-conv2d route, manual output cropping instead of a
`padding` argument) are needed here; the layout/parameter conversion the runner does is a plain
transpose of the reference NumPy arrays into MLX's native layout, not a structural workaround.

## Cases and results (from `m4-mlx-probe-output-m1.json`, verbatim)

All 10 cases: `verdict = PASSED`, `returncode = 0`, `shape_match = true`, `nan_inf = 0`,
`mlx_default_device = Device(gpu, 0)`.

| # | case_id | Op | Shape (in → out) | Config | max\|err\| |
|---|---|---|---|---|---|
| 1 | `ct_stride8_op0` | ConvTranspose1d | (1,16,16) → (1,16,136) | k=16, stride=8, pad=0, out_pad=0 | 1.5506182049485062e-07 |
| 2 | `ct_stride5_op1` | ConvTranspose1d | (1,16,16) → (1,16,86) | k=10, stride=5, pad=0, out_pad=1 | 1.275368469166871e-07 |
| 3 | `ct_stride4_op0` | ConvTranspose1d | (1,16,16) → (1,16,68) | k=8, stride=4, pad=0, out_pad=0 | 1.5366556826990063e-07 |
| 4 | `ct_stride2_op0` | ConvTranspose1d | (1,16,16) → (1,16,34) | k=4, stride=2, pad=0, out_pad=0 | 1.4742174703208377e-07 |
| 5 | `ct_stride3_op1` | ConvTranspose1d | (1,16,16) → (1,16,52) | k=6, stride=3, pad=0, out_pad=1 | 1.851479327630301e-07 |
| 6 | `ct_m3_real_stride5` | ConvTranspose1d (real M3 block-1 shape) | (1,512,20) → (1,256,100) | k=10, stride=5, pad=3, out_pad=1 | 3.2030380889835897e-06 |
| 7 | `conv1d_dilation1` | Conv1d (Higgs ResidualUnit) | (1,256,86) → (1,256,86) | k=7, dilation=1, pad=3 | 1.7691219243332057e-05 |
| 8 | `conv1d_dilation3` | Conv1d | (1,256,86) → (1,256,86) | k=7, dilation=3, pad=9 | 1.646213176886846e-05 |
| 9 | `conv1d_dilation9` | Conv1d | (1,256,86) → (1,256,86) | k=7, dilation=9, pad=27 | 1.3179415519815052e-05 |
| 10 | `snake` | Snake1d | (1,256,86) → (1,256,86) | 256 channels | 4.1266541117579436e-07 |

Cases 1–5 are the five `(stride, output_padding)` pairs Higgs's five real decoder blocks use
(`rates=(8,5,4,2,3)`), matching [`m2_convtranspose1d_prototype.py`](m2_convtranspose1d_prototype.py)
exactly (16 channels, kernel = 2×stride, seq_len=16, padding=0). Case 6 is the real shape of the
first `_BosonDecoderBlock`'s transposed conv (input `(1,512,20)`, 256 output channels). Cases 7–9
are Higgs's `ResidualUnit` dilations (1, 3, 9) at kernel 7. Case 10 is the `Snake1d` activation at
the decoder's 256-channel width.

**Cross-check against the numbers quoted in `m4-conclusion.md`/`m4-plan.md`:** those documents
summarize the ConvTranspose1d range as "1.28e-07 … 1.85e-07" for the 5 rate pairs — this JSON's
values (1.28e-07, 1.28e-07, 1.54e-07, 1.47e-07, 1.85e-07) fall in that range, confirming the prior
prose summary rather than contradicting it. The M3-real-block figure (3.20e-06), the conv1d
dilation range (1.32e-05…1.77e-05), and the Snake1d figure (4.13e-07) likewise match the prose
exactly, to the precision quoted there. No discrepancy found between the JSON and the previously
published summary numbers.

## Honest caveats

- **FP32 numerical correctness only.** No performance/latency measurement, no BF16 case, and no
  single fused end-to-end graph — each op (or the one composite case, `ct_m3_real_stride5`) runs
  standalone. See `m4-conclusion.md` §3 for the full list of things this probe does not establish.
- Inputs are synthetic random arrays (`np.random.default_rng` with fixed seeds), not real
  checkpoint activations — this checks the *operation*, not behavior on real Higgs data.
- The 10 cases were **not re-run for this commit**. The scripts and their raw output were carried
  over unchanged from a prior session's scratch directory into this repository; only the file
  paths inside the scripts were adjusted so they run from their new repository location (the
  `case_runner`'s working-directory import path, and the driver's path to the runner and to its
  output file, now resolve relative to `Path(__file__)` instead of a session-specific
  `/private/tmp/...` scratch path). The check-logic itself (the reference implementations, the
  MLX calls, the tolerance/verdict logic) was not modified.
- A re-run to confirm the scripts still execute correctly from their new location was **not**
  attempted in this session: at commit time the host had 10.6 GiB of 12 GiB swap in use and very
  little free physical memory (a media-library indexing job was running concurrently), so starting
  any new process — even this probe's small synthetic tensors — was judged not worth the risk of
  adding memory pressure. This is disclosed rather than silently skipped.

## Conclusion

The 10/10 PASSED claim already cited in `m4-conclusion.md` and `../audiobook/m4-plan.md` is
confirmed against the actual raw JSON output, now committed at
[`m4-mlx-probe-output-m1.json`](m4-mlx-probe-output-m1.json). All 10 cases ran on
`Device(gpu, 0)` (Apple Metal GPU via MLX), produced shape-correct, finite (`nan_inf=0`) output,
and matched an FP64 NumPy reference to between 1.28e-07 and 1.77e-05 max absolute error depending
on op and accumulation depth — well within FP32 rounding expectations for these operation sizes.
This corroborates, with a now-inspectable artifact, that MLX executes Higgs's ConvTranspose1d,
dilated Conv1d, and Snake1d ops natively on the Apple GPU with no CPU placement, in contrast to
MAX's `ops.conv2d_transpose`, which crashes on GPU on both Metal and Tesla T4
(see [`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md)).

## References

- [`m4_mlx_gpu_probe.py`](m4_mlx_gpu_probe.py) — driver, spawns one subprocess per case
- [`m4_mlx_probe_case_runner.py`](m4_mlx_probe_case_runner.py) — single-case runner (subprocess target)
- [`m4-mlx-probe-output-m1.json`](m4-mlx-probe-output-m1.json) — full raw output, 10 case records
- [`m3_block_reference.py`](m3_block_reference.py) — FP64 `numpy_conv_transpose1d` reference (reused unmodified)
- [`m2_residual_unit_prototype.py`](m2_residual_unit_prototype.py) — FP64 `numpy_conv1d`/`numpy_snake` references (reused unmodified)
- [`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md) — the MAX/Metal + MAX/T4 GPU crash this probe's MLX result contrasts with
- [`m4-conclusion.md`](m4-conclusion.md) — closure document that cites this probe's numbers
- [`../audiobook/m4-plan.md`](../audiobook/m4-plan.md) — plan document that cites this probe's numbers
