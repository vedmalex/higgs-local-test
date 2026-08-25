# M4-M1 — Mojo/MAX vocoder track: CLOSED as an ANSWERED question (issue #57)

Date: 2026-08-25. Track M, task M1 of [`../audiobook/m4-plan.md`](../audiobook/m4-plan.md) §5.
This document is the closing record for the M0–M3 Mojo/MAX research line. It is a judgement
document: nothing new was run for it, and every number below is quoted from a results doc that
already exists in this repository.

**Bottom line: the question "can Higgs's Code2Wav decoder be ported to Mojo/MAX and is that worth
doing?" is now ANSWERED, and the answer is "technically yes on Apple GPU, but not worth doing" —
because the decoder is already MLX-native on the GPU, and because `codec.decode` was measured at
1.78–3.76% of wall time, capping the entire vocoder lane's achievable win at ~4%. The track is
closed as completed research, not abandoned as a failure — closed at the pipeline's *current*
AR-loop/vocoder cost ratio, not closed permanently: see §4's explicit reopen conditions, including
one tied to the AR-loop speedups other M4 tracks are already pursuing.**

---

## 1. Grounds for closure

### 1.1 The decoder is already implemented in MLX and already runs on the Apple GPU

There was nothing left to port. MLX-Audio ships a native MLX implementation of exactly the block
M0–M3 were prototyping, and it executes on every `make tts` run:

```text
.venv-tts/lib/python3.12/site-packages/mlx_audio/codec/models/higgs_audio/dac.py
  :144  class AcousticDecoder          — the whole decoder stack
  :73   class AcousticDecoderBlock     — the _BosonDecoderBlock equivalent
  :86     self.conv_t1 = nn.ConvTranspose1d(...)   (padding=stride//2)
  :9    class ResidualUnit             — dilated conv + Snake, the M2 composite
  :6    from mlx_audio.codec.models.dacvae.codec import Snake1d, WNConv1d
```

`Snake1d` is not defined in `dac.py`; it is imported from
`mlx_audio/codec/models/dacvae/codec.py`. Recorded precisely so the file/line map above is not
read as claiming more than it does.

Upstream already supplies, in MLX and on Metal, every op M2 prototyped individually
(`ConvTranspose1d`, dilated `Conv1d`, `Snake1d`) and the composite M3 assembled.

### 1.2 The quantitative ground: `codec.decode` is 1.78–3.76% of wall time

This is the decisive number and it is measured, not argued.
Source: [`m4-stage-profile-results.md`](m4-stage-profile-results.md) (M4-T1, PR #95, merged
2026-08-24), `m4_stage_profile.py`, real M1 hardware, `mx.eval()` forced at every stage boundary.

| Stage | short (~5 s audio) | long (~19 s audio) |
|---|---|---|
| AR prefill + per-frame backbone passes | 95.5% | 94.3% |
| `codec.decode` (single call) | 1.78% | 3.76% |
| everything else | 2.7% | 2.0% |
| RTF | 4.084 | 4.263 |
| AR frames | 130 | 492 |

An infinitely fast, zero-cost vocoder would take the long case from **82.87 s to ≈79.75 s** — RTF
4.26 → ≈4.10. **The ceiling on the entire vocoder lane, MAX or otherwise, is ~4% of wall time.**

`m4-plan.md` §2 declared the threshold **in advance**: "vocoder relevance < 15% of wall time → do
not touch the vocoder." Measured 1.78–3.76%. The threshold fired unambiguously, and it fired on a
number that was not chosen after seeing the result.

### 1.3 MLX executes the whole decoder natively on the M1 GPU — probed, 10/10 PASSED

The MLX probe recorded in `m4-plan.md` (§0.3 and the status block, "MLX-probe run: 10/10 PASSED")
confirmed that MLX runs every op the decoder needs on the M1 GPU natively — no CPU placement, no
MAX-style workaround:

```text
5 (stride, output_padding) pairs — the real rates (8,5,4,2,3)   1.28e-07 … 1.85e-07
M3's real decoder block, stride=5                               3.20e-06
conv1d, dilation 1 / 3 / 9                                      1.32e-05 … 1.77e-05
Snake1d                                                         4.13e-07
```

**Update (2026-08-25):** this probe's script and raw output are now committed — see
[`m4-mlx-probe-results.md`](m4-mlx-probe-results.md),
[`m4_mlx_gpu_probe.py`](m4_mlx_gpu_probe.py) / [`m4_mlx_probe_case_runner.py`](m4_mlx_probe_case_runner.py),
and the raw dump at [`m4-mlx-probe-output-m1.json`](m4-mlx-probe-output-m1.json). The numbers above
are confirmed against that JSON with no discrepancy. They are quoted here because they corroborate
a conclusion that does not depend on them; the closure rests on §1.2's committed measurement.

### 1.4 MAX's structural disadvantage on this graph is concrete, not vague

`ops.conv2d_transpose` is broken on GPU upstream, on **both** backends this project can reach:

- Metal / Apple GPU — M2's prototype hard-crashed on GPU
  ([`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md)); reported upstream as
  [modular/modular#6563](https://github.com/modular/modular/issues/6563#issuecomment-5400477726)
  and [#6726](https://github.com/modular/modular/issues/6726#issuecomment-5400479775) (the latter
  reproducing the same abort on an A10), both still open.
- The consequence M3 had to design around: `conv_t1` is **forced onto the CPU** inside an otherwise
  GPU-resident graph ([`m3-device-mixing-results.md`](m3-device-mixing-results.md) — the mixed
  placement works, but only as `GPU → transfer_to(CPU) → conv2d_transpose → transfer_to(GPU) →
  GPU`). The real decoder has **5 such blocks**, so a full-decoder MAX graph pays **10 in-graph
  device transfers per call**, plus the tensor conversions at the MLX↔MAX seam on each side of the
  call.

MLX has none of this: §1.3's probe placed `ConvTranspose1d` on the GPU with no transfer at all.

### 1.5 MAX's one genuine advantage — cross-platform reach — is not currently available

Cross-platform portability was the reason to prefer MAX over MLX in the first place. Two things
removed it from the table:

- The cross-platform requirement itself was **deferred by owner decision**
  ([`m3-t4-blocked-results.md`](m3-t4-blocked-results.md) §5, 2026-08-25).
- Independently, NVIDIA Turing (`sm_75`, Tesla T4) is **blocked upstream**: the full block aborts
  with `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16` on all
  6 seeds, exit −6 (SIGABRT). Root cause confirmed upstream —
  [#4692](https://github.com/modular/modular/issues/4692) (open since 2025-05-25),
  [#6653](https://github.com/modular/modular/issues/6653) (open since 2026-06-08), fix
  [PR #6659](https://github.com/modular/modular/pull/6659) **open and NOT merged**. No workaround
  exists on this side.

So MAX's advantage would be a *future* advantage, contingent on someone else's merge.

### 1.6 What is NOT a ground for closure

[`m0-results.md`](m0-results.md)'s "GPU BLOCKED on macOS 14.6.1" finding is **stale and is not
used here**. The host was upgraded to macOS 26.6.2, and M2/M3's GPU prototypes subsequently
**passed** on that OS ([`m2-*-results.md`](m2-snake1d-results.md),
[`m3-block-results.md`](m3-block-results.md): M3-1 device mixing, M3-5 synthetic, M3-6 real
weights, M3-7 stride-8, M3-9 BF16 — all PASSED on M1/Metal). Citing the old OS blocker as a reason
to close would be false. The closure rests on §1.2 and §1.1.

---

## 2. What the research actually produced (M0–M3 was not wasted)

| Artifact | Where | Status |
|---|---|---|
| FP64 NumPy reference for the whole `_BosonDecoderBlock` | `m3_block_reference.py` | **in active reuse** as the correctness oracle for current M4 work |
| `compare()` — per-tensor max abs/rel err, NaN/Inf, exact-zero, saturation counts, combined `atol + rtol·\|ref\|` gate | `m3_divergence.py:149` | **in active reuse**; `m4-plan.md` M4-T4 uses it as the objective pre-filter ahead of the sentiment gate |
| Real-weight extraction + weight-norm fold verification (the checkpoint stores already-folded plain weights, not `weight_g`/`weight_v`) | `m3_block_weights.py`, `m3-block-results.md` §M3-3 | reference finding about this checkpoint |
| vLLM-Omni → MAX responsibility map for Code2Wav | `m1-responsibility-map.md`, `m1-facts-*.md` | the only written map of this decoder's op inventory in the project |
| Op-level correctness prototypes (Snake1d, dilated Conv1d, ConvTranspose1d, residual unit) | `m2_*_prototype.py` + results docs | reusable per-op FP64-validated harnesses |
| A metric-design correction: plain max-relative-error is meaningless at near-zero reference elements | `m3-block-results.md` §M3-5 "Metric correction" | methodology now used project-wide |
| BF16-storage / FP32-compute characterization of the block on Metal | `m3-block-results.md` §M3-9 | Metal/M1-scoped precision finding |
| Two upstream bug reports filed | modular/modular #6563, #6726 | open upstream |
| Refuted hypothesis: the T4 `ldmatrix` abort is **not** channel-count-triggered (6/6 PASSED to 1024 channels) | `m3_ldmatrix_channel_sweep.py` | negative result, saved a bisection |

M0–M3 answered their own question with real, validated results on real checkpoint weights, for two
strides (5 and 8), in FP32 and BF16-storage, against an FP64 reference. What M4-T1 changed is not
their validity but their *relevance*.

---

## 3. What remains unverified (explicitly not claimed)

- **The MLX probe measured FP32 correctness only.** It says nothing about the decoder's speed, its
  BF16 behaviour, or running the whole block as a single fused graph — all three of which M3 did
  test on the MAX side. No MLX-vs-MAX performance comparison was ever made, and none is claimed.
- **No end-to-end MAX decoder was ever built.** M3 validated one `_BosonDecoderBlock`; the full
  5-block `AcousticDecoder` chained end-to-end was never assembled, and cross-block composability
  was explicitly deferred (`m3-block-results.md` §M3-7).
- **The "10 in-graph transfers" figure is arithmetic** (5 blocks × 2 transfers), extrapolated from
  M3's single-block measurement — not a measured count from a running 5-block graph.
- The ~4% ceiling is measured for *this pipeline as currently structured*. It would have to be
  re-measured if the AR loop were ever made dramatically faster, since the vocoder's share grows as
  the AR share shrinks.
- Nothing here claims MAX is a bad framework, or that its ConvTranspose1d defects are permanent.

---

## 4. Reopen conditions

**Updated 2026-08-25 (independent audit of `m4-stage-profile-results.md`):** the track reopens if
**either** of two independent conditions holds — the cross-platform/upstream pair below, **or** a
third, separate condition tied to the AR-loop speedup that M4's other tracks (batching,
quantization) are explicitly pursuing:

**A. Cross-platform requirement returns, and upstream MAX defects are fixed — both must hold:**

1. A real cross-platform requirement returns — i.e. this project must actually run the decoder on
   non-Apple hardware, reversing the deferral in `m3-t4-blocked-results.md` §5; **and**
2. The upstream defects are fixed and released — at minimum
   [modular/modular#6659](https://github.com/modular/modular/pull/6659) merged (unblocking
   `sm_75`), and `ops.conv2d_transpose` working on GPU (#6563 / #6726), removing the forced CPU
   placement.

**B. OR — independently of A — the AR loop is sped up by ≥4x relative to the measured baseline:**

§1.2's ~4% ceiling is not a permanent property of this pipeline; it is a ratio, computed at the AR
loop's *currently measured* cost. `codec_decode` is a fixed ~3.119 s per generation call (long
case). If the AR loop (`ar_prefill` + `ar_backbone_steps`, currently ~78 s for that same case) is
accelerated **≥4x** — which is precisely what M4's batching and quantization tracks are aiming
for — the vocoder's share of wall time crosses back above the plan's own pre-declared 15% closure
threshold (`m4-plan.md` §2: solving `codec_decode / wall = 0.15` at a fixed 3.119 s decode gives
`wall ≈ 20.8 s`, i.e. RTF ≈ 1.07, roughly a 4-6x AR-loop speedup from the measured baseline). At
that point the vocoder track must be reopened on its own merits, regardless of whether condition A
ever holds.

**A speedup argument alone does not reopen the track below the ≥4x threshold** — a smaller AR-loop
speedup still leaves the vocoder under 15% of wall time, and §1.2's ceiling still caps that lane's
value. It is specifically an AR-loop speedup **in the range M4 is already targeting** that changes
this calculus, which is why condition B is recorded explicitly rather than left as an unstated
assumption. See [`m4-stage-profile-results.md`](m4-stage-profile-results.md) ("This closure is
conditional on the current AR/vocoder cost ratio, not permanent") for the full arithmetic.

---

## 5. Consequence for any future vocoder work

**If the vocoder is ever rewritten again, it goes on MLX, not on MAX.** MLX already runs every op
of this decoder natively on the Apple GPU with no CPU placement and no seam conversions; MAX
currently cannot run `ConvTranspose1d` on any GPU this project can reach. That ordering only
changes if §4's condition 2 is satisfied.

Per `m4-plan.md` §7, further Mojo/MAX vocoder porting and rewriting the decoder are both explicit
non-goals for M4.

---

## References

- [`../audiobook/m4-plan.md`](../audiobook/m4-plan.md) §0.9, §2, §5, §7 — the closure criterion and
  the pre-declared threshold
- [`m4-stage-profile-results.md`](m4-stage-profile-results.md) — the measurement this closure rests on
- [`m3-t4-blocked-results.md`](m3-t4-blocked-results.md) — upstream Turing block, owner stop decision
- [`m3-block-results.md`](m3-block-results.md) — M3-3 … M3-9, all PASSED on M1/Metal
- [`m3-device-mixing-results.md`](m3-device-mixing-results.md) — the forced CPU placement of `conv_t1`
- [`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md) — the GPU crash and the upstream reports
- [`m0-results.md`](m0-results.md) — superseded on the GPU-availability point (see §1.6)
- [`m4-mlx-probe-results.md`](m4-mlx-probe-results.md) — the committed MLX GPU correctness probe (10/10 PASSED) referenced in §1.3
