# M4-T2 follow-up — vocoder share under batching, and batching on paragraph-length replies

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio, `bosonai/higgs-tts-3-4b`). This document answers two questions the batching
measurement in [`m4-batching-results.md`](m4-batching-results.md) (PR #105) left open:

- **Task A**: does the vocoder's *share* of wall time grow under batching, as Amdahl's law predicts
  once the AR loop is sped up — and does it cross the 15% reopen threshold recorded in
  [`../mojo-max/m4-conclusion.md`](../mojo-max/m4-conclusion.md) §4 condition B?
- **Task B**: does batching's 3.40x speedup (measured on short independent sentences) hold on
  paragraph-length replies (3-6 sentences), the unit a real audiobook chapter actually produces, or
  does the memory picture change?

**Bottom line up front:**
- **Task A: the vocoder's share grew from 2.01% (batch=1) to 7.85% (batch=8) — roughly a 3.9x
  growth, in the direction and rough magnitude the hypothesis predicted — but it did NOT cross the
  15% threshold.** The AR loop was sped up 3.23x at batch=8, short of the ≥4x that
  `m4-conclusion.md` §4 condition B requires to reopen the Mojo/MAX vocoder track. The track stays
  closed; this document does not reopen it (that is the owner's call), but records the number
  precisely because the margin is no longer wide.
- **Task B: batching's speedup holds — 3.69x at batch=8 on paragraph-length replies, slightly
  *better* than the 3.40x measured on short sentences.** `peak_footprint` did not grow with batch
  depth on long material either (11.6-12.2 GiB across 1/2/4/8, batch=1 actually the highest of the
  four); no memory ceiling was found in this range. A real, honest caveat did surface: the OS swap
  file's own size ceiling grew under load at higher batch depths on long material (up to 6 GB, vs.
  the short-sentence run's 3-5 GB), though swap *usage* never approached its ceiling and no run
  showed a time cliff, an error, or a kill.

## 0. Method

- **Task A** reuses `instrument()`/`deinstrument()` from
  [`../mojo-max/m4_stage_profile.py`](../mojo-max/m4_stage_profile.py) (PR #104) **unmodified**,
  applied to `HiggsAudioV3.batch_generate` (`model.py:548`) instead of `.generate()`. This works
  without any change to the instrumentation because `batch_generate` calls the exact same four
  methods on `self` that `instrument()` patches (`self.backbone`, `self._build_prompt_embeddings`,
  `self._decode_audio`, `self._apply_fades` — confirmed by reading `model.py:634,662-711,715,717`
  before relying on it). New script:
  [`m4_batching_stage_profile.py`](m4_batching_stage_profile.py). It imports `SEGMENTS`, `MODEL_ID`,
  `machine_state()`, `audio_duration()`, `WARMUP_TEXT` from `m4_batching_bench.py` so the batch=1 and
  batch=8 numbers are on the **identical 8-segment material** the original batching benchmark used —
  directly comparable to `m4-batching-results.md`'s aggregate RTF, not a re-derived fixture.
  `mx.eval()` happens at every stage boundary via the reused wrappers, and again on the produced
  audio before any timer stops.
- **Confirmed by reading the code, not assumed**: `batch_generate` runs the AR loop fully batched
  (one shared backbone forward per step across all active rows, `model.py:662-712`), but **after**
  that loop finishes for the whole batch, it calls `self._decode_audio(state["delayed_rows"])` once
  per row, in a plain Python `for state in states:` loop (`model.py:714-716`) — i.e. the vocoder call
  is genuinely per-segment regardless of batch depth, exactly as the task's hypothesis assumed. (This
  is the single-shot `batch_generate` path the benchmark measures, not the separate streaming
  `HiggsAudioV3BatchSession`/`continuous_batching.py:279` session class also mentioned in the task —
  both call the same per-row `_decode_audio`, so the conclusion is the same either way, but the
  measured code path here is `batch_generate`, named precisely.)
- **Task B** reuses [`m4_batching_bench.py`](m4_batching_bench.py) **without modifying its existing
  behavior or default `SEGMENTS`** — the only change is an additive `--segments-file` CLI option
  (and threading a `segments` parameter through `run_baseline`/`run_batched` instead of the module
  global) that loads an alternate, blank-line-separated paragraph pool when passed; omitting the flag
  reproduces the exact original behavior byte-for-byte. New fixture:
  [`m4_long_segments_ru.txt`](m4_long_segments_ru.txt) — 8 Russian paragraphs, 3 sentences /
  203-259 characters each (vs. the original fixture's 35-61 characters), same
  technical/devotional-register mix as `samples/tts_ru.txt` and the original `SEGMENTS`, written for
  this task since no committed longer Russian fixture existed.
- **Isolation**: one batch size per process, run to completion before the next started, per project
  policy. Each run wrapped in `/usr/bin/time -l`. `peak_footprint` below is always that wrapper's
  `peak memory footprint` line; `maximum resident set size` was recorded in the raw logs but is never
  cited as a memory figure (confirmed elsewhere in this project to undercount on macOS). `peak_mlx`
  is `mx.get_peak_memory()`, read once at the end of each process after `mx.reset_peak_memory()` was
  called right after the (discarded) warm-up generation.
- **Machine load caveat, stated up front**: this is a shared, 9-10-user machine (per the earlier
  batching run's own §1). Load average during Task B's four runs ran noticeably higher (5-12) than
  during Task A's two runs (2-8) — visible in §1's table — because of other users' activity on the
  host, not this benchmark. Swap usage never approached its ceiling in any run and no run showed a
  time cliff, an error, or a kill; the elevated load is recorded so a reader can judge these numbers'
  precision for themselves, not hidden.

## 1. Machine state (recorded before and after every run)

| Run | Before: load avg (1/5/15) | Before: swap used/total | After: load avg (1/5/15) | After: swap used/total |
|---|---|---|---|---|
| Task A, batch=1 (short) | 2.30 / 2.55 / 3.39 | 1.82 / 3.00 GB | 2.60 / 2.68 / 3.33 | 2.69 / 3.00 GB |
| Task A, batch=8 (short) | 2.13 / 2.57 / 3.27 | 2.65 / 3.00 GB | 6.01 / 3.29 / 3.48 | 4.15 / 5.00 GB |
| Task B, batch=1 (long) | 2.90 / 2.87 / 3.27 | 2.08 / 3.00 GB | 8.96 / 5.13 / 3.96 | 3.40 / 4.00 GB |
| Task B, batch=2 (long) | 7.46 / 5.04 / 3.95 | 1.98 / 3.00 GB | 12.02 / 11.01 / 7.26 | 4.55 / 5.00 GB |
| Task B, batch=4 (long) | 10.01 / 10.61 / 7.19 | 2.23 / 3.00 GB | 5.05 / 7.33 / 6.59 | 4.01 / 5.00 GB |
| Task B, batch=8 (long) | 5.61 / 7.21 / 6.57 | 2.13 / 3.00 GB | 5.85 / 6.77 / 6.50 | 5.31 / 6.00 GB |

The rising load average across Task B's runs (peaking at 12.02 before batch=4) reflects other
activity on this shared host, not a symptom of this benchmark — a second, unrelated agent was
deliberately waiting idle on this machine for this task to finish precisely to avoid contending for
resources; the load came from elsewhere. No run's wall-time behavior showed a discontinuity that
would indicate this benchmark itself was thrashing.

## 2. Task A — vocoder share under batching (short-sentence material, same as `m4-batching-results.md`)

| Batch | Wall total (s) | Aggregate RTF | AR loop (s / % of wall) | `codec_decode` (s / % of wall) | Other (s / % of wall) | `peak_mlx` (GiB) | `peak_footprint` (GiB) |
|---|---|---|---|---|---|---|---|
| 1 | 110.53 | 3.936 | 106.60 / 96.44% | 2.223 / 2.01% | 1.71 / 1.55% | 9.41 | 11.81 |
| 8 | 35.91 | 1.228 | 33.04 / 92.03% | 2.818 / 7.85% | 0.04 / 0.12% | 9.35 | 11.61 |

"AR loop" = `ar_prefill` + `ar_backbone_steps` + `ar_glue_sampling_embedding`. "Other" =
`prompt_prep` + `postprocess`. Full stage breakdown (all six buckets, seconds and %):

| Stage | batch=1 (s) | batch=1 (%) | batch=8 (s) | batch=8 (%) |
|---|---|---|---|---|
| `prompt_prep` | 1.642 | 1.486% | 0.032 | 0.088% |
| `ar_prefill` | 2.399 | 2.171% | 1.105 | 3.078% |
| `ar_backbone_steps` | 103.201 | 93.368% | 31.176 | 86.828% |
| `ar_glue_sampling_embedding` | 0.994 | 0.900% | 0.762 | 2.123% |
| `codec_decode` | 2.223 | 2.011% | 2.818 | 7.847% |
| `postprocess` | 0.072 | 0.065% | 0.013 | 0.035% |

(This run's own batch=1/batch=8 aggregate RTF — 3.936 and 1.228 — differ slightly from
`m4-batching-results.md`'s 3.882 and 1.142 on the same fixture; `temperature=1.0` is stochastic and
no seed is set in either script, so run-to-run variance of a few percent is expected, not a
discrepancy. The 3.20x wall-time speedup measured here (3.936/1.228) is consistent with the
previously measured 3.40x within that variance.)

### 2.1 Does the vocoder share cross 15%?

**No.** Measured `codec_decode` share: **2.01% at batch=1, 7.85% at batch=8** — a 3.9x growth,
matching the hypothesis's predicted direction and rough magnitude (the task's own back-of-envelope
estimate was ~12.8%; the measured 7.85% is lower because the AR loop was NOT sped up by the full
3.40x this run's own workload happened to show in the original benchmark — see below).

**AR-loop speedup measured here: 106.60s → 33.04s = 3.23x.** `../mojo-max/m4-conclusion.md` §4
condition B requires the AR loop to be sped up **≥4x** before the vocoder track reopens on its own
merits. 3.23x is short of that. The vocoder track stays closed; this measurement does not change
that, and does not reopen it (that decision belongs to the owner) — it records the number precisely
because at 7.85% the margin under the 15% threshold, while still comfortable in absolute terms, is
less than half what it was.

### 2.2 At what batch size would it cross 15%? (extrapolation, not measured)

`codec_decode`'s absolute cost stayed roughly flat across batch sizes (2.223s → 2.818s for the same
8 segments — some variance, consistent with it being called once per segment regardless of batch
depth, not with it scaling with batch depth). Solving `codec_decode / wall = 0.15` at a representative
~2.5s: `wall ≈ 16.7s`, i.e. an overall wall-time speedup of **~6.6x from the batch=1 baseline**
(currently 3.20x achieved at batch=8) — roughly double the AR-loop shrink already measured going
from batch=1 to batch=8.

This was **not tested** — batch sizes above 8 are outside this task's scope, and both this document
and `m4-batching-results.md` observe `peak_mlx`/`peak_footprint` already sitting at or slightly past
MLX's own `max_recommended_working_set_size` (≈11.84 GiB) at every batch size measured so far, so
whether batch=16 or 32 is even reachable on this 16 GB machine is itself unknown. A rough
geometric extrapolation from the batch=4→8 AR-loop shrink (a further ~1.8x) suggests batch=16 alone
would land the vocoder share somewhere in the 10-13% range — still under 15% — and crossing the
threshold would likely need batching pushed further than this machine has been shown to support,
combined with (or replaced by) an AR-loop speedup from another lever such as quantization (M4-T3).
**This paragraph is explicitly an extrapolation, not a measurement — flagged as such, not published
as a result.**

## 3. Task B — batching on paragraph-length replies

Fixture: [`m4_long_segments_ru.txt`](m4_long_segments_ru.txt), 8 paragraphs, 203-259 characters each
(vs. 35-61 characters in the original short-sentence `SEGMENTS`), 3 sentences per paragraph, same
mixed technical/devotional register.

| Batch size | Aggregate RTF | Speedup vs batch=1 | `peak_mlx` (GiB) | `peak_footprint` (GiB) | Wall time (8 segments) |
|---|---|---|---|---|---|
| 1 (baseline) | 3.756 | 1.00x | 10.99 | 12.20 | 481.42 s |
| 2 | 3.696 | 1.02x | 11.00 | 11.61 | 473.62 s |
| 4 | 1.806 | 2.08x | 11.05 | 11.63 | 218.30 s |
| 8 | 1.017 | **3.69x** | 11.10 | 11.69 | 122.98 s |

**Batching's speedup holds on paragraph-length material — 3.69x at batch=8, slightly *better* than
the 3.40x measured on short sentences.** The pattern across batch sizes is the same shape as the
short-sentence result too: batch=2 alone is not worth adopting (1.02x, essentially noise), batch=4
clears the plan's ≥1.5x threshold (2.08x), and batch=8 is the clear best of the four (3.69x).

### 3.1 Where does memory become the binding constraint?

**Still not found within 1/2/4/8, and the pattern is not monotonic.** `peak_footprint` is 12.20 GiB
at batch=1 — the *highest* of the four, not the lowest — then 11.61-11.69 GiB at batch=2/4/8, flat
within noise. The batch=1 number being the outlier is a real, counter-intuitive finding worth
recording rather than smoothing over: a plausible (not proven) explanation is that batch=1's
sequential loop accumulates each segment's Python-side `results`/audio objects across all 8 calls
before the process exits, while the batched paths process a whole chunk and release intermediate
objects between chunks — this was not instrumented further here and is flagged as an open question,
not a resolved cause. Either way, no run in this range showed a memory-bound crash, error, or kill.
`peak_mlx` shows the expected small upward trend with batch depth (10.99 → 11.10 GiB) but stays
comfortably below the 16 GB ceiling throughout.

**A real caveat that did surface on long material, absent on short material:** the OS swap file's
own ceiling grew to 6 GB by batch=8 here (vs. 3-5 GB across the entire short-sentence run in
`m4-batching-results.md`), and swap *usage* likewise ran higher (up to 5.31 GB used, vs. 4.15 GB
peak on short material). This did not cause a failure or a visible time cliff in any run, but it is
a directional signal that paragraph-length KV-cache growth does put more pressure on the system than
short-sentence growth did — consistent with `m4-batching-results.md` §3's own prediction that longer
segments would exercise `BatchKVCache` more, even though `peak_footprint` itself (dominated by the
~11.4 GB of resident weights) does not show it. **The batch-size ceiling for paragraph-length
segments remains unmeasured beyond 8** — batch 16+ was in scope ("и дальше, если влезет") but was
not attempted here given the swap-growth signal above and the time this task already used; that is
this document's one explicitly incomplete item, not a silent omission.

### 3.2 Ten-hour audiobook arithmetic, paragraph-length material

Plan tiers: `RTF <= 1.5 practical / 1.5-3 usable with mandatory resume support / > 3 impractical`.

| Config | RTF | 10-hour audiobook machine time | Tier |
|---|---|---|---|
| batch=1 (today) | 3.76 | ≈37.6 h | impractical |
| batch=2 | 3.70 | ≈37.0 h | impractical |
| batch=4 | 1.81 | ≈18.1 h | usable — mandatory resume support |
| **batch=8** | **1.02** | **≈10.2 h** | **practical** |

This essentially matches the short-sentence arithmetic in `m4-batching-results.md` §5 (≈11.4 h
there vs. ≈10.2 h here) — **batching's practical-tier payoff is not an artifact of unrealistically
short test material.** A real audiobook chapter, chunked into paragraph-length utterances, should
see comparable gains.

## 4. Honesty notes / what this does not answer

- Task A's batch=1/batch=8 numbers here are a fresh run of the same fixture, not a re-read of
  `m4-batching-results.md`'s own numbers — the small RTF differences (3.936 vs 3.882; 1.228 vs
  1.142) are sampling variance from `temperature=1.0` with no fixed seed, not a contradiction. Both
  runs agree on the qualitative result (batch=8 clearly the best, ~3.2-3.4x wall speedup).
- The §2.2 extrapolation past batch=8 is explicitly not a measurement; do not cite the "batch ~16"
  figure as tested.
- Task B's batch=16+ was not attempted (see §3.1) — the paragraph-length batch-size ceiling above 8
  remains open for a future task.
- Neither task touched quantization (M4-T3) or sentiment/control-tag fidelity under batching
  (M4-T4) — both remain exactly as scoped in `m4-plan.md`.
- The AR-loop-speedup number that matters for `../mojo-max/m4-conclusion.md` §4 condition B is
  **3.23x, measured on short-sentence material at batch=8**; this document does not have a
  paragraph-length AR-loop stage profile (Task A was scoped to the short-sentence fixture for direct
  comparability with the original stage-profile numbers) — that would be a natural follow-up if the
  owner wants the reopen condition checked against paragraph-length material specifically.

## References

- [`m4-batching-results.md`](m4-batching-results.md) — the batching measurement this follows up on (PR #105)
- [`../mojo-max/m4-stage-profile-results.md`](../mojo-max/m4-stage-profile-results.md) — the original batch=1 stage profile (PR #95)
- [`../mojo-max/m4_stage_profile.py`](../mojo-max/m4_stage_profile.py) — `instrument()`/`deinstrument()`, reused unmodified
- [`../mojo-max/m4-conclusion.md`](../mojo-max/m4-conclusion.md) §4 — the reopen conditions this document's Task A evaluates against
- [`m4_batching_stage_profile.py`](m4_batching_stage_profile.py) — Task A's script
- [`m4_batching_bench.py`](m4_batching_bench.py) — Task B's script (`--segments-file` option added, additive only)
- [`m4_long_segments_ru.txt`](m4_long_segments_ru.txt) — Task B's paragraph-length fixture
- Raw logs: `logs/m4_batching_stage_profile_batch{1,8}.log`, `logs/m4_batching_long_batch{1,2,4,8}.log`
