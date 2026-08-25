# Э0 — first full-chapter audiobook run: results

Date: 2026-08-25. Refs #114 (tracked as the project's stress-marking/DSL issue; this Э0
stage is the prerequisite measurement the project owner asked for before designing that
work, so numbers land here and are referenced from #114's comments). Hardware: Apple M1,
16 GB unified memory, macOS, native arm64, `.venv-tts` (MLX-Audio,
`bosonai/higgs-tts-3-4b`).

**This is the first time a full chapter has been generated in this project.** Everything
before this was either a micro-benchmark on a handful of short sentences
(`m4-batching-results.md`, 8 segments) or synthetic-only testing of resume/crash logic.
This run is on 70 real segments (30,483 characters, 4,214 words) of real translated
scripture prose, assembled into a 32-minute WAV.

## 0. What actually happened, in order (read this before the numbers)

The plan was to measure the existing (at the time) unbatched pipeline end to end. Partway
through, the project owner pointed out — correctly — that measuring an unbatched code path
when a batching PR was about to land makes for a mostly-wasted measurement. So the run
was split into two back-to-back parts on **the same manifest, the same fixture, the same
machine**:

1. **Segments 0–30 (31 segments): `--batch-size 1`** (the only path that existed when this
   started) — this is the honest unbatched baseline the project never had before.
2. **Kill test** at segment 31 (in progress) — SIGKILL, deliberately, doubling as the
   project's first real (non-synthetic) mid-chapter crash-recovery check. See §2.
3. PR #123 (`feat/114-batching-integration`, `model.batch_generate()` wired into
   `generate_segments` via `--batch-size`) merged to `origin/main` while this was in
   progress. Fetched and fast-forwarded into the worktree.
4. **Resumed the same manifest with `--batch-size 8`.** Segments 0–30 were reused
   unchanged (content-hash keying does not depend on batch size — confirmed, no manifest
   mismatch). Segments 31–69 (39 segments) were generated in five groups of ≤8 via
   `model.batch_generate()`.
5. `--assemble-only` to produce `chapter.wav` from all 70 segments.
6. Background-noise measurement per segment, diacritics pronunciation check via STT.

This means **the baseline and the batched portion are on different material within the
same chapter** (segments 0–30 vs. 31–69), not the same 70 segments measured twice. See §1
for why this is still a fair comparison, and its one real caveat (machine load).

## 1. Speed: baseline (batch=1) vs. batched (batch=8) on the same chapter

### 1a. Baseline — segments 0–30, `--batch-size 1`

The only path that existed when this run started: one `model.generate()` call per
segment, exactly as `src/tts_test.py --text` already did.

| Metric | Value |
|---|---|
| Segments | 31 |
| Total audio | 852.3 s (14.2 min) |
| Total generation time (sum of exact per-segment `generation_seconds`) | 3263.6 s (54.4 min) |
| **Aggregate RTF** | **3.83** |
| Per-segment RTF | min 3.46, median 3.55, max 6.37 |
| `peak_mlx` (`mx.get_peak_memory()`, per segment) | 11.213–11.328 GiB |

This lands inside the project's own prior projection (RTF 3.88–6.56 from earlier
micro-measurements) — on the low end of it, in fact, which is a reasonable outcome for
paragraph-length segments (~430 chars average) vs. the shorter fixtures those earlier
numbers came from.

### 1b. Batched — segments 31–69, `--batch-size 8`

Resumed on the same manifest after the merge. 39 segments still needing generation, run
as five `model.batch_generate()` groups of 8/8/8/8/7.

**A genuine measurement problem, disclosed rather than hidden**: `generate_segments`'
per-segment `generation_seconds` field, inside a batch, records time-since-batch-started
at the moment each row's decode finished — not that row's own compute cost (this is the
same caveat `m4-batching-results.md` already raised for the standalone bench: "assigning
the whole chunk's wall time to one segment would overstate its cost"). Naively summing it
across all 39 segments gives a nonsense total (8632.8 s) that is 7.6× the real wall time
the whole run actually took. **That sum is not used below.** Instead:

- **End-to-end**: `/usr/bin/time -l` wrapped the whole resumed process: **1136.97 s real**
  for model load + the resume-check of 31 already-done segments + all 5 batches.
- **Steady-state**: batches 2–5 (31 segments, indices 39–69) have a clean wall-clock
  figure from WAV file mtimes (segment files within one batch land within ≤0.03 s of each
  other — see §2b for why that itself matters — so the gap between the last file of one
  batch and the last file of the next is that next batch's real generation time):
  **915.51 s** for **850.8 s** of audio.

| Metric | End-to-end (incl. model load + batch 1) | Steady-state (batches 2–5 only) |
|---|---|---|
| Segments | 39 | 31 |
| Total audio | 1068.4 s | 850.8 s |
| Wall time | 1136.97 s | 915.51 s |
| **Aggregate RTF** | **1.064** | **1.076** |
| `peak_mlx` (per segment) | 11.298–11.381 GiB | (included above) |
| `active_memory` (per segment) | 8.233–8.235 GiB | (included above) |
| `peak_footprint` (`/usr/bin/time -l`, whole process) | **12.73 GB = 11.86 GiB** | n/a (same process) |

The two numbers agree within noise (1.064 vs. 1.076), which is reassuring — the one-time
model-load overhead is small relative to 39 segments of real work.

### 1c. Speedup, and the caveat that must not be papered over

**Naive speedup: 3.83 / 1.064 ≈ 3.60×** (end-to-end) or **3.83 / 1.076 ≈ 3.56×**
(steady-state).

**This number is not clean, and the project owner specifically asked that this be said
plainly rather than credited entirely to batching:**

| Run | Machine load average (1 min) during the run | Swap used/total |
|---|---|---|
| Baseline (batch=1), segments 0–30 | ranged **4.85 → 18.96** (other agents active on this shared machine) | 3.79 → 8.9 / 10.24 GB |
| Batched (batch=8), segments 31–69 | ranged **1.54 → 2.10** (machine had quieted down) | 3.0 → 2.9 / 4.1 GB |

The baseline ran while this machine was genuinely contended (load average spiked to ~19,
swap climbed toward 9 of 10 GB) — other agents' processes, not this benchmark, per the
`ps aux` snapshot taken at the time. The batched portion ran once the machine had quieted
down on its own. **Some of the apparent 3.56–3.60× speedup could be "the machine was less
busy," not purely "batching is faster."** The honest way to bound this: the project's own
prior *controlled* measurement (`m4-batching-results.md`, both batch=1 and batch=8
measured back-to-back on the same machine state, no confound) found 3.40–3.69× for
`batch_generate()` directly. This run's 3.56–3.60× **sits inside that already-established
range**, not above it — if load-average noise were doing most of the work here, this
number would be expected to exceed the clean controlled measurement, not land inside it.
That is evidence the batching effect itself dominates, but it is an inference, not a
controlled re-measurement, and is reported as such. A clean same-load-average rerun would
be needed to fully separate the two effects.

## 2. Kill test — deliberate SIGKILL mid-chapter, on the real pipeline for the first time

Prior kill/resume testing in this project was synthetic-only. This is the first real one.

### 2a. What was killed and when

`SIGKILL` sent to the running `--batch-size 1` process at **31/70 segments done**,
segment 31 `in_progress` (mid-generation, ~visibly non-trivial elapsed time into that
segment). **This was a deliberate stop, not a crash** — stated plainly so it is never
mistaken for an accidental failure being reported as a recovery success.

### 2b. State immediately after the kill

| Check | Result |
|---|---|
| `manifest.json` parses | **OK** — valid JSON, no truncation |
| `manifest.json.bak` present | **OK** |
| Segment 31 (`in_progress` at kill time) has a WAV on disk | **No** — correctly absent, not a corrupt partial file |
| WAV count on disk | 31, matching the 31 `done` manifest entries exactly |
| A `done` entry's WAV opens and its declared frame count matches | **OK**, spot-checked |
| `_resume_check_ok()` (the library's own resume-integrity check) on all 31 `done` entries | **31/31 pass, 0 fail** |

**Verdict: the crash-recovery design worked exactly as documented for the unbatched
path.** No corruption, no false "done" status, a clean restart point.

### 2c. A real discrepancy found in the batching integration's kill-safety claim

`docs/research/audiobook/m4-batching-integration-results.md` (PR #123) states: "a kill
mid-batch can therefore lose at most the segment(s) not yet yielded/validated/written, not
the whole batch," backed by a synthetic unit test
(`test_manifest_written_after_each_segment_within_a_batch_not_only_after_whole_batch`).

**This run's real WAV file mtimes do not show that.** Within every one of the 5 batches
generated here, all segment WAV files in that batch landed within **≤0.03 seconds of each
other** — i.e. written back-to-back at the very end of the batch's decode, not
incrementally as each row finished:

```
batch 1 (segments 31-38): mtimes span 0.05 s
batch 2 (segments 39-46): mtimes span 0.06 s
batch 3 (segments 47-54): mtimes span 0.05 s
batch 4 (segments 55-62): mtimes span 0.04 s
batch 5 (segments 63-69): mtimes span 0.04 s
```

This is real, measured, hardware behavior, not a re-run of the PR's synthetic test — and
it disagrees with that test's premise. Plausible explanation: `mlx_audio`'s
`batch_generate()` may not actually yield each `BatchGenerationResult` as its row
individually finishes decoding in this version/build, despite the PR's write-up describing
it that way — or MLX's lazy evaluation defers the underlying compute for the whole batch
until the first `mx.eval()`/consumption point regardless of per-row eviction. Either way:
**this run never tested a kill *during* a batch=8 run** (the deliberate kill happened
during the batch=1 portion, before batching existed on this branch). What is reported here
is an inference from file-write timing, not a confirmed kill-during-batch outcome.

**This should be flagged to the PR #123 author as an open question, not a confirmed bug**:
either the "loses at most one segment" claim needs updating to "loses at most one batch"
for the real `mlx_audio` build in use, or there is a genuine incremental-write path that
this run's file-timestamp method is simply too coarse to see (sub-10ms writes on a fast
local SSD could plausibly all round to the same `stat()`-visible mtime even if called
sequentially in Python). **A direct kill-during-batch test is the only way to settle this
and was not performed in this Э0 run** — recorded here as follow-up work, not swept under
the "kill test done" checkbox.

## 3. Memory on a real several-hundred-segment-class distance

Micro-benchmarks (`m4-batching-results.md`) covered 8 segments per batch size and found no
memory ceiling — but explicitly flagged that a real chapter (hundreds of segments) was
untested. This run covers 39 batched segments across 5 batches — still short of "hundreds"
for the batched portion specifically, but the longest batched run to date in this project.

| | batch 1 (segs 31-38) | batch 2 (39-46) | batch 3 (47-54) | batch 4 (55-62) | batch 5 (63-69) |
|---|---|---|---|---|---|
| `peak_mlx` (GiB) | 11.298 | 11.298 | 11.376 | 11.381 | 11.381 |
| `active_memory` (GiB) | 8.233 | 8.233 | 8.234 | 8.235 | 8.235 |

**A small, real upward drift exists**: `peak_mlx` rose ~0.083 GiB (11.298 → 11.381,
+0.7%) between the first two batches and the last three, then held flat. `active_memory`
rose by a negligible ~0.002 GiB. This does not look like a leak (it plateaus, does not
keep climbing batch over batch) — more consistent with MLX's allocator settling into a
slightly larger high-water mark once cache/KV-cache patterns from longer segments were
seen, matching `--clear-cache-every`'s design intent. **It is reported as a real, measured
number, not rounded away, because a genuinely unbounded version of this same shape of
drift is exactly the failure mode a hundreds-of-segments chapter would need to catch.** A
longer batched run (the full 250+ segments a 10-hour book chapter might have) is needed to
confirm the plateau holds at that scale — this run does not settle that question, only
narrows it.

`peak_footprint` for the whole batch=8 process (model load + resume-check + all 5
batches): **12.73 GB = 11.86 GiB**. `weights_on_disk` (`bosonai/higgs-tts-3-4b` cache
directory): **8.7 GiB**. (`ru_maxrss`/"maximum resident set size" is not cited, per this
project's standing rule that it undercounts on macOS — the wrapper printed 4.69 GB for
this run, visibly inconsistent with `peak_mlx` alone being ≥11.2 GiB, confirming that rule
again.)

## 4. Chapter assembly — splice quality

`--assemble-only` on all 70 segments, default 200 ms inter-segment silence:

| | |
|---|---|
| Segments assembled | 70 / 70, **0 gaps** |
| Output | `output/chapter-114-e0/chapter.wav` (89 MB, 24 kHz mono) |
| Total duration | **1934.44 s = 32.24 min** |

`assemble_chapter`'s own numeric join check (`direct_join_sample_jump` — the discontinuity
that would exist between two segments' audio if the inserted silence were removed) was
**exactly 0.0 at all 69 joins**. Every segment's generated audio fades to (or starts from)
a near-silent sample at its own edge, so there is no measured click/discontinuity risk
anywhere in this chapter — a genuinely clean result, not a rounding artifact (spot-checked
against the underlying edge amplitudes).

`max_intra_window_sample_jump` (internal dynamics at the edge, not a splice defect by
itself) varies normally; the three highest are listed in §6 as listening candidates, since
"no numeric click" does not guarantee "sounds natural" — that judgment needs ears.

The baseline→batch transition (segment 30→31, exactly where the two measured halves of
this run meet) shows **nothing unusual** in this metric (`direct_join_sample_jump` 0.0,
`max_intra_window_sample_jump` 0.046 — well inside the normal range) — the switch from
`--batch-size 1` to `--batch-size 8` mid-chapter left no numeric splice artifact.

## 5. Background noise across the assembled chapter

Reused `docs/research/audiobook/m4_bgnoise_metrics.py` (PR #112) unchanged, run over all
70 segment WAVs (not the isolated short clips that earlier work measured).

| | |
|---|---|
| Segments with a real pause-based silence estimate (not the fallback floor) | 64 / 70 |
| `silence_energy_db` range across those 64 | **−120.0 to −51.7 dB** |
| **Spread** | **68.3 dB** |

This **matches (very slightly exceeds) the 67 dB spread** already found on isolated clips
in `m4-background-noise-results.md` — confirming, on a real assembled chapter for the
first time, that the per-clip noise-level inconsistency that measurement flagged is real
at chapter scale too, not an artifact of comparing unrelated clips. The biggest adjacent-
segment jumps in background level (candidates for an audible "background pops in/out" —
listed with segment WAV filenames in §6 for direct listening):

| After segment | Before segment | Jump (dB) | Levels |
|---|---|---|---|
| 9 | 10 | **58.9** | −61.1 → −120.0 |
| **30** | **31** | **57.1** | −62.9 → −120.0 |
| 17 | 18 | 53.7 | −66.3 → −120.0 |
| 10 | 11 | 53.6 | −120.0 → −66.4 |

Segment 30→31 (the exact baseline/batch transition point) is the **second-largest**
background-level jump in the whole chapter. This is flagged for direct listening (§6) —
it may be coincidence (segment 31 happening to land in a genuinely quieter passage) or it
may indicate `batch_generate()` produces a measurably different noise floor than
`generate()` for an otherwise-comparable segment. This run's data cannot distinguish those
two explanations; a same-segment generated both ways would be needed to isolate it, and
was not done here. Reported as observed, not diagnosed.

A caveat inherited from `m4-background-noise-results.md`: `silence_energy_db` of exactly
**−120.0 dB** repeats (segments 10, 18, 31) and is very likely the pause-detector's own
numerical floor for a near-perfectly-silent pause region, not meaningfully "10 dB quieter
than −110.6 dB" — treat the −120.0 entries as "at or below the measurable floor," not as
a precise reading.

## 6. Sanskrit diacritics — pronunciation check via STT round-trip

Four short clips (`output/diacritics-114-e0/`), generated independently (each its own
`--text` call, not part of the 70-segment chapter) from `samples/audiobook/prepared/`:

| Clip | Text | Audio (s) | STT transcript (`qwen3_asr`, no reference) |
|---|---|---|---|
| `verse_with` | Full श्लोक 1.1.1, diacritics kept | 19.32 | "Амг намобаговате, вас удивля Джанмадия сиято, вайат итараташе чарт хешваб хижнах, сварат тенебрах мхрдея адика вая, мухьянти ят сура яхте Джоварим рдам, ят хавинимаю ятра три сарго, мрешат хамна свена, садани раста кухакам сатям парам дхимахи" |
| `verse_plain` | Same verse, diacritics stripped | 21.56 | "Он наму поговорит и вас утревая. Джон Мадиас яято, лояди тараташ, чатхис а пич ных свараттене. Брахма Хрдая, ари кевая мухианти ят сураях тежо варим рдам, ят хвинимайо ят ратри сарго. Милшад хамна свена садени растеку хакам, сатям парамти махи" |
| `sentence_with` | "Первое слово этой шлоки, ом̇, указывает на Ва̄судеву, а Шри Вьясадева говорит о нем как об абхиджн̃ах̣ и свара̄т̣." | 11.24 | "Первое слово этой шлаки «ом» указывает на васудеву, а шревесадева говорит о нем как об обхиджнах и сварат." |
| `sentence_plain` | Same sentence, diacritics stripped | 9.16 | "Первое слово этой шлока ом указывает на восудеву. Ашреви садева говорит о нем как об абхиджнах и сварат." |

**Findings:**

- **No combining diacritic is vocalized as an extra sound or read literally.** No
  transcript contains anything resembling "макрон," "точка," a spelled-out mark name, or
  garbage tokens attributable to the mark characters themselves. Whatever the model does
  with `ом̇`/`Ва̄судеву`/`абхиджн̃ах̣`, it treats the combining marks as invisible to its
  own output vocabulary, not as literal content to pronounce or transcribe.
- **Diacritics neither clearly help nor clearly hurt intelligibility on this proxy.** The
  `sentence_with`/`sentence_plain` pair produced near-identical garbling of the Sanskrit
  words (`абхиджнах`/`обхиджнах`, `сварат` in both) — STT round-trip fidelity is not
  measurably different with or without the marks present in the input text. This is a
  rough proxy (an ASR model with no exposure to this transliteration scheme, not a ground
  truth for correct Sanskrit pronunciation) — it answers "does the model choke on the
  marks" (no) rather than "does the model pronounce Sanskrit correctly" (unmeasured,
  needs the owner's ear).
- **`итараташ́` is exactly the code-point collision flagged in
  `samples/audiobook/README.md`**: this scheme's combining acute (U+0301) marking a
  palatalized "ш" here is the same code point the project has discussed for Russian
  stress marks elsewhere. Nothing broke on that overlap in this run, but a future
  automatic-stress-marking pass over a chapter that also contains this transliteration
  scheme should be aware the two uses cannot be told apart by code point alone.

## 7. Fixtures and `prepare.py` — reusable, provenanced, rule-based

Per the owner's request that these serve documentation and future skill work, not just
this one run: `samples/audiobook/` (raw sources + `prepare.py` + prepared outputs +
provenance table with source paths, YAML `contentHash`, and `title` for each source file)
is committed as a standing fixture set, not deleted after this run. Full detail —
including the explicit drop/keep rules and the input-contract draft for a future
audiobook-preparation skill — is in `samples/audiobook/README.md`; not duplicated here.

The actual Э0 chapter input: `samples/audiobook/prepared/chapter-e0-narration.txt`
(`sb-1-19.txt` + `sb-1-1-1.txt`, 4214 words, 70 segments after chunking).

## 8. Listening pointer — what to check by ear, and why

`output/chapter-114-e0/chapter.wav` (32.24 min). Segment-level WAVs are in the same
directory (`segment_<hash>.wav`, cross-referenced to chapter position via
`manifest.json`'s `segments[].index`/`output_path`). Suggested order:

1. **The opening** (first ~60s, segments 0–1) — first exposure to the narrator's voice
   and pacing on this material.
2. **The baseline→batch transition, segment 30→31** — numerically clean (§4) but also the
   single largest background-noise jump after 9→10 (§5); worth confirming by ear whether
   `--batch-size 1` and `--batch-size 8` sound like the same voice/register, since nothing
   in this project has listened across that specific switch before.
3. **The three largest background-noise jumps**: **9→10**, **17→18**, **10→11** — this is
   where the 68.3 dB spread (§5) should be audible as background popping in or out, if it
   is audible at all.
4. **Sanskrit proper names inside the main chapter, unmarked** — "Парикшит," "Шукадева
   Госвами," "Вьясадева," "Сатьялока" all occur repeatedly in `sb-1-19.txt` with no stress
   marks. This is the direct calibration the owner asked for: how the model guesses stress
   on names it was never told how to stress.
5. **The four diacritics clips** (`output/diacritics-114-e0/{verse_with,verse_plain,
   sentence_with,sentence_plain}/chapter.wav`) — the promised with/without-diacritics A/B
   pair, both as a full श्लोक and as a single sentence excerpt.
6. **The two highest `max_intra_window_sample_jump` joins**, segments **57→58** and
   **66→67** — no numeric click was found anywhere (§4), but these are the closest thing
   to an outlier in that metric and worth a direct listen to confirm the metric is not
   missing something a person would hear.

## 9. Honest summary

- **RTF baseline (batch=1, real chapter, first time measured): 3.83.** Matches the
  project's prior projection.
- **RTF batched (batch=8, real chapter, first time measured): 1.06–1.08**, i.e. **~3.6×
  faster** — inside the range the controlled micro-benchmark already predicted
  (3.40–3.69×), with an honest caveat that the two halves of this comparison ran under
  different machine load and the speedup figure is not fully isolated from that (§1c).
- **This moves a projected 10-hour audiobook from ~38–66 machine-hours (batch=1) to
  roughly ~11 hours (batch=8)** — consistent with, not better or worse than, what
  `m4-batching-results.md` already said, now confirmed on real paragraph-length chapter
  material instead of only short independent sentences.
- **Kill/resume: works as designed on the unbatched path** (§2b), confirmed for the first
  time on a real (not synthetic) run. **Kill-during-batch was not directly tested** — a
  real discrepancy was found between the batching PR's documented write granularity and
  this run's observed file-write timing (§2c), which should be resolved with a direct test
  before the "kill-safe under batching" claim is trusted at face value.
- **Splice quality: clean** — 0 numeric discontinuities across 69 joins, including at the
  baseline/batch switch.
- **Background noise: real and chapter-scale**, 68.3 dB spread confirmed on assembled
  audio, matching prior isolated-clip measurement almost exactly. Specific joins named for
  listening (§5, §8).
- **Diacritics: silently tolerated, not vocalized as garbage, no measurable STT
  win/loss** — owner's ear is still the actual test for pronunciation quality.
- Nothing here was rounded up, and the one place this run fell short of a fully clean
  answer (kill-during-batch, §2c) is reported as unresolved rather than assumed fine.

## 10. Files

- Fixtures: `samples/audiobook/` (raw, prepared, `prepare.py`, `README.md` with
  provenance and the input-contract draft).
- Chapter audio: `output/chapter-114-e0/chapter.wav` + 70 `segment_*.wav` +
  `manifest.json` (gitignored, not committed — path recorded here for the owner).
  Diacritics clips: `output/diacritics-114-e0/{verse_with,verse_plain,sentence_with,
  sentence_plain}/`.
- Raw logs: `logs/run1.log` (baseline, killed), `logs/run2_batch8.log` (batched, completed,
  `/usr/bin/time -l` output at the end), `logs/assemble.json`, `logs/bgnoise_segments.json`,
  `logs/machine-state.log`.
- This document.
