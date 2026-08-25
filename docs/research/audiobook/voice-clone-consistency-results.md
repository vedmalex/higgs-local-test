# Voice reference (cloning) wiring: does it actually pin the voice? (Refs #57)

## 0. Why this exists

The owner listened to a fully generated chapter (`output/chapter-114-e0/chapter.wav`, 70
segments) and heard a different voice in almost every fragment. Measured: pitch (F0) median
across the 70 segments ranges 83.9-203.4 Hz (mean 135.3, stdev 28.6) -- low male to high
female inside one chapter. Fixing the random seed does not fix this: a separate seed
experiment (six phrases, one fixed seed) measured a *worse* spread, 105.7-244.9 Hz (139 Hz
range), than no seed at all (82 Hz range) -- see `.plan/voice-consistency-research-draft.md`
(a different agent's parallel cost measurement; not duplicated here). The voice is composed
autoregressively from the text as generation proceeds; the seed only picks among outcomes of
that process, not the process itself.

The only mechanism `higgs_audio_v3`'s own API exposes for pinning a voice across independent
calls is a reference waveform (`ref_audio`/`ref_audio_codes` + `ref_text`), documented and
partially wired in `docs/guides/audiobook_guide.md` sec. 2 (`register_voice()` ->
`voices/<name>.npy`) but never actually passed into `src/audiobook.py`'s generation calls
before this change. This document measures whether wiring it in actually fixes the problem,
what it costs, and whether it survives contact with `--batch-size` (issue #114's batching,
which this project cannot afford to lose -- RTF ~4 vs. ~1 is an ~11h vs. ~38h book).

## 1. What changed

`src/audiobook.py`:

- `VoiceReference` (dataclass) + `resolve_voice_reference()`/`register_voice()`/
  `load_voice_from_registry()`: resolve `--voice-name` (reads `voices/<name>.npy`+`.txt`,
  the exact format `docs/guides/audiobook_guide.md` sec. 2 already documented) or
  `--ref-audio`+`--ref-text` (ad hoc, optionally `--save-voice-as NAME` to register it in
  the same format) into one `VoiceReference`, encoded/loaded exactly **once per run**.
- `generate_segments()` -> `_generate_single_segment()`/`_generate_batch_group()`: the
  resolved reference's `codes`/`ref_text` are threaded through as `ref_audio_codes=`/
  `ref_text=` on **every** `model.generate()`/`model.batch_generate()` call, unbatched and
  batched alike -- not recomputed per call, the same pre-encoded `mx.array` (or `np.ndarray`
  loaded straight from `voices/<name>.npy`) is reused for the whole run.
- Manifest header gains a `voice_reference` field (`{"name", "source", "ref_text_hash"}` or
  `None`). `load_or_create_manifest()` now refuses to resume a manifest whose recorded
  `voice_reference` differs from the one requested this run -- the same mechanism already
  used for `model`/`max_chars`/`tag_scope` mismatches. Rationale: the reference is a property
  of the *run*, not of a segment; a segment's own content hash (`speaker` + text) does not
  change when the reference changes, so without this header check a resumed run would
  silently splice segments generated under two different voices back together -- quietly
  reproducing the exact bug this feature exists to fix. Covered by
  `TestVoiceReferenceManifestInvalidation` in `tests/test_audiobook.py` (same-reference
  resumes cleanly; different reference, no reference, or newly-added reference all raise
  `RuntimeError` naming `voice_reference` in the message).
- CLI: `--voice-name`, `--ref-audio`, `--ref-text`/`--ref-text-file`, `--save-voice-as`,
  `--voices-dir`. `--voice-name`/`--ref-audio` are mutually exclusive.

## 2. Reference selection: which segment of chapter-114-e0

Per-chapter pitch (F0) medians from all 70 `done` segments of
`output/chapter-114-e0/manifest.json`, computed with
`docs/research/audiobook/m4_prosody_metrics.py` (autocorrelation on 40ms frames -- see the
calibration caveat in sec. 5). Chapter median: **129.0 Hz**, mean 135.3, stdev 28.6.

Candidates closest to the chapter median, with pause count (>=150ms silence) over the whole
segment and voiced-frame ratio as secondary filters (excluding the two poles named in the
task: segment 27 = 83.9 Hz and segment 18 = 203.4 Hz):

| idx | text_hash | f0_median_hz | duration_s | pauses>=150ms | voiced_ratio |
| --: | --- | --: | --: | --: | --: |
| 45 | a7eaed50cad3fbec | 129.0 | 28.12 | 3 | 0.827 |
| 49 | f775c8cbb87c84f1 | 129.0 | 30.60 | 4 | 0.780 |
| 50 | 382ee0f5ad68238e | 129.0 | 27.72 | 6 | 0.769 |
| 23 | 026c9e7e159fbf82 | 127.7 | 27.04 | 9 | 0.751 |
| 43 | 6b197873cbfea728 | 130.4 | 25.52 | 9 | 0.798 |
| 14 | 91fa0330cdb8a667 | 131.9 | 30.16 | 13 | 0.713 |
| **36** | **dbaf36e0c82934a7** | **131.9** | **25.76** | **0** | **0.765** |
| 51 | 17906c7325bf726f | 125.7 | 27.24 | 0 | 0.743 |

**Chosen: segment 36** (`segment_dbaf36e0c82934a7.wav`, saved as `voices/narrator_e0.npy`+
`.txt`). Reasoning:

- Several candidates tie or beat it on raw distance to the 129.0 Hz median (45/49/50 are
  exactly 129.0), but every one of those has multiple >=150ms pauses inside the segment
  (3-13). The task's own bar was "без пауз во всю запись" (no pauses across the whole
  recording) -- a reference containing internal silence risks the codec encoding dead air as
  part of the timbre, and pads the reference with non-speech.
- Only two segments (36, 51) have **zero** detected pauses across their full ~26-30s length.
  Of those, 36 (131.9 Hz) is closer to the 129.0 Hz chapter median than 51 (125.7 Hz) -- 2.9
  Hz vs. 3.3 Hz away.
- 25.76s duration is comfortably inside the model's practical reference-length range (docs
  guide recommends 7-12s minimum; longer gives the codec more to condition on, at the cost
  described in sec. 3-4 below) and well clear of any padding threshold.
- Text is a clean, punctuation-normal narrator passage (no quotes, no control tags, no
  proper-noun apostrophes) -- nothing in it could confuse `is_stress_apostrophe` or tag
  parsing when reused as `ref_text`.

The two explicitly excluded poles (27 = 83.9 Hz, low male; 18 = 203.4 Hz, high female) are
exactly the two ends of the chapter's own drift -- using either as the reference would still
"work" mechanically but would bias the whole book toward one extreme instead of the
chapter's own center.

## 3. Does it work? Pitch consistency, 8 segments of different text

Test set: 8 sentences from 8 different, non-adjacent chapter-114-e0 segments (indices 5, 12,
20, 30, 40, 55, 62, 68), none of them the reference text itself, run as an 8-line screenplay
(`--screenplay-file`, one segment per line, `--tag-scope sentence`). Same
`m4_prosody_metrics.py` autocorrelation F0 as sec. 2.

| Run | batch_size | voice reference | n | f0 median-of-medians | mean | stdev | min | max | range |
| --- | --: | --- | --: | --: | --: | --: | --: | --: | --: |
| `voice_test_noref_b1` | 1 | none | 8 | 146.1 | 148.3 | **43.6** | 94.5 | 196.7 | **102.2** |
| `voice_test_ref_b1` | 1 | narrator_e0 (seg. 36) | 8 | 129.7 | 130.5 | **6.1** | 120.3 | 141.2 | **20.9** |
| `voice_test_noref_b4` | 4 | none | 8 | 114.8 | 128.7 | **44.8** | 85.3 | 198.3 | **113.0** |
| `voice_test_ref_b4` | 4 | narrator_e0 (seg. 36) | 8 | 134.4 | 135.4 | **8.2** | 125.7 | 147.2 | **21.5** |

Context from the chapter and from the other agent's seed experiment (not reproduced here):
whole-chapter spread (70 segments, no reference) 83.9-203.4 Hz (range 119.5, stdev 28.6);
six-phrase fixed-seed spread 105.7-244.9 Hz (range 139.2); six-phrase no-seed spread 82 Hz.

**Reading this number honestly:** with the reference wired in (`ref_b1`), stdev drops from
43.6 Hz to 6.1 Hz and range from 102.2 Hz to 20.9 Hz -- roughly a **5x** reduction in spread
on this 8-sentence sample, and the median-of-medians (129.7 Hz) lands almost exactly on the
reference segment's own pitch (131.9 Hz) and the whole chapter's median (129.0 Hz). This is
the strongest evidence in this document that the mechanism works: it did not just narrow the
spread, it pulled every sample toward the *specific* voice the reference encodes, not toward
some other attractor.

`voice_test_ref_b4` (reference + batching together) is the load-bearing remaining cell --
see sec. 4.

## 4. Batching compatibility (does the reference survive `--batch-size`)

`_generate_batch_group` passes `ref_audio_codes`/`ref_text` as a single shared value (not a
per-row list) to `model.batch_generate()`. Per `higgs_audio_v3/model.py`'s
`_normalize_batch_references` (read directly, Refs #57 audit): when `ref_audio_codes` is
given this way, it is encoded/broadcast to every row of the batch from one shared
`ReferenceCodes` object -- so cloning should not need N encodes for an N-row batch, and
should not force `--batch-size 1`.

Measured on the same 8-sentence, non-adjacent test set as sec. 3, `--batch-size 4`:

**It works.** `voice_test_ref_b4` (reference + batch_size=4) completed all 8 segments
without falling back to single-segment generation (no retries, no degrade-to-smaller-batch
in the log), and the pitch spread stayed tight: stdev 8.2 Hz, range 21.5 Hz (sec. 3 table) --
close to `voice_test_ref_b1`'s 6.1 Hz/20.9 Hz (unbatched reference) and nowhere near
`voice_test_noref_b4`'s 44.8 Hz/113.0 Hz (batched, no reference). Batching does not disable
or degrade the voice-pinning effect. This directly answers the task's most consequential
open question: **the reference does not force a retreat to `--batch-size 1`.**

## 5. Calibration caveat

`m4_prosody_metrics.py`'s F0 estimator is a from-scratch autocorrelation over 40ms frames --
no `librosa`/`parselmouth` (not present in `.venv-tts`). It is useful as a **relative**
measure -- "is this the same voice as that clip" -- because the same estimator is applied
consistently to every clip compared here. It is **not** a calibrated absolute pitch
measurement: octave errors (picking a harmonic instead of the fundamental) are possible on
individual frames, and nothing here should be read as "this speaker is exactly N Hz."

## 6. Cost: RTF, memory, batching interaction

All four runs generate the same 8-sentence test set (sec. 3), each in its own process (model
loaded fresh), on the M1 hardware this project targets. `RTF = generation_seconds /
audio_duration_seconds`.

**Important measurement note, not a code bug fixed here:** for `--batch-size > 1`, the
manifest's per-segment `generation_seconds` field is `BatchGenerationResult
.processing_time_seconds`, which `higgs_audio_v3.batch_generate()` reports as elapsed time
*since the batch call started*, not per-row -- so every segment in the same batch carries
almost the whole batch's wall time, and a naive per-segment sum over-counts by roughly
`batch_size`x. The RTF figures below use one representative (the max, i.e. the
last-to-finish row) per batch group instead of summing every row -- this matched the
observed wall-clock (`experiments.log` timestamps) to within a few seconds. This is a
pre-existing characteristic of `mlx_audio`'s batched result objects, not something this
change introduced; flagged here as a caveat for the next person reading `manifest.json`'s
`generation_seconds` field on a batched run, not fixed in this change (out of scope: this
task is about wiring the voice reference, not batching internals, and `_generate_batch_group`
is shared with issue #114's work).

| Run | batch_size | reference | audio_s | generation_s (per-batch-corrected) | RTF | peak_mlx_memory (mx.get_peak_memory) | peak_footprint (`/usr/bin/time -l`) |
| --- | --: | --- | --: | --: | --: | --: | --: |
| `voice_test_noref_b1` | 1 | none | 65.04 | 251.38 | **3.87** | 11.24 GB | 12.51 GB |
| `voice_test_ref_b1` | 1 | narrator_e0 | 56.40 | 314.55 | **5.58** | 11.37 GB | 12.54 GB |
| `voice_test_noref_b4` | 4 | none | 71.68 | 155.21 | **2.17** | 11.75 GB | 12.54 GB |
| `voice_test_ref_b4` | 4 | narrator_e0 | 54.92 | 211.68 | **3.85** | 11.43 GB | 12.42 GB |

`weights_on_disk`: the Higgs TTS 3 4B checkpoint cache is **8.7 GB**
(`~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b`).

Reading these numbers:

- Without a reference, batch_size=4 gives RTF 2.17 vs. 3.87 at batch_size=1 -- **1.78x**
  speedup on this 8-segment sample (smaller than the 3.4-3.69x measured at batch=8 over dozens
  of segments in `m4-batching-integration-results.md`; consistent with only 2 full batches of
  4 here, plus per-run model-load overhead this small sample doesn't amortize).
- With a reference and no batching, RTF rises from 3.87 to 5.58 -- a **~44% slowdown**,
  smaller than but in the same direction as the pre-existing measurement in
  `docs/guides/audiobook_guide.md` sec. 3a (RTF 7.73 vs. 6.56, ~18% slower there -- different
  reference/text/sample, not directly comparable, but the same sign).
- **Reference + batching together: RTF 3.85** -- almost exactly `voice_test_noref_b1`'s
  baseline (3.87, no reference, no batching), and much better than reference alone (5.58).
  Batching's per-token throughput gain and the reference's per-call overhead partially
  cancel out, landing back near where an unbatched, unclonded chapter already was. In other
  words: adding the reference does **not** erase batching's win outright, but it does spend
  most of it -- going from `noref_b4`'s 2.17 to `ref_b4`'s 3.85 gives back roughly
  three-quarters of the 1.78x batching speedup measured in sec. 6's first bullet. This is a
  real cost, not a rounding error, but it is NOT the "RTF ~4 -> ~11h book" floor the task
  worried about -- reference+batch4 (RTF 3.85) is still measurably faster than reference
  alone at batch=1 (RTF 5.58), and close to the pre-batching, pre-reference baseline this
  project already shipped a chapter at.
- Peak memory (both `mx.get_peak_memory()` and `/usr/bin/time -l`'s footprint) does not grow
  meaningfully with the reference: 11.24-11.75 GB `mx` peak / 12.42-12.54 GB process footprint
  across all four configurations, well inside the 16 GB unified-memory target this project
  runs on -- the reference's own encoded size (24 KB `voices/narrator_e0.npy`) is negligible
  next to the model's own footprint.

## 7. What this means for a full book

Rough scaling from measured RTF to wall-clock time for an audiobook whose *finished audio*
runs ~11h (this project's batching-only estimate, `m4-batching-integration-results.md`) or
~38h if batching were lost entirely (the same document's no-batching baseline):

| Configuration | RTF | Estimated wall time for an 11h-of-audio book |
| --- | --: | --: |
| No reference, `--batch-size 1` (pre-#57, pre-#114) | ~4 (measured 3.87 here) | ~38-44h |
| No reference, `--batch-size 4` (issue #114, shipped) | ~2.2 | ~24h |
| Reference, `--batch-size 1` | ~5.6 | ~55-62h |
| **Reference, `--batch-size 4` (this change)** | **~3.85** | **~38-42h** |

This is the honest bottom line: wiring in the reference costs real time -- a full-book run
with cloning and batching together lands close to where the project was BEFORE issue #114's
batching win (not before this change's cost, after it), not close to the ~24h batched-only
number. It is not the doomsday case from the task brief (reference disabling batching
outright, ~38-60h+ blowing past even the pre-batching baseline) -- reference+batching stays
faster than reference-without-batching (55-62h) by a wide margin -- but it does spend most of
what #114 bought. Whether that trade is worth it is an ownership call this document does not
make: it depends entirely on whether the owner is willing to run a chapter's worth of extra
wall time in exchange for a chapter that does not audibly change narrator mid-read.

## 8. What was NOT checked

- **Multi-speaker (`speaker` column) voice selection.** This change pins ONE reference for
  the whole run, applied identically regardless of `speaker`. `chunk_screenplay`'s existing
  multi-speaker warning is unchanged and still correct: a screenplay with more than one
  speaker still gets one voice.
- **Listening confirmation.** Sec. 3's numbers are the from-scratch autocorrelation
  estimator (sec. 5 caveat) on synthetic 8-segment samples, not the owner's ear. A
  comparison task (`src/sentiment_survey/task_sets/voice_consistency.json`,
  `output/voice_consistency_survey/{with_reference,without_reference}.wav`) is provided for
  that -- this document reports what was measured, not what was heard.
- **Voice *quality/naturalness* with the reference active.** Nothing here checks whether the
  cloned voice sounds as good as the model's unconditioned default, only whether it stays
  consistent segment to segment.
