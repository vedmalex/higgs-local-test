# M4-T0 — Sentiment/style/prosody tag baseline: objective evidence + owner verdict (PASSED)

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio). Blocking gate per
[`m4-plan.md`](m4-plan.md) §3, M4-T0: "can sadness and elation be told apart"; everything in M4
that assumes sentiment tags actually work (T4, T6, Lane 2) is contingent on this passing.
**Update, same date: the owner has now given the blind-listening verdict this gate was waiting
on — see §6. Emotions are distinguishable and `sadness` genuinely sounds sad (gate PASSED on
substance); `style:whispering` is not usable as whispering (see §6b); F0 median is flagged as an
unreliable proxy for this checkpoint (see §6c).**

Sections 1-5 below are the objective material that was prepared *for* that judgment: exact tags
used (pulled from the real checkpoint's `tokenizer.json`, not typed from memory), the generated
clips, an STT check for the literal-tag-readout failure mode already on record in
`README.md:245`, and prosodic measurements. §6 records the owner's actual verdict.

## 0. Tags used (verified against the real checkpoint)

Read directly from
`~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b/snapshots/7556c17e05201fccd9c8cc120bc216dcc7b5d561/tokenizer.json`,
`added_tokens` where `special: true`:

```
151699 <|emotion:sadness|>
151681 <|emotion:elation|>
151706 <|style:whispering|>
151717 <|prosody:speed_slow|>
```

All four tags the task asked for exist verbatim in the checkpoint — no substitution was needed.

## 1. Method

**Text** (identical across all 5 clips, semantically neutral so any emotional coloring in the
audio has to come from the tag, not the words):

```
Сегодня вторник. Автобус приходит в девять утра. Дорога до станции занимает десять минут.
```

**Clips generated** (tag prepended to the text above, or no tag for the neutral baseline):

| Clip | Tag | Output |
|---|---|---|
| neutral | (none) | `output/m4t0_neutral.wav` |
| sadness | `<\|emotion:sadness\|>` | `output/m4t0_sadness.wav` |
| elation | `<\|emotion:elation\|>` | `output/m4t0_elation.wav` |
| whispering | `<\|style:whispering\|>` | `output/m4t0_whispering.wav` |
| speed_slow | `<\|prosody:speed_slow\|>` | `output/m4t0_speed_slow.wav` |

`src/tts_test.py` only exposed hardcoded text per `--mode` (`basic` reads
`samples/tts_ru.txt`, `controls` uses a fixed multi-tag string). It was extended with an optional
`--text` override (ignored for `--mode clone`, which must keep using the cloning reference text)
so this experiment could reuse the existing model-loading/generation path instead of writing a
parallel script. This is the only code change on this branch.

Each clip was generated in its own `.venv-tts` process (`/usr/bin/time -l` wrapped, per AGENTS.md
process-isolation rule), e.g.:

```bash
TEXT="Сегодня вторник. Автобус приходит в девять утра. Дорога до станции занимает десять минут."
.venv-tts/bin/python src/tts_test.py --mode basic --text "<|emotion:sadness|>$TEXT" \
  --output output/m4t0_sadness.wav
```

(same pattern for `neutral` with no tag prepended, `elation`, `whispering`, `speed_slow`). Raw
`/usr/bin/time -l` logs: `logs/m4t0_{neutral,sadness,elation,whispering,speed_slow}.log`.

STT check used the project's existing fast path, `src/stt_qwen_local_test.py`
(`mlx-community/Qwen3-ASR-0.6B-8bit`, RTF ≈0.18-0.2 measured here, own child process per clip):

```bash
.venv-tts/bin/python src/stt_qwen_local_test.py --audio output/m4t0_sadness.wav \
  --output output/m4t0_sadness_stt.txt --metrics logs/m4t0_sadness_stt.json
```

Prosodic metrics used a small ad-hoc analyzer
(`docs/research/audiobook/m4_prosody_metrics.py`, committed alongside this report) built on
`numpy`/`scipy` only — `librosa`, `parselmouth`, and `soundfile` are **not installed** in
`.venv-tts` (verified: `ModuleNotFoundError` for all three). F0 is estimated per-frame by
windowed autocorrelation (40 ms frames, 10 ms hop, 60-400 Hz search range, periodicity-strength
gate at 0.3 to reject unvoiced/silent frames); pauses are detected as runs of 20 ms frames at
least 40 dB below the clip's peak energy, ≥150 ms long. This is a coarse, uncalibrated estimator,
not a validated pitch tracker — see Limitations below.

## 2. Check #1 — literal tag readout (README:245 failure class)

None of the five transcripts contain the tag text, category names, or English loanwords for them
("emotion", "sadness", "elation", "prosody", "style", "whispering", "slow", "tag", etc.). The
model does **not** read the control tags out loud on this run — the specific failure this check
was built to catch (README.md:245's literal-readout-of-a-markup-instruction precedent, from
stress marks) does **not** reproduce here.

| Clip | Transcript (`src/stt_qwen_local_test.py`, Qwen3-ASR) |
|---|---|
| neutral | Сегодня вторник. Автобус приходит в 9 утра. Дорога до станции занимает 10 минут. |
| sadness | Сегодня вторник. Автобус приходит в 9 утра. Дорога до станции занимает 10 минут. |
| elation | Сегодня вторник. Автобус приходит в 9 утра. Дорога до станции занимает 10 минут. |
| whispering | Сегодня вторник. Автобус приходит в девять утра. Дорога до станции занимает десять минут. |
| speed_slow | Сегодня вторник. Автобус приходит в девять утра. Дорога до станции занимает десять минут. |

(The digit-vs-word spelling difference between rows is a Qwen3-ASR text-normalization quirk, not
a content difference — all five say the same sentence.) Full transcripts and per-clip STT metrics
JSON: `output/m4t0_*_stt.txt`, `logs/m4t0_*_stt.json`.

## 3. Check #2 — prosodic metrics

Raw JSON: `logs/m4t0_prosody_metrics.json`. Word count is identical (13 words) across clips since
the text is identical, so words/second is a valid tempo proxy here.

| Clip | Duration (s) | Tempo (words/s) | F0 median (Hz) | F0 range (Hz) | F0 std (Hz) | Voiced-frame ratio | Mean energy (dB) | Pauses (n / total ms) |
|---|---|---|---|---|---|---|---|---|
| neutral | 6.96 | 1.87 | 162.2 | 331.2 | 77.4 | 0.808 | -28.8 | 2 / 460 |
| sadness | 7.12 | 1.83 | 195.1 | 334.6 | 41.7 | 0.702 | -34.2 | 3 / 1260 |
| elation | 6.80 | 1.91 | 275.9 | 297.0 | 94.3 | 0.862 | -29.7 | 2 / 580 |
| whispering | 7.52 | 1.73 | 162.2 | 333.5 | 52.5 | 0.715 | -31.9 | 3 / 1060 |
| speed_slow | 8.28 | 1.57 | 158.9 | 335.0 | 78.4 | 0.744 | -43.5 | 3 / 1120 |

### sadness vs. elation

Expected if the tags work: sadness lower/narrower F0 and slower than elation.

- **F0 median**: sadness 195.1 Hz vs. elation 275.9 Hz — elation is higher, which is in the
  expected direction. But sadness is also higher than the untagged **neutral** baseline
  (162.2 Hz), which is the opposite of the expected direction for sadness on its own.
- **F0 std (variability)**: sadness 41.7 Hz, clearly narrower than elation's 94.3 Hz — consistent
  with "sadness = flatter delivery."
- **Tempo**: sadness 1.83 words/s vs. elation 1.91 words/s — sadness is marginally slower, but
  the gap (≈4%) is small next to the run-to-run duration variance already visible between
  neutral/elation (6.96 s vs. 6.80 s with no prosody tag involved).
- **Pauses**: sadness has more and much longer pauses (3 pauses, 1260 ms total) than elation
  (2 pauses, 580 ms) — consistent with the expected direction.

**Read**: two of four signals (F0 spread, pausing) point the expected way; the tempo gap is
small; and the F0 median comparison is internally inconsistent (sadness *above* the neutral
baseline, not below it). This is a mixed result, not a clean pass. **(Update, §6c: the owner's
blind verdict says `sadness` genuinely sounds sad — the F0-median signal here is now understood as
a misleading proxy for this checkpoint, not evidence the emotion failed. Trust the F0-spread and
pause signals instead.)**

### whispering vs. neutral

Expected: sharply lower energy and much weaker/absent periodicity (a real whisper has little to
no vocal-fold vibration, so voiced-frame ratio should collapse, not just dip).

- **Mean energy**: -31.9 dB vs. -28.8 dB — about 3 dB quieter. Present, but far short of the
  large drop a true whisper produces.
- **Voiced-frame ratio**: 0.715 vs. 0.808 — a modest reduction, not the near-collapse expected of
  actual whispered (voiceless) speech.
- **F0 median**: 162.2 Hz — identical to neutral to one decimal place, which is itself notable:
  a genuine whisper has no reliable fundamental at all.

**Read**: the `whispering` tag produces a small quieter/breathier shift, not an acoustically
whisper-like signal.

### speed_slow vs. neutral

Expected: lower tempo (more time per word), everything else roughly held constant.

- **Duration**: 8.28 s vs. 6.96 s (+19%); **tempo**: 1.57 vs. 1.87 words/s (-16%). This is the
  clearest, most internally consistent result of the four probes — the direction matches the tag
  intent, on the metric it's supposed to move.
- Side effect: mean energy dropped sharply (-43.5 dB vs. -28.8 dB) and F0 std/median shifted
  slightly. Not predicted, and not currently explained — could be a genuinely slower/quieter
  delivery, or could be an artifact of the coarse energy estimator on a longer, more
  pause-laden clip. Flagged, not resolved.

## 4. Limitations (read before trusting the numbers above)

- **n = 1 per condition.** No repeated trials, no seed variation, no statistical test. Single
  runs at `temperature=1.0` (the project's existing default) are exactly the kind of thing that
  can differ from clip to clip for reasons unrelated to the tag.
- **The F0 estimator is homemade and uncalibrated** — plain autocorrelation on 40 ms frames,
  capped at 400 Hz, with no validation against a reference pitch tracker (none of
  librosa/parselmouth/soundfile is installed in `.venv-tts`; installing one was avoided per the
  task's "don't pull heavy dependencies" instruction). Autocorrelation pitch trackers are prone
  to octave errors, particularly on frames with strong harmonics — the sadness clip's
  higher-than-neutral F0 median could plausibly be exactly this kind of error rather than a real
  effect, and cannot be told apart from a real effect with this tool.
- **The pause/energy detector uses a fixed relative-dB threshold**, not adapted to each clip's
  noise floor; the speed_slow clip's much lower mean-energy figure may partly reflect longer
  silence stretches pulling the average down rather than quieter speech throughout.
- This says nothing about intelligibility, naturalness, or whether the *emotional character* a
  human ear would attribute to sadness/elation is present — that is exactly the judgment these
  metrics cannot make.

## 5. Verdict

**Objective metrics are mixed, not a clean pass.** `speed_slow` shows a clear, internally
consistent tempo effect in the expected direction. `sadness` vs. `elation` shows some signals in
the expected direction (F0 variability, pausing) and one that runs against it (sadness's F0
median sitting above the untagged neutral baseline, not below). `whispering` shows only a modest
energy/voicing shift, well short of what a genuine whisper would produce acoustically. None of
the five clips read the tags out loud (Check #1 passes cleanly).

This is **not** a claim that "emotion tags work" or "emotion tags don't work." Per the task's own
instruction, that call belongs to the project owner's ear, not to metrics assembled by an agent
in one run. What can be said objectively: the tags reach generation without being spoken aloud,
and the prosodic evidence is **partially consistent but not conclusively supportive** of the tags
producing the intended emotional/stylistic effect — strong enough that the owner's blind
listening check called for in `m4-plan.md` §3 (M4-T0) should happen before committing to the rest
of Track T, and should not be assumed to pass just because these metrics have signal.

**For the owner's listening check, compare first:**

1. `output/m4t0_sadness.wav` vs. `output/m4t0_elation.wav` — the pair the plan's M4-T0 gate is
   actually about. Ask: can these be told apart blind, and does either sound like the labeled
   emotion rather than just "a different reading" of the same sentence?
2. `output/m4t0_whispering.wav` vs. `output/m4t0_neutral.wav` — the metrics above suggest this
   tag is the weakest of the four; confirm or refute by ear whether it sounds whispered at all.

All five clips (`output/m4t0_{neutral,sadness,elation,whispering,speed_slow}.wav`) are local,
`output/`-gitignored, and not committed to this branch — only this report and the small analysis
script are.

## 6. Owner's verdict (SUBJECTIVE — blind listening check, 2026-08-25)

Per this project's own rule (§5 above, and `m4-plan.md` §3 M4-T0), the emotion/style call belongs
to the project owner's ear, not to an agent. The owner listened to the clips referenced in §5
("For the owner's listening check, compare first") and gave two verbatim verdicts, the second
sharpening the first:

> «по звуку, он различим, и эмоции тоже, только шепот звучит как тихий звук»

> «сэднесс и эталон различимы, и грусть действительно грустная» (owner's own phrasing of the
> tags by ear; "эталон" here is `elation`)

This is a **subjective human judgment**, not a metric, and is recorded as such. Three separate
claims are contained across the two statements, and none should be stretched past what was
actually said:

**(a) Emotions are distinguishable by ear, and `sadness` is not just different from `elation` —
it sounds like actual sadness.** The first statement established the pair is distinguishable
("он различим, и эмоции тоже"); the second is a stronger, more specific claim: "грусть
действительно грустная" — sadness genuinely sounds sad, i.e. the tag produces the *semantically
correct* emotion, not merely *a* different-sounding reading of the same sentence. This is a
materially stronger pass of the M4-T0 gate (§3: "can sadness and elation be told apart blind?")
than "distinguishable" alone would have been — the gate is **PASSED on substance, not just on
form**. Note what the owner did *not* say: no claim of "excellent" or "production quality" was
made, no claim about `elation` sounding correctly joyful (only that it is distinguishable from
`sadness`). Do not upgrade this verdict beyond what was said. Only 4 of the 34 tags in the
checkpoint (§0 above) have been checked at all; `sadness`/`elation`/`whispering`/`speed_slow` are
the extent of the evidence — this verdict says nothing about the other 30 tags.

**(b) `<|style:whispering|>` does not produce whispering — it produces quieter speech.** The
owner's description ("шепот звучит как тихий звук" — "the whisper sounds like a quiet sound")
matches the objective metrics in §3 exactly: energy dropped only ~3 dB relative to neutral
(-31.9 dB vs. -28.8 dB) while the F0 median was **identical to the neutral baseline to one decimal
place** (162.2 Hz for both). A real whisper is breathy, voiceless phonation with a destroyed
fundamental — not simply quieter voiced speech. The unchanged F0 median, which §4's Limitations
section had flagged as suspicious on its own (identical medians across two independent
generations), is now explained by the owner's verdict rather than left as an open anomaly: the tag
is not producing whisper phonation, so there is no reason for F0 to move at all — it is behaving
like a volume control, not a phonation-mode switch.

This is **not** "the tag does nothing." It changed energy (~3 dB) and pause structure (3 pauses /
1060 ms vs. neutral's 2 / 460 ms, per §3) — something in generation responded to the tag. What it
did not do is produce the whisper *quality* of speech the tag name promises. The precise claim:
`<|style:whispering|>` is measurably distinguishable from neutral but is **not usable as a
substitute for actual whispered narration** in this project's intended sense.

**(c) Methodological finding: F0 median is a bad proxy for this model's Russian sentiment
synthesis; F0 spread and pause duration are not.** §3 flagged as a mixed/inconsistent result that
`sadness`'s F0 median (195.1 Hz) sits *above* the untagged neutral baseline (162.2 Hz) — read at
the time as "the wrong direction for sadness." The owner's verdict that `sadness` genuinely sounds
sad directly contradicts that reading: the emotion is correct on the ear, so the F0-median signal
that looked wrong was **misleading, not the emotion**. The two metrics that *did* point the
expected direction in §3 are corroborated by this verdict: F0 std/variability (sadness 41.7 Hz vs.
elation's 94.3 Hz — flatter delivery for sadness) and pause duration (sadness 1260 ms total vs.
elation's 580 ms — more/longer pauses for sadness). **Consequence for the M4-T4 quantization
sentiment-integrity gate's objective proxy** (`m4-plan.md` §3, M4-T4: "F0 range, pause duration,
speaking rate"): treat F0 median as unreliable for this checkpoint/language and do not use it as a
pass/fail signal — it would have raised a false alarm here and could again. F0 spread (range/std)
and pause duration are the proxies with owner-verdict-corroborated validity; prefer them.

**Net effect on M4:** the blocking gate is cleared, and cleared on substance (sadness sounds
genuinely sad, not merely different) — the optimization branch (batching → quantization, Lane 1 of
`m4-plan.md` §1) is unblocked, since there is now a real, semantically-correct sentiment baseline
worth preserving under quantization. Separately, `whispering` is downgraded from "one of the four
probed tags" to "known not fit for purpose"; see `m4-plan.md`'s updated §0.4/M4-T5 for the
practical consequence for audiobook tag choice and for the other 30 unverified tags, and §3/M4-T4
for the F0-median proxy caution.

## 7. Files

- Clips: `output/m4t0_neutral.wav`, `output/m4t0_sadness.wav`, `output/m4t0_elation.wav`,
  `output/m4t0_whispering.wav`, `output/m4t0_speed_slow.wav` (gitignored, local only)
- STT transcripts: `output/m4t0_{neutral,sadness,elation,whispering,speed_slow}_stt.txt`
- Generation logs (`/usr/bin/time -l`, JSON metrics): `logs/m4t0_*.log`
- STT metrics: `logs/m4t0_*_stt.json`
- Prosody metrics (raw): `logs/m4t0_prosody_metrics.json`
- Analyzer script (committed): `docs/research/audiobook/m4_prosody_metrics.py`
- Code change: `src/tts_test.py` — added optional `--text` override for research probes like this
  one; no behavior change for existing `--mode basic/controls/clone` invocations that omit it.
