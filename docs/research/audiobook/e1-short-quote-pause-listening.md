# E1: audible pause around a short embedded quote (issue #114, owner decision follow-up)

The owner's requirement (dsl-spec.md sec. 2.9, sec. 7) was explicit and stated on the
strength of listening to `output/chapter-114-e0/chapter.wav`: whatever mechanism marks a
short quote embedded in narration, **the result must be an audibly distinct pause, not
the smooth, blended transition the un-marked Э0 narration had.** This document records
what was generated to let the owner judge that by ear, plus a numeric proxy measurement
that is explicitly *not* a substitute for that judgment.

**This agent cannot hear the generated audio.** Nothing below should be read as "I
listened and confirmed it sounds right" — only the owner's own listening pass in the
sentiment survey app is that check. What follows is (a) what was generated and how, and
(b) a transparent numeric signal computed from the waveform, offered only as a secondary,
reproducible data point alongside the owner's ear, never in place of it.

## What was generated

Reproduction script: `docs/research/audiobook/e1_short_quote_pause_probe.py`
(`.venv-tts/bin/python3 docs/research/audiobook/e1_short_quote_pause_probe.py`, real
Higgs TTS 3 via `mlx_audio`, same model/call convention as `src/audiobook.py`).

The exact sentence the owner flagged, from `samples/audiobook/prepared/sb-1-19.txt`:

> Все великие мудрецы, собравшиеся там, восторженно приняли решение Махараджи
> Парикшита и выразили свое одобрение словами: «Очень хорошо!»

- **`output/m4_dsl_short_quote/before.wav`** — this sentence exactly as it sits in the
  Э0 fixture today (one un-split `#prose` paragraph, no pause markup at all).
- **`output/m4_dsl_short_quote/after.wav`** — the same sentence compiled from a real
  `.abs` snippet through the actual compiler (`scripts/audiobook_dsl.py`), using the
  decided mechanism (dsl-spec.md sec. 2.9): `[пауза]` immediately before and after the
  quote, inside the same `#prose` block:

  ```
  #prose
  Все великие мудрецы, собравшиеся там, восторженно приняли решение Махараджи
  Парикшита и выразили свое одобрение словами: [пауза] «Очень хорошо!» [пауза]
  ```

  which compiles to (verified via `dsl.compile_source`, not hand-typed):

  ```
  ...словами: <|prosody:pause|> «Очень хорошо!» <|prosody:pause|>
  ```

Both clips: 24000 Hz mono PCM16, generated with `temperature=1.0, max_new_tokens=4096`,
no batching (single `model.generate()` call each), on the same model load. `before.wav`:
10.44s audio for 136 characters of input. `after.wav`: 12.12s audio for the same 136
spoken characters plus the two pause tags.

## Where the owner can listen

`src/sentiment_survey/task_sets/dsl_short_quote_pause.json` adds one `pair_compare` task
(`dsl-short-quote-pause-audible`) pointing at these two clips, discoverable the same way
as every other hand-curated task set: `make sentiment-survey`, then the usual survey UI.
The question asked matches the owner's own wording of the requirement ("отчётливо,
отдельно... а не слитно").

## Numeric proxy (not a listening check)

`e1_short_quote_pause_probe.py` also runs a simple, transparent heuristic over each
clip: 10ms RMS windows, anything under a fixed amplitude threshold counted as "quiet",
consecutive quiet windows of at least 40ms reported as a silence run. This is a crude
energy-based measure, not a real VAD, and it flags *every* natural inter-word/inter-
clause pause in normal speech, not just the one around the quote — so the useful signal
is not "does after.wav have a silence" (both clips have many, from ordinary sentence
prosody) but **whether `after.wav` has a silence run near the quote's expected position
that is longer than anything `before.wav` produces anywhere in the sentence.**

Measured (full output in `docs/research/audiobook/e1_short_quote_pause_probe.py`'s run
log; both lists below are every silence run >= 40ms the heuristic found in each clip):

- **`before.wav`** (10.44s total): 18 silence runs detected, longest non-trailing run
  **420ms** (at ~2.45s and again at ~6.41s — ordinary comma/clause pauses), longest
  overall **400ms** near the end (~9.0s, plausibly around "словами:" before the quote).
  Nothing over ~420ms anywhere in the clip.
- **`after.wav`** (12.12s total): two runs stand out from the rest — **630ms** at
  ~6.39s and **740ms** at ~9.21s — both noticeably longer than `before.wav`'s longest
  anywhere in the sentence (420ms), and both sit close to where the quote is expected
  given the sentence's proportional position (the quote is the last ~12% of the spoken
  text). A third long run (960ms at ~10.99s) is most likely trailing silence at the
  clip's end, not part of the mid-sentence effect being measured.

**Read as a proxy, not a verdict**: this is consistent with the pause tags producing a
longer gap than this sentence's own natural prosody does anywhere else, which is the
kind of thing that could sound "audibly distinct" rather than "blended" — but a 600-
900ms energy-based gap is not the same claim as "a human ear finds this natural and
clearly separated," and the heuristic has no way to tell a real perceptual pause from,
e.g., a soft trailing consonant. **The owner's listening pass on the task set above is
the actual check this requirement asked for.**
