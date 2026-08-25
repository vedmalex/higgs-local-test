# M4 mid-sentence split probe: does cutting a sentence mid-clause create an audible defect?

Issue #57 follow-up. See `docs/research/audiobook/m4-tag-inventory-results.md` §5 item 6 for
why the earlier `boundary_1_complete_thought.wav` / `boundary_2_continuing_thought.wav` pair
did **not** actually test this question (both texts were grammatically complete sentences
ending in a period; only their *narrative* continuation was open). This document reports the
first real test of the actual risk: `_force_split_long_sentence()` in `src/audiobook.py`
(the `--max-chars` overflow path, F2) can cut a single long sentence at a `;`, then a `,`,
then a whitespace boundary — **never at a sentence end**, because by construction it only runs
on a sentence that is already too long to fit as a whole chunk. If the model's terminal
intonation falls the way it does at a real full stop even when cut at a comma, a listener
would hear a false completion in the middle of a sentence, and the join would be audible.

## Method

Reused `samples/audiobook/prepared/sb-1-19.txt` (owner's existing example text — narration
about the sages recognizing Śukadeva Gosvāmī), verbatim, not invented text. One real sentence:

> Хотя он и старался скрыть свое естественное величие, великие мудрецы, собравшиеся там, были
> искушены в физиономистике и потому почтили его, поднявшись со своих мест.

Cut after `"...естественное величие,"` — a comma before the main clause, exactly the kind of
boundary `_force_split_long_sentence()`'s `,`-split branch would choose, and genuinely not a
sentence end (the first half is a bare subordinate clause starting with "Хотя", grammatically
incomplete on its own — unlike the earlier flawed probe).

Script: `docs/research/audiobook/m4_midsentence_split_bench.py`. Generated with the SAME pinned
voice reference throughout (`--voice-name narrator_e0` equivalent, PR #133's
`ref_audio_codes`/`ref_text`, `voices/narrator_e0.npy`) so voice drift between calls cannot be
mistaken for a prosody effect:

- `whole.wav` — the full sentence, ONE `model.generate()` call (control: what this material
  sounds like when the model knows the sentence continues).
- `fragment.wav` — only the first half, generated ALONE — exactly what `generate_segments`
  would produce for an independent chunk with no knowledge of what follows.
- `remainder.wav` — the second half, generated ALONE (needed to build the spliced clip).
- `spliced.wav` — `fragment.wav` + 200 ms silence (`assemble_chapter()`'s default `silence_ms`)
  + `remainder.wav` — reproduces exactly what chapter assembly would output for this cut.

Raw numbers: `logs/m4_midsentence_split.json`. Machine: M1, 16 GB, MLX. `peak_mlx_gib≈10.5`,
model load 9.7 s, run wall 133 s for 3 generate() calls (`whole` 9.4 s audio/40.2 s wall,
`fragment` 2.88 s audio/59.1 s wall, `remainder` 7.16 s audio/32.4 s wall — `fragment`'s wall
time is disproportionate to its short output; likely decode running toward
`max_new_tokens=4096`'s stopping behavior on a short, comma-terminated prompt rather than
anything specific to this text, not otherwise investigated here).

Analysis reuses `docs/research/audiobook/m4_prosody_metrics.py`'s primitives (`read_wav`,
`frame_signal`, `autocorr_f0`, `detect_pauses`, `energy_db`) — no second F0/pause estimator was
written. **Caveat repeated from that script's own docstring: plain autocorrelation on 40 ms
frames, no librosa/parselmouth. Good for "does the contour rise or fall", not for exact Hz —
octave errors are visible in the raw traces below (isolated jumps to ~250 Hz and ~400 Hz that
are almost certainly doubled/halved octave misreads, not real pitch).**

## What was measured (objective, n=1 — one sentence, one cut point)

1. **F0 trend over the last 600 ms of `fragment.wav`** (the isolated cut clip): slope
   **+217.9 Hz/s** — RISING toward the end, from ~81 Hz up to ~160 Hz over roughly the first
   400 ms of that window, before trailing off into a noisy, partly-unvoiced tail (see raw trace
   in `logs/m4_midsentence_split.json`). A genuine sentence-final cadence would fall, not rise;
   this trace does not show a fall.
2. **F0 trend at the same textual position inside `whole.wav`**, located by proportional
   character-offset (52 of 165 chars ⇒ ≈3.02 s of 9.4 s — **not forced alignment, an
   approximation**, flagged in the script's output as such): slope **+91.7 Hz/s** over the
   600 ms window ending there — also rising, and the last ~10 frames before the estimated cut
   point climb fairly steadily (121→207 Hz). Noisy (several ~250 Hz octave-artifact frames in
   the middle of the window), but the qualitative direction agrees with `fragment.wav`: rising,
   not falling, at this comma.
3. **The model's own natural pause at this comma, inside `whole.wav`**: ≈100 ms of
   below-threshold energy near the estimated cut position — under `detect_pauses()`'s 150 ms
   minimum, so it does not show up as one of `whole.wav`'s 2 counted pauses (380 ms total,
   elsewhere in the sentence). The artificial join silence inserted into `spliced.wav` is
   **200 ms — roughly double** this natural comma-pause length.

**Reading of the objective numbers**: on this one example, the F0 contour does NOT show the
"false full stop" signature (a falling terminal cadence) that would make a mid-clause cut sound
like a genuine sentence end — both the isolated fragment and the same position inside the whole
sentence show a rising, continuation-like contour. The one concrete, measurable defect found is
timing, not pitch: the assembly's fixed 200 ms join silence is about 2x longer than the model's
own natural pause at a comma, which could read as a slightly unnatural hesitation at the splice
even if the pitch itself doesn't lie about completion.

This is a single case (n=1, one sentence, one cut point, one voice). It is evidence, not proof,
and the F0 estimator's own noise (octave jumps visible in the traces) means small effects are
not resolvable at all with this tool.

## What the owner needs to listen for (I cannot hear the clips)

All 4 clips are in `output/m4_midsentence_split/` (`fragment.wav`, `remainder.wav`,
`whole.wav`, `spliced.wav`) and also loaded blind into the sentiment-survey app as a new task
set, **`src/sentiment_survey/task_sets/midsentence_split.json`** (a separate file — the shared
task_sets files were left untouched per the concurrent-work boundary):

- `midsplit-false-fullstop` — listen to `fragment.wav` alone; does the ending sound finished or
  cut off? (Objectively "cut off" is correct here — the text is a bare subordinate clause with
  no main clause and no terminal punctuation, unlike the earlier flawed
  `final-boundary-continuing` task.)
- `midsplit-seam-audible` — `spliced.wav` vs `whole.wav`, blind, order randomized: which one has
  an audible seam?

Run via `make sentiment-survey` (`docs/guides/sentiment_survey_guide.md`). If the owner hears no
seam and no false stop, that directly confirms the objective reading above: mid-clause force
splits are safe on this kind of material. If the owner DOES hear a seam or a false stop despite
the F0 numbers above, trust the ear over this uncalibrated estimator — that is exactly the
failure mode its own docstring warns about (octave/threshold errors), and it would mean the
defect is real but expressed in something this tool doesn't measure well (timbre discontinuity,
energy/loudness step, a pitch move too small for 40 ms-frame autocorrelation to resolve, etc).

## If the owner confirms a real defect: remediation options (not implemented here — owner picks)

1. **Move split points closer to real sentence/clause boundaries.** `_force_split_long_sentence`
   already prefers `;` then `,` over a hard whitespace/character cut (src/audiobook.py:493-544)
   — the mechanism to prefer "softer" boundaries already exists. Cost: none new to build;
   tuning is mostly about `--max-chars` (raising it reduces how often force-split triggers at
   all, since `chunk_sentences()` never splits WITHIN a sentence unless the sentence alone
   exceeds `max_chars` — see chunk_sentences docstring, src/audiobook.py:547). Trade-off: a
   larger `max_chars` means longer per-call generations (worse latency/memory per segment) and
   does not eliminate the case entirely — some real sentences (as here) legitimately exceed any
   practical budget.
2. **A harder per-piece cap specifically for force-split pieces**, separate from the general
   `max_chars` budget used for normal chunking — e.g. only invoke force-split when a sentence
   exceeds some larger threshold than the standard chunk budget, so it fires rarely rather than
   routinely. Cost: a new parameter/threshold, more surface to document and test; does not
   change what happens on the sentences that DO still trigger it.
3. **Pass context into generation, if the model accepts it.** Not verified in this project
   whether Higgs TTS 3 / MLX-Audio exposes any "this text continues" signal to the generator
   beyond the raw text (unlike prefix/continuation-aware ASR or streaming TTS APIs elsewhere).
   If it does not, this option requires a change in `mlx_audio`/the checkpoint's prompting
   contract, not just this project's code — should be checked against `PROMPTING.md` before
   committing to this path.
4. **Splice with overlap/crossfade instead of a fixed silence gap.** `assemble_chapter()`
   currently does a hard int16 concatenation with `silence_ms` of zeros between segments (no
   crossfade — src/audiobook.py:1575 area). A short crossfade could mask a small
   energy/timbre discontinuity at the join, and using the MEASURED natural pause length (≈100 ms
   here) instead of the fixed 200 ms default would remove the specific timing mismatch found
   above. Cost: assemble_chapter's splice-quality metrics (F3/F13, already computed from the
   small edge windows) would need to account for a variable or measured join length rather than
   the current fixed constant; a crossfade also changes the numeric splice-quality metric's
   definition and would need its own validation.

No option above was implemented in this task — the owner should confirm by ear first (this
agent cannot hear the clips), and choose based on the actual severity heard, since the
objective read here leans toward "not obviously broken on this one example."

## Open / not done

- n=1: one sentence, one cut point (comma before a main clause), one voice. Not tested: cuts at
  `;`, at a conjunction ("и"/"а"/"но"), inside a deeply nested clause, or on dialogue (the
  owner's brief specifically mentioned dialogue as available in the prepared samples — only
  narration was used here for the first probe).
- The "same textual position inside `whole.wav`" comparison uses a proportional character-offset
  estimate, not forced alignment (no aligner available in this project) — treat the ~3.02 s
  timestamp as approximate.
- `spliced.wav`'s join uses the assembly's default 200 ms silence; the ~100 ms "natural pause"
  number is itself a rough estimate from a fixed -40 dB-below-peak threshold, not a calibrated
  pause detector.
- Blind listening (`midsentence_split` task set in the survey app) has not been run by the owner
  as of this writing — the two tasks above are unanswered.
