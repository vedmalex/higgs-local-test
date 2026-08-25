# M4-T5 — Full 34-tag inventory, terminal intonation, and stress-mark notation: results

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio, `bosonai/higgs-tts-3-4b`). Scope: `docs/research/audiobook/m4-plan.md`
M4-T5, plus three owner additions made mid-task: (1) terminal/falling intonation at sentence
end and whether our own segmentation preserves it, (2) Russian stress-mark notation on Higgs
(never tested before this document), (3) a full sample dump for the owner's own listening.

**Before this document, only 4 of the 34 control tags had ever been checked at all**
(`docs/research/audiobook/m4-sentiment-results.md`), and one of those four
(`<|style:whispering|>`) turned out to be a near-total no-op on inspection. **This document is
the first pass over all 34.**

## 0. Honesty summary (read this first)

- **All 34 tags were generated and measured.** Nothing was skipped for time.
- **The final judgment on every tag is NOT made here.** Per the owner's brief, this document
  produces an objective triage (groups A/B/C, real numbers) and a prioritized listening list;
  whether a clip actually sounds like "anger" or "shame" is a human judgment this document
  explicitly does not make.
- **The terminal-intonation F0 slope metric is noisy and only a hint, not a verdict** — see §2.
  The autocorrelation pitch tracker produces genuine octave-jump artifacts near sentence-final
  creaky voice (documented with raw numbers below); the boundary-continuity question (§2.3),
  which matters most for the codebase, is flagged for listening rather than decided by metric.
- **Stress-mark notation on Higgs had never been tested before this run.** An earlier revision
  of `README.md` implied a cross-backend failure; that was wrong and has been corrected (§3.0).
- **One base carrier sentence for the stress "fabric" meaning (атлас) produced garbled
  transcripts across ALL notations** (§3.3) — this looks like a sentence-quality problem with
  that specific carrier, not a stress-notation effect, and is reported as inconclusive for that
  one word rather than folded into the notation verdict.
- Every number below is a real measurement from the actual runs; the raw JSON is at
  `logs/m4_tag_inventory.json` (generation) and `logs/m4_tag_inventory_metrics.json` (prosody +
  terminal-contour metrics) on this machine — **local only, not committed**, per this project's
  `logs/*`/`output/*` gitignore convention (same as every other M4 measurement doc). Re-run
  `docs/research/audiobook/m4_tag_inventory_bench.py` then
  `docs/research/audiobook/m4_tag_inventory_metrics.py logs/m4_tag_inventory.json` to reproduce.

## 1. Method

- **Generation**: `docs/research/audiobook/m4_tag_inventory_bench.py`, reusing PR #105's
  `model.batch_generate(texts=[...], temperature=1.0, max_new_tokens=4096)` batch=8 path
  instead of one-by-one calls — 76 clips in 10 batches. Aggregate RTF **1.58** (run wall 724.6s
  / 459.1s total audio), peak MLX **10.80 GiB**. Machine before: `load averages: 13.74 8.80
  7.25` (elevated from a just-finished, unrelated heavy run this project shares the host with;
  swap grew from ~2.0 GB to ~4.6 GB used over the run, no actual swap events reported by
  `/usr/bin/time -l`). One run for all 76 clips — GPU access was serialized against the
  project's other in-flight profiling run per the owner's explicit instruction (confirmed via
  `ps`/`uptime` polling before starting; one earlier attempt was killed by a 2-minute tool
  timeout mid-warm-up, before any generation batch executed, and produced no output).
- **Base text** (same for every emotion/prosody/style tag, so any acoustic difference is
  attributable to the tag, not the words): *"Сегодня я занимался повседневными делами. Утром я
  выпил чай и почитал книгу. Потом вышел на улицу и немного прошёлся."* — 3 sentences,
  semantically neutral.
- **Tag placement**: sentence-level tags (all 21 emotion, 8 of 10 prosody, all 3 style) are
  **reopened at the start of every sentence** in the base text, mirroring `chunk_sentences()`'s
  own reopening behavior in `src/audiobook.py` for a tag that must persist across an entire
  chunk — this measures the tag as it is actually used in production, not a weaker
  first-sentence-only variant. `<|prosody:pause|>` / `<|prosody:long_pause|>`
  (`INLINE_ONE_SHOT_PROSODY` in `src/audiobook.py`) are inserted **once**, between sentence 1
  and 2, never reopened, per `PROMPTING.md`.
- **Metrics**: `docs/research/audiobook/m4_prosody_metrics.py` (F0 median/std/range, energy,
  pause count/duration — unchanged, reused as-is) plus a new terminal-contour extension,
  `docs/research/audiobook/m4_tag_inventory_metrics.py`, described in §2. Per the owner's
  standing instruction, **F0 median is recorded but never used alone to draw a conclusion** —
  F0 std, pause count/duration, and tempo (words/sec) are the trusted discriminators.

## 2. Terminal / falling intonation at sentence end

### 2.1 Is there a dedicated tag for this?

**No.** The full 43-entry tag catalog in the checkpoint's own `PROMPTING.md`
(`~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b/snapshots/*/PROMPTING.md`) was read
in full for this document. It documents 21 emotion tags, 10 prosody tags (`speed_*`,
`pitch_*`, `expressive_*`, `pause`/`long_pause`), 3 style tags, and 9 inline `sfx` tags. **None
of it — no tag, no tip, no example — mentions sentence finality, a terminal/falling contour, or
punctuation-driven intonation.** The closest entries are `<|prosody:pitch_low|>` and
`<|prosody:expressive_low|>`, and neither is documented as a terminal-contour control — one is
a general pitch-register shift, the other a general expressiveness reduction. **Direct answer:
finality is not tag-controlled in this model; if it exists at all, it can only come from the
model's own punctuation-conditioned prosody.**

### 2.2 Does the model do this on its own? (punctuation probe)

Same one-sentence text, *"Я закрыл дверь и сел за стол"*, generated three times with only the
final punctuation mark changed (`.` / `?` / `!`). Terminal contour = linear-regression slope of
F0 (Hz/s) over the last ~300–400 ms of **actual speech before trailing silence** (silence
trimmed via the same energy threshold as pause detection, to keep silence-tail artifacts out of
the window — see §2.4 for why that trimming matters):

| Clip | Duration | F0 slope (Hz/s) | Direction | F0 start→end in window |
|---|---:|---:|---|---|
| `punct_period` (`.`) | 1.82s | +553.9 | rising | 91.6 → 375.0 Hz |
| `punct_question` (`?`) | 1.54s | +298.1 | rising | 156.9 → 142.0 Hz |
| `punct_exclaim` (`!`) | 2.20s | **-523.0** | falling | 369.2 → 76.7 Hz |

These three numbers are **not consistent with a simple "period falls, question rises" model** —
`punct_period` itself measured a large rise, and `punct_exclaim` measured the sharpest fall of
the three. Given the pitch tracker's known fragility near sentence-final creaky voice (§2.4),
this table is reported as a real measurement, not a conclusion: **it does not establish that
punctuation reliably drives terminal contour, and it does not rule it out either.** This is
exactly the kind of question that needs an ear, not a slope number — hence `punct_1/2/3` are in
the priority listening list.

### 2.3 Does OUR segmentation break it? (the important one — this is our code, not the model)

Two single, independently generated sentences, exactly as `generate_segments` would produce
them as two separate chunks with zero knowledge of each other:

- `boundary_complete`: *"Сегодня утром я пил чай и читал книгу."* — a genuinely self-contained
  thought (the normal, common case — `chunk_sentences()` only ever splits on a sentence
  boundary).
- `boundary_continuing`: *"Он медленно подошёл к двери и взялся за ручку."* — grammatically a
  complete sentence (so it still gets its own chunk under `chunk_sentences()`'s rules), but
  narratively a cliffhanger — the reader expects the thought to continue in whatever sentence
  comes next, which this clip deliberately does NOT include, matching exactly what
  `generate_segments` does in production (each chunk generated with no forward context).

| Clip | Duration | F0 slope (Hz/s) | Direction |
|---|---:|---:|---|
| `boundary_complete` | 2.76s | +127.1 | rising |
| `boundary_continuing` | 2.80s | -3.3 | flat |

**This is the opposite of the naive expectation** (complete thought → falling/settled;
continuing thought → held/rising) **and it is exactly the kind of result that must not be
trusted from the metric alone** — the underlying autocorrelation F0 track near an
utterance-final syllable is measurably noisy (§2.4). This pair is the single most important
listening item in this document for the codebase itself: if a human listener hears
`boundary_complete` as HELD/UNFINISHED, that is a real defect in the segmentation pipeline
(every independently-generated chunk boundary in every book would carry the same risk); if it
sounds properly finished despite the "rising" number, the metric is simply not sensitive enough
to catch it and no code defect is implied.

### 2.4 Why the terminal-slope numbers get an explicit noise warning

Directly inspecting `neutral_baseline`'s last ~400ms of nominally-voiced F0 track (before the
silence-trim fix was added) showed: 152.9 → 102.6 → 110.6 → **400.0 (a raw autocorrelation
octave-jump artifact)** → 65.9 → 73.4 → 99.6 → … → 166.7 Hz, all within about 340ms. This is not
a plausible pitch contour for a human voice; it is the known failure mode of a simple
autocorrelation pitch tracker applied to a devoicing/creaky-voice tail, which is common exactly
at sentence-final position — the region this whole section needs to measure. `m4_tag_inventory_metrics.py`
was extended (relative to `m4_prosody_metrics.py`) to (a) trim trailing silence via the same
energy threshold `detect_pauses` already uses before selecting the terminal window, and (b)
median-filter the F0 track before the linear fit — both applied uniformly to every clip in this
document, tags included — but neither eliminates the underlying tracker fragility, which is why
the slope numbers in §2.2/§2.3 are reported as measurements, explicitly not verdicts.

## 3. Stress-mark notation on Higgs (owner addition; first time tested on this backend)

### 3.0 Correcting the record — this experiment was never run on Higgs before today

`README.md` (until this change) read as if a stress-notation workaround had "actively made it
worse" for **both** backends. That was wrong, and the owner is the one who caught it. The
actual prior history, confirmed by re-reading `docs/research/qwen3-tts-notes.md`'s "Stress
control" section and the artifact filenames it produced:

- The motivation paragraph names the model explicitly: *"`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`
  (the local Apple Silicon path above) mispronounces stress..."* — **Qwen3-TTS only**.
- Every garbled example (`Вриндава́н` → "ВриндаваЮн", `+Это` → "Plus, Эйхолог") is from that same
  Qwen run.
- The artifacts are literally named `output/qwen_tts_ru_basic.wav` etc. — there is no
  `higgs_*` artifact from that experiment, because Higgs was never run in it.
- The only sentence in that section that legitimately covers both backends is the *documentation*
  claim: *"Neither Higgs nor Qwen3-TTS documents a stress-control mechanism"* — a statement about
  missing docs, not about measured behavior.

`README.md` has been corrected in this change (see the diff) to say plainly: undocumented on
both backends; experimentally-confirmed-worse on Qwen only; **Higgs's actual behavior was
untested until this document.**

### 3.1 What was tested

3 Russian homograph pairs where stress position changes the word's meaning, each word rendered
in 6 notations, in a disambiguating carrier sentence (36 clips total):

| Pair | Meaning 1 | Meaning 2 | Carrier template |
|---|---|---|---|
| замок | за́мок "castle" | замо́к "lock" | "На холме стоит старинный {}." / "На двери висит крепкий {}." |
| стоит | сто́ит "costs" | стои́т "stands" | "Эта книга дорого {}." / "Этот дом давно {} пустым." |
| атлас | а́тлас "atlas (book)" | атла́с "satin (fabric)" | "На полке лежит географический {}." / "Платье сшито из блестящего {}." |

Notations: combining acute accent U+0301 (`за́мок`), capital letter on the stressed vowel
(`зАмок`), apostrophe after the stressed vowel (`за'мок`), `+` before the stressed vowel
(`з+амок`), doubled stressed vowel (`заамок`), and a bare unmarked control (`замок`).

### 3.2 Does the notation break our own text pipeline? (static check, no model call)

Checked directly against `src/audiobook.py`'s `split_sentences` and `validate_control_tags` on
all 6 notations of the "замок" pair (representative text with normal end-of-sentence
punctuation): **all 6 notations split into exactly 2 sentences as expected, and
`validate_control_tags` raised nothing for any of them** — none of the notation characters
(combining acute, capital letters, apostrophe, `+`, doubled letters) collide with the tag-shape
regex or the quote-parity tracker (which only tracks `"`, `„`, `“`, `‘`, `«`, etc. — a bare `'`
is not one of those paired characters). This check used representative sentences, not an
exhaustive fuzz of every possible position (e.g. a stress mark placed adjacent to an actual `"`
quote was not separately tested) — flagged as untested, not claimed as proven for every case.

### 3.3 Is the notation spoken aloud literally? (STT round-trip check, `src/stt_qwen_local_test.py`)

Every one of the 36 clips was transcribed back with Qwen3-ASR (fast, RTF ≈0.09–0.44 here) and
the transcript inspected for the README:245 failure signature (the model vocalizing the mark
itself, e.g. a spelled-out symbol name).

| Notation | Literal vocalization of the mark? | Notes |
|---|---|---|
| acute (U+0301) | **No** in 5/6 words | `stress_atlas_book_acute` transcribed as Latin-script "atlas" (see caveat below) |
| capital letter | **No** in 5/6 words | same `atlas_book` anomaly; other 5 clean |
| apostrophe | **No**, 6/6 clean | cleanest notation across all six words |
| `+` before vowel | **Yes, 1/6** | `stress_stoit_stands_plus` ("сто+ит") transcribed as the garbled non-word "оступлить" — same failure class as Qwen's `+Это` → "Plus, Эйхолог", on a smaller scale |
| doubled vowel | **Possibly 1/6** | `stress_zamok_lock_doubled` ("замок" doubled) transcribed with an extra vowel, "замаок" — could be the doubling literally adding a syllable, needs a listen |
| none (control) | n/a | `atlas_book` anomaly also appears here (see below) |

**The `атлас` "book" meaning is an unexplained anomaly, not a notation failure**: `acute`,
`capital`, and `none` (3 of 6 notations, including the unmarked control) all transcribed back as
the *Latin-script* word "atlas" rather than Cyrillic "атлас" — i.e. the ASR heard something
foreign-sounding regardless of stress marking. Because this happens with the bare, unmarked
control too, it cannot be a stress-notation artifact; it looks like a word-specific
pronunciation quirk (possibly the model treating "атлас" as a loanword) and is reported as
**inconclusive**, not folded into any notation's verdict.

**The `атлас` "fabric" meaning (carrier: "Платье сшито из блестящего {}.") produced garbled
transcripts across every single notation** (e.g. "Клатя изшита из блестящего атласа.",
"Накершито из блестящего атлас"), including the unmarked control. This points to a
carrier-sentence quality problem (possibly the consonant cluster in "сшито"), not a
stress-notation effect, and this word/carrier pair is excluded from the notation verdict below.

### 3.4 Does the notation actually shift stress to the intended syllable? (cannot be answered here)

**This is the central question and objective metrics cannot answer it** — Russian ASR output
does not mark stress, exactly as the owner anticipated. Duration/energy per syllable were
considered as an objective proxy but were not computed for this document (time-boxed); **a
human listening comparison of the paired clips in `output/m4_tags/04_stress/` is the only way to
answer this**, and that pairing is built into the priority listening list.

### 3.5 Verdict table

| Notation | Spoken literally? | Meaning applied? | Breaks our text pipeline? | Verdict |
|---|---|---|---|---|
| acute (U+0301) | No (usable words clean) | **Unknown — needs listen** | No | Candidate — listen to confirm |
| capital letter | No (usable words clean) | **Unknown — needs listen** | No | Candidate — listen to confirm |
| apostrophe | No, cleanest of all 6 | **Unknown — needs listen** | No | Best-behaved candidate so far — listen to confirm |
| `+` before vowel | **Yes, at least once** (mirrors Qwen's known failure) | Unknown | No (static check) | Risky — likely unreliable, same failure class as Qwen |
| doubled vowel | Possibly once (extra syllable) | Unknown | No | Candidate, one flagged anomaly — listen to confirm |
| none (control) | n/a | n/a (no attempt to mark) | n/a | Baseline only |

**No notation can be declared fit for the book from this document alone** — the STT check only
rules out the "spoken literally" failure mode (and does so for acute/capital/apostrophe/doubled
in the overwhelming majority of cases; `+` shows the same failure class Qwen had). Whether
stress actually moves is unverified pending listening; the `04_stress/` folder and `LISTEN.md`
are built specifically to make that judgment fast.

## 4. The 34-tag inventory: groups A/B/C

Delta columns are `clip - neutral_baseline` (neutral: F0 std 60.3 Hz, mean energy -24.5 dB,
2.23 words/sec, 1 pause / 540ms total, F0 median 134.1 Hz, duration 8.52s). Per the owner's
standing rule, **F0 median deltas are shown but never used alone to assign a group** — grouping
uses F0 std, pause count/duration, and tempo (words/sec) as the trusted signals, cross-checked
against duration.

- **A = clear, direction-consistent objective signal.**
- **B = weak, mixed, or direction-ambiguous signal** — plausible but not clean.
- **C = signal close to neutral (or in the wrong direction) on the metric that should show it
  most directly** — suspected weak/no effect, same class as the previously-confirmed
  `whispering` pustyshka.

| Group | Tag | ΔF0 std (Hz) | ΔPauses (n) | ΔPause (ms) | ΔTempo (wps) | ΔF0 median (Hz)* | ΔDuration (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | emotion:affection | +0.8 | +7 | +2680 | -0.72 | +39.8 | +4.04 |
| A | emotion:amusement | +14.1 | +5 | +1660 | -0.77 | +92.3 | +4.52 |
| A | emotion:awe | +26.3 | +7 | +1980 | -0.71 | +7.1 | +4.00 |
| A | emotion:bitterness | +8.5 | +5 | +1640 | -0.52 | -22.0 | +2.56 |
| A | emotion:contemplation | -4.1 | +5 | +1980 | -0.58 | +27.0 | +3.00 |
| A | emotion:contentment | +17.3 | +7 | +2100 | -0.75 | +82.1 | +4.36 |
| A | emotion:determination | +64.3 | +0 | -280 | -0.40 | -9.1 | +1.88 |
| A | emotion:disgust | -11.9 | +5 | +1400 | -0.63 | +9.6 | +3.32 |
| A | emotion:elation | +17.1 | +3 | +400 | -0.36 | +88.1 | +1.64 |
| A | emotion:fear | -11.0 | +3 | +1460 | -0.48 | +126.8 | +2.36 |
| A | emotion:pride | +28.8 | +3 | +380 | -0.75 | -1.9 | +4.36 |
| A | emotion:relief | +6.2 | +7 | +2340 | -0.59 | +7.1 | +3.04 |
| A | emotion:sadness | -26.0 | +5 | +1780 | -0.50 | -3.7 | +2.44 |
| A | emotion:shame | -16.8 | +2 | +300 | -0.05 | +28.1 | +0.20 |
| A | emotion:surprise | +21.7 | +1 | +60 | -0.01 | +43.0 | +0.04 |
| B | emotion:arousal | -6.4 | +4 | +660 | -0.55 | +76.4 | +2.80 |
| B | emotion:confusion | -10.9 | +5 | +1580 | -0.11 | +32.6 | +0.44 |
| B | emotion:helplessness | -2.5 | +1 | +440 | -0.16 | -55.9 | +0.68 |
| B | emotion:longing | -10.7 | +4 | +800 | -0.37 | +5.4 | +1.72 |
| C | emotion:anger | +2.5 | +4 | +320 | -0.16 | +59.4 | +0.68 |
| C | emotion:enthusiasm | -2.0 | +2 | +60 | -0.23 | +51.9 | +0.96 |
| A | prosody:expressive_low | -10.6 | +6 | +1640 | -0.42 | +3.8 | +1.96 |
| A | prosody:long_pause | -9.6 | +3 | +1100 | -0.17 | -34.9 | +0.72 |
| A | prosody:pitch_high | -3.9 | +2 | +120 | +0.01 | +173.6 | -0.04 |
| A | prosody:pitch_low | +47.3 | +7 | +1340 | -0.72 | -44.5 | +4.04 |
| B | prosody:expressive_high | +9.3 | +3 | +1040 | -0.43 | -11.3 | +2.04 |
| B | prosody:pause | +14.4 | +5 | +1400 | -0.46 | -7.8 | +2.24 |
| C | prosody:speed_fast | -21.1 | +3 | +980 | **+0.11** | +0.0 | -0.40 |
| C | prosody:speed_slow | -29.3 | -1 | -540 | **-0.03** | -34.7 | +0.12 |
| C | prosody:speed_very_fast | -17.8 | +2 | +220 | **-0.08** | -0.4 | +0.32 |
| C | prosody:speed_very_slow | -5.6 | +1 | +560 | **+0.02** | -15.3 | -0.08 |
| A | style:shouting | +1.5 | +3 | +640 | -0.28 | +135.6 | +1.20 |
| B | style:singing | +9.9 | +2 | +300 | -0.22 | -23.0 | +0.92 |
| C | style:whispering | +0.8 | -1 | -540 | -0.45 | -27.4 | +2.16 |

**Headline finding: all four `speed_*` prosody tags are Group C.** Words-per-second — the metric
built specifically to detect a tempo change — sits within ±0.11 wps of neutral for every one of
them, and `speed_very_fast` (-0.08 wps, +0.32s duration) and `speed_very_slow` (+0.02 wps,
-0.08s duration) both move in the **wrong direction** from what their names promise. This
matches `PROMPTING.md`'s own tip that `speed_very_slow` "only slows the model to roughly ~5s,"
suggesting these tags are weakly implemented or get overridden by other decoding factors at
`temperature=1.0` — but this document only measures the effect, not why. `<|prosody:pause|>`
(Group B) also shows *more* total pause time (1940ms) than `<|prosody:long_pause|>` (1640ms,
Group A) despite the name implying the reverse — flagged, not resolved, pending a listen to
`prosody_pause.wav` / `prosody_long_pause.wav` for actual pause placement/length.

`style:whispering` (Group C) reconfirms the pre-existing finding from
`m4-sentiment-results.md`: energy is *higher*, not lower, than neutral (+3.5 dB in this run's
carrier text), and pitch spread is essentially unchanged (+0.8 Hz) — consistent evidence across
two independent generations that this tag is a near no-op.

## 5. Priority listening list

See `output/m4_tags/LISTEN.md` for the full guide (local path only — `output/` is gitignored,
nothing here is committed). The 8 highest-priority items, condensed:

1. `03_style/style_whispering.wav` vs neutral — confirmed-suspect pustyshka.
2. `02_prosody/prosody_speed_slow.wav` vs neutral — near-zero measured tempo change.
3. `02_prosody/prosody_speed_very_fast.wav` vs neutral — measured tempo change is *backwards*.
4. `01_emotion/emotion_sadness.wav` vs neutral — sanity-check clip: largest, most direction-consistent Group A signal.
5. `04_stress/stress_zamok_acute_1_castle.wav` vs `..._acute_2_lock.wav` — does stress notation
   actually shift meaning?
5b. `04_stress/stress_zamok_plus_1_castle.wav` vs `..._plus_2_lock.wav` — the notation with a
   confirmed STT-transcript corruption elsewhere.
6. `05_final_intonation/boundary_1_complete_thought.wav` vs `..._2_continuing_thought.wav` —
   **the segmentation-defect question**: does our own chunk-per-chunk generation leave sentences
   sounding cut off.
7. `05_final_intonation/punct_3_exclaim.wav` vs `punct_1_period.wav` — does punctuation alone
   change delivery.

## 6. What was NOT verified (honesty)

- **No tag's emotional/stylistic identity was confirmed by ear in this document.** Groups A/B/C
  are an objective triage only, exactly as scoped by the owner.
- **Terminal-contour F0 slope is a noisy proxy** (§2.4); the segmentation-boundary question
  (§2.3), which is the one about our own code, is explicitly left to a human listener rather
  than decided from the metric.
- **Stress-shift itself (does the meaning actually change) is unverified** for all 3 homograph
  pairs and all 6 notations — Russian ASR cannot detect stress placement, so only a human
  listen can answer it (§3.4).
- **Duration/energy-per-syllable was not computed** for the stress probe (would have been a
  cheaper objective proxy for stress placement than nothing, but was time-boxed out of this
  pass).
- **The atlas "fabric" carrier sentence is unusable as constructed** — garbled across every
  notation including the control — and its stress-notation results are excluded from the
  verdict rather than misreported as evidence against any notation.
- **A stress mark colliding with an actual quote character was not tested** (§3.2) — only
  representative sentences without quotes were checked against `split_sentences`.
- Machine contention: this run shares the host with another in-progress heavy profiling task;
  GPU access was serialized (confirmed via polling before starting), but the "before" load
  average (13.74) reflects residual system load from that task's immediately-preceding run, not
  a clean-machine baseline — recorded here rather than smoothed over.
