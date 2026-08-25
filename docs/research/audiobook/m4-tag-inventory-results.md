# M4-T5 — 34-of-43-tag inventory, terminal intonation, and stress-mark notation: results

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio, `bosonai/higgs-tts-3-4b`). Scope: `docs/research/audiobook/m4-plan.md`
M4-T5, plus three owner additions made mid-task: (1) terminal/falling intonation at sentence
end and whether our own segmentation preserves it, (2) Russian stress-mark notation on Higgs
(never tested before this document), (3) a full sample dump for the owner's own listening.

**Correction (2026-08-25, later same-day pass, issue #57 sfx tag fix):** at the time this
document was written, the checkpoint's control-tag inventory was itself believed to be 34 tags
(21 emotion + 10 prosody + 3 style) — the `<|sfx:*|>` category (9 tags: `cough`, `laughter`,
`crying`, `screaming`, `burping`, `humming`, `sigh`, `sniff`, `sneeze`; ids 151707-151715,
documented in the checkpoint's `PROMPTING.md` "Full tag catalog (43)") was missed entirely, both
in this document and in `src/audiobook.py`'s `VALID_TAGS` (which rejected all 9 as "unknown"
until the fix). **This document's "full inventory" claim below is therefore false as originally
written: this is a triage of 34 of the checkpoint's real 43 control tags, not all of them.** The
9 `sfx:*` tags remain completely unchecked by this document or any other — no generation, no
metrics, no listening pass. That is separate, still-open work; see §6.

**Before this document, only 4 of the (as later corrected) 43 control tags had ever been
checked at all** (`docs/research/audiobook/m4-sentiment-results.md`), and one of those four
(`<|style:whispering|>`) turned out to be a near-total no-op on inspection. **This document is
the first pass over the 34 emotion/prosody/style tags — not, as originally claimed, over the
full inventory; the 9 `sfx:*` tags were not covered.**

## 0. Honesty summary (read this first)

- **All 34 emotion/prosody/style tags were generated and measured.** Nothing was skipped for
  time. **The 9 `sfx:*` tags were not part of this run at all** — they were not yet known to be
  part of the checkpoint's real 43-tag inventory when this document was written (see the
  correction above). This is separate, unfinished work.
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
  **Update (2026-08-25, post-listening): the owner has now listened to the `04_stress/` clips
  and delivered a verdict — apostrophe notation works, and stress marking is NOT optional for
  a Russian book. See §3.6.**
- **Update (2026-08-25, post-listening): the owner also listened to the emotion-tag clips and
  reports emotions are broadly distinguishable and usable — see §4.1.** This is a cautious,
  not a strong, confirmation (see §4.1 for exact wording and scope).
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

### 3.6 Owner's listening verdict (2026-08-25) — SUBJECTIVE, human judgment

The owner listened to the `04_stress/` clips (and others) and delivered this verdict, quoted
verbatim per this project's convention of preserving subjective human judgments word-for-word
rather than paraphrasing them:

> «там где нет ударений, он выбирает случайное ударение
> двойные буквы проговариваются как двойные буквы - удлиняя звук
> плюс иногда проговаривает букву п
> апостроф дает ударение
> заглавная буква иногда создает ударение, а иногда создает ощущение акцента на слове
> медленный голос не всегда сильно медленнее
>
> причем видно что если ударение непроставлено, то он выбирает свое, и не всегда правильное»

**This is a subjective human judgment, not a re-run of the objective metrics above.** It
resolves §3.4's "cannot be answered here" and §4's `speed_slow` open question (see §4.1a), but
it is scoped to exactly what the owner listened to: the 3 homograph pairs and their 6 notations
in `04_stress/`, plus the `speed_*` clips. It is **not** a general claim about Russian stress
behavior beyond those words — see the Honesty note at the top of this document.

| Нотация (as tested in §3.1) | Owner's verdict | Interpretation |
|---|---|---|
| **апостроф** (`за'мок`) | **РАБОТАЕТ — даёт ударение.** | The one notation confirmed both objectively (§3.3, cleanest STT round-trip) and now subjectively (correct stress placement). This is the only notation this document can now call fit for use. |
| удвоение гласной (`заамок`) | НЕ ударение — проговаривается как удвоенная буква, удлиняя звук. | Corrects §3.5's "candidate, needs listen" — by ear this is not a stress mechanism at all, it is a literal vowel-length effect. See §3.7 for why the STT round-trip in §3.3 could not have caught this. |
| `+` перед гласной (`з+амок`) | Иногда произносится вслух как буква «п» — provisionally read as the same class of failure §3.3 already caught once (`сто+ит` → «оступлить») and that Qwen's `+Это` → «Plus, Эйхолог» failure represents; the owner's exact wording names the letter "п", not the word "плюс", which may be a distinct artifact of Higgs's tokenizer rather than literally the same failure — recorded verbatim rather than reinterpreted. | Confirms §3.5's "risky — likely unreliable" verdict; do not use. |
| заглавная буква (`зАмок`) | Ненадёжна: иногда даёт ударение, иногда только смысловое выделение слова (аналог логического ударения/акцента), а не сдвиг словесного ударения. | Downgrades §3.5's "candidate — listen to confirm" to **not reliable**: the effect exists but is not consistently the effect this project needs (a specific-syllable stress cue). |
| без пометки (`замок`, control) | **Модель выбирает ударение сама, случайно, и не всегда верно.** | This is the load-bearing finding for §3.8 below: leaving a homograph unmarked is not a neutral/safe default — it is an unforced, occasionally-wrong guess. |

Separately, on `<|prosody:speed_slow|>`: *«медленный голос не всегда сильно медленнее»* — see
§4.1a, which folds this into the resolution of the `m4-sentiment-results.md` vs. this
document's contradiction on that tag.

### 3.7 Lesson: the STT round-trip proxy has a real, now-demonstrated blind spot

`m4-sentiment-results.md`'s predecessor investigation into Qwen3-TTS
(`docs/research/qwen3-tts-notes.md`, "Stress control" section) judged doubled-vowel notation
(`Вриндаваан`) as "the least-bad of the three" workarounds because the ASR round-trip came back
"close to the clean baseline" — i.e. it did not introduce spurious syllables/garbling the way
U+0301 and `+` did. That document was explicit that this only showed the workaround "doesn't
damage the *segmentation* of the audio, not that it corrects the *stress*" and flagged it as
needing "an actual listen before being treated as anything more than 'didn't make it worse.'"
The owner's listen has now happened (§3.6, this document, on Higgs rather than Qwen), and the
answer is exactly the failure mode that caveat anticipated: doubled-vowel notation does not
move stress at all — it lengthens the vowel sound, which a speech-recognition transcript
literally cannot represent (ASR outputs normalized spelling, not duration), so the proxy read
it as harmless when by ear it was never doing the intended job in the first place.

**General lesson for this project**: an objective proxy metric (here, ASR transcript fidelity)
can only detect the failure modes it is structurally capable of representing. A proxy that
cannot represent duration, stress placement, or pitch contour will silently pass an intervention
that fails on exactly that axis. Every "clean STT round-trip" finding in this document and in
`m4-sentiment-results.md` should be read as "did not corrupt words/segmentation," never as "did
what the tag/notation claims to do" — the two are different questions, and the doubled-vowel
case is the concrete proof that they can disagree.

### 3.8 Practical conclusion: stress marking is NECESSARY, not optional, for this book

Because §3.6 establishes that an unmarked homograph is resolved by the model **guessing**, and
that guess is **not always correct**, stress-mark placement is a correctness requirement for a
Russian-language book on this backend, not a nice-to-have quality improvement. A homograph left
unmarked does not fail safely (it does not, say, default to the more common reading) — it is
non-deterministic per the owner's own listening, which means the same unmarked chapter text
could plausibly render a wrong-meaning word on one generation and a right-meaning word on
another. The only notation confirmed to reliably move stress to the intended syllable, by both
the objective STT check (§3.3) and the owner's ear (§3.6), is the apostrophe. Pipeline support
for apostrophe-as-stress-mark is implemented in `src/audiobook.py` (see the module-level
comment above `RUSSIAN_VOWELS_LOWER`) and documented for the scenario author in
`docs/guides/audiobook_guide.md`.

This conclusion is scoped honestly: it rests on 3 homograph pairs and 6 notations, not a survey
of Russian stress behavior in general (see the Honesty note at the top of this document and
§6 below).

## 4. The 34-of-43-tag inventory: groups A/B/C

(Covers only the 21 emotion + 10 prosody + 3 style tags. The 9 `sfx:*` tags are not in this
table and have no group assignment — see the correction note above and §6.)

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

**Headline finding (revised 2026-08-25, see §4.1a): all four `speed_*` prosody tags are Group
C on the objective metric, but this is a weak/inconsistent effect, not a confirmed no-op.**
Words-per-second — the metric built specifically to detect a tempo change — sits within ±0.11
wps of neutral for every one of them, and `speed_very_fast` (-0.08 wps, +0.32s duration) and
`speed_very_slow` (+0.02 wps, -0.08s duration) both move in the **wrong direction** from what
their names promise on this particular carrier text. This matches `PROMPTING.md`'s own tip that
`speed_very_slow` "only slows the model to roughly ~5s," suggesting these tags are weakly
implemented or get overridden by other decoding factors at `temperature=1.0` — but this document
only measures the effect, not why. `<|prosody:pause|>` (Group B) also shows *more* total pause
time (1940ms) than `<|prosody:long_pause|>` (1640ms, Group A) despite the name implying the
reverse — flagged, not resolved, pending a listen to `prosody_pause.wav` /
`prosody_long_pause.wav` for actual pause placement/length.

`style:whispering` (Group C) reconfirms the pre-existing finding from
`m4-sentiment-results.md`: energy is *higher*, not lower, than neutral (+3.5 dB in this run's
carrier text), and pitch spread is essentially unchanged (+0.8 Hz) — consistent evidence across
two independent generations that this tag is a near no-op.

### 4.1 Owner's listening verdict on emotion tags (2026-08-25) — SUBJECTIVE, human judgment

The owner listened to the emotion-tag clips from this inventory and reported:

> «эмоции вроде различимы и работают»

This is quoted verbatim per the same convention as §3.6. Scoped precisely, per the owner's own
qualifier «вроде» (moderate confidence, not a categorical claim):

- This covers **emotion tags as a group**, from listening across the inventory — a broader
  statement than the earlier, stronger confirmation in `m4-sentiment-results.md` §6 ("грусть
  действительно грустная"), which was about the single `sadness`/`elation` pair specifically.
- «вроде» is doing real work here: this is a cautious, moderate-confidence confirmation, not
  "emotions work reliably." It should not be strengthened into a stronger claim than the owner
  made.
- It corroborates **Group A** above (15 emotion tags with a clear, direction-consistent
  objective signal): the ear and the metric agree on the group most likely to matter, which is
  useful evidence that the A/B/C triage methodology itself is tracking something real, for
  future tag checks.
- It does **not** extend to Group B (`arousal`, `confusion`, `helplessness`, `longing`) or to
  the two Group C emotions (`anger`, `enthusiasm`) — the owner said nothing about these, and
  they remain unverified by ear.

### 4.1a Resolving the `speed_slow` contradiction between this document and `m4-sentiment-results.md`

Two independent measurements of `<|prosody:speed_slow|>` disagree on the surface:
`m4-sentiment-results.md` (T0, single clip, one measurement) found a clean, internally
consistent tempo drop — duration 8.28s vs. 6.96s neutral, tempo 1.57 vs. 1.87 words/s (-16%).
This document's inventory run (§4 table above) found `speed_slow` in Group C — ΔTempo only
-0.03 wps, essentially indistinguishable from neutral on this document's carrier text.

The owner's listening verdict resolves this rather than leaving it as an unexplained
discrepancy: *«медленный голос не всегда сильно медленнее»* — "the slow voice isn't always much
slower." **The correct reading is not that one measurement is wrong and the other right; it is
that the tag's effect on tempo is real but weak and inconsistent across carrier text/generation
runs** — strong enough for T0's single clip and carrier to show a clean -16% tempo delta, weak
enough for this document's inventory carrier and generation to land within noise of neutral.
This is a materially different conclusion from "no-op" (which `style:whispering`, confirmed
inert on two independent measurements, actually is): `speed_slow` is a tag with SOME effect that
one run can catch and another can miss, not a tag with no effect at all. Both measurement
documents are corrected to state this rather than the sharper "Group C / near-zero" framing that
implied the tag simply does nothing.

## 5. Priority listening list

See `output/m4_tags/LISTEN.md` for the full guide (local path only — `output/` is gitignored,
nothing here is committed). The 8 highest-priority items, condensed:

1. `03_style/style_whispering.wav` vs neutral — confirmed-suspect pustyshka.
2. `02_prosody/prosody_speed_slow.wav` vs neutral — near-zero measured tempo change.
   **Resolved 2026-08-25 by owner listening: not a no-op, effect is real but weak/inconsistent
   — see §4.1a.**
3. `02_prosody/prosody_speed_very_fast.wav` vs neutral — measured tempo change is *backwards*.
4. `01_emotion/emotion_sadness.wav` vs neutral — sanity-check clip: largest, most direction-consistent Group A signal.
5. `04_stress/stress_zamok_acute_1_castle.wav` vs `..._acute_2_lock.wav` — does stress notation
   actually shift meaning? **Resolved 2026-08-25 by owner listening for the apostrophe notation
   specifically — see §3.6.** Acute/capital/doubled-vowel notations were also covered by the
   same listening pass; see §3.6 for their individual verdicts.
5b. `04_stress/stress_zamok_plus_1_castle.wav` vs `..._plus_2_lock.wav` — the notation with a
   confirmed STT-transcript corruption elsewhere. **Resolved 2026-08-25: owner confirms this
   notation is unreliable — see §3.6.**
6. `05_final_intonation/boundary_1_complete_thought.wav` vs `..._2_continuing_thought.wav` —
   was framed as **the segmentation-defect question**: does our own chunk-per-chunk generation
   leave sentences sounding cut off. **Update 2026-08-25 (issue #57 follow-up): this pair does
   NOT answer that question, and the framing above is wrong.** `boundary_2_continuing_thought.wav`'s
   text (`BOUNDARY_CONTINUING` in `m4_tag_inventory_bench.py`) is a grammatically COMPLETE
   sentence ending in a period — `"Он медленно подошёл к двери и взялся за ручку."` — only its
   *narrative* continuation (what happens next in the story) is left open; nothing about the
   chunk boundary itself is tested here, since the clip is not actually cut off mid-sentence. The
   owner listened and correctly heard "мысль завершена" (the sentence IS grammatically finished);
   the task had asked for "мысль не завершена" as if the model were expected to intuit unwritten
   narrative intent from a full stop, which is not a defect in chunking or in the model's
   intonation — it is a badly posed question. The corresponding survey task
   (`final-boundary-continuing` in `src/sentiment_survey/task_sets/final_intonation.json`) has
   had its `correct_answer` retracted to `null` (no gradable expectation) rather than kept as a
   false miss; see that file's `hidden.A.note` and `src/sentiment_survey/server.py`'s
   `record_is_still_gradable()` for how the historical answer is excluded from scoring without
   being rewritten. **The real segmentation-defect question — what happens when a long sentence
   is force-split mid-clause (comma/conjunction, not a period) via `_force_split_long_sentence()`
   — was still untested by this pair and is the subject of a separate, targeted probe:
   `docs/research/audiobook/m4_midsentence_split_bench.py` (issue #57 follow-up).**
7. `05_final_intonation/punct_3_exclaim.wav` vs `punct_1_period.wav` — does punctuation alone
   change delivery.

## 6. What was NOT verified (honesty)

- **The 9 `sfx:*` tags (`cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`,
  `sigh`, `sniff`, `sneeze`) were not generated, measured, or listened to at all, in this
  document or anywhere else in this project.** This document's original title/framing claimed
  a "full" tag inventory, but the `sfx:*` category was not even known to belong to the
  checkpoint's inventory at the time (see the correction note at the top). This is separate,
  currently unclosed work — issue #57 tracks it. Before shipping `sfx` tags in a real audiobook
  chapter: (1) confirm each of the 9 is audibly the effect its name promises (not, e.g., a
  no-op like `style:whispering` turned out to be for whispering), (2) confirm the inline,
  one-shot placement semantics from `PROMPTING.md` (`<|sfx:tag|>onomatopoeia, then the line`,
  no reopening across a chunk boundary) actually match what the model produces at a real chunk
  boundary, not just what `src/audiobook.py`'s `chunk_sentences()` was coded to do.
- **Update 2026-08-25**: the two items below this note that concerned tag identity/stress-shift
  by ear are now partially addressed by the owner's listening (§3.6, §4.1) — updated in place
  rather than deleted, so the original scoping of this document (objective-only) stays visible.
- **Emotion tag identity was confirmed by ear only in aggregate and only cautiously** (§4.1,
  «эмоции вроде различимы и работают») — this covers Group A as a set; Group B and the two
  Group C emotions remain unverified by ear. No *individual* emotion beyond `sadness`/`elation`
  (confirmed separately in `m4-sentiment-results.md`) has an individual by-ear confirmation.
- **Terminal-contour F0 slope is a noisy proxy** (§2.4); the segmentation-boundary question
  (§2.3), which is the one about our own code, is explicitly left to a human listener rather
  than decided from the metric — still unresolved as of this update.
- **Stress-shift itself (does the meaning actually change) is now verified for the apostrophe
  notation, by the owner's ear** (§3.6: «апостроф дает ударение») **— but only for the 3
  homograph pairs and 6 notations actually tested here, not as a general claim about Russian
  stress.** The other notations now have owner verdicts too (§3.6): doubled-vowel is confirmed
  NOT a stress mechanism (vowel-length effect only), `+` and capital-letter are confirmed
  unreliable, acute accent was not separately called out by the owner and remains without an
  individual verdict.
- **Duration/energy-per-syllable was not computed** for the stress probe (would have been a
  cheaper objective proxy for stress placement than nothing, but was time-boxed out of this
  pass) — moot for the apostrophe notation now that a human listen has answered the question
  directly, but still true for any future notation this document has not covered.
- **The atlas "fabric" carrier sentence is unusable as constructed** — garbled across every
  notation including the control — and its stress-notation results are excluded from the
  verdict rather than misreported as evidence against any notation.
- **A stress mark colliding with an actual quote character was not tested** (§3.2) — only
  representative sentences without quotes were checked against `split_sentences`.
- Machine contention: this run shares the host with another in-progress heavy profiling task;
  GPU access was serialized (confirmed via polling before starting), but the "before" load
  average (13.74) reflects residual system load from that task's immediately-preceding run, not
  a clean-machine baseline — recorded here rather than smoothed over.
- **Open question, not investigated at all: `<|env:music|>`, `<|env:noise|>`, and a standalone
  `<|chatml|>`.** The checkpoint's `tokenizer.json` `added_tokens` carries these three beyond
  the 43 documented control tags (ids 151702, 151703, 151724), but `PROMPTING.md` never
  mentions any of them — no syntax, no example, no description. They are deliberately excluded
  from `src/audiobook.py`'s `VALID_TAGS` (Refs #57) rather than guessed at: they may be internal
  training-time scaffolding for labeling background audio (music/noise beds) rather than a
  usable prompt-time control, or `<|chatml|>` may be leftover from a shared tokenizer base with
  no role in TTS at all — but that is speculation, not a verified fact. No one has tried
  inserting them into a generation call to see what happens. If audiobook work ever needs
  background music/noise cues, a controlled probe of `env:music`/`env:noise` (does it change
  anything audible, does it error, is it silently ignored) is the first step, done in isolation
  before trusting it in a real chapter.
