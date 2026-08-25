# M4-T5/T7 — Segmentation + chunk-boundary tag continuity: results

> **Correction (2026-08-25, Refs #57).** An independent audit of `src/audiobook.py` found 14
> defects, 5 of them critical, that this document's §1 and §0 claims did not account for. In
> particular:
> - §1's "no incorrect split occurred" claim was **not representative**: `quote_depth` was a
>   single counter fed by both the open- and close-quote branches with the *same* `"`
>   character, so it only ever grew and never came back down. The hand-verified paragraph in
>   §1 happens to be the one case (a closing `"` immediately after a period) that this bug
>   does not break; any other quote pattern — a `"…"` pair anywhere earlier in the text, or
>   `„…“`-nested quotes inside `«…»` — collapsed the rest of the input into one sentence. This
>   was not caught because the verification was hand-checked against a single paragraph, not
>   covered by a test suite exercising the general case.
> - §0/§1 said nothing about numbered lists: `"1. Первый пункт. 2. Второй пункт."` split into
>   four segments, two of them the bare, meaningless one-character strings `"1."` and `"2."`
>   — each would have been sent to the model as its own generation call and read aloud as
>   "один." / "два." in the finished audiobook.
> - §0/§1 did not discuss what happens when an abbreviation or initial sits at the very end of
>   a sentence or paragraph (e.g. `"...и т.д."` immediately followed by end-of-input or a
>   quote) — an edge case the original hand-verification paragraph did not exercise and this
>   correction still does not claim full coverage of; treat it as untested.
> - §0's "resume support" claim did not satisfy M4-T8's resumability requirement: manifest
>   writes were a plain in-place `path.write_text()` (not atomic), so a kill mid-write left an
>   unparsable JSON file with no recovery path and no way to know which segments were actually
>   done; and a `"done"` segment was trusted on `out_path.exists()` alone, with no check that
>   the WAV was non-empty, complete, or the length its text implied.
>
> All of the above (and 9 further findings, several also critical: an unbounded chunk size for
> a single over-budget sentence, unbounded assembly memory for a full chapter, one failed
> segment aborting the entire multi-hour run, a `max_chars` accounting bug under
> `tag_scope="sentence"`, a mid-sentence tag-reopen conflict, silently-ignored malformed
> control tags, and more) are fixed in the same change that added this correction. Every fix
> has a regression test in `tests/test_audiobook.py` that first reproduces the original
> defect against the pre-fix code, then confirms the fix — see that file and its module
> docstring's F1–F14 mapping for the full list, and the PR that lands this correction for the
> real test-run output. The rest of this document (§2 onward: the tag-continuity acoustic
> measurement) is unaffected by this correction and stands as originally recorded.

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio). Scope: the segmentation lane of
[`m4-plan.md`](m4-plan.md) (M4-T5/M4-T7, Lane 2's prerequisite). **This is not the full-chapter
run (M4-TX)** — per the task's explicit instruction, no long generation was run on this pass
because the machine may be shared with other work; every clip below is a few seconds of audio.
A full multi-hour chapter run is a separate, later task once the machine is free.

## 0. What is implemented vs. what is reused

**Implemented in this branch (`src/audiobook.py`), as corrected 2026-08-25 (see the correction
notice above):**
- Russian-aware sentence splitting (`split_sentences`) that does not break on common
  abbreviations (`т.е.`, `и т.д.`, `гг.`, `н.э.`, …), single-letter initials (`А. С. Пушкин`),
  numbered-list markers (`"1. Первый пункт."`), or inside quoted spans (`«…»`, `"…"`,
  `„…“`-nested-inside-`«…»`) — quote state is now tracked per pair-type with a paragraph-
  boundary reset and a force-reset safety valve, not the single ever-growing counter this
  document originally described.
- Sentence-preserving chunking (`chunk_sentences`) with control-tag continuity tracking across
  chunk boundaries and (optionally, see §2) across untagged sentences within one chunk, a hard
  per-chunk character budget (a single over-budget sentence is now force-split rather than
  passed through whole), and a validated tag vocabulary (an unrecognized/mistyped control tag
  now raises before generation starts instead of being silently ignored).
- A per-segment, content-hash-keyed manifest with atomic, crash-safe writes and integrity-
  checked resume (`build_manifest` / `load_or_create_manifest` / `generate_segments`) — each
  segment is its own WAV file, manifest status is written atomically after every segment (with
  a `.bak` fallback recovered from on a corrupt read), and a segment already marked `"done"` is
  re-validated against the actual WAV on disk (non-empty, plausible duration, matching sample
  count) rather than trusted on `out_path.exists()` alone. A single segment's generation
  failure is retried with backoff and, under `--continue-on-error`, no longer aborts the rest
  of a multi-hour run.
- A separate, streaming assembly step (`assemble_chapter`) with a numeric splice-quality check
  (edge amplitude + max sample-to-sample jump at each join) that never materializes a full
  chapter's audio in memory, and an `--allow-gaps` mode that inserts reported placeholder
  silence for missing segments instead of refusing all-or-nothing.

**Reused as-is:** `mlx_audio.tts.utils.load` and `HiggsAudioV3.generate(text=..., temperature=...,
max_new_tokens=...)` — the exact call convention `src/tts_test.py --text` already established
(M4-T0, PR #97). `generate_segments` is the only place this module calls the model, and it calls
it exactly that way. Fades (`fade_in_ms=30`, `fade_out_ms=15`) are already applied inside
`HiggsAudioV3.generate()` per call (`model.py:756-757`) — not new code, but relevant to the
splice-quality check in §4. The prosody analyzer used for the numeric evidence below
(`m4_prosody_metrics.py`) is the same script committed for M4-T0, not reimplemented.

**Not implemented / not run:**
- The full chapter (M4-TX) — explicitly out of scope for this pass.
- `--tag-scope` behavior beyond what §2 tests directly.
- Any change to `src/tts_test.py`.

## 1. Sentence splitter — verification

> **This section's original claim was wrong for the general case** (see the correction notice
> above). The single hand-verified paragraph below happened to avoid the specific quote
> pattern (`quote_depth` as a single ever-growing counter) that broke on most other inputs,
> so "no incorrect split occurred" described one lucky paragraph, not the splitter's actual
> behavior. The splitter has since been rewritten (per-pair-type quote tracking, numbered-list
> handling) and is now covered by `tests/test_audiobook.py`, including tests built specifically
> from the audit's failing examples (straight-quote pairs, `„…“`-inside-`«…»` nesting, and
> numbered lists) run against both the original and the fixed code. The paragraph below is kept
> for continuity but should be read as "one hand-checked example that still passes," not as
> general verification.

Tested against a synthetic paragraph exercising every case called out in the task (abbreviations,
initials, quotes):

```
Академик А. С. Пушкин писал стихи, т.е. занимался литературой, и т.д. «Это цитата с точкой
внутри. И ещё одна.» — сказал он. Она ответила: "Хорошо." Потом ушла. Сегодня 25 г. до н.э.
это было давно.
```

`split_sentences()` output (5 sentences, verified by hand against the source):

1. `Академик А. С. Пушкин писал стихи, т.е. занимался литературой, и т.д. «Это цитата с точкой внутри. И ещё одна.»` — correctly kept together: neither "А." / "С." (initials) nor "т.е." / "и т.д." (abbreviations) nor the periods *inside* the quoted span end the sentence.
2. `— сказал он.`
3. `Она ответила: "Хорошо."` — the quote closes and the period after it correctly ends the sentence.
4. `Потом ушла.`
5. `Сегодня 25 г. до н.э. это было давно.` — "г." and "н.э." correctly treated as abbreviations, not sentence ends.

No incorrect split occurred on any of the abbreviation/initial/quote cases in this test. This is
a hand-verified regression case, not a formal test suite — a real chapter will surface cases this
paragraph doesn't cover.

## 2. THE key result — does a control tag survive an independent chunk boundary?

**What actually ran.** The full 8-clip design below (§2 table) was planned but the machine hit
severe, *unrelated* contention while this task ran: `uptime` load averages spiked to **77-80**
and `vm.swapusage` showed **17.3-17.9 GB of 18 GB swap used** (94-97%), driven by macOS system
daemons unrelated to this task (`mediaanalysisd` at 76-132% CPU, `BackgroundShortcutRunner`,
`siriactionsd`, `mobileassetd` — confirmed via `ps aux`, not this script or another agent's TTS/STT
job). A first attempt at the full design (`m4_boundary_check.py`) was killed after the very first
clip took over 5 minutes of wall time with almost no progress. **The design was cut down to the
single most important pair** — `chunk2_noreopen` vs. `chunk2_reopen`, `max_new_tokens` capped at
200 to bound worst-case run length — and re-run once load had visibly dropped
(`m4_boundary_check_minimal.py`). That run completed: `chunk2_noreopen` in 116.9 s (2.88 s audio),
`chunk2_reopen` in 315.2 s (3.52 s audio) — both wall-clock numbers are **contention-inflated, not
clean RTF measurements** (load average was still 27-40 during this run), and are not cited as
performance figures anywhere in this project. **Not run in this pass, for the same reason:**
`ref_neutral_s1`, `ref_elation_s1`, `whole_call_once` (the "does an unrepeated tag persist to a
later sentence within ONE call" question) and all of §3/§4 below. The harness scripts for the
full design remain committed and ready to run once the machine is free.

**Method.** Two short sentences, `S1` = "Сегодня удивительный день, полный радости." and `S2` =
"Мы наконец дождались этой прекрасной новости.", with `<|emotion:elation|>` as the test tag. Five
clips, generated with the exact same `model.generate(text=..., temperature=1.0,
max_new_tokens=4096)` call `src/tts_test.py` uses, model loaded once for all clips:

| Clip | Text sent to the model | What it represents |
|---|---|---|
| `ref_neutral_s1` | `S1` (no tag) | neutral baseline for S1's wording — **not run**, see above |
| `ref_elation_s1` | `<\|emotion:elation\|>S1` | what "elation" sounds like on S1 — **not run** |
| `chunk2_noreopen` | `S2` (no tag) | S2 generated as an independent second chunk, **tag NOT reopened** — **run** |
| `chunk2_reopen` | `<\|emotion:elation\|>S2` | S2 generated as an independent second chunk, **tag reopened** — **run** |
| `whole_call_once` | `<\|emotion:elation\|>S1 S2` | tag stated once, not repeated on S2, in ONE call — **not run** |

`chunk2_noreopen` vs. `chunk2_reopen` is the direct test of chunk-boundary tag survival, and it is
the one pair that actually ran: both are generated as **fully independent `model.generate()`
calls** (architecturally unavoidable — each call builds a fresh prompt and a fresh KV-cache via
`make_prompt_cache(self)`, `model.py:786`; nothing carries over between two separate calls). The
only question is whether the *text* fed to the second call needs to repeat the tag for the audio
to carry the same emotional coloring as the first chunk would have.

### Results (real numbers, `logs/m4_boundary_check_minimal.json`)

Both clips generated with `temperature=1.0`, `max_new_tokens=200` (capped to bound run length
under the contention described above — this caps generation length, not audio content; both
clips finished well under the cap). Scored with the unmodified M4-T0 analyzer
(`m4_prosody_metrics.py`):

| Metric | `chunk2_noreopen` (no tag) | `chunk2_reopen` (tag reopened) | Direction |
|---|---|---|---|
| Duration (s) | 2.88 | 3.52 | reopen longer |
| F0 median (Hz) | **104.8** | **171.4** | reopen +63.5% |
| F0 range (Hz) | 316.4 | 325.2 | ~flat |
| F0 std (Hz) | **54.3** | **91.9** | reopen +69.2% |
| Voiced-frame ratio | **0.687** | **0.830** | reopen +0.14 |
| Mean energy (dB) | -25.0 | -25.2 | ~flat |
| Pauses (n / total ms) | 0 | 1 / 220 | reopen adds one pause |

For context, M4-T0's own measurements on a different sentence (`m4-sentiment-results.md`) put
untagged neutral speech at F0 median 162.2 Hz / std 77.4 Hz, and `<|emotion:elation|>`-tagged
speech at F0 median 275.9 Hz / std 94.3 Hz — i.e. in that prior data, elation runs *higher and
more variable* than neutral. `chunk2_reopen`'s profile (171.4 Hz median, 91.9 Hz std, high voiced
ratio) sits in that same "elevated and variable" direction, while `chunk2_noreopen`'s profile
(104.8 Hz median, 54.3 Hz std) is *lower and flatter than even the untagged neutral baseline from
a different sentence* — consistent with a flat, low-energy, unemotional reading, not merely "a
different sentence read neutrally."

### Verdict and design decision

**The tag does not survive the chunk boundary on its own, and reopening it recovers a
substantially different (and elation-like) delivery.** Without reopening, `chunk2_noreopen`'s F0
profile is lower and flatter than the tagged clip by a wide margin (median -66 Hz, std -37.6 Hz,
voiced ratio -0.14) — this is not a marginal or ambiguous difference; on every prosodic axis this
analyzer measures except energy, the two clips move in the same direction the T0 elation-vs-neutral
comparison did. Architecturally this is expected — each chunk is an independent `generate()` call
with its own fresh cache, so there is no mechanism by which state from a prior call could survive
regardless of what the text says — but the plan asked for evidence, not an inference from code
structure, and this is that evidence for the one pair that could be run given the machine's state.

**Design decision confirmed by this result:** `chunk_sentences()` in `src/audiobook.py` reopens
the last-active emotion/prosody/style tag at the start of every chunk (default `tag_scope`
now set to `"sentence"`, i.e. even more conservatively — re-declaring before every sentence, not
just every chunk boundary — precisely because PROMPTING.md documents these tags as
sentence-level, "colors the whole sentence," singular). This result directly supports not shipping
a segmenter that lets a tag silently expire at a chunk boundary.

**What this does NOT establish:** whether an owner listening blind would call `chunk2_reopen`
"elation" in an absolute sense — that is M4-T0's still-open, "mixed, not a clean pass" question,
and this document does not relitigate it. What is established here is *relative*: reopening
produces a measurably different, more elevated/variable delivery than not reopening, on the
exact same text. **n = 1 per condition, same estimator caveats as T0** (see §5). The owner's ear
is the actual arbiter; both WAVs are named explicitly in §6 for that listening check.

## 3. Voice/timbre consistency across independently generated segments

**Not run.** The planned test (`voice_a` fresh, `voice_b_noref` fresh, `voice_b_withref` using
`voice_a`'s audio+text as `ref_audio`/`ref_text`) is written into
`docs/research/audiobook/m4_boundary_check.py` but was never executed — the machine-contention
cutdown in this pass kept only the single highest-priority pair (§2). This is not implemented
evidence; do not treat it as such. It is queued to run alongside the deferred parts of §2 once the
machine is free.

## 4. Splice quality — numeric join check

**Not run against real generated segments**, for the same reason as §3. `assemble_chapter()`
itself (the numeric-join-check code in `src/audiobook.py`) was smoke-tested with two synthetic
sine-wave WAVs (not model output) to confirm the join math executes and produces sane numbers —
e.g. `prev_tail_edge_abs_amplitude`/`next_head_edge_abs_amplitude` correctly report the
(deliberately nonzero, ~0.5) edge amplitude of two unfaded synthetic tones, and
`max_intra_window_sample_jump`/`direct_join_sample_jump` came back as small positive floats, not
zero or NaN. That confirms the *code path works*, nothing about *real chapter audio's* join
quality — for that, `HiggsAudioV3.generate()`'s built-in 30 ms fade-in / 15 ms fade-out per call
should drive both edge-amplitude numbers close to zero on real segments, but this is a prediction
from reading `model.py:756-757`, not a measurement. The intended check — run `assemble_chapter()`
on the two clips generated in §2, or on a short real chapter once the machine is free — is
queued, not completed.

## 5. Limitations and honesty notes

- **n = 1 per condition**, same as M4-T0 — no repeated trials, no seed variation. Everything
  here is a single run at `temperature=1.0`.
- The prosody analyzer is the same homemade, uncalibrated autocorrelation-based F0 estimator
  from M4-T0 (`m4_prosody_metrics.py`) — no `librosa`/`parselmouth`/`soundfile` in `.venv-tts`.
  Octave errors and other tracking artifacts are possible and cannot be distinguished from real
  effects with this tool alone.
- M4-T0's own verdict on whether Higgs's emotion tags are *audibly* distinguishable at all is
  **mixed, not a clean pass** (`m4-sentiment-results.md`) — the owner's blind listening check has
  not yet returned a verdict. The results in this document are about tag *continuity/scope*
  mechanics (does a tag's effect, whatever its strength, carry across a chunk boundary), not a
  re-litigation of whether the effect itself is strong. If the owner's listening check later
  finds the elation tag has no reliable audible effect at all, this document's continuity finding
  would need to be re-run once a more reliably audible tag is confirmed.
- The machine was under heavy, variable background load while these clips were generated
  (`uptime` load averages seen between ~15 and ~80 across this task, swap 94-97% full at the
  worst point, `mediaanalysisd` and other unrelated macOS daemons active) — generation wall-time
  numbers here are not clean RTF measurements and should not be cited as such; only the audio
  itself and its measured acoustic properties are the evidence.
- **No full chapter was generated.** M4-TX (a real chapter, overnight) is explicitly deferred to
  a separate task once the machine is free, per the task's instruction.
- Voice-consistency testing (§3) was not run at all this pass (see §3) — it is not merely weak
  evidence, there is no evidence yet. When it does run, note in advance that the planned proxy
  (F0 median + a homemade FFT spectral centroid) is coarse and not a validated speaker-similarity
  metric, so even a completed run should be read as a directional observation, not a settled one.

## 6. Files

- Code: `src/audiobook.py` (production segmentation/generation/assembly module).
- Regression tests (added 2026-08-25, Refs #57): `tests/test_audiobook.py` — one test class per
  audited finding (F1-F14), each reproducing the original defect against the pre-fix code before
  asserting the fix. Run with `.venv-tts/bin/python3 -m unittest discover -s tests -v`.
- Verification harnesses: `docs/research/audiobook/m4_boundary_check.py` (full 8-clip design,
  planned but not completed this pass — see §2), `docs/research/audiobook/
  m4_boundary_check_minimal.py` (the 2-clip cutdown that actually ran).
- Raw results actually produced: `logs/m4_boundary_check_minimal.json`.
- Clips that actually exist (gitignored, local only, for the owner's listening check):
  `output/m4_boundary_check/chunk2_noreopen.wav`, `output/m4_boundary_check/chunk2_reopen.wav`.
- `logs/m4_boundary_check.json` does **not** exist — the full-design run that would have produced
  it was killed before completing its first clip; do not look for it.
