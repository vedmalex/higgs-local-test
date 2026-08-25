# Voice consistency research — draft findings (issue #57 follow-up, owner's ear-verdict)

## 3. Measured F0 spread on chapter-114-e0 (70 segments)

Tool: docs/research/audiobook/m4_prosody_metrics.py (autocorrelation F0, no calibrated
pitch tracker) run against all 70 segment WAVs in output/chapter-114-e0/, in manifest
`index` order (0..69), raw per-file results in /tmp/all_segments_prosody.json.

Overall (n=70):
- min 83.9 Hz, max 203.4 Hz, range 119.5 Hz
- mean 135.3 Hz, median 129.0 Hz, population stdev 28.6 Hz

Batch-boundary split (owner-supplied claim: segments 0-30 generated at --batch-size 1,
segments 31-69 at --batch-size 8 — NOT present as an explicit field in manifest.json
itself, taken from task framing):
- First 31 (batch=1): min 83.9, max 203.4, mean 135.2, stdev 27.6, range 119.5
- Remaining 39 (batch=8): min 88.2, max 196.7, mean 135.5, stdev 29.4, range 108.5

No discontinuity at the batch boundary: segment 30 -> 31 is 142.0 -> 149.1 Hz (~7 Hz),
smaller than several adjacent same-batch jumps elsewhere (e.g. 18->19: 203.4->106.2 Hz,
~97 Hz; 38->39: 98.0->162.2 Hz, ~64 Hz). Both sub-groups have statistically
indistinguishable dispersion (stdev 27.6 vs 29.4, range 119.5 vs 108.5) — batch size does
not appear to be a driver of the voice variance; the variance looks uniform across the
whole chapter regardless of batch size.

Per-segment F0 median list (index: Hz):
0:163.3 1:132.6 2:109.1 3:123.7 4:154.3 5:120.0 6:151.9 7:149.1 8:158.9 9:117.6
10:120.0 11:152.9 12:140.4 13:110.1 14:131.9 15:105.7 16:145.5 17:114.3 18:203.4 19:106.2
20:151.9 21:193.5 22:100.4 23:127.7 24:172.7 25:109.1 26:112.1 27:83.9 28:115.4 29:171.4
30:142.0 31:149.1 32:195.1 33:179.1 34:150.9 35:102.6 36:131.9 37:108.6 38:98.0 39:162.2
40:196.7 41:102.1 42:165.5 43:130.4 44:88.2 45:129.0 46:110.1 47:149.1 48:127.0 49:129.0
50:129.0 51:125.7 52:115.9 53:145.5 54:104.3 55:123.1 56:180.5 57:115.9 58:115.9 59:94.9
60:130.4 61:186.0 62:120.0 63:111.1 64:186.0 65:123.1 66:170.2 67:123.7 68:164.4 69:113.2

Caveat carried over from m4-chapter-results.md §5: this is a homemade, uncalibrated
autocorrelation F0 estimator (no librosa/parselmouth in .venv-tts) — octave errors and
tracking artifacts are possible; still directionally sound as "same speaker across
segments should cluster tighter than this."

## TODO still running
- Batch-vs-no-batch already done above.
- Seed experiment (fixed seed, different text x4-6; fixed seed + same text bitwise check) — IN PROGRESS.
- Cloning cost / batching-compat / caching section — pending confirmation from code (model.py already read, see notes below).

## Code facts already gathered (source: model.py in .venv-tts mlx_audio higgs_audio_v3)

- `generate()` (model.py:738) and `batch_generate()` (model.py:548) BOTH accept
  `ref_audio`, `ref_text`, `references`, `ref_audios`, `ref_texts`, `ref_audio_codes`,
  `ref_audio_codes_list` — i.e. batch_generate DOES support a reference/cloning voice per
  call, contrary to nothing said elsewhere; the project's own docs/guides/audiobook_guide.md
  §3a and src/audiobook.py's `_generate_batch_group` (calls `model.batch_generate(texts=...,
  temperature=..., max_new_tokens=...)` at line ~1210) simply never PASS these kwargs.
- `seed: Optional[int] = None` is a real parameter on BOTH `generate` (line 755) and
  `batch_generate` (line 569): `if seed is not None: mx.random.seed(int(seed))` (lines
  762-763 / 609-610) — this is a GLOBAL mx.random seed set once before the decode loop
  starts, not a per-row/per-sequence seed.
- `encode_reference_audio(audio)` (line 211-229) computes the delayed reference codes ONCE
  from a waveform; the returned `mx.array` is documented as reusable directly in
  `generate(..., ref_audio_codes=...)`. `_normalize_batch_references` (line 391-454): when
  the same ref_audio/ref_text is shared across the whole batch (`has_explicit_shared_ref` or
  `has_equal_per_item_refs`), the reference is encoded ONCE (`shared_refs`) and reused for
  every sequence in the batch — NOT re-encoded per item. So caching ref_audio_codes across
  segments/batches is both possible (precompute once, pass via `ref_audio_codes`) and
  already partially what batch_generate does internally when ref_audio is identical across
  a batch's rows.
- Sampling: `generation.py`'s `sample_independent`/`sample_batch` use
  `mx.random.categorical(logits, axis=-1)` (lines 95, 112) when temperature > 1e-5 — this is
  exactly what `mx.random.seed` affects. `step()` (generation.py:115-159) is the per-frame,
  per-codebook sampler with the SGLang-style delay pattern; nothing in it retains state
  between separate `generate()`/`batch_generate()` calls (no history_prompt-like carryover
  found anywhere in higgs_audio_v3/).
- `src/audiobook.py:1070` `_generate_single_segment` calls
  `model.generate(text=entry["text"], temperature=temperature, max_new_tokens=max_new_tokens)`
  — no ref_audio/ref_text/seed passed, confirmed directly (matches task's claim).
- `docs/guides/audiobook_guide.md` lines 50-58 `register_voice()`: calls
  `model.encode_reference_audio(wav_path)` and saves to `voices/<name>.npy` — this IS a real
  documented model API (`encode_reference_audio` exists in the actual mlx_audio source,
  verified independently above), not an invented wrapper API. But per audiobook_guide.md
  lines 61-65 and 100-115, `src/audiobook.py` never reads anything from `voices/` — the
  registration code works standalone but is not wired into generation.
- RTF cost of cloning: README.md lines 229-231 (Apple M1, native MLX):
  TTS basic RTF 6.56 (no ref); TTS clone (7.4s ref) RTF 7.73. audiobook_guide.md lines
  112-115 attributes not-doing-cloning-yet partly to this being "slower, not just
  unverified." README line 54 + audiobook_guide history: an EARLIER 60-second reference
  gave clone RTF 822.09 (essentially broken/unusable) vs the current 7.4s reference's 7.73 —
  i.e. clone cost scales steeply with reference length on this MLX/M1 path (60s ref: 822,
  7.4s ref: 7.73). README line 255 states explicitly: "Higgs's clone cost itself scales
  with reference length while Qwen's stayed comparably cheap" — direct quote/citation for
  the scaling claim, contrasted with Qwen3-TTS whose clone RTF stayed ~1.4-1.6 regardless.
  Mechanistic reason (README line 56): "reference audio is tokenized at 25 frames/sec into
  the prompt KV cache (e.g. 5-15s clips are ~125-375 tokens for faster RTF, while a 60s
  clip introduces ~1500 prompt audio tokens)" — longer reference = longer prompt = more
  KV-cache/prefill cost paid on every single call unless ref_audio_codes are precomputed
  AND the prefill itself is amortized (batch_generate's shared-ref-across-batch behavior
  amortizes the CODE computation, not necessarily the per-sequence prefill cost, since each
  batch row still needs the reference tokens in its own prompt embedding sequence — see
  `_build_prompt_embeddings` per-sequence in the batch loop, model.py:634-637).


## Seed experiment — IN PROGRESS (background), first partial results

Script: .plan/seed_experiment.py. Same fixed seed=42 passed to model.generate(seed=42)
for each of 6 different short Russian fragments (independent generate() calls, exactly
matching how audiobook segments are produced). Early partial results (from
.plan/seed_experiment_results.json while still running):

- seeded_0 ("Сегодня прекрасная погода..."): F0 median 244.9 Hz (wall 20.3s, 3.32s audio)
- seeded_1 ("Компьютер медленно загружался..."): F0 median 105.7 Hz (wall 10.6s, 2.64s audio)

Same seed=42 for both calls, F0 median differs by ~139 Hz — a huge swing, comparable to
or larger than the whole-chapter spread measured in §3 (119.5 Hz range across 70
segments with NO seed fixation at all). This is only 2/6 fragments so far; full run
(6 seeded + 6 unseeded + bitwise-identical check) still in progress in background,
polling via `grep DONE .plan/seed_experiment.log`.

## Clone cost experiment — NOT YET STARTED

Script ready at .plan/clone_cost_experiment.py (basic vs clone-with-path vs
clone-with-cached-codes, each also in batch_generate() form with N=4 texts). Will run
sequentially after the seed experiment finishes (one heavy run at a time on this
machine, per explicit instruction). Weights on disk (for the memory section):
~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b = 8.7 GB.
