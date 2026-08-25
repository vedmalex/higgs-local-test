# M4 plan — audiobook production track (issue #57)

Date: 2026-08-25. Milestone M4 of issue #57. **This is a planning document.** Except where a task
is explicitly marked done below (M4-T1, S0, M4-M1), nothing in this plan has been executed. Per
repo policy (`AGENTS.md`) a box is ticked only after the thing has actually been RUN and its real
output recorded in a results doc. (M4-M1 is the one exception by construction: it is a
judgement/write-up task with no run of its own — it is ticked because its deliverable,
[`../mojo-max/m4-conclusion.md`](../mojo-max/m4-conclusion.md), exists and rests entirely on
M4-T1's already-recorded measurement.)

Prior milestones: [`../mojo-max/m0-results.md`](../mojo-max/m0-results.md) through
[`../mojo-max/m3-plan.md`](../mojo-max/m3-plan.md) (Mojo/MAX vocoder porting — see §5 for how this
plan closes that track), and the M4 measurement that actually motivates this plan:
[`../mojo-max/m4-stage-profile-results.md`](../mojo-max/m4-stage-profile-results.md) (PR #95,
merged 2026-08-24).

## Where things stand before this plan (facts, not re-derived here)

```text
Vocoder (Mojo/MAX)      M0-M3: full port of one _BosonDecoderBlock validated on M1, BLOCKED on
                        Tesla T4 by an upstream MAX/Turing gap (m3-t4-blocked-results.md).
Stage profile (M4-T1)   MEASURED, PR #95. AR loop (prefill + per-frame backbone) = 94.3-95.5% of
                        wall time; codec.decode (one call) = 1.78-3.76%. See §3, M4-T1.
Decoder                 already MLX, already on GPU. MLX-probe run: 10/10 PASSED (correctness
                        only — no performance or sentiment claim, see §8).
Sentiment tags          34 added_tokens exist in the tokenizer, all special/non-normalized,
                        NOBODY has verified Higgs audibly obeys them. See §0.4.
Batching                already implemented (`continuous_batching.py`), never benchmarked.
Quantization            infrastructure complete (`python -m mlx_audio.convert`), never run on
                        this checkpoint. Codec fate under conversion is UNVERIFIED and risky —
                        see §0.6.
Segmentation            does not exist for higgs_audio_v3. A neighbor model has one to copy from.
STT                     two working local models; existing WER number is invalid; larger model
                        never run.
```

---

## 0. Findings

**0.1 Higgs is required for inline sentiment tags; Qwen is not a substitute.** Qwen3-TTS's
`generate_custom_voice(instruct=...)` (`qwen3_tts.py:445-469`) takes one style string that gets
concatenated as a prefix to the *entire* utterance — one style per call, no way to change emotion
mid-sentence. Higgs Audio v3's control tags are inline tokens inside the text stream itself, at an
arbitrary position, so a single generation call can shift emotion/prosody/style mid-paragraph.
This is the entire reason the project stays on Higgs rather than switching to the faster,
already-working Qwen3-TTS backend for anything beyond flat narration.

**0.2 Audiobook arithmetic (for BOTH measured RTF values — see §3, M4-T1b for why there are two).**
At RTF 4.26 (this plan's own PR #95 measurement): a 10-hour audiobook needs ≈43 hours of machine
time. At RTF 6.56 (the pre-existing `logs/tts_basic.log` figure for the same text fixture): ≈66
hours. **Neither number is treated as final** — M4-T1b exists specifically to resolve which one
this project should plan against. Both land in the "impractical" tier (§2: RTF > 3), and both
require a 3–4.4x speedup before a full book is practical on this machine. This is the central
arithmetic problem M4's optimization lane exists to attack.

**0.3 The decoder is already MLX on GPU.** Nothing about the vocoder needs porting to run on
Metal — it already does. The MLX-Audio probe recorded 10/10 PASSED, but that is a correctness
check only (load + inference completes, output is finite/well-shaped); it says nothing about
speed or sentiment fidelity (§8).

**0.4 Sentiment tag inventory — measured directly against the real checkpoint's `tokenizer.json`,
not assumed.** `bosonai/higgs-tts-3-4b`'s tokenizer has **84 `added_tokens`**, of which **34 are
control-markup tags**, all `special: true`, `normalized: false`:

```text
<|emotion:*|>   21 tokens, ids 151681-151701
<|prosody:*|>   10 tokens, ids 151716-151726
<|style:*|>      3 tokens, ids 151704-151706
```

(Verified by parsing `~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b/snapshots/*/tokenizer.json`
directly — 21 emotion / 10 prosody / 3 style / 34 total, not "20 emotion / 33 total" as an earlier
draft of this plan stated.) None of the 34 appear in the base BPE vocabulary. `prompt.py:46`'s
`encode(text, add_special_tokens=False)` suppresses BOS/EOS, **not** added-tokens, so all 34 tags
do reach the model as intended token ids. **But "reaches the model" is not "the model obeys it" —
nobody has listened.** The repo's own existing control-tag fixture (`src/tts_test.py:14`) already
exercises 7 of these 34 tags across all three categories (`contentment`, `speed_slow`, `pause`,
`enthusiasm`, `expressive_high`, `long_pause`, `whispering`) and has never been graded blind for
whether the emotion/prosody actually changed. There is a direct precedent for tags landing wrong:
`README.md:245` records the model literally speaking a stress-mark annotation aloud instead of
using it prosodically. This gap — tags reach the model, nobody has confirmed they are heard — is
the reason T0 is a blocking first task (§3).

**0.5 The bottleneck is the AR loop, and this is now a MEASURED fact, not an inference from code
structure.** M4-T1 (§3, DONE, PR #95) measured: the autoregressive Talker loop (prefill +
per-frame backbone forward) is 94.3% (long case) to 95.5% (short case) of wall time; the single
`codec.decode` call is 3.76% / 1.78%. Batching is already implemented in
`continuous_batching.py` (`BatchKVCache`, `add`/`step`/`cancel`, `_admit_pending`,
`_advance_active`, `_decode_audio:279`), with the batched forward path at `model.py:715` — it has
never been benchmarked on this hardware. Batching changes *scheduling*, not per-token arithmetic,
so it does not put sentiment fidelity at risk the way quantization might. One wrinkle, harmless
for this project: `generate_batch` explicitly rejects the `instructs` argument
(`model.py:596-598`: `"Higgs Audio v3 batch_generate does not support instructs"`). This does
**not** affect this plan — `instructs` is the Qwen-style "one style string per call" convention
(§0.1); Higgs's sentiment travels as inline tokens inside the text itself, which the batched path
passes through unmodified. Recorded here explicitly because the error message looks alarming out
of context and would otherwise cost someone time re-discovering this.

**0.6 Quantization — two separate facts about the codec that must not be merged into one.**
The conversion tool this plan uses is specifically `mlx_audio/convert.py:583 def convert`
(argparse `configure_parser:710`, `main:795`) — **not** `tts/utils.py:211 def convert`, a
different function of the same name that has neither the `higgs_multimodal_qwen3` detection nor
the `sanitize`-before-quantization sequencing (both confirmed present only in
`mlx_audio/convert.py`). Infrastructure in the correct tool is complete: `higgs_multimodal_qwen3` →
`higgs_audio_v3` detection covered by `tts/tests/test_convert.py:18-24`, `sanitize` runs before
quantization (`convert.py:660-661`), quantized-checkpoint loading already wired
(`apply_quantization`, `utils.py:209-254`, called at `:402-403`), `mlx_audio.tts.load()` accepts a
local path (`tts/utils.py:132-155`), and `get_model_path` (`convert.py:234-262`) already tries a
local `Path(...)` first and only calls `snapshot_download` if it doesn't exist — none of this needs
new code. Start with 8-bit and, within 8-bit, predicate variant B first (§ M4-T3); mixed recipes
(`QUANT_RECIPES`, `convert.py:22`, CLI `--quant-predicate` `:753-757`, built by
`build_quant_predicate:496-516`) come only after variant B, not before.

- **(a) The codec is never quantized — and this needs no protective work, it already can't be
  reached.** `_codec` is constructed inside `post_load_hook` (`model.py:88-110`, verified against
  the installed package) entirely outside the model's `nn.Module` parameter tree — it is assigned
  onto `model._codec` after `load_model` returns, so MLX's quantization walk over named parameters
  never sees it and the `model_quant_predicate` (`model.py:82-85`, confirmed:
  `isinstance(module, (nn.Linear, nn.Embedding))` with a `multimodal_embedding` exclusion) never
  runs against it. **Delete "keep the codec at full precision" as a task — it is already
  structurally true, for a reason unrelated to file copying.**
- **(b) But the codec's WEIGHTS can vanish from the converted output entirely — and this is a live
  risk requiring a blocking check.** Verified directly against the cached checkpoint: the
  `bosonai/higgs-tts-3-4b` snapshot has **no `audio_tokenizer/` subdirectory** — only a sharded
  `model.safetensors` + `model.safetensors.index.json`. That sends `post_load_hook` down the
  `HiggsAudioTokenizer.from_higgs_tts_checkpoint(...)` branch (`codec/models/higgs_audio/
  higgs_audio.py:252-264`, confirmed), whose own docstring states: "Higgs Audio v3 stores codec
  tensors in the main TTS safetensors shard under
  `tied.embedding.modality_embeddings.0.model.*`". But `sanitize` (`model.py:129-130`, confirmed
  exact lines) hits an `elif key.startswith("tied.embedding.modality_embeddings.0.model."):
  continue` — it discards exactly those keys before conversion/quantization ever sees them — and
  `copy_model_files` (`convert.py:536-542`, confirmed) skips any file matching `model*.safetensors`
  when copying supporting files. **Net effect: after conversion, the codec's weights exist in
  neither the sanitized weight dict nor the copied files — nowhere in the output directory.**
  `from_higgs_tts_checkpoint` on the converted checkpoint would find zero matching tensors and
  silently return a codec with uninitialized weights. The failure mode is noise or silence on an
  apparently-successful conversion — exactly the shape of bug that gets misattributed to
  "quantization degraded quality" and produces a false T4 verdict. **M4-T3's first step is a
  blocking check for this** (see §3).
- Do not write "CODEC PROTECTED structurally via `copy_model_files`" — that claim is false; the
  correct statement is the two bullets above, kept separate.

**Memory — four numbers, and the apparent 1.16-GiB-vs-11.93-GB "contradiction" is not one; it is
four different metrics, one of which was simply never recorded before now.**

```text
peak_rss_maxrss   1 248 624 640 B = 1.16 GiB. This is resource.getrusage(RUSAGE_SELF).ru_maxrss,
                  which src/tts_test.py:66 records under the AMBIGUOUS key "peak_memory_bytes" —
                  identical to /usr/bin/time -l's "maximum resident set size" line. UNDERCOUNTS:
                  does not account for MLX's unified-memory allocations. Never cite this as "the"
                  memory figure.
peak_footprint    12 205 576 640 B = 11.37 GiB (README.md:206's "11.37 GB", GiB mislabeled GB) =
                  /usr/bin/time -l's "peak memory footprint" line, the real whole-process number.
peak_mlx          mx.get_peak_memory() — the MLX allocator's own peak (activations + KV-cache).
                  logs/tts_basic.log NEVER recorded this at all — src/tts_test.py never calls it
                  (only the Qwen runner does). PR #95's profiler added the call and read up to
                  11.93 GB. There was never a conflicting pair of MLX-allocator readings; the
                  older log simply has no entry for this metric.
weights_on_disk   8.7 GB = 8.1 GiB (model.safetensors).
```

**So there is no unresolved 1.16-vs-11.93 discrepancy to chase** — the confusion arose entirely
from `peak_memory_bytes` being an ambiguous field name for `ru_maxrss`, which is exactly the class
of naming mistake this plan otherwise commits to avoiding (§6). The arithmetic is consistent:
≈8.1 GiB weights + the MLX allocator's own peak + runtime overhead ≈ 11.37 GiB peak footprint.
**Weights dominate, and quantization cuts them specifically**: 8-bit ≈8.1 → ≈4.3 GiB, 4-bit ≈8.1 →
≈2.3 GiB. Quantization does **not** shrink the MLX-allocator slice, which grows with utterance
length and batch size instead. **The success metric for memory is therefore not "did total GB go
down" but "how many batch slots now fit"** — quantization frees headroom specifically *for* the
batch (§0.5/M4-T2); the two levers multiply rather than compete. 4-bit quantization frees roughly
≈5.8 GiB of the weight footprint — that is what makes M4-T2's batch size of 2-4 achievable in the
first place.
§6 and any future README update must report exactly three quantities under these names —
`peak_mlx (GiB)`, `peak_footprint (GiB)`, `weights_on_disk (GiB)` — and must never cite
`peak_rss_maxrss`. **Add a task to rename the ambiguous `peak_memory_bytes` key in
`src/tts_test.py` (and its README references) to `peak_rss_maxrss`**, so this exact confusion
cannot recur.

**0.7 Segmentation does not exist for this model.** `higgs_audio_v3` has no chunking path. A
neighbor implementation does: `qwen3_tts.py:1271`'s `split_pattern` and `_decode_chunk:1037`
(`chunk_tokens=300`, `left_context_size=25`) is the closest existing pattern to adapt.

**0.8 STT status.** Two local models work end-to-end; the previously logged WER 1.5 is invalid and
must not be cited as a quality number; the 1.7B variant has never been run; no standalone Whisper
integration exists.

**0.9 Mojo/MAX closure now has a MEASURED basis, not just a structural argument.** §0.5's number —
codec.decode at 1.78-3.76% of wall time — is the measured answer to the question M0-M3 existed to
answer indirectly. State explicitly: `../mojo-max/m0-results.md`'s "GPU BLOCKED on macOS 14.6.1"
finding is **stale and is not the basis for anything in this plan** — the machine has since been
upgraded to macOS 26.6.2, and M2/M3's own GPU prototypes passed on that OS. The closure argument in
§5 rests entirely on the M4-T1 wall-time share, not on the old GPU-availability finding.

**0.10 Quantization predicate inventory.** Eleven quant predicates exist in the tree:
`higgs_audio_v3:82`, `higgs_audio:69`, `qwen3_tts:275`, `sesame:487`, `spark:110`,
`moss_tts:371`, `moss_tts_nano:61`, `longcat_audiodit:134`, `voxtral_tts:315`,
`minimax_music3:132`. Note `minimax_music3.py` lives under `mlx_audio/music/models/`, **not**
`tts/models/`, and its logic is an allow-list via `path.startswith((...))`, not a
vocoder-exclusion pattern like Higgs's — do not describe it as analogous to `model_quant_
predicate`'s `multimodal_embedding` exclusion.

---

## 1. Tracks

**Track T (TTS/audiobook, primary)** now runs as **two parallel lanes** rather than one linear
list, because everything a real chapter needs already works today at the current (slow) speed:

- **Lane 1 — optimization**: batching (free, already implemented) → quantization (risky, gated by
  sentiment fidelity).
- **Lane 2 — the slow-but-real chapter**: generate one real chapter at today's speed. Starts
  immediately, in parallel with Lane 1, not after it.

**Track S (STT, secondary)**, **Track M (Mojo/MAX, closure)**.

Order: T0 → **[M4-T1 already DONE]** → T1b → Lane 1 and Lane 2 run concurrently (they compete for
GPU time, not for a human, so alternate machine time between them) → S0 runs in parallel with both
→ stability/resume work once a full-length run exists → STT comparison once chapter-scale
reference audio exists (after T6/T7) → Mojo/MAX closure (M1) can be written up at any point once
T1's number is in hand — it is not gated by anything else in this plan.

## 2. Pre-declared thresholds

```text
sentiment baseline (T0)     can sadness and elation be told apart blind? NO -> escalate to the
                            owner immediately, before any further TTS work.
vocoder relevance           < 15% of wall time -> do not touch the vocoder.
                            MEASURED (M4-T1): 1.78-3.76%. Threshold triggered; vocoder is closed
                            (see §5).
T1 failure threshold        if no configuration reaches < 24h for a 10h audiobook -> escalate to
                            the owner with an explicit choice. NOTE (§0.2): at BOTH currently
                            measured RTF values (4.26 -> ~43h, 6.56 -> ~66h) this threshold is
                            already tripped; §3's optimization lane exists to answer it.
batching payoff             >= 1.5x wall-time improvement to be worth keeping.
quantization payoff         >= 1.5x improvement over the best batching result.
sentiment integrity gate    operationalized (§7, U7): an objective pre-filter (KL divergence +
                            top-k logit overlap over AR steps, `m3_divergence.py:149`) screens
                            out grossly broken configurations before any listening; then a MINIMUM
                            of 8 blind pairs, generated from REAL CHAPTER material (not isolated
                            test phrases, once Lane 2 has produced one), with the verdict cast by
                            the OWNER — opus only aggregates the owner's answers, it does not judge
                            audio itself (an agent judging a text description of audio is not a
                            substitute for a human listening). FAIL if the owner prefers the
                            unquantized version in more than 6 of 8 pairs. Any audible sentiment
                            smoothing is a failure regardless of any speed gain.
suitability tiers           RTF <= 1.5 practical / 1.5-3 usable with mandatory resume support /
                            > 3 impractical.
```

## 3. Track T

- [ ] **M4-T0 (BLOCKING). Sentiment baseline, blind, 15 minutes.** Generate `sadness` vs
  `elation` on the same short text; the owner listens blind and says whether they're
  distinguishable. NO → escalate to the owner immediately; everything downstream in this plan that
  assumes sentiment tags work (T4, T6, Lane 2) is contingent on this passing.
  *Tier:* **opus** for the verdict framing/aggregation; the generation itself is mechanical.

- [x] **M4-T1. Profile the existing MLX pipeline stage-by-stage.** **DONE — measured, PR #95
  merged 2026-08-24.** `docs/research/mojo-max/m4_stage_profile.py`,
  `docs/research/mojo-max/m4-stage-profile-results.md`. Measured on real M1 hardware with
  `mx.eval()` inserted at every stage boundary (mandatory under MLX's lazy evaluation — otherwise
  work silently migrates to whichever stage forces the next sync):

  | Stage | short (~5 s audio) | long (~19 s audio) |
  |---|---|---|
  | AR prefill + per-frame backbone passes | 95.5% | 94.3% |
  | `codec.decode` (single call) | 1.78% | 3.76% |
  | everything else (tokenize, sampling glue, WAV write) | 2.7% | 2.0% |
  | RTF (wall / audio duration) | 4.08 | 4.26 |
  | AR frames | 130 | 492 |

  Conclusion: an infinitely fast, zero-cost vocoder would take the long case from 82.87 s to
  ≈79.75 s wall time — RTF 4.26 → ≈4.10. **The vocoder's ceiling on any achievable speedup is
  ~4%.** §2's pre-declared "<15% → don't touch the vocoder" threshold fired; the vocoder track is
  closed (§5).
  Memory (kept as separate, named claims per `AGENTS.md`): `mx.get_peak_memory()` up to **11.93
  GB**; whole-process peak footprint via `/usr/bin/time -l` **12.30 GB**; that same tool's
  "maximum resident set size" line reads only **1.32 GB** — confirmed to undercount on macOS and
  must **not** be quoted as a memory figure. (§0.6 resolves what initially looked like a
  contradiction between this 11.93 GB figure and an older log's 1.16 GiB figure — it is not one;
  they are different metrics, and the older log simply never recorded the MLX-allocator number.)
  **Honest caveats, recorded verbatim, not smoothed over:** the machine was not fully idle during
  this run — `uptime` reported load average 5.94, with Docker Desktop and an IDE language server
  active in the background. The measured long-case RTF of **4.26 does not match** the previously
  logged **6.56** for the nominally same fixture (`logs/tts_basic.log`). Plausible contributing
  causes are named (background load, added `mx.eval()` sync points, package-version drift, natural
  AR-sampling-length variance) but **the number was not adjusted or tuned to match** — the gap is
  reported open, and M4-T1b (below) exists to actually resolve it rather than guess at it.
  Consequently, §0.5's "the AR loop dominates" moves from an inference drawn from code structure
  to a **measured fact**, and the corresponding line is removed from §8 ("what this plan does not
  claim").

- [ ] **M4-T1b. Re-run M4-T1's profile on an idle machine.** The 4.26-vs-6.56 discrepancy is not
  cosmetic: it's the difference between a ≈43-hour and a ≈66-hour full-book run (§0.2), which
  matters for every downstream speedup-target calculation in this plan. Close Docker Desktop, IDE
  language servers, and any other background load; confirm `uptime`'s load average is near
  baseline-idle before running; re-run `m4_stage_profile.py` unchanged and compare against both
  prior numbers.
  *Done when:* a clean-load run's RTF is recorded and reconciled against 4.26 and 6.56 — either it
  lands close to one of them (identifying which was the anomaly) or a third number requires its
  own explanation.
  *Files:* reuses `docs/research/mojo-max/m4_stage_profile.py`; appends a section to
  `m4-stage-profile-results.md`.
  *Devices:* M1, idle.
  *Tier:* **sonnet**, size **S**.

### Lane 2 — the slow, real chapter (starts now, in parallel with Lane 1)

- [ ] **M4-TX. Generate one full real chapter at today's speed, no optimization applied.**
  Rationale: everything a chapter needs already works today — the model, the control tags, the
  codec. The only missing piece is segmentation (§0.7), and even that has a neighbor
  implementation to adapt. A full chapter is a night of *machine* time, not human time. This gives
  (a) the owner a tangible result inside the first week, (b) the sentiment gate (T4/T0) real
  chapter-scale material for blind A/B instead of isolated test phrases, and (c) an early,
  concrete answer to the open risk of whether a tag survives a chunk boundary — well before Lane
  1's optimization work would otherwise surface it. Lane 1 and Lane 2 compete for GPU time, not
  for a person, so alternate machine time between them rather than serializing.
  Depends on T0 passing (sentiment must be at least baseline-distinguishable before spending a
  night of compute on a chapter that assumes it works) and on enough of T7's segmentation work to
  produce one stitched chapter — this task and T7 are effectively the same delivery, listed
  separately here to make Lane 2's priority explicit against Lane 1.
  *Files:* produces the first real audiobook output artifact; segmentation code lands wherever T7
  places it.
  *Devices:* M1, overnight.
  *Tier:* **sonnet** for the harness, **opus** for reviewing the first tag-across-chunk-boundary
  result.

### Lane 1 — optimization

- [ ] **M4-T2. Measure the existing batching implementation at batch sizes 2 and 4.** This is a
  measurement of code that already exists (`continuous_batching.py`), not new development. **Do
  not test batch size 8**: ≈11.9 GB of resident weights on a 16 GB machine leaves too little
  headroom for KV-cache growth at that size (§0.6's memory arithmetic) — state this as a known
  constraint going in, not a surprise discovered via OOM. If batch sizes 2/4 hit memory pressure
  anyway, a quantized KV-cache is the fallback enabler (it raises the batch ceiling; it does not
  by itself deliver a speedup — see the rejected-levers table below).
  *Done when:* wall-time-per-utterance at batch 2 and 4 is measured against the batch-1 baseline
  and checked against the >=1.5x threshold (§2).
  *Files:* new measurement script under `docs/research/audiobook/`.
  *Devices:* M1.
  *Tier:* **sonnet**, size **S** (measurement, not development).

- [ ] **M4-T3 (conditional on T2). Convert to 8-bit and verify the result is actually usable.**
  **First step, blocking, before anything else in this task**: after conversion, explicitly check
  that `HiggsAudioTokenizer.from_higgs_tts_checkpoint(...)` finds nonzero codec tensors in the
  converted output. If it does not — the expected outcome per §0.6(b) unless this is fixed — before
  converting, manually extract `tied.embedding.modality_embeddings.0.model.*` from the source
  checkpoint's `model.safetensors` into an `audio_tokenizer/` directory placed next to the
  conversion's output, so `post_load_hook` takes its `audio_tokenizer/`-present branch instead.
  State the "codec stays full precision" fact correctly: it is true because the codec sits outside
  the quantized parameter tree entirely (§0.6a), not because `copy_model_files` protects it.
  **Third-party weight rejection criterion, applied before downloading anything** (P7): pull
  `added_tokens` from the candidate checkpoint's `tokenizer.json` and confirm the same 34 markup
  tags at the same ids (151681-151701 emotion, 151704-151706 style, 151716-151726 prosody, per
  §0.4's measured inventory). A mismatch means a different tokenizer generation — do not download
  those weights.
  **Predicate variant hypothesis, to compare in T4**: the upstream predicate
  (`model.py:82-85`, confirmed `isinstance(module, (nn.Linear, nn.Embedding))`) quantizes
  `nn.Embedding`, which means the embeddings for all 34 markup tokens (ids 151681+) land in
  quantization groups of 64 alongside high-frequency ordinary tokens — the most plausible
  mechanism for an "emotion gets smoothed out" failure. **Variant A** = the upstream predicate
  as-is. **Variant B** = the same predicate restricted to `isinstance(module, nn.Linear)` only
  (i.e. never quantize `nn.Embedding`). T4 compares A and B blind. Indirect support for this
  hypothesis: upstream already special-cases and excludes `multimodal_embedding` from
  quantization, which is the same category of concern.
  *Done when:* the converted checkpoint's codec is confirmed present and functional, both
  quantization variants (A/B) are produced, and the third-party-weight rejection check has been
  run (even though this is the project's own first-party checkpoint, applying the check here
  establishes the procedure before it's needed for any future substitute model).
  *Files:* extends `mlx_audio/convert.py`'s `convert()` usage (not `tts/utils.py:211`); new
  verification script under `docs/research/audiobook/`.
  *Devices:* M1.
  *Tier:* **sonnet-deterministic-code**, **opus** review on the predicate-variant framing (this is
  an interpretive claim about a failure mechanism, not a mechanical check).

- [ ] **M4-T4 (GATE). Sentiment integrity gate, conditional on T3.** Objective pre-filter first:
  `m3_divergence.py:149`'s `compare()` (KL divergence + top-k logit overlap across AR steps)
  cheaply screens out grossly broken configurations before any listening — **including the case
  where the codec never made it into the converted checkpoint at all** (§0.6b); a lost codec is
  exactly the kind of gross break this pre-filter is cheap insurance against. **Low KL does not
  prove emotion survived, and high KL does not prove it was lost** — this tool is a pre-filter,
  never the gate itself (P10). The actual, only gate is >= 8 blind pairs of real-chapter material
  (§2), owner verdict, opus only aggregates. Additional objective proxy, reported *alongside* the
  listening verdict, never as a substitute for it — replacing an earlier plan to use a third-party
  SER classifier (rejected: wav2vec2-style SER models are trained on English acted-emotion
  corpora; their calibration on Russian synthetic speech is unknown and would add an unvalidated
  external dependency to judge this project's own model): **ASR round-trip + prosodic metrics**
  (F0 range, pause duration, speaking rate) measured on pairs of `<|emotion:sadness|>` /
  `<|emotion:elation|>` output, computed without any third-party model. Emotion smoothing, if
  present, must show up as a narrowed F0 range under this proxy.
  **Escalation option if no configuration clears tier <= 3 (§2's suitability tiers): a hybrid, not
  abandoning Higgs.** Route emotionally loaded scenes through Higgs (full tag fidelity) and
  neutral narration through Qwen3-TTS (RTF 1.57 on this same machine, `README.md:225`). This
  removes the "emotion vs. time budget" dilemma without giving up Higgs's control tags where they
  actually matter.
  *Done when:* pre-filter run on both variants, >= 8 blind pairs judged by the owner on real
  chapter material, verdict recorded per §2's fail condition.
  *Files:* reuses `m3_divergence.py`; new blind-pairing harness and results doc under
  `docs/research/audiobook/`.
  *Devices:* M1 for generation; human ears for the gate.
  *Tier:* **sonnet** for harness/pre-filter code, **opus** to aggregate the owner's blind-pair
  verdicts (never to substitute for them).

**Rejected optimization levers (recorded here so they are not re-proposed without new evidence):**

| Lever | Why rejected |
|---|---|
| Speculative decoding | No draft model exists for this stack; building/validating one is expensive relative to expected gain. |
| Reduce codec frame rate | 25 fps is baked into the trained codec; changing it means retraining, not an inference-time knob. |
| Quantized KV-cache | Frees memory, does not by itself deliver speed — kept only as an *enabler* to raise T2's batch-size ceiling if 2/4 hit memory pressure. |
| Vocoder kernel optimization (the whole M0-M3 Mojo/MAX effort) | Measured ceiling ≈4% of wall time (M4-T1). Rejected on data, not on suspicion. |

- [ ] **M4-T5. Dump the full 34-tag dictionary and check chunk-boundary behavior.** Per §0.4, this
  is now a lookup, not research — the tag inventory (21 emotion / 10 prosody / 3 style, ids
  151681-151701 / 151704-151706 / 151716-151726) is already measured from `tokenizer.json`; this
  task exports it into project docs and tests each tag once across a synthetic chunk boundary
  (relevant once T7's segmentation exists).
  *Tier:* **sonnet**, size **S**.

- [ ] **M4-T6. Russian quality at chapter scale; emotion drift check.** N=10 paragraphs each using
  one tag; blind comparison of paragraph 1 vs. paragraph 10 to check for drift over a long run.

- [ ] **M4-T7. Chapter generation: segmentation, stitching, tag continuity.** Adapts
  `qwen3_tts.py:1271`'s `split_pattern` / `_decode_chunk:1037` pattern (`chunk_tokens=300`,
  `left_context_size=25`) since `higgs_audio_v3` has no chunking path of its own (§0.7). This is
  the task that actually produces Lane 2's chapter (M4-TX above).

- [ ] **M4-T8. Multi-hour run stability, SIGKILL handling.** Criterion: a partial WAV file must
  remain readable, AND the resume point must be recoverable, after a kill mid-run. Use the
  three-named-quantities memory reporting from §0.6/§6 throughout — this is exactly the run shape
  where the MLX-allocator-vs-process-vs-disk distinction matters most, and where batching's real
  payoff (more segments per batch, not fewer GB) should be visible.

- [ ] **M4-T9. Resume support + a practical hours-per-book number + `make audiobook`.**

- [ ] **M4-T10. Rename the ambiguous `peak_memory_bytes` key.** `src/tts_test.py:66` stores
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` under the misleading name `peak_memory_bytes` — this
  is what caused §0.6's apparent 1.16-GiB-vs-11.93-GB "contradiction" to look like a real
  discrepancy when it was actually two different metrics with confusingly similar names. Rename
  the key to `peak_rss_maxrss` in the JSON output and in every README reference to it.
  *Tier:* **sonnet**, size **S**.

## 4. Track S

- [x] **S0 (30 min, no code).** **DONE.** Ran Qwen3-ASR-1.7B-8bit through the existing `--model`
  flag (`stt_qwen_local_test.py:70`) — RTF 0.291, `peak_mlx` 3.577 GiB, `weights_on_disk` 2.298
  GiB (`logs/qwen17_stt.log`, `logs/qwen17_stt.json`) — and retracted the invalid WER 1.5 figure
  from `README.md`. Draft comparison doc: `docs/research/stt/m4-stt-comparison.md`. No valid WER
  exists for any STT model yet — that is S1's job, still open.
- [ ] **S1 (after T6/T7).** Build verified reference transcripts once chapter-scale audio exists.
- [ ] **S2.** Compare models, with the mandatory caveat that this is an 8-bit vs. FP16 comparison,
  not an architecture comparison.
- [ ] **S3.** Recommendation + model selection exposed as a parameter.

## 5. Track M — Mojo/MAX closure

- [x] **M1. Close the Mojo/MAX track on a MEASURED basis. DONE — written up 2026-08-25,
  [`../mojo-max/m4-conclusion.md`](../mojo-max/m4-conclusion.md).** Readiness criterion: "the
  `codec.decode` share of wall time is measured and below 15%" — **satisfied**, per M4-T1: 1.78%
  (short) / 3.76% (long), both far under the 15% threshold declared in advance (§2). If this had
  come back >= 15%, the track would **not** close and the question would reopen — it did not.
  M0-M3 are written up as completed research; `m3_block_reference.py` and `m3_divergence.py`
  (the latter already reused by M4-T4 above) remain in active use. State explicitly:
  `m0-results.md`'s "GPU BLOCKED on macOS 14.6.1" finding is **stale and not the basis for this
  closure** — the machine is now macOS 26.6.2, and M2/M3's GPU prototypes passed there; the
  closure argument rests solely on the M4-T1 wall-time measurement. Reopen conditions: a return of
  the cross-platform requirement AND the upstream MAX/Turing fix (`modular/modular#6659`) landing.
  Consequence for the future: if the vocoder is ever rewritten again, it goes on MLX, not MAX.

## 6. Verification method

Same rigor as M2/M3:

```text
gates            main gates listed in §2; sentiment fidelity (T0, T4) is a hard gate, not advisory.
RTF              measured with mx.eval() at every stage boundary (mandatory under MLX's lazy
                 evaluation, exactly as M4-T1 did it) — never a wall-clock number taken across
                 unforced lazy ops.
memory           reported as THREE separate, explicitly named quantities, never merged:
                 peak_mlx (GiB) = mx.get_peak_memory(); peak_footprint (GiB) = /usr/bin/time -l's
                 "peak memory footprint" line (the real whole-process number); weights_on_disk
                 (GiB) = checkpoint size on disk. NEVER cite peak_rss_maxrss (ru_maxrss / "maximum
                 resident set size") as a memory figure — confirmed to undercount on macOS by
                 excluding MLX's unified-memory allocations. README updates and results docs must
                 use these three names (§0.6, M4-T10).
SIGKILL          partial-WAV-readable AND resume-point-recoverable is the pass condition (T8);
                 neither alone is sufficient.
tool reuse       m3_divergence.py's compare() (per-tensor max abs/rel err, NaN/Inf, exact-zero,
                 saturation counts) is reused as-is for the KL/logit pre-filter tooling, not
                 reimplemented.
process isolation TTS and STT stay in separate processes/environments, per AGENTS.md.
provenance       every numeric claim cites its script and results doc; PR #95 is the standing
                 example for M4-T1.
extrapolation    flagged explicitly wherever a conclusion crosses from measured to expected
                 (e.g. §0.9's "AR-loop share only grows with longer text" note).
honesty          a partial or failing stage is written up as partial or failing. Branch per issue,
                 Refs #57 on commits, results docs land next to the scripts that produced them.
```

## 7. Explicit non-goals for M4

- **NOT** replacing Higgs with Qwen3-TTS wholesale without an explicit owner decision — the hybrid
  in M4-T4 is an opt-in escalation path, not a default.
- **NOT** further Mojo/MAX vocoder porting — closed on measured grounds (§5).
- **NOT** rewriting the decoder — it is already MLX/GPU.
- **NOT** NVIDIA, vLLM, or any non-local/non-MLX inference path.
- **NOT** training or fine-tuning anything.
- **NOT** building on third-party quantized weights without first passing the §0.6(b)/M4-T3
  tokenizer-tag rejection check.
- **NOT** starting quantization at 4-bit, and **NOT** starting with mixed `QUANT_RECIPES` — 8-bit
  first, and within 8-bit, predicate variant B first (§ M4-T3); mixed recipes only after B, with
  evidence.
- **NOT** using `tts/utils.py:211 convert` — the conversion tool this plan uses is
  `mlx_audio/convert.py:583 convert` specifically; the other function of the same name lacks the
  `higgs_multimodal_qwen3` detection and the pre-quantization `sanitize` sequencing.
- **NOT** writing new code to load quantized weights or resolve a local checkpoint path — both
  already work (`mlx_audio.tts.load()` accepts a local path, `tts/utils.py:132-155`;
  `get_model_path`, `convert.py:234-262`, tries `Path(...)` first and only calls
  `snapshot_download` if that path doesn't exist).
- **NOT** relying on any claim that "the codec is protected structurally by `copy_model_files`" —
  it is not protected; per §0.6(b) it is actively discarded for this checkpoint's layout, and the
  M4-T3 blocking check exists specifically because of that.
- **NOT** an architecture-level STT comparison — S2's comparison is explicitly scoped to an
  8-bit-vs-FP16 caveat, not a clean model-vs-model claim.
- **NOT** streaming generation.
- **NOT** a standalone Whisper integration without a separate justification.
- **NOT** inventing new control tags — the 34 that exist are the complete inventory (§0.4).
- **NOT** using a third-party SER classifier to judge sentiment fidelity — replaced by the ASR
  round-trip + prosodic-metric proxy (M4-T4) and the owner's own blind-pair verdict.

## 8. What this plan does not claim

- That Higgs audibly obeys its control tags — this is the single biggest unverified premise in the
  whole plan, which is exactly why T0 is first and blocking.
- That batching will actually deliver >= 1.5x (T2 measures this; it is not assumed).
- That 8-bit quantization will be sufficient, or safe for sentiment, before T4's gate runs.
- That sentiment survives chunk boundaries — untested until T5/T7.
- That Russian quality holds at full-chapter scale — untested until T6.
- No claim about STT accuracy before S1 produces verified reference transcripts.
- The MLX-Audio probe's 10/10 PASSED measured correctness only, not speed or sentiment.
- The result of this plan is not production-grade.
- Which of the two measured RTF values (4.26 vs. 6.56, §0.2) is the "real" one for planning
  purposes — that is exactly what M4-T1b exists to resolve, and until it does, this plan reports
  arithmetic for both rather than picking one.

(Note, unlike M3's equivalent section: "the AR loop dominates wall time" has been **removed** from
this list — per §0.5/M4-T1, that is now a measured fact, not an unverified premise.)
