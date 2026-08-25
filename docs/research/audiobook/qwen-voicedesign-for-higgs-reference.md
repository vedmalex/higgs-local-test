# Qwen3-TTS VoiceDesign as a reference-clip generator for Higgs cloning

Research snapshot: 2026-08-25. Question: can Qwen3-TTS's VoiceDesign (voice built
from a natural-language description) generate reference clips that Higgs Audio v3
then clones from, so the audiobook narrator's timbre is chosen by description
instead of by trawling the existing segment library? No models were downloaded or
run in this pass — this is a code/documentation read only, per the run order in
the owning task (machine busy with the Mahabharata translation and a parallel
cloning-benchmark agent).

## Verdict, up front

1. **MLX path exists and is already partially proven locally.** Not hypothetical.
2. **Description language/attributes are underdocumented** upstream; only the
   `instruct` free-text field and `language` code are confirmed parameters.
3. **Higgs's reference-audio format requirements are simple and already satisfied
   by Qwen's own output** — no conversion step needed.
4. **The owner's core assumption is half right.** Timbre transfers, and the
   specific *words* of the reference clip do not leak into the Higgs output. But
   Higgs's own documentation states plainly that "manner of pronunciation"
   (манера произношения) is imitated from the reference, not just timbre — so if
   Qwen's stress-mangled reference clip is not just wrong-stressed but audibly
   glitchy/unstable in its articulation, that instability is a genuine, not
   theoretical, risk of bleeding into the Higgs clone. This must be checked by
   ear, not assumed away.
5. **Given the 71-segment library already on disk, Qwen is worth it only for
   precision, not for capability** — the library already contains real Higgs
   voices; Qwen's only genuine value-add is *targeted* tembr selection instead of
   accept/reject sampling from what the default voice happens to produce.

## 1. Does Qwen3-TTS VoiceDesign run on M1 via MLX at all — yes

`mlx-audio==0.5.0`, already installed in this project's `.venv-tts`, implements
VoiceDesign natively:

- `.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`
  has a dedicated `generate_voice_design()` method (lines 2145-2200), gated by
  `self.config.tts_model_type != "voice_design"` — i.e. it is a real, separate
  code path from `generate_custom_voice()` (predefined speakers, line 2068) and
  plain `generate()` (Base cloning), not a stub:

  ```python
  def generate_voice_design(
      self, text: str, instruct: str, language: str = "auto",
      temperature: float = 0.9, max_tokens: int = 4096, top_k: int = 50,
      top_p: float = 1.0, repetition_penalty: float = 1.05,
      verbose: bool = False, stream: bool = False, streaming_interval: float = 2.0,
  ) -> Generator[GenerationResult, None, None]:
      ...
      if self.config.tts_model_type != "voice_design":
          raise ValueError(
              f"Model type '{self.config.tts_model_type}' does not support generate_voice_design. "
              "Please use a VoiceDesign model (e.g., Qwen/Qwen3-TTS-12Hz-*-VoiceDesign)."
          )
      yield from self._generate_with_instruct(
          text=text, speaker=None, language=language, instruct=instruct, ...
      )
  ```

- The top-level `model.generate(..., tts_model_type=...)` dispatcher (same file,
  ~line 1142-1229) already routes `"voice_design"` to this method: "voice_design:
  Uses generate_voice_design() with instruct as voice description."

**Published MLX weights exist** (checked via web search + the model page,
2026-08-25, not merely inferred from the code path being present):
`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{4bit,5bit,6bit,8bit,bf16}`,
converted from `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` with mlx-audio. Sizes:
4bit ≈2.31 GB, 5bit ≈2.5 GB, 6bit ≈2.69 GB, 8bit ≈3.08 GB, bf16 ≈4.52 GB. License
Apache-2.0 (same as the rest of the Qwen3-TTS family, `docs/research/qwen3-tts-notes.md`).
There is **no 0.6B VoiceDesign checkpoint** — 1.7B is the only size, consistent
with what `docs/research/qwen3-tts-notes.md` already recorded for the CUDA path.

**This project has never run VoiceDesign, only Base and CustomVoice**, both
locally on M1 (`output/qwen_tts_basic.json`, `qwen_tts_clone.json`,
`qwen_tts_custom.json`, all `"status": "PASSED"`, `"device": "mlx"`,
`mlx_audio 0.5.0`) and Base/CustomVoice on a T4 (`docs/research/qwen3-tts-notes.md`).
Those PASSED runs are strong indirect evidence the MLX Qwen3-TTS stack works on
this exact M1/mlx-audio install, but VoiceDesign itself is untested here — this
pass only confirms the code path and the checkpoint exist, not that the checkpoint
loads and runs on this machine's 16 GB. **No weights were downloaded in this
pass** — the 8bit variant (≈3.1 GB) is the recommended size/quality trade-off if
a real run is authorized; ask before pulling it, per the run order.

## 2. How the voice description is specified

- Single parameter: `instruct` (free natural-language text), plus `language`
  (`"auto"`, `"chinese"`, `"english"`, etc. — code accepts any string, no
  enumerated whitelist found in `qwen3_tts.py`).
- No structured fields for gender/age/timbre — it is entirely prose. The
  upstream Qwen3-TTS model card states only that the system "supports speech
  generation driven by natural language instructions for flexible control over
  timbre, emotion, and prosody" — no enumeration of exactly which words map to
  which acoustic property, and no confirmed Russian-language example. All
  documented `instruct` examples on the model card and in `qwen3_tts.py`'s own
  docstring are in **Chinese** (e.g. `'体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显'`); English also appears
  in this project's own local test (`instruct: "Speak calmly and warmly, like a
  narrator reading a spiritual book."`, `output/qwen_tts_custom.json` — though
  that was CustomVoice's instruct field, not VoiceDesign's). **Whether a
  Russian-language description works for VoiceDesign specifically is unverified**
  by any primary source found — this needs an actual run to confirm, not an
  assumption from the CustomVoice case.
- Per `docs/research/qwen3-tts-notes.md`'s own prior finding (Discussion #185,
  #53 in `QwenLM/Qwen3-TTS`): a voice description does **not** pin a voice
  identity across separate calls — the community's own recommended workaround is
  exactly what the owner proposes: generate once with `instruct`, then use that
  clip as a reference for cloning downstream. This is now cross-checked and
  stands.

## 3. Does Qwen's output satisfy Higgs's reference-audio format — yes, without conversion

Read directly from
`.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/higgs_audio_v3/model.py`:

```python
# lines 192-209
def _normalize_audio(self, audio: Any) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        from mlx_audio.utils import load_audio
        audio = load_audio(str(audio), sample_rate=self.sample_rate)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] <= 2:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1)
    return arr.reshape(-1).astype(np.float32, copy=False)

def _prepare_reference_waveform(self, audio: Any) -> np.ndarray:
    audio_np = self._normalize_audio(audio)
    if audio_np.shape[0] < self.sample_rate:
        audio_np = np.pad(audio_np, (0, self.sample_rate - audio_np.shape[0]))
    return np.ascontiguousarray(audio_np, dtype=np.float32)
```

- **Any sample rate**: a path/string is loaded through `load_audio(...,
  sample_rate=self.sample_rate)`, which resamples to Higgs's own rate
  (`config.py` line 75: `sample_rate: int = 24000`). Qwen's own MLX output is
  already 24 kHz (`output/qwen_tts_basic.json`/`clone.json`/`custom.json`:
  `"sample_rate": 24000` in every one of this project's own local Qwen runs) —
  so no resampling is even needed in practice, but it would happen automatically
  either way.
- **Mono forced**: any 2D array (stereo) is averaged down to mono. Qwen outputs
  mono already.
- **Minimum duration ≥1 second**: shorter clips are zero-padded up to
  `self.sample_rate` samples (1 second). No documented maximum in the MLX path —
  `docs/guides/voice_cloning_guide.md` records a 60 s clip actually completing
  locally (at RTF 822, i.e. ~5.75 hours for 60 s of audio) and recommends
  **7-12 seconds** as the practical target (~175-250 audio tokens in the
  prompt); the CUDA/vLLM-Omni path caps at 30 s server-side, but that limit does
  not apply to the local MLX path this project actually uses.
- **`encode_reference_audio()`** (same file, lines 211-229) runs this
  normalization, encodes through the Higgs codec, applies the delay pattern, and
  returns codes reusable in `generate(..., ref_audio_codes=...)` — exactly the
  `.npy`-cacheable path `docs/guides/voice_cloning_guide.md` §4 already documents.

**Conclusion: a Qwen VoiceDesign WAV can be fed straight into
`encode_reference_audio()`/`generate(ref_audio=...)` with zero format
massaging** — same sample rate, mono, well above the 1-second floor at the
recommended 7-12 s target.

## 4. Does anything besides timbre transfer from the reference clip — the owner's key question

**Two separate mechanisms, and they answer differently.**

### Lexical content of the reference: does not transfer

`.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/higgs_audio_v3/prompt.py`
builds the prompt as:

```python
# build_prompt(), lines ~47-70
ids: list[int] = [self.tts_id]
for reference in references:
    if reference.text and self.ref_text_id is not None:
        ids.append(self.ref_text_id)
        ids.extend(self.encode_text(reference.text))   # ref transcript, as context only
    ids.append(self.ref_audio_id)
    ...
    segments.append((start, reference.codes))            # ref audio codes, as context only
ids.append(self.text_id)
ids.extend(self.encode_text(text))                        # the NEW text to synthesize
ids.append(self.audio_id)
```

The model only ever generates codes *after* `<|audio|>`, conditioned on the new
`<|text|>` block — the reference transcript (`ref_text`) and reference codes sit
earlier in context purely as conditioning (this is the standard zero-shot
in-context mechanism `docs/guides/voice_cloning_guide.md` §1 describes: "Аудиофайл
эталона… преобразуется в… Voice Tokens… подставляются в начало контекста… перед
синтезируемым текстом"). Nothing in this structure causes specific reference
*words* — or their stress pattern — to appear in the generated audio for
unrelated book text. This matches the owner's framing: mispronounced words in
Qwen's reference clip are not spoken again by Higgs, because Higgs is generating
new codes for new text, not aligning/copying phonemes from the reference
transcript.

### Manner of pronunciation / prosody / pace: transfers, by this project's own documentation — not just timbre

`docs/guides/voice_cloning_guide.md` §1, step 4, states the mechanism in its own
words (this project's canonical description of how Higgs cloning works, not an
inference from this pass):

> «Модель генерирует речь, **подражая тембру, акустике и манере произношения**
> из промпта.» — "The model generates speech **imitating the timbre, acoustics,
> and manner of pronunciation** from the prompt."

And its own recording rules (§3, rule 4) treat pace/breathing as part of what a
good reference clip must get right: "Естественный темп и дыхание… избегая
задувания в микрофон" — natural pace and breathing, avoiding mic-popping — i.e.
the guide already treats the reference clip's *articulation quality*, not only
its spectral timbre, as something that ends up in the clone.

**This is the load-bearing finding for the owner's plan.** Qwen's documented
failure mode against `samples/tts_ru.txt` (`docs/research/qwen3-tts-notes.md`,
"Stress control" section) is not merely "wrong syllable stressed" — it is
audible corruption: `Вриндава́н` → "Вриндава**Юн**" (an inserted syllable),
`Радхара́ни` → "Радхра**Юни**". That is exactly a manner-of-pronunciation defect
(garbled articulation, spurious phonemes), the category this project's own guide
says Higgs imitates from the reference — not a lexical-content defect that the
prompt structure above would block. **If a Qwen-generated reference clip
contains this kind of glitch anywhere in its 7-12 seconds, there is a real risk
it reproduces as a general instability/garbling quality in Higgs's clone
output — not the specific mis-stressed word, but an imitated rough-articulation
"accent."** This is not proven either way by source-reading alone; it is an
empirical question the code structure cannot settle, because "does the codec
encode glitch-articulation as an entangled acoustic feature that's genuinely
timbre-adjacent, or as a discardable artifact" is a property of the trained
codec/talker weights, not the prompt format.

**What would settle it, concretely**: generate one Qwen VoiceDesign reference
clip using **only stress-safe vocabulary** (short common Russian words, no
Sanskrit/Vaishnava proper nouns — i.e. avoid the exact category of word Qwen is
known to mangle) at 7-12 s, listen to it clean by ear first, then clone with
Higgs and listen to the Higgs output for any of the same glitch character (extra
syllables, unnatural pacing) even though the words differ. If the Qwen reference
itself is clean (plausible for simple vocabulary — the mangling was specifically
tied to complex/foreign proper nouns and to the abandoned stress-mark
workarounds, not to Russian TTS in general), this risk may simply not arise in
practice: **restrict Qwen's reference-clip text to plain, common Russian
words and this failure mode is likely avoidable by construction**, without
needing to prove the transfer mechanism either way.

## 5. Is Qwen worth adding, given the 71-segment library already on disk

`output/chapter-114-e0/manifest.json` confirms **71** `narrator`-speaker segments
already generated with Higgs's own default voice (no cloning, no `voice`/`ref_audio`
argument — `AGENTS.md`'s "Voice cloning must use only user-provided authorized
audio" plus `src/tts_qwen_local_test.py`'s architecture confirm the default-voice
path takes no speaker argument, so Higgs's own generation is naturally stochastic
across calls without a fixed reference). The owner's framing — that this is
already "a library of dictors" to pick from — matches: because no `ref_audio`/seed
is pinned, each of the 71 segments is an independent stochastic draw from Higgs's
own default-voice distribution, which is why they land on a spread of pitches the
owner is now sorting through by ear.

Given that, Qwen's honest value proposition is narrow and specific:

- **What it does NOT offer**: a capability the library lacks. The segments are
  already real Higgs voices, already in the target language, already reflecting
  Higgs's own acoustic character (which is what will actually narrate the book).
  Qwen adds a second model family, a second license (Apache-2.0, fine, but still
  a second thing to track per `AGENTS.md`), 2.3-4.5 GB of additional disk, and — per
  `docs/research/qwen3-tts-notes.md` — a documented Russian stress/pronunciation
  weakness of its own.
- **What it DOES offer**: control. Right now the owner is accept/reject sampling
  71 free draws hoping one lands near a target timbre (the 83.9-203.4 Hz spread
  implies a wide range of pitches to sift through). VoiceDesign lets the owner
  *aim* — describe "a warm, low male narrator voice" in words and get something
  close on the first or second try, then use §4's reference-clip-becomes-clone-source
  pattern to lock it into a reusable Higgs voice profile. That is a real,
  non-trivial time savings if the desired timbre is far from what the 71
  existing segments happen to contain (e.g. if none of them land near the
  target register) — but if the target register is already well-represented in
  the existing 71 segments, Qwen adds cost for no benefit.

**Honest answer: contingent, not unconditional.** Look at the existing 71
segments' pitch spread against the actual target register the owner wants
before deciding to spend the ~3 GB and a VoiceDesign run. If the target is
already covered, Qwen is not worth it — the "another backend in the project"
cost (a second TTS stack to keep working, per `AGENTS.md`'s isolation rules)
outweighs a convenience win that's already available for free. If the target
register genuinely isn't represented in the 71 segments (or the owner wants a
timbre distinct from anything Higgs's default voice tends to produce), Qwen
VoiceDesign is a reasonable, cheap (one 7-12 s generation) way to get there —
*conditional on* the manner-of-pronunciation risk in §4 being checked by ear
first, using stress-safe reference text.

## Open questions (not resolved by this pass)

1. Whether `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` (or another
   quantization) actually loads and runs within this M1's 16 GB alongside
   everything else already cached (~24 GB of models on disk currently: Higgs
   TTS 8.7G, Higgs STT 5.0G, Qwen CustomVoice 4.2G, Qwen Base 2.3G, Qwen ASR
   2.3G+964M). Untested — no download happened in this pass.
2. Whether a Russian-language `instruct` description produces a controllable,
   sensible voice for VoiceDesign specifically (confirmed only for CustomVoice's
   separate `instruct` field, in English, in this project's own prior local run).
3. Whether the manner-of-pronunciation transfer risk in §4 is real in practice —
   settled only by generating a stress-safe Qwen reference clip and listening to
   the resulting Higgs clone, not by source-reading.
4. No enumerated enumeration of exactly which voice attributes (gender/age/pitch
   range/accent) VoiceDesign's `instruct` text reliably controls was found in any
   primary source (Hugging Face model card or `QwenLM/Qwen3-TTS` repo) — only the
   generic "timbre, emotion, prosody" claim.

## Recommended next step (not taken in this pass)

If the owner decides the target timbre is not already covered by the 71-segment
library: download `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` (~3.1 GB —
flagging this explicitly since it is more than a few gigabytes, per the run
order) once the machine is free, generate one 7-12 s clip with a plain-vocabulary
Russian `instruct` description, listen to it by ear for the glitch pattern
described in §4, and only then feed it into
`HiggsAudioV3.encode_reference_audio()` and listen to the resulting clone before
trusting the pipeline for the actual book.

**Checked against this project's Google Drive model cache (2026-08-25,
`MyDrive/higgs-benchmark/model-cache/`, via the `gcloud`-authenticated Drive API,
`scripts/gdrive_sync.py`'s own auth path):** the cache holds tars for
`qwen3-tts-0.6b-base` (2.5 GB), `qwen3-tts-1.7b-customvoice` (4.5 GB),
`qwen3-asr-0.6b`/`1.7b`, `whisper-large-v3-processor`, `higgs-tts-3-4b`, and
`higgs-audio-v3-stt` — **no VoiceDesign tar exists there.** So "the models are
already in Google Drive" is true for Base/CustomVoice (already mirrored locally
in `~/.cache/huggingface/hub` anyway, per §1) but not for VoiceDesign — a fresh
pull is still needed regardless of source (local HF download, or add
VoiceDesign to `notebooks/model_prefetch_to_drive.ipynb`'s prefetch list and run
it through Colab's network first, matching this project's existing workaround
for an unstable local network). Either path is the same ~2.3-4.5 GB depending on
quantization; neither was started in this pass.
