# Qwen3-TTS: current APIs, licensing, and T4 implications (issue #52)

Research snapshot: 2026-08-24. Only primary sources (Hugging Face model cards, the
`QwenLM/Qwen3-TTS` and `vllm-project/vllm-omni` repositories/docs) were consulted.
Nothing was installed or downloaded. Where a primary source did not state an
answer, this is recorded explicitly rather than inferred.

## Model identity and variants

Confirmed exact Hugging Face IDs (all under the `Qwen` org):

| Variant | 0.6B | 1.7B |
| --- | --- | --- |
| Base (voice cloning) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| CustomVoice (predefined timbres + instruction control) | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| VoiceDesign (voice from natural-language description) | not published | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |

`Qwen/Qwen3-TTS-12Hz-0.6B-Base` matches the issue's Phase 1 model exactly — no ID
correction needed. VoiceDesign is **1.7B-only**; there is no 0.6B VoiceDesign
checkpoint, so Phase 2 VoiceDesign testing must use the 1.7B model.

Capability matrix, per model card text (do not assume all three variants expose
identical controls — the issue explicitly warns against this):

| Variant | Voice cloning | Predefined timbres | Instruction/style control | Voice from description |
| --- | --- | --- | --- | --- |
| Base | ✅ (`generate_voice_clone`, 3s reference) | — | — | — |
| CustomVoice | — | ✅ (9 premium timbres) | ✅ (`instruct=` free-text) | — |
| VoiceDesign | — | — | ✅ (drives the design) | ✅ (natural-language persona) |

Sources:

- [`Qwen/Qwen3-TTS-12Hz-0.6B-Base` model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [`QwenLM/Qwen3-TTS` repository](https://github.com/QwenLM/Qwen3-TTS)

## Languages and Russian support

All three variants' model cards state the same 10-language list: **Chinese,
English, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
Italian**. Russian is an explicit, first-party claim here — unlike Higgs STT,
where the model card only claims English and Russian is empirical (see
`docs/research/higgs-current-apis.md`). This is a meaningful difference to record
in the comparison matrix: Qwen3-TTS's Russian support is a documented claim, not
an assumption; actual output quality is still an empirical question.

Source: model cards above (all three repeat the same language list).

## License

**Apache-2.0** on every published model card checked (0.6B-Base, 1.7B-CustomVoice,
1.7B-VoiceDesign). This is a material difference from Higgs TTS 3, which uses
Boson's research/non-commercial license. Apache-2.0 removes the "no production
use" caveat this project has to carry for Higgs — the README's Higgs-vs-Qwen
summary should state this explicitly, since it directly answers the issue's
"licensing implications for generated audiobooks" requirement.

Source: model cards above.

## Native Python API (not the serving path used here)

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
wavs, sr = model.generate_voice_clone(
    text="...", language="Russian", ref_audio="...", ref_text="...",
)
```

CustomVoice/VoiceDesign use `generate_custom_voice(text=..., language=..., speaker=...,
instruct=...)`. The model card's own load path uses `dtype=torch.bfloat16` and
`flash_attention_2` unconditionally — the same pattern already seen with Higgs's
CUDA card, and the same reason this project does not simply copy a model card's
device/dtype choice onto unverified hardware. This native path is not what the
runner will use (vLLM-Omni server mode is), but it confirms the checkpoint's
native dtype is bf16, same as Higgs.

Source: [`Qwen/Qwen3-TTS-12Hz-0.6B-Base` model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

## vLLM-Omni serving path (the one this runner will use)

vLLM-Omni states **day-0 support** for Qwen3-TTS, added in
`vllm-project/vllm-omni#895` (targeting the project's `v0.14.0rc1` milestone;
online serving landed later in `#968`). This project already pins
`vllm-omni==0.26.0` for Higgs (`src/tts_cuda.py`); whether that PyPI release
actually contains the merged Qwen3-TTS support and the online-serving change is
**not confirmed from the model card or PR text alone** — the runner must check at
install/serve time (does `vllm_omni/deploy/qwen3_tts.yaml` exist in the installed
package; does `vllm serve --omni` accept `task_type`) rather than assume it.

### Server command

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091 --trust-remote-code --enforce-eager
```

The same `vllm serve MODEL --trust-remote-code --omni` shape used for Higgs; the
model ID selects the variant (Base/CustomVoice/VoiceDesign), one server per
variant, matching this project's "one server, one benchmark stage" pattern.

### `POST /v1/audio/speech` request schema

- `input` (string, required) — text to synthesize.
- `voice` (string, default `"vivian"`) — CustomVoice predefined speaker name.
- `response_format` (default `"wav"`).
- `speed` (float, default 1.0, range 0.25–4.0).
- `task_type` (string, default `"CustomVoice"`) — `"CustomVoice"` | `"VoiceDesign"` | `"Base"`.
- `language` (string, default `"Auto"`).
- `instructions` (string, default `""`) — style/emotion/design directive text.
- `ref_audio`, `ref_text` — Base (voice-clone) task only.
- `max_new_tokens` (int, default 2048).
- `stream` / `stream_format` — not needed for this benchmark (no streaming requirement in #52).

Mapping to this project's three required tests:

| Test | `task_type` | Model | Key fields |
| --- | --- | --- | --- |
| Basic Russian TTS | `CustomVoice` | `0.6B-CustomVoice` (Phase 1) / `1.7B-CustomVoice` (Phase 2) | `input`, `language="Russian"`, `voice=<one of 9 timbres>` |
| Voice cloning | `Base` | `0.6B-Base` (Phase 1) / `1.7B-Base` (Phase 2) | `input`, `ref_audio`, `ref_text`, `language="Russian"` |
| Emotion/style instruction | `CustomVoice` (or `VoiceDesign` for a fully designed narrator) | `1.7B-CustomVoice` / `1.7B-VoiceDesign` | `instructions="<real natural-language directive>"` |

**Note on Phase 1 (0.6B):** the model card lists 0.6B in **Base** and
**CustomVoice** only — there is no 0.6B VoiceDesign checkpoint. The T4 control run
therefore covers basic Russian TTS (CustomVoice) and voice cloning (Base) with the
0.6B model; the emotion/style instruction test on 0.6B uses CustomVoice's
`instructions` field, not VoiceDesign, since VoiceDesign has no small checkpoint.
Full VoiceDesign coverage is Phase 2 (1.7B) as the issue specifies.

Sources:

- [vLLM-Omni Speech API reference](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/speech_api.md)
- [vLLM-Omni Qwen3-TTS offline-inference guide](https://docs.vllm.ai/projects/vllm-omni/en/v0.18.0/user_guide/examples/offline_inference/qwen3_tts/)
- [vLLM-Omni Qwen3-TTS support PR #895](https://github.com/vllm-project/vllm-omni/pull/895)
- [vLLM-Omni default Qwen3-TTS deploy profile](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts.yaml)

## T4 / Turing compatibility — MEASURED on a real Colab T4 (2026-08-23, reproduced 2026-08-24)

**Re-run confirms this is deterministic, not a one-off.** A second full
Colab T4 run (2026-08-24), after the log-naming and
`attention_backend_observed` fixes below were applied, hit the exact same
`RuntimeError: index_copy_(): ... Half ... BFloat16` at the exact same
`gpu_model_runner.py:1723 flush_decode_batch` call site, on both
`0.6b-base` and `0.6b-customvoice`, with separate per-variant logs
(`qwen_vllm_server_0.6b-base.log`, `qwen_vllm_server_0.6b-customvoice.log`)
confirming it independently for each. `attention_backend_observed` now
correctly reports `["TRITON_ATTN"]` on both runs (previously the
FLASHINFER-substring false positive) -- so both runner fixes are validated
against a real second run, and the underlying Qwen3-TTS/T4 failure is
confirmed reproducible rather than transient.

**Result: FAILED for every job on both `0.6b-base` and `0.6b-customvoice`.**
`src/tts_qwen_cuda.py` was actually run on a Colab Tesla T4 (compute 7.5,
vllm-omni 0.26.0, `--dtype float16` forced as this project's runner already
does below compute 8.0). Full server logs recovered from Google Drive
(`qwen_vllm_server_0.6b-customvoice.log`; the `0.6b-base` run's log was lost to
a since-fixed filename-clobbering bug, see below) show:

- **The FlashInfer compute-8.0 abort predicted as a possibility below does
  NOT happen.** The log's own line confirms it: `Using TRITON_ATTN attention
  backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION']`. So
  the open question from the first research pass is answered: vLLM's
  auto-selector avoided FlashInfer on this T4 without any deploy-config
  override, unlike Higgs.
- **A different, previously unmeasured failure occurs instead**, on every
  synthesis request, at the point where the Talker stage batches per-request
  embeddings:

  ```
  File ".../vllm_omni/worker/gpu_model_runner.py", line 1723, in flush_decode_batch
      inputs_embeds.index_copy_(0, offsets_t, req_embeds)
  RuntimeError: index_copy_(): self and source expected to have the same dtype, but got (self) Half and (source) BFloat16
  ```

  `inputs_embeds` follows the forced `--dtype float16`, but `req_embeds` (the
  embeddings for a specific request, produced by a submodule this research
  pass has not yet localized in vllm-omni's Qwen3-TTS integration — see the
  "Выводы" section below for what to check next) stays in the checkpoint's
  native `bfloat16` regardless of that global override. This is an
  `EngineDeadError` that kills the whole server process for every subsequent
  request in that run, not a silent bad-audio case like Higgs's #48 — it is a
  harder failure (a crash, not degenerate output), but the same root category:
  a component that the global `--dtype float16` override does not actually
  reach on a GPU with no BF16 hardware support.
- This happened for `qwen_tts_clone` (`0.6b-base`) and for both
  `qwen_tts_basic` and `qwen_tts_style` (`0.6b-customvoice`) — i.e. for every
  attempted request regardless of `task_type`, so it is not specific to voice
  cloning or to CustomVoice's `instructions` field.
- Two bugs in this project's own runner were found from this run and are
  already fixed (`src/tts_qwen_cuda.py`, `fix/52-qwen-t4-run-followups`): (1)
  `attention_backend_diagnostics()` did a bare-substring search for
  `"FLASHINFER"` that false-positived on the unrelated
  `VLLM_USE_FLASHINFER_SAMPLER` env-var name in an unrelated warning line,
  reporting `FLASHINFER` even though `TRITON_ATTN` was what actually ran; (2)
  the per-run server log used a fixed filename (`qwen_vllm_server.log`)
  shared across every `--model-variant`, so the notebook's per-variant loop
  silently clobbered the previous variant's log — the `0.6b-base` run's full
  log is lost for exactly this reason. Both are fixed; logs are now named
  `qwen_vllm_server_<variant>.log` and the diagnostic matches vLLM's actual
  `"Using X attention backend"` line.

### Where `req_embeds`'s dtype comes from, and related upstream evidence

Reading `vllm_omni/worker/gpu_model_runner.py` at the `v0.26.0` tag
([source](https://raw.githubusercontent.com/vllm-project/vllm-omni/v0.26.0/vllm_omni/worker/gpu_model_runner.py)):
`inputs_embeds` is allocated lazily inside `flush_decode_batch` and its dtype
is copied **from** `req_embeds`, not forced to the engine's configured dtype:

```python
if inputs_embeds is None:
    inputs_embeds = torch.empty(
        (preprocess_input_ids.shape[0], req_embeds.shape[-1]),
        device=req_embeds.device,
        dtype=req_embeds.dtype,
    )
```

So the crash means two *different* `req_embeds` tensors arrived with two
different dtypes across calls (one float16, one still bfloat16) — `req_embeds`
itself comes from an optional per-model hook,
`getattr(self.model, "preprocess_decode_batch", None)`. Which specific
Qwen3-TTS submodule that hook calls, and why it doesn't uniformly follow
`--dtype float16`, was **not resolved** — the model's own `modeling_qwen3_tts`
source was not reachable via raw GitHub fetch in this research pass. This is
recorded as unresolved, not guessed.

Two directly relevant, independent upstream data points found:

- **[vllm-omni PR #3253](https://github.com/vllm-project/vllm-omni/pull/3253)
  (merged)**: "`[Bugfix][Qwen3TTS] Use float32 for code predictor on
  fp16-only GPUs`" — wraps `CodePredictorBaseModel` in a `torch.amp.autocast`
  to `float32` specifically when running fp16-only on T4/SM75-class hardware,
  tested on a `g4dn.xlarge` (T4). This is Qwen3-TTS-specific and already
  merged, so it likely **is** present in the pinned `vllm-omni==0.26.0` — but
  it targets the *code predictor* submodule, not whatever produces the
  Talker-stage `req_embeds` that actually crashed here. It confirms this
  general class of fp16-only-GPU dtype bug in Qwen3-TTS's vllm-omni
  integration was already known and partially fixed upstream, in a different
  submodule than the one this run hit.
- **[vllm-omni issue #4838](https://github.com/vllm-project/vllm-omni/issues/4838)
  (open)**: a **different** Omni TTS model (Voxtral TTS) crashes on a T4 with
  the same error class — "expected scalar type BFloat16 but found Half" —
  described as "Stage-1 has hardcoded `torch.bfloat16`, Stage-0's fallback to
  float16 doesn't apply." This is not Qwen3-TTS and does not confirm the exact
  same code path, but it is independent evidence that this class of
  "one submodule silently keeps bfloat16 despite a global fp16 override, and
  it crashes rather than degrades on hardware with no bf16 support" bug is a
  recurring, still-open pattern in vllm-omni's serving of multi-stage TTS
  models on Turing-class GPUs — not something specific to this project's
  configuration.
- `engine_extras`'s per-stage dtype override (used for Higgs's stage-1 codec
  decoder in `configs/higgs_multimodal_qwen3_turing.yaml`) is **not
  documented** anywhere in vllm-omni's docs found by this search — it is only
  known to work empirically from that existing config. Whether a similar
  `engine_extras: {dtype: float16}` on Qwen3-TTS's stage 0 would reach the
  submodule that produces `req_embeds` is unconfirmed and was not testable
  without a GPU. **No speculative `configs/qwen3_tts_turing.yaml` is added
  in this pass** — inventing an unverified deploy-config fix and reporting it
  as a solution would violate this project's own reproducibility rule, and a
  GPU is required to test whether any override actually reaches the right
  submodule.

No primary source (model card, `QwenLM/Qwen3-TTS`, vLLM-Omni docs, the Qwen3-TTS
deploy YAML, or the vLLM-Omni PR/RFC issues found) states a minimum GPU compute
capability for Qwen3-TTS, and none mentions T4/Turing/sm75 specifically. This
mirrors Higgs's situation exactly (also undocumented on Turing) — the issue's own
framing ("diagnostic control on T4") anticipates this gap rather than a resolved
answer. The paragraphs below are the original, pre-run analysis; the actual
measured result is above.

What is verifiable from the deploy config and the checkpoint metadata:

- `vllm_omni/deploy/qwen3_tts.yaml` sets `max_num_seqs`, `gpu_memory_utilization`,
  `max_num_batched_tokens`, `max_model_len` per stage but **does not pin
  `attention_backend`** (unlike `higgs_multimodal_qwen3.yaml`, which pins
  `FLASHINFER` and is exactly what breaks Higgs on a T4 — see
  `docs/research/higgs-current-apis.md`). This means the FlashInfer
  compute-capability-8.0 startup abort that forced
  `configs/higgs_multimodal_qwen3_turing.yaml` for Higgs is **not automatically
  reproduced** for Qwen3-TTS; vLLM's default attention-backend auto-selection
  applies instead, which does have Turing-capable fallbacks (Triton). This is a
  structural difference worth testing for, not an assumption to report as fact —
  vLLM's auto-selector could still choose FlashInfer opportunistically depending
  on the installed vLLM version.
- The checkpoint's native dtype is **bfloat16** (model card load example). On a
  T4 (no BF16 hardware support), vLLM refuses BF16 below compute capability 8.0 —
  the same gate already measured for Higgs. `--dtype float16` is therefore the
  same forced choice, and the same downcast risk Higgs hit for its stage-1 codec
  decoder (issue #48) is a plausible failure mode for Qwen3-TTS's Code2Wav stage
  too, given the two-stage Talker/Code2Wav architecture is architecturally
  analogous. This must be measured on the actual T4 run, not assumed.
- No Qwen3-TTS-specific Turing deploy-config equivalent to
  `configs/higgs_multimodal_qwen3_turing.yaml` exists upstream or in this
  repository yet.

### FlashInfer SM75 regression, general (not Qwen3-TTS-specific)

A general vLLM issue confirms FlashInfer dropped SM75 as a candidate backend in a
recent vLLM release for some code paths (FP8 KV cache), independent of any
specific model — consistent with the same floor already measured for Higgs
(`FlashInferBackend.supports_compute_capability` requiring `>= (8, 0)` in vLLM
0.26.0). This is general vLLM/FlashInfer platform behavior, not a Qwen3-TTS
model-card claim, and is cited here only as corroborating context.

Source: [vLLM issue #47549 — FlashInfer no longer available on SM75](https://github.com/vllm-project/vllm/issues/47549)

## Open questions / not found in primary sources

- Whether the pinned `vllm-omni==0.26.0` (already used for Higgs in this repo)
  actually contains the Qwen3-TTS `deploy/qwen3_tts.yaml` and the `task_type`
  request field — PR #895/#968 dates and version numbers did not resolve this
  cleanly from the pages fetched. **Verify at install/serve time** (check for the
  file inside the installed package; a missing deploy config or unsupported
  `task_type` should be a genuine `SKIPPED`, not a silent downgrade).
- Whether Qwen3-TTS's default sample rate over the HTTP API is 24 kHz (matching
  Higgs, `wav_duration`/`audio_statistics` reuse) or 16 kHz (the offline
  `end2end.py` example's default `--sampling-rate`) — must be read from the
  actual returned WAV header, not assumed.
- Whether vLLM's auto-selected attention backend on a T4 for the Qwen3 dense
  decoder architecture used by the Talker stage actually avoids a FlashInfer
  compute-capability abort in the installed vLLM version, or whether a
  Turing-specific `--deploy-config` (forcing `TRITON_ATTN`, by analogy with
  Higgs) will still be required. Unconfirmed until measured.
- No commercial/redistribution restriction beyond standard Apache-2.0 terms was
  found on any Qwen3-TTS model card; no separate "generated audio" licensing
  clause (unlike some TTS providers) was located. This is stated as "not found",
  not as a guarantee — recheck the model card at implementation time in case of
  updates.

## Stress control — no official mechanism; three in-text workarounds tried, none confirmed to work (2026-08-24)

Motivation: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (the local Apple
Silicon path above) mispronounces stress on several `samples/tts_ru.txt`
words, especially the Sanskrit/Vaishnava proper nouns. Before touching the
shared fixture, this was researched, not guessed:

- **Neither Higgs nor Qwen3-TTS documents a stress-control mechanism.**
  Higgs's `PROMPTING.md` covers only emotion/prosody/style tags (see
  `docs/research/higgs-current-apis.md`) and says nothing about diacritics or
  pronunciation control. Qwen3-TTS has no tag-based DSL for anything word- or
  phoneme-level either — its only structured markup is the `<|im_start|>` /
  `<|im_end|>` ChatML wrapper around the free-text `instruct` argument (style
  direction for the whole utterance, not per-word pronunciation); confirmed
  by reading `mlx_audio/tts/models/qwen3_tts/qwen3_tts.py` directly in this
  repository's own `.venv-tts`.
- **This is a known, open, unresolved community request for Qwen3-TTS**:
  [`QwenLM/Qwen3-TTS` Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185)
  ("Russian language TTS issues") reports the same mispronunciation and shows
  users experimenting with the Unicode combining acute accent (U+0301);
  [Discussion #53](https://github.com/QwenLM/Qwen3-TTS/discussions/53)
  ("Adding Pronunciation/Stress Control") shows a community vote (76% for
  SSML tags, 23% for a pronunciation dictionary) with no maintainer response
  found. A related project, [`k2-fsa/OmniVoice` issue #65](https://github.com/k2-fsa/OmniVoice/issues/65),
  states plainly that U+0301 is "ignored by model" for its own TTS stack —
  a warning sign, not a guarantee either way for Qwen3-TTS specifically.

Three in-text workarounds were tried against
`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (local M1, `basic` mode),
verified only by transcribing the generated WAV back through
`mlx-community/Qwen3-ASR-0.6B-8bit` and comparing the round-trip text — this
detects garbling but **cannot confirm correct stress placement**, since
Russian ASR output does not mark stress. None of the three modified
`samples/tts_ru.txt`, tested, then reverted it — the fixture stays unmarked
in the repository until a workaround is actually confirmed to help by ear:

1. **Combining acute accent (U+0301)** immediately after the stressed vowel
   (e.g. `Вриндава́н`): made pronunciation measurably **worse**. The ASR
   round-trip introduced spurious syllables exactly at the marked positions —
   `Вриндава́н` → "Вриндава**Юн**", `Радхара́ни` → "Радхра**Юни**",
   `Шрима́д` → "Шрема**Юд**" — where the unmarked baseline produced clean
   `Вриндаван`, `Радхарани`, `Шри Мадбахагаватом`. The model does not ignore
   the mark; it appears to vocalize something in response to it.
2. **A literal `+` before the stressed vowel** (a convention used by some
   Russian auto-accentuation tools, e.g. `Вриндав+ан`): **worse still** — the
   symbol is read out loud as the word "plus"/"плюс" or similar noise
   (`Сег+одня` → "Секплюгня", `+Это` → "Plus, Эйхолог", `Кр+ишна` →
   "Карплюфишно"). Consistent with unrecognized symbols being "read
   literally" rather than ignored.
3. **Doubling the stressed vowel letter** (a length/emphasis hint using only
   existing Cyrillic letters, no special symbols — e.g. `Вриндаваан`,
   `Радхараани`): did **not** introduce the garbling of (1) or (2). The ASR
   round-trip came back close to the clean baseline (`Вриндаван` →
   "в рендован", `Радхарани` → "Радхарани", `Раадха-Раман` → "Раадха Раман",
   with `Раадха` notably transcribed with its doubling still audible in some
   form). This is the least-bad of the three, but **not a confirmed fix**:
   ASR normalizes spelling regardless of where the stress actually landed, so
   this only shows the workaround doesn't damage the *segmentation* of the
   audio, not that it corrects the *stress*. Needs an actual listen before
   being treated as anything more than "didn't make it worse."

**Conclusion for this pass**: stress control is an unresolved upstream
limitation for both TTS backends this project uses, not something to work
around with an invented per-project convention until one is verified by ear.
`samples/tts_ru.txt` is left unmarked. If pursued further, the vowel-doubling
result is the only one of the three worth a human-listened follow-up; U+0301
and `+` are now recorded as actively harmful for this checkpoint and should
not be retried without new evidence.

## Выводы для реализации (`src/tts_qwen_cuda.py`)

Follow the same architecture as `src/tts_cuda.py`, not a rewrite:

1. **Backend**: default and only backend is `vllm` (vllm-omni). SGLang-Omni is not
   evaluated for Qwen3-TTS in this pass — Higgs's SGLang path already failed
   structurally on T4 (`KeyError: 'sm_75'`) for unrelated reasons (CUTLASS-DSL
   RMSNorm), and no Qwen3-TTS SGLang recipe was found in this research pass.
2. **Model IDs**: parametrize `--model-variant` over
   `{0.6b-base, 0.6b-customvoice, 1.7b-base, 1.7b-customvoice, 1.7b-voicedesign}`,
   mapping to the exact HF IDs listed above. Default to
   `Qwen/Qwen3-TTS-12Hz-0.6B-Base` for the Phase 1 T4 diagnostic per the issue.
3. **Server command**: `vllm serve <model_id> --trust-remote-code --omni --host
   127.0.0.1 --port <port>`, reusing this project's `wait_for_server`/`/health`
   pattern verbatim. Below compute capability 8.0, force `--dtype float16` (same
   gate as Higgs) and log whatever attention backend vLLM actually selects
   (do not assume Triton — read it from the server log, same spirit as
   `fp16_cast_diagnostics`).
4. **Deploy config — UPDATED after the real T4 run (2026-08-23)**: the
   FlashInfer/attention-backend concern this point originally flagged did
   **not** materialize (TRITON_ATTN was selected correctly, unmodified). A
   *different* failure did: `RuntimeError: index_copy_(): ... Half ...
   BFloat16` inside vllm-omni's own `flush_decode_batch`, on every request,
   independent of `task_type` — see the measured section above. This is
   evidenced as a known, still-partially-unresolved upstream pattern
   (merged fix for a *different* Qwen3-TTS submodule in
   [vllm-omni#3253](https://github.com/vllm-project/vllm-omni/pull/3253); an
   open, same-symptom issue for a *different* model in
   [vllm-omni#4838](https://github.com/vllm-project/vllm-omni/issues/4838)),
   not something this project introduced by misconfiguration. **Still no
   speculative `configs/qwen3_tts_turing.yaml` is authored**: `engine_extras`'s
   per-stage dtype override is undocumented, and which submodule actually
   produces the crashing `req_embeds` tensor was not located in this research
   pass — writing an unverified override and calling it a fix would be exactly
   the kind of unearned "PASSED" this project's rules forbid. The next
   concrete step, when GPU time is available again, is either (a) try
   `engine_extras: {dtype: float16}` on stage 0 as an experiment and record
   whether it actually changes anything, or (b) file/track an upstream issue
   analogous to #4838 but for Qwen3-TTS's Talker-stage `req_embeds` path.
5. **Payload**: `POST /v1/audio/speech` with `task_type`, `language="Russian"`,
   and `instructions` fields as documented above. Reuse `audio_statistics()` /
   `audio_defect()` from `src/tts_cuda.py` verbatim (import or factor into a
   shared module) — the issue requires the *same* anti-false-positive waveform
   checks Higgs uses; do not write a second, weaker validator.
6. **Jobs** (Phase 1, 0.6B, T4 diagnostic):
   - `qwen_tts_basic`: `task_type=CustomVoice`, plain Russian sentence, a fixed
     predefined `voice`.
   - `qwen_tts_clone`: `task_type=Base`, same reference audio/text this project
     already uses for Higgs cloning (`samples/reference.wav`/`.txt`), `SKIPPED`
     when absent — matching Higgs's own skip semantics.
   - `qwen_tts_style`: `task_type=CustomVoice` with a real `instructions` string
     (e.g. "Read slowly and thoughtfully, with a warm, contemplative tone,
     becoming slightly more emotional toward the end." in Russian) — never an
     invented tag DSL, per the issue's explicit constraint.
7. **Phase 2** (1.7B, separate follow-up run once Phase 1 is verified): add
   `qwen_tts_voicedesign` (`task_type=VoiceDesign`, no `voice`, `instructions`
   describing the target narrator persona) against `1.7B-VoiceDesign`, and repeat
   basic/clone/style against `1.7B-CustomVoice` / `1.7B-Base` for the audiobook
   capability comparison the issue asks for.
8. **Report fields**: mirror `tts_cuda.py`'s JSON report shape (`status`,
   `results[]`, `peak_device_vram_bytes`, `server_command`, `server_env`) so the
   notebook's existing metrics-reading and summary-table code needs only
   additive changes, not a parallel format.
9. **GPU-detection reporting**: the notebook must record GPU name/VRAM/compute
   capability once (already implemented in cell 5) and pass the same `GPU`
   dict's capability into the Qwen stage's gating, exactly like the existing TTS
   stage — no separate detection logic.
