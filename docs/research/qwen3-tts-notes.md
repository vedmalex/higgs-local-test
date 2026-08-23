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

## T4 / Turing compatibility — not documented, must be measured

No primary source (model card, `QwenLM/Qwen3-TTS`, vLLM-Omni docs, the Qwen3-TTS
deploy YAML, or the vLLM-Omni PR/RFC issues found) states a minimum GPU compute
capability for Qwen3-TTS, and none mentions T4/Turing/sm75 specifically. This
mirrors Higgs's situation exactly (also undocumented on Turing) — the issue's own
framing ("diagnostic control on T4") anticipates this gap rather than a resolved
answer.

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
4. **Deploy config**: start with vLLM-Omni's own
   `vllm_omni/deploy/qwen3_tts.yaml` unmodified (it does not pin FlashInfer, so it
   may not need a Turing override at all). Add a
   `configs/qwen3_tts_turing.yaml` **only if** a real T4 run shows a
   startup-time attention-backend or dtype failure analogous to Higgs's — do not
   pre-author a speculative override before that evidence exists, unlike the
   Higgs case where the FlashInfer pin was verified in the upstream YAML before
   the override was written.
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
