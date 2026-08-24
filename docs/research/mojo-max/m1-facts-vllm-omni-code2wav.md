# Raw facts: vLLM-Omni HiggsAudioV3Code2Wav (haiku extraction, v0.26.0)

Source: `vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py` (+ pipeline.py,
stage_input_processors/higgs_audio_v3.py, transformers_utils/configs/higgs_audio_v3.py)
in `.research-scratch/vllm-omni` (tag v0.26.0). Pure fact extraction, no interpretation.

## 1. Classes/functions

- `class HiggsAudioV3Code2Wav(nn.Module)` — `__init__`, `_load_from_tokenizer_repo`,
  `_resolve_model_dir`, `_read_bundled_codec_state`, `_ensure_codec_loaded`, `embed_input_ids`,
  `compute_logits`, `load_weights`, `_load_from_bundled_state`, `decode_codes`,
  `forward_chunk`, `forward`, `_validate_codes`, `_split_request_ids` (staticmethod).
- `HiggsAudioV3Code2WavForConditionalGeneration = HiggsAudioV3Code2Wav` (alias).
- stage_input_processors/higgs_audio_v3.py: `_empty_code2wav_prompt`, `_revert_delay_pattern`,
  `_filter_real_code_frames`, `talker2code2wav` (sync), `_extract_last_step_row`,
  `talker2code2wav_async_chunk` (streaming).

## 2. Layers actually instantiated in this file

- `HiggsAudioRVQ(num_quantizers=8, codebook_size=1024, codebook_dim=64, hidden_size=<from codec_state>)`
  → `self.quantizer`.
- `nn.Linear(fc2_w.shape[1], fc2_w.shape[0])` → `self.fc2`.
- `build_boson_dac_decoder(device)` and `build_higgs_audio_acoustic_decoder(tokenizer_config, device)`
  → `self.acoustic_decoder` — both imported factory functions from
  `higgs_audio_v2.higgs_audio_decoder`, NOT defined in this file. The actual conv/GroupNorm/etc.
  DAC decoder layers live there, not in the v3 code2wav file itself.
- DAC config passed to the factory: `codebook_dim=8, codebook_size=1024, decoder_hidden_size=1024,
  downsampling_ratios=[8,5,4,2,3], encoder_hidden_size=64, hidden_size=256, hop_length=960,
  model_type="dac", n_codebooks=9, sampling_rate=16000, upsampling_ratios=[8,5,4,2,3]`.

## 3. Dtype references

No `torch.bfloat16`/`torch.float16`/autocast anywhere in this file. Dtype is always resolved
dynamically from already-loaded weights:
- `quantized = quantized.to(dtype=self.fc2.weight.dtype)`
- `quantized = quantized.to(dtype=first_param.dtype)` (first param of acoustic_decoder)
- Output PCM is forced to `torch.float32` before returning (`.to(torch.float32)`).
- Everything else that's explicitly typed is int/long (codes) or float32 (empty/placeholder
  tensors), never a speech-relevant compute dtype.

**This confirms the earlier #52 finding was specific to the Qwen3-TTS talker, not a pattern
repeated in Higgs v3's Code2Wav file — this file's own code has no hardcoded compute dtype.
Any bf16/fp16 exposure comes from whatever dtype the DAC decoder's own layers (in
higgs_audio_v2/higgs_audio_decoder.py, not yet read) were loaded/instantiated with, and from the
Talker stage's output dtype before it reaches Code2Wav.**

## 4. Forward data flow (decode_codes)

- Input: `audio_codes` shape `[B, num_codebooks=8, T]`, validated then `.long()`.
- `rvq_codes = codes.transpose(0,1).long()` → `[8, B, T]`.
- `quantized = self.quantizer.decode(rvq_codes)` (internal RVQ shape not traced).
- dtype-align to fc2 weight dtype.
- `quantized = self.fc2(quantized.transpose(1,2)).transpose(1,2)`.
- dtype-align to acoustic_decoder's first param dtype.
- `audio = self.acoustic_decoder(quantized)` → DAC decode.
- unsqueeze to 3D if needed, return on `codes.device`.
- Output: `[B, 1, T*960]` PCM, forced float32, sample_rate=24000 (from config, not the 16000 in
  the bundled DAC tokenizer_config — the tokenizer_config's `sampling_rate: 16000` looks like an
  internal/encoder-side value distinct from the model's public 24000 output rate; worth
  double-checking against actual decoder upsample math before assuming either number, not
  resolved by this extraction pass).

`forward_chunk` wraps `decode_codes` and trims `left_context_size*hop_length` samples from the
front and `right_holdback_size*hop_length` from the back — streaming-chunk overlap handling.

## 5. Conditioning inputs

None beyond `runtime_additional_information` carrying `left_context_size`/`right_holdback_size`
(streaming chunk boundaries, not speaker/style conditioning). Code2Wav itself is deterministic
given codes — no speaker embedding or style vector enters at this stage in v3's architecture.

## 6. Custom kernels / vendored code

No custom CUDA kernels defined in this file. All codec layers (RVQ internals, DAC conv stack)
are imported from `higgs_audio_v2.higgs_audio_decoder` — **not yet read**; that's the file that
actually contains the Conv1d/ConvTranspose1d/GroupNorm layer definitions and is the real target
for a from-scratch MAX port, not the v3 wrapper file extracted here.

## 7. Talker → Code2Wav boundary

- Talker emits codes with a delay pattern (BOC=1024 padding); `_revert_delay_pattern` strips
  `Q-1` leading BOC pads and trailing EOC entries per codebook.
- Out-of-range codes (≥1024) get zeroed via `torch.where`.
- Sync path: flat `[Q*num_frames]` long tensor via `OmniTokensPrompt`.
- Async/streaming path: `OmniPayloadStruct` with `codes.audio` (flat long tensor) plus
  `meta.left_context_size` / `meta.right_holdback_size` / `meta.finished`.
- Code2Wav's declared output: `OmniOutput` with `multimodal_outputs["model_outputs"]` = list of
  float32 CPU waveforms, `multimodal_outputs["sr"]` = list of int32 sample-rate tensors (24000).

## 8. Config fields (HiggsAudioV3Config)

`num_codebooks=8`, `codebook_size=1026` (1024 real + BOS 1024 + EOS 1025), `audio_stream_bos_id
=1024`, `audio_stream_eos_id=1025`, `sample_rate=24000`, `frame_rate=25`. No dtype field in the
config at all — dtype is purely a load-time/runtime property of the weights.

## What M1's MAX-side research still needs to answer (not covered by this extraction)

This extraction only covers the *wrapper* (`higgs_audio_v3_code2wav.py`). The actual layer
definitions (the real "Code2Wav" numerically, i.e. the DAC-style conv/GroupNorm/upsample stack
and the RVQ codebook lookup) live in `higgs_audio_v2/higgs_audio_decoder.py`, which this pass did
**not** read. Before finalizing the M1 responsibility map, that file needs the same fact
extraction treatment — it, not this wrapper, is what actually needs a MAX-equivalent per-layer.
