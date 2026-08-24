# Upstream PR — vllm-project/vllm-omni (SUBMITTED 2026-08-24)

Prepared 2026-08-24 for issue #52; the one-line fix below was confirmed on a real
Colab Tesla T4 the same day (all three previously-crashing jobs now PASS with real,
independently re-verified waveform statistics — see
`docs/research/qwen3-tts-notes.md`, "Workaround B"). The PR was opened from
`vedmalex/vllm-omni@fix/qwen3-tts-embedding-dtype` against
`vllm-project/vllm-omni:main`. Everything below is either quoted from source read
at a named tag or from this project's own measured Colab T4 runs; no claim is
inferred.

---

## Title

`[Bug][Qwen3TTS] Talker hardcodes _embedding_dtype = torch.bfloat16, crashing every request on fp16-only GPUs (T4/sm75)`

## Body

### Summary

`Qwen3TTSTalkerForConditionalGeneration` reads the engine's configured dtype into a
local `model_dtype`, uses it for the `_tts_pad_embed` buffer, and then hardcodes
`self._embedding_dtype = torch.bfloat16` on the next line. On a GPU with no bf16
support, where vLLM forces `--dtype float16`, this makes the talker's per-request
decode embeddings bfloat16 while the engine's `inputs_embeds` is float16, and the
engine dies with `RuntimeError: index_copy_(): self and source expected to have the
same dtype` on the first synthesis request.

### Where

`vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`, in
`Qwen3TTSTalkerForConditionalGeneration.__init__` — line 446 at tag `v0.26.0`,
line 468 on `main` (checked 2026-08-24, still present):

```python
model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
self.register_buffer(
    "_tts_pad_embed",
    torch.zeros(1, int(self.talker_config.hidden_size), dtype=model_dtype),
    persistent=False,
)
self._embedding_dtype = torch.bfloat16   # <-- ignores model_dtype computed above
```

`_embedding_dtype` is the dtype every per-request embedding is cast to in
`preprocess_decode_batch` (the `preprocess_decode_batch` hook `OmniGPUModelRunner`
calls), in the scalar `preprocess()` decode branch, and in `talker_mtp`. So the
inconsistency is not cosmetic: it is the dtype of the tensor handed back to the
runner.

The mismatch surfaces in `vllm_omni/worker/gpu_model_runner.py`
(`flush_decode_batch`), which allocates `inputs_embeds` from the *first*
`req_embeds` it sees and then `index_copy_`s later ones into it:

```python
if inputs_embeds is None:
    inputs_embeds = torch.empty(
        (preprocess_input_ids.shape[0], req_embeds.shape[-1]),
        device=req_embeds.device,
        dtype=req_embeds.dtype,
    )
...
inputs_embeds.index_copy_(0, offsets_t, req_embeds)
```

### Reproduction

- GPU: NVIDIA Tesla T4 (compute capability 7.5, no bf16 support), Google Colab.
- `vllm-omni==0.26.0`.
- Models: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` and
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
- `vllm serve <model> --trust-remote-code --omni --dtype float16` (`float16` is
  forced because vLLM refuses bf16 below compute capability 8.0).
- Any `POST /v1/audio/speech` request.

```
File ".../vllm_omni/worker/gpu_model_runner.py", line 1723, in flush_decode_batch
    inputs_embeds.index_copy_(0, offsets_t, req_embeds)
RuntimeError: index_copy_(): self and source expected to have the same dtype,
              but got (self) Half and (source) BFloat16
```

The attention backend selected was `TRITON_ATTN` (`Using TRITON_ATTN attention
backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION']`), so this is
not a FlashInfer/sm75 issue. It is an `EngineDeadError`: the server process dies
for every subsequent request too.

Observed on every request in two independent full runs (2026-08-23 and
2026-08-24), for `task_type` `CustomVoice` and `Base`, with and without
`instructions`, with and without `ref_audio` — so it is not specific to a task
type, to voice cloning, or to the `instructions` field.

### Suggested fix

One line — use the `model_dtype` the same `__init__` already computes:

```python
-        self._embedding_dtype = torch.bfloat16
+        self._embedding_dtype = model_dtype
```

The prefill path needs the same treatment: `Qwen3TTSPromptEmbedsBuilder.__init__`
(`prompt_embeds_builder.py:332`) hardcodes `self._embedding_dtype =
torch.bfloat16` as well, and takes no dtype argument. Since the talker is its only
constructor (`qwen3_tts_talker.py:480`), threading `model_dtype` through as a
keyword argument that defaults to `torch.bfloat16` would fix it without touching
any existing caller. On bf16-capable hardware `model_dtype` already resolves to
`bfloat16`, so both changes are no-ops there.

Two related hardcodes in the same class are deliberately left out of this
suggestion because they are not what crashed here and may be intentional:
`self.encoder.to(dtype=torch.bfloat16)` in `__init__` and again in `load_weights`.
The encoder's `encode()` returns `torch.long` codec ids, so it does not feed the
`_embedding_dtype`-typed path — but if bf16 convolutions are also unsupported on
sm75 for that submodule, a second fix may be needed for the `Base` (ref_audio)
task type. Whether float16 is numerically adequate for the talker embedding path
is likewise a maintainer question, not something this report can answer.

### Related

- **[#3253](https://github.com/vllm-project/vllm-omni/pull/3253) (merged)** —
  `[Bugfix][Qwen3TTS] Use float32 for code predictor on fp16-only GPUs`, tested on
  a `g4dn.xlarge` (T4). Same model, same class of bug (a submodule that a global
  fp16 override does not reach), different submodule: it changes
  `vllm_omni/model_executor/models/common/qwen3_code_predictor.py` (the only
  file in its diff), not the talker's embedding dtype. Its merge commit
  (`b9789091`) is an ancestor of the `v0.26.0` tag, and the autocast-to-float32
  guard is present in that file at `v0.26.0` — so this crash happens **with
  #3253's fix already installed**, in a different, earlier code path
  (`_embedding_dtype` in `qwen3_tts_talker.py`).
- **[#4838](https://github.com/vllm-project/vllm-omni/issues/4838) (open since
  2026-07-02)** — a *different* Omni TTS model (Voxtral TTS) failing on a T4 with
  the same error class, described as "Stage-1 has hardcoded `torch.bfloat16`,
  Stage-0's fallback to float16 doesn't apply." Not the same code path, but the
  same recurring pattern: a hardcoded `torch.bfloat16` in a multi-stage TTS
  integration that a global fp16 override never reaches.
- **[vllm#47549](https://github.com/vllm-project/vllm/issues/47549)** — general
  vLLM context on sm75 backend availability; cited only to rule out an
  attention-backend explanation for this particular crash.

### Environment

- `vllm-omni==0.26.0`
- NVIDIA Tesla T4, compute capability 7.5, CUDA driver as shipped by Google Colab
- `--dtype float16` (forced by vLLM's bf16 gate below compute capability 8.0)
- Attention backend actually selected: `TRITON_ATTN`
