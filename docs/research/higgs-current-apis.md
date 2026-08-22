# Higgs Audio v3: current APIs and Apple-Silicon implications

Research snapshot: 2026-08-21. Only primary sources were consulted. No package or model was installed or downloaded.

## TTS: model identity and MLX-Audio API

Use the canonical Hugging Face ID:

```text
bosonai/higgs-tts-3-4b
```

The current MLX-Audio Higgs v3 guide still spells the ID as
`bosonai/higgs-audio-v3-tts-4b`, but that Hugging Face URL currently returns
an HTTP 307 redirect to `bosonai/higgs-tts-3-4b`. MLX-Audio's current model
detector recognizes both the upstream `higgs_multimodal_qwen3` model type and
hyphenated `higgs-audio-v3` paths, so the canonical redirected repository is
the sensible ID to pin in this test project.

Sources:

- [MLX-Audio Higgs Audio v3 guide](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/higgs_audio_v3.md)
- [MLX-Audio Higgs v3 detector](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/higgs_audio_v3/__init__.py)
- [Canonical Boson model card](https://huggingface.co/bosonai/higgs-tts-3-4b)
- [Legacy ID which redirects to the canonical model](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

Current MLX-Audio supports Python `>=3.10`; Python 3.11 is the conservative
choice for this M1 test (it also avoids being on the newest interpreter edge).
The actual installed version and resolved model ID should still be logged by
`bootstrap.sh`, because the PyPI release can lag `main`.

Source: [MLX-Audio package metadata](https://github.com/Blaizzy/mlx-audio/blob/main/pyproject.toml)

### CLI

The supported command shape is:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-tts-3-4b \
  --text "Сегодня мы проверяем русский синтез речи."
```

For cloning, repeat `--ref_audio` and `--ref_text` as paired flags. A single
reference is sufficient for this benchmark:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-tts-3-4b \
  --text "Тестовая русская фраза." \
  --ref_audio samples/reference.wav \
  --ref_text "Точная транскрипция reference audio."
```

### Python API

The current model-specific guide uses this API (not `load_model`):

```python
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write as audio_write

model = load("bosonai/higgs-tts-3-4b")
for result in model.generate(
    text=text,
    temperature=1.0,
    max_new_tokens=2048,
):
    audio_write(output_path, result.audio, result.sample_rate)
```

`model.generate()` is a generator. Benchmark generation through completion,
then compute WAV duration from the saved sample count/rate or the written WAV.
For cloning, add `ref_audio` and its exact `ref_text`. The model also exposes
`encode_reference_audio()` so repeated runs can reuse codes, but that would
distort a cold-load benchmark and is unnecessary here.

Source: [MLX-Audio Higgs Audio v3 guide](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/higgs_audio_v3.md)

## TTS control tags

The official format is `<|category:value|>`. Sentence-level tags go at the
start of a sentence. Only pause tags are inserted inline at the desired
position. A suitable Russian controls test using only documented tags is:

```text
<|emotion:contentment|><|prosody:speed_slow|>Сегодня мы начинаем спокойный рассказ. <|prosody:pause|> Теперь голос становится выразительнее.
<|emotion:enthusiasm|><|prosody:expressive_high|>Это важная и радостная часть нашего теста! <|prosody:long_pause|> <|style:whispering|>А теперь рассказ завершается тихо и спокойно.
```

Documented catalog:

- Emotion (sentence-level): `affection`, `amusement`, `anger`, `arousal`,
  `awe`, `bitterness`, `confusion`, `contemplation`, `contentment`,
  `determination`, `disgust`, `elation`, `enthusiasm`, `fear`, `helplessness`,
  `longing`, `pride`, `relief`, `sadness`, `shame`, `surprise`.
- Prosody sentence-level: `speed_very_slow`, `speed_slow`, `speed_fast`,
  `speed_very_fast`, `pitch_low`, `pitch_high`, `expressive_high`,
  `expressive_low`.
- Prosody inline: `pause`, `long_pause`.
- Style sentence-level: `singing`, `shouting`, `whispering`.
- SFX inline: `cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`,
  `sigh`, `sniff`, `sneeze`. SFX require matching onomatopoeia immediately
  after the tag with no intervening space.

Source: [Boson PROMPTING.md](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md)

The weights use Boson's research/non-commercial license (with a separately
described creator-use grant), not Apache-2.0. The benchmark README should link
the license and avoid implying production permission.

Source: [Boson TTS model card and license notice](https://huggingface.co/bosonai/higgs-tts-3-4b)

## STT: authoritative checkpoint and API

Use only:

```text
bosonai/higgs-audio-v3-stt
```

The card describes a 2.68B model: Whisper Large-v3 encoder plus Qwen3-1.7B
decoder, with 16 kHz mono audio input. It is tagged English, not multilingual;
there is no primary-source claim establishing Russian support. Russian must
therefore be treated strictly as an empirical result.

Source: [Boson STT model card](https://huggingface.co/bosonai/higgs-audio-v3-stt)

Loading requires custom code and eager attention:

```python
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
model.to(device)
```

Do not copy the card's `device_map="cuda:0"` or BF16 choice to the Mac. For
the requested experiment, load on CPU first, then explicitly move to MPS. Log
`torch.backends.mps.is_built()` and `.is_available()` independently.

### `boson_multimodal`: current safe resolution

The model card still says preprocessing requires `boson_multimodal` and imports
it from the official `boson-ai/higgs-audio` repository. That repository is the
only authoritative source; its distribution name is `boson_multimodal==0.1.0`.
It is not appropriate to install an unrelated similarly named PyPI project.

However, installing the old repository wholesale is a bad fit for this STT
environment: its official requirements pin `transformers>=4.45.1,<4.47.0`,
while the current STT card requires `transformers>=4.51.0`. The repository's
current README also says Higgs v3 no longer depends on that repository.

More importantly, the current STT model repository now bundles
`higgs_audio_collator.py`, and its bundled `transcribe.py` imports that sibling
collator directly. Therefore the preferred current implementation is:

1. install `torch`, a compatible current `transformers`, `soundfile`, `numpy`,
   `huggingface_hub`, and the small dependencies actually reported by an import
   smoke test;
2. fetch the checkpoint's small `transcribe.py` and required sibling Python
   files at the **same pinned model revision** using `hf_hub_download` (this is
   code, not model weights);
3. call its public `transcribe(model, tokenizer, audio, sample_rate=16000)`;
4. keep `trust_remote_code=True` for the custom model architecture.

This follows the model repository's current evaluation path and avoids the
incompatible legacy package. If direct use of `boson_multimodal` is still
needed, install only from the official Git commit, pinned explicitly, and
expect to resolve its old Transformers constraint rather than silently mixing
versions:

```bash
pip install "boson_multimodal @ git+https://github.com/boson-ai/higgs-audio.git@<PINNED_COMMIT>"
```

Sources:

- [Current STT repository files](https://huggingface.co/bosonai/higgs-audio-v3-stt/tree/main)
- [Bundled STT transcribe helper](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/transcribe.py)
- [Official Boson package metadata](https://github.com/boson-ai/higgs-audio/blob/main/setup.cfg)
- [Official Boson legacy dependency pins](https://github.com/boson-ai/higgs-audio/blob/main/requirements.txt)
- [Official Boson v3 migration notice](https://github.com/boson-ai/higgs-audio)

Pin the Hugging Face revision in benchmark logs. The `main` checkpoint and
helper code were updated in June 2026; reproducibility requires recording the
resolved commit SHA, not merely the model ID.

### MPS risk identified in current remote code

There is no official MPS support claim. The current custom architecture has a
device-independent path when using eager attention, but it also forcibly casts
`audio_features` to `torch.bfloat16` inside `modeling_higgs_audio.py`, regardless
of the dtype requested at load time. This creates two concrete risks on M1:

- BF16 operation support may be incomplete for the model's MPS operations;
- loading FP16 or FP32 weights can encounter mixed-dtype failures because the
  audio features are forced to BF16.

The same file contains CUDA graph code and hard-coded CUDA allocations, but
those are in the optional CUDA-graph preparation path and should not be reached
by the normal eager MPS/CPU transcription path. This is a source audit, not
proof of inference compatibility.

Source: [Current STT custom model implementation](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/modeling_higgs_audio.py)

Implementation order for an honest test:

1. MPS availability checks.
2. MPS FP16 load and one short inference; retain full traceback on failure.
3. If the error is specifically a dtype mismatch/unsupported BF16 operation,
   record it before trying anything else.
4. A minimal MPS FP32 attempt is informative, but the hard-coded BF16 cast may
   make it fail for the same reason; do not patch remote model code and call
   that stock support.
5. Fully terminate the failed MPS process, then try CPU FP32 in a fresh process.
6. Report load success separately from successful generation.

Do not use BF16 on M1 merely because the CUDA model card does. Do not claim MPS
support until `generate()` completes on real audio.

## Condensed implementation decisions

- Python: 3.11.
- TTS model: `bosonai/higgs-tts-3-4b` (canonical target of MLX guide's legacy ID).
- TTS API: `mlx_audio.tts.utils.load`, iterate `model.generate`, write with
  `mlx_audio.audio_io.write`.
- Controls: use only values from Boson's `PROMPTING.md`; pauses are inline.
- STT model: `bosonai/higgs-audio-v3-stt`, never the 8B checkpoint.
- STT preprocessing/inference helper: pin and use the model repository's bundled
  `transcribe.py` plus siblings; do not install a random `boson_multimodal`.
- MPS: experimental; FP16 first, CPU FP32 fallback in a separate process; log
  the entire failure and resolved package/model revisions.
- Russian STT: unknown until measured; the official metadata is English.

## CUDA (Google Colab) inference paths

Research snapshot: 2026-08-22. Sources are the pinned checkpoints and first-party
repositories; nothing was installed or downloaded.

### TTS has no `transformers` path

`bosonai/higgs-tts-3-4b` at revision `7556c17e05201fccd9c8cc120bc216dcc7b5d561`
contains no `.py` files at all, so `trust_remote_code=True` resolves no
implementation. Its `config.json` declares `model_type: higgs_multimodal_qwen3`,
`architectures: [HiggsMultimodalQwen3ForConditionalGeneration]`, and
`transformers_version: 5.5.0`, but `higgs_multimodal_qwen3` is **not implemented in
`transformers`**: `src/transformers/models/` on `main` carries only `higgs_audio_v2`
and `higgs_audio_v2_tokenizer`.

The only first-party CUDA implementation is `sglang_omni/models/higgs_tts` in
SGLang-Omni, which the model card presents as the sole serving path. Verified present
in the pinned release `sglang-omni==0.1.3` (18 modules under `models/higgs_tts/`,
`hf_config.py` matching `higgs_multimodal_qwen3`).

Server API actually exposed by that release: `GET /health`,
`POST /v1/audio/speech` with `input`, `response_format`, `max_new_tokens`,
`temperature`, `top_k`, and `references: [{audio_path, text}]` for cloning. Local
reference files require `--allowed-local-media-path`. `sgl-omni serve` has **no
`--revision` flag**, so a pinned run must `snapshot_download(..., revision=...)`
first and serve the resolved directory.

Hardware floor: the package pins `flash-attn-4>=4.0.0b18` and
`flashinfer_python[cu13]==0.6.14`, whose kernels require an Ampere-class device.
Colab's T4 is compute capability 7.5 and therefore cannot run this stack; L4 (8.9)
and A100 (8.0) can. Consequently a T4 Colab session can benchmark STT but must
report TTS as `SKIPPED`, not as a placeholder result.

Sources:

- [Boson TTS model card and SGLang-Omni usage](https://huggingface.co/bosonai/higgs-tts-3-4b)
- [TTS checkpoint file listing (no `.py` files)](https://huggingface.co/bosonai/higgs-tts-3-4b/tree/main)
- [`transformers` model registry](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
- [SGLang-Omni `higgs_tts` implementation](https://github.com/sgl-project/sglang-omni/tree/main/sglang_omni/models/higgs_tts)
- [SGLang-Omni dependency pins](https://github.com/sgl-project/sglang-omni/blob/main/pyproject.toml)
- [SGLang-Omni OpenAI-compatible speech API](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/serve/openai_api.py)

### STT runs on CUDA in the existing runner

`bosonai/higgs-audio-v3-stt` at the pinned revision
`2ffd1aa39f5a1266931e405cba12e404a9f994b2` declares `transformers_version: 4.51.0`
and ships its own remote code, so it cannot share an environment with the TTS stack.
`src/stt_test.py` therefore accepts `--device cuda`, streaming shards straight into
VRAM with `device_map={"": 0}` so host RAM never holds a full copy.

### Isolation requirement on Colab

Because the two stacks pin incompatible `transformers` versions, and because a
notebook kernel cannot reliably release a model held by a global name, each model
must run as a **separate subprocess in its own virtual environment**. Process exit is
what returns VRAM and host RAM; `del` on a function argument does not.
