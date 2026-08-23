# Higgs Audio v3 local M1 benchmark

Reproducible, local-only compatibility and Russian-quality test stand for Higgs Audio v3. TTS and STT use isolated environments and separate processes; no cloud fallback exists.

## System

Target: Apple M1, 16 GB unified memory, macOS 14.6.1, native arm64, Python 3.11 or 3.12. Python 3.11 is the tested project choice. The initial inspection found the current terminal running under Rosetta (`uname -m = x86_64`), so setup and inference were intentionally not run. Open a native arm64 terminal first.

```bash
cd ~/work/higgs-local-test
make info
make setup
```

Before inference, close Docker Desktop, local LLMs, virtual machines, and other memory-heavy processes. Do not alter macOS swap settings.

## Models and dependencies

- TTS: `bosonai/higgs-tts-3-4b`, the canonical Boson repository. MLX-Audio's older documented ID redirects to it, and the current detector recognizes its `higgs_multimodal_qwen3` model type.
- STT: `bosonai/higgs-audio-v3-stt` (Whisper Large-v3 encoder + Qwen3-1.7B decoder, approximately 2.68B). The 8B model is never used.
- STT preprocessing: the checkpoint's own `transcribe.py` and `higgs_audio_collator.py`, pinned to revision `2ffd1aa39f5a1266931e405cba12e404a9f994b2`. The legacy `boson_multimodal` distribution is deliberately not installed because its official Transformers constraint conflicts with the current STT card.
- Models remain in the Hugging Face cache and are not stored in this repository.

Primary references: [MLX-Audio Higgs v3 guide](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/higgs_audio_v3.md), [Boson TTS prompting guide](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md), and [Boson STT model card](https://huggingface.co/bosonai/higgs-audio-v3-stt). The detailed source audit is in [docs/research/higgs-current-apis.md](docs/research/higgs-current-apis.md).

## Commands

```bash
make setup            # environments/imports only; does not pre-download both models
make download-models  # pre-caches TTS, STT, and Whisper processor weights upfront
make info
make tts              # basic, official control tags, optional clone; fresh process each
make stt              # MPS FP16 inference, then CPU FP32 fallback if needed
make benchmark        # TTS then STT, sequentially
make upload-gdrive    # upload samples/, output/, and notebooks/ to Google Drive
make download-gdrive FOLDER_ID=<id> # download results from Google Drive
make clean-output
```

Add `samples/stt_ru.wav` before STT. It is normalized with system `ffmpeg` to mono 16 kHz. Optional voice cloning needs both `samples/reference.wav` and its exact transcript in `samples/reference.txt` (can be copied directly from `samples/stt_ru.wav` and `output/stt_ru.txt`); otherwise it reports `SKIPPED`. Note on cloning latency: reference audio is tokenized at 25 frames/sec into the prompt KV cache (e.g. 5–15s clips are ~125–375 tokens for faster RTF, while a 60s clip introduces ~1500 prompt audio tokens).

The STT runner does not equate loading with support: it attempts complete MPS inference. Any MPS exception and traceback are written to `logs/stt.log`; CPU FP32 is then tested in a fresh process so the failed model cannot remain resident. No BF16 is requested on M1, although the current remote model code contains a forced BF16 feature cast that may itself prove incompatible. WER uses the fixed Russian sample transcript. For 1–3 minute behavior, record the same kind of continuous Russian speech and replace `samples/stt_ru.wav`; interpret WER against a matching reference (the built-in WER is only valid for the prescribed text).

## Google Colab Benchmark & Playground

For comparing Apple Silicon M1 performance against server GPUs:

- Notebook: [`notebooks/higgs_colab_benchmark.ipynb`](notebooks/higgs_colab_benchmark.ipynb)
- Open directly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedmalex/higgs-local-test/blob/main/notebooks/higgs_colab_benchmark.ipynb)

The local M1 run and the Colab run do not share an implementation. On Apple Silicon, TTS goes through MLX-Audio, an independent MLX port of Higgs v3. The checkpoint itself ships no remote code and its `higgs_multimodal_qwen3` architecture is not implemented in `transformers`, so on CUDA there is no `from_pretrained` + `generate` path. Two first-party stacks implement it, and `src/tts_cuda.py` drives either through `--backend`:

| Backend | Server | Notes |
| --- | --- | --- |
| `vllm` (default) | `vllm serve --omni` from [vllm-omni](https://github.com/vllm-project/vllm-omni) | Supports Python 3.13, runs the Higgs stages eager, does not reach flashinfer's CuTe kernels. Installed with `uv` as `vllm==0.26.0 --torch-backend=auto` then `vllm-omni==0.26.0`, in that order — the vllm-omni wheel does not declare `vllm` as a dependency. Voice reference passed as a `data:` base64 URL. |
| `sglang` | `sgl-omni serve` — the path named on the model card | Needs Python <3.13 and CUDA graph capture, which fails on a T4 with `KeyError: 'sm_75'`. Voice reference read from an allowlisted directory. |

Neither stack claims Turing support, so a T4 remains an experiment — but nothing identified in advance rules out the vLLM path, unlike the SGLang one. Below compute 8.0 the runner passes `--dtype float16`, because the checkpoint declares bfloat16 and vLLM refuses it on pre-Ampere hardware. `--min-capability` turns the expectation into a deliberate skip, `--server-arg KEY=VALUE` and `--server-env KEY=VALUE` pass extra server flags and environment variables through.

`SKIPPED` is reported only for genuinely missing prerequisites: no CUDA device, a missing server CLI, an unsupported interpreter, a capability below an explicitly requested `--min-capability`, or a missing input file.

The notebook runs TTS first, since TTS is this project's purpose, and STT second. It **disconnects the runtime unconditionally** when finished, so Colab quota is not spent idling. Everything needed to review a run is therefore written to Drive as it happens — per-stage metrics, install logs, the server log, transcripts, WAV files, and the full output of every child process in `metrics/session.log` — and stages record failures as statuses instead of raising, because an exception would stop `Run all` before the disconnect.

Each model runs as a separate subprocess in its own virtual environment, so VRAM and host RAM are reclaimed by process exit rather than by a `del` inside the kernel. Separate environments are required and not merely tidy: STT pins `transformers==4.51.0` plus the checkpoint's remote code, while either TTS stack brings its own incompatible `transformers` and `torch`. They are created with `--without-pip --system-site-packages` (Colab's Debian build has no `ensurepip`, so plain `python -m venv` fails) and inherit Colab's preinstalled `torch` through a `.pth` file, while their own pinned packages still shadow it.

Inputs and outputs live in `MyDrive/higgs-benchmark`; model weights are cached on the VM's local disk, since 15 GB through the Drive FUSE mount would be far slower and would consume Drive quota. A missing `stt_ru.wav`, TTS text, or cloning reference yields `SKIPPED` rather than synthetic input.

## Voice Cloning & Audiobook Production Guides

- **Voice Cloning Guide & Reference Texts**: [`docs/guides/voice_cloning_guide.md`](docs/guides/voice_cloning_guide.md) — rules for recording clean 7–12s audio samples, phonetically balanced text templates, and voice profile export.
- **Audiobook Production Guide**: [`docs/guides/audiobook_guide.md`](docs/guides/audiobook_guide.md) — multi-character dialogue switching, screenplay JSON format, natural pause insertion, and complete chapter audio stitching.

## Status

Benchmarked on Apple Silicon M1 (16 GB unified memory, macOS 14.6.1, native `arm64`, Python 3.11.7).

### TTS status

```text
MLX: PASSED
TTS: PASSED (bosonai/higgs-tts-3-4b via MLX-Audio)
Russian: PASSED (Natural Cyrillic Russian speech generated)
Voice cloning: PASSED (60s reference audio cloned into 25.2s Russian speech)
Control tags: PASSED ([whispering], [sigh], [laughter], [screaming] generated)
```

### STT status

```text
MPS loading/inference: PASSED (Complete FP16 inference on Metal GPU)
CPU loading/inference: NOT NEEDED (MPS FP16 succeeded end-to-end)
Russian transcription: PASSED (Accurate Cyrillic transcription and Vaishnava terminology)
```

## Benchmark

Measurements recorded from real sequential runs on native Apple Silicon M1 (16 GB unified memory):

| Test | Device | Load | Processing | Audio | RTF | Peak RAM (RSS) | Peak Footprint |
| ---- | ------ | ---: | ---------: | ----: | --: | -------------: | -------------: |
| TTS basic | MLX | 14.15s | 145.51s | 20.72s | 7.02 | 1.72 GB | 11.22 GB |
| TTS controls | MLX | 16.58s | 201.34s | 15.96s | 12.61 | 3.68 GB | 11.02 GB |
| TTS clone (60s ref) | MLX | 17.00s | 20716.63s | 25.20s | 822.09 | 1.33 GB | 11.67 GB |
| STT | MPS (FP16) | 19.89s | 83.76s | 60.00s | 1.40 | 3.29 GB | 9.25 GB |

RTF is processing seconds divided by output audio duration (TTS) or input audio duration (STT). Values below 1.0 are faster than real time.

### Google Colab, Tesla T4 (recorded 2026-08-23)

| Test | Device | Load | Processing | Audio | RTF | Peak VRAM | Peak RSS |
| ---- | ------ | ---: | ---------: | ----: | --: | --------: | -------: |
| STT | CUDA T4 (FP16) | 81.23s | 10.71s | 60.00s | 0.178 | 5.97 GB | 6.25 GB |
| TTS (all modes) | CUDA T4 | — | — | — | — | 12.95 GB peak during startup | — |

STT transcribed 60 s of Russian speech 7.9× faster than the M1 MPS run (0.178 vs 1.40). WER is not stated: the recording used was not the repository fixture and no matching reference transcript was supplied, so no WER was measured — a comparison against the fixture text would have produced a number describing nothing.

TTS is a **documented reproducible failure on T4, not a pass and not a skip**. `sglang-omni==0.1.3` installed on a `uv`-fetched Python 3.12, the weights downloaded, and `sgl-omni serve` loaded the model — then died during CUDA graph capture:

```
File ".../flashinfer/norm/kernels/rmsnorm.py", line 1148, in _get_compiled_rmsnorm_kernel
File ".../cutlass/base_dsl/arch.py", line 106, in from_string
KeyError: 'sm_75'
```

flashinfer's CUTLASS-DSL RMSNorm kernel has no entry for Turing. flashinfer documents `FLASHINFER_USE_CUDA_NORM=1` as the CUDA-JIT fallback for that exact path, and SGLang-Omni honours a pre-set value while auto-applying it only for sm100+. `src/tts_cuda.py` now sets it for anything below compute 8.0; whether that carries the T4 through the rest of startup is not yet measured.

## Expected honest outcomes

Success or partial success are both valid. A load-only result is not a pass. Keep TTS and STT in separate processes and never try to keep both models resident.
