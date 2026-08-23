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

The 60-second reference behind the recorded cloning result was a copy of `samples/stt_ru.wav`; `samples/reference.wav` now holds a 7.4-second clip instead, so reproducing that figure means copying `samples/stt_ru.wav` back over it. The 60-second pair is also preserved on Drive as `reference_60s.wav` / `reference_60s.txt`.

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

Neither stack claims Turing support, so a T4 remains an experiment. Measured so far: SGLang-Omni fails inside a CuTe kernel lookup (`KeyError: 'sm_75'`), while the vLLM path fails only on configuration — vllm-omni's auto-discovered deploy profile pins `attention_backend: FLASHINFER`, which vLLM 0.26.0 gates at compute 8.0. Every component the vLLM path needs has a Turing-capable implementation, so `configs/higgs_multimodal_qwen3_turing.yaml` selects `TRITON_ATTN` and a single-request memory budget; the runner applies it automatically below compute 8.0 and `--deploy-config` overrides it. Below compute 8.0 the runner passes `--dtype float16`, because the checkpoint declares bfloat16 and vLLM refuses it on pre-Ampere hardware. `--min-capability` turns the expectation into a deliberate skip, `--server-arg KEY=VALUE` and `--server-env KEY=VALUE` pass extra server flags and environment variables through.

`SKIPPED` is reported only for genuinely missing prerequisites: no CUDA device, a missing server CLI, an unsupported interpreter, a capability below an explicitly requested `--min-capability`, or a missing input file.

The notebook runs TTS first, since TTS is this project's purpose, and STT second. It **disconnects the runtime unconditionally** when finished, so Colab quota is not spent idling. Everything needed to review a run is therefore written to Drive as it happens — per-stage metrics, install logs, the server log, transcripts, WAV files, and the full output of every child process in `metrics/session.log` — and stages record failures as statuses instead of raising, because an exception would stop `Run all` before the disconnect.

Each model runs as a separate subprocess in its own virtual environment, so VRAM and host RAM are reclaimed by process exit rather than by a `del` inside the kernel. Separate environments are required and not merely tidy: STT pins `transformers==4.51.0` plus the checkpoint's remote code, while either TTS stack brings its own incompatible `transformers` and `torch`. They are created with `--without-pip --system-site-packages` (Colab's Debian build has no `ensurepip`, so plain `python -m venv` fails) and inherit Colab's preinstalled `torch` through a `.pth` file, while their own pinned packages still shadow it.

Inputs and outputs live in `MyDrive/higgs-benchmark`; model weights are cached on the VM's local disk, since 15 GB through the Drive FUSE mount would be far slower and would consume Drive quota. A missing `stt_ru.wav`, TTS text, or cloning reference yields `SKIPPED` rather than synthetic input.

## Voice Cloning & Audiobook Production Guides

- **Voice Cloning Guide & Reference Texts**: [`docs/guides/voice_cloning_guide.md`](docs/guides/voice_cloning_guide.md) — rules for recording clean 7–12s audio samples, phonetically balanced text templates, and voice profile export.
- **Audiobook Production Guide**: [`docs/guides/audiobook_guide.md`](docs/guides/audiobook_guide.md) — multi-character dialogue switching, screenplay JSON format, natural pause insertion, and complete chapter audio stitching.

## Status

### Apple Silicon M1 (16 GB unified memory, macOS 14.6.1, native `arm64`, Python 3.11.7)

```text
TTS via MLX-Audio:     PASSED (bosonai/higgs-tts-3-4b)
Russian speech:        PASSED (natural Cyrillic Russian generated)
Voice cloning:         PASSED (60s reference cloned into 25.2s Russian speech, RTF 822.09)
Voice cloning (7.4s):  NOT MEASURED (aborted after 30+ min; the host became unresponsive)
Control tags:          PASSED
STT MPS FP16:          PASSED (complete inference on Metal GPU)
STT CPU fallback:      NOT NEEDED (MPS FP16 succeeded end-to-end)
Russian transcription: PASSED (accurate Cyrillic and Vaishnava terminology)
```

### Google Colab, Tesla T4 (15 GB, compute 7.5, Python 3.13, driver 580.82.07 / CUDA 13.0)

```text
TTS via vLLM-Omni:     FAILED (server runs, but every sample of the output is -32768;
                               constant full-scale DC, not speech)
TTS via SGLang-Omni:   FAILED (KeyError: 'sm_75' in flashinfer CUTLASS-DSL RMSNorm)
STT CUDA FP16:         PASSED (RTF 0.18-0.21 across four runs)
Russian transcription: PASSED (coherent Cyrillic; WER not measured, no matching reference)
```

**No usable Russian speech has been produced on a T4.** With [`configs/higgs_multimodal_qwen3_turing.yaml`](configs/higgs_multimodal_qwen3_turing.yaml) the vLLM path completes startup and answers every request with a correctly sized, correctly headed 24 kHz mono WAV — whose payload is the two bytes `00 80` repeated end to end, i.e. a constant `-32768`. Basic, control-tag and voice-cloning requests all produce this. Timings were recorded and are reported below as diagnostics only; an RTF for the production of a constant is not a synthesis measurement.

Suspected cause, not yet isolated: `--dtype float16`, which the runner must force because vLLM refuses the checkpoint's declared bfloat16 below compute 8.0, and possibly the `TRITON_ATTN` substitution the same profile makes. Distinguishing "Turing configuration" from "this profile" needs an Ampere-class run, where neither applies. Tracked in [#48](https://github.com/vedmalex/higgs-local-test/issues/48).

Neither Boson nor either serving stack claims Turing support.

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
| STT (run 1) | CUDA T4 (FP16) | 81.23s | 10.71s | 60.00s | 0.178 | 5.97 GB | 6.25 GB |
| STT (run 2) | CUDA T4 (FP16) | 76.94s | 11.69s | 60.00s | 0.195 | 5.97 GB | 6.25 GB |
| STT (run 3) | CUDA T4 (FP16) | 89.34s | 12.40s | 60.00s | 0.207 | 5.97 GB | 6.25 GB |
| STT (run 4) | CUDA T4 (FP16) | 92.30s | 11.93s | 60.00s | 0.199 | 5.97 GB | 6.25 GB |
| TTS all modes | CUDA T4 (FP16, vLLM) | 205.30s | see note | — | not measurable | 11.18 GB | 2.97 GB |

**TTS has produced no usable audio on a T4.** `vllm serve --omni` completes startup with `configs/higgs_multimodal_qwen3_turing.yaml` (7.61 GiB weights, 1.25 GiB KV cache, 11.18 GB device peak of 14.56 GB) and answers every request 200 with a correctly headed 24 kHz mono WAV. Every sample in every file is `-32768`. Verified byte-wise:

```
00000020: 0200 1000 6461 7461 809b 1100 0080 0080  ....data........
00000030: 0080 0080 0080 0080 0080 0080 0080 0080  ................
```

Measured on all three outputs: peak 32768, RMS 32768.0, 100.00% of samples at full scale, one distinct value across the file. For contrast, the repository's own speech reference measures peak 16338, RMS 1989.4, 0.00% at full scale.

Request timings for the record — **diagnostics, not synthesis measurements**: basic 61.05 s for 24.04 s of output, control tags 25.40 s for 13.68 s, cloning 43.47 s for 18.60 s. `src/tts_cuda.py` now inspects the samples and reports such a job `FAILED`; the earlier `PASSED` results and their RTF figures were wrong and have been removed from this table.

Four STT runs of the same input give RTF 0.178, 0.195, 0.199 and 0.207, so read it as approximately 0.18–0.21 — roughly 7–8× faster than the M1 MPS run (1.40) rather than a single exact ratio. WER is not stated: the recording used was not the repository fixture and no matching reference transcript was supplied, so no WER was measured — a comparison against the fixture text would have produced a number describing nothing.

The `sglang` backend fails earlier and differently — a **documented reproducible failure on T4**: it installs and loads the model, then dies during CUDA graph capture with `KeyError: 'sm_75'` in flashinfer's CUTLASS-DSL RMSNorm, which has no Turing entry. `FLASHINFER_USE_CUDA_NORM=1` is applied below compute 8.0 as flashinfer's documented CUDA-JIT fallback for that path, but whether it carries a T4 through the rest of startup is untested — the vLLM backend made it unnecessary to find out.

## Expected honest outcomes

Success or partial success are both valid. A load-only result is not a pass. Keep TTS and STT in separate processes and never try to keep both models resident.
