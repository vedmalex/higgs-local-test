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
make setup       # environments/imports only; does not pre-download both models
make info
make tts         # basic, official control tags, optional clone; fresh process each
make stt         # MPS FP16 inference, then CPU FP32 fallback if needed
make benchmark   # TTS then STT, sequentially
make clean-output
```

Add `samples/stt_ru.wav` before STT. It is normalized with system `ffmpeg` to mono 16 kHz. Optional voice cloning needs both `samples/reference.wav` and its exact transcript in `samples/reference.txt`; otherwise it reports `SKIPPED`.

The STT runner does not equate loading with support: it attempts complete MPS inference. Any MPS exception and traceback are written to `logs/stt.log`; CPU FP32 is then tested in a fresh process so the failed model cannot remain resident. No BF16 is requested on M1, although the current remote model code contains a forced BF16 feature cast that may itself prove incompatible. WER uses the fixed Russian sample transcript. For 1–3 minute behavior, record the same kind of continuous Russian speech and replace `samples/stt_ru.wav`; interpret WER against a matching reference (the built-in WER is only valid for the prescribed text).

## Status

Not yet benchmarked because the invoking terminal is x86_64/Rosetta.

### TTS status

```text
MLX: NOT RUN
TTS: NOT RUN
Russian: NOT RUN
Voice cloning: NOT RUN / SKIPPED when no reference
Control tags: NOT RUN
```

### STT status

The official checkpoint metadata is English. Russian suitability must be decided only from actual output, including Cyrillic preservation and the names Кришна, Радхарани, Вриндаван, Чайтанья, Шримад-Бхагаватам, Гопала Бхатта Госвами, and Радха-Раман. If that fails, report: `Higgs open STT checkpoint is not currently suitable for Russian on this configuration.`

```text
MPS loading/inference: NOT RUN
CPU loading/inference: NOT RUN
Russian transcription: NOT RUN
```

## Benchmark

Populate this table from the JSON printed in `logs/tts_*.log` and `logs/stt.log`. `/usr/bin/time -l` also records native peak resident memory (`maximum resident set size`).

| Test | Device | Load | Processing | Audio | RTF | Peak RAM |
| ---- | ------ | ---: | ---------: | ----: | --: | -------: |
| TTS basic | MLX | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| TTS controls | MLX | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| TTS clone | MLX | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| STT | MPS/CPU | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

RTF is processing seconds divided by output audio duration (TTS) or input audio duration (STT). Values below 1.0 are faster than real time.

## Expected honest outcomes

Success or partial success are both valid. A load-only result is not a pass. Keep TTS and STT in separate processes and never try to keep both models resident.
