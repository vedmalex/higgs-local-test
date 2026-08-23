# Qwen3-ASR: current APIs, licensing, and local/T4 implications (issue #52 extension)

Research snapshot: 2026-08-24. Primary sources consulted: Hugging Face model
cards, the `QwenLM/Qwen3-ASR` GitHub repository/README, and direct inspection
of the `mlx-audio` 0.5.0 package already installed in this repository's
`.venv-tts` (its source is the authority for the Apple Silicon / MLX claims
below — a package-metadata cross-check, not a web search, since MLX support
for Qwen3-ASR is recent enough that general web search missed it entirely).
Nothing about Russian WER/CER was found published anywhere; that gap is
recorded as "unknown", not guessed.

## Model identity, license, languages

| Model | HF ID | Params | Release | License |
| --- | --- | --- | --- | --- |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | 1.7B | 2026-01-28 | Apache-2.0 |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` | 0.6B | 2026-01-28 | Apache-2.0 |
| Qwen3-ASR (native transformers) | `Qwen/Qwen3-ASR-1.7B-hf` | 1.7B | 2026-06-26 | Apache-2.0 |
| Qwen3-ForcedAligner (timestamps) | `Qwen/Qwen3-ForcedAligner-0.6B` | 0.6B | 2026-01-28 | Apache-2.0 |

Source: [`QwenLM/Qwen3-ASR` README](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/README.md), model cards on Hugging Face (`Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B-hf`).

**Do not confuse with `Qwen/Qwen-Audio` / `Qwen-Audio-Chat` (2023)** — those are
general-purpose audio-language models covering many tasks including ASR as
one of several, documented only for Chinese/English, and are **not** the
specialized ASR line this issue asks about. `Qwen3-Omni` is a text-to-audio /
any-to-any model, also not an ASR model. Neither should be treated as "a Qwen
ASR model" per the issue's explicit instruction not to infer ASR capability
from a generic audio-language model.

**Languages**: 30 languages documented, explicitly including **Russian**.
`Qwen3-ForcedAligner-0.6B` supports timestamps for a narrower 11-language set
that also explicitly includes Russian (measured alignment error 40.2 ms on
its own benchmark, versus 200.7 ms for the NFA baseline it compares against).

Source: [`QwenLM/Qwen3-ASR` README](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/README.md).

## Capabilities

- **Long-form**: supported; `max_new_tokens` is adjustable for longer audio.
- **Language detection**: automatic, or can be forced.
- **Timestamps**: not from the ASR model itself — `Qwen3-ForcedAligner-0.6B`
  is a separate, non-autoregressive alignment model for word/character-level
  timestamps.
- **Prompt/context/hotwords**: a `prompt` argument to
  `apply_transcription_request()` documented for biasing toward
  domain vocabulary/names — directly relevant to this issue's Russian
  specialist-terminology test (Шримад-Бхагаватам, Чайтанья Махапрабху, etc.).
  Source: [`Qwen/Qwen3-ASR-1.7B-hf` model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf).
- **Code-switching**: not explicitly documented either way in the README —
  recorded as unknown, not assumed.
- **Speech understanding beyond transcription**: not documented as a Qwen3-ASR
  capability in the README found — this line item stays `unknown`, per the
  issue's explicit instruction not to assume it.
- **WER/CER benchmarks**: published for English, Chinese, and other dialects;
  **no Russian WER/CER was found published** by Qwen or on the HF Open ASR
  Leaderboard snapshot checked (2026-06-26) for either Qwen3-ASR or
  Whisper Large v3. This project's own Russian specialist-terminology test
  (below) is therefore the only source of a Russian WER/CER number until one
  is run.

## Serving paths

- **transformers**: native pipeline support from `transformers>=5.13.0`
  (`Qwen/Qwen3-ASR-1.7B-hf`), with `torch.compile` support (~2.4x speedup
  reported on A100, batch=4 — not verified on T4 or M1 by this research pass).
- **vLLM**: officially supported and recommended for throughput; suggested
  `gpu_memory_utilization=0.7-0.8` with FlashAttention-2. FlashAttention-2 is
  CUDA/Ampere+-oriented — whether it is available at all on a T4 (compute 7.5)
  is the same open question already tracked for Higgs and Qwen3-TTS, and is
  not resolved by this research pass.
- No explicit VRAM figure is published; the vLLM guidance above is the only
  official sizing hint found.

Source: [`Qwen/Qwen3-ASR-1.7B` model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [`Qwen/Qwen3-ASR-1.7B-hf` model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf).

## Apple Silicon / MLX — found by direct package inspection, not web search

The general web search this research pass ran did **not** surface an MLX port
of Qwen3-ASR. Direct inspection of this repository's own `.venv-tts`
(`mlx-audio==0.5.0`, the same package this project already uses for Higgs TTS
on M1) found one exists and is already installed:

```
.venv-tts/lib/python3.11/site-packages/mlx_audio/stt/models/qwen3_asr/
    qwen3_asr.py, config.py, qwen3_forced_aligner.py
```

`config.py`'s `AudioEncoderConfig`/`TextConfig` describe exactly Qwen3-ASR's
documented architecture (a Whisper-style mel-spectrogram audio encoder feeding
a Qwen3 text decoder), and `mlx-audio`'s own package metadata (`METADATA`)
names ready-made MLX weight conversions:

- `mlx-community/Qwen3-ASR-1.7B-8bit`
- `mlx-community/Qwen3-ASR-0.6B-8bit`

CLI: `python -m mlx_audio.stt.generate --model mlx-community/Qwen3-ASR-0.6B-8bit --audio <file> --output-path <path> --format json` (flags confirmed by reading `mlx_audio/stt/generate.py`'s `argparse` definition directly, not assumed). Equivalent Qwen3-TTS support (`mlx_audio.tts.models.qwen3_tts`, including voice cloning via `model.generate(ref_audio=..., ref_text=...)`) is documented separately in `docs/research/qwen3-tts-notes.md`'s local-M1 addendum.

This is a **community MLX port bundled with an already-used dependency**, not
an official Qwen/Alibaba Apple Silicon release — treat its Russian quality and
performance as unverified until this project's own local test records real
numbers (see `src/stt_qwen_local_test.py`, `docs/README.md`'s Benchmark
section).

## Comparison target (per issue's requested table)

| Capability | Higgs STT 3 | Qwen3-ASR | Whisper Large v3 |
| --- | --- | --- | --- |
| Open/local weights | yes | yes | yes |
| Russian | yes (empirical; model card claims English only) | yes (explicit, documented) | yes (explicit, 99 languages) |
| Russian WER/CER | measured on this project's fixture (see README) | unknown — not published upstream; this project's local test is the only source once run | unknown — not published upstream |
| Long-form | current test uses ~60s clips | yes (documented) | yes |
| Timestamps | not tested here | via separate `Qwen3-ForcedAligner-0.6B` model, Russian included | yes, native |
| Language detection | model-dependent, not the focus of this project's test | yes, automatic or forced | yes |
| Code switching | unknown | unknown — not documented | yes (documented) |
| Context/prompt | unknown | yes — `prompt` arg for vocabulary/name biasing (relevant to specialist terminology) | limited prompt |
| Hotwords/terminology | unknown | via the same `prompt` mechanism, unverified for Russian specialist terms until tested | limited |
| Speech understanding beyond ASR | claimed by Higgs, not yet verified by this project | unknown — not documented as a Qwen3-ASR capability | no, primarily ASR |
| T4 16 GB | Higgs STT already runs on T4 in this project (PASSED, RTF ~0.18-0.21) | not yet run on T4 by this project; vLLM/FlashAttention-2 path on Turing is unconfirmed | yes, widely used on T4 |
| M1 16 GB | Higgs STT PASSED on M1 (MPS FP16) | not yet run by this project; MLX port exists (`mlx-community/Qwen3-ASR-0.6B-8bit`), unverified quality/RTF | yes, via `mlx-audio`/`whisper.cpp` and similar |
| License | document exact checkpoint (Boson STT card) | Apache-2.0 | MIT |

Cells stay `unknown` where no primary source states an answer, per the issue's
instruction not to fill them from assumption.

## Выводы для реализации

1. **Local M1 test**: use `mlx-community/Qwen3-ASR-0.6B-8bit` via
   `mlx_audio.stt.generate` (already installed, no new dependency) to
   transcribe `samples/stt_ru.wav`, reusing this project's existing WER
   mechanism from `src/stt_test.py` rather than inventing a second one. Report
   real RTF/WER, never a placeholder.
2. **Russian specialist-terminology benchmark**: build the small fixture the
   issue specifies (Шримад-Бхагаватам, Чайтанья Махапрабху, Вриндаван,
   Радхарани, Гопала Бхатта Госвами, Радха-Раман, Бхакти-расамрита-синдху) as
   a committed sample text + a human reference transcript, and run it through
   both Qwen3-ASR and Higgs STT (and Whisper if/when added) without
   normalizing away terminology errors, per the issue's explicit rule.
3. **Colab/Kaggle T4 path**: not attempted by this research pass. If pursued,
   reuse this project's GPU-detection-first pattern and the same
   PASSED/FAILED/SKIPPED discipline as `src/tts_qwen_cuda.py` — do not assume
   FlashAttention-2 or vLLM behave the same on a T4 as on Ampere+ until
   measured, exactly as Qwen3-TTS's T4 run (above) turned an "unconfirmed"
   into a measured, different-than-expected failure.
4. **Round-trip test**: out of scope for this pass; requires both a working
   Qwen3-TTS path and a working Qwen3-ASR path first.
