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
make sentiment-survey # local blind-listening survey app (issue #57 sentiment gate)
```

`make download-models` (`scripts/download_models.sh`) retries each model through a flaky
network on its own (disables the Xet accelerated-download backend, which has been observed
to stall indefinitely on some VPNs; retries transient timeouts with backoff). If the local
network is unusable for a multi-gigabyte download even with that, prefetch on Colab instead —
its network doesn't share the same problem — and pull the result down as an archive:
[`notebooks/model_prefetch_to_drive.ipynb`](notebooks/model_prefetch_to_drive.ipynb)
(<a href="https://colab.research.google.com/github/vedmalex/higgs-local-test/blob/main/notebooks/model_prefetch_to_drive.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>)
downloads every model this project needs, packs each into a `.tar` preserving the Hugging
Face cache's internal blob/snapshot symlink layout, and uploads it to
`MyDrive/higgs-benchmark/model-cache/`. Locally: `tar -xf <name>.tar -C
~/.cache/huggingface/hub/` for each downloaded archive, and `huggingface_hub` recognizes it as
already cached.

The 60-second reference behind the recorded cloning result was a copy of `samples/stt_ru.wav`; `samples/reference.wav` now holds a 7.4-second clip instead, so reproducing that figure means copying `samples/stt_ru.wav` back over it. The 60-second pair is also preserved on Drive as `reference_60s.wav` / `reference_60s.txt`.

Add `samples/stt_ru.wav` before STT. It is normalized with system `ffmpeg` to mono 16 kHz. Optional voice cloning needs both `samples/reference.wav` and its exact transcript in `samples/reference.txt` (can be copied directly from `samples/stt_ru.wav` and `output/stt_ru.txt`); otherwise it reports `SKIPPED`. Note on cloning latency: reference audio is tokenized at 25 frames/sec into the prompt KV cache (e.g. 5–15s clips are ~125–375 tokens for faster RTF, while a 60s clip introduces ~1500 prompt audio tokens).

The STT runner does not equate loading with support: it attempts complete MPS inference. Any MPS exception and traceback are written to `logs/stt.log`; CPU FP32 is then tested in a fresh process so the failed model cannot remain resident. No BF16 is requested on M1, although the current remote model code contains a forced BF16 feature cast that may itself prove incompatible. WER uses the fixed Russian sample transcript. For 1–3 minute behavior, record the same kind of continuous Russian speech and replace `samples/stt_ru.wav`; interpret WER against a matching reference (the built-in WER is only valid for the prescribed text).

## Google Colab Benchmark & Playground

For comparing Apple Silicon M1 performance against server GPUs:

- Notebook: [`notebooks/higgs_colab_benchmark.ipynb`](notebooks/higgs_colab_benchmark.ipynb)
- Open directly in Colab: <a href="https://colab.research.google.com/github/vedmalex/higgs-local-test/blob/main/notebooks/higgs_colab_benchmark.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

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

### Qwen3-TTS — second, independent TTS backend (#52)

Qwen3-TTS is added as a **second, independent** TTS backend, never a replacement for Higgs: it serves two purposes — a diagnostic control on the same Colab T4 that reproducibly fails to produce real speech with Higgs ([#48](https://github.com/vedmalex/higgs-local-test/issues/48)), and a separate audiobook-production candidate to evaluate on its own merits. `src/tts_qwen_cuda.py` drives it through the same `vllm serve --omni` stack already used for Higgs, so the notebook installs, runs, and tears it down with the same subprocess-per-stage pattern. Results are written to a separate metrics file per model variant and are never merged into or reported as a Higgs result.

Model variants (exact upstream IDs; sources and full research trail in [`docs/research/qwen3-tts-notes.md`](docs/research/qwen3-tts-notes.md)):

| Variant | 0.6B (Phase 1, T4 diagnostic) | 1.7B (Phase 2, audiobook capability) | Supports |
| --- | --- | --- | --- |
| Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning from a ~3s reference |
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 predefined timbres + free-text `instructions` (style/emotion) |
| VoiceDesign | not published | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | narrator voice from a natural-language description |

One server serves one variant at a time — a request's `task_type` only works against the checkpoint trained for it, so `--model-variant` selects which of `qwen_tts_basic` / `qwen_tts_clone` / `qwen_tts_style` / `qwen_tts_voicedesign` the run can actually attempt; the rest are reported `SKIPPED` with the loaded variant named in the reason, never silently attempted against the wrong checkpoint. Every produced WAV goes through the exact same anti-false-positive waveform checks as Higgs (`src/tts_cuda_common.py`, shared by both runners after #52's refactor) — an `HTTP 200` and a well-formed WAV are not `PASSED` on their own.

All three variants' model cards claim the same 10 languages, including **Russian**, as a first-party, documented claim (unlike Higgs STT, where Russian is empirical). Every published Qwen3-TTS model card checked so far uses the **Apache-2.0** license — no research/non-commercial restriction, unlike Higgs TTS 3 — which is the direct answer to this project's "licensing implications for generated audiobooks" question, pending the actual quality/RTF numbers from a real run.

The FlashInfer/attention-backend concern above did not materialize: `TRITON_ATTN` is selected correctly on a T4 with no override needed. A *different* failure did — a hardcoded `torch.bfloat16` in vllm-omni's Qwen3-TTS talker (`self._embedding_dtype`) ignored the engine's actual configured dtype, crashing every request with a `Half`/`BFloat16` mismatch. **This is now fixed and confirmed on real hardware** (2026-08-24): a one-line source fix, upstreamed as [vllm-project/vllm-omni#6545](https://github.com/vllm-project/vllm-omni/pull/6545), installable today via this notebook's `QWEN_OMNI_SOURCE="fork"` (from `github.com/vedmalex/vllm-omni@fix/qwen3-tts-embedding-dtype`) while the upstream PR is reviewed. Full root-cause and verification detail: `docs/research/qwen3-tts-notes.md`.

**Status: Phase 1 (0.6B Base + CustomVoice) PASSED on a real Colab Tesla T4** with the fork fix — see the comparison matrix and Benchmark section below for real numbers. Phase 2 (1.7B) has not been run yet.

## Kaggle Notebooks

The same CUDA benchmark (Higgs and Qwen3-TTS, plus STT) can be tried outside Colab on [Kaggle Notebooks](https://www.kaggle.com/code). Kaggle's GPU type and availability are **dynamic and not guaranteed** — it does not promise a T4 or any particular GPU for a given session, so the notebook's GPU-detection cell (name, VRAM, compute capability, printed prominently) is exactly as necessary there as on Colab; do not assume the T4-specific code paths in `configs/higgs_multimodal_qwen3_turing.yaml` or the Turing dtype/attention-backend forcing apply until that cell confirms a T4.

The notebook and `src/*.py` runners are the shared implementation for both platforms — nothing here is Colab-specific beyond the `google.colab.drive` mount (guarded by `USE_DRIVE`) and the Colab-badge link above. To run on Kaggle: upload or import [`notebooks/higgs_colab_benchmark.ipynb`](notebooks/higgs_colab_benchmark.ipynb) as a Kaggle Notebook, enable a GPU accelerator in the session settings, set `USE_DRIVE = False` (Kaggle has its own persistent `/kaggle/working` output, not a Drive mount) or adapt `WORKSPACE` to a Kaggle-writable path, and run all cells. If Kaggle's filesystem layout needs a small adapter beyond that, keep it minimal and documented here rather than maintaining a second notebook.

## Mojo/MAX feasibility spike (#57) — CLOSED, question answered

Separate research track, exploring whether [Mojo/MAX](https://www.modular.com/) can host a ported Higgs/Qwen speech pipeline as a portable, precision-controlled execution layer across Apple Silicon and NVIDIA T4 — not a replacement for the vLLM-Omni stack above until evidence says otherwise. Full plan, hardware/numerical M0 probe results, and the staged M0–M6 roadmap: [`docs/research/mojo-max/m0-results.md`](docs/research/mojo-max/m0-results.md) and issue [#57](https://github.com/vedmalex/higgs-local-test/issues/57).

**Verdict (2026-08-25): the track is closed as ANSWERED, not abandoned.** Two grounds, both
measured: (1) the decoder is already a native MLX implementation running on the Apple GPU
(`mlx_audio/codec/models/higgs_audio/dac.py` — `AcousticDecoder:144`, `AcousticDecoderBlock:73`),
so there was nothing left to port; (2) the vocoder is not the bottleneck — `codec.decode` is
**1.78% (short) / 3.76% (long)** of wall time against the AR loop's **94.3–95.5%**, so a perfect
zero-cost vocoder would move the long case only from 82.87 s to ≈79.75 s. That is under the
pre-declared "<15% → don't touch the vocoder" threshold. MAX additionally has to place
`ConvTranspose1d` on the CPU inside a GPU graph (upstream, both Metal and CUDA); MLX does not.
If the vocoder is ever rewritten, it goes on MLX, not MAX. M0–M3's artifacts stay in active use
(`m3_block_reference.py`'s FP64 oracle, `m3_divergence.py:149`'s `compare()`).

**This closure is at the pipeline's *current* AR-loop/vocoder cost ratio, not permanent.** It
reopens if **either**: (a) a real cross-platform requirement returns **and** the upstream MAX fixes
land; **or** (b) **the AR loop is sped up by ≥4x** relative to the measured baseline above — at
that point `codec_decode`'s fixed ~3.1 s cost crosses back above the plan's own pre-declared 15%
closure threshold (RTF ≈ 1.07), and the vocoder track must be reopened on its own merits. Condition
(b) does not require condition (a), and it is squarely in the range M4's batching/quantization
tracks are targeting — an independent audit (2026-08-25) added this condition explicitly rather
than leaving it implicit. Full reasoning, reuse inventory, and both reopen conditions:
[`docs/research/mojo-max/m4-conclusion.md`](docs/research/mojo-max/m4-conclusion.md).

- Notebook (Colab T4 only, deliberately minimal — no TTS/STT/Qwen stack install): [`notebooks/mojo_max_m0_t4.ipynb`](notebooks/mojo_max_m0_t4.ipynb)
- Open directly in Colab: <a href="https://colab.research.google.com/github/vedmalex/higgs-local-test/blob/main/notebooks/mojo_max_m0_t4.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

Status: M0 **PASSED** on both Apple M1 and Colab T4 GPUs. M1 (vLLM-Omni → MAX responsibility map) is done. M2 (correctness prototypes) on M1: Snake1d and Conv1d **PASSED**; ConvTranspose1d **PASSED on CPU, hard-crashed on Metal GPU** (`cudnnCreate` symbol not found — attempting an NVIDIA-only library load on Apple Silicon).

- M2 T4 runner (same three prototypes, unchanged): [`notebooks/mojo_max_m2_t4.ipynb`](notebooks/mojo_max_m2_t4.ipynb) — <a href="https://colab.research.google.com/github/vedmalex/higgs-local-test/blob/main/notebooks/mojo_max_m2_t4.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**NVIDIA Turing / `sm_75` (Tesla T4, including the default Colab GPU) is NOT supported for this
project's MAX graphs on MAX 26.5.0** — the full decoder-block graph fatally aborts
(`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16`). This is an
upstream Modular defect (the `ldmatrix` intrinsic is not configured for Turing GPUs in this MAX
version), not a defect in this project's code, and cannot be fixed on our side. Full evidence,
upstream references, and status: [`docs/research/mojo-max/m3-t4-blocked-results.md`](docs/research/mojo-max/m3-t4-blocked-results.md).

## Voice Cloning & Audiobook Production Guides

- **Voice Cloning Guide & Reference Texts**: [`docs/guides/voice_cloning_guide.md`](docs/guides/voice_cloning_guide.md) — rules for recording clean 7–12s audio samples, phonetically balanced text templates, and voice profile export.
- **Audiobook Production Guide**: [`docs/guides/audiobook_guide.md`](docs/guides/audiobook_guide.md) — the `[{"speaker": ..., "text": ...}]` screenplay JSON DSL, run through `src/audiobook.py --screenplay-file ...` (the same sentence-splitting/chunking/tag-reopening/resumable-manifest engine used for plain text, not a separate pipeline), incremental per-line regeneration via a speaker+text content hash, pause insertion, and chapter stitching. **A single voice reference is now wired into every segment** (`--voice-name`/`--ref-audio`, Refs #57) — without one, a full chapter's pitch median drifts 83.9–203.4 Hz segment to segment (measured on `output/chapter-114-e0/chapter.wav`, 70 segments); with one, an 8-sentence spread test dropped from stdev 43.6 Hz to 6.1 Hz (unbatched) / 8.2 Hz (`--batch-size 4`, confirmed compatible with batching) — see `docs/research/audiobook/voice-clone-consistency-results.md` for the full measurement, including the RTF cost (baseline 3.87 → reference+batch4 3.85, reference alone 5.58) and the caveat that per-`speaker` (multi-character) voice selection is still **not** wired — one run still uses one voice for every speaker.
- **Sentiment/Tag Blind-Listening Survey**: [`docs/guides/sentiment_survey_guide.md`](docs/guides/sentiment_survey_guide.md) — `make sentiment-survey` starts a local stdlib-only web app that plays clip pairs blind (tag identity hidden until you answer), records every answer to disk immediately, and is resumable across sessions. Auto-discovers task sets from `output/m4_tag_catalog/`, `output/m4_tags/`, `output/m4t0_*`, and `output/m4_boundary_check/`, prioritized: unheard sfx/env first, then metrics-disputed tags, then everything else. This is the tool the M4 sentiment-integrity gate (`docs/research/audiobook/m4-plan.md` §2/§3) runs on.

## Status

### Apple Silicon M1 (16 GB unified memory, macOS 26.6.2, native `arm64`, Python 3.12.11)

Re-recorded 2026-08-24 after the host's macOS was upgraded (14.6.1 → 26.6.2, for issue #57's
Mojo/MAX work) and the model cache was fully cleared and re-downloaded — a genuinely fresh
environment, not a re-run of stale state. The prior 60-second cloning reference was replaced by
the current `samples/reference.wav` (7.4s) earlier in this project's history; this run measures
against that 7.4s reference, which is why clone RTF looks nothing like the old 822.09 figure —
different reference length, not a regression fixed.

```text
TTS via MLX-Audio:     PASSED (bosonai/higgs-tts-3-4b)
Russian speech:        PASSED (natural Cyrillic Russian generated)
Voice cloning (7.4s):  PASSED (7.4s reference cloned into 18.6s Russian speech, RTF 7.73)
Control tags:          PASSED
STT MPS FP16:          PASSED (complete inference on Metal GPU, RTF 0.48)
STT CPU fallback:      NOT NEEDED (MPS FP16 succeeded end-to-end)
Russian transcription: PASSED (accurate Cyrillic and Vaishnava terminology)
```

### Qwen3-TTS / Qwen3-ASR on the same M1 host — separate backend, separate result (#52)

```text
Qwen3-TTS basic:       PASSED (Qwen3-TTS-12Hz-0.6B-Base-bf16, RTF 1.71)
Qwen3-TTS clone:       PASSED (same 60s reference as Higgs, RTF 1.58 vs Higgs 822.09)
Qwen3-TTS instruct:    PASSED (1.7B CustomVoice, generate_custom_voice(instruct=...))
Qwen3-ASR:             PASSED (Qwen3-ASR-0.6B-8bit, RTF 0.093)
Russian speech:        PASSED (all nine fixture lines round-trip through ASR)
Sanskrit terminology:  WEAK (proper nouns garbled in both TTS and ASR)
Qwen3-ASR WER:         NOT MEASURED (no reference transcript matches samples/stt_ru.wav)
```

Numbers, waveform validity checks and the round-trip evidence are in the Benchmark section below. This is a Qwen result and stays one; it does not turn any Higgs failure into a Higgs pass.

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

### Higgs vs Qwen3-TTS comparison (#52)

Qwen3-TTS is an additional backend evaluated on its own merits, not a fallback that turns a Higgs failure into a Higgs pass. The two are always reported separately.

| Backend | Model | GPU | Russian | Basic TTS | Clone | Emotion/style | RTF | Peak VRAM | Audio valid | License |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| Higgs | `higgs-tts-3-4b` | T4 | claimed (empirical failure below) | FAILED — constant `-32768` (#48) | blocked by the same common audio failure | blocked | not meaningful for invalid audio | 11.18 GB | **FAIL** | Boson research/non-commercial |
| Qwen3-TTS (`QWEN_OMNI_SOURCE="pypi"`) | `12Hz-0.6B-Base` / `-CustomVoice` | T4 | documented claim (model card) | FAILED — `index_copy_` dtype crash, every request | same crash | same crash | not meaningful for a crash | 10.4 GB | **FAIL** (upstream bug, fixed below) | Apache-2.0 |
| Qwen3-TTS (`QWEN_OMNI_SOURCE="fork"`, [PR #6545](https://github.com/vllm-project/vllm-omni/pull/6545)) | `12Hz-0.6B-Base` / `-CustomVoice` | T4 | **PASSED** — real Cyrillic speech | **PASSED**, RTF 1.234 | **PASSED**, RTF 1.392 | **PASSED** (`instructions`), RTF 1.118 | 1.12–1.39 | 10.2–10.4 GB | **PASS** | Apache-2.0 |

Recorded 2026-08-24 on a real Colab Tesla T4. The PyPI row is the reproducible upstream bug (`vllm-omni==0.26.0`'s Qwen3-TTS talker hardcodes `torch.bfloat16` for per-request embeddings regardless of the engine's configured dtype); the fork row is the one-line fix (`self._embedding_dtype = model_dtype`), verified via this project's own `audio_statistics()`/`audio_defect()` checks against a human-reference waveform — not just an absence of a crash. Full detail: `docs/research/qwen3-tts-notes.md`.

RTF is never published for invalid audio, per this project's own rule — hence the PyPI row's `—` there. Phase 2 (1.7B CustomVoice/VoiceDesign/Base) has not been run yet.

## Benchmark

### Higgs Audio v3, Apple Silicon M1

Measurements recorded from real sequential runs on native Apple Silicon M1 (16 GB unified
memory, macOS 26.6.2, Python 3.12.11), 2026-08-24:

| Test | Device | Load | Processing | Audio | RTF | Peak RAM (RSS) | Peak Footprint |
| ---- | ------ | ---: | ---------: | ----: | --: | -------------: | -------------: |
| TTS basic | MLX | 24.34s | 123.91s | 18.88s | 6.56 | 1.16 GB | 11.37 GB |
| TTS controls | MLX | 14.43s | 126.10s | 14.36s | 8.78 | 3.82 GB | 11.32 GB |
| TTS clone (7.4s ref) | MLX | 10.13s | 143.83s | 18.60s | 7.73 | 3.87 GB | 11.37 GB |
| STT | MPS (FP16) | 15.83s | 28.56s | 60.00s | 0.48 | 3.38 GB | 7.80 GB |

**WER retracted:** the previously published WER 1.5 for Higgs STT is invalid — the transcript above reads as correct Russian for the actual audio content, so the fixture's `REFERENCE` text (`src/stt_test.py:24-26`) simply does not match `samples/stt_ru.wav`; the number measured a mismatched reference, not model accuracy. No valid WER currently exists for any model tested here (Qwen3-ASR 0.6B/1.7B report `"wer": null` for the same reason — no matching reference was supplied, see `logs/qwen_stt.log:19-21` and `logs/qwen17_stt.log`). Verified reference transcripts are a separate, not-yet-started task (M4 plan, Track S, S1); see `docs/research/stt/m4-stt-comparison.md`.

RTF is processing seconds divided by output audio duration (TTS) or input audio duration (STT). Values below 1.0 are faster than real time.

### Qwen3-TTS / Qwen3-ASR — local Apple Silicon M1 test (recorded 2026-08-24)

A **separate** benchmark from the Higgs rows above — different model family, different weights, never merged into or reported as a Higgs result. Motivation: Higgs voice cloning on this machine runs at RTF 822, which is not usable for iterative testing. This path uses the native MLX implementations shipped in `mlx-audio` 0.5.0 (`mlx_audio.tts.models.qwen3_tts`, `mlx_audio.stt.models.qwen3_asr`) with published MLX weights, so no CUDA and no server process is involved.

Same host as the Higgs rows: native `arm64` M1, 16 GB unified memory, macOS 26.6.2,
Python 3.12.11, `mlx` 0.32.1, `mlx-audio` 0.5.0. Re-recorded 2026-08-24 alongside the Higgs
re-run above (fresh OS, fresh model cache).

| Test | Model | Load | Processing | Audio | RTF | Peak RSS | Peak MLX | Status |
| ---- | ----- | ---: | ---------: | ----: | --: | -------: | -------: | ------ |
| TTS basic | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | 6.24s | 37.72s | 24.08s | 1.57 | 1.05 GB | 4.64 GB | PASSED |
| TTS clone (7.4s ref) | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | 4.24s | 25.22s | 18.00s | 1.40 | 2.74 GB | 6.26 GB | PASSED |
| TTS CustomVoice (`instruct`) | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | 4.88s | 41.87s | 24.40s | 1.72 | 4.36 GB | 8.07 GB | PASSED |
| ASR | `mlx-community/Qwen3-ASR-0.6B-8bit` | 4.47s | 5.18s | 60.00s | 0.086 | 1.40 GB | 2.18 GB | PASSED |

Peak RSS is the worker process's `ru_maxrss`; Peak MLX is `mx.get_peak_memory()`, which counts MLX's unified-memory allocations and is therefore the larger and more relevant ceiling on a 16 GB machine.

**Voice cloning is still far faster than the Higgs clone path on the same host and reference** (`samples/reference.wav` + `samples/reference.txt`, 7.4s): RTF 1.40 against Higgs's 7.73 for the same reference — roughly 5.5× faster, a smaller margin than the old 60s-reference comparison (which showed ~520×) because Higgs's clone cost itself scales with reference length while Qwen's stayed comparably cheap; this is the practical reason the Qwen path exists here regardless of which reference length is used.

Generated audio was checked for the same false-positive failure mode the CUDA runners guard against — a well-formed WAV that is not speech. All three outputs are real waveforms with speech-like statistics, next to the repository's own human reference for scale:

| File | Peak | RMS | % at full scale | Distinct values |
| ---- | ---: | --: | --------------: | --------------: |
| `output/qwen_tts_ru_basic.wav` | 23278 | 1573.6 | 0.00% | 16948 |
| `output/qwen_tts_ru_clone.wav` | 13502 | 1683.7 | 0.00% | 15215 |
| `output/qwen_tts_ru_custom.wav` | 14860 | 1920.0 | 0.00% | 16900 |
| `samples/reference.wav` (human) | 16338 | 1989.4 | 0.00% | 14061 |

The cloning and CustomVoice paths return the whole utterance as one segment rather than splitting on newlines like the basic path, so coverage was verified rather than assumed: each generated WAV was transcribed back with the same Qwen3-ASR model and all nine lines of `samples/tts_ru.txt` are present in all three. Russian words come back accurate; Sanskrit proper nouns are the weak spot in both directions (`Шри Чайтанья Махапрабху` round-trips as `Шричая таня маха правку` in the clone output).

**Stress control has no *documented* mechanism in either backend; the experimental workaround failure below is Qwen3-TTS-only, not Higgs.** (Correction, Refs #57: an earlier revision of this paragraph read as if both backends had been tried and both got worse — they had not; only Qwen3-TTS was ever exercised, and the artifacts (`output/qwen_tts_ru_basic.wav` etc.) are named accordingly. See `docs/research/audiobook/m4-tag-inventory-results.md` for the first actual measurement of stress notations against Higgs.) Neither Higgs nor Qwen3-TTS *documents* a way to mark word stress; it is an open, unresolved request in the Qwen3-TTS community (`QwenLM/Qwen3-TTS` discussions #53 and #185). Against **Qwen3-TTS only** (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`), inserting a Unicode combining acute accent after the stressed vowel, or a literal `+` before it, were both tried against the fixture text and reverted — the model vocalizes the mark itself (e.g. `Вриндава́н` round-tripped as "Вриндава**Юн**"; `+Это` came back as "Plus, Эйхолог") rather than using it to shift stress. Doubling the stressed vowel letter (no special symbols) did not introduce that garbling on Qwen, but an ASR round-trip cannot confirm the stress itself moved — that needs an actual listen, so it is recorded as inconclusive, not as a fix. Full experiment log: `docs/research/qwen3-tts-notes.md`. **Update (2026-08-25, Refs #57): Higgs's stress-notation behavior has now been tested and the project owner has confirmed a working notation by ear** — an apostrophe placed right after the stressed vowel (`за'мок`) is neither vocalized nor garbled and gives correct stress on the 3 homograph pairs the owner listened to; the doubled-vowel/`+`/capital-letter notations do not reliably work (doubling only lengthens the vowel sound; `+` and capitals are unreliable or read aloud); the unmarked default is confirmed unsafe (the model picks a stress on its own, not always correctly). **For a Russian book this makes stress marking a correctness requirement, not an optional polish step** — see `docs/research/audiobook/m4-tag-inventory-results.md` §3.6-§3.8 for the verbatim verdict and `docs/guides/audiobook_guide.md` §5b for the notation table, the apostrophe-vs-proper-noun disambiguation rule implemented in `src/audiobook.py`, and the open question of automating stress placement across a whole book. This finding is scoped to the 3 homograph pairs/6 notations actually tested, not a general claim about Russian stress.

**ASR WER is not reported, and this is not an omission.** `samples/stt_ru.wav` is a real lecture recording, not the scripted fixture text that `REFERENCE` in `src/stt_test.py` holds, and no matching reference transcript exists in the repository — exactly the situation already noted for the T4 STT runs below. Measuring against the fixture anyway produced WER 2.73, a number that describes nothing; the recorded run therefore passes no `--reference` and reports `wer: null` with the reason. Supply `--reference <path>` once a matching transcript exists. Qualitatively, the transcript is usable Russian with garbled Sanskrit terminology (`Хари Крешна`, `вошнавы`, `брамана`, and Sanskrit verse quotation reduced to phonetic mush) and is written out as returned — no transliteration, no LLM repair.

Emotion and style go through the documented `generate_custom_voice(instruct=...)` argument only; no control tags are invented for this model. The 1.7B CustomVoice checkpoint reports nine speakers: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. The recorded run used `serena` with `instruct="Speak calmly and warmly, like a narrator reading a spiritual book."`.

Commands, each running the model in its own child process so the runner never holds a model resident:

```bash
.venv-tts/bin/python src/tts_qwen_local_test.py --mode basic \
  --output output/qwen_tts_ru_basic.wav --metrics output/qwen_tts_basic.json
.venv-tts/bin/python src/tts_qwen_local_test.py --mode clone \
  --output output/qwen_tts_ru_clone.wav --metrics output/qwen_tts_clone.json
.venv-tts/bin/python src/tts_qwen_local_test.py --mode custom_voice \
  --speaker serena --language russian \
  --instruct "Speak calmly and warmly, like a narrator reading a spiritual book." \
  --output output/qwen_tts_ru_custom.wav --metrics output/qwen_tts_custom.json
.venv-tts/bin/python src/stt_qwen_local_test.py --audio samples/stt_ru.wav \
  --language ru --output output/qwen_stt_ru.txt --metrics output/qwen_stt.json
```

`--mode clone` reports `SKIPPED` when either reference file is missing, and the ASR runner reports `SKIPPED` when the audio file is missing; neither substitutes synthetic input. WER needs `jiwer` in `.venv-tts` (`scripts/bootstrap.sh` installs it); without it the run still passes and says so in `wer_note`.

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

### Qwen3-TTS on Google Colab, Tesla T4 (recorded 2026-08-24)

| Test | Model | vLLM-Omni source | Load | Processing | Audio | RTF | Peak VRAM | Status |
| ---- | ----- | ----------------- | ---: | ---------: | ----: | --: | --------: | ------ |
| `qwen_tts_clone` | `0.6B-Base` | pypi (`vllm-omni==0.26.0`) | 32.6s | — | — | — | 10.4 GB | **FAILED** — `index_copy_` dtype crash |
| `qwen_tts_basic`/`qwen_tts_style` | `0.6B-CustomVoice` | pypi | 58.7s | — | — | — | 10.2 GB | **FAILED** — same crash |
| `qwen_tts_clone` | `0.6B-Base` | fork ([PR #6545](https://github.com/vllm-project/vllm-omni/pull/6545)) | 21.0s | 26.51s | 19.04s | 1.392 | 10.4 GB | **PASSED** |
| `qwen_tts_basic` | `0.6B-CustomVoice` | fork | 58.7s | 29.22s | 23.68s | 1.234 | 10.2 GB | **PASSED** |
| `qwen_tts_style` (`instructions`) | `0.6B-CustomVoice` | fork | 58.7s | 29.86s | 26.72s | 1.118 | 10.2 GB | **PASSED** |

`server_startup_seconds` (not shown as a per-job "Load" figure above, since it's paid once per server, not per job) was 573.7s (`0.6B-Base`) and 490.1s (`0.6B-CustomVoice`) — roughly 9-10 minutes, dominated by `torch.compile`/CUDA-graph-capture warmup across 17 batch sizes vLLM prepares for (1 through 128) that this project's one-request-at-a-time usage never needs. `--enforce-eager` (skips both) is queued as the next experiment — see `docs/research/qwen3-tts-notes.md`.

Waveform validity independently re-checked against a human reference (`samples/reference.wav`: peak 16338, RMS 1989.4, 0.00% full scale):

| Output | Peak | RMS | Full-scale fraction | Distinct values (first 4096) |
| --- | --: | --: | --: | --: |
| `qwen_tts_ru_clone.wav` | 13406 | 1577.3 | 0.0% | 20 |
| `qwen_tts_ru_basic.wav` | 19517 | 2803.6 | 0.0% | 38 |
| `qwen_tts_ru_style.wav` | 24785 | 2964.9 | 0.0% | 7 |

None show the `-32768`-constant signature that made the Higgs T4 result above a FAIL — these are real, varying waveforms.

## Expected honest outcomes

Success or partial success are both valid. A load-only result is not a pass. Keep TTS and STT in separate processes and never try to keep both models resident.
