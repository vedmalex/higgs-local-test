# Higgs Local Test: Development Guidelines

## Project mission

Maintain a reproducible, local-only compatibility benchmark for Higgs Audio v3 on macOS, Apple Silicon M1, and 16 GB unified memory. The project measures Russian TTS and STT quality and performance; it is an experimental test stand, not a production service.

An honest partial result is valid. Never turn a load-only success into an inference pass, and never hide unsupported Russian behavior with post-processing or another model.

## Supported environment

- macOS on native `arm64` Apple Silicon.
- Python 3.11 is preferred; Python 3.12 may be supported after verification.
- CUDA, Docker inference, Rosetta Python, and cloud APIs are out of scope.
- Model weights belong in the Hugging Face cache and must never be committed.

Before changing dependency setup or model IDs, verify the current first-party MLX-Audio source, Boson model card, bundled checkpoint code, and package metadata. Record source links and pinned revisions in `docs/research/` and update `README.md` when a decision changes.

## Architecture

- `scripts/bootstrap.sh`: validates the host and creates isolated environments without downloading both models.
- `scripts/test_tts.sh`: runs basic TTS, control-tag TTS, and optional cloning in separate Python processes.
- `scripts/test_stt.sh`: normalizes input, tries complete MPS inference, and starts CPU fallback in a fresh process after failure.
- `src/tts_test.py`: MLX model loading, WAV generation, timing, RTF, and memory metrics.
- `src/stt_test.py`: one device/dtype attempt per process, transcript output, timing, RTF, WER, versions, and full failures.
- `src/stt_helper.py`: resolves checkpoint-owned helper code at a pinned Hugging Face revision.
- `src/benchmark.py`: sequential orchestration only; it must not retain either model.
- `samples/`: committed text fixtures and instructions; user audio is ignored.
- `output/` and `logs/`: generated artifacts; only `.gitkeep` files are tracked.

Keep TTS and STT environments and processes isolated. Do not introduce a shared daemon or module that holds both models in memory.

## Model constraints

- TTS must remain Higgs TTS 3 through MLX-Audio. Do not silently substitute another speech model.
- STT must remain `bosonai/higgs-audio-v3-stt`; never switch to the 8B checkpoint on M1/16 GB.
- Do not install an unrelated package named like `boson_multimodal`. Prefer the helper code bundled with the pinned STT checkpoint.
- Do not request BF16 on M1 unless complete inference has been demonstrated for the exact code and package versions.
- MPS support means successful end-to-end inference, not merely `torch.backends.mps.is_available()` or model loading.

## Memory and data safety

- TTS and STT benchmarks must run sequentially in separate processes.
- A failed MPS process must terminate before CPU fallback starts.
- Do not start other large models, change macOS swap settings, or add heavyweight monitoring dependencies.
- Never send sample audio, transcripts, or generated output to external APIs.
- Voice cloning must use only user-provided authorized audio and must report `SKIPPED` when either reference file is absent.

## Development rules

1. Run every command from the repository root.
2. Preserve native architecture checks and clear Rosetta errors.
3. Pin remote custom code to an immutable revision and log model/package versions.
4. Preserve complete exception type, operation context, and traceback for failed MPS and CPU attempts.
5. Keep official Higgs control tags synchronized with the current `PROMPTING.md`; never invent tags.
6. Keep Russian transcript output untouched except for the checkpoint's documented deterministic processing. Do not transliterate or repair it with an LLM.
7. Keep generated models, audio samples, outputs, logs, virtual environments, and caches out of Git.
8. Update the README status and benchmark table only from real recorded runs on the stated hardware.

## Validation

For changes that do not require downloading models, run:

```bash
bash -n scripts/*.sh
python3 -m py_compile src/*.py
git diff --check
```

On a native arm64 host, also run:

```bash
make info
make setup
make tts
make stt
```

`make benchmark` must execute TTS and STT sequentially. Validate produced WAV files, transcript encoding, JSON metrics, `/usr/bin/time -l` memory output, and logs rather than relying only on exit status.

## Completion criteria

A change is complete only when documentation matches the implemented commands, static checks pass, and relevant runtime tests have either succeeded or produced a clearly documented reproducible failure. Keep TTS status, STT status, Russian quality, device, dtype, RTF, WER, and peak memory as separate claims.
