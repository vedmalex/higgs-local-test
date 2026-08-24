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

## GitHub Issues are the primary workflow

Substantive project work must be driven through GitHub Issues in `vedmalex/higgs-local-test`. Before starting implementation, dependency research, compatibility investigation, benchmark execution, or a behavioral documentation change:

1. Search open and closed issues for an existing matching task and reuse it instead of creating a duplicate.
2. If none exists, create an issue that states the problem, scope, constraints, acceptance criteria, and expected evidence.
3. Record resolving progress, root cause analyses, model or package revisions, technical decisions, code/dependency fixes, discovered limitations, benchmark commands, results, and blockers directly in comments on that issue as the work progresses.
4. Name the working branch so it can be associated with the issue, and reference the issue number in commits and pull requests.
5. Use `Closes #N` only when the implemented and verified result fully satisfies the issue. Otherwise use `Refs #N` and leave the issue open with its remaining work documented.
6. Before closing an issue, run every relevant validation for its scope and record the results or reproducible failures in the issue.
7. Review every acceptance-criteria checkbox against that evidence and explicitly mark an item complete only when its intended outcome is actually achieved. Never rewrite, weaken, or reinterpret acceptance criteria merely to make a partial result pass.
8. Do not close an issue while the original user-requested outcome or any acceptance criterion remains incomplete. A reproducible failure, documented blocker, completed check, commit, successful model load, or partial benchmark is progress evidence, not completion; keep the issue open and record the remaining work.
9. Close the issue only after the original goal is genuinely complete, validation evidence is recorded, and every acceptance-criteria checkbox is checked against that completed outcome.

GitHub Issues are the canonical surface for task state and development history. Record every intermediate troubleshooting and resolution step directly in issue comments so all technical findings remain fully traceable. README status tables and local logs are supporting evidence, not substitutes for an issue. Trivial typo fixes and repository administration that do not change behavior may be performed without a new issue, but should reference an existing issue when one is relevant.

## Architecture

- `scripts/bootstrap.sh`: validates the host and creates isolated environments without downloading both models.
- `scripts/test_tts.sh`: runs basic TTS, control-tag TTS, and optional cloning in separate Python processes.
- `scripts/test_stt.sh`: normalizes input, tries complete MPS inference, and starts CPU fallback in a fresh process after failure.
- `src/tts_test.py`: MLX model loading, WAV generation, timing, RTF, and memory metrics.
- `src/stt_test.py`: one device/dtype attempt per process, transcript output, timing, RTF, WER, versions, and full failures.
- `src/stt_helper.py`: resolves checkpoint-owned helper code at a pinned Hugging Face revision.
- `src/tts_qwen_local_test.py` and `src/stt_qwen_local_test.py`: the local Apple Silicon Qwen3-TTS / Qwen3-ASR benchmarks, running MLX-Audio's native `qwen3_tts` / `qwen3_asr` implementations. Each spawns its own child process for the model call, so the runner never holds a model resident. A separate backend with separately reported numbers — never merged into a Higgs row.
- `src/benchmark.py`: sequential orchestration only; it must not retain either model.
- `samples/`: committed text fixtures and instructions; user audio is ignored.
- `output/` and `logs/`: generated artifacts; only `.gitkeep` files are tracked.
- `notebooks/mojo_max_m0_t4.ipynb`: standalone Colab T4 runner for issue #57's M0 hardware probe (`docs/research/mojo-max/m0_smoke_test.mojo`). Deliberately separate from `higgs_colab_benchmark.ipynb` — it only installs `pixi`/`modular` and runs the probe, so it can be checked quickly without pulling in the TTS/STT/Qwen stack.
- `skills-lock.json`: pins the official Modular agent skills used for issue #57's Mojo/MAX research (`import-model`, `debug-model`, `serve-model`, `benchmark-model`, `profile-model`, `eval-model`, `mojo-syntax`, `mojo-gpu-fundamentals`, `mojo-python-interop`, `closure_migration`, `new-modular-project`). Restore them into `.agents/`/`.claude/skills` (both gitignored, regenerated like `node_modules`) with `npx skills experimental_install`; a fresh `npx skills add modular/skills` also works but may pick up newer skill revisions than the lockfile pins.
- `scripts/gdrive_sync.py`: uploads/downloads `samples/`, `output/`, and `notebooks/` to/from Google Drive (`make upload-gdrive`, `make download-gdrive FOLDER_ID=<id>`). It authenticates via the local `gcloud` CLI (`gcloud auth print-access-token`, optionally scoped with `--account=<email>`) or an explicit `GDRIVE_ACCESS_TOKEN` env var — there is no separate `gdrive` binary in this project. An agent may use this existing `gcloud` authentication directly for Drive file operations (list/upload/download) without prompting the user to log in again, as long as a credentialed account is already present (`gcloud auth list`).

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
