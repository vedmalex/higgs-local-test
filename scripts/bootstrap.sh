#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this benchmark targets macOS." >&2
  exit 2
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: current shell is $(uname -m). Open a native arm64 Terminal (not Rosetta)." >&2
  exit 2
fi
if [[ "$(python3 -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
  echo "ERROR: python3 is running as x86_64/Rosetta. Install/use native arm64 Python 3.11 or 3.12." >&2
  exit 2
fi

python3 - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(f"ERROR: Python 3.11 or 3.12 required, found {sys.version.split()[0]}")
PY

if ! command -v brew >/dev/null; then
  echo "ERROR: Homebrew is required: https://brew.sh" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg is missing. Install now with: brew install ffmpeg"
  read -r -p "Install ffmpeg now? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then brew install ffmpeg; else exit 2; fi
fi

echo "Creating isolated TTS environment..."
python3 -m venv .venv-tts
.venv-tts/bin/python -m pip install --upgrade pip setuptools wheel
# jiwer is for the Qwen3-ASR local test's WER, which runs in this environment
# because Qwen3-ASR is an MLX-Audio model, not part of the Torch STT stack.
.venv-tts/bin/python -m pip install mlx-audio torch jiwer
.venv-tts/bin/python -c "import mlx, mlx_audio, torch; print('MLX, MLX Audio, and Torch OK')"

echo "Creating isolated STT environment..."
python3 -m venv .venv-stt
.venv-stt/bin/python -m pip install --upgrade pip setuptools wheel
.venv-stt/bin/python -m pip install "transformers==4.51.0" torch torchaudio librosa soundfile numpy jiwer sentencepiece accelerate huggingface_hub
.venv-stt/bin/python -c "import torch, torchaudio, librosa, transformers, soundfile, huggingface_hub; print('STT base imports OK'); print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
.venv-stt/bin/python src/stt_helper.py --prepare-only

cat <<'EOF'
Installation completed.

Close Docker Desktop, local LLMs, large IDE processes, and virtual machines before a benchmark.

TTS:
./scripts/test_tts.sh

STT:
place samples/stt_ru.wav
./scripts/test_stt.sh
EOF
