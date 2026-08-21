#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -x .venv-tts/bin/python && -x .venv-stt/bin/python ]] || { echo "ERROR: run make setup first." >&2; exit 2; }

echo "=== Pre-downloading TTS models into Hugging Face cache ==="
.venv-tts/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print("Fetching bosonai/higgs-tts-3-4b...")
snapshot_download("bosonai/higgs-tts-3-4b")
print("TTS model cached successfully.")
PY

echo "=== Pre-downloading STT models into Hugging Face cache ==="
.venv-stt/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import WhisperProcessor, AutoTokenizer

MODEL_ID = "bosonai/higgs-audio-v3-stt"
REVISION = "2ffd1aa39f5a1266931e405cba12e404a9f994b2"

print(f"Fetching {MODEL_ID}@{REVISION}...")
snapshot_download(MODEL_ID, revision=REVISION)

print("Fetching Whisper processor openai/whisper-large-v3...")
WhisperProcessor.from_pretrained("openai/whisper-large-v3")
print("STT models cached successfully.")
PY

echo "=== All models preloaded successfully ==="
