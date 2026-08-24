#!/usr/bin/env bash
# Pre-downloads every model this project's local benchmarks need, straight into the
# Hugging Face cache, so later `make tts` / `make stt` / Qwen local runs never re-fetch.
#
# Robust to a flaky/VPN network: disables the Xet accelerated-download backend (hf_xet),
# which has been observed to stall indefinitely on some VPNs while still reporting an
# ESTABLISHED TCP connection -- plain HTTP downloads recover via normal read-timeout retries,
# Xet does not. Wraps each snapshot_download in a bounded retry loop with backoff so a single
# transient timeout does not abort the whole run.
#
# Model list must stay in sync with notebooks/model_prefetch_to_drive.ipynb's MODELS list
# (the Colab fallback that prefetches the same models onto Google Drive when the local
# network can't sustain a multi-gigabyte download even with the retry loop below) -- add
# or remove a model in both places together. One intentional divergence: this script uses
# WhisperProcessor.from_pretrained() for openai/whisper-large-v3 (a few MB of tokenizer/
# config files -- the STT encoder's real weights live inside bosonai/higgs-audio-v3-stt's
# own checkpoint), while the notebook achieves the same narrow fetch via
# snapshot_download(..., allow_patterns=["*.json", "*.txt"]) since it needs the result as
# a directory to tar, not just importable via transformers.
set -uo pipefail
cd "$(dirname "$0")/.."

[[ -x .venv-tts/bin/python && -x .venv-stt/bin/python ]] || { echo "ERROR: run make setup first." >&2; exit 2; }

export HF_HUB_DISABLE_XET=1

retry_python() {
  local label="$1" python_bin="$2" script="$3"
  local attempt=1 max_attempts=20
  while true; do
    echo "=== $label (attempt $attempt/$max_attempts) ==="
    if "$python_bin" -c "$script"; then
      echo "=== $label: done ==="
      return 0
    fi
    if (( attempt >= max_attempts )); then
      echo "ERROR: $label failed after $max_attempts attempts." >&2
      return 1
    fi
    local backoff=$(( attempt < 6 ? attempt * 5 : 30 ))
    echo "$label failed (attempt $attempt) -- retrying in ${backoff}s. huggingface_hub resumes partial downloads itself; this loop only restarts the process if it exits."
    sleep "$backoff"
    attempt=$(( attempt + 1 ))
  done
}

echo "HF_HUB_DISABLE_XET=1 (Xet backend disabled -- known to stall on some VPNs without erroring)"

retry_python "Higgs TTS 3 (bosonai/higgs-tts-3-4b)" .venv-tts/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("bosonai/higgs-tts-3-4b")'

retry_python "Higgs STT (bosonai/higgs-audio-v3-stt)" .venv-stt/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("bosonai/higgs-audio-v3-stt", revision="2ffd1aa39f5a1266931e405cba12e404a9f994b2")'

retry_python "Whisper processor (openai/whisper-large-v3)" .venv-stt/bin/python \
  'from transformers import WhisperProcessor; WhisperProcessor.from_pretrained("openai/whisper-large-v3")'

retry_python "Qwen3-TTS Base 0.6B (basic + clone)" .venv-tts/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")'

retry_python "Qwen3-TTS CustomVoice 1.7B" .venv-tts/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")'

retry_python "Qwen3-ASR 0.6B" .venv-tts/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/Qwen3-ASR-0.6B-8bit")'

retry_python "Qwen3-ASR 1.7B" .venv-tts/bin/python \
  'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/Qwen3-ASR-1.7B-8bit")'

echo "=== All models preloaded successfully ==="
echo "Cache location: ~/.cache/huggingface/hub -- subsequent make tts / make stt / Qwen runs will not re-download."
