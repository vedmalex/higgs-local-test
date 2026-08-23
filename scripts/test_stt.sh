#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -m)" == "arm64" ]] || { echo "ERROR: native arm64 shell required." >&2; exit 2; }
[[ -x .venv-stt/bin/python ]] || { echo "ERROR: run make setup first." >&2; exit 2; }
[[ -f samples/stt_ru.wav ]] || { echo "STT: SKIPPED — place a Russian recording at samples/stt_ru.wav"; exit 0; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg missing; run brew install ffmpeg." >&2; exit 2; }
mkdir -p output logs
echo "Close Docker, local LLMs, large IDEs, and VMs. STT runs alone."
ffmpeg -hide_banner -loglevel error -y -i samples/stt_ru.wav -ac 1 -ar 16000 output/stt_ru_16k.wav
: > logs/stt.log
if .venv-stt/bin/python -c 'import torch,sys; sys.exit(0 if torch.backends.mps.is_available() else 1)'; then
  echo "Attempting complete MPS FP16 inference..." | tee -a logs/stt.log
  if /usr/bin/time -l .venv-stt/bin/python src/stt_test.py --device mps --dtype float16 --reference builtin --audio output/stt_ru_16k.wav --output output/stt_ru.txt 2>&1 | tee -a logs/stt.log; then
    exit 0
  fi
  echo "MPS inference failed. Starting CPU fallback in a fresh process..." | tee -a logs/stt.log
else
  echo "MPS unavailable. Starting CPU compatibility test..." | tee -a logs/stt.log
fi
/usr/bin/time -l .venv-stt/bin/python src/stt_test.py --device cpu --dtype float32 --reference builtin --audio output/stt_ru_16k.wav --output output/stt_ru.txt 2>&1 | tee -a logs/stt.log
