#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -m)" == "arm64" ]] || { echo "ERROR: native arm64 shell required." >&2; exit 2; }
[[ -x .venv-tts/bin/python ]] || { echo "ERROR: run make setup first." >&2; exit 2; }
mkdir -p output logs
echo "Close Docker, local LLMs, large IDEs, and VMs. TTS runs alone in separate processes."

run_timed() {
  local name="$1"; shift
  /usr/bin/time -l "$@" 2>&1 | tee "logs/${name}.log"
}

run_timed tts_basic .venv-tts/bin/python src/tts_test.py --mode basic --output output/tts_ru_basic.wav
run_timed tts_controls .venv-tts/bin/python src/tts_test.py --mode controls --output output/tts_ru_controls.wav

if [[ -f samples/reference.wav && -f samples/reference.txt ]]; then
  run_timed tts_clone .venv-tts/bin/python src/tts_test.py --mode clone --output output/tts_ru_clone.wav
else
  echo "Voice cloning: SKIPPED (add samples/reference.wav and samples/reference.txt)" | tee logs/tts_clone.log
fi

