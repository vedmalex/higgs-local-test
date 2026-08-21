#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "uname -m: $(uname -m)"
sw_vers
echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory bytes: $(sysctl -n hw.memsize)"
echo "Python: $(python3 --version 2>&1)"
echo "Python executable: $(python3 -c 'import sys; print(sys.executable)')"
echo "Python machine: $(python3 -c 'import platform; print(platform.machine())')"

if [[ "$(uname -m)" != "arm64" ]] || [[ "$(python3 -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
  echo "ERROR: native arm64 shell and Python are required; Rosetta/x86_64 detected." >&2
  exit 2
fi

