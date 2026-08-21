#!/usr/bin/env python3
"""Run benchmarks sequentially; each shell script starts fresh model processes."""
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cmd_prefix = ["arch", "-arm64"] if platform.system() == "Darwin" else []

for script in ("scripts/test_tts.sh", "scripts/test_stt.sh"):
    subprocess.run([*cmd_prefix, str(ROOT / script)], cwd=ROOT, check=True)

