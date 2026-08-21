#!/usr/bin/env python3
"""Run benchmarks sequentially; each shell script starts fresh model processes."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for script in ("scripts/test_tts.sh", "scripts/test_stt.sh"):
    subprocess.run([str(ROOT / script)], cwd=ROOT, check=True)

