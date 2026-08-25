#!/usr/bin/env python3
"""Minimal, tightly-bounded variant of m4_boundary_check.py -- just the two clips that
answer the single key question (does a control tag survive an independent chunk-boundary
generate() call), with a small max_new_tokens cap so a heavily-loaded shared machine
cannot turn this into a long run. See m4_boundary_check.py for the full test set docstring.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audiobook import write_wav  # noqa: E402
from m4_prosody_metrics import analyze  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"
OUT_DIR = ROOT / "output" / "m4_boundary_check"
LOG_PATH = ROOT / "logs" / "m4_boundary_check_minimal.json"

SENT2 = "Мы наконец дождались этой прекрасной новости."
TAG = "<|emotion:elation|>"

CLIPS = {
    "chunk2_noreopen": SENT2,
    "chunk2_reopen": TAG + SENT2,
}

MAX_NEW_TOKENS = 200  # bound worst-case run length under heavy contention


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    from mlx_audio.tts.utils import load

    print(f"loading {MODEL_ID} ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    print(f"loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)

    results = {}
    for name, text in CLIPS.items():
        print(f"generating {name}: {text!r}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        gens = list(model.generate(text=text, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS))
        gen_seconds = time.perf_counter() - started
        sr = gens[0].sample_rate
        audio = np.concatenate([np.asarray(g.audio).reshape(-1) for g in gens])
        out_path = OUT_DIR / f"{name}.wav"
        write_wav(out_path, audio, sr)
        prosody = analyze(out_path)
        results[name] = {
            "text": text,
            "output": str(out_path),
            "generation_seconds": gen_seconds,
            "duration_seconds": len(audio) / sr,
            "prosody": prosody,
        }
        print(f"  done in {gen_seconds:.1f}s, {results[name]['duration_seconds']:.2f}s audio",
              file=sys.stderr, flush=True)
        LOG_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
