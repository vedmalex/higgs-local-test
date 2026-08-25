#!/usr/bin/env python3
"""Listening probe for issue #114's short-embedded-quote pause requirement (owner
decision, dsl-spec.md sec. 2.9): does wrapping a short quote in `[пауза]` sugar inside a
`#prose` block produce an audibly distinct pause, not the blended transition the
un-marked Э0 narration had?

Generates two clips from the exact sentence the owner named
(`...выразили свое одобрение словами: «Очень хорошо!»`, from
`samples/audiobook/prepared/sb-1-19.txt`):
  before.wav -- the plain sentence, no pause markup (matches how Э0 was actually marked
                up: this exact text sits inside one un-split #prose paragraph today).
  after.wav  -- the same sentence compiled from a `.abs` snippet using the accepted
                `[пауза]` mechanism from dsl-spec.md sec. 2.9, via the REAL compiler
                (scripts/audiobook_dsl.py), not a hand-typed tag.

Also measures silence duration around the quote in both clips (RMS-below-threshold
run length) as a numeric, reproducible signal -- explicitly NOT a substitute for the
owner's own listening judgment, just a number to report alongside it.

Usage (from repo root, needs the project's .venv-tts -- mlx_audio, numpy):
    .venv-tts/bin/python3 docs/research/audiobook/e1_short_quote_pause_probe.py
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import audiobook_dsl as dsl  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"

SENTENCE_BEFORE = (
    "Все великие мудрецы, собравшиеся там, восторженно приняли решение Махараджи "
    "Парикшита и выразили свое одобрение словами: «Очень хорошо!»"
)

ABS_AFTER = (
    "#prose\n"
    "Все великие мудрецы, собравшиеся там, восторженно приняли решение Махараджи "
    "Парикшита и выразили свое одобрение словами: [пауза] «Очень хорошо!» [пауза]\n"
)

OUT_DIR = ROOT / "output" / "m4_dsl_short_quote"


def compile_after_text() -> str:
    _doc, segs = dsl.compile_source(ABS_AFTER)
    assert len(segs) == 1
    return segs[0]["text"]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return audio, sr


def measure_silences(audio: np.ndarray, sr: int, threshold: float = 0.02, min_ms: float = 40.0):
    """Return a list of (start_ms, duration_ms) for runs of RMS-below-threshold audio,
    computed over 10ms windows. A cheap, transparent heuristic -- not a perfect VAD --
    used only to report a number alongside the owner's own listening judgment, never as
    a replacement for it."""
    win = max(1, int(sr * 0.010))
    n_windows = len(audio) // win
    silences = []
    run_start = None
    for i in range(n_windows):
        seg = audio[i * win : (i + 1) * win]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        is_quiet = rms < threshold
        t_ms = i * win / sr * 1000.0
        if is_quiet and run_start is None:
            run_start = t_ms
        elif not is_quiet and run_start is not None:
            dur = t_ms - run_start
            if dur >= min_ms:
                silences.append((run_start, dur))
            run_start = None
    if run_start is not None:
        dur = n_windows * win / sr * 1000.0 - run_start
        if dur >= min_ms:
            silences.append((run_start, dur))
    return silences


def generate(model, text: str, out_path: Path) -> dict:
    from mlx_audio.audio_io import write as audio_write

    started = time.perf_counter()
    results = list(model.generate(text=text, temperature=1.0, max_new_tokens=4096))
    generation_seconds = time.perf_counter() - started
    if not results:
        raise RuntimeError(f"model.generate produced no result for: {text!r}")
    sample_rate = results[0].sample_rate
    audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_write(str(out_path), audio, sample_rate)
    return {
        "path": str(out_path),
        "chars": len(text),
        "generation_seconds": generation_seconds,
        "audio_duration_seconds": len(audio) / sample_rate,
        "sample_rate": sample_rate,
    }


def main() -> None:
    from mlx_audio.tts.utils import load

    after_text = compile_after_text()
    print("BEFORE text:", repr(SENTENCE_BEFORE))
    print("AFTER  text:", repr(after_text))

    print("Loading model...")
    model = load(MODEL_ID, model_type="higgs_audio_v3")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before_path = OUT_DIR / "before.wav"
    after_path = OUT_DIR / "after.wav"

    print("Generating BEFORE (no pause markup)...")
    before_meta = generate(model, SENTENCE_BEFORE, before_path)
    print(before_meta)

    print("Generating AFTER ([пауза] around the quote, via real compiler)...")
    after_meta = generate(model, after_text, after_path)
    print(after_meta)

    for label, path in (("BEFORE", before_path), ("AFTER", after_path)):
        audio, sr = read_wav(path)
        silences = measure_silences(audio, sr)
        print(f"\n{label} ({path}): {len(audio) / sr:.2f}s total")
        if not silences:
            print("  no silence run >= 40ms detected anywhere in the clip")
        for start_ms, dur_ms in silences:
            print(f"  silence at {start_ms:.0f}ms, duration {dur_ms:.0f}ms")


if __name__ == "__main__":
    main()
