#!/usr/bin/env python3
"""Prosody + terminal-intonation metrics for M4-T5 (issue #57).

Extends m4_prosody_metrics.py's analyze() (unchanged, imported directly) with:
  - words-per-second tempo (from the clip's own text, split on whitespace)
  - terminal F0 contour: linear-regression slope (Hz/s) of the voiced F0 track
    over the LAST 300-500ms of voiced audio in the clip, to answer the owner's
    question about falling/flat/rising terminal intonation. Negative slope =
    falling (terminal-sounding); ~0 = flat/held; positive = rising.

Median F0 is computed (per m4_prosody_metrics.py) but per the owner's explicit
instruction is NOT used to draw conclusions on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_prosody_metrics import (  # noqa: E402
    analyze,
    autocorr_f0,
    energy_db,
    frame_signal,
    read_wav,
)


def _speech_end_time_s(audio, sr, silence_db=-40, energy_frame_ms=20):
    """Last time (s) that is NOT part of trailing silence, using the same
    energy-threshold convention as m4_prosody_metrics.detect_pauses. Needed
    because the raw last-voiced-F0-frame includes trailing breath/room-noise
    artifacts the autocorrelation pitch tracker mislabels as weakly voiced --
    those octave-jumping artifacts were observed directly on neutral_baseline
    (F0 jumping 400/65/110 Hz within the same 100ms) and must not be counted
    as the speech's own terminal contour."""
    frame_len = int(sr * energy_frame_ms / 1000)
    hop_len = frame_len
    energies = []
    for start in range(0, len(audio) - frame_len, hop_len):
        energies.append(energy_db(audio[start:start + frame_len]))
    energies = np.array(energies)
    if len(energies) == 0:
        return len(audio) / sr
    ref = np.max(energies)
    active = np.nonzero(energies >= (ref + silence_db))[0]
    if len(active) == 0:
        return len(audio) / sr
    last_active = active[-1]
    return (last_active + 1) * hop_len / sr


def terminal_f0_slope(audio, sr, window_ms=400):
    """Linear-regression slope (Hz/s) of voiced F0 over the last `window_ms` ms of
    ACTUAL SPEECH (trailing silence trimmed via energy threshold first -- see
    _speech_end_time_s -- so a noisy pitch-tracker artifact in the silence tail
    cannot masquerade as the sentence's terminal contour). A 3-point median
    filter is applied to the voiced F0 track before the linear fit to suppress
    isolated octave-jump errors from the autocorrelation estimator."""
    frames, hop_len = frame_signal(audio, sr, frame_ms=40, hop_ms=10)
    if len(frames) == 0:
        return {"slope_hz_per_s": None, "n_points": 0, "note": "no frames"}
    f0s = np.array([autocorr_f0(f, sr) for f in frames])
    speech_end_s = _speech_end_time_s(audio, sr)
    frame_times = np.arange(len(frames)) * (hop_len / sr)
    voiced_idx = np.nonzero((f0s > 0) & (frame_times <= speech_end_s + 0.02))[0]
    if len(voiced_idx) < 3:
        return {"slope_hz_per_s": None, "n_points": int(len(voiced_idx)), "note": "too few voiced frames before speech end"}
    last_voiced = voiced_idx[-1]
    window_frames = max(3, int(window_ms / 10))  # hop_ms=10
    start = max(0, last_voiced - window_frames + 1)
    window_idx = voiced_idx[(voiced_idx >= start) & (voiced_idx <= last_voiced)]
    if len(window_idx) < 3:
        return {"slope_hz_per_s": None, "n_points": int(len(window_idx)), "note": "too few voiced frames in window"}
    times_s = frame_times[window_idx]
    f0_vals = f0s[window_idx]
    # 3-point median filter to suppress isolated octave-jump artifacts.
    if len(f0_vals) >= 3:
        padded = np.concatenate(([f0_vals[0]], f0_vals, [f0_vals[-1]]))
        f0_vals = np.array([np.median(padded[i:i + 3]) for i in range(len(f0_vals))])
    # linear fit f0 = a*t + b; slope a in Hz/s
    a, b = np.polyfit(times_s, f0_vals, 1)
    return {
        "slope_hz_per_s": round(float(a), 2),
        "n_points": int(len(window_idx)),
        "window_ms_requested": window_ms,
        "window_ms_actual": round(float(times_s[-1] - times_s[0]) * 1000, 1) if len(times_s) > 1 else 0.0,
        "speech_end_s": round(float(speech_end_s), 3),
        "f0_start_hz": round(float(f0_vals[0]), 1),
        "f0_end_hz": round(float(f0_vals[-1]), 1),
        "direction": (
            "falling" if a < -20 else "rising" if a > 20 else "flat"
        ),
    }


def analyze_full(path, text=None):
    result = analyze(path)
    audio, sr = read_wav(path)
    result["terminal_f0"] = terminal_f0_slope(audio, sr)
    if text:
        n_words = len([w for w in text.split() if w.strip()])
        result["words_per_second"] = round(n_words / result["duration_s"], 2) if result["duration_s"] else None
        result["n_words"] = n_words
    return result


def main():
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if manifest_path and manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        clips = data["per_clip"]
        results = []
        for c in clips:
            r = analyze_full(Path(c["wav_path"]), text=c["text"])
            r["clip_id"] = c["clip_id"]
            results.append(r)
    else:
        paths = sys.argv[1:]
        results = [analyze_full(Path(p)) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
