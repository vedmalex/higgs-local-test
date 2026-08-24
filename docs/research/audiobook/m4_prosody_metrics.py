#!/usr/bin/env python3
"""Prosody metrics for M4-T0 sentiment-tag baseline probe (issue #57).

Uses only numpy/scipy (available in .venv-tts); no librosa/parselmouth/soundfile
present, so F0 is estimated via autocorrelation on 40ms frames.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sampwidth {sampwidth}")
    channels = w.getnchannels() if hasattr(w, "getnchannels") else 1
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def frame_signal(audio, sr, frame_ms=40, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    frames = []
    for start in range(0, len(audio) - frame_len, hop_len):
        frames.append(audio[start:start + frame_len])
    return np.array(frames), hop_len


def autocorr_f0(frame, sr, fmin=60, fmax=400):
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-6:
        return 0.0
    windowed = frame * np.hanning(len(frame))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr) // 2:]
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return 0.0
    segment = corr[min_lag:max_lag]
    if len(segment) == 0 or corr[0] <= 0:
        return 0.0
    peak_idx = np.argmax(segment)
    peak_val = segment[peak_idx]
    if peak_val / corr[0] < 0.3:  # weak periodicity -> unvoiced/silence
        return 0.0
    lag = min_lag + peak_idx
    return sr / lag if lag > 0 else 0.0


def energy_db(frame):
    rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def detect_pauses(audio, sr, silence_db=-40, min_pause_ms=150):
    frame_len = int(sr * 0.02)
    hop_len = frame_len
    energies = []
    for start in range(0, len(audio) - frame_len, hop_len):
        f = audio[start:start + frame_len]
        energies.append(energy_db(f))
    energies = np.array(energies)
    ref = np.max(energies) if len(energies) else 0
    is_silence = energies < (ref + silence_db)
    min_frames = int(min_pause_ms / 20)
    pauses = []
    count = 0
    for s in is_silence:
        if s:
            count += 1
        else:
            if count >= min_frames:
                pauses.append(count * 20)
            count = 0
    if count >= min_frames:
        pauses.append(count * 20)
    return pauses


def analyze(path):
    audio, sr = read_wav(path)
    duration = len(audio) / sr
    frames, hop_len = frame_signal(audio, sr)
    f0s = np.array([autocorr_f0(f, sr) for f in frames])
    voiced_f0 = f0s[f0s > 0]
    energies = np.array([energy_db(f) for f in frames])
    pauses = detect_pauses(audio, sr)

    result = {
        "file": str(path),
        "duration_s": round(duration, 3),
        "f0_median_hz": round(float(np.median(voiced_f0)), 1) if len(voiced_f0) else None,
        "f0_min_hz": round(float(np.min(voiced_f0)), 1) if len(voiced_f0) else None,
        "f0_max_hz": round(float(np.max(voiced_f0)), 1) if len(voiced_f0) else None,
        "f0_range_hz": round(float(np.max(voiced_f0) - np.min(voiced_f0)), 1) if len(voiced_f0) else None,
        "f0_std_hz": round(float(np.std(voiced_f0)), 1) if len(voiced_f0) else None,
        "voiced_frame_ratio": round(float(len(voiced_f0) / len(f0s)), 3) if len(f0s) else None,
        "mean_energy_db": round(float(np.mean(energies)), 1),
        "num_pauses": len(pauses),
        "total_pause_ms": sum(pauses),
        "mean_pause_ms": round(sum(pauses) / len(pauses), 1) if pauses else 0,
    }
    return result


def main():
    paths = sys.argv[1:]
    results = [analyze(Path(p)) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
