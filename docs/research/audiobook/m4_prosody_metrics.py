#!/usr/bin/env python3
"""Prosody metrics for M4-T0 sentiment-tag baseline probe (issue #57).

Uses only numpy/scipy (available in .venv-tts); no librosa/parselmouth/soundfile
present, so F0 is estimated via autocorrelation on 40ms frames.

Also the one place for every other homemade acoustic proxy this project uses
(issue #57/#118 voice-casting follow-up: "тембр голоса мы сами можем
измерить?") -- spectral centroid/tilt/sibilance and a reverberation-decay
proxy live here too, so there is exactly one analyzer, not one per feature.
`src/sentiment_survey/pitch.py` imports this module by path and caches its
`analyze()` output.

CALIBRATION CAVEAT (applies to every field below, not just F0): every
estimator here is homemade, with no librosa/parselmouth/scipy.signal.stft to
check against. They are good for *relative, within-this-project* comparisons
between clips (this file's own precedent for `f0_median_hz`, carried over
verbatim from the M4-T0 baseline: "good as a 'same speaker or not' measure,
not as an exact pitch reading") -- never report any of them as a calibrated
physical quantity (real dB SPL, real RT60 in seconds, a calibrated Hz
centroid). A human-vs-measurement disagreement is a reason to re-check the
estimator, not automatically a human error.
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


def spectral_features(frames, sr, sibilance_band=(5000, 8000), tilt_band=(200, 6000)):
    """Brightness (centroid), dullness/muffled-ness (tilt), sibilance
    ("шипящесть"), and a proximity-to-mic proxy (low-band energy ratio),
    all from one pass of per-frame FFT magnitude spectra -- avoids a
    separate full-audio pass per feature. `frames` is `frame_signal()`'s
    (40ms/10ms) output, same frames F0 is measured on.

    - `spectral_centroid_hz`: the magnitude-weighted mean frequency --
      the algorithm already used once in this project for a timbre proxy
      (docs/research/audiobook/m4_boundary_check.py's `spectral_centroid`),
      reused verbatim here rather than re-derived, per "one calculator, not
      two". Higher = perceptually brighter.
    - `spectral_tilt_db_per_khz`: slope of a linear fit to magnitude (dB)
      vs. frequency (kHz) over `tilt_band`. More negative = energy falls
      off faster with frequency = duller/more muffled ("глуховато").
    - `sibilance_ratio`: fraction of each frame's spectral energy inside
      `sibilance_band` (5-8 kHz, where /s/ /sh/ /ch/ energy concentrates in
      Russian speech) -- proxy for "шипящий" strength.
    - `low_band_ratio`: fraction of each frame's spectral energy below
      300 Hz. A closer microphone boosts low frequencies (the proximity
      effect on directional mics) -- higher values lean "close to mic".
    """
    if len(frames) == 0:
        return {
            "spectral_centroid_hz": None, "spectral_tilt_db_per_khz": None,
            "sibilance_ratio": None, "low_band_ratio": None,
        }
    frame_len = frames.shape[1]
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    sib_mask = (freqs >= sibilance_band[0]) & (freqs <= sibilance_band[1])
    tilt_mask = (freqs >= tilt_band[0]) & (freqs <= tilt_band[1])
    low_mask = freqs <= 300
    window = np.hanning(frame_len)

    centroids, sib_ratios, low_ratios, tilts = [], [], [], []
    for frame in frames:
        mag = np.abs(np.fft.rfft(frame * window))
        total = mag.sum()
        if total < 1e-9:
            continue
        centroids.append(float((mag * freqs).sum() / total))
        sib_ratios.append(float(mag[sib_mask].sum() / total))
        low_ratios.append(float(mag[low_mask].sum() / total))
        if tilt_mask.sum() > 8:
            mag_db = 20 * np.log10(mag[tilt_mask] + 1e-9)
            slope = np.polyfit(freqs[tilt_mask] / 1000.0, mag_db, 1)[0]
            tilts.append(float(slope))

    return {
        "spectral_centroid_hz": round(float(np.mean(centroids)), 1) if centroids else None,
        "spectral_tilt_db_per_khz": round(float(np.mean(tilts)), 2) if tilts else None,
        "sibilance_ratio": round(float(np.mean(sib_ratios)), 4) if sib_ratios else None,
        "low_band_ratio": round(float(np.mean(low_ratios)), 4) if low_ratios else None,
    }


def estimate_reverb_tail_ms(audio, sr, frame_ms=20, active_db=-15, silence_db=-40, max_tail_frames=25):
    """Reverberation proxy: how long energy takes to decay from an active
    (loud) frame down toward the silence floor, averaged over every
    loud-to-quiet transition in the clip. A slower decay (longer tail)
    means more reverberant energy hanging on after speech -- "есть
    реверберация" -- rather than a clean, dry cutoff.

    Same 20ms frame grid and silence_db floor as detect_pauses(), so this
    reads as "the shape of the same transitions detect_pauses() already
    finds the boundaries of", not a second unrelated measurement.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(audio) < frame_len:
        return None
    energies = np.array([
        energy_db(audio[start:start + frame_len])
        for start in range(0, len(audio) - frame_len, frame_len)
    ])
    if len(energies) == 0:
        return None
    ref = np.max(energies)
    tails = []
    i, n = 0, len(energies)
    while i < n - 1:
        if energies[i] >= ref + active_db:
            j = i + 1
            tail_len = 0
            while (j < n and energies[j] < ref + active_db
                   and energies[j] >= ref + silence_db and tail_len < max_tail_frames):
                tail_len += 1
                j += 1
            if tail_len > 0:
                tails.append(tail_len)
            i = j
        else:
            i += 1
    return round(float(np.mean(tails)) * frame_ms, 1) if tails else None


def analyze(path):
    audio, sr = read_wav(path)
    duration = len(audio) / sr
    frames, hop_len = frame_signal(audio, sr)
    f0s = np.array([autocorr_f0(f, sr) for f in frames])
    voiced_f0 = f0s[f0s > 0]
    energies = np.array([energy_db(f) for f in frames])
    pauses = detect_pauses(audio, sr)
    spectral = spectral_features(frames, sr)
    reverb_tail_ms = estimate_reverb_tail_ms(audio, sr)

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
        # Issue #57/#118 voice-casting follow-up -- see spectral_features()/
        # estimate_reverb_tail_ms() docstrings and the module-level
        # calibration caveat above.
        "reverb_tail_ms": reverb_tail_ms,
    }
    result.update(spectral)
    return result


def main():
    paths = sys.argv[1:]
    results = [analyze(Path(p)) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
