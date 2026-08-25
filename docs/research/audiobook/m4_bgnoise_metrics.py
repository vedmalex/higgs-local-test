#!/usr/bin/env python3
"""Background-noise metrics for issue #57's M4 background-noise probe.

Extends m4_prosody_metrics.py (imported directly, unchanged) with metrics
aimed specifically at the owner's observation that some generated clips carry
audible background sound ("иногда добавляет окружение, некоторые семплы
шумные"). For an audiobook stitched from hundreds of independently generated
segments, the concern is not a constant noise floor (the ear adapts to that)
but a noise floor that VARIES from clip to clip, so it appears/disappears at
segment boundaries.

Method (numpy/scipy only, same convention as the rest of M4):
  1. Reuse detect_pauses()'s 20ms-frame energy classification (silence_db=-40
     relative to the clip's own peak) to find non-speech regions.
  2. From ONLY the non-speech (pause) frames, not silence in general -- if a
     clip has no pause >= min_pause_ms we fall back to the bottom 10% of
     frames by energy, flagged via `used_fallback_floor`.
  3. silence_energy_db: median dB of samples inside detected pause regions.
     This is the direct proxy for background level: near the noise floor of
     an empty/quiet recording, or elevated if something (room tone, hum,
     music bed) is present under the pause.
  4. speech_energy_db: median dB of the top-50%-by-energy frames (voice-active
     estimate, cheaper and more robust than a VAD).
  5. snr_db = speech_energy_db - silence_energy_db. Lower SNR does not by
     itself prove background sound (it could be that the model has quiet
     speech), but combined with an elevated silence_energy_db it is
     informative.
  6. Spectral shape of the pause-region audio via a single FFT over the
     concatenated pause samples (or fallback frames):
       - spectral_flatness: geometric_mean(power) / arithmetic_mean(power)
         over 80Hz-8kHz. Near 0 => a few dominant narrow peaks (tonal, as
         music/hum would produce). Near 1 => broadband/white-noise-like.
       - dominant_freq_hz / dominant_freq_prominence_db: the single strongest
         bin and how far above the spectral median it sits, another tonality
         signal independent of flatness.
       - broadband_energy_ratio: fraction of pause-region spectral power
         outside the typical voiced speech band (80-1000 Hz F0 range extended
         to 4kHz for formants) -- energy present above ~4kHz in "silence" is
         inconsistent with residual breath/room tone and more consistent with
         an synthesized ambience bed or hiss.

None of this proves cause; it is a measurement layer only. All numbers are
descriptive, not a VAD/ASR-grade silence detector -- see docs' "honesty"
section for known limitations (autocorrelation/energy heuristics were already
flagged as fragile in m4_tag_inventory_metrics.py's terminal-F0 module).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_prosody_metrics import energy_db, read_wav  # noqa: E402


def _frame_energies(audio, sr, frame_ms=20):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len
    frames = []
    energies = []
    for start in range(0, len(audio) - frame_len, hop_len):
        f = audio[start:start + frame_len]
        frames.append((start, start + frame_len))
        energies.append(energy_db(f))
    return frames, np.array(energies)


def _pause_regions(audio, sr, silence_db=-40, min_pause_ms=150, frame_ms=20):
    """Same convention as m4_prosody_metrics.detect_pauses, but returns the
    (start_sample, end_sample) spans instead of just durations, so the raw
    samples can be analyzed spectrally."""
    frames, energies = _frame_energies(audio, sr, frame_ms)
    if len(energies) == 0:
        return [], energies
    ref = np.max(energies)
    is_silence = energies < (ref + silence_db)
    min_frames = max(1, int(min_pause_ms / frame_ms))
    regions = []
    run_start = None
    count = 0
    for i, s in enumerate(is_silence):
        if s:
            if run_start is None:
                run_start = i
            count += 1
        else:
            if count >= min_frames:
                regions.append((frames[run_start][0], frames[i - 1][1]))
            run_start = None
            count = 0
    if count >= min_frames:
        regions.append((frames[run_start][0], frames[-1][1]))
    return regions, energies


def _spectral_shape(samples, sr, band=(80, 8000)):
    if len(samples) < 64:
        return {
            "spectral_flatness": None,
            "dominant_freq_hz": None,
            "dominant_freq_prominence_db": None,
            "broadband_energy_ratio_above_4k": None,
            "n_samples": int(len(samples)),
        }
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    p = power[band_mask] + 1e-20
    f = freqs[band_mask]
    log_p = np.log(p)
    geo_mean = np.exp(np.mean(log_p))
    arith_mean = np.mean(p)
    flatness = float(geo_mean / arith_mean) if arith_mean > 0 else None
    dom_idx = int(np.argmax(p))
    dom_freq = float(f[dom_idx])
    median_p_db = 10 * np.log10(np.median(p))
    dom_p_db = 10 * np.log10(p[dom_idx])
    prominence = float(dom_p_db - median_p_db)
    above_4k_mask = freqs > 4000
    below_mask = (freqs >= band[0]) & (freqs <= 4000)
    total = np.sum(power[band_mask])
    above_4k = np.sum(power[above_4k_mask & (freqs <= band[1])])
    ratio = float(above_4k / total) if total > 0 else None
    return {
        "spectral_flatness": round(flatness, 4) if flatness is not None else None,
        "dominant_freq_hz": round(dom_freq, 1),
        "dominant_freq_prominence_db": round(prominence, 1),
        "broadband_energy_ratio_above_4k": round(ratio, 4) if ratio is not None else None,
        "n_samples": int(len(samples)),
    }


def analyze_bgnoise(path, silence_db=-40, min_pause_ms=150):
    audio, sr = read_wav(path)
    duration = len(audio) / sr
    regions, energies = _pause_regions(audio, sr, silence_db, min_pause_ms)

    used_fallback = False
    if regions:
        pause_samples = np.concatenate([audio[s:e] for s, e in regions])
        pause_frame_energies = []
        frame_len = int(sr * 0.02)
        for s, e in regions:
            for start in range(s, e - frame_len, frame_len):
                pause_frame_energies.append(energy_db(audio[start:start + frame_len]))
        pause_frame_energies = np.array(pause_frame_energies) if pause_frame_energies else np.array([])
    else:
        used_fallback = True
        # Fallback: bottom 10% of frames by energy (no true pause detected).
        if len(energies) == 0:
            pause_samples = np.array([])
            pause_frame_energies = np.array([])
        else:
            n = max(1, int(len(energies) * 0.1))
            order = np.argsort(energies)
            low_idx = order[:n]
            frame_len = int(sr * 0.02)
            pause_samples = np.concatenate(
                [audio[i * frame_len: i * frame_len + frame_len] for i in low_idx]
            )
            pause_frame_energies = energies[low_idx]

    if len(energies) > 0:
        n_top = max(1, int(len(energies) * 0.5))
        speech_energy_db = float(np.median(np.sort(energies)[-n_top:]))
    else:
        speech_energy_db = None

    silence_energy_db = float(np.median(pause_frame_energies)) if len(pause_frame_energies) else None
    snr_db = (speech_energy_db - silence_energy_db) if (speech_energy_db is not None and silence_energy_db is not None) else None

    spectral = _spectral_shape(pause_samples, sr)

    total_pause_ms = sum((e - s) / sr * 1000 for s, e in regions)

    return {
        "file": str(path),
        "duration_s": round(duration, 3),
        "num_pause_regions": len(regions),
        "total_pause_ms": round(total_pause_ms, 1),
        "used_fallback_floor": used_fallback,
        "silence_energy_db": round(silence_energy_db, 1) if silence_energy_db is not None else None,
        "speech_energy_db": round(speech_energy_db, 1) if speech_energy_db is not None else None,
        "snr_db": round(snr_db, 1) if snr_db is not None else None,
        **spectral,
    }


def main():
    paths = sys.argv[1:]
    results = [analyze_bgnoise(Path(p)) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
