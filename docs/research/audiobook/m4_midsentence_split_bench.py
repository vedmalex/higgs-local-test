#!/usr/bin/env python3
"""M4 mid-sentence split probe (issue #57).

The `final-boundary-continuing` task in `src/sentiment_survey/task_sets/
final_intonation.json` tested the wrong thing: it asked whether a
grammatically COMPLETE sentence (ending in a period) sounds "finished",
which has no honest correct answer -- see docs/research/audiobook/
m4-tag-inventory-results.md sec. 5 item 6 for the writeup.

The real open question from that investigation is different: what happens
when `_force_split_long_sentence()` (src/audiobook.py) cuts a sentence that
is over `--max-chars` at a point that is NOT a sentence boundary -- a comma,
a conjunction, mid-clause? The model generating the first half has no idea
a continuation exists. Does its terminal contour still fall like a genuine
sentence end (a "false full stop"), and is the resulting splice audible as
a defect?

This script generates 4 clips from one real sentence taken from
`samples/audiobook/prepared/sb-1-19.txt` (owner's existing example text,
not invented), cut at a comma before a subordinate clause -- NOT at a
period:

    FULL:  "Хотя он и старался скрыть свое естественное величие, великие
            мудрецы, собравшиеся там, были искушены в физиономистике и
            потому почтили его, поднявшись со своих мест."
    CUT:   after "...естественное величие," (comma, subordinate clause,
           no sentence-final punctuation)

  - `whole`    : FULL generated in ONE call (the control -- what this
                 exact material sounds like when the model knows the
                 sentence continues).
  - `fragment` : the first half ("Хотя он ... величие,") generated ALONE,
                 exactly as `generate_segments` would generate it as an
                 independent chunk with no knowledge of what follows.
  - `remainder`: the second half ("великие мудрецы ... мест.") generated
                 ALONE, needed to build the spliced clip.
  - `spliced`  : `fragment` + `assemble_chapter`'s default 200 ms join
                 silence + `remainder` -- reproduces exactly what chapter
                 assembly would produce if `_force_split_long_sentence` cut
                 here.

All 4 clips use the SAME pinned voice reference (`voices/narrator_e0`,
PR #133) so a difference in F0 contour is never just a different sampled
voice -- both `fragment`/`remainder` and `whole` are anchored to one voice.

Analysis reuses docs/research/audiobook/m4_prosody_metrics.py's primitives
(`read_wav`, `frame_signal`, `autocorr_f0`, `detect_pauses`) -- no second
F0/pause estimator is written here. It compares:

  1. The F0 trajectory over the last ~600 ms of `fragment` (does it fall
     toward the end, like a sentence-final cadence?) against the F0
     trajectory at the SAME approximate textual position inside `whole`
     (found by proportional character-offset of the cut point in `whole`'s
     duration -- a rough estimate, not forced alignment; flagged as such).
  2. The pause/silence duration right after "величие," inside `whole`
     (the model's own natural comma-pause) against the artificial
     200 ms join silence in `spliced`.

Caveat repeated from m4_prosody_metrics.py's own docstring: this estimator
is a plain autocorrelation pitch tracker (no librosa/parselmouth), good
for "does the contour fall or not", not for absolute Hz accuracy -- treat
octave-level errors as possible.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mlx_audio.audio_io import write as audio_write  # noqa: E402
from mlx_audio.tts.utils import load  # noqa: E402

from audiobook import load_voice_from_registry  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"
MAX_NEW_TOKENS = 4096
VOICE_NAME = "narrator_e0"
JOIN_SILENCE_MS = 200  # assemble_chapter()'s default silence_ms

# The real sentence, taken verbatim from samples/audiobook/prepared/sb-1-19.txt
# (paragraph about the sages recognizing Sukadeva Gosvami despite his
# disguise), cut at a comma before a subordinate clause -- not at a period.
FRAGMENT = "Хотя он и старался скрыть свое естественное величие,"
REMAINDER = "великие мудрецы, собравшиеся там, были искушены в физиономистике и потому почтили его, поднявшись со своих мест."
FULL = f"{FRAGMENT} {REMAINDER}"

PROSODY_METRICS_PATH = ROOT / "docs" / "research" / "audiobook" / "m4_prosody_metrics.py"


def _load_prosody_metrics():
    spec = importlib.util.spec_from_file_location("m4_prosody_metrics", PROSODY_METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PROSODY_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prosody = _load_prosody_metrics()


def machine_state() -> dict:
    import subprocess

    def run(cmd):
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<error: {exc}>"

    return {"uptime": run(["uptime"]), "vm_swapusage": run(["sysctl", "vm.swapusage"])}


def write_wav_concat(frag_path: Path, rem_path: Path, out_path: Path, silence_ms: int) -> dict:
    """Splice frag + silence + rem exactly like assemble_chapter() does:
    int16 PCM, single mono stream, no crossfade."""
    frag_audio, frag_sr = prosody.read_wav(frag_path)
    rem_audio, rem_sr = prosody.read_wav(rem_path)
    if frag_sr != rem_sr:
        raise RuntimeError(f"sample rate mismatch: fragment={frag_sr} remainder={rem_sr}")
    sr = frag_sr
    silence_samples = int(silence_ms / 1000 * sr)
    silence = np.zeros(silence_samples, dtype=np.float64)
    spliced = np.concatenate([frag_audio, silence, rem_audio])
    pcm = np.clip(spliced * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return {
        "sample_rate": sr,
        "fragment_samples": len(frag_audio),
        "silence_samples": silence_samples,
        "remainder_samples": len(rem_audio),
        "splice_at_seconds": len(frag_audio) / sr,
    }


def f0_tail_trend(path: Path, tail_ms: float = 600.0) -> dict:
    """F0 (Hz) over 40ms/10ms frames for the LAST `tail_ms` of the clip,
    plus a linear-fit slope (Hz/s) -- a falling slope is the acoustic
    signature of a sentence-final cadence."""
    audio, sr = prosody.read_wav(path)
    frames, hop_len = prosody.frame_signal(audio, sr)
    f0s = np.array([prosody.autocorr_f0(f, sr) for f in frames])
    hop_s = hop_len / sr
    n_tail = max(1, int(tail_ms / 1000 / hop_s))
    tail = f0s[-n_tail:]
    times = np.arange(len(tail)) * hop_s
    voiced_mask = tail > 0
    slope = None
    if voiced_mask.sum() >= 3:
        slope = float(np.polyfit(times[voiced_mask], tail[voiced_mask], 1)[0])
    return {
        "tail_ms_requested": tail_ms,
        "tail_frames": len(tail),
        "tail_f0_hz": [round(float(x), 1) for x in tail],
        "tail_f0_slope_hz_per_s": round(slope, 1) if slope is not None else None,
        "tail_voiced_ratio": round(float(voiced_mask.mean()), 3) if len(tail) else None,
    }


def f0_at_proportional_offset(path: Path, char_offset: float, total_chars: int, tail_ms: float = 600.0) -> dict:
    """Approximate the F0 trend AT the textual position `char_offset` /
    `total_chars` inside a longer clip, by mapping the character fraction
    onto the clip's duration (constant-speaking-rate assumption -- no
    forced alignment available in this project). Returns the trend over
    the `tail_ms` window ending at that estimated timestamp."""
    audio, sr = prosody.read_wav(path)
    duration_s = len(audio) / sr
    frac = char_offset / total_chars
    approx_t = frac * duration_s
    frames, hop_len = prosody.frame_signal(audio, sr)
    f0s = np.array([prosody.autocorr_f0(f, sr) for f in frames])
    hop_s = hop_len / sr
    end_frame = min(len(f0s), int(approx_t / hop_s))
    n_tail = max(1, int(tail_ms / 1000 / hop_s))
    start_frame = max(0, end_frame - n_tail)
    window = f0s[start_frame:end_frame]
    times = np.arange(len(window)) * hop_s
    voiced_mask = window > 0
    slope = None
    if voiced_mask.sum() >= 3:
        slope = float(np.polyfit(times[voiced_mask], window[voiced_mask], 1)[0])
    return {
        "approx_cut_time_s": round(approx_t, 3),
        "clip_duration_s": round(duration_s, 3),
        "window_f0_hz": [round(float(x), 1) for x in window],
        "window_f0_slope_hz_per_s": round(slope, 1) if slope is not None else None,
        "note": "position estimated by proportional character offset, NOT forced alignment",
    }


def pause_after_cut(path: Path, char_offset: float, total_chars: int, search_window_ms: float = 1500.0) -> dict:
    """Find the pause (if any) closest to the estimated cut position inside
    `whole`, using prosody.detect_pauses()'s own 20ms-frame silence
    detector -- this is the model's OWN comma-pause, to compare against the
    artificial join silence in `spliced`."""
    audio, sr = prosody.read_wav(path)
    duration_s = len(audio) / sr
    approx_t = (char_offset / total_chars) * duration_s
    frame_len = int(sr * 0.02)
    hop_len = frame_len
    energies = []
    for start in range(0, len(audio) - frame_len, hop_len):
        f = audio[start:start + frame_len]
        energies.append(prosody.energy_db(f))
    energies = np.array(energies)
    ref = np.max(energies) if len(energies) else 0
    is_silence = energies < (ref - 40)
    frame_times = np.arange(len(is_silence)) * 0.02
    window_mask = np.abs(frame_times - approx_t) < (search_window_ms / 1000)
    candidate = is_silence & window_mask
    run_ms = int(candidate.sum()) * 20
    return {
        "approx_cut_time_s": round(approx_t, 3),
        "silence_ms_near_cut": run_ms,
        "note": "counts silent 20ms frames within +/- search window of the estimated cut position",
    }


def main() -> None:
    out_dir = ROOT / "output" / "m4_midsentence_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"FRAGMENT ({len(FRAGMENT)} chars): {FRAGMENT!r}", flush=True)
    print(f"REMAINDER ({len(REMAINDER)} chars): {REMAINDER!r}", flush=True)
    print(f"FULL ({len(FULL)} chars): {FULL!r}", flush=True)

    state_before = machine_state()
    print("machine state before run:", json.dumps(state_before, ensure_ascii=False), flush=True)

    codes, ref_text, source = load_voice_from_registry(VOICE_NAME, ROOT / "voices")
    print(f"voice reference: {VOICE_NAME!r} from {source}", flush=True)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    t0 = time.perf_counter()
    list(model.generate(text="Это короткая прогревочная фраза перед замером.",
                         temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
                         ref_audio_codes=codes, ref_text=ref_text))
    warmup_seconds = time.perf_counter() - t0
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded)", flush=True)

    mx.reset_peak_memory()
    run_start = time.perf_counter()

    clips = [("whole", FULL), ("fragment", FRAGMENT), ("remainder", REMAINDER)]
    per_clip = []
    for clip_id, text in clips:
        t0 = time.perf_counter()
        results = list(model.generate(
            text=text, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
            ref_audio_codes=codes, ref_text=ref_text,
        ))
        result = results[-1]
        mx.eval(result.audio)
        wall = time.perf_counter() - t0
        audio = np.asarray(result.audio).reshape(-1)
        sr = result.sample_rate
        wav_path = out_dir / f"{clip_id}.wav"
        audio_write(str(wav_path), audio, sr)
        per_clip.append({
            "clip_id": clip_id, "text": text, "chars": len(text),
            "wall_seconds": wall, "audio_duration_seconds": len(audio) / sr,
            "wav_path": str(wav_path),
        })
        print(f"  {clip_id}: wall={wall:.3f}s duration={len(audio)/sr:.3f}s -> {wav_path}", flush=True)

    run_wall = time.perf_counter() - run_start
    peak_mlx_gib = mx.get_peak_memory() / (1024 ** 3)
    state_after = machine_state()

    frag_path = out_dir / "fragment.wav"
    rem_path = out_dir / "remainder.wav"
    whole_path = out_dir / "whole.wav"
    spliced_path = out_dir / "spliced.wav"
    splice_info = write_wav_concat(frag_path, rem_path, spliced_path, JOIN_SILENCE_MS)
    print(f"  spliced: {splice_info} -> {spliced_path}", flush=True)

    cut_char_offset = len(FRAGMENT) + 1  # +1 for the space joining the two halves in FULL

    analysis = {
        "fragment_tail_f0": f0_tail_trend(frag_path),
        "whole_at_cut_f0": f0_at_proportional_offset(whole_path, cut_char_offset, len(FULL)),
        "whole_pause_near_cut": pause_after_cut(whole_path, cut_char_offset, len(FULL)),
        "spliced_join_silence_ms": JOIN_SILENCE_MS,
        "spliced_full_analysis": prosody.analyze(spliced_path),
        "whole_full_analysis": prosody.analyze(whole_path),
        "fragment_full_analysis": prosody.analyze(frag_path),
        "remainder_full_analysis": prosody.analyze(rem_path),
    }

    result = {
        "sentence_full": FULL,
        "cut_fragment": FRAGMENT,
        "cut_remainder": REMAINDER,
        "voice_name": VOICE_NAME,
        "voice_source": source,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "run_wall_seconds": run_wall,
        "peak_mlx_gib": peak_mlx_gib,
        "per_clip": per_clip,
        "splice_info": splice_info,
        "analysis": analysis,
        "machine_state_before": state_before,
        "machine_state_after": state_after,
    }
    out_json = ROOT / "logs" / "m4_midsentence_split.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
