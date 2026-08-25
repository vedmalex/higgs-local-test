#!/usr/bin/env python3
"""M4-T5/T7 empirical check: control-tag continuity across an independent chunk
boundary, and voice/timbre stability across independently generated segments
(issue #57, M4 Lane 2). Refs docs/research/audiobook/m4-plan.md §3.

Deliberately kept short (all clips a few seconds each, well under the "no long
generation on a shared machine" constraint for this pass) -- a full-chapter run is a
separate, later task.

Loads the model ONCE (matching src/tts_test.py's load()/generate() call convention)
and generates a small fixed set of clips, then scores them with the already-committed
prosody analyzer (m4_prosody_metrics.py, from M4-T0) plus a small spectral-centroid
proxy for timbre added here (no new heavy dependency -- FFT via numpy only).
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
from m4_prosody_metrics import analyze, read_wav  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"
OUT_DIR = ROOT / "output" / "m4_boundary_check"
LOG_PATH = ROOT / "logs" / "m4_boundary_check.json"

SENT1 = "Сегодня удивительный день, полный радости."
SENT2 = "Мы наконец дождались этой прекрасной новости."
TAG = "<|emotion:elation|>"

VOICE_A_TEXT = "Дорога до станции была спокойной и совсем короткой."
VOICE_B_TEXT = "Вечером мы уже сидели дома и пили горячий чай."

CLIPS = {
    # --- Part 1: does an emotion tag survive an independent generate() call boundary?
    "ref_neutral_s1": SENT1,
    "ref_elation_s1": TAG + SENT1,
    "chunk2_noreopen": SENT2,
    "chunk2_reopen": TAG + SENT2,
    "whole_call_once": TAG + SENT1 + " " + SENT2,
    # --- Part 2: does voice/timbre drift across independent segments, and does
    # passing the previous chunk as ref_audio/ref_text stabilize it?
    "voice_a": VOICE_A_TEXT,
    "voice_b_noref": VOICE_B_TEXT,
}


def spectral_centroid(audio: np.ndarray, sr: int, frame_ms: int = 40, hop_ms: int = 10) -> float:
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    centroids = []
    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start : start + frame_len] * np.hanning(frame_len)
        mag = np.abs(np.fft.rfft(frame))
        total = mag.sum()
        if total < 1e-9:
            continue
        centroids.append(float((mag * freqs).sum() / total))
    return round(float(np.mean(centroids)), 1) if centroids else 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    from mlx_audio.tts.utils import load

    print(f"loading {MODEL_ID} ...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    print(f"loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    results = {}
    for name, text in CLIPS.items():
        print(f"generating {name}: {text!r}", file=sys.stderr)
        started = time.perf_counter()
        gens = list(model.generate(text=text, temperature=1.0, max_new_tokens=4096))
        gen_seconds = time.perf_counter() - started
        sr = gens[0].sample_rate
        audio = np.concatenate([np.asarray(g.audio).reshape(-1) for g in gens])
        out_path = OUT_DIR / f"{name}.wav"
        write_wav(out_path, audio, sr)
        results[name] = {
            "text": text,
            "output": str(out_path),
            "generation_seconds": gen_seconds,
            "duration_seconds": len(audio) / sr,
        }

    # voice_b_withref: generate VOICE_B_TEXT again, this time passing voice_a's audio
    # + text as a voice-cloning reference (the model's existing ref_audio/ref_text path,
    # already exercised by src/tts_test.py --mode clone).
    name = "voice_b_withref"
    print(f"generating {name} (with ref_audio=voice_a)", file=sys.stderr)
    started = time.perf_counter()
    gens = list(
        model.generate(
            text=VOICE_B_TEXT,
            ref_audio=str(OUT_DIR / "voice_a.wav"),
            ref_text=VOICE_A_TEXT,
            temperature=1.0,
            max_new_tokens=4096,
        )
    )
    gen_seconds = time.perf_counter() - started
    sr = gens[0].sample_rate
    audio = np.concatenate([np.asarray(g.audio).reshape(-1) for g in gens])
    out_path = OUT_DIR / f"{name}.wav"
    write_wav(out_path, audio, sr)
    results[name] = {
        "text": VOICE_B_TEXT,
        "output": str(out_path),
        "generation_seconds": gen_seconds,
        "duration_seconds": len(audio) / sr,
        "ref_audio": "voice_a.wav",
    }

    # Score every clip with the M4-T0 prosody analyzer + a spectral-centroid timbre proxy.
    for name, entry in results.items():
        prosody = analyze(Path(entry["output"]))
        audio, sr = read_wav(Path(entry["output"]))
        entry["prosody"] = prosody
        entry["spectral_centroid_hz"] = spectral_centroid(audio, sr)

    # --- Part 3: numeric splice check using the actual assemble_chapter() path,
    # reconstructing "the split-with-reopen chunk sequence" as a 2-segment chapter.
    from audiobook import assemble_chapter

    manifest = {
        "segments": [
            {
                "index": 0,
                "status": "done",
                "output_path": results["ref_elation_s1"]["output"],
            },
            {
                "index": 1,
                "status": "done",
                "output_path": results["chunk2_reopen"]["output"],
            },
        ]
    }
    splice = assemble_chapter(manifest, OUT_DIR / "spliced_chapter.wav", silence_ms=200)
    results["_splice_check"] = splice

    LOG_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
