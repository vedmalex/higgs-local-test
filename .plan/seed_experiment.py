#!/usr/bin/env python3
"""Issue #57 follow-up: does mx.random.seed fix voice across DIFFERENT text?
Also: same seed + same text -> bitwise identical output?

Writes results incrementally to .plan/seed_experiment_results.json as it goes,
so a kill loses at most the current in-flight call.
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/research/audiobook"))
from m4_prosody_metrics import analyze  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".plan" / "seed_experiment_wavs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / ".plan" / "seed_experiment_results.json"

MODEL_ID = "bosonai/higgs-tts-3-4b"

FRAGMENTS = [
    "Сегодня прекрасная погода для прогулки в парке.",
    "Компьютер медленно загружался этим утром.",
    "Она приготовила вкусный борщ на обед.",
    "Дети играли во дворе до самого вечера.",
    "Поезд опоздал на пятнадцать минут.",
    "Река тихо несла свои воды мимо старого моста.",
]

SEED = 42


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"seeded": [], "unseeded": [], "bitwise_check": None}


def save_results(results):
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def gen_one(model, text, seed, tag):
    started = time.perf_counter()
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed
    results = list(model.generate(text=text, temperature=1.0, max_new_tokens=2048, **kwargs))
    elapsed = time.perf_counter() - started
    r = results[0]
    audio = np.asarray(r.audio).reshape(-1)
    sr = r.sample_rate
    out_path = OUT_DIR / f"{tag}.wav"
    write_wav(out_path, audio, sr)
    prosody = analyze(out_path)
    return {
        "tag": tag,
        "text": text,
        "seed": seed,
        "wall_seconds": round(elapsed, 2),
        "duration_s": prosody["duration_s"],
        "f0_median_hz": prosody["f0_median_hz"],
        "f0_std_hz": prosody["f0_std_hz"],
        "audio_sha256_prefix": None,  # filled below
        "path": str(out_path),
    }


def sha256_of_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print("Loading model...", flush=True)
    started = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)

    results = load_results()
    done_tags = {e["tag"] for e in results["seeded"]} | {e["tag"] for e in results["unseeded"]}

    # Part A: same fixed seed=42, different text, one call each (fresh generate() call
    # per fragment -- this is exactly how audiobook segments are generated: independent
    # calls). mx.random.seed(42) is set fresh inside generate() every time seed is passed.
    for i, text in enumerate(FRAGMENTS):
        tag = f"seeded_{i}"
        if tag in done_tags:
            print(f"skip {tag} (already done)", flush=True)
            continue
        print(f"generating {tag}: {text[:40]}...", flush=True)
        entry = gen_one(model, text, SEED, tag)
        entry["audio_sha256_prefix"] = sha256_of_file(Path(entry["path"]))[:16]
        results["seeded"].append(entry)
        save_results(results)
        print(f"  -> F0 median {entry['f0_median_hz']} Hz, wall {entry['wall_seconds']}s", flush=True)

    # Part B: no seed fixation (seed=None -> global RNG state just continues from
    # wherever it is, non-reproducible), same fragments.
    for i, text in enumerate(FRAGMENTS):
        tag = f"unseeded_{i}"
        if tag in done_tags:
            print(f"skip {tag} (already done)", flush=True)
            continue
        print(f"generating {tag}: {text[:40]}...", flush=True)
        entry = gen_one(model, text, None, tag)
        entry["audio_sha256_prefix"] = sha256_of_file(Path(entry["path"]))[:16]
        results["unseeded"].append(entry)
        save_results(results)
        print(f"  -> F0 median {entry['f0_median_hz']} Hz, wall {entry['wall_seconds']}s", flush=True)

    # Part C: same seed + same text, twice -> bitwise identical audio?
    if results.get("bitwise_check") is None:
        print("bitwise check: same seed + same text twice", flush=True)
        text = FRAGMENTS[0]
        e1 = gen_one(model, text, SEED, "bitwise_a")
        e2 = gen_one(model, text, SEED, "bitwise_b")
        h1 = sha256_of_file(Path(e1["path"]))
        h2 = sha256_of_file(Path(e2["path"]))
        identical = h1 == h2
        results["bitwise_check"] = {
            "text": text,
            "seed": SEED,
            "sha256_a": h1,
            "sha256_b": h2,
            "identical": identical,
            "f0_a": e1["f0_median_hz"],
            "f0_b": e2["f0_median_hz"],
        }
        save_results(results)
        print(f"  -> identical={identical}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
