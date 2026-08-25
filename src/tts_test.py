#!/usr/bin/env python3
import argparse
import json
import resource
import time
import wave
from pathlib import Path

import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load

MODEL_ID = "bosonai/higgs-tts-3-4b"
# Per issue #57 M4-T0 (owner blind-listening verdict, docs/research/audiobook/m4-sentiment-
# results.md §6): `emotion`/`prosody` tags below are confirmed audibly distinguishable.
# `<|style:whispering|>` is confirmed NOT to produce whispering — it renders as quieter speech,
# not breathy/voiceless phonation. Kept here only to exercise the control-tag code path, not as a
# demonstrated-working whisper example.
CONTROL_TEXT = """<|emotion:contentment|><|prosody:speed_slow|>Начнём спокойно и внимательно. <|prosody:pause|> Теперь голос становится выразительнее. <|emotion:enthusiasm|><|prosody:expressive_high|>Это важная и радостная проверка! <|prosody:long_pause|><|style:whispering|>А теперь тихое завершение."""


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("basic", "controls", "clone"), default="basic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default=None,
                        help="Override the text for this generation (research use, e.g. "
                             "issue #57 M4-T0 sentiment-tag baseline probes). Ignored for "
                             "--mode clone, which always uses the cloning reference text.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    text = (root / "samples/tts_ru.txt").read_text(encoding="utf-8").strip()
    kwargs = {}
    if args.mode == "controls":
        text = CONTROL_TEXT
    elif args.mode == "clone":
        ref_audio, ref_text = root / "samples/reference.wav", root / "samples/reference.txt"
        if not ref_audio.exists() or not ref_text.exists():
            print("Voice cloning: SKIPPED")
            return
        kwargs = {"ref_audio": str(ref_audio), "ref_text": ref_text.read_text(encoding="utf-8").strip()}

    if args.text is not None and args.mode != "clone":
        text = args.text

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    results = list(model.generate(text=text, temperature=1.0, max_new_tokens=4096, **kwargs))
    generation_seconds = time.perf_counter() - started
    if not results:
        raise RuntimeError("MLX-Audio produced no generation result")
    sample_rate = results[0].sample_rate
    audio = np.concatenate([np.asarray(result.audio).reshape(-1) for result in results])
    audio_write(str(args.output), audio, sample_rate)
    duration = wav_duration(args.output)
    metrics = {
        "test": f"tts_{args.mode}", "model": MODEL_ID, "device": "mlx",
        "model_load_seconds": load_seconds, "processing_seconds": generation_seconds,
        "audio_duration_seconds": duration,
        "rtf": generation_seconds / duration if duration else None,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": str(args.output),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
