#!/usr/bin/env python3
"""Local Apple Silicon Qwen3-TTS test via the native MLX implementation.

This is a *separate* benchmark from the Higgs TTS test in `src/tts_test.py`:
different model family, different weights, reported separately and never
merged into the Higgs numbers.

The real `model.generate(...)` call always runs in its own child process
(`--worker`), so the runner never keeps a model resident. The child writes the
metrics JSON, the parent prints it. One mode per process, like the Higgs test.
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Real published MLX weights. Base = voice cloning (ref_audio + ref_text),
# CustomVoice = predefined speakers with an optional `instruct` style string.
BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
CUSTOM_VOICE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
DEFAULT_MODEL = {"basic": BASE_MODEL, "clone": BASE_MODEL, "custom_voice": CUSTOM_VOICE_MODEL}


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def peak_memory() -> dict:
    """Peak host RSS for this process plus the MLX device peak when available."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kibibytes.
    peak = {"peak_host_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}
    try:
        import mlx.core as mx

        peak["peak_mlx_memory_bytes"] = mx.get_peak_memory()
    except Exception:
        peak["peak_mlx_memory_bytes"] = None
    return peak


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("basic", "clone", "custom_voice"), default="basic",
                        help="basic: plain synthesis with the Base model. clone: Base model with "
                             "samples/reference.wav + samples/reference.txt. custom_voice: "
                             "CustomVoice model with a predefined --speaker and optional --instruct.")
    parser.add_argument("--output", type=Path, required=True, help="WAV file to write.")
    parser.add_argument("--metrics", type=Path, default=None, help="Write the metrics JSON here.")
    parser.add_argument("--model", default=None, help="Override the model repo id.")
    parser.add_argument("--text", type=Path, default=ROOT / "samples/tts_ru.txt",
                        help="File with the text to synthesize.")
    parser.add_argument("--speaker", default=None,
                        help="custom_voice only: predefined speaker name. Validated against the "
                             "model's own get_supported_speakers().")
    parser.add_argument("--instruct", default=None,
                        help="custom_voice only: documented emotion/style instruction string for "
                             "generate_custom_voice(instruct=...). Not a control tag.")
    parser.add_argument("--language", default="auto", help="custom_voice only: language code.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--worker", action="store_true",
                        help="Internal: run the model in this process instead of spawning a child.")
    return parser


def skipped(reason: str, mode: str, metrics_path: Path | None) -> dict:
    record = {"test": f"qwen3_tts_{mode}", "status": "SKIPPED", "reason": reason}
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_worker(args: argparse.Namespace) -> int:
    """The isolated process that actually loads the model and synthesizes."""
    import importlib.metadata as metadata

    import mlx.core as mx
    import numpy as np
    from mlx_audio.audio_io import write as audio_write
    from mlx_audio.tts.utils import load_model

    model_id = args.model or DEFAULT_MODEL[args.mode]
    diagnostics = {
        "test": f"qwen3_tts_{args.mode}", "model": model_id, "device": "mlx",
        "python": platform.python_version(),
        "mlx": metadata.version("mlx"), "mlx_audio": metadata.version("mlx-audio"),
        "machine": platform.machine(),
    }
    try:
        text = args.text.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{args.text} is empty")

        call_kwargs: dict = {"text": text, "temperature": args.temperature,
                             "max_tokens": args.max_tokens}
        if args.mode == "clone":
            ref_audio = ROOT / "samples/reference.wav"
            ref_text = ROOT / "samples/reference.txt"
            # Guarded again in the child: the parent guard is the fast path, this
            # one keeps the worker honest when invoked directly.
            if not ref_audio.exists() or not ref_text.exists():
                raise FileNotFoundError("samples/reference.wav and samples/reference.txt required")
            call_kwargs["ref_audio"] = str(ref_audio)
            call_kwargs["ref_text"] = ref_text.read_text(encoding="utf-8").strip()
            diagnostics["ref_audio"] = str(ref_audio)

        mx.reset_peak_memory()
        started = time.perf_counter()
        model = load_model(model_id)
        load_seconds = time.perf_counter() - started

        if args.mode == "custom_voice":
            speakers = list(model.get_supported_speakers())
            diagnostics["supported_speakers"] = speakers
            if args.speaker not in speakers:
                raise ValueError(f"--speaker must be one of {speakers}, got {args.speaker!r}")
            call_kwargs = {"text": text, "speaker": args.speaker, "language": args.language,
                           "instruct": args.instruct, "temperature": args.temperature,
                           "max_tokens": args.max_tokens}
            diagnostics.update({"speaker": args.speaker, "instruct": args.instruct,
                                "language": args.language})
            generate = model.generate_custom_voice
        else:
            generate = model.generate

        started = time.perf_counter()
        results = list(generate(**call_kwargs))
        generation_seconds = time.perf_counter() - started
        if not results:
            raise RuntimeError("MLX-Audio produced no generation result")

        sample_rate = results[0].sample_rate
        audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        audio_write(str(args.output), audio, sample_rate)
        duration = wav_duration(args.output)
        diagnostics.update({
            "status": "PASSED",
            "model_load_seconds": load_seconds,
            "processing_seconds": generation_seconds,
            "audio_duration_seconds": duration,
            "sample_rate": sample_rate,
            "rtf": generation_seconds / duration if duration else None,
            "segments": len(results),
            "output": str(args.output),
            **peak_memory(),
        })
    except Exception as exc:
        diagnostics.update({"status": "FAILED", "exception": repr(exc),
                            "traceback": traceback.format_exc(), **peak_memory()})
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return 1

    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.worker:
        return run_worker(args)

    if args.mode == "clone":
        missing = [str(p.relative_to(ROOT)) for p in
                   (ROOT / "samples/reference.wav", ROOT / "samples/reference.txt")
                   if not p.exists()]
        if missing:
            record = skipped(f"missing reference file(s): {', '.join(missing)}",
                             args.mode, args.metrics)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            return 0
    if args.mode == "custom_voice" and not args.speaker:
        record = skipped("custom_voice needs --speaker; run --mode custom_voice --speaker <name> "
                         "and the worker prints the model's supported speakers on a mismatch",
                         args.mode, args.metrics)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    child = [sys.executable, str(Path(__file__).resolve()), "--worker",
             "--mode", args.mode, "--output", str(args.output),
             "--text", str(args.text), "--max-tokens", str(args.max_tokens),
             "--temperature", str(args.temperature), "--language", args.language]
    if args.metrics:
        child += ["--metrics", str(args.metrics)]
    if args.model:
        child += ["--model", args.model]
    if args.speaker:
        child += ["--speaker", args.speaker]
    if args.instruct:
        child += ["--instruct", args.instruct]

    started = time.perf_counter()
    completed = subprocess.run(child, cwd=str(ROOT))
    wall = time.perf_counter() - started
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({
        "runner": f"qwen3_tts_{args.mode}",
        "child_exit_code": completed.returncode,
        "child_wall_seconds": wall,
        "child_peak_rss_bytes": child_rss if platform.system() == "Darwin" else child_rss * 1024,
    }, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
