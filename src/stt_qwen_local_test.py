#!/usr/bin/env python3
"""Local Apple Silicon Qwen3-ASR test via the native MLX implementation.

A *separate* benchmark from the Higgs STT test in `src/stt_test.py`: different
model family, different weights, reported separately and never merged into the
Higgs numbers.

The real transcription runs in its own child process (`--worker`) so the runner
never keeps a model resident. Cyrillic output is written through untouched — no
transliteration, no LLM repair (AGENTS.md rule 6).

The WER mechanism is the one `src/stt_test.py` already uses: `jiwer.wer` over
lowercased reference/hypothesis, with `--reference builtin` selecting the same
repository fixture text. WER is reported only when a reference is supplied.
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"


def builtin_reference() -> str:
    """The repository's Russian fixture text, owned by src/stt_test.py.

    Read out of that file rather than copied here, so the two benchmarks compare
    against the exact same string — a divergent copy would silently produce
    incomparable WER. Parsed with `ast` instead of imported because stt_test.py
    pulls in the torch/transformers stack that lives only in `.venv-stt`.
    """
    import ast

    source = (Path(__file__).resolve().parent / "stt_test.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REFERENCE" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("REFERENCE not found in src/stt_test.py")


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
    parser.add_argument("--audio", type=Path, default=ROOT / "samples/stt_ru.wav")
    parser.add_argument("--output", type=Path, required=True, help="Transcript text file.")
    parser.add_argument("--metrics", type=Path, default=None, help="Write the metrics JSON here.")
    parser.add_argument("--model", default=MODEL_ID, help="Override the model repo id.")
    parser.add_argument("--language", default="ru", help="Language code passed to the model.")
    parser.add_argument("--reference", default=None,
                        help="Reference transcript for WER: a file path, or the literal 'builtin' "
                             "for the repository's Russian fixture. Omit it and no WER is "
                             "reported — comparing arbitrary audio against the fixture produces a "
                             "number that measures nothing.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--worker", action="store_true",
                        help="Internal: run the model in this process instead of spawning a child.")
    return parser


def resolve_reference(spec: str | None) -> tuple[str | None, str | None]:
    if spec is None:
        return None, None
    if spec == "builtin":
        return builtin_reference(), "builtin"
    path = Path(spec)
    return path.read_text(encoding="utf-8").strip(), str(path)


def run_worker(args: argparse.Namespace) -> int:
    """The isolated process that actually loads the model and transcribes."""
    import importlib.metadata as metadata

    import mlx.core as mx
    # MLX-Audio's own loader, so this benchmark needs nothing beyond `.venv-tts`.
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.stt.utils import load_model

    diagnostics = {
        "test": "qwen3_asr", "model": args.model, "device": "mlx",
        "language": args.language, "python": platform.python_version(),
        "mlx": metadata.version("mlx"), "mlx_audio": metadata.version("mlx-audio"),
        "machine": platform.machine(),
    }
    try:
        # Measured on the file as it is, before the model does its own resampling,
        # so the RTF denominator is the true wall-clock length of the recording.
        audio, sample_rate = audio_read(args.audio, dtype="float32")
        channels = 1 if audio.ndim == 1 else audio.shape[-1]
        duration = audio.shape[0] / sample_rate
        diagnostics.update({"audio": str(args.audio), "input_sample_rate": sample_rate,
                            "input_channels": channels})

        mx.reset_peak_memory()
        started = time.perf_counter()
        model = load_model(args.model)
        load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        result = model.generate(str(args.audio), language=args.language,
                                max_tokens=args.max_tokens)
        processing = time.perf_counter() - started

        transcript = result.text.strip()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(transcript + "\n", encoding="utf-8")

        reference, reference_label = resolve_reference(args.reference)
        wer_value, wer_note = None, None
        if reference:
            try:
                from jiwer import wer

                wer_value = wer(reference.lower(), transcript.lower())
            except ImportError:
                wer_note = ("jiwer is not installed in this environment; install it to measure "
                            "WER (pip install jiwer)")
        else:
            wer_note = "no reference transcript supplied; WER not measured"

        diagnostics.update({
            "status": "PASSED",
            "model_load_seconds": load_seconds,
            "processing_seconds": processing,
            "audio_duration_seconds": duration,
            "rtf": processing / duration if duration else None,
            "wer": wer_value, "wer_reference": reference_label, "wer_note": wer_note,
            "transcript": transcript, "output": str(args.output),
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

    if not args.audio.exists():
        record = {"test": "qwen3_asr", "status": "SKIPPED",
                  "reason": f"missing audio file: {args.audio}"}
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    child = [sys.executable, str(Path(__file__).resolve()), "--worker",
             "--audio", str(args.audio), "--output", str(args.output),
             "--model", args.model, "--language", args.language,
             "--max-tokens", str(args.max_tokens)]
    if args.metrics:
        child += ["--metrics", str(args.metrics)]
    if args.reference:
        child += ["--reference", args.reference]

    started = time.perf_counter()
    completed = subprocess.run(child, cwd=str(ROOT))
    wall = time.perf_counter() - started
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({
        "runner": "qwen3_asr",
        "child_exit_code": completed.returncode,
        "child_wall_seconds": wall,
        "child_peak_rss_bytes": child_rss if platform.system() == "Darwin" else child_rss * 1024,
    }, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
