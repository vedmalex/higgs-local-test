#!/usr/bin/env python3
"""M4-T2 measurement: does the already-implemented continuous-batching path
(`mlx_audio.tts.models.higgs_audio_v3.model.HiggsAudioV3.batch_generate`,
backed by `continuous_batching.py`'s `BatchKVCache`) actually speed up
multi-segment Higgs TTS on this M1/16GB machine, and where does memory become
the binding constraint? Issue #57, docs/research/audiobook/m4-plan.md §3, M4-T2.

This is a MEASUREMENT of existing code, not new development -- nothing in
`mlx_audio` is modified. One batch size is measured per process invocation
(repo isolation rule): run this script once per --batch-size value, in
separate `/usr/bin/time -l` invocations, never concurrently.

Design: a fixed pool of NUM_SEGMENTS independent short Russian sentences
(comparable in register/length to the existing samples/tts_ru.txt and
m4_boundary_check.py fixtures -- an audiobook is a sequence of short
independent utterances, not one long one, which is exactly the shape
batching is meant to exploit) is synthesized two ways depending on
--batch-size:

  --batch-size 1  -> NUM_SEGMENTS sequential model.generate() calls
                     (today's production path, i.e. the baseline).
  --batch-size N  -> NUM_SEGMENTS split into ceil(NUM_SEGMENTS / N) calls to
                     model.batch_generate(), each admitting up to N segments
                     at once through the continuous-batching machinery.

Both paths process the SAME NUM_SEGMENTS segments, so the comparison is
apples-to-apples: total wall time and total synthesized audio duration for
the identical workload, at different batch depths.

Metrics recorded (per docs/research/audiobook/m4-plan.md's honesty rules):
  - per-segment and aggregate RTF (wall / audio_duration)
  - aggregate throughput (audio_duration / wall) -- the actual optimization
    target; batching improves this, not necessarily single-segment RTF
  - peak_mlx (GiB) via mx.get_peak_memory() -- MLX's own allocator peak,
    covering the whole process (model weights + KV cache + activations)
  - machine state (uptime, vm.swapusage) captured immediately before and
    after the run and embedded in the JSON, per the project's
    "never trust a number without recording machine load next to it" rule
  - NOT reported: ru_maxrss / "maximum resident set size" -- confirmed
    elsewhere in this project to undercount on macOS; the results doc pulls
    "peak memory footprint" instead from the /usr/bin/time -l wrapper that
    invokes this script, never from getrusage inside the process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mlx_audio.audio_io import write as audio_write  # noqa: E402
from mlx_audio.tts.utils import load  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"

# Independent short Russian sentences, comparable register/length to
# samples/tts_ru.txt (the first 6 are that fixture's own non-empty lines,
# reused verbatim for comparability; segments 7-8 are new, added here only
# because batch size 8 needs 8 distinct segments and the existing fixture
# has 6 -- same style/length, documented as an extension in the results doc).
SEGMENTS = [
    "Сегодня мы проверяем работу системы синтеза речи Higgs Audio.",
    "Это локальная генерация на компьютере Apple M1.",
    "Шри Вриндаван — святое место паломничества.",
    "Шри Чайтанья Махапрабху учил повторению святых имён.",
    "Кришна. Радхарани. Шримад-Бхагаватам.",
    "Гопала Бхатта Госвами. Радха-Раман.",
    "Харе Кришна маха-мантра звучит каждый день на киртане.",
    "Вриндаванский лес хранит память об играх Кришны.",
]
NUM_SEGMENTS = len(SEGMENTS)

WARMUP_TEXT = "Это короткая прогревочная фраза перед замером."


def machine_state() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<error: {exc}>"

    return {
        "uptime": run(["uptime"]),
        "vm_swapusage": run(["sysctl", "vm.swapusage"]),
    }


def audio_duration(samples: int, sample_rate: int) -> float:
    return samples / sample_rate if sample_rate else 0.0


def run_baseline(model, out_dir: Path, max_new_tokens: int) -> list[dict]:
    """batch-size 1: NUM_SEGMENTS independent, sequential model.generate() calls."""
    per_segment = []
    for index, text in enumerate(SEGMENTS):
        t0 = time.perf_counter()
        results = list(model.generate(text=text, temperature=1.0, max_new_tokens=max_new_tokens))
        mx.eval(*[r.audio for r in results])
        wall = time.perf_counter() - t0
        if not results:
            raise RuntimeError(f"segment {index}: model.generate produced no result")
        sample_rate = results[0].sample_rate
        audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
        duration = audio_duration(len(audio), sample_rate)
        audio_write(str(out_dir / f"seg_{index}.wav"), audio, sample_rate)
        per_segment.append(
            {
                "segment": index,
                "chars": len(text),
                "wall_seconds": wall,
                "audio_duration_seconds": duration,
                "rtf": (wall / duration) if duration else None,
            }
        )
        print(
            f"  [batch=1] segment {index}: wall={wall:.3f}s audio={duration:.3f}s "
            f"rtf={(wall / duration) if duration else float('nan'):.3f}",
            flush=True,
        )
    return per_segment


def run_batched(model, out_dir: Path, batch_size: int, max_new_tokens: int) -> list[dict]:
    """batch-size N>1: NUM_SEGMENTS split into ceil(NUM_SEGMENTS/N) batch_generate calls."""
    per_segment = []
    for chunk_start in range(0, NUM_SEGMENTS, batch_size):
        chunk = SEGMENTS[chunk_start : chunk_start + batch_size]
        t0 = time.perf_counter()
        chunk_results = list(
            model.batch_generate(texts=chunk, temperature=1.0, max_new_tokens=max_new_tokens)
        )
        mx.eval(*[r.audio for r in chunk_results])
        chunk_wall = time.perf_counter() - t0
        chunk_results.sort(key=lambda r: r.sequence_idx)
        if len(chunk_results) != len(chunk):
            raise RuntimeError(
                f"chunk starting at {chunk_start}: expected {len(chunk)} results, got {len(chunk_results)}"
            )
        for offset, result in enumerate(chunk_results):
            index = chunk_start + offset
            audio = np.asarray(result.audio).reshape(-1)
            sample_rate = result.sample_rate
            duration = audio_duration(len(audio), sample_rate)
            audio_write(str(out_dir / f"seg_{index}.wav"), audio, sample_rate)
            # Individual wall time is not separable inside one batched call --
            # all rows share forward passes until each finishes and is evicted
            # (continuous_batching.py's _advance_active). Reporting the whole
            # chunk's wall time against this segment's own duration would
            # overstate its cost; instead this field is left null per-segment
            # and only the chunk-level and run-level aggregates are trusted.
            per_segment.append(
                {
                    "segment": index,
                    "chars": len(SEGMENTS[index]),
                    "chunk_wall_seconds": chunk_wall,
                    "chunk_size": len(chunk),
                    "audio_duration_seconds": duration,
                    "rtf": None,
                }
            )
        print(
            f"  [batch={batch_size}] chunk {chunk} ({len(chunk)} segs): "
            f"wall={chunk_wall:.3f}s",
            flush=True,
        )
    return per_segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 2, 4, 8))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to output/m4_batching/batch<N>/",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (ROOT / "output" / "m4_batching" / f"batch{args.batch_size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_before = machine_state()
    print("machine state before run:", json.dumps(state_before, ensure_ascii=False), flush=True)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    # Warm-up: absorbs first-call MLX graph construction so it does not
    # contaminate the measured segments (same convention as
    # docs/research/mojo-max/m4_stage_profile.py). Discarded, not scored.
    t0 = time.perf_counter()
    list(model.generate(text=WARMUP_TEXT, temperature=1.0, max_new_tokens=args.max_new_tokens))
    warmup_seconds = time.perf_counter() - t0
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded)", flush=True)

    mx.reset_peak_memory()
    run_start = time.perf_counter()
    if args.batch_size == 1:
        per_segment = run_baseline(model, out_dir, args.max_new_tokens)
    else:
        per_segment = run_batched(model, out_dir, args.batch_size, args.max_new_tokens)
    run_wall = time.perf_counter() - run_start

    total_audio = sum(s["audio_duration_seconds"] for s in per_segment)
    aggregate_rtf = (run_wall / total_audio) if total_audio else None
    throughput = (total_audio / run_wall) if run_wall else None
    peak_mlx_gib = mx.get_peak_memory() / (1024**3)

    state_after = machine_state()

    result = {
        "batch_size": args.batch_size,
        "num_segments": NUM_SEGMENTS,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "run_wall_seconds": run_wall,
        "total_audio_duration_seconds": total_audio,
        "aggregate_rtf": aggregate_rtf,
        "aggregate_throughput_audio_seconds_per_wall_second": throughput,
        "peak_mlx_gib": peak_mlx_gib,
        "per_segment": per_segment,
        "machine_state_before": state_before,
        "machine_state_after": state_after,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
