#!/usr/bin/env python3
"""M4-T2 follow-up: measure the vocoder's SHARE of wall time under batching.

Issue #57. Tests the hypothesis recorded in the task that spawned this
script: batching (`docs/research/audiobook/m4-batching-results.md`, PR #105)
speeds up the shared autoregressive (AR) loop, but `Model._decode_audio`
(the codec vocoder) is called once PER FINISHED SEGMENT regardless of batch
depth -- confirmed by reading
`mlx_audio/tts/models/higgs_audio_v3/model.py:714-722` inside
`batch_generate`: after the fully-batched AR loop finishes for every row,
the function iterates `for state in states: audio = self._decode_audio(...)`
one row at a time. So the vocoder's aggregate wall-clock cost per N segments
should stay roughly constant across batch sizes, while the AR loop's
aggregate cost shrinks with batching -- meaning the vocoder's PERCENTAGE
SHARE of total wall time should grow roughly in proportion to the batching
speedup. This script measures that directly instead of extrapolating from
the batch=1 stage-profile numbers in `../mojo-max/m4-stage-profile-results.md`
(RTF 4.26, codec_decode 3.76%) and the batch=8 aggregate RTF in
`m4-batching-results.md` (1.14, no stage breakdown).

Reuses, unmodified:
  - `instrument()` / `deinstrument()` from `../mojo-max/m4_stage_profile.py`
    (PR #104) -- these monkey-patch `model.backbone`, `_build_prompt_embeddings`,
    `_decode_audio`, `_apply_fades` and work identically whether the model is
    driven through `model.generate()` (batch=1) or `model.batch_generate()`
    (batch>1), because `batch_generate` calls the exact same four methods on
    `self` (see `model.py:634,662-711,715,717`).
  - `SEGMENTS`, `MODEL_ID`, `machine_state()`, `audio_duration()`, `WARMUP_TEXT`
    from `m4_batching_bench.py` (PR #105) -- the identical 8-segment Russian
    material the batching benchmark used, so batch=1/batch=8 stage-profile
    numbers here are directly comparable to that benchmark's aggregate RTF.

Run ONE batch size per process (project isolation rule -- never run two
batch sizes concurrently):

    /usr/bin/time -l .venv-tts/bin/python \\
        docs/research/audiobook/m4_batching_stage_profile.py --batch-size 1 \\
        | tee logs/m4_batching_stage_profile_batch1.log
    /usr/bin/time -l .venv-tts/bin/python \\
        docs/research/audiobook/m4_batching_stage_profile.py --batch-size 8 \\
        | tee logs/m4_batching_stage_profile_batch8.log

Methodology notes (same conventions as `m4_stage_profile.py` and
`m4_batching_bench.py`):
  * MLX is lazy -- `mx.eval()` is called at every stage boundary via the
    reused `instrument()` wrappers, and again on the produced audio arrays
    before any timer stops.
  * `ar_glue_sampling_embedding` is the remainder of `wall_total` after
    subtracting every explicitly-timed stage (sampling/embedding/control-flow
    glue between backbone calls); a negative remainder prints a warning
    instead of being silently clamped (same fix as the m4_stage_profile.py
    2026-08-25 audit).
  * `peak_mlx` (`mx.get_peak_memory()`) is reported; `peak_footprint` must be
    read from the `/usr/bin/time -l` wrapper's "peak memory footprint" line
    in the invoking shell, not from anything inside this process --
    `ru_maxrss`/"maximum resident set size" is never cited as a memory figure
    in this project (confirmed to undercount on macOS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "research" / "mojo-max"))
sys.path.insert(0, str(ROOT / "docs" / "research" / "audiobook"))

from mlx_audio.audio_io import write as audio_write  # noqa: E402
from mlx_audio.tts.utils import load  # noqa: E402

from m4_stage_profile import instrument, deinstrument  # noqa: E402
from m4_batching_bench import (  # noqa: E402
    MODEL_ID,
    WARMUP_TEXT,
    audio_duration,
    machine_state,
)


def load_segments(segments_file: Path | None) -> list[str]:
    """Default: the exact 8-segment pool `m4_batching_bench.py` measured.

    `--segments-file` (Task B use, paragraph-length material) overrides this
    with blank-line-separated paragraph blocks from an external text file --
    an additive CLI option, not a change to `m4_batching_bench.py`'s own
    `SEGMENTS` list or measurement logic.
    """
    if segments_file is None:
        from m4_batching_bench import SEGMENTS  # noqa: E402

        return list(SEGMENTS)
    raw = segments_file.read_text(encoding="utf-8").strip()
    blocks = [block.strip().replace("\n", " ") for block in raw.split("\n\n") if block.strip()]
    return blocks


def run_batch1(model, segments: list[str], out_dir: Path, max_new_tokens: int) -> list[dict]:
    per_segment = []
    for index, text in enumerate(segments):
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
            {"segment": index, "chars": len(text), "wall_seconds": wall, "audio_duration_seconds": duration}
        )
        print(f"  [batch=1] segment {index}: wall={wall:.3f}s audio={duration:.3f}s", flush=True)
    return per_segment


def run_batch_n(model, segments: list[str], out_dir: Path, batch_size: int, max_new_tokens: int) -> list[dict]:
    per_segment = []
    n = len(segments)
    for chunk_start in range(0, n, batch_size):
        chunk = segments[chunk_start : chunk_start + batch_size]
        t0 = time.perf_counter()
        chunk_results = list(model.batch_generate(texts=chunk, temperature=1.0, max_new_tokens=max_new_tokens))
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
            per_segment.append({"segment": index, "chars": len(segments[index]), "audio_duration_seconds": duration})
        print(f"  [batch={batch_size}] chunk of {len(chunk)}: wall={chunk_wall:.3f}s", flush=True)
    return per_segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--segments-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    segments = load_segments(args.segments_file)
    tag = args.segments_file.stem if args.segments_file else "short"
    out_dir = args.output_dir or (
        ROOT / "output" / "m4_batching_stage_profile" / f"{tag}_batch{args.batch_size}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state_before = machine_state()
    print("machine state before run:", json.dumps(state_before, ensure_ascii=False), flush=True)
    print(f"num_segments={len(segments)} tag={tag}", flush=True)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    # Warm-up: un-instrumented, discarded -- same convention as
    # m4_batching_bench.py (absorbs first-call MLX graph construction).
    t0 = time.perf_counter()
    list(model.generate(text=WARMUP_TEXT, temperature=1.0, max_new_tokens=args.max_new_tokens))
    warmup_seconds = time.perf_counter() - t0
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded)", flush=True)

    stats: dict = {}
    originals = instrument(model, stats)
    mx.reset_peak_memory()
    wall_start = time.perf_counter()
    try:
        if args.batch_size == 1:
            per_segment = run_batch1(model, segments, out_dir, args.max_new_tokens)
        else:
            per_segment = run_batch_n(model, segments, out_dir, args.batch_size, args.max_new_tokens)
    finally:
        deinstrument(model, originals)
    wall_total = time.perf_counter() - wall_start

    total_audio = sum(s["audio_duration_seconds"] for s in per_segment)
    aggregate_rtf = (wall_total / total_audio) if total_audio else None

    ar_prefill = stats.get("ar_prefill", 0.0)
    ar_backbone = stats.get("ar_backbone_steps", 0.0)
    prompt_prep = stats.get("prompt_prep", 0.0)
    codec_decode = stats.get("codec_decode", 0.0)
    postprocess = stats.get("postprocess", 0.0)
    accounted = ar_prefill + ar_backbone + prompt_prep + codec_decode + postprocess
    ar_glue = wall_total - accounted
    if ar_glue < 0:
        print(
            f"WARNING: negative ar_glue ({ar_glue:.6f}s) -- stage timers over-account "
            "for wall_total; treat this run's breakdown as suspect.",
            flush=True,
        )

    stage_seconds = {
        "prompt_prep": prompt_prep,
        "ar_prefill": ar_prefill,
        "ar_backbone_steps": ar_backbone,
        "ar_glue_sampling_embedding": ar_glue,
        "codec_decode": codec_decode,
        "postprocess": postprocess,
    }
    stage_pct = {k: (100.0 * v / wall_total if wall_total else 0.0) for k, v in stage_seconds.items()}
    ar_loop_total = ar_prefill + ar_backbone + ar_glue

    result = {
        "batch_size": args.batch_size,
        "tag": tag,
        "num_segments": len(segments),
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "wall_total_seconds": wall_total,
        "total_audio_duration_seconds": total_audio,
        "aggregate_rtf": aggregate_rtf,
        "ar_frame_count": stats.get("ar_backbone_steps_count", 0),
        "stage_seconds": stage_seconds,
        "stage_pct_of_wall": stage_pct,
        "ar_loop_total_seconds": ar_loop_total,
        "ar_loop_pct_of_wall": (100.0 * ar_loop_total / wall_total if wall_total else 0.0),
        "peak_mlx_gib": mx.get_peak_memory() / (1024**3),
        "per_segment": per_segment,
        "machine_state_before": state_before,
        "machine_state_after": machine_state(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
