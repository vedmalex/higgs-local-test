#!/usr/bin/env python3
"""M4 stage profile: break Higgs TTS 3 (MLX) generation down by pipeline stage.

Issue #57, milestone M4. Verifies (does not assume) whether the Amdahl's-law
hypothesis holds: that the codec vocoder decode (`Model._decode_audio`,
called exactly once after the autoregressive loop finishes — see
`mlx_audio/tts/models/higgs_audio_v3/model.py:341-352,793-817`) is a small
share of total wall time compared to the per-frame autoregressive Talker loop.

Run with the project's TTS venv, alone, with nothing else heavy running:

    /usr/bin/time -l .venv-tts/bin/python docs/research/mojo-max/m4_stage_profile.py \
        | tee logs/m4_stage_profile.log

Methodology notes (see AGENTS.md / issue #57 for the full rationale):
  * MLX is lazy. Every stage boundary below calls `mx.eval()` on the boundary
    tensor(s) before stopping the stage's clock, otherwise work silently
    leaks into whichever stage happens to force the sync later.
  * The autoregressive backbone (`Model.backbone`, a `Qwen3Model`) is wrapped
    so that its single big "prefill" call (processing the whole prompt) is
    timed separately from each single-token decode-step call inside the
    per-frame loop. Sampling/embedding glue between backbone calls is timed
    as a third AR-loop bucket ("ar_glue") by difference.
  * A short throwaway warm-up generation runs once after model load and is
    discarded, to separate one-time MLX graph/compile overhead from the
    short/long measurements that follow.
  * `mx.get_peak_memory()` (MLX's own device-buffer accounting) and the
    process's peak RSS (from `/usr/bin/time -l`, wrapping this whole script)
    are reported as two separate, never-merged numbers, per AGENTS.md.
  * RTF here follows the project's existing convention in `src/tts_test.py`:
    processing_seconds / audio_duration_seconds.
  * `write_seconds` (writing the WAV file) is measured AFTER `wall_total`'s
    clock already stopped and is reported as its own field, never folded into
    `wall_total` or any of its stage percentages. An earlier version of this
    script folded it into `postprocess` before computing `ar_glue` by
    difference, which let accounted-for time exceed `wall_total` and made a
    `max(0.0, ...)` clamp mask the resulting negative remainder as a
    misleading "0.00%" ar_glue bucket. Audited 2026-08-25 (issue #57); see
    `m4-stage-profile-results.md` for the corrected published numbers. A
    negative `ar_glue` now prints a warning instead of being silently
    clamped.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load

MODEL_ID = "bosonai/higgs-tts-3-4b"
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "output" / "m4_stage_profile"

WARMUP_TEXT = "Привет."
SHORT_TEXT = "Сегодня мы проверяем работу системы синтеза речи Higgs Audio."
LONG_TEXT = (ROOT / "samples/tts_ru.txt").read_text(encoding="utf-8").strip()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


class _BackboneProfiler:
    """Wraps `Model.backbone` to split prefill vs. per-frame decode calls.

    A plain instance attribute cannot override `__call__` (dunder lookup
    bypasses the instance dict), so this replaces `model.backbone` itself
    with a proxy object whose *class* defines `__call__`. Every other
    attribute access (`.layers`, etc.) is forwarded to the wrapped module.
    """

    def __init__(self, wrapped, stats: dict):
        self._wrapped = wrapped
        self._stats = stats

    def __call__(self, inputs, cache=None, input_embeddings=None):
        is_prefill = inputs.shape[1] > 1
        t0 = time.perf_counter()
        out = self._wrapped(inputs, cache=cache, input_embeddings=input_embeddings)
        mx.eval(out)
        dt = time.perf_counter() - t0
        key = "ar_prefill" if is_prefill else "ar_backbone_steps"
        self._stats[key] += dt
        self._stats[key + "_count"] += 1
        return out

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def instrument(model, stats: dict) -> dict:
    """Monkey-patch one model instance in place.

    Returns a dict of the original (pre-patch) attributes so a caller can
    remove the wrappers afterwards via `deinstrument()`. Without this,
    calling `instrument()` again on the same model instance (e.g. once per
    case, plus once for the warm-up) stacks a fresh wrapper layer on top of
    the previous one every time: `model.backbone` becomes a
    `_BackboneProfiler` wrapping a `_BackboneProfiler` wrapping the real
    module, and likewise for `_decode_audio` / `_apply_fades` /
    `_build_prompt_embeddings`. Audited 2026-08-25 (issue #57): on the
    numbers this script has published so far, the outer-most layer always
    writes into the caller's live `stats` dict and correctly wraps (and
    therefore still times) the inner layers, so the stacking did not corrupt
    any published stage time — the extra per-call Python-attribute-access
    overhead from the stacked layers is on the order of single-digit
    milliseconds, well below this script's own measurement noise. It is
    nonetheless fragile: anything that loops over more cases or
    re-instruments this model instance in the future would keep growing the
    stack indefinitely. Restoring originals after each case removes that
    risk going forward.
    """

    stats.setdefault("ar_prefill", 0.0)
    stats.setdefault("ar_prefill_count", 0)
    stats.setdefault("ar_backbone_steps", 0.0)
    stats.setdefault("ar_backbone_steps_count", 0)
    stats.setdefault("prompt_prep", 0.0)
    stats.setdefault("codec_decode", 0.0)
    stats.setdefault("postprocess", 0.0)

    originals = {
        "backbone": model.backbone,
        "_build_prompt_embeddings": model._build_prompt_embeddings,
        "_decode_audio": model._decode_audio,
        "_apply_fades": model._apply_fades,
    }

    model.backbone = _BackboneProfiler(model.backbone, stats)

    original_build_prompt = model._build_prompt_embeddings

    def timed_build_prompt(text, references):
        t0 = time.perf_counter()
        embeds, n_tokens = original_build_prompt(text, references)
        mx.eval(embeds)
        stats["prompt_prep"] += time.perf_counter() - t0
        return embeds, n_tokens

    model._build_prompt_embeddings = timed_build_prompt

    original_decode_audio = model._decode_audio

    def timed_decode_audio(delayed_rows):
        t0 = time.perf_counter()
        audio = original_decode_audio(delayed_rows)
        mx.eval(audio)
        stats["codec_decode"] += time.perf_counter() - t0
        return audio

    model._decode_audio = timed_decode_audio

    original_apply_fades = model._apply_fades

    def timed_apply_fades(audio, *, fade_in_ms, fade_out_ms):
        t0 = time.perf_counter()
        out = original_apply_fades(audio, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms)
        mx.eval(out)
        stats["postprocess"] += time.perf_counter() - t0
        return out

    model._apply_fades = timed_apply_fades

    return originals


def deinstrument(model, originals: dict) -> None:
    """Undo `instrument()`: restore the model's original attributes.

    Call this after each case/warm-up so consecutive `instrument()` calls on
    the same model instance never stack wrapper layers (see `instrument()`'s
    docstring for why that matters and why it was harmless so far).
    """

    model.backbone = originals["backbone"]
    model._build_prompt_embeddings = originals["_build_prompt_embeddings"]
    model._decode_audio = originals["_decode_audio"]
    model._apply_fades = originals["_apply_fades"]


def run_case(model, name: str, text: str, output_path: Path) -> dict:
    stats: dict = {}
    originals = instrument(model, stats)

    try:
        mx.reset_peak_memory()
        wall_start = time.perf_counter()
        results = list(model.generate(text=text, temperature=1.0, max_new_tokens=4096))
        wall_total = time.perf_counter() - wall_start
    finally:
        # Restore the model's un-wrapped attributes immediately so a later
        # `instrument()` call (the next case, or a future extra run) never
        # stacks a new wrapper layer on top of this one. See `instrument()`'s
        # docstring (issue #57 audit, 2026-08-25) for why the previous
        # never-restored version was fragile but did not corrupt the numbers
        # already published in `m4-stage-profile-results.md`.
        deinstrument(model, originals)

    if not results:
        raise RuntimeError(f"no generation result for case {name!r}")

    sample_rate = results[0].sample_rate
    audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    audio_write(str(output_path), audio, sample_rate)
    write_seconds = time.perf_counter() - write_start
    duration = wav_duration(output_path)

    ar_backbone = stats["ar_backbone_steps"]
    ar_prefill = stats["ar_prefill"]
    prompt_prep = stats["prompt_prep"]
    codec_decode = stats["codec_decode"]
    postprocess = stats["postprocess"]
    # Sampling / audio-code embedding / control-flow glue between backbone
    # calls inside the AR loop, measured by difference against `wall_total`
    # so no MLX op inside `model.generate()` is left unaccounted for.
    #
    # `write_seconds` (the WAV write) happens strictly *after* `wall_total`
    # already stopped (`model.generate()` returned), so it must NOT be added
    # to any stage that gets subtracted from `wall_total` — doing that here
    # previously (folding it into `postprocess` before computing `ar_glue`)
    # let accounted time exceed `wall_total` by exactly `write_seconds` and a
    # `max(0.0, ...)` silently clamped the resulting negative `ar_glue` to
    # zero, which read as "glue costs nothing" when it was really "the ledger
    # doesn't balance." Audited 2026-08-25 (issue #57): fixed here by keeping
    # `write_seconds` as its own line item outside of `wall_total`/`ar_glue`
    # accounting entirely — this is the more honest choice of the two
    # options, since `wall_total` is meant to mirror `model.generate()`'s own
    # cost (matching `src/tts_test.py`'s RTF convention), and the WAV write is
    # profiling-script overhead, not part of what a real caller measures as
    # "how long did generation take." No underlying number here is
    # recomputed — this only changes which bucket each already-measured
    # second is reported under.
    accounted = ar_prefill + ar_backbone + prompt_prep + codec_decode + postprocess
    ar_glue = wall_total - accounted
    if ar_glue < 0:
        # A negative remainder means the stage timers accounted for MORE than
        # the wall clock measured — a bug in this script's own bookkeeping,
        # not a valid "zero-cost" stage. Surface it instead of masking it.
        print(
            f"WARNING: case {name!r} has negative ar_glue ({ar_glue:.6f}s) — "
            "stage timers over-account for wall_total; treat this run's "
            "breakdown as suspect and investigate before trusting it.",
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

    return {
        "case": name,
        "text_chars": len(text),
        "audio_duration_seconds": duration,
        "wall_total_seconds": wall_total,
        "rtf": wall_total / duration if duration else None,
        "ar_frame_count": stats["ar_backbone_steps_count"],
        "stage_seconds": stage_seconds,
        "stage_pct_of_wall": stage_pct,
        # Reported separately, NOT inside wall_total/stage_seconds/stage_pct:
        # this happens after model.generate() already returned, so folding it
        # into any stage that participates in the wall_total ledger would
        # make accounted-for time exceed wall_total again (see comment above).
        "write_seconds": write_seconds,
        "mlx_peak_memory_bytes": mx.get_peak_memory(),
        "output": str(output_path),
    }


def main() -> None:
    print(f"host: {platform.platform()}", flush=True)
    print("machine load before run:", flush=True)
    subprocess.run(["uptime"], check=False)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    # Warm-up: absorbs first-call MLX graph construction / lazy compile so it
    # does not contaminate the short/long measurements below. The warm-up's
    # own generated audio is discarded (this is a throwaway run), but its
    # stage breakdown is not thrown away — see the print below (issue #57
    # audit P5): those numbers are the direct, measured evidence for how much
    # of the warm-up's cost is one-time graph/kernel compilation overhead
    # versus ordinary per-frame AR cost, which is exactly the question the
    # cold-start-vs-6.56-RTF discrepancy discussion in
    # `m4-stage-profile-results.md` needs a third data point for.
    warmup_stats: dict = {}
    warmup_originals = instrument(model, warmup_stats)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    list(model.generate(text=WARMUP_TEXT, temperature=1.0, max_new_tokens=4096))
    warmup_seconds = time.perf_counter() - t0
    deinstrument(model, warmup_originals)
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded audio, stage breakdown kept below)", flush=True)
    warmup_frame_count = warmup_stats.get("ar_backbone_steps_count", 0)
    warmup_stage_seconds = {
        "prompt_prep": warmup_stats.get("prompt_prep", 0.0),
        "ar_prefill": warmup_stats.get("ar_prefill", 0.0),
        "ar_backbone_steps": warmup_stats.get("ar_backbone_steps", 0.0),
        "codec_decode": warmup_stats.get("codec_decode", 0.0),
        "postprocess": warmup_stats.get("postprocess", 0.0),
    }
    warmup_accounted = sum(warmup_stage_seconds.values())
    warmup_stage_seconds["ar_glue_sampling_embedding_or_unaccounted"] = warmup_seconds - warmup_accounted
    warmup_avg_frame_ms = (
        1000.0 * warmup_stats["ar_backbone_steps"] / warmup_frame_count if warmup_frame_count else None
    )
    print(
        json.dumps(
            {
                "warmup_seconds": warmup_seconds,
                "warmup_frame_count": warmup_frame_count,
                "warmup_stage_seconds": warmup_stage_seconds,
                "warmup_avg_backbone_step_ms": warmup_avg_frame_ms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    results = []
    for name, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
        out_path = OUTPUT_DIR / f"m4_{name}.wav"
        result = run_case(model, name, text, out_path)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    ideal_vocoder_savings = {
        r["case"]: r["stage_pct_of_wall"]["codec_decode"] for r in results
    }
    print(
        "ideal_instant_vocoder_time_saved_pct_of_total="
        + json.dumps(ideal_vocoder_savings, ensure_ascii=False),
        flush=True,
    )
    print(
        json.dumps(
            {"model_load_seconds": load_seconds, "warmup_seconds": warmup_seconds, "results": results},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
