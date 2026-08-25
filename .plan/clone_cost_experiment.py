#!/usr/bin/env python3
"""Issue #57 follow-up: price of cloning with a FIXED reference sample.

Measures, on the SAME reference (samples/reference.wav / .txt, 7.4s), sequentially,
one heavy call at a time, with mx.eval() at every boundary already guaranteed by
model.generate()/batch_generate() internals (they mx.eval() audio + prompt embeds):

  1. basic single-segment generate(), no ref               (compare to README 6.56)
  2. clone single-segment generate(), ref_audio=path        (compare to README 7.73)
  3. clone single-segment generate(), ref_audio_codes=precomputed (cached-ref cost)
  4. batch_generate() of N=4 texts, NO ref                  (batching alone)
  5. batch_generate() of N=4 texts, SAME ref_audio for all rows (batching + cloning:
     does it even run, and what does it cost)
  6. batch_generate() of N=4 texts, SAME precomputed ref_audio_codes for all rows
     (batching + cached cloning)

Each step's peak MLX memory (mx.get_peak_memory()) is reset at the very start of that
step via mx.metal / mx.reset_peak_memory if available, so numbers are per-step not
cumulative. Also record RSS is explicitly NOT used as the peak-memory citation per
instruction (peak_mlx / peak_footprint / weights are the three numbers to report;
weights-on-disk is a static du -sh of the HF cache snapshot, done separately in shell).

Writes to .plan/clone_cost_results.json incrementally.
"""
import json
import time
import wave
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / ".plan" / "clone_cost_results.json"
REF_AUDIO = ROOT / "samples/reference.wav"
REF_TEXT_PATH = ROOT / "samples/reference.txt"
REF_TEXT = REF_TEXT_PATH.read_text(encoding="utf-8").strip()

MODEL_ID = "bosonai/higgs-tts-3-4b"

BASIC_TEXT = "Сегодня прекрасная погода для прогулки в парке, и хочется просто дышать свежим воздухом."

BATCH_TEXTS = [
    "Сегодня прекрасная погода для прогулки в парке.",
    "Компьютер медленно загружался этим утром.",
    "Она приготовила вкусный борщ на обед.",
    "Дети играли во дворе до самого вечера.",
]


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results):
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def reset_peak():
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass


def mem_snapshot():
    return {
        "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 3),
        "active_mlx_gb": round(mx.get_active_memory() / 1e9, 3),
        "cache_mlx_gb": round(mx.get_cache_memory() / 1e9, 3),
    }


def audio_duration(results_list):
    total_samples = sum(int(np.asarray(r.audio).size) for r in results_list)
    sr = results_list[0].sample_rate
    return total_samples / sr if sr else 0.0, sr


def run_step_generate(name, results, kwargs, text=BASIC_TEXT):
    if name in results:
        print(f"skip {name} (already done)", flush=True)
        return
    print(f"=== {name} ===", flush=True)
    reset_peak()
    started = time.perf_counter()
    gen_results = list(_MODEL.generate(text=text, temperature=1.0, max_new_tokens=4096, **kwargs))
    elapsed = time.perf_counter() - started
    dur, sr = audio_duration(gen_results)
    rtf = elapsed / dur if dur else None
    entry = {
        "elapsed_s": round(elapsed, 2),
        "audio_duration_s": round(dur, 2),
        "rtf": round(rtf, 3) if rtf else None,
        "sample_rate": sr,
        **mem_snapshot(),
    }
    print(f"  elapsed={entry['elapsed_s']}s audio={entry['audio_duration_s']}s RTF={entry['rtf']} "
          f"peak_mlx={entry['peak_mlx_gb']}GB", flush=True)
    results[name] = entry
    save_results(results)


def run_step_batch(name, results, kwargs, texts=BATCH_TEXTS):
    if name in results:
        print(f"skip {name} (already done)", flush=True)
        return
    print(f"=== {name} ===", flush=True)
    reset_peak()
    started = time.perf_counter()
    try:
        gen_results = list(_MODEL.batch_generate(texts=texts, temperature=1.0, max_new_tokens=4096, **kwargs))
        elapsed = time.perf_counter() - started
        dur, sr = audio_duration(gen_results)
        rtf = elapsed / dur if dur else None
        entry = {
            "status": "ok",
            "n_texts": len(texts),
            "elapsed_s": round(elapsed, 2),
            "total_audio_duration_s": round(dur, 2),
            "rtf_aggregate": round(rtf, 3) if rtf else None,
            "sample_rate": sr,
            **mem_snapshot(),
        }
        print(f"  OK elapsed={entry['elapsed_s']}s total_audio={entry['total_audio_duration_s']}s "
              f"RTF_aggregate={entry['rtf_aggregate']} peak_mlx={entry['peak_mlx_gb']}GB", flush=True)
    except Exception as exc:  # noqa: BLE001 -- must record the failure, not crash the run
        elapsed = time.perf_counter() - started
        entry = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(elapsed, 2),
            **mem_snapshot(),
        }
        print(f"  FAILED after {entry['elapsed_s']}s: {entry['error_type']}: {entry['error']}", flush=True)
    results[name] = entry
    save_results(results)


def main():
    global _MODEL
    print("Loading model...", flush=True)
    started = time.perf_counter()
    _MODEL = load(MODEL_ID, model_type="higgs_audio_v3")
    print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)

    results = load_results()

    # Precompute reference codes once, up front, to reuse in the "cached" steps.
    if "ref_codes_precompute" not in results:
        print("=== ref_codes_precompute ===", flush=True)
        reset_peak()
        started = time.perf_counter()
        ref_codes = _MODEL.encode_reference_audio(str(REF_AUDIO))
        mx.eval(ref_codes)
        elapsed = time.perf_counter() - started
        entry = {"elapsed_s": round(elapsed, 2), "codes_shape": list(ref_codes.shape), **mem_snapshot()}
        print(f"  encode_reference_audio: {entry['elapsed_s']}s, shape={entry['codes_shape']}", flush=True)
        results["ref_codes_precompute"] = entry
        save_results(results)
        np.save(str(ROOT / ".plan" / "ref_codes_cache.npy"), np.array(ref_codes))
    ref_codes_np = np.load(str(ROOT / ".plan" / "ref_codes_cache.npy"))

    # 1. basic, no ref
    run_step_generate("1_basic_no_ref", results, {})

    # 2. clone, ref_audio=path (recomputes ref codes internally every call)
    run_step_generate("2_clone_ref_audio_path", results, {"ref_audio": str(REF_AUDIO), "ref_text": REF_TEXT})

    # 3. clone, ref_audio_codes=precomputed (cached ref, no re-encode)
    run_step_generate("3_clone_ref_audio_codes_cached", results,
                       {"ref_audio_codes": ref_codes_np, "ref_text": REF_TEXT})

    # 4. batch, no ref
    run_step_batch("4_batch_no_ref", results, {})

    # 5. batch, same ref_audio path for all rows (shared -> encoded once internally per
    #    _normalize_batch_references, per earlier code reading)
    run_step_batch("5_batch_ref_audio_shared", results, {"ref_audio": str(REF_AUDIO), "ref_text": REF_TEXT})

    # 6. batch, same precomputed ref_audio_codes for all rows
    run_step_batch("6_batch_ref_audio_codes_cached", results,
                    {"ref_audio_codes": ref_codes_np, "ref_text": REF_TEXT})

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
