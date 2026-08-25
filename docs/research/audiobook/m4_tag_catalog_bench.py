#!/usr/bin/env python3
"""Full tag-reference generation for issue #57's tag catalog (`docs/guides/tag_reference.md`).

Generates ONE comparable clip per each of the 43 official Higgs TTS 3 control tags
(`src/audiobook.py`'s `VALID_TAGS`) plus the 2 undocumented `<|env:*|>` tokens, plus a
neutral baseline -- 46 clips total, all on the SAME neutral Russian carrier text used by
`m4_tag_inventory_bench.py` (S1/S2/S3), so results are directly comparable to the earlier
34-tag inventory (PR #108) and to each other. Not new development: reuses
`model.batch_generate` (batch=8, PR #105) exactly like `m4_tag_inventory_bench.py`.

Placement rules (per PROMPTING.md, verified against the pinned snapshot):
- Sentence-level tags (emotion, style, prosody speed_*/pitch_*/expressive_*) are REOPENED
  at the start of every sentence, mirroring `chunk_sentences()`.
- Inline one-shot tags (prosody pause/long_pause, ALL sfx) are inserted ONCE between
  sentence 1 and sentence 2 -- never as a sentence prefix. `sfx` additionally follows
  PROMPTING.md's literal format: `<|sfx:tag|>onomatopoeia, ` (tag immediately followed by
  the onomatopoeia, no space, then a comma continuing the same sentence).
- `<|env:music|>` / `<|env:noise|>` are UNDOCUMENTED. Tried as a sentence-level prefix
  (owner's explicit instruction), same reopening convention as emotion/style, so their
  behavior is measured the same way as everything else even though PROMPTING.md says
  nothing about them.
"""
from __future__ import annotations

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

from audiobook import VALID_TAGS, INLINE_ONE_SHOT_PROSODY  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 4096

# Same neutral-by-meaning 3-sentence Russian text as m4_tag_inventory_bench.py (PR #108),
# for direct comparability across both runs.
S1 = "Сегодня я занимался повседневными делами."
S2 = "Утром я выпил чай и почитал книгу."
S3 = "Потом вышел на улицу и немного прошёлся."
NEUTRAL_TEXT = f"{S1} {S2} {S3}"

# Undocumented env tokens (owner's explicit ask -- not in PROMPTING.md's 43-tag catalog).
ENV_TAGS = {"<|env:music|>", "<|env:noise|>"}

# Onomatopoeia per PROMPTING.md's literal sfx format:
# "<|sfx:tag|>onomatopoeia, then the line" (tag immediately followed, no space).
SFX_ONOMATOPOEIA = {
    "cough": "Кхе-кхе",
    "laughter": "Ха-ха-ха",
    "crying": "Всхлип",
    "screaming": "А-а-а",
    "burping": "Ыгх",
    "humming": "М-м-м",
    "sigh": "Ах",
    "sniff": "Шмыг",
    "sneeze": "Апчхи",
}


def sentence_level_clip(tag: str) -> str:
    """Reopen `tag` at the start of every sentence, mirroring chunk_sentences()."""
    return f"{tag}{S1} {tag}{S2} {tag}{S3}"


def inline_prosody_clip(tag: str) -> str:
    """Insert an inline one-shot prosody tag once, between sentence 1 and 2."""
    return f"{S1} {tag}{S2} {S3}"


def inline_sfx_clip(tag: str, name: str) -> str:
    """PROMPTING.md's literal sfx format, inserted between sentence 1 and 2:
    `<|sfx:name|>Onomatopoeia, <lowercased S2> S3` -- one continuous sentence for the
    sfx + its carrier clause, exactly as the doc's own example
    (`<|sfx:cough|>Ahem, welcome everyone, let's get started.`) does it."""
    onoma = SFX_ONOMATOPOEIA[name]
    s2_lower = S2[0].lower() + S2[1:]
    return f"{S1} {tag}{onoma}, {s2_lower} {S3}"


def build_clips() -> list[tuple[str, str]]:
    """Returns list of (clip_id, text). 46 total: neutral + 43 official tags + 2 env."""
    clips: list[tuple[str, str]] = [("neutral_baseline", NEUTRAL_TEXT)]

    for tag in sorted(VALID_TAGS):
        name = tag.split(":", 1)[1].rstrip("|>")
        category = tag.split(":", 1)[0].lstrip("<|")
        clip_id = f"tag_{category}_{name}"
        if category == "sfx":
            clips.append((clip_id, inline_sfx_clip(tag, name)))
        elif name in INLINE_ONE_SHOT_PROSODY:
            clips.append((clip_id, inline_prosody_clip(tag)))
        else:
            clips.append((clip_id, sentence_level_clip(tag)))

    for tag in sorted(ENV_TAGS):
        name = tag.split(":", 1)[1].rstrip("|>")
        clips.append((f"tag_env_{name}", sentence_level_clip(tag)))

    assert len(clips) == 1 + len(VALID_TAGS) + len(ENV_TAGS), (
        len(clips), len(VALID_TAGS), len(ENV_TAGS)
    )
    return clips


def machine_state() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<error: {exc}>"

    return {"uptime": run(["uptime"]), "vm_swapusage": run(["sysctl", "vm.swapusage"])}


def audio_duration(samples: int, sample_rate: int) -> float:
    return samples / sample_rate if sample_rate else 0.0


def main() -> None:
    out_dir = ROOT / "output" / "m4_tag_catalog"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = build_clips()
    print(f"total clips: {len(clips)}", flush=True)
    for clip_id, text in clips:
        print(f"  {clip_id}: {text!r}", flush=True)

    state_before = machine_state()
    print("machine state before run:", json.dumps(state_before, ensure_ascii=False), flush=True)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    # Warm-up (discarded), same convention as m4_batching_bench.py / m4_tag_inventory_bench.py.
    t0 = time.perf_counter()
    list(model.generate(text="Это короткая прогревочная фраза перед замером.",
                         temperature=1.0, max_new_tokens=MAX_NEW_TOKENS))
    warmup_seconds = time.perf_counter() - t0
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded)", flush=True)

    mx.reset_peak_memory()
    run_start = time.perf_counter()

    per_clip = []
    manifest = []
    for chunk_start in range(0, len(clips), BATCH_SIZE):
        chunk = clips[chunk_start: chunk_start + BATCH_SIZE]
        texts = [t for _, t in chunk]
        t0 = time.perf_counter()
        results = list(
            model.batch_generate(texts=texts, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS)
        )
        mx.eval(*[r.audio for r in results])
        chunk_wall = time.perf_counter() - t0
        results.sort(key=lambda r: r.sequence_idx)
        if len(results) != len(chunk):
            raise RuntimeError(
                f"chunk starting at {chunk_start}: expected {len(chunk)} results, got {len(results)}"
            )
        for offset, result in enumerate(results):
            clip_id, text = chunk[offset]
            audio = np.asarray(result.audio).reshape(-1)
            sample_rate = result.sample_rate
            duration = audio_duration(len(audio), sample_rate)
            wav_path = out_dir / f"{clip_id}.wav"
            audio_write(str(wav_path), audio, sample_rate)
            per_clip.append(
                {
                    "clip_id": clip_id,
                    "text": text,
                    "chars": len(text),
                    "chunk_wall_seconds": chunk_wall,
                    "chunk_size": len(chunk),
                    "audio_duration_seconds": duration,
                    "wav_path": str(wav_path),
                }
            )
            manifest.append(clip_id)
        print(f"  [batch={len(chunk)}] {manifest[-len(chunk):]}: wall={chunk_wall:.3f}s", flush=True)

    run_wall = time.perf_counter() - run_start
    total_audio = sum(c["audio_duration_seconds"] for c in per_clip)
    aggregate_rtf = (run_wall / total_audio) if total_audio else None
    peak_mlx_gib = mx.get_peak_memory() / (1024 ** 3)

    state_after = machine_state()

    result = {
        "num_clips": len(clips),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "run_wall_seconds": run_wall,
        "total_audio_duration_seconds": total_audio,
        "aggregate_rtf": aggregate_rtf,
        "peak_mlx_gib": peak_mlx_gib,
        "per_clip": per_clip,
        "machine_state_before": state_before,
        "machine_state_after": state_after,
    }
    out_json = ROOT / "logs" / "m4_tag_catalog.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
