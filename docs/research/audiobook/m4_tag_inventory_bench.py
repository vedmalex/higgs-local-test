#!/usr/bin/env python3
"""M4-T5 measurement: generate one clip per each of the 34 known-valid Higgs TTS 3
control tags (`src/audiobook.py`'s `VALID_TAGS`), plus a neutral baseline and a small
set of terminal-intonation probe clips, using the already-measured batching path
(`model.batch_generate`, batch=8, PR #105) instead of 34+ sequential calls.

Issue #57. Not new development: measures existing `mlx_audio` code exactly like
`m4_batching_bench.py` does, reusing its API and machine-state/peak-memory
conventions.

Text design (owner's brief for M4-T5):
- A short (2-3 sentence) semantically NEUTRAL Russian text, so that any emotional
  color in the output must come from the tag, not from the words. The same base
  text is used for every tag so metrics differences are attributable to the tag.
- Sentence-level tags (emotion/style/prosody except pause/long_pause) are
  REOPENED at the start of every sentence in the 2-3 sentence text -- this mirrors
  `chunk_sentences()`'s own reopening behavior in `src/audiobook.py` for tags that
  persist across sentences within one chunk, so the measurement reflects how the
  tag is actually used in production, not a weaker one-sentence-only variant.
- `<|prosody:pause|>` / `<|prosody:long_pause|>` are INLINE one-shot effects
  (`INLINE_ONE_SHOT_PROSODY` in `src/audiobook.py`): inserted ONCE, between
  sentence 1 and sentence 2 of the neutral text, not reopened.

Additional terminal-intonation probes (owner addendum, same run):
- `punct_period` / `punct_question` / `punct_exclaim`: the identical one-sentence
  neutral text ending in `.` / `?` / `!` respectively, to see whether the model's
  terminal F0 contour differs by punctuation alone.
- `boundary_complete`: a single, semantically self-contained sentence (nothing
  pending), generated standalone -- the normal `chunk_sentences()` case where a
  chunk ends on a genuine sentence boundary.
- `boundary_continuing`: a single, grammatically complete sentence that is
  narratively a "cliffhanger" (the thought continues in a following sentence that
  is NOT included here) -- generated standalone, exactly as `generate_segments`
  would generate it as an independent chunk with no knowledge of what follows.
  Compares whether the terminal contour still falls even when the narrative isn't
  "done".

All 39 clips are generated in 5 batches of <=8 via `model.batch_generate`.
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

# Neutral-by-meaning 3-sentence Russian text (no emotionally loaded words).
S1 = "Сегодня я занимался повседневными делами."
S2 = "Утром я выпил чай и почитал книгу."
S3 = "Потом вышел на улицу и немного прошёлся."
NEUTRAL_TEXT = f"{S1} {S2} {S3}"

# Terminal-intonation punctuation probes (single self-contained sentence, base form).
PUNCT_BASE = "Я закрыл дверь и сел за стол"
BOUNDARY_COMPLETE = "Сегодня утром я пил чай и читал книгу."
BOUNDARY_CONTINUING = "Он медленно подошёл к двери и взялся за ручку."

# --- Stress-mark notation probe (owner addendum) ------------------------------
# Higgs has NEVER been tested with stress notation before this run (see
# docs/research/qwen3-tts-notes.md -- all prior stress experiments were Qwen3-TTS
# only). 3 Russian homograph pairs where stress position changes MEANING, each
# word tested in 6 notations, in a disambiguating carrier sentence so a human
# listener can judge whether the intended meaning came through.
STRESS_NOTATIONS = ("acute", "capital", "apostrophe", "plus", "doubled", "none")


def notate(word: str, idx: int, mode: str) -> str:
    """`idx` = 0-based index of the stressed vowel letter in `word`."""
    if mode == "acute":
        return word[: idx + 1] + "́" + word[idx + 1:]
    if mode == "capital":
        return word[:idx] + word[idx].upper() + word[idx + 1:]
    if mode == "apostrophe":
        return word[: idx + 1] + "'" + word[idx + 1:]
    if mode == "plus":
        return word[:idx] + "+" + word[idx:]
    if mode == "doubled":
        return word[: idx + 1] + word[idx] + word[idx + 1:]
    if mode == "none":
        return word
    raise ValueError(mode)


# (pair_id, meaning, carrier_sentence_template, base_word, stressed_vowel_index)
HOMOGRAPH_CASES = [
    ("zamok", "castle", "На холме стоит старинный {}.", "замок", 1),   # за́мок
    ("zamok", "lock", "На двери висит крепкий {}.", "замок", 3),        # замо́к
    ("stoit", "costs", "Эта книга дорого {}.", "стоит", 2),             # сто́ит
    ("stoit", "stands", "Этот дом давно {} пустым.", "стоит", 3),       # стои́т
    ("atlas", "book", "На полке лежит географический {}.", "атлас", 0),  # а́тлас
    ("atlas", "fabric", "Платье сшито из блестящего {}.", "атлас", 3),  # атла́с
]


def sentence_level_clip(tag: str) -> str:
    """Reopen `tag` at the start of every sentence, mirroring chunk_sentences()."""
    return f"{tag}{S1} {tag}{S2} {tag}{S3}"


def inline_once_clip(tag: str) -> str:
    """Insert an inline one-shot prosody tag once, between sentence 1 and 2."""
    return f"{S1} {tag}{S2} {S3}"


def build_clips() -> list[tuple[str, str]]:
    """Returns list of (clip_id, text)."""
    clips: list[tuple[str, str]] = [("neutral_baseline", NEUTRAL_TEXT)]

    for tag in sorted(VALID_TAGS):
        # tag looks like "<|category:name|>"
        name = tag.split(":", 1)[1].rstrip("|>")
        clip_id = f"tag_{name}"
        if name in INLINE_ONE_SHOT_PROSODY:
            clips.append((clip_id, inline_once_clip(tag)))
        else:
            clips.append((clip_id, sentence_level_clip(tag)))

    assert len(clips) == 1 + len(VALID_TAGS), (len(clips), len(VALID_TAGS))

    # Terminal-intonation probes (owner addendum).
    clips.append(("punct_period", PUNCT_BASE + "."))
    clips.append(("punct_question", PUNCT_BASE + "?"))
    clips.append(("punct_exclaim", PUNCT_BASE + "!"))
    clips.append(("boundary_complete", BOUNDARY_COMPLETE))
    clips.append(("boundary_continuing", BOUNDARY_CONTINUING))

    # Stress-mark notation probe: 6 homograph-meanings x 6 notations = 36 clips.
    for pair_id, meaning, template, base_word, vowel_idx in HOMOGRAPH_CASES:
        for mode in STRESS_NOTATIONS:
            word = notate(base_word, vowel_idx, mode)
            text = template.format(word)
            clips.append((f"stress_{pair_id}_{meaning}_{mode}", text))

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
    out_dir = ROOT / "output" / "m4_tag_inventory"
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

    # Warm-up (discarded), same convention as m4_batching_bench.py.
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
    out_json = ROOT / "logs" / "m4_tag_inventory.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
