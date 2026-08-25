#!/usr/bin/env python3
"""Chapter-scale segmentation and generation for Higgs Audio v3 (issue #57, M4 Lane 2).

`higgs_audio_v3` has no chunking path of its own (see docs/research/audiobook/m4-plan.md
§0.7) -- the closest existing pattern is the neighbor `qwen3_tts.py`'s `split_pattern`
(qwen3_tts.py:1271) and `_decode_chunk` (qwen3_tts.py:1037, chunk_tokens=300,
left_context_size=25). That pattern is "split on a literal separator, generate each segment
as an independent model call, concatenate the audio" -- it does not solve Higgs's specific
problem (inline control tags whose scope must survive a chunk boundary), so this module
adapts the *shape* of that approach (independent per-chunk model calls, concatenated audio)
rather than porting its code, and adds tag-continuity tracking on top, which Qwen's
`generate_custom_voice(instruct=...)` never needed because it only supports one style per
whole call (m4-plan.md §0.1).

What is newly written here:
  - Russian-aware sentence splitting that does not break on abbreviations/initials or
    inside quotes (`split_sentences`).
  - Chunking that groups sentences up to a character budget without ever splitting a
    sentence (`chunk_sentences`).
  - Control-tag continuity tracking across chunk boundaries -- re-emitting (\"reopening\")
    the last-seen emotion/prosody/style tag at the start of a new chunk, or before every
    sentence, depending on `--tag-scope` (see docs/research/audiobook/m4-chapter-results.md
    for the empirical basis of the default).
  - A per-segment manifest with resume support, and a separate assembly step with a
    numeric splice-quality check (`assemble_chapter`).

What is reused as-is: `mlx_audio.tts.utils.load` and `HiggsAudioV3.generate(text=...,
temperature=..., max_new_tokens=...)` -- the exact call convention already used by
`src/tts_test.py --text`. This module does not build a second/parallel generation path;
`generate_segments` below is the only place that calls the model, and it calls it exactly
the way `src/tts_test.py` does.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_ID = "bosonai/higgs-tts-3-4b"

# ---------------------------------------------------------------------------
# Sentence splitting (Russian-aware: abbreviations, initials, quotes)
# ---------------------------------------------------------------------------

SENTENCE_END_CHARS = ".!?…"
OPEN_QUOTES = "«\"“‘("
CLOSE_QUOTES = "»\"”’)"

# Common Russian abbreviations that end in a period but do not end a sentence.
# Matched case-insensitively against the token immediately preceding the period run.
ABBREVIATIONS = {
    "т.е.", "т.д.", "т.п.", "т.к.", "т.н.", "т.о.", "т.г.",
    "др.", "пр.", "см.", "гл.", "рис.", "табл.", "стр.", "гг.", "вв.", "г.",
    "им.", "проф.", "акад.", "доц.", "канд.", "д-р",
    "мин.", "сек.", "час.", "руб.", "коп.", "тыс.", "млн.", "млрд.",
    "напр.", "включ.", "исключ.", "обл.", "р-н", "ул.", "д.", "кв.", "корп.",
    "и.о.", "с.г.", "н.э.", "до н.э.",
}


def _token_ending_at(text: str, end_idx: int) -> str:
    """Return the run of letters/periods immediately before (and including) end_idx."""
    start = end_idx
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start : end_idx + 1]


def _is_non_terminal_period(text: str, run_start: int) -> bool:
    """True if the period run starting at run_start should NOT end a sentence."""
    token = _token_ending_at(text, run_start).lower()
    if token in ABBREVIATIONS:
        return True
    bare = token.rstrip(".")
    # Single-letter initial, e.g. "А." in "А. С. Пушкин".
    if len(bare) == 1 and bare.isalpha():
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking abbreviations, initials, or quotes.

    Control tags (``<|category:tag|>``) contain no sentence-ending punctuation, so they
    pass through untouched regardless of where they sit in a sentence.
    """
    sentences: list[str] = []
    n = len(text)
    i = 0
    start = 0
    quote_depth = 0
    while i < n:
        ch = text[i]
        if ch in OPEN_QUOTES:
            quote_depth += 1
            i += 1
            continue
        if ch in CLOSE_QUOTES:
            quote_depth = max(0, quote_depth - 1)
            i += 1
            continue
        if ch in SENTENCE_END_CHARS:
            run_start = i
            j = i
            while j < n and text[j] in SENTENCE_END_CHARS:
                j += 1
            end_punct = j
            # Swallow a directly-following closing quote into the sentence.
            k = end_punct
            trailing_quote_depth = quote_depth
            while k < n and text[k] in CLOSE_QUOTES:
                trailing_quote_depth = max(0, trailing_quote_depth - 1)
                k += 1
            if trailing_quote_depth > 0:
                # Still inside an open quote -- this punctuation does not end the sentence.
                i = j
                continue
            if _is_non_terminal_period(text, run_start):
                i = j
                continue
            m = k
            while m < n and text[m].isspace():
                m += 1
            boundary_ok = True
            if m < n and text[m].islower():
                boundary_ok = False
            if boundary_ok:
                quote_depth = trailing_quote_depth
                sentence = text[start:k].strip()
                if sentence:
                    sentences.append(sentence)
                start = k
                i = k
                continue
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# Control-tag tracking and chunking
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<\|(emotion|prosody|style):([a-z_]+)\|>")

# Per PROMPTING.md (bosonai/higgs-tts-3-4b, verified against the cached snapshot):
# "pause" / "long_pause" are INLINE, one-shot effects at an exact position in a
# sentence -- they are not a sustained state and must never be reopened as one.
INLINE_ONE_SHOT_PROSODY = {"pause", "long_pause"}


@dataclass
class Chunk:
    index: int
    sentences: list[str]
    reopened_tags: dict[str, str]
    text: str


def _sentence_own_tags(sentence: str) -> list[tuple[str, str]]:
    return TAG_RE.findall(sentence)


def _reopen_prefix(active: dict[str, Optional[str]], sentence: str) -> str:
    """Build the tag prefix to prepend to `sentence` so active state survives."""
    prefix_parts = []
    stripped = sentence.lstrip()
    for category, tag in active.items():
        if tag is None:
            continue
        if stripped.startswith(f"<|{category}:"):
            continue  # sentence already (re)declares this category itself
        prefix_parts.append(tag)
    return "".join(prefix_parts)


def chunk_sentences(
    sentences: list[str],
    max_chars: int = 500,
    tag_scope: str = "chunk",
) -> list[Chunk]:
    """Group sentences into chunks under `max_chars`, never splitting a sentence.

    tag_scope="chunk"    reopen the last-active emotion/prosody/style tag only at the
                         start of a new chunk (i.e. only where a real generation-call
                         boundary exists).
    tag_scope="sentence" reopen the last-active tag before every sentence that does not
                         already declare its own tag of that category. This matches
                         PROMPTING.md's literal semantics ("sentence-level ... colors the
                         whole sentence") -- see m4-chapter-results.md for the empirical
                         test that motivated making this the default.

    Inline one-shot prosody (`pause`, `long_pause`) is never reopened; it fires once at
    its authored position and carries no state.
    """
    if tag_scope not in ("chunk", "sentence"):
        raise ValueError(f"unknown tag_scope: {tag_scope!r}")

    chunks: list[Chunk] = []
    active: dict[str, Optional[str]] = {"emotion": None, "prosody": None, "style": None}

    cur_sentences: list[str] = []
    cur_len = 0
    chunk_start_active: dict[str, Optional[str]] = dict(active)

    def flush() -> Optional[Chunk]:
        nonlocal cur_sentences, cur_len
        if not cur_sentences:
            return None
        if tag_scope == "sentence":
            text = " ".join(cur_sentences)
            reopened = {k: v for k, v in chunk_start_active.items() if v is not None}
        else:
            prefix = _reopen_prefix(chunk_start_active, cur_sentences[0])
            text = prefix + " ".join(cur_sentences)
            reopened = {
                cat: tag
                for cat, tag in chunk_start_active.items()
                if tag is not None and tag in prefix
            }
        chunk = Chunk(
            index=len(chunks),
            sentences=list(cur_sentences),
            reopened_tags=reopened,
            text=text,
        )
        cur_sentences = []
        cur_len = 0
        return chunk

    for sent in sentences:
        sent_len = len(sent)
        if cur_sentences and cur_len + 1 + sent_len > max_chars:
            chunk = flush()
            if chunk is not None:
                chunks.append(chunk)
            chunk_start_active = dict(active)

        own_tags = _sentence_own_tags(sent)
        if tag_scope == "sentence":
            prefix = _reopen_prefix(active, sent)
            sent_to_store = prefix + sent
        else:
            sent_to_store = sent
        cur_sentences.append(sent_to_store)
        cur_len += sent_len + 1

        for category, tag_name in own_tags:
            if category == "prosody" and tag_name in INLINE_ONE_SHOT_PROSODY:
                continue
            active[category] = f"<|{category}:{tag_name}|>"

    if cur_sentences:
        chunk = flush()
        if chunk is not None:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# WAV I/O (stdlib + numpy only -- no mlx_audio dependency for read/assemble)
# ---------------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(n)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sampwidth}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Manifest + resumable per-segment generation
# ---------------------------------------------------------------------------


def build_manifest(chunks: list[Chunk], output_dir: Path) -> dict:
    segments = []
    for c in chunks:
        segments.append(
            {
                "index": c.index,
                "sentences": c.sentences,
                "text": c.text,
                "reopened_tags": c.reopened_tags,
                "status": "pending",
                "output_path": str(output_dir / f"segment_{c.index:04d}.wav"),
                "sample_rate": None,
                "audio_duration_seconds": None,
                "generation_seconds": None,
                "error": None,
            }
        )
    return {"model": MODEL_ID, "created": time.time(), "segments": segments}


def load_or_create_manifest(manifest_path: Path, chunks: list[Chunk], output_dir: Path) -> dict:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_texts = [seg["text"] for seg in manifest["segments"]]
        new_texts = [c.text for c in chunks]
        if existing_texts != new_texts:
            raise RuntimeError(
                "Existing manifest's segment text does not match the current chunking "
                "plan -- refusing to resume blindly. Re-run with a fresh --output-dir, "
                "or confirm the input text/chunking parameters have not changed."
            )
        return manifest
    manifest = build_manifest(chunks, output_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_segments(
    model,
    manifest: dict,
    manifest_path: Path,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
) -> None:
    """Generate every pending segment, writing progress to disk after each one.

    On restart, a segment already marked "done" with its WAV present on disk is
    skipped -- this is the resume mechanism required for multi-hour runs (m4-plan.md
    §3 M4-TX/T8): a crash mid-chapter only loses the segment in flight, not everything
    generated so far.
    """
    for entry in manifest["segments"]:
        out_path = Path(entry["output_path"])
        if entry["status"] == "done" and out_path.exists():
            continue

        entry["status"] = "in_progress"
        save_manifest(manifest, manifest_path)
        try:
            started = time.perf_counter()
            results = list(
                model.generate(
                    text=entry["text"],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            )
            generation_seconds = time.perf_counter() - started
            if not results:
                raise RuntimeError("model.generate produced no result")
            sample_rate = results[0].sample_rate
            audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
            write_wav(out_path, audio, sample_rate)
            entry["status"] = "done"
            entry["sample_rate"] = sample_rate
            entry["audio_duration_seconds"] = len(audio) / sample_rate if sample_rate else None
            entry["generation_seconds"] = generation_seconds
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 -- must record and keep the manifest resumable
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            save_manifest(manifest, manifest_path)
            raise
        save_manifest(manifest, manifest_path)


# ---------------------------------------------------------------------------
# Assembly (separate step) + numeric splice-quality check
# ---------------------------------------------------------------------------


def assemble_chapter(manifest: dict, output_path: Path, silence_ms: int = 200) -> dict:
    segments = manifest["segments"]
    if not segments:
        raise RuntimeError("manifest has no segments to assemble")

    sample_rate: Optional[int] = None
    audio_parts: list[np.ndarray] = []
    join_reports = []
    edge_window_ms = 20

    for i, entry in enumerate(segments):
        if entry["status"] != "done":
            raise RuntimeError(
                f"segment {entry['index']} is not done (status={entry['status']!r}) "
                "-- cannot assemble a chapter with missing segments"
            )
        audio, sr = read_wav(Path(entry["output_path"]))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                f"sample rate mismatch: segment {entry['index']} is {sr} Hz, "
                f"expected {sample_rate} Hz"
            )

        if audio_parts:
            edge_n = int(edge_window_ms / 1000 * sample_rate)
            prev_tail = audio_parts[-1][-edge_n:] if len(audio_parts[-1]) >= edge_n else audio_parts[-1]
            this_head = audio[:edge_n] if len(audio) >= edge_n else audio
            prev_edge_amp = float(np.max(np.abs(prev_tail))) if len(prev_tail) else 0.0
            head_edge_amp = float(np.max(np.abs(this_head))) if len(this_head) else 0.0
            # Largest sample-to-sample jump within each edge window, i.e. what an
            # audible click looks like numerically within a single segment's edge.
            max_intra_jump = float(
                max(
                    np.max(np.abs(np.diff(prev_tail))) if len(prev_tail) > 1 else 0.0,
                    np.max(np.abs(np.diff(this_head))) if len(this_head) > 1 else 0.0,
                )
            )
            # The jump directly AT the join point: last sample of the previous segment
            # vs. first sample of the next one. Meaningful mainly at silence_ms=0, where
            # the two segments are directly concatenated with nothing between them.
            direct_join_jump = (
                float(abs(this_head[0] - prev_tail[-1])) if len(prev_tail) and len(this_head) else 0.0
            )
            join_reports.append(
                {
                    "after_segment": segments[i - 1]["index"],
                    "before_segment": entry["index"],
                    "prev_tail_edge_abs_amplitude": prev_edge_amp,
                    "next_head_edge_abs_amplitude": head_edge_amp,
                    "max_intra_window_sample_jump": max_intra_jump,
                    "direct_join_sample_jump": direct_join_jump,
                }
            )
            if silence_ms > 0:
                silence = np.zeros(int(silence_ms / 1000 * sample_rate), dtype=audio.dtype)
                audio_parts.append(silence)
        audio_parts.append(audio)

    full_audio = np.concatenate(audio_parts)
    write_wav(output_path, full_audio, sample_rate)

    return {
        "output": str(output_path),
        "sample_rate": sample_rate,
        "num_segments": len(segments),
        "silence_ms_between_segments": silence_ms,
        "total_duration_seconds": len(full_audio) / sample_rate if sample_rate else None,
        "join_reports": join_reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--tag-scope", choices=("chunk", "sentence"), default="sentence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--silence-ms", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        parser.error("one of --text or --text-file is required")
        return

    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences, max_chars=args.max_chars, tag_scope=args.tag_scope)

    if args.dry_run:
        for c in chunks:
            print(
                json.dumps(
                    {"index": c.index, "reopened_tags": c.reopened_tags, "text": c.text},
                    ensure_ascii=False,
                )
            )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_or_create_manifest(manifest_path, chunks, args.output_dir)

    if not args.assemble_only:
        from mlx_audio.tts.utils import load

        model = load(MODEL_ID, model_type="higgs_audio_v3")
        generate_segments(
            model,
            manifest,
            manifest_path,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    if args.assemble or args.assemble_only:
        result = assemble_chapter(
            manifest, args.output_dir / "chapter.wav", silence_ms=args.silence_ms
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
