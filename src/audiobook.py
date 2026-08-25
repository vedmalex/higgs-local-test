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
  - Russian-aware sentence splitting that does not break on abbreviations/initials, inside
    quotes, or on numbered-list markers (`split_sentences`).
  - Chunking that groups sentences up to a character budget without ever splitting a
    sentence, with a forced hard split for any single sentence that is itself over budget
    (`chunk_sentences`).
  - Control-tag continuity tracking across chunk boundaries -- re-emitting (\"reopening\")
    the last-seen emotion/prosody/style tag at the start of a new chunk, or before every
    sentence, depending on `--tag-scope` (see docs/research/audiobook/m4-chapter-results.md
    for the empirical basis of the default) -- plus hard validation of every control tag
    against the model's actual tag vocabulary.
  - A per-segment, content-hash-keyed manifest with atomic writes, crash recovery, and
    resume-integrity checks, and a separate streaming assembly step with a numeric
    splice-quality check (`assemble_chapter`).

What is reused as-is: `mlx_audio.tts.utils.load` and `HiggsAudioV3.generate(text=...,
temperature=..., max_new_tokens=...)` -- the exact call convention already used by
`src/tts_test.py --text`. This module does not build a second/parallel generation path;
`generate_segments` below is the only place that calls the model, and it calls it exactly
the way `src/tts_test.py` does.

Independent audit findings (2026-08, Refs #57) fixed in this revision -- see
docs/research/audiobook/m4-chapter-results.md for the corrected write-up of what actually
works and what does not:
  F1  -- quote-depth tracking was a single counter that both quote-branches fed with the
         same character (`"`), so it only ever grew; fixed with per-pair-type stacks, a
         parity toggle for the ambiguous straight quote, a paragraph-boundary reset, and a
         force-reset safety valve after `QUOTE_FORCE_RESET_CHARS`.
  F2  -- a single sentence longer than `max_chars` used to bypass the budget entirely;
         `chunk_sentences` now force-splits any such sentence on `;`, then `,`, then
         whitespace, then a hard character cut, and warns.
  F3  -- chapter assembly used to materialize the whole chapter as float64 arrays twice;
         `assemble_chapter` now streams: one segment's audio in memory at a time, written
         directly into the output `wave` handle.
  F4  -- manifest writes used to truncate-in-place; `save_manifest` now writes to a temp
         file, fsyncs, and `os.replace()`s, keeping a `.bak`; `load_or_create_manifest`
         recovers from `.bak` on a `JSONDecodeError` instead of crashing.
  F5  -- one failed segment used to abort the whole run; `generate_segments` now retries
         with backoff and, under `continue_on_error=True`, marks a segment `failed` and
         keeps going instead of raising.
  F6  -- a `done` segment was trusted on `exists()` alone; generation now validates
         audio length/duration plausibility and resume now checks the stored sample count
         against the actual file.
  F7  -- segments are now keyed by a hash of their own text (reused across chunking-plan
         changes elsewhere in the chapter) instead of requiring the whole plan to match
         byte-for-byte, `output_path` is stored relative to the manifest's directory, and
         the manifest header records `max_chars`/`tag_scope`/`model` for a precise mismatch
         error.
  F8  -- `max_chars` accounting under `tag_scope="sentence"` now counts the stored
         (prefixed) text, not the bare sentence.
  F9  -- a bare numbered-list marker ("1.", "2.") at the start of a sentence followed by a
         capitalized word no longer ends a sentence on its own.
  F10 -- a tag declared mid-sentence (not just at the very start) now suppresses reopening
         that category for the sentence.
  F11 -- every control-tag-shaped span is validated against the actual tag vocabulary
         (`VALID_TAGS`, extracted from the pinned tokenizer.json) and raises immediately.
  F12 -- `read_wav` now checks the declared frame count against the bytes actually present.
  F13 -- `assemble_chapter` takes `allow_gaps=True` to insert placeholder silence for
         non-`done` segments and report them, instead of refusing all-or-nothing.
  F14 -- `generate_segments` clears the MLX cache every `clear_cache_every` segments and
         records best-effort memory metrics per segment.

Screenplay format (2026-08, Refs #57): `docs/guides/audiobook_guide.md` sec. 3 already
documented a JSON "script" format (a list of ``{"speaker": ..., "text": ...}`` lines) as the
intended authoring format for a chapter, but nothing in this file ever read it -- the guide's
own example pipeline (sec. 4) called `model.generate()` directly, once per line, with none of
the sentence-splitting/chunking/tag-continuity/manifest/resume machinery above, so a long
reply would silently lose an emotion tag after its first chunk-worth of text (see
docs/research/audiobook/m4-chapter-results.md sec 2) and a one-line edit would force
regenerating the whole chapter. `parse_screenplay` + `chunk_screenplay` below read that same
JSON format into the existing `Chunk`/manifest pipeline instead of adding a second, parallel
generation path:
  - `parse_screenplay` validates each line (non-blank `speaker`/`text`, valid control tags)
    and drops any other JSON key a line might carry (notes, ids, ...) -- those cannot affect
    the generated audio, so they must not affect resume/regeneration either.
  - `chunk_screenplay` runs each line through the same `split_sentences`/`chunk_sentences` as
    plain text, so a long reply is still chunked and its tags still reopened across chunk
    boundaries exactly as for a plain chapter. Control-tag/emotion state is reset at every
    new speaker line (a character's leftover emotional state has no textual basis once a
    different speaker starts talking).
  - Segment identity (`_segment_hash_input`) is now `speaker + text`, not just `text`: two
    lines with identical wording said by different speakers must not collide onto the same
    cached segment once per-speaker voices exist, and a speaker rename with the same wording
    is a real change to what should be synthesized.
  - `speaker` is threaded into every manifest segment and `assemble_chapter` can use a
    different pause length across a speaker change (`speaker_change_silence_ms`) than within
    one speaker's own sentences, matching the guide's pause table (sec. 5). What is
    deliberately NOT done: no code path here selects, loads, or synthesizes a per-speaker
    voice -- `generate_segments` still calls `model.generate(text=...)` with the model's one
    default voice for every segment regardless of `speaker`. Per-character voice cloning has
    never been verified for a multi-voice book in this project, and the one cloning
    measurement that exists (RTF 7.73 vs. 6.56 for plain generation, see
    docs/research/audiobook/) shows it is *slower*, not just unimplemented; `chunk_screenplay`
    raises a `UserWarning` naming every distinct speaker whenever a screenplay uses more than
    one, so this is never silently mistaken for "it just works."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import wave
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_ID = "bosonai/higgs-tts-3-4b"

# ---------------------------------------------------------------------------
# Sentence splitting (Russian-aware: abbreviations, initials, quotes, lists)
# ---------------------------------------------------------------------------

SENTENCE_END_CHARS = ".!?…"

# Paired quote/bracket openers and their expected closers. `"` is deliberately NOT here --
# it is the same glyph for open and close, so it is tracked separately as a parity toggle
# (see split_sentences). Note "“" (") appears both as a value (closer for „) and as a
# key (opener for the plain English-style “…” pair); a stack-based match-top-first resolves
# the ambiguity contextually instead of by character identity alone.
PAIR_OPEN_TO_CLOSE = {
    "«": "»",
    "„": "“",  # „quote“ -- opens with „, closes with “
    "“": "”",  # “quote” -- plain English-style curly quotes
    "‘": "’",  # ‘quote’
    "(": ")",
}

# Safety valve: if quote-nesting state has been continuously non-empty for longer than this
# many characters, force-reset it. Real prose does not hold an open quote this long; this
# exists so a single unmatched opening quote cannot silently swallow the rest of a chapter
# into "one sentence" (the catastrophic failure mode fixed here) even though it cannot, by
# itself, tell where the missing closing quote was actually meant to go.
QUOTE_FORCE_RESET_CHARS = 400

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


def _token_bounds(text: str, end_idx: int) -> tuple[int, str]:
    """Return (start_idx, token) for the run of letters/digits/periods ending at end_idx."""
    start = end_idx
    while start > 0 and (
        text[start - 1].isalpha() or text[start - 1] == "." or text[start - 1].isdigit()
    ):
        start -= 1
    return start, text[start : end_idx + 1]


def _is_non_terminal_period(
    text: str, run_start: int, run_end: int, sentence_start: int
) -> bool:
    """True if the period run starting at run_start should NOT end a sentence."""
    token_start, token_raw = _token_bounds(text, run_start)
    token = token_raw.lower()
    if token in ABBREVIATIONS:
        return True
    bare = token.rstrip(".")
    # Single-letter initial, e.g. "А." in "А. С. Пушкин".
    if len(bare) == 1 and bare.isalpha():
        return True
    if bare.isdigit() and bare:
        # A bare numbered-list marker ("1.", "23.") at the very start of the current
        # sentence-in-progress, followed by a capitalized word, is a list label, not a
        # sentence -- e.g. "1. Первый пункт." must not split into "1." + "Первый пункт.".
        # A number mid-sentence ("Их было 5. Потом ушли.") is unaffected: sentence_start
        # to token_start is not blank there, so this branch does not fire.
        if text[sentence_start:token_start].strip() == "":
            k = run_end
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[k].isupper():
                return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking abbreviations, initials, lists, or quotes.

    Control tags (``<|category:tag|>``) contain no sentence-ending punctuation, so they
    pass through untouched regardless of where they sit in a sentence.
    """
    sentences: list[str] = []
    n = len(text)
    i = 0
    start = 0
    quote_stack: list[str] = []
    straight_quote_open = False
    stuck_chars = 0

    def inside_quote() -> bool:
        return bool(quote_stack) or straight_quote_open

    def reset_quote_state() -> None:
        nonlocal straight_quote_open, stuck_chars
        quote_stack.clear()
        straight_quote_open = False
        stuck_chars = 0

    while i < n:
        ch = text[i]

        # Paragraph boundary (blank line): a pending quote state cannot legitimately
        # cross it, and neither can an in-progress sentence -- both are force-flushed
        # here rather than letting a missing closing quote (or a missing final period)
        # in one paragraph merge into the next paragraph's real sentences.
        if ch == "\n" and i + 1 < n and text[i + 1] == "\n":
            reset_quote_state()
            para_text = text[start:i].strip()
            if para_text:
                sentences.append(para_text)
            start = i  # leftover newline(s)/whitespace are stripped when the next
            i += 1  # sentence is appended
            continue

        if inside_quote():
            stuck_chars += 1
            if stuck_chars > QUOTE_FORCE_RESET_CHARS:
                reset_quote_state()
        else:
            stuck_chars = 0

        if ch == '"':
            straight_quote_open = not straight_quote_open
            i += 1
            continue
        if quote_stack and ch == quote_stack[-1]:
            quote_stack.pop()
            i += 1
            continue
        if ch in PAIR_OPEN_TO_CLOSE:
            quote_stack.append(PAIR_OPEN_TO_CLOSE[ch])
            i += 1
            continue

        if ch in SENTENCE_END_CHARS:
            run_start = i
            j = i
            while j < n and text[j] in SENTENCE_END_CHARS:
                j += 1
            end_punct = j
            # Swallow a directly-following closing quote into the sentence, tracking a
            # LOCAL copy of quote state so we can tell whether the sentence terminator is
            # still "inside" an open quote afterward without mutating real state yet.
            k = end_punct
            local_stack = list(quote_stack)
            local_straight = straight_quote_open
            while k < n and (
                (local_stack and text[k] == local_stack[-1])
                or (text[k] == '"' and local_straight)
            ):
                if text[k] == '"':
                    local_straight = False
                else:
                    local_stack.pop()
                k += 1
            if local_stack or local_straight:
                # Still inside an open quote -- this punctuation does not end the sentence.
                i = j
                continue
            if _is_non_terminal_period(text, run_start, end_punct, start):
                i = j
                continue
            m = k
            while m < n and text[m].isspace():
                m += 1
            boundary_ok = True
            if m < n and text[m].islower():
                boundary_ok = False
            if boundary_ok:
                quote_stack = local_stack
                straight_quote_open = local_straight
                stuck_chars = 0
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

TAG_RE = re.compile(r"<\|(emotion|prosody|style|sfx):([a-z0-9_]+)\|>")

# Loose shape used only for validation -- deliberately wider than TAG_RE (any category
# word, any case, digits) so a typo'd/mis-cased tag is *caught*, not silently ignored.
_TAG_SHAPE_RE = re.compile(r"<\|[A-Za-z0-9_]+:[A-Za-z0-9_]+\|>")

# The full, exact set of the model's 43 emotion/prosody/style/sfx control tags, extracted
# directly from bosonai/higgs-tts-3-4b's tokenizer.json `added_tokens`
# (snapshot 7556c17e05201fccd9c8cc120bc216dcc7b5d561, the pinned revision this project
# uses -- see AGENTS.md's model-constraints section) and cross-checked against that
# snapshot's PROMPTING.md "Full tag catalog (43)": 21 emotion + 10 prosody + 3 style +
# 9 sfx. Any control-tag-shaped span in the input that is not exactly one of these is a
# bug in the source text (typo, wrong case, or an invented tag) and must fail loudly
# before a multi-hour run starts, per PROMPTING.md's instruction to never invent tags.
VALID_TAGS = {
    "<|emotion:affection|>", "<|emotion:amusement|>", "<|emotion:anger|>",
    "<|emotion:arousal|>", "<|emotion:awe|>", "<|emotion:bitterness|>",
    "<|emotion:confusion|>", "<|emotion:contemplation|>", "<|emotion:contentment|>",
    "<|emotion:determination|>", "<|emotion:disgust|>", "<|emotion:elation|>",
    "<|emotion:enthusiasm|>", "<|emotion:fear|>", "<|emotion:helplessness|>",
    "<|emotion:longing|>", "<|emotion:pride|>", "<|emotion:relief|>",
    "<|emotion:sadness|>", "<|emotion:shame|>", "<|emotion:surprise|>",
    "<|prosody:expressive_high|>", "<|prosody:expressive_low|>", "<|prosody:long_pause|>",
    "<|prosody:pause|>", "<|prosody:pitch_high|>", "<|prosody:pitch_low|>",
    "<|prosody:speed_fast|>", "<|prosody:speed_slow|>", "<|prosody:speed_very_fast|>",
    "<|prosody:speed_very_slow|>", "<|style:shouting|>", "<|style:singing|>",
    "<|style:whispering|>",
    "<|sfx:burping|>", "<|sfx:cough|>", "<|sfx:crying|>", "<|sfx:humming|>",
    "<|sfx:laughter|>", "<|sfx:screaming|>", "<|sfx:sigh|>", "<|sfx:sneeze|>",
    "<|sfx:sniff|>",
}

# Deliberately NOT in VALID_TAGS (Refs #57): the tokenizer's `added_tokens` also carries
# `<|env:music|>` (id 151702), `<|env:noise|>` (id 151703), and a standalone `<|chatml|>`
# (id 151724). None of these three appear anywhere in the checkpoint's PROMPTING.md --
# no syntax, no example, no mention. They may be internal training-time scaffolding for
# background-audio labeling rather than a usable prompt-time control, but that is a guess,
# not a verified fact -- using an undocumented tag on a real book run would be a blind bet.
# Open question, not yet investigated: what (if anything) do `env:music`/`env:noise` do to
# generation, and is it worth a controlled probe? See m4-tag-inventory-results.md.

# Per PROMPTING.md (bosonai/higgs-tts-3-4b, verified against the cached snapshot):
# "pause" / "long_pause" are INLINE, one-shot effects at an exact position in a
# sentence -- they are not a sustained state and must never be reopened as one.
INLINE_ONE_SHOT_PROSODY = {"pause", "long_pause"}

# Per the same PROMPTING.md: ALL sfx tags are inline, one-shot events ("Insert at the
# exact position in the sentence where the effect should occur"), never a sustained
# state -- unlike emotion/prosody/style, there is no sentence-level sfx placement at
# all. Reopening an sfx tag at the start of every subsequent chunk/sentence would make a
# character cough or sneeze again at the start of each one, which is not what the source
# text authored. `active` (in chunk_sentences) therefore never gains an "sfx" key -- see
# the `category == "sfx"` skip below -- so _reopen_prefix can never reopen it.


def validate_control_tags(text: str) -> None:
    """Raise ValueError on any control-tag-shaped span that is not one of the 43 known-valid
    Higgs TTS 3 tags (F11). Without this, a typo like ``<|emotion:Elation|>`` silently fails
    to match TAG_RE, is never tracked as an active tag, and is read aloud as literal text (or
    silently dropped) for the rest of the chapter with no warning -- exactly the kind of
    defect that should fail before a multi-hour unattended run, not after it.
    """
    for m in _TAG_SHAPE_RE.finditer(text):
        candidate = m.group(0)
        if candidate not in VALID_TAGS:
            raise ValueError(
                f"unknown control tag {candidate!r} at text offset {m.start()} -- not one "
                f"of the {len(VALID_TAGS)} tags known from bosonai/higgs-tts-3-4b's "
                "tokenizer.json (added_tokens); fix the source text before starting a "
                "multi-hour run"
            )


# ---------------------------------------------------------------------------
# Stress-mark notation (owner-verified by ear, Refs #57, docs/research/audiobook/
# m4-tag-inventory-results.md sec. 3): an apostrophe placed directly after the
# stressed vowel (e.g. "за'мок") is the one notation the owner confirmed gives
# Higgs a stress cue without being read aloud or corrupting the transcript.
#
# `split_sentences`/`chunk_sentences`/`validate_control_tags` already treat a bare
# apostrophe (U+0027) as ordinary text -- it is not in PAIR_OPEN_TO_CLOSE (only the
# curly single quote pair '‘'/'’' is tracked there), it contains no
# SENTENCE_END_CHARS, and it does not match _TAG_SHAPE_RE -- so the notation already
# passes through the whole pipeline untouched (see tests below). What the pipeline
# does NOT do on its own is tell a stress-mark apostrophe apart from the *same*
# character used legitimately in a transliterated name ("д'Артаньян", "О'Генри"):
# both are just "'" to every function above.
#
# Disambiguation used here (documented compromise, not a proven rule -- see
# docs/guides/audiobook_guide.md "Расстановка ударений"): a "'" counts as a stress
# mark only when it sits strictly INSIDE one lowercase word -- immediately preceded
# by a lowercase Russian vowel AND immediately followed by a lowercase Russian
# letter continuing the same word. Both of the name examples above fail this test:
# "д'Артаньян" has a consonant before the apostrophe and an uppercase letter after
# it; "О'Генри" has a vowel before it but an uppercase letter after it. This rule is
# a heuristic on the two examples this project actually has, not a linguistic proof
# -- a name that happened to be lowercase on both sides of the apostrophe (none
# found in the sample text) would still be misread as a stress mark.
RUSSIAN_VOWELS_LOWER = "аеёиоуыэюя"
RUSSIAN_LETTERS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def is_stress_apostrophe(text: str, pos: int) -> bool:
    """True if the "'" at `text[pos]` looks like a stress mark, not a name apostrophe.

    See the module-level comment above `RUSSIAN_VOWELS_LOWER` for the exact rule and
    its known blind spot.
    """
    if text[pos] != "'":
        return False
    if pos == 0 or pos + 1 >= len(text):
        return False
    return text[pos - 1] in RUSSIAN_VOWELS_LOWER and text[pos + 1] in RUSSIAN_LETTERS_LOWER


def count_stress_marks(text: str) -> int:
    """Count apostrophes in `text` that match the stress-mark heuristic."""
    return sum(1 for i, ch in enumerate(text) if ch == "'" and is_stress_apostrophe(text, i))


def count_ambiguous_apostrophes(text: str) -> int:
    """Count apostrophes in `text` that do NOT match the stress-mark heuristic.

    Informational only (e.g. surfaced in the manifest) -- these are apostrophes
    `is_stress_apostrophe` treats as ordinary text (most likely a name like
    "д'Артаньян"), so a large count next to a small `count_stress_marks` count can
    hint that stress notation did not make it into a chapter as intended, or vice
    versa. Never raises and never blocks generation -- the heuristic has a known
    blind spot (see above) and must not be treated as ground truth.
    """
    return sum(
        1 for i, ch in enumerate(text) if ch == "'" and not is_stress_apostrophe(text, i)
    )


DEFAULT_SPEAKER = "narrator"


@dataclass
class Chunk:
    index: int
    sentences: list[str]
    reopened_tags: dict[str, str]
    text: str
    # Screenplay support (Refs #57): who speaks this chunk. Plain-text chunking (chunk_sentences
    # called directly, as from --text/--text-file) never sets this explicitly, so every such
    # chunk gets the same DEFAULT_SPEAKER -- segment hashes for plain-text runs are therefore
    # shifted by a constant, not made speaker-dependent in any way that matters for them.
    speaker: str = DEFAULT_SPEAKER


def _sentence_own_tags(sentence: str) -> list[tuple[str, str]]:
    return TAG_RE.findall(sentence)


def _reopen_prefix(active: dict[str, Optional[str]], sentence: str) -> str:
    """Build the tag prefix to prepend to `sentence` so active state survives.

    A category is skipped if `sentence` declares a tag of that category ANYWHERE in it
    (F10) -- not just at the very start -- since two tags of the same category in one
    sentence has no defined meaning in PROMPTING.md.
    """
    own_categories = {cat for cat, _ in TAG_RE.findall(sentence)}
    prefix_parts = []
    for category, tag in active.items():
        if tag is None:
            continue
        if category in own_categories:
            continue
        prefix_parts.append(tag)
    return "".join(prefix_parts)


def _force_split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split a single sentence that exceeds max_chars on its own (F2).

    `chunk_sentences` can never place a sentence that is already over budget into any
    chunk, so without this, one long sentence used to become one unbounded chunk (and, if
    it came from a broken split, potentially the entire remaining chapter). Tries `;`
    boundaries first, then `,`, then whitespace, then a hard character cut -- always makes
    progress, so this cannot infinite-loop or return a piece over max_chars.
    """
    sentence = sentence.strip()
    if not sentence:
        return []
    if len(sentence) <= max_chars:
        return [sentence]

    for sep in (";", ","):
        if sep in sentence:
            raw_parts = sentence.split(sep)
            rebuilt = []
            for idx, p in enumerate(raw_parts):
                p = p.strip()
                if not p:
                    continue
                rebuilt.append(p + sep if idx < len(raw_parts) - 1 else p)
            if len(rebuilt) > 1:
                out: list[str] = []
                for piece in rebuilt:
                    out.extend(_force_split_long_sentence(piece, max_chars))
                return out

    words = sentence.split(" ")
    if len(words) > 1:
        grouped: list[str] = []
        cur = ""
        for w in words:
            candidate = f"{cur} {w}".strip() if cur else w
            if len(candidate) > max_chars and cur:
                grouped.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            grouped.append(cur)
        if len(grouped) > 1:
            out = []
            for piece in grouped:
                out.extend(_force_split_long_sentence(piece, max_chars))
            return out

    # Last resort: a single unbroken token longer than max_chars -- hard character cut.
    # This is the guarantee that no piece this function returns ever exceeds max_chars.
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]


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

    Inline one-shot prosody (`pause`, `long_pause`) and all `sfx` tags are never
    reopened; each fires once at its authored position and carries no state.

    Any sentence longer than `max_chars` on its own is force-split (F2, see
    `_force_split_long_sentence`) with a warning, so `max_chars` is a real hard cap on
    every chunk's text, not merely a per-add check.
    """
    if tag_scope not in ("chunk", "sentence"):
        raise ValueError(f"unknown tag_scope: {tag_scope!r}")

    expanded: list[str] = []
    for sent in sentences:
        if len(sent) > max_chars:
            warnings.warn(
                f"sentence of {len(sent)} chars exceeds max_chars={max_chars}; "
                "force-splitting on ';'/','/whitespace boundaries -- audio at the forced "
                "split point(s) may sound less natural than a real sentence boundary",
                stacklevel=2,
            )
            expanded.extend(_force_split_long_sentence(sent, max_chars))
        else:
            expanded.append(sent)
    sentences = expanded

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
        own_tags = _sentence_own_tags(sent)
        if tag_scope == "sentence":
            prefix = _reopen_prefix(active, sent)
            sent_to_store = prefix + sent
        else:
            sent_to_store = sent
        # F8: count the length of what is actually stored (including any reopened-tag
        # prefix under tag_scope="sentence"), not the bare sentence -- otherwise the
        # budget check below silently underestimates a chunk's real size.
        store_len = len(sent_to_store)

        if cur_sentences and cur_len + 1 + store_len > max_chars:
            chunk = flush()
            if chunk is not None:
                chunks.append(chunk)
            chunk_start_active = dict(active)

        cur_sentences.append(sent_to_store)
        cur_len += store_len + 1

        for category, tag_name in own_tags:
            if category == "sfx":
                # Always inline/one-shot (PROMPTING.md) -- never tracked as active state,
                # so it can never be reopened at the top of a later chunk/sentence.
                continue
            if category == "prosody" and tag_name in INLINE_ONE_SHOT_PROSODY:
                continue
            active[category] = f"<|{category}:{tag_name}|>"

    if cur_sentences:
        chunk = flush()
        if chunk is not None:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Screenplay format (docs/guides/audiobook_guide.md sec. 3) -- reads the
# `[{"speaker": ..., "text": ...}, ...]` DSL into the same Chunk pipeline as plain text.
# ---------------------------------------------------------------------------


def parse_screenplay(data: object) -> list[dict]:
    """Validate and normalize a parsed screenplay JSON document into a plain list of
    ``{"speaker": str, "text": str}`` dicts.

    Deliberately keeps ONLY `speaker` and `text` from each line -- any other key a line
    happens to carry (an editor's note, a stable id, a source-page reference, ...) is
    dropped here, before chunking/hashing ever sees it, so editing such a field can never
    change a segment's content hash and trigger a needless regeneration (see
    `_segment_hash_input`).

    Raises ``ValueError`` (never silently drops/guesses) on:
      - the document not being a JSON array,
      - an empty array,
      - a line that is not an object,
      - a missing/blank/non-string `speaker` or `text`,
      - any control-tag-shaped span in `text` that is not one of the known-valid tags
        (delegates to `validate_control_tags`, same as plain-text mode).
    """
    if not isinstance(data, list):
        raise ValueError(
            "screenplay must be a JSON array of {'speaker': ..., 'text': ...} objects, got "
            f"{type(data).__name__}"
        )
    if not data:
        raise ValueError("screenplay is empty -- no lines to generate")

    lines: list[dict] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"screenplay line {i}: expected an object with 'speaker'/'text', got "
                f"{type(entry).__name__}"
            )
        speaker = entry.get("speaker")
        text = entry.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"screenplay line {i}: missing or blank 'speaker'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"screenplay line {i}: missing or blank 'text'")
        validate_control_tags(text)
        lines.append({"speaker": speaker.strip(), "text": text})
    return lines


def chunk_screenplay(
    lines: list[dict],
    max_chars: int = 500,
    tag_scope: str = "sentence",
) -> list[Chunk]:
    """Turn a parsed screenplay (`parse_screenplay`'s output) into a flat list of `Chunk`s
    using the exact same `split_sentences`/`chunk_sentences` machinery as plain text -- a
    long reply is still chunked under `max_chars` and its tags still reopened across chunk
    boundaries (F1-F11 all apply per line, unchanged).

    Each screenplay LINE is chunked independently: active emotion/prosody/style state is
    reset to "none active" at the start of every line, since a character's leftover
    emotional state carrying over onto a different speaker's first sentence has no basis in
    PROMPTING.md (tags are documented as attached to the speech they're written into, not to
    the chapter as a whole). A line that is itself long may still produce more than one
    `Chunk` (chunk_sentences' normal budget-splitting) -- those internal chunks keep that
    same line's tag-continuity as usual; only the FIRST chunk of each line is a genuine
    speaker-boundary chunk.

    Warns once (naming every distinct speaker) if the screenplay uses more than one speaker,
    since no code path downstream selects a per-speaker voice -- see this module's top
    docstring and docs/guides/audiobook_guide.md sec. 3a.
    """
    chunks: list[Chunk] = []
    speakers_seen: list[str] = []
    for line in lines:
        speaker = line["speaker"]
        if speaker not in speakers_seen:
            speakers_seen.append(speaker)
        sentences = split_sentences(line["text"])
        line_chunks = chunk_sentences(sentences, max_chars=max_chars, tag_scope=tag_scope)
        for lc in line_chunks:
            lc.speaker = speaker
            lc.index = len(chunks)
            chunks.append(lc)

    if len(speakers_seen) > 1:
        warnings.warn(
            f"screenplay has {len(speakers_seen)} distinct speakers {speakers_seen} but "
            "per-speaker voice cloning is NOT wired into generate_segments -- every line "
            "will be synthesized with the model's single default voice regardless of "
            "'speaker'. Per-character voice cloning has never been verified for a "
            "multi-voice book in this project; the one measurement that exists found "
            "cloned generation SLOWER than plain generation (RTF 7.73 vs. 6.56), not merely "
            "unimplemented. Wiring an actual per-speaker voice into generation is tracked "
            "as separate follow-up work -- see docs/guides/audiobook_guide.md sec. 3a.",
            stacklevel=2,
        )
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
    # F12: a WAV whose header claims more frames than the file actually contains (e.g.
    # truncated by a kill mid-write) used to be accepted silently.
    expected_bytes = n * sampwidth * channels
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path}: truncated WAV data -- header declares {n} frames "
            f"({expected_bytes} bytes) but only {len(raw)} bytes were read"
        )
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


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Voice reference (cloning): issue #57, "every fragment reads in a different voice".
#
# Without a reference, Higgs picks a voice by sampling from its speaker distribution as it
# autoregresses through each segment's own text -- independently per `model.generate()`/
# `model.batch_generate()` call, since nothing pins it across calls (fixing the random seed
# does NOT help here: a fixed-seed, six-phrase measurement still spanned 105.7-244.9 Hz
# median pitch, WORSE than the 82 Hz spread with no seed at all -- see
# docs/research/audiobook/voice-clone-consistency-results.md). The only lever that has ever
# been shown to pin the voice is `references`/`ref_audio_codes` -- an encoded reference
# waveform fed into *every* segment's generation call, not just the first.
#
# `VoiceReference.codes` is computed exactly ONCE per run (`resolve_voice_reference`, called
# from `main()` before the per-segment loop starts) and threaded down through
# `generate_segments` -> `_generate_single_segment`/`_generate_batch_group` -> every
# `model.generate()`/`model.batch_generate()` call as `ref_audio_codes=`/`ref_text=` --
# never recomputed per segment or per batch.
# ---------------------------------------------------------------------------

VOICES_DIR = Path("voices")


@dataclass
class VoiceReference:
    """One reference voice: pre-encoded codes (`model.encode_reference_audio()`'s output,
    or a `voices/<name>.npy` array loaded straight off disk -- both are accepted directly
    by `generate(..., ref_audio_codes=...)`) plus the reference transcript.

    `source` and `ref_text` (not the codes array itself, which is large and not worth
    hashing) are what identify this reference in the manifest header (`manifest_id()`) --
    see `load_or_create_manifest`'s mismatch check: swapping the reference on an
    in-progress chapter must invalidate the whole manifest, the same way changing
    `max_chars`/`tag_scope`/`model` already does, because the reference is a property of
    the *run*, not of any one segment -- a resumed run that silently kept old segments
    generated against a different voice would quietly re-introduce exactly the
    inconsistent-voice bug this feature exists to fix.
    """

    name: str
    ref_text: str
    codes: object  # mx.array (or a plain np.ndarray straight from voices/<name>.npy)
    source: str  # "voices/<name>.npy" or the literal --ref-audio path, for diagnostics

    def manifest_id(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "ref_text_hash": _text_hash(self.ref_text),
        }


def load_voice_from_registry(
    name: str, voices_dir: Path = VOICES_DIR
) -> tuple[np.ndarray, str, str]:
    """Load a voice already registered per docs/guides/audiobook_guide.md sec. 2
    (`register_voice()` -> `voices/<name>.npy` + `voices/<name>.txt`).

    Returns (codes_array, ref_text, source_description). Raises `FileNotFoundError` naming
    exactly which of the two files is missing, rather than a generic I/O error -- a chapter
    run should fail fast and legibly if the voice was never registered, or was registered
    under a different name.
    """
    npy_path = voices_dir / f"{name}.npy"
    txt_path = voices_dir / f"{name}.txt"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"voice {name!r} not found: {npy_path} does not exist -- register it first "
            "(docs/guides/audiobook_guide.md sec. 2, register_voice(), or "
            "--ref-audio/--ref-text with --save-voice-as to do it from this CLI)"
        )
    if not txt_path.exists():
        raise FileNotFoundError(
            f"voice {name!r} is missing its reference transcript: {txt_path} does not exist"
        )
    codes = np.load(npy_path)
    ref_text = txt_path.read_text(encoding="utf-8").strip()
    return codes, ref_text, f"voices/{name}.npy"


def register_voice(
    model, name: str, ref_audio_path: Path, ref_text: str, voices_dir: Path = VOICES_DIR
) -> VoiceReference:
    """Encode `ref_audio_path` once and save it as `voices/<name>.npy` + `.txt`, exactly
    the format docs/guides/audiobook_guide.md sec. 2's standalone Python snippet writes --
    this is the same mechanism, callable from the CLI (`--save-voice-as`) instead of a
    separate one-off script, so there is exactly one registered-voice format, not two.
    """
    voices_dir.mkdir(parents=True, exist_ok=True)
    codes = model.encode_reference_audio(str(ref_audio_path))
    np.save(voices_dir / f"{name}.npy", np.array(codes))
    (voices_dir / f"{name}.txt").write_text(ref_text, encoding="utf-8")
    return VoiceReference(
        name=name, ref_text=ref_text, codes=codes, source=f"voices/{name}.npy"
    )


def resolve_voice_reference(
    model,
    voice_name: Optional[str],
    ref_audio: Optional[Path],
    ref_text: Optional[str],
    save_voice_as: Optional[str] = None,
    voices_dir: Path = VOICES_DIR,
) -> Optional["VoiceReference"]:
    """Resolve `--voice-name`/`--ref-audio`+`--ref-text` into a `VoiceReference`, or `None`
    if neither was given (unchanged, no-reference behavior). `model` is only used to encode
    a raw `--ref-audio` waveform -- a `--voice-name` load is pre-encoded already (`.npy` on
    disk), so it costs nothing beyond a file read, which is also why `--voice-name` is the
    cheaper, preferred path for any voice reused across more than one run.
    """
    if voice_name is not None and ref_audio is not None:
        raise ValueError("--voice-name and --ref-audio are mutually exclusive")
    if voice_name is not None:
        codes, text, source = load_voice_from_registry(voice_name, voices_dir)
        return VoiceReference(name=voice_name, ref_text=text, codes=codes, source=source)
    if ref_audio is not None:
        if not ref_text:
            raise ValueError("--ref-audio requires --ref-text (or --ref-text-file)")
        name = save_voice_as or ref_audio.stem
        if save_voice_as is not None:
            return register_voice(model, name, ref_audio, ref_text, voices_dir)
        codes = model.encode_reference_audio(str(ref_audio))
        return VoiceReference(name=name, ref_text=ref_text, codes=codes, source=str(ref_audio))
    return None


def _segment_hash_input(chunk: "Chunk") -> str:
    """Everything about `chunk` that can change the audio it generates: who speaks it
    (`speaker` -- selects the voice, once per-speaker voices exist) and its final text
    (already includes any reopened emotion/prosody/style tag prefix). Nothing else --
    `chunk.index`, or any field a screenplay line originally carried besides speaker/text
    (`parse_screenplay` already dropped those) -- is allowed to affect this, so an edit to
    an unrelated field, or a chunk simply shifting position in the chapter, never forces a
    needless regeneration of an otherwise-unchanged line (F7's per-segment content hash,
    extended to cover `speaker`).

    Uses a \\x1f (unit separator, cannot occur in the input text) join so that e.g.
    speaker="ab" + text="cd" cannot collide with speaker="a" + text="bcd".
    """
    return f"{chunk.speaker}\x1f{chunk.text}"


def _new_segment_entry(chunk: "Chunk") -> dict:
    text_hash = _text_hash(_segment_hash_input(chunk))
    return {
        "index": chunk.index,
        "text_hash": text_hash,
        "speaker": chunk.speaker,
        "sentences": chunk.sentences,
        "text": chunk.text,
        "reopened_tags": chunk.reopened_tags,
        "status": "pending",
        # F7: relative to the manifest's own directory, and named by content hash so an
        # untouched segment keeps its file even if surrounding chunks shift index.
        "output_path": f"segment_{text_hash}.wav",
        "sample_rate": None,
        "num_samples": None,
        "audio_duration_seconds": None,
        "generation_seconds": None,
        "memory": None,
        "error": None,
    }


def build_manifest(
    chunks: list[Chunk],
    max_chars: int,
    tag_scope: str,
    voice_reference: Optional["VoiceReference"] = None,
) -> dict:
    # Auto-detected, not author-set (Refs #57): the author of a screenplay/chapter has no
    # other way to tell a later reader/tool whether stress marks were already placed in the
    # text. Recording the counts here -- rather than requiring a manual flag the author could
    # forget to pass -- means the manifest itself answers "was this text stress-marked?" for
    # anyone inspecting it later (e.g. before spending an evening hand-marking a chapter
    # that already has marks, or before assuming an unmarked chapter is done). Counts, not a
    # single bool, so a mostly-marked chapter with a few misses is visible as such.
    full_text = "\n".join(c.text for c in chunks)
    return {
        "model": MODEL_ID,
        "max_chars": max_chars,
        "tag_scope": tag_scope,
        # Refs #57: the voice reference is a property of the whole run, not of any one
        # segment -- see VoiceReference's docstring for why this must invalidate resume on
        # mismatch the same way model/max_chars/tag_scope already do (load_or_create_manifest).
        "voice_reference": voice_reference.manifest_id() if voice_reference else None,
        "created": time.time(),
        "stress_marks_detected": count_stress_marks(full_text),
        "ambiguous_apostrophes_detected": count_ambiguous_apostrophes(full_text),
        "segments": [_new_segment_entry(c) for c in chunks],
    }


def _load_manifest_with_recovery(manifest_path: Path) -> dict:
    """Load the manifest, recovering from `.bak` on a JSONDecodeError (F4).

    A `save_manifest` writes atomically and keeps the previous good version as `.bak`, so a
    kill mid-write can only ever leave the *new* temp file incomplete (and it is never
    renamed into place until fsynced) -- but a manifest from before this fix, or corruption
    from any other cause, is still handled here rather than crashing on `json.loads`.
    """
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        bak_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        if not bak_path.exists():
            raise RuntimeError(
                f"manifest {manifest_path} is corrupt ({exc}) and no .bak backup exists -- "
                "cannot recover automatically; the segment WAVs on disk are still there but "
                "the manifest linking them to text/order is gone"
            ) from exc
        try:
            recovered = json.loads(bak_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as bak_exc:
            raise RuntimeError(
                f"manifest {manifest_path} is corrupt ({exc}) and its .bak backup "
                f"{bak_path} is also corrupt ({bak_exc}) -- cannot recover automatically"
            ) from bak_exc
        # Restore the recovered content as the primary file so the next save_manifest()
        # call has a consistent starting point instead of re-reading the corrupt one.
        manifest_path.write_text(bak_path.read_text(encoding="utf-8"), encoding="utf-8")
        return recovered


def load_or_create_manifest(
    manifest_path: Path,
    chunks: list[Chunk],
    max_chars: int,
    tag_scope: str,
    voice_reference: Optional["VoiceReference"] = None,
) -> dict:
    if manifest_path.exists():
        manifest = _load_manifest_with_recovery(manifest_path)

        # F7: compare the settings that change every segment's text, and name exactly
        # which one differs instead of a blanket "text doesn't match" error.
        mismatches = []
        if manifest.get("model") != MODEL_ID:
            mismatches.append(f"model: manifest={manifest.get('model')!r} vs current={MODEL_ID!r}")
        if manifest.get("max_chars") != max_chars:
            mismatches.append(
                f"max_chars: manifest={manifest.get('max_chars')!r} vs current={max_chars!r}"
            )
        if manifest.get("tag_scope") != tag_scope:
            mismatches.append(
                f"tag_scope: manifest={manifest.get('tag_scope')!r} vs current={tag_scope!r}"
            )
        # Refs #57: a manifest built with one reference voice (or none) must never be
        # silently resumed under a different one -- every "done" segment on disk was
        # generated against the OLD reference, so blind resume would splice a chapter
        # containing two different voices at whatever point the reference was swapped,
        # exactly reproducing the bug this feature exists to fix, just quietly instead of
        # audibly. `.get("voice_reference")` on an older manifest predating this field
        # reads as `None`, which correctly compares unequal to any real reference below.
        current_ref_id = voice_reference.manifest_id() if voice_reference else None
        if manifest.get("voice_reference") != current_ref_id:
            mismatches.append(
                f"voice_reference: manifest={manifest.get('voice_reference')!r} vs "
                f"current={current_ref_id!r}"
            )
        if mismatches:
            raise RuntimeError(
                "Existing manifest was built with different settings, which changes every "
                "segment's text -- refusing to resume blindly: " + "; ".join(mismatches)
            )

        # F7: reuse any segment whose own text is unchanged (keyed by content hash), no
        # matter where it now sits in the chapter, instead of requiring the entire plan to
        # match byte-for-byte (which previously discarded 40 hours of prior work for a
        # one-character edit anywhere in the chapter).
        existing_by_hash = {
            seg["text_hash"]: seg for seg in manifest["segments"] if "text_hash" in seg
        }
        new_segments = []
        for c in chunks:
            text_hash = _text_hash(_segment_hash_input(c))
            prior = existing_by_hash.get(text_hash)
            if prior is not None:
                seg = dict(prior)
                seg["index"] = c.index
                seg["speaker"] = c.speaker
                seg["sentences"] = c.sentences
                seg["reopened_tags"] = c.reopened_tags
                seg["output_path"] = f"segment_{text_hash}.wav"
            else:
                seg = _new_segment_entry(c)
            new_segments.append(seg)
        manifest["segments"] = new_segments
        # Refresh on every resume, not just at first creation, so an edit that adds/removes
        # stress marks in an already-in-progress chapter is reflected here too (only the
        # touched segment's text_hash changes and gets regenerated -- these counts are purely
        # informational and never gate resume).
        full_text = "\n".join(c.text for c in chunks)
        manifest["stress_marks_detected"] = count_stress_marks(full_text)
        manifest["ambiguous_apostrophes_detected"] = count_ambiguous_apostrophes(full_text)
        save_manifest(manifest, manifest_path)
        return manifest

    manifest = build_manifest(chunks, max_chars, tag_scope, voice_reference=voice_reference)
    save_manifest(manifest, manifest_path)
    return manifest


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    """Write the manifest atomically (F4).

    Writes to a temp file in the same directory, `flush()` + `fsync()`, then
    `os.replace()`s it into place, keeping the previous version as `.bak`. A kill at any
    point during this either leaves the old manifest untouched (temp file incomplete, never
    renamed) or leaves the new one complete (renamed only after fsync) -- there is no window
    where `manifest_path` itself is a truncated file, unlike the old `path.write_text(...)`
    in-place truncate.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if manifest_path.exists():
        bak_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        try:
            os.replace(manifest_path, bak_path)
        except OSError:
            pass
    os.replace(tmp_path, manifest_path)


MIN_SECONDS_PER_CHAR = 0.03
MAX_SECONDS_PER_CHAR = 0.30


def _validate_generated_audio(audio: np.ndarray, sample_rate: int, text: str) -> Optional[str]:
    """Return None if `audio` plausibly matches `text`'s length, else a reason string (F6).

    A `done` segment used to be trusted purely on `out_path.exists()` -- an empty array
    (0 samples) or a wildly implausible duration (a runaway/looping generation, or a
    silent truncation at `max_new_tokens`) was written and accepted with no complaint.
    """
    if audio.size == 0:
        return "generated audio is empty (0 samples)"
    duration = len(audio) / sample_rate if sample_rate else 0.0
    text_len = max(len(text), 1)
    min_expected = MIN_SECONDS_PER_CHAR * text_len
    max_expected = MAX_SECONDS_PER_CHAR * text_len
    if duration < min_expected:
        return (
            f"generated audio ({duration:.3f}s) is implausibly short for {text_len} chars "
            f"of text (expected >= {min_expected:.3f}s) -- likely a truncated/aborted "
            "generation"
        )
    if duration > max_expected:
        return (
            f"generated audio ({duration:.3f}s) is implausibly long for {text_len} chars "
            f"of text (expected <= {max_expected:.3f}s) -- likely a looping/runaway "
            "generation"
        )
    return None


def _resume_check_ok(entry: dict, out_path: Path) -> tuple[bool, Optional[str]]:
    """Whether a segment marked `done` actually has valid, matching audio on disk (F6)."""
    if not out_path.exists():
        return False, "output file missing"
    try:
        with wave.open(str(out_path), "rb") as w:
            actual_samples = w.getnframes()
            actual_sr = w.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        return False, f"could not read existing WAV: {exc}"
    if actual_samples == 0:
        return False, "existing WAV has 0 frames"
    expected_samples = entry.get("num_samples")
    if expected_samples is not None and actual_samples != expected_samples:
        return False, f"manifest expects {expected_samples} samples, file has {actual_samples}"
    expected_sr = entry.get("sample_rate")
    if expected_sr is not None and actual_sr != expected_sr:
        return False, f"manifest expects {expected_sr} Hz, file is {actual_sr} Hz"
    return True, None


def _capture_memory_metrics() -> dict:
    """Best-effort MLX memory metrics per segment (F14). Never raises: returns all-None
    fields when `mlx` is unavailable (e.g. running the pure-Python paths under test)."""
    metrics: dict = {
        "active_memory_bytes": None,
        "peak_memory_bytes": None,
        "cache_memory_bytes": None,
    }
    try:
        import mlx.core as mx  # type: ignore

        metrics["active_memory_bytes"] = int(mx.get_active_memory())
        metrics["peak_memory_bytes"] = int(mx.get_peak_memory())
        metrics["cache_memory_bytes"] = int(mx.get_cache_memory())
    except Exception:
        pass
    return metrics


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx  # type: ignore

        mx.clear_cache()
    except Exception:
        pass


def _generate_single_segment(
    model,
    entry: dict,
    out_path: Path,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    retry_base_delay: float,
    manifest: dict,
    manifest_path: Path,
    ref_audio_codes=None,
    ref_text: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Generate one segment via `model.generate()` with retry/backoff (F5) and per-segment
    audio-sanity validation (F6). Mutates `entry` in place and calls `save_manifest` after
    every status change, so a kill at any point leaves the manifest resumable (F4).

    This is the exact single-segment call convention `generate_segments` always used before
    batching existed (`--batch-size 1` still goes through this function only, unchanged) --
    factored out so the batched path (`_generate_batch_group` below) can also fall back to
    it, one segment at a time, when a batch cannot be trusted as a whole.

    `ref_audio_codes`/`ref_text` (Refs #57): the pre-encoded voice reference, computed once
    per run by `resolve_voice_reference` and passed to EVERY segment's call here -- this is
    the actual fix for "every fragment reads in a different voice" (see VoiceReference's
    docstring). `None` (the default) is the old, unchanged, no-reference behavior.
    """
    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        entry["status"] = "in_progress"
        save_manifest(manifest, manifest_path)
        try:
            started = time.perf_counter()
            results = list(
                model.generate(
                    text=entry["text"],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    ref_audio_codes=ref_audio_codes,
                    ref_text=ref_text,
                )
            )
            generation_seconds = time.perf_counter() - started
            if not results:
                raise RuntimeError("model.generate produced no result")
            sample_rate = results[0].sample_rate
            audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
            bad_reason = _validate_generated_audio(audio, sample_rate, entry["text"])
            if bad_reason:
                raise RuntimeError(bad_reason)
            write_wav(out_path, audio, sample_rate)
            entry["status"] = "done"
            entry["sample_rate"] = sample_rate
            entry["num_samples"] = int(len(audio))
            entry["audio_duration_seconds"] = len(audio) / sample_rate if sample_rate else None
            entry["generation_seconds"] = generation_seconds
            entry["memory"] = _capture_memory_metrics()
            entry["error"] = None
            save_manifest(manifest, manifest_path)
            return True, None
        except Exception as exc:  # noqa: BLE001 -- must record and keep the manifest resumable
            last_error = f"{type(exc).__name__}: {exc}"
            entry["error"] = last_error
            if attempt < max_retries:
                entry["status"] = "pending"  # explicit reset before retrying (F5)
                save_manifest(manifest, manifest_path)
                time.sleep(retry_base_delay * (2 ** (attempt - 1)))
                continue
            entry["status"] = "failed"
            save_manifest(manifest, manifest_path)
    return False, last_error


def _generate_batch_group(
    model,
    group: list[tuple[int, dict]],
    base_dir: Path,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    retry_base_delay: float,
    manifest: dict,
    manifest_path: Path,
    ref_audio_codes=None,
    ref_text: Optional[str] = None,
) -> list[tuple[int, dict, bool, Optional[str]]]:
    """Generate every entry in `group` (list of `(seg_i, entry)`, len <= --batch-size)
    through `model.batch_generate()`, with a self-narrowing fallback on failure.

    Segment identity across the batch boundary (this task's most dangerous spot): `group`
    preserves manifest order, so position `pos` in `texts` is passed to `batch_generate` and
    the returned `BatchGenerationResult.sequence_idx` is used -- never return order -- to map
    each decoded result back to `group[pos]`'s own entry. `sequence_idx` equals `pos` in the
    current mlx_audio implementation (both are built from the same `enumerate(texts)`), but
    this still keys off `sequence_idx` explicitly, not off yield order, in case a future
    mlx_audio revision reorders yields (continuous batching evicts finished rows early) --
    exactly the m4_batching_bench.py convention (`chunk_results.sort(key=lambda r:
    r.sequence_idx)`).

    Manifest writes happen per segment, not once per batch -- but this does NOT bound a kill
    mid-batch to losing "at most one segment" (an earlier version of this docstring, and of
    `docs/research/audiobook/m4-batching-integration-results.md`, claimed exactly that; both
    were wrong -- see Refs #114, "batching kill-loses-a-whole-batch" follow-up). Measured
    fact, both by reading `mlx_audio`'s `Higgs*V3.batch_generate()` and by a real
    kill-mid-batch experiment: the whole `for _ in range(limit): ...` decode loop for every
    row in the batch runs to completion (or hits the token limit) BEFORE the function's
    single `for state in states: yield BatchGenerationResult(...)` loop runs at all -- so
    nothing is yielded until the entire batch has finished decoding. On top of that,
    `_generate_batch_group` itself drains the whole `model.batch_generate()` generator into
    `result_by_pos` (the loop right below this docstring) before writing or saving anything;
    the per-segment `write_wav`/`save_manifest` calls happen in a second loop afterwards, so
    even a truly-incremental `batch_generate()` would still be fully buffered here. A kill
    during the (dominant, GPU-bound) generation phase therefore loses the *entire* batch --
    up to `--batch-size` segments, not one. The per-segment writes in the second loop only
    protect against a kill during that fast, CPU-bound tail (numpy write + JSON dump), which
    is a tiny fraction of a batch's wall time.

    Retries and isolation (F5, extended to batches): a whole-batch exception (or a
    per-segment audio-sanity failure inside it, F6) is retried whole up to `max_retries`
    times, since `batch_generate`'s shared forward pass gives no way to blame one row before
    the batch completes. If the batch keeps failing at that size, the *unfinished* entries
    (already-done ones are kept, not redone) are split in half and each half is retried
    independently, recursing down to single-segment `_generate_single_segment` calls -- this
    isolates exactly the segment(s) actually at fault instead of writing off the whole batch,
    and doubles as automatic degradation toward a smaller effective batch size if the failure
    is memory pressure (F6/"memory ceiling" risk) rather than a bad segment.

    `ref_audio_codes`/`ref_text` (Refs #57): passed as a single shared value (not a
    per-row list) to `model.batch_generate()` -- `_normalize_batch_references` in
    `higgs_audio_v3/model.py` encodes/broadcasts a shared reference exactly once across
    every row of the batch when it is given this way, so voice cloning does not cost an
    extra encode per row and does not disable batching (verified in
    docs/research/audiobook/voice-clone-consistency-results.md's batching-compatibility
    measurement).
    """
    if len(group) == 1:
        seg_i, entry = group[0]
        out_path = base_dir / entry["output_path"]
        ok, err = _generate_single_segment(
            model, entry, out_path, temperature, max_new_tokens, max_retries,
            retry_base_delay, manifest, manifest_path,
            ref_audio_codes=ref_audio_codes, ref_text=ref_text,
        )
        return [(seg_i, entry, ok, err)]

    for _, entry in group:
        entry["status"] = "in_progress"
    save_manifest(manifest, manifest_path)

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        # Recomputed every attempt: a segment already written to "done" earlier in a
        # previous partially-successful attempt (see the per-row loop below, which can
        # raise partway through) is never resent to batch_generate again on retry.
        remaining_now = [(seg_i, entry) for seg_i, entry in group if entry["status"] != "done"]
        if not remaining_now:
            break
        texts = [entry["text"] for _, entry in remaining_now]
        try:
            result_by_pos: dict[int, object] = {}
            for r in model.batch_generate(
                texts=texts, temperature=temperature, max_new_tokens=max_new_tokens,
                ref_audio_codes=ref_audio_codes, ref_text=ref_text,
            ):
                result_by_pos[r.sequence_idx] = r
            if len(result_by_pos) != len(remaining_now):
                raise RuntimeError(
                    f"batch_generate returned {len(result_by_pos)} result(s) for "
                    f"{len(remaining_now)} requested segment(s)"
                )
            for pos, (seg_i, entry) in enumerate(remaining_now):
                r = result_by_pos[pos]
                out_path = base_dir / entry["output_path"]
                audio = np.asarray(r.audio).reshape(-1)
                sample_rate = r.sample_rate
                # F6, applied per segment, not to the batch as a whole -- one bad row in an
                # otherwise-fine batch must not silently pass because the other rows are fine.
                bad_reason = _validate_generated_audio(audio, sample_rate, entry["text"])
                if bad_reason:
                    raise RuntimeError(f"segment {entry['index']}: {bad_reason}")
                write_wav(out_path, audio, sample_rate)
                entry["status"] = "done"
                entry["sample_rate"] = sample_rate
                entry["num_samples"] = int(len(audio))
                entry["audio_duration_seconds"] = len(audio) / sample_rate if sample_rate else None
                entry["generation_seconds"] = getattr(r, "processing_time_seconds", None)
                entry["memory"] = _capture_memory_metrics()
                entry["error"] = None
                save_manifest(manifest, manifest_path)  # per segment, not per batch
        except Exception as exc:  # noqa: BLE001 -- must record and keep the manifest resumable
            last_error = f"{type(exc).__name__}: {exc}"
            for _, entry in remaining_now:
                if entry["status"] != "done":
                    entry["error"] = last_error
            if attempt < max_retries:
                for _, entry in remaining_now:
                    if entry["status"] != "done":
                        entry["status"] = "pending"
                save_manifest(manifest, manifest_path)
                time.sleep(retry_base_delay * (2 ** (attempt - 1)))
                continue
            break  # exhausted whole-batch retries at this size -- degrade below

    already_done = [
        (seg_i, entry, True, None) for seg_i, entry in group if entry["status"] == "done"
    ]
    remaining = [(seg_i, entry) for seg_i, entry in group if entry["status"] != "done"]
    if not remaining:
        return already_done
    mid = max(1, len(remaining) // 2)
    left = _generate_batch_group(
        model, remaining[:mid], base_dir, temperature, max_new_tokens, max_retries,
        retry_base_delay, manifest, manifest_path,
        ref_audio_codes=ref_audio_codes, ref_text=ref_text,
    )
    right = (
        _generate_batch_group(
            model, remaining[mid:], base_dir, temperature, max_new_tokens, max_retries,
            retry_base_delay, manifest, manifest_path,
            ref_audio_codes=ref_audio_codes, ref_text=ref_text,
        )
        if remaining[mid:]
        else []
    )
    return already_done + left + right


def generate_segments(
    model,
    manifest: dict,
    manifest_path: Path,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    continue_on_error: bool = False,
    clear_cache_every: int = 50,
    batch_size: int = 1,
    ref_audio_codes=None,
    ref_text: Optional[str] = None,
) -> dict:
    """Generate every pending segment, writing progress to disk after each one.

    `ref_audio_codes`/`ref_text` (Refs #57): a pre-encoded voice reference (see
    `VoiceReference`/`resolve_voice_reference`), computed exactly once by the caller and
    passed to EVERY segment's `model.generate()`/`model.batch_generate()` call below --
    both the unbatched and batched paths. `None` (default) reproduces the old,
    no-reference behavior exactly.

    On restart, a segment already marked "done" is re-validated against the WAV actually on
    disk (F6) -- a missing file, an empty/corrupt WAV, or a sample count/rate mismatch
    against the manifest resets it to "pending" instead of being trusted blindly. A segment
    left "in_progress" by a killed run is handled identically to "pending" here -- neither
    status short-circuits the loop below, both just fall through to (re)generation.

    `batch_size=1` (default) is the exact original per-segment path: one `model.generate()`
    call per segment, in order, via `_generate_single_segment`. `batch_size > 1` groups the
    segments that still need generating (after the resume check above) into groups of at
    most `batch_size` and generates each group through `model.batch_generate()` via
    `_generate_batch_group`, which maps results back to segments by `sequence_idx` (never by
    return order) and falls back to smaller groups / single-segment generation on failure --
    see that function's docstring for the full retry/isolation contract. Both paths write the
    manifest after every individual segment completes, validate every individual segment's
    audio before accepting it, and clear the MLX cache every `clear_cache_every` *segments
    actually generated* (a resumed/skipped segment does not count, matching the pre-batching
    behavior) -- batching only changes how many segments are requested from the model per
    call, not any of the per-segment bookkeeping around it.

    On a generation failure, the segment is retried (with exponential backoff) up to
    `max_retries` times -- per segment when `batch_size=1`, per batch (then per smaller
    sub-batch, then per segment) when `batch_size>1`. If every retry is exhausted for a given
    segment: under `continue_on_error=True` it is marked "failed" and the loop keeps going
    (losing only that segment, not the rest of a multi-hour run); otherwise the original
    all-or-nothing behavior is preserved and the exception is re-raised immediately.

    Returns {"failures": [{"index": ..., "error": ...}, ...]} -- empty when everything
    either succeeded or was skipped via a valid resume.
    """
    base_dir = manifest_path.parent
    failures: list[dict] = []

    if batch_size <= 1:
        processed = 0
        for entry in manifest["segments"]:
            out_path = base_dir / entry["output_path"]

            if entry["status"] == "done":
                ok, reason = _resume_check_ok(entry, out_path)
                if ok:
                    continue
                entry["status"] = "pending"
                entry["error"] = f"resume check failed, regenerating: {reason}"

            succeeded, last_error = _generate_single_segment(
                model, entry, out_path, temperature, max_new_tokens, max_retries,
                retry_base_delay, manifest, manifest_path,
                ref_audio_codes=ref_audio_codes, ref_text=ref_text,
            )
            if not succeeded:
                if not continue_on_error:
                    raise RuntimeError(
                        f"segment {entry['index']} failed after {max_retries} attempt(s): "
                        f"{last_error} -- pass --continue-on-error to keep going and report "
                        "failed segments at the end"
                    )
                failures.append({"index": entry["index"], "error": last_error})

            processed += 1
            if clear_cache_every and processed % clear_cache_every == 0:
                _clear_mlx_cache()

        return {"failures": failures}

    # batch_size > 1: resume-check every segment first (unchanged rule), then hand only the
    # segments that still need generating to the batched path, `batch_size` at a time.
    pending: list[tuple[int, dict]] = []
    for seg_i, entry in enumerate(manifest["segments"]):
        out_path = base_dir / entry["output_path"]
        if entry["status"] == "done":
            ok, reason = _resume_check_ok(entry, out_path)
            if ok:
                continue
            entry["status"] = "pending"
            entry["error"] = f"resume check failed, regenerating: {reason}"
        pending.append((seg_i, entry))

    processed = 0
    for group_start in range(0, len(pending), batch_size):
        group = pending[group_start : group_start + batch_size]
        outcomes = _generate_batch_group(
            model, group, base_dir, temperature, max_new_tokens, max_retries,
            retry_base_delay, manifest, manifest_path,
            ref_audio_codes=ref_audio_codes, ref_text=ref_text,
        )
        for _seg_i, entry, succeeded, last_error in outcomes:
            if not succeeded:
                if not continue_on_error:
                    raise RuntimeError(
                        f"segment {entry['index']} failed after retries: {last_error} -- "
                        "pass --continue-on-error to keep going and report failed segments "
                        "at the end"
                    )
                failures.append({"index": entry["index"], "error": last_error})

            processed += 1
            if clear_cache_every and processed % clear_cache_every == 0:
                _clear_mlx_cache()

    return {"failures": failures}


# ---------------------------------------------------------------------------
# Assembly (separate step) + numeric splice-quality check
# ---------------------------------------------------------------------------


def assemble_chapter(
    manifest: dict,
    output_path: Path,
    base_dir: Path,
    silence_ms: int = 200,
    allow_gaps: bool = False,
    gap_silence_ms: int = 1000,
    speaker_change_silence_ms: Optional[int] = None,
) -> dict:
    """Stream-assemble the chapter from per-segment WAVs (F3, F13).

    Only one segment's audio is ever held in memory at a time -- the output file is opened
    once and written to with `writeframes` per segment, in int16, with no intermediate
    `np.concatenate` over the whole chapter. Splice-quality metrics are computed from the
    small edge windows of consecutive segments only, never from the full chapter buffer.

    With `allow_gaps=False` (default), any non-`done` segment raises before anything is
    written (F13: fail fast, not after writing most of a chapter). With `allow_gaps=True`,
    each non-`done` segment is replaced with `gap_silence_ms` of silence and reported in
    `gaps`, once the sample rate is known from a real segment; a run of gaps before the
    first real segment cannot be sample-rate-stamped and is reported but not written.

    `speaker_change_silence_ms` (screenplay format, Refs #57): when given, the join silence
    before a segment uses this duration instead of `silence_ms` if its manifest entry's
    `speaker` differs from the previous non-gap segment's `speaker` -- matching
    docs/guides/audiobook_guide.md sec. 5's pause table, which recommends a longer pause on a
    speaker change than between sentences of the same speaker. A manifest entry with no
    `speaker` key (plain-text runs predating this feature) is treated as unchanged from
    whatever the previous entry's speaker was, so `silence_ms` alone still governs those.
    """
    segments = manifest["segments"]
    if not segments:
        raise RuntimeError("manifest has no segments to assemble")

    not_done = [s for s in segments if s["status"] != "done"]
    if not_done and not allow_gaps:
        first = not_done[0]
        raise RuntimeError(
            f"{len(not_done)} segment(s) are not done (first: index={first['index']}, "
            f"status={first['status']!r}) -- cannot assemble a chapter with missing "
            "segments; pass allow_gaps=True / --allow-gaps to insert silence and report "
            "the gaps instead"
        )

    sample_rate: Optional[int] = None
    join_reports: list[dict] = []
    gap_reports: list[dict] = []
    pending_gap_segments: list[dict] = []
    edge_window_ms = 20
    total_samples = 0
    prev_tail: Optional[np.ndarray] = None
    prev_index: Optional[int] = None
    prev_speaker: Optional[str] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)

        for entry in segments:
            if entry["status"] != "done":
                if sample_rate is None:
                    # Cannot write silence of the right length/framerate before the output
                    # framerate is known; report it, do not fabricate an assumed rate.
                    pending_gap_segments.append(entry)
                    continue
                gap_samples = int(gap_silence_ms / 1000 * sample_rate)
                out.writeframes(np.zeros(gap_samples, dtype=np.int16).tobytes())
                total_samples += gap_samples
                gap_reports.append(
                    {
                        "index": entry["index"],
                        "status": entry["status"],
                        "inserted_silence_ms": gap_silence_ms,
                    }
                )
                prev_tail = None  # no meaningful edge to join across a gap
                prev_index = entry["index"]
                prev_speaker = None  # speaker on the far side of a gap is unknown here
                continue

            audio, sr = read_wav(base_dir / entry["output_path"])
            if sample_rate is None:
                sample_rate = sr
                out.setframerate(sample_rate)
                for pending in pending_gap_segments:
                    gap_reports.append(
                        {
                            "index": pending["index"],
                            "status": pending["status"],
                            "inserted_silence_ms": 0,
                            "note": "before the first done segment -- sample rate unknown, "
                            "no silence could be written for it",
                        }
                    )
                pending_gap_segments = []
            elif sr != sample_rate:
                raise RuntimeError(
                    f"sample rate mismatch: segment {entry['index']} is {sr} Hz, "
                    f"expected {sample_rate} Hz"
                )

            edge_n = int(edge_window_ms / 1000 * sample_rate)
            if prev_tail is not None:
                this_head = audio[:edge_n] if len(audio) >= edge_n else audio
                prev_edge_amp = float(np.max(np.abs(prev_tail))) if len(prev_tail) else 0.0
                head_edge_amp = float(np.max(np.abs(this_head))) if len(this_head) else 0.0
                max_intra_jump = float(
                    max(
                        np.max(np.abs(np.diff(prev_tail))) if len(prev_tail) > 1 else 0.0,
                        np.max(np.abs(np.diff(this_head))) if len(this_head) > 1 else 0.0,
                    )
                )
                direct_join_jump = (
                    float(abs(this_head[0] - prev_tail[-1]))
                    if len(prev_tail) and len(this_head)
                    else 0.0
                )
                join_reports.append(
                    {
                        "after_segment": prev_index,
                        "before_segment": entry["index"],
                        "prev_tail_edge_abs_amplitude": prev_edge_amp,
                        "next_head_edge_abs_amplitude": head_edge_amp,
                        "max_intra_window_sample_jump": max_intra_jump,
                        "direct_join_sample_jump": direct_join_jump,
                    }
                )
                this_speaker = entry.get("speaker")
                effective_silence_ms = silence_ms
                if (
                    speaker_change_silence_ms is not None
                    and prev_speaker is not None
                    and this_speaker is not None
                    and this_speaker != prev_speaker
                ):
                    effective_silence_ms = speaker_change_silence_ms
                if effective_silence_ms > 0:
                    silence_samples = int(effective_silence_ms / 1000 * sample_rate)
                    out.writeframes(np.zeros(silence_samples, dtype=np.int16).tobytes())
                    total_samples += silence_samples

            clipped = np.clip(audio, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
            out.writeframes(pcm.tobytes())
            total_samples += len(pcm)

            prev_tail = audio[-edge_n:] if len(audio) >= edge_n else audio
            prev_index = entry["index"]
            prev_speaker = entry.get("speaker")
            del audio, clipped, pcm  # release this segment's memory before the next one

        if sample_rate is None:
            raise RuntimeError(
                "no 'done' segments were available to assemble -- "
                f"{len(pending_gap_segments)} segment(s) were skipped as gaps"
            )

    return {
        "output": str(output_path),
        "sample_rate": sample_rate,
        "num_segments": len(segments),
        "num_segments_assembled": len(segments) - len(gap_reports),
        "gaps": gap_reports,
        "silence_ms_between_segments": silence_ms,
        "speaker_change_silence_ms": speaker_change_silence_ms,
        "total_duration_seconds": total_samples / sample_rate if sample_rate else None,
        "join_reports": join_reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--screenplay-file",
        type=Path,
        default=None,
        help=(
            "JSON screenplay file: a list of {'speaker': ..., 'text': ...} lines "
            "(docs/guides/audiobook_guide.md sec. 3). Mutually exclusive with "
            "--text/--text-file. Per-SPEAKER (multi-character) voice selection is still NOT "
            "wired in -- every line is synthesized with the same single voice reference "
            "(--voice-name/--ref-audio, or the model default if neither is given); see "
            "chunk_screenplay."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--tag-scope", choices=("chunk", "sentence"), default="sentence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--clear-cache-every", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Segments generated per model.batch_generate() call (Refs #114). Default 4: "
            "measured speedup was 3.40x-3.69x at batch=8 with no memory ceiling found up to "
            "8, but that measurement covered only dozens of segments, not the hundreds in a "
            "real chapter, so 4 is a deliberately smaller starting point with headroom to "
            "raise; --batch-size 8 reproduces the exact measured configuration. "
            "--batch-size 1 is the original, unbatched, one-model.generate()-call-per-segment "
            "path -- byte-for-byte the pre-#114 behavior -- for comparison or rollback."
        ),
    )
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--silence-ms", type=int, default=200)
    parser.add_argument("--gap-silence-ms", type=int, default=1000)
    parser.add_argument(
        "--speaker-change-silence-ms",
        type=int,
        default=None,
        help=(
            "Pause length to use between segments whose 'speaker' differs (screenplay "
            "format only); defaults to --silence-ms for every join when unset."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--voice-name",
        type=str,
        default=None,
        help=(
            "Use a voice already registered under voices/<name>.npy + voices/<name>.txt "
            "(docs/guides/audiobook_guide.md sec. 2, register_voice()) as the reference for "
            "EVERY segment generated this run (Refs #57 -- pins the voice across the whole "
            "chapter instead of letting it drift segment to segment). Mutually exclusive "
            "with --ref-audio. Cheapest option: the reference is loaded pre-encoded from "
            "disk, not re-encoded."
        ),
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help=(
            "Reference WAV to encode and use as the voice for every segment this run "
            "(Refs #57). Requires --ref-text or --ref-text-file. Encoded exactly once for "
            "the whole run, not per segment. Mutually exclusive with --voice-name. Combine "
            "with --save-voice-as to also register it under voices/<name>.npy for reuse via "
            "--voice-name next time -- there is one registered-voice format, not two."
        ),
    )
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--ref-text-file", type=Path, default=None)
    parser.add_argument(
        "--save-voice-as",
        type=str,
        default=None,
        help="With --ref-audio, also save the encoded reference under voices/<name>.npy+.txt.",
    )
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR)
    args = parser.parse_args()

    given = [
        name
        for name, val in (
            ("--text", args.text),
            ("--text-file", args.text_file),
            ("--screenplay-file", args.screenplay_file),
        )
        if val is not None
    ]
    if len(given) != 1:
        parser.error(
            "exactly one of --text, --text-file, --screenplay-file is required "
            f"(got: {given or 'none'})"
        )
        return

    if args.screenplay_file is not None:
        data = json.loads(args.screenplay_file.read_text(encoding="utf-8"))
        lines = parse_screenplay(data)
        chunks = chunk_screenplay(lines, max_chars=args.max_chars, tag_scope=args.tag_scope)
    else:
        text = args.text if args.text is not None else args.text_file.read_text(
            encoding="utf-8"
        ).strip()
        validate_control_tags(text)
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, max_chars=args.max_chars, tag_scope=args.tag_scope)

    if args.dry_run:
        for c in chunks:
            print(
                json.dumps(
                    {
                        "index": c.index,
                        "speaker": c.speaker,
                        "reopened_tags": c.reopened_tags,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
            )
        return

    if args.voice_name is not None and args.ref_audio is not None:
        parser.error("--voice-name and --ref-audio are mutually exclusive")
    ref_text = args.ref_text
    if args.ref_text_file is not None:
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    # A voice reference is resolved before the manifest is (re)loaded, whether or not this
    # run will generate anything, so the manifest header comparison below (Refs #57) sees
    # the real value on assemble-only runs too, not a placeholder that would falsely
    # "mismatch" against a manifest built with a real reference. --voice-name never needs
    # the model (its codes are already encoded on disk); --ref-audio does, so the model is
    # loaded early in that one case even for an otherwise assemble-only invocation.
    model = None
    if not args.assemble_only:
        from mlx_audio.tts.utils import load

        model = load(MODEL_ID, model_type="higgs_audio_v3")

    voice_reference = None
    if args.voice_name is not None or args.ref_audio is not None:
        if model is None:
            from mlx_audio.tts.utils import load

            model = load(MODEL_ID, model_type="higgs_audio_v3")
        voice_reference = resolve_voice_reference(
            model, args.voice_name, args.ref_audio, ref_text, args.save_voice_as, args.voices_dir
        )

    manifest = load_or_create_manifest(
        manifest_path,
        chunks,
        max_chars=args.max_chars,
        tag_scope=args.tag_scope,
        voice_reference=voice_reference,
    )

    if not args.assemble_only:
        gen_result = generate_segments(
            model,
            manifest,
            manifest_path,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            continue_on_error=args.continue_on_error,
            clear_cache_every=args.clear_cache_every,
            batch_size=args.batch_size,
            ref_audio_codes=voice_reference.codes if voice_reference else None,
            ref_text=voice_reference.ref_text if voice_reference else None,
        )
        if gen_result["failures"]:
            print(
                f"WARNING: {len(gen_result['failures'])} segment(s) failed after retries "
                f"and were skipped (--continue-on-error): {gen_result['failures']}"
            )

    if args.assemble or args.assemble_only:
        result = assemble_chapter(
            manifest,
            args.output_dir / "chapter.wav",
            base_dir=args.output_dir,
            silence_ms=args.silence_ms,
            allow_gaps=args.allow_gaps,
            gap_silence_ms=args.gap_silence_ms,
            speaker_change_silence_ms=args.speaker_change_silence_ms,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
