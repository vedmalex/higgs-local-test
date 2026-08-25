#!/usr/bin/env python3
"""Shared library for the audiobook markup DSL compiler/checkers (issue #114, stage E1).

Turns a `.abs` chapter file into the *existing* screenplay JSON that `src/audiobook.py
--screenplay-file` already reads (`parse_screenplay`/`chunk_screenplay`) -- the engine
itself is never modified. See `.claude/skills/audiobook-markup/references/dsl-spec.md`
for the normative syntax description; this module is the implementation of that spec.

Design invariant carried over unchanged from `src/audiobook.py` and deliberately reused,
not re-implemented, here:
  - `VALID_TAGS` (the 43-tag catalog) is imported from `src/audiobook.py`, never
    duplicated -- a new tag added there is automatically recognized here too.
  - `is_stress_apostrophe` is imported from the same module for the same reason: the
    stress-mark heuristic (and its documented blind spot -- a name apostrophe with
    lowercase letters on both sides) must behave identically in the DSL's canonical
    round-trip check as it does at generation time.

Four independently useful entry points other than `compile_dsl.py`'s main path:
  - `parse_dsl` / `compile_document` -- turn `.abs` source into the screenplay JSON list.
  - `strip_markup` -- reconstruct the canonical (pre-markup) chapter text from a `.abs`
    file; used by `check_canonical.py`'s byte-for-byte invariant.
  - `collect_lint_issues` -- non-fail-fast structural/tag/attribute checks with line
    addresses, used by `lint_dsl.py`.
  - `estimate_budget` -- per-segment character counts plus the measured RTF constants
    from `docs/research/audiobook/m4-full-chapter-results.md`, used by `check_budget.py`.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiobook as ab  # noqa: E402 -- VALID_TAGS / is_stress_apostrophe reused, not duplicated

DEFAULT_SPEAKER = "narrator"
UNIT_PROSE = "prose"
UNIT_SAY = "say"
UNIT_RECITE = "recite"
VALID_UNITS = {UNIT_PROSE, UNIT_SAY, UNIT_RECITE}

ATTR_KEYS = {"emotion", "prosody", "style"}
# Fixed, deterministic emission order for the leading tag prefix, regardless of the
# order the author typed the attributes in -- matches the order VALID_TAGS documents
# the categories in (emotion, then prosody, then style).
ATTR_ORDER = ("emotion", "prosody", "style")

# The only two accepted pause-sugar spellings (dsl-spec.md). Longest-first order matters
# for the substring check in _check_no_stray_brackets below.
PAUSE_SUGAR = {
    "[долгая пауза]": "<|prosody:long_pause|>",
    "[пауза]": "<|prosody:pause|>",
}

TAG_SPAN_RE = re.compile(r"<\|[^|<>]*\|>")
SPEED_TAG_RE = re.compile(r"<\|prosody:speed_[a-z_]+\|>")


class DslError(ValueError):
    """A .abs source error, always carrying the 1-indexed source line it came from."""

    def __init__(self, message: str, src_line: Optional[int] = None):
        self.src_line = src_line
        self.message = message
        loc = f"line {src_line}: " if src_line is not None else ""
        super().__init__(f"{loc}{message}")


@dataclass
class Block:
    unit: str
    speaker: str
    attrs: dict
    body_lines: list
    scene: Optional[str]
    src_line: int  # line number of the directive line (1-indexed)


@dataclass
class Document:
    chapter: Optional[str]
    blocks: list = field(default_factory=list)
    notes: list = field(default_factory=list)  # [(src_line, text)] -- dropped, never in JSON


# ---------------------------------------------------------------------------
# Tag validation (shared by compile and lint -- catches env:*/chatml + typos, F11-style)
# ---------------------------------------------------------------------------


def _tag_error_message(candidate: str) -> str:
    """Return the error message for an invalid control-tag-shaped span, or None if valid."""
    if candidate in ab.VALID_TAGS:
        return None
    if candidate.startswith("<|env:") or candidate == "<|chatml|>":
        return (
            f"tag {candidate!r} is undocumented tokenizer scaffolding (env:*/chatml), not "
            "one of PROMPTING.md's usable prompt-time control tags -- compilation refused "
            "(see src/audiobook.py's VALID_TAGS comment)"
        )
    return (
        f"unknown control tag {candidate!r} -- not one of the {len(ab.VALID_TAGS)} valid "
        "Higgs TTS 3 tags in src/audiobook.py's VALID_TAGS"
    )


def _validate_text_tags(text: str, src_line: int) -> None:
    for m in TAG_SPAN_RE.finditer(text):
        err = _tag_error_message(m.group(0))
        if err:
            raise DslError(err, src_line)


def _check_no_stray_brackets(line: str, src_line: int) -> None:
    if "[" in line or "]" in line:
        raise DslError(
            f"unrecognized bracket marker in {line!r} -- only the pause sugars "
            f"{sorted(PAUSE_SUGAR)} are supported; fix the typo or write the raw "
            "<|prosody:...|> tag directly",
            src_line,
        )


def _apply_inline_sugar(line: str) -> str:
    for sugar, tag in PAUSE_SUGAR.items():
        line = line.replace(sugar, tag)
    return line


def _attrs_to_prefix(attrs: dict, src_line: int) -> str:
    parts = []
    for key in ATTR_ORDER:
        if key not in attrs:
            continue
        tag = f"<|{key}:{attrs[key]}|>"
        err = _tag_error_message(tag)
        if err:
            raise DslError(err, src_line)
        parts.append(tag)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_header_tokens(tokens: list, src_line: int, allow_speaker: bool):
    speaker = None
    attrs: dict = {}
    seen_attr = False
    for tok in tokens:
        if "=" in tok:
            key, _, val = tok.partition("=")
            key = key.strip()
            val = val.strip()
            if key not in ATTR_KEYS:
                raise DslError(
                    f"unknown attribute {key!r} (expected one of {sorted(ATTR_KEYS)})",
                    src_line,
                )
            if not val:
                raise DslError(f"attribute {key!r} has an empty value", src_line)
            if key in attrs:
                raise DslError(f"attribute {key!r} repeated in one block header", src_line)
            attrs[key] = val
            seen_attr = True
        else:
            if not allow_speaker:
                raise DslError(
                    f"unexpected token {tok!r} -- #prose takes only key=value attributes, "
                    "no speaker",
                    src_line,
                )
            if seen_attr:
                raise DslError(
                    f"speaker {tok!r} must come before any attribute, not after", src_line
                )
            if speaker is not None:
                raise DslError(
                    f"speaker must be a single token, got an extra {tok!r} -- multi-word "
                    "speaker names are not supported",
                    src_line,
                )
            speaker = tok
    return speaker, attrs


_DIRECTIVE_KEYWORDS = ("#chapter", "#scene", "#note", "#prose", "#say", "#recite")


def _directive_rest(line: str, keyword: str) -> Optional[str]:
    """Return the text after `keyword` if `line` is exactly that directive, else None."""
    if line == keyword:
        return ""
    if line.startswith(keyword + " "):
        return line[len(keyword) :].strip()
    return None


def parse_dsl(text: str) -> Document:
    """Parse `.abs` source into a `Document`. Raises `DslError` with a 1-indexed line
    number on any structural problem -- never silently drops or guesses."""
    lines = text.split("\n")
    n = len(lines)
    chapter: Optional[str] = None
    current_scene: Optional[str] = None
    blocks: list = []
    notes: list = []

    i = 0
    while i < n:
        raw = lines[i]
        lineno = i + 1
        stripped = raw.strip()

        if stripped == "":
            i += 1
            continue

        if not raw.startswith("#"):
            raise DslError(
                f"text outside of a block header (no preceding #prose/#say/#recite): "
                f"{raw[:60]!r}",
                lineno,
            )

        first_word = raw.split(None, 1)[0]
        if first_word not in _DIRECTIVE_KEYWORDS:
            raise DslError(f"unknown directive {first_word!r}", lineno)

        rest = _directive_rest(raw, first_word)
        if rest is None:
            # e.g. "#prosewrong" glued to the keyword with no space
            raise DslError(f"malformed directive line {raw!r}", lineno)

        if first_word == "#chapter":
            chapter = rest
            i += 1
            continue
        if first_word == "#scene":
            current_scene = rest
            i += 1
            continue
        if first_word == "#note":
            notes.append((lineno, rest))
            i += 1
            continue

        unit = first_word[1:]  # "prose" | "say" | "recite"
        tokens = rest.split() if rest else []
        allow_speaker = unit in (UNIT_SAY, UNIT_RECITE)
        speaker, attrs = _parse_header_tokens(tokens, lineno, allow_speaker)
        if unit == UNIT_SAY and speaker is None:
            raise DslError("#say requires a speaker as its first token", lineno)
        if speaker is None:
            speaker = DEFAULT_SPEAKER

        body: list = []
        j = i + 1
        while j < n and lines[j].strip() != "":
            if lines[j].startswith("#"):
                raise DslError(
                    f"block opened at line {lineno} is missing a blank line before the "
                    f"next directive at line {j + 1} -- a blank line must close every block",
                    j + 1,
                )
            body.append(lines[j])
            j += 1
        if not body:
            raise DslError(f"#{unit} block has no body text", lineno)

        blocks.append(
            Block(
                unit=unit,
                speaker=speaker,
                attrs=attrs,
                body_lines=body,
                scene=current_scene,
                src_line=lineno,
            )
        )
        i = j

    if not blocks:
        raise DslError("document has no content blocks (#prose/#say/#recite)", 1)
    return Document(chapter=chapter, blocks=blocks, notes=notes)


# ---------------------------------------------------------------------------
# Compilation: Document -> screenplay JSON segments
# ---------------------------------------------------------------------------


def _compile_prose_or_say(block: Block) -> str:
    prefix = _attrs_to_prefix(block.attrs, block.src_line)
    processed = []
    for raw_line in block.body_lines:
        line = _apply_inline_sugar(raw_line.strip())
        _check_no_stray_brackets(line, block.src_line)
        processed.append(line)
    body_text = "\n".join(processed)
    text = prefix + body_text
    _validate_text_tags(text, block.src_line)
    return text


def _compile_recite(block: Block) -> str:
    prefix = _attrs_to_prefix(block.attrs, block.src_line)
    if block.attrs.get("prosody", "").startswith("speed_"):
        raise DslError(
            "#recite must not use prosody=speed_* -- verse rhythm is carried by pauses, "
            "not tempo (owner-verified unreliable by ear; see dsl-spec.md)",
            block.src_line,
        )

    lines = block.body_lines
    last_i = len(lines) - 1
    compiled_lines = []
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.rstrip()
        suppress = False
        if stripped.endswith("\\"):
            suppress = True
            stripped = stripped[:-1].rstrip()
        line = _apply_inline_sugar(stripped.strip())
        _check_no_stray_brackets(line, block.src_line)
        if SPEED_TAG_RE.search(line):
            raise DslError(
                "#recite must not use an inline <|prosody:speed_*|> tag -- verse rhythm is "
                "carried by pauses, not tempo",
                block.src_line,
            )
        already_paused = line.endswith("<|prosody:pause|>") or line.endswith(
            "<|prosody:long_pause|>"
        )
        if already_paused and suppress:
            raise DslError(
                f"line {idx + 1} of this #recite block both ends with an explicit pause "
                "tag and suppresses the auto-pause with a trailing '\\' -- redundant, "
                "remove one",
                block.src_line,
            )
        if not already_paused and not suppress:
            line = line + ("<|prosody:long_pause|>" if idx == last_i else "<|prosody:pause|>")
        compiled_lines.append(line)

    text = prefix + " ".join(compiled_lines)
    _validate_text_tags(text, block.src_line)
    return text


def compile_document(doc: Document) -> list:
    """Compile a parsed `Document` into the screenplay JSON list
    (`[{"speaker": ..., "text": ..., "unit": ..., "scene": ..., "src_line": ...}, ...]`)
    that `src/audiobook.py --screenplay-file` reads unmodified. `unit`/`scene`/`src_line`
    are the ignored-by-the-engine structural keys; `speaker`/`text` are exactly what
    `parse_screenplay` keeps."""
    segments = []
    for b in doc.blocks:
        if b.unit == UNIT_RECITE:
            text = _compile_recite(b)
        else:
            text = _compile_prose_or_say(b)
        segments.append(
            {
                "speaker": b.speaker,
                "text": text,
                "unit": b.unit,
                "scene": b.scene,
                "src_line": b.src_line,
            }
        )
    return segments


def compile_source(text: str) -> tuple:
    """Parse + compile in one call. Returns (Document, segments)."""
    doc = parse_dsl(text)
    return doc, compile_document(doc)


# ---------------------------------------------------------------------------
# Canonical text reconstruction (strip_markup) -- the byte-for-byte invariant
# ---------------------------------------------------------------------------


def _strip_stress_apostrophes(s: str) -> str:
    """Drop every apostrophe `ab.is_stress_apostrophe` recognizes as a stress mark.

    Reuses `ab.is_stress_apostrophe` verbatim (not re-implemented) so this check has the
    exact same blind spot as generation time: a name apostrophe with lowercase letters on
    both sides is treated as a stress mark and stripped here too, which is precisely how
    this round-trip check is meant to surface that case loudly instead of silently.
    """
    return "".join(
        ch for i, ch in enumerate(s) if not (ch == "'" and ab.is_stress_apostrophe(s, i))
    )


def _canon_line(raw_line: str, is_recite: bool) -> str:
    stripped = raw_line.rstrip()
    if is_recite and stripped.endswith("\\"):
        stripped = stripped[:-1].rstrip()
    line = stripped.strip()
    for sugar in PAUSE_SUGAR:
        line = re.sub(r"\s*" + re.escape(sugar) + r"\s*", " ", line).strip()
    line = TAG_SPAN_RE.sub("", line)
    line = _strip_stress_apostrophes(line)
    return line


def strip_markup(source_text: str) -> str:
    """Reconstruct the canonical (pre-markup) chapter text from `.abs` source.

    Directives (`#chapter`/`#scene`/`#note` and every block header line) are removed
    entirely. Each block's body lines are rejoined with the same "\\n" they had in the
    source (preserving a genuine embedded line break inside one paragraph, e.g. two
    sentences originally joined by a single newline rather than a blank line); blocks
    themselves are rejoined with a blank line, mirroring the blank line that closed each
    block in the source. Pause sugar, raw control tags, `#recite`'s `\\`-suppression
    marker, and stress apostrophes (via `ab.is_stress_apostrophe`) are all removed.
    """
    doc = parse_dsl(source_text)
    paragraphs = []
    for b in doc.blocks:
        is_recite = b.unit == UNIT_RECITE
        lines_out = [_canon_line(rl, is_recite) for rl in b.body_lines]
        paragraphs.append("\n".join(lines_out))
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Lint (contour 3): structure/tags/attributes, collecting every issue, not fail-fast
# ---------------------------------------------------------------------------


@dataclass
class LintIssue:
    src_line: Optional[int]
    message: str

    def __str__(self) -> str:
        loc = f"line {self.src_line}" if self.src_line is not None else "(document)"
        return f"{loc}: {self.message}"


def collect_lint_issues(text: str) -> list:
    """Return a list of `LintIssue`s. Unlike `compile_source`, this never raises on the
    first problem -- it parses defensively and keeps collecting so one run reports
    everything wrong with a chapter instead of one error at a time."""
    issues: list = []
    try:
        doc = parse_dsl(text)
    except DslError as exc:
        issues.append(LintIssue(exc.src_line, exc.message))
        return issues

    for b in doc.blocks:
        try:
            if b.unit == UNIT_RECITE:
                _compile_recite(b)
            else:
                _compile_prose_or_say(b)
        except DslError as exc:
            issues.append(LintIssue(exc.src_line, exc.message))

        # Redundant-but-harmless authoring smells worth flagging even though the compile
        # step above already raises on the genuinely broken cases:
        if b.unit != UNIT_RECITE and any(
            raw_line.rstrip().endswith("\\") for raw_line in b.body_lines
        ):
            issues.append(
                LintIssue(
                    b.src_line,
                    f"trailing '\\' has no effect outside #recite (only #recite lines "
                    "suppress an auto-pause with it)",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Budget (contour 4): character-based time estimate from measured RTF constants
# ---------------------------------------------------------------------------

# docs/research/audiobook/m4-full-chapter-results.md sec. 1a/1b: measured aggregate RTF
# (generation_seconds / audio_duration_seconds) on the Э0 chapter run.
RTF_NO_BATCH = 3.83
RTF_BATCH8 = 1.08  # upper end of the measured 1.06-1.08 range at --batch-size 8

# Empirical seconds-of-audio-per-character-of-input-text, computed from the same Э0 run's
# manifest (output/chapter-114-e0/manifest.json: total audio_duration_seconds / total
# len(text) across all 70 segments = 1920.64s / 30483 chars = 0.0630). Not a universal
# constant -- Russian prose at this model's default settings, this text's average sentence
# length. Recompute if the model, language mix, or temperature changes materially.
SECONDS_PER_CHAR = 0.0630

# Heuristic (not measured) floor for a single #recite segment's character count, below
# which it risks both `ab.MIN_SECONDS_PER_CHAR`'s runtime plausibility floor (a very short
# clip can round down to implausibly-short audio) and an unnaturally abrupt cutoff -- the
# failure mode stanza-gluing (_compile_recite joining all lines into one sentence) is
# meant to avoid. A single-line #recite block can still land under this if that one line
# is itself short.
DEFAULT_MIN_RECITE_CHARS = 40


def estimate_budget(segments: list, min_recite_chars: int = DEFAULT_MIN_RECITE_CHARS) -> dict:
    total_chars = sum(len(s["text"]) for s in segments)
    audio_seconds_est = total_chars * SECONDS_PER_CHAR
    short_recite = [
        {"src_line": s["src_line"], "speaker": s["speaker"], "chars": len(s["text"])}
        for s in segments
        if s["unit"] == UNIT_RECITE and len(s["text"]) < min_recite_chars
    ]
    return {
        "num_segments": len(segments),
        "total_chars": total_chars,
        "estimated_audio_seconds": round(audio_seconds_est, 1),
        "estimated_generation_seconds_no_batch": round(audio_seconds_est * RTF_NO_BATCH, 1),
        "estimated_generation_seconds_batch8": round(audio_seconds_est * RTF_BATCH8, 1),
        "rtf_no_batch": RTF_NO_BATCH,
        "rtf_batch8": RTF_BATCH8,
        "seconds_per_char": SECONDS_PER_CHAR,
        "min_recite_chars": min_recite_chars,
        "short_recite_segments": short_recite,
    }
