#!/usr/bin/env python3
"""Prepare BhaktiVaibhava scripture markdown for TTS narration (issue #114, Э0).

Turns a "raw/" scholarly-edition markdown export into a "prepared/" plain-text file
that can be fed to `src/audiobook.py` unmodified. This is a draft of the future
audiobook-preparation skill's input contract (issue #114 stage A) -- a script, not a
narrative description, so it stays reproducible and testable.

Two source shapes are handled, distinguished by which `##` sections a file has:

1. **Dialogue-narrative chapters** (e.g. `sb-1-16.md`, `sb-1-19.md`): sections
   `## Вступление`, `## Повествование`, `## Заключение` are editorial prose meant to be
   read aloud, in that order. `## Навигация` (links only) is dropped.
2. **Verse-commentary files** (e.g. `sb-1-1-1.md`): four sections --
   `## Текст` (Sanskrit, Cyrillic transliteration with combining diacritics),
   `## Пословный перевод` (word-by-word gloss), `## Перевод` (short literary
   translation), `## Комментарий` (long prose commentary, dialogue-free). Only
   `Перевод` and `Комментарий` are narration text; `Текст` is extracted separately
   (kept, diacritics intact, for a dedicated pronunciation experiment, never merged
   into the narration file) and `Пословный перевод` is dropped entirely -- a
   word-by-word gloss is not read aloud in a real audiobook.

## Rules applied (this IS the input-contract draft for the future skill)

Dropped, always:
  - The YAML frontmatter block between the leading `---` markers
    (`telegraphUrl`, `accessToken`, `contentHash`, ...).
  - The `## Навигация` section (bare links to other chapters).
  - The `## Пословный перевод` section (word-by-word gloss; never narrated).
  - Every verse-anchor markdown link, e.g. `[1](19/01.19.01.md)` or
    `[9-10](16/01.16.09-10.md)` -- both the visible number AND the `.md` target,
    removed as one unit. Left in, the TTS model would read the file path aloud.
  - A lone footnote-style link to a same-page anchor, e.g. `[(Б.-г., 4.7),1](#)`.
  - Markdown emphasis markers `**bold**` / `*italic*` -- the marker only, the words
    inside are kept as plain narration text.
  - The `***` horizontal-rule marker.

Kept, always:
  - Speaker labels (`Сута Госвами сказал:`, `Мудрецы сказали:`, ...) as ordinary
    narration text -- this project stage does not build a screenplay/speaker
    structure, and a real audiobook narrator reads these aloud too.
  - Editorial bracketed asides (`[Царь Парикшит думал:]`, `[Ганга, у которой постился
    царь]`) -- the square brackets are stripped but the words inside are kept as plain
    text, so nothing sounds like a stage direction.
  - Sanskrit diacritics (combining macron U+0304, combining dot below U+0323,
    combining tilde U+0303, combining caron U+030C, combining dot above U+0307, ...)
    wherever they occur in narration text (`## Перевод`, `## Комментарий`) -- this is
    native transliteration signal, not an added stress mark, and stripping it is a
    separate, explicit experiment (`--diacritics-experiment`), never the default.
  - Paragraph order exactly as in the source.

## Verification (run automatically after every prepare)

The output MUST NOT contain: `](`, literal `.md`, `[`, `]`, `*`, or a line starting
with `#`. `--check` (or normal script exit) enforces this and fails loudly otherwise.

## Usage

    python3 prepare.py raw/sb-1-19.md prepared/sb-1-19.txt --kind dialogue
    python3 prepare.py raw/sb-1-1-1.md prepared/sb-1-1-1.txt --kind verse-commentary
    python3 prepare.py raw/sb-1-1-1.md prepared/sb-1-1-1.verse-diacritics.txt \
        --kind verse-commentary --emit-verse
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.S)
HEADING_RE = re.compile(r"^##\s+(.+)$", re.M)
VERSE_LINK_RE = re.compile(r"\[\d+(?:-\d+)?\]\([^)]*\.md\)")
ANCHOR_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]\(#\)")
HRULE_RE = re.compile(r"^\*\*\*\s*$", re.M)
BRACKETS_RE = re.compile(r"\[([^\[\]]*)\]")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"\*([^*]+)\*")
MULTISPACE_RE = re.compile(r"[ \t]+")
TRAILING_SPACE_RE = re.compile(r" \n")
BLANK_RUN_RE = re.compile(r"\n{3,}")


def split_sections(text: str) -> dict[str, str]:
    text = FRONTMATTER_RE.sub("", text, count=1)
    parts = HEADING_RE.split(text)
    sections: dict[str, str] = {}
    it = iter(parts[1:])
    for heading, body in zip(it, it):
        sections[heading.strip()] = body
    return sections


def strip_markup(body: str) -> str:
    body = VERSE_LINK_RE.sub("", body)
    body = ANCHOR_FOOTNOTE_RE.sub("", body)
    body = HRULE_RE.sub("", body)
    body = BRACKETS_RE.sub(r"\1", body)
    body = BOLD_RE.sub(r"\1", body)
    body = ITALIC_RE.sub(r"\1", body)
    body = MULTISPACE_RE.sub(" ", body)
    body = TRAILING_SPACE_RE.sub("\n", body)
    body = BLANK_RUN_RE.sub("\n\n", body)
    return body.strip()


def prepare_dialogue(text: str) -> str:
    sections = split_sections(text)
    keep = [
        sections[name]
        for name in ("Вступление", "Повествование", "Заключение")
        if name in sections
    ]
    return strip_markup("\n\n".join(keep))


def prepare_verse_commentary(text: str) -> tuple[str, str]:
    """Returns (narration_text, verse_text). `verse_text` keeps diacritics intact."""
    sections = split_sections(text)
    perevod = strip_markup(sections.get("Перевод", ""))
    kommentarii = strip_markup(sections.get("Комментарий", ""))
    narration = "\n\n".join(p for p in (perevod, kommentarii) if p)

    verse_raw = sections.get("Текст", "")
    verse = HRULE_RE.sub("", verse_raw)
    verse = ITALIC_RE.sub(r"\1", verse)  # only strip *italic* markers, keep diacritics
    verse = MULTISPACE_RE.sub(" ", verse).strip()
    return narration, verse


DIACRITIC_MARKS = {
    0x0304,  # combining macron (vowel length)
    0x0323,  # combining dot below (retroflex/vocalic r,l)
    0x0303,  # combining tilde (palatal nasal marker in this scheme)
    0x030C,  # combining caron (palatal sibilant marker)
    0x0307,  # combining dot above (anusvara marker)
    0x0331,  # combining macron below
    0x0301,  # combining acute accent -- here marks sha-with-acute (ш́ = palatal ś),
    # NOT a Russian stress mark; same code point, different job in this corpus. See
    # docs/research/audiobook/m4-full-chapter-results.md for the ambiguity this
    # creates against the project's existing U+0301 stress-mark pipeline.
}

# Precomposed Cyrillic letters this scheme also uses for long vowels, which combining()
# does not flag (they are single code points, not base+mark) -- map each to its plain
# counterpart for a genuinely diacritic-free comparison.
PRECOMPOSED_MAP = {
    "ӣ": "и",  # CYRILLIC SMALL LETTER I WITH MACRON (ӣ)
    "Ӣ": "И",  # CYRILLIC CAPITAL LETTER I WITH MACRON
    "ӯ": "у",  # CYRILLIC SMALL LETTER U WITH MACRON (ӯ)
    "Ӯ": "У",  # CYRILLIC CAPITAL LETTER U WITH MACRON
}


def strip_sanskrit_diacritics(s: str) -> str:
    """Remove only the specific combining marks this transliteration scheme adds.

    Deliberately does NOT run unicodedata.normalize('NFD', ...) over the whole
    string first: several ordinary Cyrillic letters (е.g. 'й' = U+0439) have a
    canonical decomposition into base+combining-breve, and NFD-then-filter would
    silently corrupt those letters too (turns "асйа" into "асиа"). Only the
    specific diacritic code points this project's transliteration uses are
    stripped, on the already-composed string; the scheme's few precomposed
    long-vowel letters (ӣ, ӯ) are remapped to their plain counterpart separately.
    """
    s = "".join(PRECOMPOSED_MAP.get(c, c) for c in s)
    return "".join(c for c in s if ord(c) not in DIACRITIC_MARKS)


FORBIDDEN_SUBSTRINGS = ("](", ".md", "[", "]", "*")


def verify(text: str) -> list[str]:
    problems = []
    for token in FORBIDDEN_SUBSTRINGS:
        if token in text:
            problems.append(f"forbidden substring present: {token!r}")
    for line in text.splitlines():
        if line.startswith("#"):
            problems.append(f"markdown heading survived: {line[:60]!r}")
            break
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--kind", choices=("dialogue", "verse-commentary"), required=True)
    ap.add_argument(
        "--emit-verse",
        action="store_true",
        help="for --kind verse-commentary: write the ## Текст block (diacritics kept) "
        "to <output>.verse.txt instead of the narration text",
    )
    ap.add_argument(
        "--emit-verse-plain",
        action="store_true",
        help="also write a diacritics-stripped copy of the verse to "
        "<output>.verse-plain.txt, for the A/B pronunciation experiment",
    )
    args = ap.parse_args()

    raw = args.input.read_text(encoding="utf-8")

    if args.kind == "dialogue":
        prepared = prepare_dialogue(raw)
    else:
        narration, verse = prepare_verse_commentary(raw)
        prepared = narration
        if args.emit_verse:
            verse_path = args.output.with_suffix(".verse.txt")
            verse_path.write_text(verse + "\n", encoding="utf-8")
            print(f"wrote {verse_path} ({len(verse.split())} words, diacritics kept)")
            if args.emit_verse_plain:
                plain_path = args.output.with_suffix(".verse-plain.txt")
                plain_path.write_text(strip_sanskrit_diacritics(verse) + "\n", encoding="utf-8")
                print(f"wrote {plain_path} (diacritics stripped, for A/B comparison)")

    problems = verify(prepared)
    if problems:
        print("VERIFICATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prepared + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(prepared.split())} words) -- verification OK")


if __name__ == "__main__":
    main()
