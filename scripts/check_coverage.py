#!/usr/bin/env python3
"""Verify completeness of a `.abs` chapter's compilation (issue #114 DSL v0.1): every
word of the canonical narration text must be represented in the compiled screenplay
JSON, with nothing left over.

Unlike `check_canonical.py` (which checks the `.abs` *source* against an external
canonical text file, catching a mis-authored chapter), this checks the *compiler's own*
completeness: it derives canonical text from the same `.abs` file's parsed blocks and
compares it, word-normalized, against what `compile_document` actually produced. A bug
that silently drops, duplicates, or mangles a block while building JSON would show up
here even if the `.abs` file itself is perfectly authored. See
`.claude/skills/audiobook-markup/references/dsl-spec.md` sec. 5.2 -- this pattern is
carried over from the suno-music-producer skill's `check-stress.py`, and from the exact
failure mode it was built to catch: fragments silently dropped while an authenticity
check stayed green.

Usage:
    python3 scripts/check_coverage.py chapter.abs

Exit code 0 if the recovered text and the canonical text match word-for-word after
normalization, 1 otherwise (with the words present in one but not the other).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiobook_dsl as dsl  # noqa: E402

_WS_RE = re.compile(r"\s+")


def _normalize_words(text: str) -> list:
    text = dsl.TAG_SPAN_RE.sub(" ", text)
    text = dsl._strip_stress_apostrophes(text)
    text = _WS_RE.sub(" ", text).strip()
    return text.split(" ") if text else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("abs_file", type=Path)
    args = ap.parse_args()

    source = args.abs_file.read_text(encoding="utf-8")
    try:
        doc, segments = dsl.compile_source(source)
        canon = dsl.strip_markup(source)
    except dsl.DslError as exc:
        print(f"PARSE ERROR in {args.abs_file}: {exc}", file=sys.stderr)
        return 1

    canon_words = _normalize_words(canon)
    recovered_words = _normalize_words("\n\n".join(s["text"] for s in segments))

    if canon_words == recovered_words:
        print(f"OK: {args.abs_file} -- {len(canon_words)} word(s), fully covered, no remainder")
        return 0

    canon_set = _multiset(canon_words)
    recovered_set = _multiset(recovered_words)
    missing = _subtract(canon_set, recovered_set)
    extra = _subtract(recovered_set, canon_set)

    print(f"COVERAGE MISMATCH in {args.abs_file}", file=sys.stderr)
    if missing:
        print(
            f"  {sum(missing.values())} word(s) in canonical text but NOT in compiled JSON "
            f"(dropped): {_sample(missing)}",
            file=sys.stderr,
        )
    if extra:
        print(
            f"  {sum(extra.values())} word(s) in compiled JSON but NOT in canonical text "
            f"(fabricated/duplicated): {_sample(extra)}",
            file=sys.stderr,
        )
    if canon_words != recovered_words and not missing and not extra:
        print("  same words, different order/count somewhere -- structural mismatch", file=sys.stderr)
    return 1


def _multiset(words: list) -> dict:
    counts: dict = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


def _subtract(a: dict, b: dict) -> dict:
    out = {}
    for w, n in a.items():
        d = n - b.get(w, 0)
        if d > 0:
            out[w] = d
    return out


def _sample(counts: dict, limit: int = 15) -> str:
    items = list(counts.items())[:limit]
    more = "" if len(counts) <= limit else f", ... ({len(counts) - limit} more)"
    return ", ".join(f"{w!r}x{n}" for w, n in items) + more


if __name__ == "__main__":
    sys.exit(main())
