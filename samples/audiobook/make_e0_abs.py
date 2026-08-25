#!/usr/bin/env python3
"""Regenerate `prepared/chapter-e0-narration.abs` from `prepared/chapter-e0-narration.txt`
(issue #114, stage E1) -- one `#prose` block per source paragraph (paragraphs are
separated by a blank line in the `.txt`, which is exactly what closes a DSL block), no
speaker/attribute markup, matching how the Э0 chapter run was actually marked up
(minimally, to measure the pipeline -- see `docs/research/audiobook/m4-full-chapter-
results.md`).

This is the fixture `check_canonical.py`/`check_coverage.py`/the hash-reproduction test
in `tests/test_dsl.py` are run against. Mechanical and re-runnable, like `prepare.py`.

Usage:
    cd samples/audiobook
    python3 make_e0_abs.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).parent / "prepared" / "chapter-e0-narration.txt"
DST = Path(__file__).parent / "prepared" / "chapter-e0-narration.abs"


def main() -> None:
    text = SRC.read_text(encoding="utf-8").strip()
    paragraphs = text.split("\n\n")
    lines = ["#chapter Э0 fixture reproduction (sb-1-19 + sb-1-1-1, issue #114)", ""]
    for p in paragraphs:
        lines.append("#prose")
        lines.append(p)
        lines.append("")
    DST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DST} ({len(paragraphs)} #prose blocks)")


if __name__ == "__main__":
    main()
