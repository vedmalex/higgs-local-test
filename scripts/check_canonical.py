#!/usr/bin/env python3
"""Verify the byte-for-byte canonical-text invariant for a `.abs` chapter (issue #114
DSL v0.1): `strip_markup(chapter.abs)` must reproduce the original narration text
exactly.

See `.claude/skills/audiobook-markup/references/dsl-spec.md` sec. 4.

Usage:
    python3 scripts/check_canonical.py chapter.abs --canon original-narration.txt

Exit code 0 on an exact match, 1 on any mismatch (with a diff of the first differing
line/offset) or a `.abs` parse error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiobook_dsl as dsl  # noqa: E402


def _first_diff(a: str, b: str) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            line = a.count("\n", 0, i) + 1
            ctx_a = a[max(0, i - 30) : i + 30]
            ctx_b = b[max(0, i - 30) : i + 30]
            return (
                f"first difference at char offset {i} (line ~{line}):\n"
                f"  computed: ...{ctx_a!r}...\n"
                f"  expected: ...{ctx_b!r}..."
            )
    return f"one text is a prefix of the other -- computed len={len(a)}, expected len={len(b)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("abs_file", type=Path)
    ap.add_argument(
        "--canon", type=Path, required=True, help="the original, pre-markup narration text file"
    )
    args = ap.parse_args()

    source = args.abs_file.read_text(encoding="utf-8")
    try:
        computed = dsl.strip_markup(source)
    except dsl.DslError as exc:
        print(f"PARSE ERROR in {args.abs_file}: {exc}", file=sys.stderr)
        return 1

    expected = args.canon.read_text(encoding="utf-8")
    # A prepared narration file conventionally ends with exactly one trailing newline
    # (samples/audiobook/prepare.py writes `prepared + "\n"`); strip_markup's block-join
    # logic does not manufacture one. Allow that single, well-understood difference and
    # nothing else -- comparing raw otherwise so any real content drift still fails.
    computed_cmp = computed if computed.endswith("\n") else computed + "\n"
    expected_cmp = expected if expected.endswith("\n") else expected + "\n"

    if computed_cmp == expected_cmp:
        print(f"OK: {args.abs_file} canonical text matches {args.canon} byte-for-byte")
        return 0

    print(f"CANONICAL MISMATCH: {args.abs_file} vs {args.canon}", file=sys.stderr)
    print(_first_diff(computed_cmp, expected_cmp), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
