#!/usr/bin/env python3
"""Estimate generation time for a `.abs` chapter (issue #114 DSL v0.1) from measured RTF
constants, and flag any `#recite` segment that risks the short-verse-segment failure mode
even after stanza-gluing.

See `.claude/skills/audiobook-markup/references/dsl-spec.md` sec. 5.4.

Usage:
    python3 scripts/check_budget.py chapter.abs [--min-recite-chars N] [--allow-short-recite]

Exit code 0 unless a `#recite` segment is under the minimum character threshold and
--allow-short-recite was not passed (1 in that case), or the file fails to compile (1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiobook_dsl as dsl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("abs_file", type=Path)
    ap.add_argument("--min-recite-chars", type=int, default=dsl.DEFAULT_MIN_RECITE_CHARS)
    ap.add_argument(
        "--allow-short-recite",
        action="store_true",
        help="report short #recite segments but exit 0 anyway",
    )
    args = ap.parse_args()

    text = args.abs_file.read_text(encoding="utf-8")
    try:
        _doc, segments = dsl.compile_source(text)
    except dsl.DslError as exc:
        print(f"COMPILE ERROR in {args.abs_file}: {exc}", file=sys.stderr)
        return 1

    report = dsl.estimate_budget(segments, min_recite_chars=args.min_recite_chars)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["short_recite_segments"] and not args.allow_short_recite:
        print(
            f"\n{len(report['short_recite_segments'])} #recite segment(s) under "
            f"{args.min_recite_chars} chars -- risks the short-verse-segment failure mode "
            "even after stanza-gluing (see dsl-spec.md sec. 2.7). Pass --allow-short-recite "
            "to accept this.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
