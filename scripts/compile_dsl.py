#!/usr/bin/env python3
"""Compile a `.abs` audiobook chapter (issue #114 DSL v0.1) into the screenplay JSON
that `src/audiobook.py --screenplay-file` reads unmodified.

See `.claude/skills/audiobook-markup/references/dsl-spec.md` for the format. This is a
thin CLI over `scripts/audiobook_dsl.py`; the parsing/compilation logic lives there so
`lint_dsl.py`/`check_canonical.py`/`check_coverage.py`/`check_budget.py` share it.

Usage:
    python3 scripts/compile_dsl.py chapter.abs --output chapter.json
    python3 scripts/compile_dsl.py chapter.abs            # prints JSON to stdout
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
    ap.add_argument("--output", type=Path, default=None, help="write JSON here instead of stdout")
    args = ap.parse_args()

    text = args.abs_file.read_text(encoding="utf-8")
    try:
        _doc, segments = dsl.compile_source(text)
    except dsl.DslError as exc:
        print(f"COMPILE ERROR in {args.abs_file}: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(segments, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({len(segments)} segment(s))")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
