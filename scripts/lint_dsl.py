#!/usr/bin/env python3
"""Lint a `.abs` audiobook chapter (issue #114 DSL v0.1): structure, the 43-tag catalog,
attribute syntax, and `#recite` pause rules -- every finding reported with its source
line, not fail-fast.

See `.claude/skills/audiobook-markup/references/dsl-spec.md` sec. 5.3.

Usage:
    python3 scripts/lint_dsl.py chapter.abs [chapter2.abs ...]

Exit code 0 if every file has no issues, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiobook_dsl as dsl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("abs_files", type=Path, nargs="+")
    args = ap.parse_args()

    had_issues = False
    for path in args.abs_files:
        text = path.read_text(encoding="utf-8")
        issues = dsl.collect_lint_issues(text)
        if not issues:
            print(f"{path}: OK")
            continue
        had_issues = True
        print(f"{path}: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  {issue}")

    return 1 if had_issues else 0


if __name__ == "__main__":
    sys.exit(main())
