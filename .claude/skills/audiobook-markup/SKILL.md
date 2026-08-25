---
name: audiobook-markup
description: Reference for the `.abs` audiobook chapter markup DSL (issue #114) — its syntax, compilation into the existing screenplay JSON read by `src/audiobook.py --screenplay-file`, and the canon/coverage/tag/budget validation contours. Use when authoring or reviewing a `.abs` chapter file, or when working on `scripts/compile_dsl.py`, `scripts/lint_dsl.py`, `scripts/check_canonical.py`, `scripts/check_coverage.py`, or `scripts/check_budget.py`.
---

# Audiobook markup DSL

The normative spec lives in `references/dsl-spec.md` — read it before authoring or
changing anything about the `.abs` format. It covers directive/block syntax, attribute
compilation, `#recite` pause rhythm, the canonical byte-for-byte invariant, the four
validation contours, and the open editorial questions the project owner still needs to
answer before a real chapter is marked up (§7).

The DSL compiles to the screenplay JSON `src/audiobook.py --screenplay-file` already
reads — the engine itself is never modified. Compile with:

```bash
python3 scripts/compile_dsl.py chapter.abs --output chapter.json
python3 scripts/lint_dsl.py chapter.abs
python3 scripts/check_canonical.py chapter.abs --canon original-narration.txt
python3 scripts/check_coverage.py chapter.abs
python3 scripts/check_budget.py chapter.abs
```
