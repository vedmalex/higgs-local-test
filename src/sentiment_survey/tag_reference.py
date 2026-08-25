"""Parse docs/guides/tag_reference.md into structured per-tag facts.

tag_reference.md (PR #113) is the source of truth for what each control tag is
supposed to do (from PROMPTING.md) and what has already been measured/heard.
This module extracts, per "category:name" tag key:

  - id            : tokenizer id (string, as printed in the doc)
  - sample        : repo-relative path to the doc's own catalog sample
  - status_text   : the full "status on listen" bullet text for this tag
  - group         : "A" | "B" | "C" | None — the M4-T5 objective-triage bucket,
                    parsed out of status_text (e.g. "Группа B по объективной триаге")
  - confirmed     : True if status_text says the owner already gave an
                    individual blind verdict for this exact tag (e.g.
                    "Подтверждено владельцем")

This is intentionally a light regex scrape of a hand-written Markdown doc, not
a general Markdown parser — it only needs to survive the specific table/bullet
shapes PR #113 established. If the doc's shape changes, parse_tag_reference()
degrades to an empty dict rather than crashing the server (checked by the
caller); a task set losing its doc-sourced status text is a soft failure, not
a broken app.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROW_RE = re.compile(r"^\|\s*\*\*([a-z_]+:[a-z_]+)\*\*\s*\|\s*(\d+)\s*\|(.*)\|\s*`([^`]+)`\s*\|\s*$", re.MULTILINE)
_STATUS_RE = re.compile(r"^- `([a-z_]+:[a-z_]+)`:\s*(.+)$", re.MULTILINE)
_GROUP_RE = re.compile(r"[Гг]руппа\s+([ABC])\b")
_CONFIRMED_RE = re.compile(r"[Пп]одтверждено(?:\s+владельцем)?", re.IGNORECASE)


def parse_tag_reference(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    tags: dict[str, dict] = {}

    for m in _ROW_RE.finditer(text):
        tag_key, tag_id, _metrics, sample = m.groups()
        tags[tag_key] = {
            "id": tag_id,
            "sample": sample,
            "status_text": None,
            "group": None,
            "confirmed": False,
        }

    for m in _STATUS_RE.finditer(text):
        tag_key, status_text = m.groups()
        status_text = status_text.strip()
        if tag_key not in tags:
            continue
        tags[tag_key]["status_text"] = status_text
        group_m = _GROUP_RE.search(status_text)
        tags[tag_key]["group"] = group_m.group(1) if group_m else None
        tags[tag_key]["confirmed"] = bool(_CONFIRMED_RE.search(status_text))

    return tags


CATEGORY_RU = {
    "emotion": "эмоция",
    "prosody": "просодия",
    "style": "манера речи",
    "sfx": "звуковой эффект",
    "env": "фон/окружение",
}
