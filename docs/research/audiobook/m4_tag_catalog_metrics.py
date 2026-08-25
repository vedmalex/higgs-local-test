#!/usr/bin/env python3
"""Prosody + terminal-intonation metrics for the 46-clip tag-catalog run
(`m4_tag_catalog_bench.py`, issue #57, `docs/guides/tag_reference.md`).

Thin wrapper: reuses `m4_tag_inventory_metrics.analyze_full` (which itself reuses
`m4_prosody_metrics.analyze`) unchanged -- no new metric logic, just pointed at the
new manifest (`logs/m4_tag_catalog.json`) produced by `m4_tag_catalog_bench.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_tag_inventory_metrics import analyze_full  # noqa: E402


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if manifest_path and manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        clips = data["per_clip"]
        results = []
        for c in clips:
            r = analyze_full(Path(c["wav_path"]), text=c["text"])
            r["clip_id"] = c["clip_id"]
            results.append(r)
    else:
        paths = sys.argv[1:]
        results = [analyze_full(Path(p)) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
