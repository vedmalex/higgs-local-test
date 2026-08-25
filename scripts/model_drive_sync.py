#!/usr/bin/env python3
"""
Reconcile and upload the local Hugging Face model cache against Google Drive
(MyDrive/higgs-benchmark/model-cache/) and notebooks/model_catalog.json.

notebooks/model_prefetch_to_drive.ipynb only ever writes TO Drive (internet -> Drive,
via Colab). This script is the other direction: local cache -> Drive, for models that
were already downloaded locally (over a working connection, or restored from an earlier
Drive archive) but never got packed and uploaded, so a future cache clear or a different
machine would have to re-fetch them from the internet instead of just re-extracting.

It packs models with scripts/model_cache_lib.py -- the SAME recipe the notebook's cell 6
uses (see that module's docstring for why the notebook can't literally import it) -- so an
archive built here restores exactly like one built there:
  tar -xf <name>.tar -C ~/.cache/huggingface/hub/

Never deletes anything, locally or on Drive. Never re-uploads an archive that already
matches what's on Drive. Never overwrites a differing archive without --force.

Usage:
  # Read-only: what's local, what's on Drive, what's in the catalog, where they disagree.
  python3 scripts/model_drive_sync.py reconcile

  # Read-only: what a full upload run would do and how many bytes it would move.
  python3 scripts/model_drive_sync.py plan

  # Pack + upload one model (skips if an identically-sized archive is already on Drive).
  python3 scripts/model_drive_sync.py upload --model whisper-large-v3-processor

  # Overwrite a differing archive already on Drive (asked for explicitly, never implied).
  python3 scripts/model_drive_sync.py upload --model whisper-large-v3-processor --force

  # Re-stamp model_catalog.json's status/notes/verified fields from what reconcile found.
  python3 scripts/model_drive_sync.py refresh-catalog
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gdrive_sync import (  # noqa: E402
    get_token,
    list_files,
    resolve_path,
    upload_file,
    upload_file_resumable,
    delete_file,
)
from model_cache_lib import (  # noqa: E402
    archive_name,
    hub_model_dir,
    is_locally_complete,
    local_hf_hub_dir,
    pack_model_dir,
    repo_id_from_dirname,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT_DIR / "notebooks" / "model_catalog.json"
DRIVE_PATH = "higgs-benchmark/model-cache"


def load_catalog() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_catalog(catalog: dict) -> None:
    with open(CATALOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def drive_archives(token: str):
    """(archives, folder_id) where archives is {'name.tar': {'id':..., 'size': int_bytes}}
    for everything in the Drive model-cache folder. Empty dict + None folder_id (not an
    error) if the folder doesn't exist yet."""
    folder_id = resolve_path(DRIVE_PATH, token)
    if folder_id is None:
        return {}, None
    out = {}
    for f in list_files(folder_id, token):
        if f["name"].endswith(".tar"):
            out[f["name"]] = {"id": f["id"], "size": int(f.get("size", 0))}
    return out, folder_id


def local_hub_dirs() -> Dict[str, Path]:
    """{repo_id: path} for every complete models--... directory in the local HF hub cache."""
    hub = local_hf_hub_dir()
    out = {}
    if not hub.is_dir():
        return out
    for entry in hub.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        repo_id = repo_id_from_dirname(entry.name)
        if repo_id and is_locally_complete(entry):
            out[repo_id] = entry
    return out


def build_reconciliation(token: str) -> dict:
    catalog = load_catalog()
    models = catalog["models"]
    by_repo_id = {m["repo_id"]: m for m in models if m.get("repo_id")}

    archives, _ = drive_archives(token)
    local_dirs = local_hub_dirs()

    rows = []
    for m in models:
        repo_id = m.get("repo_id")
        name = m["name"]
        local_present = repo_id in local_dirs if repo_id else False
        drive_present = archive_name(name) in archives
        declared = m["status"]
        expected = {
            "local+drive": local_present and drive_present,
            "drive_only": drive_present and not local_present,
            "local_only": local_present and not drive_present,
        }.get(declared, not local_present and not drive_present)
        rows.append(
            {
                "name": name,
                "repo_id": repo_id,
                "declared_status": declared,
                "local_present": local_present,
                "drive_present": drive_present,
                "drive_bytes": archives.get(archive_name(name), {}).get("size"),
                "matches_declared": expected,
            }
        )

    # Local hub directories that map to no catalog entry at all -- the "gets forgotten"
    # case the owner specifically flagged.
    orphans = [
        repo_id
        for repo_id in local_dirs
        if repo_id not in by_repo_id
    ]

    return {"rows": rows, "orphans": orphans, "orphan_paths": {r: str(local_dirs[r]) for r in orphans}}


def cmd_reconcile(args: argparse.Namespace) -> None:
    token = get_token(args.token, args.account)
    result = build_reconciliation(token)
    rows = result["rows"]

    local_not_drive = [r for r in rows if r["local_present"] and not r["drive_present"]]
    drive_not_local = [r for r in rows if r["drive_present"] and not r["local_present"]]
    mismatched = [r for r in rows if not r["matches_declared"]]

    print("=== Model cache reconciliation ===")
    print(f"Catalog entries: {len(rows)}")
    print()

    print(f"Local but NOT on Drive ({len(local_not_drive)}):")
    for r in local_not_drive:
        print(f"  - {r['name']} ({r['repo_id']})")
    if not local_not_drive:
        print("  (none)")
    print()

    print(f"On Drive but NOT local ({len(drive_not_local)}):")
    for r in drive_not_local:
        size_mb = (r["drive_bytes"] or 0) / (1024 * 1024)
        print(f"  - {r['name']} ({size_mb:.2f} MB on Drive)")
    if not drive_not_local:
        print("  (none)")
    print()

    print(f"On disk but NOT in the catalog at all ({len(result['orphans'])}):")
    for repo_id in result["orphans"]:
        print(f"  - {repo_id} -> {result['orphan_paths'][repo_id]}")
    if not result["orphans"]:
        print("  (none)")
    print()

    print(f"Catalog status disagrees with reality ({len(mismatched)}):")
    for r in mismatched:
        print(
            f"  - {r['name']}: declared '{r['declared_status']}' but "
            f"local={r['local_present']} drive={r['drive_present']}"
        )
    if not mismatched:
        print("  (none -- catalog matches reality)")


def cmd_plan(args: argparse.Namespace) -> None:
    """What a full `upload --all` run would move, without moving anything."""
    token = get_token(args.token, args.account)
    result = build_reconciliation(token)
    candidates = [r for r in result["rows"] if r["local_present"] and not r["drive_present"]]

    if not candidates:
        print("Nothing to upload: every locally-present model already has a Drive archive.")
        return

    print(f"Would upload {len(candidates)} model(s):")
    total_bytes = 0
    hub = local_hf_hub_dir()
    for r in candidates:
        model_dir = hub / (
            "models--" + r["repo_id"].replace("/", "--")
        )
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        total_bytes += size_bytes
        print(f"  - {r['name']}: ~{size_bytes / (1024 * 1024):.2f} MB (uncompressed dir size)")
    print(f"\nTotal (uncompressed, tar overhead not included): ~{total_bytes / (1024 * 1024 * 1024):.2f} GB")
    print("Nothing was uploaded. Run `upload --model <name>` per entry, or add --all-confirmed.")


def _upload_one(name: str, token: str, force: bool, dry_run: bool) -> str:
    catalog = load_catalog()
    entry = next((m for m in catalog["models"] if m["name"] == name), None)
    if entry is None:
        raise SystemExit(f"'{name}' is not in {CATALOG_PATH}")
    repo_id = entry.get("repo_id")
    if not repo_id:
        raise SystemExit(f"'{name}' has no repo_id pinned in the catalog -- nothing to upload")

    model_dir = hub_model_dir(repo_id, local_hf_hub_dir().parent)
    if not is_locally_complete(model_dir):
        raise SystemExit(f"'{name}': no complete local cache at {model_dir}")

    archives, folder_id = drive_archives(token)
    tar_name = archive_name(name)
    existing = archives.get(tar_name)

    with tempfile.TemporaryDirectory(prefix="model-drive-sync-") as tmp:
        local_tar = pack_model_dir(model_dir, Path(tmp) / tar_name)
        local_size = local_tar.stat().st_size

        if existing is not None:
            if existing["size"] == local_size and not force:
                return (
                    f"SKIPPED: '{tar_name}' already on Drive with the same size "
                    f"({local_size / (1024*1024):.2f} MB) -- nothing to do"
                )
            if existing["size"] != local_size and not force:
                return (
                    f"BLOCKED: '{tar_name}' on Drive is {existing['size'] / (1024*1024):.2f} MB, "
                    f"local pack is {local_size / (1024*1024):.2f} MB -- differs. "
                    f"Re-run with --force to overwrite (this deletes the current Drive copy first)."
                )
            # existing and force: overwrite regardless of whether sizes matched.
            if dry_run:
                return f"WOULD OVERWRITE: '{tar_name}' ({local_size / (1024*1024):.2f} MB) -- --force given"
            delete_file(existing["id"], token)
        elif dry_run:
            return f"WOULD UPLOAD: '{tar_name}' ({local_size / (1024*1024):.2f} MB), new on Drive"

        if local_size > 5 * 1024 * 1024:
            upload_file_resumable(local_tar, folder_id, token)
        else:
            upload_file(local_tar, folder_id, token)
        return f"UPLOADED: '{tar_name}' ({local_size / (1024*1024):.2f} MB)"


def cmd_upload(args: argparse.Namespace) -> None:
    token = get_token(args.token, args.account)
    print(_upload_one(args.model, token, force=args.force, dry_run=args.dry_run))


def cmd_refresh_catalog(args: argparse.Namespace) -> None:
    token = get_token(args.token, args.account)
    result = build_reconciliation(token)
    catalog = load_catalog()
    by_name = {m["name"]: m for m in catalog["models"]}

    changed = []
    for r in result["rows"]:
        entry = by_name[r["name"]]
        if r["local_present"] and r["drive_present"]:
            new_status = "local+drive"
        elif r["drive_present"]:
            new_status = "drive_only"
        elif r["local_present"]:
            new_status = "local_only"
        else:
            # Neither present: leave needed/candidate/rejected alone -- refresh only
            # touches facts this script can actually observe (presence), never the
            # editorial candidate/rejected/needed judgment calls.
            continue
        if entry["status"] != new_status:
            changed.append((entry["name"], entry["status"], new_status))
            entry["status"] = new_status
        if r["drive_present"] and r["drive_bytes"]:
            size_mb = r["drive_bytes"] / (1024 * 1024)
            marker = f"On Drive as {archive_name(r['name'])} ("
            fresh_fact = f"{marker}{size_mb:.2f} MB)."
            existing_notes = entry.get("notes") or ""
            if marker in existing_notes:
                # Replace only the stale "On Drive as ... MB)." fragment (size may have
                # changed since a re-pack), never the rest of the note -- refresh-catalog
                # must not silently drop editorial context (rejection reasons, warnings,
                # "do not switch to X" guidance) that has nothing to do with Drive presence.
                start = existing_notes.index(marker)
                end = existing_notes.index(")", start) + 1
                if end < len(existing_notes) and existing_notes[end] == ".":
                    end += 1
                entry["notes"] = existing_notes[:start] + fresh_fact + existing_notes[end:]
            elif fresh_fact not in existing_notes:
                entry["notes"] = (existing_notes + " " + fresh_fact).strip() if existing_notes else fresh_fact

    catalog["verified"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%d")
        + " via scripts/model_drive_sync.py reconcile"
    )

    if args.dry_run:
        print("DRY RUN -- would change:")
        for name, old, new in changed:
            print(f"  - {name}: {old} -> {new}")
        if not changed:
            print("  (no status changes; 'verified' timestamp would still be refreshed)")
        return

    save_catalog(catalog)
    print(f"Wrote {CATALOG_PATH}")
    for name, old, new in changed:
        print(f"  - {name}: {old} -> {new}")
    if not changed:
        print("  (no status changes; refreshed 'verified' timestamp only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--token", help="Explicit OAuth2 token")
        p.add_argument("--account", help="Specific gcloud account email")

    p_rec = sub.add_parser("reconcile", help="Read-only diff: local cache vs Drive vs catalog")
    add_common(p_rec)
    p_rec.set_defaults(func=cmd_reconcile)

    p_plan = sub.add_parser("plan", help="Read-only: what a full upload run would move")
    add_common(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_up = sub.add_parser("upload", help="Pack and upload one catalog entry's local cache")
    add_common(p_up)
    p_up.add_argument("--model", required=True, help="Catalog 'name' to upload")
    p_up.add_argument("--force", action="store_true", help="Overwrite an existing differing Drive archive")
    p_up.add_argument("--dry-run", action="store_true", help="Pack nothing skipped, but do not touch Drive")
    p_up.set_defaults(func=cmd_upload)

    p_ref = sub.add_parser("refresh-catalog", help="Re-stamp status/notes/verified from reconcile facts")
    add_common(p_ref)
    p_ref.add_argument("--dry-run", action="store_true", help="Print what would change, don't write the file")
    p_ref.set_defaults(func=cmd_refresh_catalog)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
