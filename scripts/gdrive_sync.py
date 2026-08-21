#!/usr/bin/env python3
"""
Google Drive synchronization utility for Higgs Audio benchmark files.
Pure Python standard library implementation using Google Drive REST API v3
and gcloud access tokens (or GDRIVE_ACCESS_TOKEN environment variable).

Adapted from bs-search upload-to-drive / download-from-drive architecture.

Usage:
  # 1. Upload samples/ to Google Drive (creates "higgs-benchmark/samples" folder)
  python scripts/gdrive_sync.py upload --source samples --folder-name higgs-benchmark

  # 2. Upload outputs/ to Google Drive
  python scripts/gdrive_sync.py upload --source output --folder-name higgs-benchmark

  # 3. Upload everything (samples, output, notebooks)
  python scripts/gdrive_sync.py upload --all --folder-name higgs-benchmark

  # 4. Download from a specific Drive folder
  python scripts/gdrive_sync.py download --folder-id <DRIVE_FOLDER_ID> --dest output

  # 5. List contents of a Drive folder
  python scripts/gdrive_sync.py list --folder-id <DRIVE_FOLDER_ID>
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DRIVE_API = "https://www.googleapis.com/upload/drive/v3/files"
DRIVE_META = "https://www.googleapis.com/drive/v3/files"
ROOT_DIR = Path(__file__).resolve().parents[1]


def get_token(custom_token: Optional[str] = None, account: Optional[str] = None) -> str:
    """Obtain OAuth2 token from custom arg, env var, or gcloud CLI."""
    if custom_token:
        return custom_token.strip()

    env_token = os.environ.get("GDRIVE_ACCESS_TOKEN")
    if env_token:
        return env_token.strip()

    cmd = ["gcloud", "auth", "print-access-token"]
    if account:
        cmd.extend([f"--account={account}"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    print("ERROR: Could not obtain Google Drive access token.", file=sys.stderr)
    print("Please log in with gcloud:", file=sys.stderr)
    print("  gcloud auth login --enable-gdrive-access", file=sys.stderr)
    print("Or set the GDRIVE_ACCESS_TOKEN environment variable.", file=sys.stderr)
    sys.exit(1)


def list_files(folder_id: str, token: str) -> List[dict]:
    """List all files and subfolders in a Drive folder."""
    url = (
        f"{DRIVE_META}?q='{folder_id}'+in+parents+and+trashed=false"
        f"&fields=files(id,name,mimeType,size,modifiedTime)&pageSize=1000"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()).get("files", [])
    except urllib.error.HTTPError as e:
        print(f"API Error listing folder {folder_id}: {e.code} {e.reason}", file=sys.stderr)
        return []


def create_folder(name: str, parent_id: Optional[str], token: str) -> str:
    """Create a folder on Drive, return its ID."""
    payload: Dict[str, object] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        payload["parents"] = [parent_id]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{DRIVE_META}?fields=id",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["id"]


def get_or_create_folder(name: str, parent_id: Optional[str], token: str) -> str:
    """Find an existing folder by name or create it if absent."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"

    url = f"{DRIVE_META}?q={urllib.request.quote(query)}&fields=files(id,name)&pageSize=10"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            files = json.loads(resp.read()).get("files", [])
            if files:
                return files[0]["id"]
    except urllib.error.HTTPError:
        pass

    return create_folder(name, parent_id or "root", token)


def delete_file(file_id: str, token: str) -> None:
    """Delete a file from Drive."""
    req = urllib.request.Request(
        f"{DRIVE_META}/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        method="DELETE",
    )
    urllib.request.urlopen(req)


def upload_file(local_path: Path, parent_id: str, token: str) -> str:
    """Upload a file using multipart or resumable upload. Return file ID."""
    file_size = local_path.stat().st_size

    # Resumable upload for files > 5MB
    if file_size > 5 * 1024 * 1024:
        return upload_file_resumable(local_path, parent_id, token)

    file_name = local_path.name
    boundary = "----HiggsDriveBoundary7MA4YWxkTrZu0gW"
    metadata = json.dumps({"name": file_name, "parents": [parent_id]})

    with open(local_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{metadata}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{DRIVE_API}?uploadType=multipart&fields=id",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["id"]


def upload_file_resumable(local_path: Path, parent_id: str, token: str) -> str:
    """Resumable upload for large audio/weights files."""
    file_name = local_path.name
    file_size = local_path.stat().st_size

    metadata = json.dumps({"name": file_name, "parents": [parent_id]}).encode("utf-8")
    req = urllib.request.Request(
        f"{DRIVE_API}?uploadType=resumable&fields=id",
        data=metadata,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        upload_url = resp.headers["Location"]

    with open(local_path, "rb") as f:
        file_data = f.read()

    req = urllib.request.Request(
        upload_url,
        data=file_data,
        headers={
            "Content-Length": str(file_size),
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["id"]


def upload_directory(
    local_dir: Path,
    dest_folder_id: str,
    token: str,
    overwrite: bool = True,
) -> Tuple[int, int]:
    """Upload all files in a directory to the target Drive folder."""
    existing = {f["name"]: f["id"] for f in list_files(dest_folder_id, token)}
    files = [f for f in sorted(local_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]

    total_files = len(files)
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"Uploading {total_files} files ({total_bytes / (1024 * 1024):.2f} MB) from {local_dir}...")

    uploaded_count = 0
    uploaded_bytes = 0
    for idx, fpath in enumerate(files, 1):
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  [{idx}/{total_files}] {fpath.name} ({size_mb:.2f} MB)...", end=" ", flush=True)

        if fpath.name in existing:
            if overwrite:
                try:
                    delete_file(existing[fpath.name], token)
                except Exception:
                    pass
            else:
                print("SKIPPED (already exists)")
                continue

        try:
            upload_file(fpath, dest_folder_id, token)
            uploaded_count += 1
            uploaded_bytes += fpath.stat().st_size
            pct = (uploaded_bytes / total_bytes * 100) if total_bytes else 100
            print(f"OK ({pct:.0f}%)")
        except urllib.error.HTTPError as e:
            print(f"FAILED ({e.code} {e.reason})")

    return uploaded_count, uploaded_bytes


def download_file(file_id: str, dest_path: Path, token: str) -> None:
    """Download a file from Drive in streaming chunks."""
    url = f"{DRIVE_META}/{file_id}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as resp:
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(128 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def download_folder_recursive(folder_id: str, local_dir: Path, token: str) -> int:
    """Recursively download all files from a Drive folder."""
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(folder_id, token)
    downloaded = 0
    for f in sorted(files, key=lambda x: x["name"]):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            downloaded += download_folder_recursive(f["id"], local_dir / f["name"], token)
        else:
            dest = local_dir / f["name"]
            size = int(f.get("size", 0))
            if dest.exists() and dest.stat().st_size == size:
                print(f"  [Skip] {dest.name} (identical size: {size / (1024 * 1024):.2f} MB)")
                continue

            size_mb = size / (1024 * 1024)
            print(f"  [Download] {dest.name} ({size_mb:.2f} MB)...", end=" ", flush=True)
            try:
                download_file(f["id"], dest, token)
                downloaded += 1
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive synchronization for Higgs Audio benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload command
    p_upload = subparsers.add_parser("upload", help="Upload local benchmark files to Google Drive")
    p_upload.add_argument("--source", choices=["samples", "output", "notebooks"], help="Source directory to upload")
    p_upload.add_argument("--all", action="store_true", help="Upload samples, outputs, and notebooks")
    p_upload.add_argument("--folder-id", help="Target Google Drive folder ID")
    p_upload.add_argument("--folder-name", default="higgs-benchmark", help="Root folder name on Drive")
    p_upload.add_argument("--token", help="Explicit OAuth2 token")
    p_upload.add_argument("--account", help="Specific gcloud account email")

    # download command
    p_down = subparsers.add_parser("download", help="Download files from Google Drive")
    p_down.add_argument("--folder-id", required=True, help="Source Google Drive folder ID")
    p_down.add_argument("--dest", required=True, help="Local destination folder (e.g., output, samples)")
    p_down.add_argument("--token", help="Explicit OAuth2 token")
    p_down.add_argument("--account", help="Specific gcloud account email")

    # list command
    p_list = subparsers.add_parser("list", help="List files in a Google Drive folder")
    p_list.add_argument("--folder-id", required=True, help="Google Drive folder ID")
    p_list.add_argument("--token", help="Explicit OAuth2 token")
    p_list.add_argument("--account", help="Specific gcloud account email")

    args = parser.parse_args()
    token = get_token(getattr(args, "token", None), getattr(args, "account", None))

    if args.command == "upload":
        root_folder_id = args.folder_id or get_or_create_folder(args.folder_name, None, token)
        print(f"Target Drive folder: https://drive.google.com/drive/folders/{root_folder_id}")

        sources = []
        if args.all:
            sources = ["samples", "output", "notebooks"]
        elif args.source:
            sources = [args.source]
        else:
            sources = ["samples", "output"]

        for src in sources:
            src_dir = ROOT_DIR / src
            if not src_dir.exists():
                print(f"Skipping {src}: directory {src_dir} does not exist.")
                continue
            subfolder_id = get_or_create_folder(src, root_folder_id, token)
            print(f"\nSyncing {src}/ -> Drive folder ID: {subfolder_id}")
            upload_directory(src_dir, subfolder_id, token)

        print(f"\nUpload complete: https://drive.google.com/drive/folders/{root_folder_id}")

    elif args.command == "download":
        dest_dir = Path(args.dest) if Path(args.dest).is_absolute() else ROOT_DIR / args.dest
        print(f"Downloading Drive folder {args.folder_id} -> {dest_dir}...")
        count = download_folder_recursive(args.folder_id, dest_dir, token)
        print(f"\nDownload complete: {count} new/updated files in {dest_dir}")

    elif args.command == "list":
        files = list_files(args.folder_id, token)
        print(f"Files in folder {args.folder_id} ({len(files)} items):")
        for f in files:
            size = f"{int(f.get('size', 0)) / (1024 * 1024):.2f} MB" if "size" in f else "<dir>"
            print(f"  - {f['name']} ({size}) [id: {f['id']}]")


if __name__ == "__main__":
    main()
