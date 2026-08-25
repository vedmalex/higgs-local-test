#!/usr/bin/env python3
"""
Shared Hugging Face model-cache archiving logic.

This is the ONE place that knows how to turn a `~/.cache/huggingface/hub/models--...`
directory into a `.tar` that is safe to hand to Google Drive and safe to restore later.
Both directions of the sync use it:

  - notebooks/model_prefetch_to_drive.ipynb (download direction, Drive -> nowhere, it just
    fills Drive from the internet) packs the model it just downloaded with this exact
    recipe, inlined in its own code cell -- Colab notebooks run on a separate machine and
    only fetch notebooks/model_catalog.json over raw.githubusercontent.com; they do not
    check out this repo, so the notebook CANNOT `import scripts.model_cache_lib`. Its cell
    is a deliberate copy, not a second design.
  - scripts/model_drive_sync.py (upload direction, local cache -> Drive) imports this
    module directly, since it runs on the local machine that already has the checkout.

Because the notebook's copy cannot be a real import, the invariant that keeps the two
directions restore-compatible is enforced instead by
tests/test_model_cache_lib.py::test_notebook_pack_cell_matches_invariants, which parses
the notebook JSON and asserts its packing cell still contains the exact call shapes below
(TAR_OPEN_MODE, `tf.add(model_dir, arcname=model_dir.name)`). If you change how this module
packs a model, that test will fail until the notebook cell is edited to match -- treat it as
the drift guard rather than trusting review alone.

Golden rule: `tarfile.open(dest, TAR_OPEN_MODE)` then `tf.add(model_dir, arcname=model_dir.name)`,
no `dereference=True`, no re-fnamed members. Never let tarfile follow symlinks -- the whole
point of archiving is to carry the Hugging Face cache's blob/snapshot symlink layout, which
Google Drive's FUSE mount does not preserve on loose files.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

# Kept as a module constant (not a literal in each call site) so the notebook-drift test
# can quote it instead of hardcoding "w" a second time.
TAR_OPEN_MODE = "w"


def safe_repo_dirname(repo_id: str) -> str:
    """Hugging Face's own cache directory naming: 'org/name' -> 'models--org--name'."""
    return "models--" + repo_id.replace("/", "--")


def repo_id_from_dirname(dirname: str) -> Optional[str]:
    """Reverse of safe_repo_dirname. Returns None for anything not shaped like a HF model
    cache directory. Assumes a single 'org/name' split (the convention every repo_id in
    this project's catalog follows) -- good enough for reconciliation/orphan-detection,
    not a general-purpose HF cache parser."""
    prefix = "models--"
    if not dirname.startswith(prefix):
        return None
    rest = dirname[len(prefix):]
    if "--" not in rest:
        return None
    org, _, name = rest.partition("--")
    return f"{org}/{name}"


def hub_model_dir(repo_id: str, hf_home: Path) -> Path:
    """Where a given repo_id's snapshot lives under an HF_HOME (local: ~/.cache/huggingface,
    Colab: /content/hf-cache -- both use HF_HOME/hub/models--...)."""
    return hf_home / "hub" / safe_repo_dirname(repo_id)


def local_hf_hub_dir() -> Path:
    """The local machine's default Hugging Face hub cache directory."""
    return Path.home() / ".cache" / "huggingface" / "hub"


def archive_name(model_name: str) -> str:
    """Catalog `name` -> the .tar filename used on Drive, e.g. 'higgs-tts-3-4b.tar'."""
    return f"{model_name}.tar"


def is_locally_complete(model_dir: Path) -> bool:
    """True if model_dir exists and has no leftover .incomplete blobs -- i.e. the snapshot
    finished downloading and is safe to pack. Same check the notebook's cell 6 makes before
    deciding to skip a fresh snapshot_download call."""
    if not model_dir.is_dir():
        return False
    blobs = model_dir / "blobs"
    if not blobs.is_dir():
        return False
    return not any(blobs.glob("*.incomplete"))


def pack_model_dir(model_dir: Path, dest_tar: Path) -> Path:
    """Pack an HF cache model directory into dest_tar, preserving the blob/snapshot symlink
    layout untouched (no dereferencing, no renaming beyond dropping the parent path --
    arcname is model_dir.name so extracting with `tar -xf dest_tar -C .../hub/` recreates
    the exact `models--org--name` directory `huggingface_hub` expects).

    This is byte-for-shape the same recipe as notebooks/model_prefetch_to_drive.ipynb's
    cell 6 -- see the module docstring for why that cell can't just call this function."""
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_tar, TAR_OPEN_MODE) as tf:
        tf.add(model_dir, arcname=model_dir.name)
    return dest_tar
