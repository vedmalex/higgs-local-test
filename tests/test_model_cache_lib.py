#!/usr/bin/env python3
"""Tests for scripts/model_cache_lib.py (shared HF-cache archiving recipe used by
scripts/model_drive_sync.py) and notebooks/model_catalog.json's structure, plus a drift
guard tying notebooks/model_prefetch_to_drive.ipynb's inlined packing cell back to the
same recipe -- see model_cache_lib.py's module docstring for why the notebook cell can't
just import this module (it runs on Colab, which only fetches the catalog JSON, not this
repo checkout).

Run with:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import model_cache_lib as mcl  # noqa: E402

CATALOG_PATH = ROOT / "notebooks" / "model_catalog.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "model_prefetch_to_drive.ipynb"

VALID_STATUSES = {
    "local+drive",
    "drive_only",
    "local_only",
    "needed",
    "candidate",
    "rejected",
}


class TestNamingHelpers(unittest.TestCase):
    def test_safe_repo_dirname(self):
        self.assertEqual(
            mcl.safe_repo_dirname("bosonai/higgs-tts-3-4b"),
            "models--bosonai--higgs-tts-3-4b",
        )

    def test_repo_id_from_dirname_round_trip(self):
        for repo_id in ["bosonai/higgs-tts-3-4b", "mlx-community/Qwen3-ASR-0.6B-8bit"]:
            dirname = mcl.safe_repo_dirname(repo_id)
            self.assertEqual(mcl.repo_id_from_dirname(dirname), repo_id)

    def test_repo_id_from_dirname_rejects_non_hf_dirs(self):
        self.assertIsNone(mcl.repo_id_from_dirname(".locks"))
        self.assertIsNone(mcl.repo_id_from_dirname("CACHEDIR.TAG"))
        self.assertIsNone(mcl.repo_id_from_dirname("models--nosplit"))

    def test_archive_name(self):
        self.assertEqual(mcl.archive_name("higgs-tts-3-4b"), "higgs-tts-3-4b.tar")

    def test_hub_model_dir(self):
        hf_home = Path("/tmp/fake-hf-home")
        got = mcl.hub_model_dir("bosonai/higgs-tts-3-4b", hf_home)
        self.assertEqual(got, hf_home / "hub" / "models--bosonai--higgs-tts-3-4b")


class TestIsLocallyComplete(unittest.TestCase):
    def test_missing_dir_is_incomplete(self):
        self.assertFalse(mcl.is_locally_complete(Path("/tmp/does-not-exist-model-cache-lib-test")))

    def test_dir_without_blobs_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models--org--name"
            model_dir.mkdir()
            self.assertFalse(mcl.is_locally_complete(model_dir))

    def test_dir_with_incomplete_blob_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models--org--name"
            (model_dir / "blobs").mkdir(parents=True)
            (model_dir / "blobs" / "abc123.incomplete").write_bytes(b"partial")
            self.assertFalse(mcl.is_locally_complete(model_dir))

    def test_dir_with_only_finished_blobs_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models--org--name"
            (model_dir / "blobs").mkdir(parents=True)
            (model_dir / "blobs" / "abc123").write_bytes(b"done")
            self.assertTrue(mcl.is_locally_complete(model_dir))


class TestPackModelDir(unittest.TestCase):
    """The one thing that must never regress: extracting the tar reproduces the exact
    blob/snapshot symlink layout huggingface_hub expects -- real files as real files,
    symlinks as symlinks pointing at relative blob paths, nothing dereferenced."""

    def _make_fake_hf_model(self, root: Path) -> Path:
        model_dir = root / "models--acme--tiny-model"
        blobs = model_dir / "blobs"
        snapshots = model_dir / "snapshots" / "deadbeef"
        blobs.mkdir(parents=True)
        snapshots.mkdir(parents=True)
        blob_path = blobs / "0123456789abcdef"
        blob_path.write_bytes(b"fake weight bytes")
        # Mirrors the real HF cache layout: snapshot files are symlinks into blobs/.
        (snapshots / "config.json").symlink_to(Path("..") / ".." / "blobs" / "0123456789abcdef")
        return model_dir

    def test_pack_and_extract_preserves_symlink_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._make_fake_hf_model(root)
            dest_tar = root / "archive" / "tiny-model.tar"

            returned = mcl.pack_model_dir(model_dir, dest_tar)
            self.assertEqual(returned, dest_tar)
            self.assertTrue(dest_tar.is_file())

            extract_dir = root / "extracted"
            extract_dir.mkdir()
            with tarfile.open(dest_tar, "r") as tf:
                tf.extractall(extract_dir)

            extracted_model_dir = extract_dir / "models--acme--tiny-model"
            self.assertTrue(extracted_model_dir.is_dir())

            extracted_symlink = extracted_model_dir / "snapshots" / "deadbeef" / "config.json"
            self.assertTrue(extracted_symlink.is_symlink(), "snapshot entry must stay a symlink")
            self.assertEqual(
                extracted_symlink.read_bytes(),
                b"fake weight bytes",
                "symlink must still resolve to the real blob content after extraction",
            )

    def test_pack_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._make_fake_hf_model(root)
            dest_tar = root / "a" / "b" / "c" / "tiny-model.tar"
            mcl.pack_model_dir(model_dir, dest_tar)
            self.assertTrue(dest_tar.is_file())


class TestNotebookPackingCellMatchesInvariants(unittest.TestCase):
    """Drift guard: the notebook cannot import model_cache_lib.py (Colab only fetches
    model_catalog.json over HTTP, it never checks out this repo -- see model_cache_lib.py's
    module docstring). Its packing cell is therefore a deliberate copy of the same recipe,
    and this test is what keeps that copy honest instead of trusting review alone."""

    def _find_packing_cell_source(self) -> str:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if "tarfile.open" in source:
                return source
        self.fail("No code cell in the notebook calls tarfile.open(...) -- packing cell missing or renamed")

    def test_packing_cell_uses_the_same_tar_open_mode(self):
        source = self._find_packing_cell_source()
        self.assertIn(f'tarfile.open(local_tar, "{mcl.TAR_OPEN_MODE}")', source)

    def test_packing_cell_does_not_dereference_symlinks(self):
        source = self._find_packing_cell_source()
        self.assertNotIn("dereference=True", source)

    def test_packing_cell_uses_bare_dirname_as_arcname(self):
        source = self._find_packing_cell_source()
        self.assertIn("tf.add(model_dir, arcname=model_dir.name)", source)


class TestModelCatalogStructure(unittest.TestCase):
    """Schema sanity for notebooks/model_catalog.json, the single source of truth both
    the notebook and scripts/model_drive_sync.py read."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_top_level_shape(self):
        self.assertIn("models", self.catalog)
        self.assertIsInstance(self.catalog["models"], list)
        self.assertGreater(len(self.catalog["models"]), 0)

    def test_names_are_unique(self):
        names = [m["name"] for m in self.catalog["models"]]
        self.assertEqual(len(names), len(set(names)), "duplicate 'name' entries in the catalog")

    def test_every_entry_has_required_fields(self):
        required = {
            "name",
            "repo_id",
            "revision",
            "allow_patterns",
            "role",
            "license",
            "size_gb",
            "status",
            "fetch_now",
            "notes",
        }
        for m in self.catalog["models"]:
            missing = required - set(m.keys())
            self.assertFalse(missing, f"{m.get('name')}: missing fields {missing}")

    def test_status_values_are_known(self):
        for m in self.catalog["models"]:
            self.assertIn(
                m["status"],
                VALID_STATUSES,
                f"{m['name']}: unknown status {m['status']!r}",
            )

    def test_fetch_now_entries_have_a_pinned_repo_id(self):
        # Mirrors the notebook's own assertion in cell 4 -- catch this here instead of
        # discovering it only when the notebook runs in Colab.
        for m in self.catalog["models"]:
            if m["fetch_now"]:
                self.assertTrue(
                    m["repo_id"],
                    f"{m['name']}: fetch_now=true but repo_id is not pinned",
                )

    def test_local_or_drive_status_implies_pinned_repo_id(self):
        for m in self.catalog["models"]:
            if m["status"] in ("local+drive", "drive_only", "local_only"):
                self.assertTrue(
                    m["repo_id"],
                    f"{m['name']}: status={m['status']!r} but repo_id is not pinned",
                )


if __name__ == "__main__":
    unittest.main()
