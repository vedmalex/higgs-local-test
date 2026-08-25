#!/usr/bin/env python3
"""Regression tests for src/sentiment_survey/ (issue #57 blind sentiment survey app).

Pure-function tests only -- no server socket, no audio playback, no GPU. Run with:
    .venv-tts/bin/python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVEY_DIR = REPO_ROOT / "src" / "sentiment_survey"
sys.path.insert(0, str(SURVEY_DIR))

import tag_reference  # noqa: E402
import catalog  # noqa: E402
import server  # noqa: E402

OUTPUT_PRESENT = (REPO_ROOT / "output" / "m4_tag_catalog" / "neutral_baseline.wav").is_file()


class TestTagReferenceParser(unittest.TestCase):
    FIXTURE = """\
| **emotion:sadness** | 151699 | -52.1 | -0.9 | 1 | -100 | -0.38 | 1.4 | -38.2 | `output/m4_tag_catalog/tag_emotion_sadness.wav` |
| **emotion:anger** | 151695 | -30.2 | -8.2 | 2 | 60 | -0.35 | 1.28 | 44.2 | `output/m4_tag_catalog/tag_emotion_anger.wav` |

- `emotion:sadness`: Подтверждено владельцем вслепую (M4-T0): звучит грустно, PASSED.
- `emotion:anger`: Группа C по объективной триаге M4-T5 (PR #108) -- близкий к нейтрали.
"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8")
        self.tmp.write(self.FIXTURE)
        self.tmp.close()

    def test_parses_table_rows_and_status(self):
        tags = tag_reference.parse_tag_reference(Path(self.tmp.name))
        self.assertEqual(set(tags), {"emotion:sadness", "emotion:anger"})
        self.assertEqual(tags["emotion:sadness"]["id"], "151699")
        self.assertTrue(tags["emotion:sadness"]["confirmed"])
        self.assertIsNone(tags["emotion:sadness"]["group"])
        self.assertFalse(tags["emotion:anger"]["confirmed"])
        self.assertEqual(tags["emotion:anger"]["group"], "C")

    def test_missing_file_returns_empty_dict_not_crash(self):
        self.assertEqual(tag_reference.parse_tag_reference(Path("/no/such/file.md")), {})


@unittest.skipUnless(OUTPUT_PRESENT, "output/m4_tag_catalog/ not present in this checkout (gitignored)")
class TestCatalogScanAgainstRealOutput(unittest.TestCase):
    def test_catalog_sets_cover_sfx_and_env(self):
        sets = catalog.build_catalog_sets()
        by_id = {s["id"]: s for s in sets}
        self.assertIn("unheard_sfx_env", by_id)
        tags_seen = {t["hidden"]["A"]["tag"] for t in by_id["unheard_sfx_env"]["tasks"]}
        self.assertEqual(len(tags_seen), 11)  # 9 sfx + 2 env
        self.assertTrue(all(t.startswith("sfx:") or t.startswith("env:") for t in tags_seen))

    def test_disputed_set_matches_group_b_or_c(self):
        sets = catalog.build_catalog_sets()
        by_id = {s["id"]: s for s in sets}
        if "disputed_tags" not in by_id:
            self.skipTest("tag_reference.md not present or no B/C tags found")
        tags_ref = tag_reference.parse_tag_reference(catalog.TAG_REF_PATH)
        for task in by_id["disputed_tags"]["tasks"]:
            tag_key = task["hidden"]["A"]["tag"]
            self.assertIn(tags_ref[tag_key]["group"], ("B", "C"))

    def test_no_dynamic_set_mixes_two_runs(self):
        # every clip pair inside one dynamic task set must come from the same
        # output/ subdirectory (never m4_tag_catalog vs m4_tags, etc.)
        for doc in catalog.build_all_dynamic_sets():
            for task in doc["tasks"]:
                dirs = {Path(p).parent for p in task["clips"].values()}
                self.assertEqual(len(dirs), 1, f"{doc['id']}/{task['id']} mixes clips from {dirs}")


class TestTaskSetJSONFilesValidate(unittest.TestCase):
    """The hand-written JSON sets must at least be internally consistent;
    file-existence is only checked when output/ is actually present."""

    def test_all_task_set_json_parse_and_have_unique_ids(self):
        for path in sorted((SURVEY_DIR / "task_sets").glob("*.json")):
            with path.open(encoding="utf-8") as fh:
                doc = json.load(fh)
            ids = [t["id"] for t in doc["tasks"]]
            self.assertEqual(len(ids), len(set(ids)), f"{path} has duplicate task ids")
            if OUTPUT_PRESENT:
                ts = server.TaskSet(doc, path)  # raises on missing clip files
                self.assertTrue(ts.tasks)


class TestServerAnswerLogic(unittest.TestCase):
    def test_compute_matches_expected_differ(self):
        task = {"answer_kind": "differ", "hidden": {"correct_answer": "Да, отличаются"}}
        matches, expected = server.compute_matches_expected(task, None, "Да, отличаются", "Да, отличаются")
        self.assertTrue(matches)
        self.assertEqual(expected, "Да, отличаются")

    def test_compute_matches_expected_which_uses_role(self):
        task = {"answer_kind": "which", "hidden": {"correct_answer": "A"}}
        matches, _ = server.compute_matches_expected(task, None, "A", "Клип 1")
        self.assertTrue(matches)
        matches, _ = server.compute_matches_expected(task, None, "B", "Клип 2")
        self.assertFalse(matches)

    def test_compute_matches_expected_no_ground_truth(self):
        task = {"answer_kind": None, "hidden": {}}
        matches, expected = server.compute_matches_expected(task, None, "Да", "Да")
        self.assertIsNone(matches)
        self.assertIsNone(expected)


class TestAtomicResultsRoundTrip(unittest.TestCase):
    """append_answer() must be resumable and never leave a stray temp file
    behind, matching the project's own atomic-write lesson."""

    def setUp(self):
        self._orig_results_dir = server.RESULTS_DIR
        self.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = self.tmp_dir

    def tearDown(self):
        server.RESULTS_DIR = self._orig_results_dir

    def test_append_then_reload_resumes(self):
        rec1 = {"task_id": "t1", "correct_answer": "yes", "matches_expected": True,
                "type": "single_rating", "listen_ms": 100, "timestamp": "x", "hidden": {}}
        server.append_answer("demo", rec1)
        loaded = server.load_answers("demo")
        self.assertEqual(set(loaded), {"t1"})

        rec2 = {"task_id": "t2", "correct_answer": None, "matches_expected": None,
                 "type": "single_rating", "listen_ms": 50, "timestamp": "y", "hidden": {}}
        server.append_answer("demo", rec2)
        loaded = server.load_answers("demo")
        self.assertEqual(set(loaded), {"t1", "t2"})

        # no leftover .tmp* files after two atomic writes
        leftovers = list((self.tmp_dir / "demo").glob("*.tmp*"))
        self.assertEqual(leftovers, [])

    def test_answers_jsonl_is_one_json_object_per_line(self):
        server.append_answer("demo2", {"task_id": "a", "correct_answer": None,
                                         "matches_expected": None, "type": "x",
                                         "listen_ms": 0, "timestamp": "t", "hidden": {}})
        jsonl_path = self.tmp_dir / "demo2" / "answers.jsonl"
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        json.loads(lines[0])  # must not raise


if __name__ == "__main__":
    unittest.main()
