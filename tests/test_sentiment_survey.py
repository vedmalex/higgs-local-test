#!/usr/bin/env python3
"""Regression tests for src/sentiment_survey/ (issue #57 blind sentiment survey app).

Pure-function tests only -- no server socket, no audio playback, no GPU. Run with:
    .venv-tts/bin/python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
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


class TestCorrectionHistory(unittest.TestCase):
    """Requirement 2 (issue #57 follow-up): a correction must be a new,
    separately timestamped record, never an in-place overwrite -- and
    grading/`load_answers()` must resolve to the latest one."""

    def setUp(self):
        self._orig_results_dir = server.RESULTS_DIR
        self.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = self.tmp_dir

    def tearDown(self):
        server.RESULTS_DIR = self._orig_results_dir

    def test_correction_appends_new_line_and_keeps_old(self):
        rec1 = {"task_id": "t1", "answer_label": "Да", "correct_answer": "Да",
                "matches_expected": True, "type": "single_rating", "listen_ms": 100,
                "timestamp": "x", "hidden": {}}
        server.append_answer("demo", dict(rec1))
        rec2 = {"task_id": "t1", "answer_label": "Нет", "correct_answer": "Да",
                 "matches_expected": False, "type": "single_rating", "listen_ms": 50,
                 "timestamp": "y", "hidden": {}}
        server.append_answer("demo", dict(rec2))

        history = server.load_answer_history("demo")
        self.assertEqual(len(history), 2, "both the original and the correction must survive on disk")
        self.assertEqual(history[0]["answer_label"], "Да")
        self.assertEqual(history[0]["revision"], 1)
        self.assertFalse(history[0]["is_correction"])
        self.assertEqual(history[1]["answer_label"], "Нет")
        self.assertEqual(history[1]["revision"], 2)
        self.assertTrue(history[1]["is_correction"])
        self.assertTrue(history[1]["answered_after_reveal"])
        self.assertEqual(history[1]["replaces_revision"], 1)

    def test_load_answers_resolves_to_latest_revision(self):
        server.append_answer("demo", {"task_id": "t1", "answer_label": "Да",
                                        "correct_answer": "Да", "matches_expected": True,
                                        "type": "single_rating", "listen_ms": 0,
                                        "timestamp": "x", "hidden": {}})
        server.append_answer("demo", {"task_id": "t1", "answer_label": "Нет",
                                        "correct_answer": "Да", "matches_expected": False,
                                        "type": "single_rating", "listen_ms": 0,
                                        "timestamp": "y", "hidden": {}})
        answers = server.load_answers("demo")
        self.assertEqual(set(answers), {"t1"})
        self.assertEqual(answers["t1"]["answer_label"], "Нет")
        self.assertTrue(answers["t1"]["is_correction"])

    def test_old_record_without_new_fields_reads_as_non_correction(self):
        # Simulates an answers.jsonl written before this feature existed
        # (e.g. the owner's real unheard_sfx_env/answers.jsonl).
        d = self.tmp_dir / "legacy"
        d.mkdir()
        old_line = json.dumps({
            "task_id": "catalog-env-noise", "set_id": "legacy", "type": "single_rating",
            "answer_kind": None, "question": "Слышен ли фоновый шум?",
            "answer_label": "Да, фоновый шум слышен, речь не пострадала",
            "answer_role": None, "listen_ms": 0, "timestamp": "2026-08-25T12:25:24+0300",
            "hidden": {"A": {"tag": "env:noise"}}, "correct_answer": None,
            "matches_expected": None, "skipped_prior": False,
        }, ensure_ascii=False)
        (d / "answers.jsonl").write_text(old_line + "\n", encoding="utf-8")

        history = server.load_answer_history("legacy")
        self.assertEqual(len(history), 1)
        self.assertNotIn("revision", history[0])
        answers = server.load_answers("legacy")
        self.assertEqual(set(answers), {"catalog-env-noise"})
        self.assertFalse(answers["catalog-env-noise"].get("is_correction"))
        self.assertFalse(answers["catalog-env-noise"].get("answered_after_reveal"))

        # A correction on top of a legacy record must still work and not
        # crash on the missing key (prior_count derived by task_id match).
        server.append_answer("legacy", {"task_id": "catalog-env-noise",
                                          "answer_label": "Нет", "correct_answer": None,
                                          "matches_expected": None, "type": "single_rating",
                                          "listen_ms": 0, "timestamp": "z", "hidden": {}})
        answers = server.load_answers("legacy")
        self.assertEqual(answers["catalog-env-noise"]["revision"], 2)
        self.assertTrue(answers["catalog-env-noise"]["is_correction"])


@unittest.skipUnless(OUTPUT_PRESENT, "output/ audio fixtures not present in this checkout (gitignored)")
class TestNavigationOverHTTP(unittest.TestCase):
    """End-to-end check that the actual HTTP server (issue #57 follow-up:
    back/forward navigation, jump-to-task list, corrections) behaves as
    documented -- not just the pure functions above."""

    @classmethod
    def setUpClass(cls):
        cls._orig_results_dir = server.RESULTS_DIR
        cls.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = cls.tmp_dir
        cls.httpd = server.ThreadingHTTPServer((server.HOST, 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        # A JSON-defined set whose clip files are copied into this checkout's
        # output/ for the test fixtures (see worktree setup).
        cls.set_id = "final_intonation"
        assert cls.set_id in server.TASK_SETS, "final_intonation task set failed to load"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        server.RESULTS_DIR = cls._orig_results_dir

    def _get(self, path):
        conn = http.client.HTTPConnection(server.HOST, self.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
        finally:
            conn.close()

    def _post_json(self, path, obj):
        conn = http.client.HTTPConnection(server.HOST, self.port, timeout=5)
        try:
            data = json.dumps(obj).encode("utf-8")
            conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
        finally:
            conn.close()

    def test_tasks_list_and_jump_to_arbitrary_task(self):
        status, body = self._get(f"/api/sets/{self.set_id}/tasks")
        self.assertEqual(status, 200)
        self.assertEqual(body["total"], len(server.TASK_SETS[self.set_id].tasks))
        ids = [t["id"] for t in body["tasks"]]
        self.assertIn("final-punct-question", ids)

        # Jump directly to a task that is neither first nor next-unanswered.
        status, detail = self._get(f"/api/sets/{self.set_id}/task/final-punct-question")
        self.assertEqual(status, 200)
        self.assertEqual(detail["task"]["id"], "final-punct-question")
        self.assertIsNone(detail["previous_answer"])

    def test_returning_to_an_answered_task_returns_the_prior_answer(self):
        task_id = "final-punct-exclaim"
        status, before = self._get(f"/api/sets/{self.set_id}/task/{task_id}")
        self.assertEqual(status, 200)
        self.assertIsNone(before["previous_answer"])

        status, ans = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "answer_label": "Восклицание", "listen_ms": 1200,
        })
        self.assertEqual(status, 200)
        self.assertTrue(ans["ok"])

        # Navigate away and back -- the server must hand back the same answer.
        status, after = self._get(f"/api/sets/{self.set_id}/task/{task_id}")
        self.assertEqual(status, 200)
        self.assertIsNotNone(after["previous_answer"])
        self.assertEqual(after["previous_answer"]["answer_label"], "Восклицание")
        self.assertTrue(after["previous_answer"]["matches_expected"])
        self.assertIsNotNone(after["reveal"])  # already revealed, honestly shown again

    def test_correction_is_recorded_and_counted_as_non_blind_in_summary(self):
        task_id = "final-punct-period"
        status, ans1 = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "answer_label": "Вопрос", "listen_ms": 500,
        })
        self.assertEqual(status, 200)
        self.assertFalse(server.load_answers(self.set_id)[task_id]["is_correction"])

        # Owner realizes the mistake and corrects it.
        status, ans2 = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "answer_label": "Утверждение (точка)", "listen_ms": 300,
        })
        self.assertEqual(status, 200)

        history = server.load_answer_history(self.set_id)
        task_history = [r for r in history if r["task_id"] == task_id]
        self.assertEqual(len(task_history), 2, "the wrong first answer must not be discarded")
        self.assertEqual(task_history[0]["answer_label"], "Вопрос")
        self.assertEqual(task_history[1]["answer_label"], "Утверждение (точка)")

        latest = server.load_answers(self.set_id)[task_id]
        self.assertTrue(latest["is_correction"])
        self.assertTrue(latest["answered_after_reveal"])
        self.assertTrue(latest["matches_expected"], "grading must use the corrected (latest) answer")

        # Task list reflects the correction.
        status, tasks = self._get(f"/api/sets/{self.set_id}/tasks")
        row = next(t for t in tasks["tasks"] if t["id"] == task_id)
        self.assertTrue(row["is_correction"])
        self.assertTrue(row["answered_after_reveal"])

        # Summary: latest (correct) answer is used, but since it was given
        # after the reveal it must NOT count in the blind graded bucket.
        status, summary = self._get(f"/api/sets/{self.set_id}/summary")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(summary["answered_after_reveal"], 1)
        graded_task_ids = {r["task_id"] for r in summary["answers"]
                            if r["task_id"] == task_id and not r.get("answered_after_reveal")}
        self.assertEqual(graded_task_ids, set(), "the corrected answer must not be double counted as blind")


if __name__ == "__main__":
    unittest.main()
