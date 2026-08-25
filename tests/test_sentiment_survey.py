#!/usr/bin/env python3
"""Regression tests for src/sentiment_survey/ (issue #57 blind sentiment survey app).

Pure-function tests only -- no server socket, no audio playback, no GPU. Run with:
    .venv-tts/bin/python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import http.client
import json
import math
import struct
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVEY_DIR = REPO_ROOT / "src" / "sentiment_survey"
sys.path.insert(0, str(SURVEY_DIR))

import tag_reference  # noqa: E402
import catalog  # noqa: E402
import server  # noqa: E402
import pitch  # noqa: E402

OUTPUT_PRESENT = (REPO_ROOT / "output" / "m4_tag_catalog" / "neutral_baseline.wav").is_file()
CHAPTER114E0_PRESENT = (REPO_ROOT / "output" / "chapter-114-e0" / "manifest.json").is_file()

try:
    import numpy  # noqa: F401
    NUMPY_PRESENT = True
except ImportError:
    NUMPY_PRESENT = False


def _write_tone_wav(path: Path, freq_hz: float, dur_s: float = 1.0, sr: int = 16000) -> None:
    """A pure sine tone WAV -- synthetic stand-in for "a clip whose median F0
    is (approximately) freq_hz", so pitch.py's tests don't depend on real
    generated speech."""
    n = int(dur_s * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            val = int(16000 * math.sin(2 * math.pi * freq_hz * i / sr))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))


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


class TestMissingClipDoesNotTakeDownTheApp(unittest.TestCase):
    """A missing wav in ONE set must not make the whole app unloadable.

    Regression: `output/` is gitignored, so a cleaned output/, a fresh
    checkout, or a removed agent worktree that held the only copy of a clip
    left `load_task_sets()` raising at import time — which took down every
    other set too, including voice casting whose own audio was present.
    Structural authoring faults must still refuse to load, so the two cases
    are asserted together here rather than in separate tests.
    """

    def _doc(self, clip_path):
        return {
            "id": "probe_set",
            "title": "probe",
            "tasks": [{
                "id": "probe-task",
                "type": "pair_compare",
                "answer_kind": "which",
                "question": "?",
                "options": ["A", "B"],
                "clips": {"A": clip_path, "B": clip_path},
                "hidden": {"correct_answer": "A"},
            }],
        }

    def test_missing_clip_raises_the_transient_subclass(self):
        doc = self._doc("output/definitely_not_here_12345/nope.wav")
        with self.assertRaises(server.MissingClipError):
            server.TaskSet(doc, Path("<probe>"))

    def test_structural_faults_stay_plain_value_errors(self):
        """Not a MissingClipError — these are real bugs, never skipped."""
        doc = self._doc("output/whatever.wav")
        doc["tasks"][0]["type"] = "no_such_type"
        with self.assertRaises(ValueError) as ctx:
            server.TaskSet(doc, Path("<probe>"))
        self.assertNotIsInstance(ctx.exception, server.MissingClipError)

    def test_escaping_clip_path_is_not_treated_as_transient(self):
        doc = self._doc("../../etc/passwd")
        with self.assertRaises(ValueError) as ctx:
            server.TaskSet(doc, Path("<probe>"))
        self.assertNotIsInstance(ctx.exception, server.MissingClipError)

    def test_load_task_sets_survives_a_set_with_a_missing_clip(self):
        """The real loader, with a genuinely broken set dropped into the
        live task_sets dir — every other set must still come back."""
        broken = SURVEY_DIR / "task_sets" / "_zz_probe_broken.json"
        with broken.open("w", encoding="utf-8") as fh:
            json.dump(self._doc("output/definitely_not_here_12345/nope.wav"), fh)
        try:
            sets = server.load_task_sets()
        finally:
            broken.unlink()
        self.assertNotIn("probe_set", sets, "the broken set should be skipped")
        self.assertTrue(sets, "every other set must still load")


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


class TestFreeTextNotes(unittest.TestCase):
    """Free-text observation field (issue #57 follow-up): optional, editable
    via the same append/revision mechanism as a correction, but must NOT by
    itself flip is_correction / answered_after_reveal -- only an actual
    change of answer does that (see server._answer_value())."""

    def setUp(self):
        self._orig_results_dir = server.RESULTS_DIR
        self.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = self.tmp_dir

    def tearDown(self):
        server.RESULTS_DIR = self._orig_results_dir

    def _base_record(self, task_id="t1", answer_label="Да", note=""):
        return {
            "task_id": task_id, "answer_label": answer_label, "answer_role": None,
            "note": note, "correct_answer": None, "matches_expected": None,
            "type": "single_rating", "listen_ms": 0, "timestamp": "x", "hidden": {},
        }

    def test_note_saved_and_read_back(self):
        server.append_answer("demo", self._base_record(note="фон гуляет"))
        answers = server.load_answers("demo")
        self.assertEqual(answers["t1"]["note"], "фон гуляет")

    def test_note_only_edit_does_not_mark_correction_or_non_blind(self):
        server.append_answer("demo", self._base_record(note=""))
        server.append_answer("demo", self._base_record(note="голос между фрагментами разный"))
        answers = server.load_answers("demo")
        latest = answers["t1"]
        self.assertEqual(latest["note"], "голос между фрагментами разный")
        self.assertEqual(latest["revision"], 2)
        self.assertFalse(latest["is_correction"], "same answer + new note must not count as a correction")
        self.assertFalse(latest["answered_after_reveal"], "a note-only edit must not leave the blind bucket")

        history = server.load_answer_history("demo")
        self.assertEqual(len(history), 2, "both revisions must survive on disk")
        self.assertEqual(history[0]["note"], "")

    def test_note_change_combined_with_answer_change_is_still_a_correction(self):
        server.append_answer("demo", self._base_record(answer_label="Да", note=""))
        server.append_answer("demo", self._base_record(answer_label="Нет", note="передумал, послушав внимательнее"))
        latest = server.load_answers("demo")["t1"]
        self.assertTrue(latest["is_correction"])
        self.assertTrue(latest["answered_after_reveal"])
        self.assertEqual(latest["note"], "передумал, послушав внимательнее")

    def test_empty_note_does_not_break_answer(self):
        # No "note" key at all in the submitted record (client omitted it).
        rec = self._base_record()
        del rec["note"]
        answers = server.append_answer("demo", rec)
        self.assertEqual(answers["t1"]["answer_label"], "Да")
        self.assertNotIn("note", answers["t1"])  # append_answer doesn't invent one

    def test_old_record_without_note_field_reads_as_empty_string(self):
        d = self.tmp_dir / "legacy"
        d.mkdir()
        old_line = json.dumps({
            "task_id": "catalog-env-noise", "answer_label": "Да", "type": "single_rating",
            "listen_ms": 0, "timestamp": "x", "hidden": {}, "correct_answer": None,
            "matches_expected": None,
        }, ensure_ascii=False)
        (d / "answers.jsonl").write_text(old_line + "\n", encoding="utf-8")
        answers = server.load_answers("legacy")
        self.assertEqual(answers["catalog-env-noise"].get("note", ""), "")

        # Adding a note on top of a legacy (note-less) record works and is
        # correctly recognized as a note-only edit (answer unchanged).
        server.append_answer("legacy", self._base_record(
            task_id="catalog-env-noise", answer_label="Да", note="шёпот звучит как тихий звук"))
        latest = server.load_answers("legacy")["catalog-env-noise"]
        self.assertEqual(latest["note"], "шёпот звучит как тихий звук")
        self.assertFalse(latest["is_correction"])
        self.assertFalse(latest["answered_after_reveal"])


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

    def test_note_round_trips_and_note_only_edit_stays_in_blind_bucket(self):
        task_id = "final-boundary-complete"
        status, ans1 = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "answer_label": "Утверждение (точка)", "listen_ms": 400,
            "note": "шёпот звучит как тихий звук",
        })
        self.assertEqual(status, 200)

        # Navigate away and back: the note must come back with the answer.
        status, detail = self._get(f"/api/sets/{self.set_id}/task/{task_id}")
        self.assertEqual(status, 200)
        self.assertEqual(detail["previous_answer"]["note"], "шёпот звучит как тихий звук")
        self.assertFalse(detail["previous_answer"]["answered_after_reveal"])

        # Edit the note only (same answer_label), as the "Сохранить заметку"
        # button does client-side.
        status, ans2 = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "answer_label": "Утверждение (точка)", "listen_ms": 0,
            "note": "шёпот звучит как тихий звук; фон гуляет",
        })
        self.assertEqual(status, 200)

        latest = server.load_answers(self.set_id)[task_id]
        self.assertEqual(latest["note"], "шёпот звучит как тихий звук; фон гуляет")
        self.assertEqual(latest["revision"], 2)
        self.assertFalse(latest["is_correction"], "editing only the note must not register as a correction")
        self.assertFalse(latest["answered_after_reveal"], "editing only the note must not leave the blind bucket")

        history = server.load_answer_history(self.set_id)
        task_history = [r for r in history if r["task_id"] == task_id]
        self.assertEqual(len(task_history), 2, "the original note-bearing answer must survive on disk")
        self.assertEqual(task_history[0]["note"], "шёпот звучит как тихий звук")

        # The summary's blind bucket must still include this task (note edit
        # is not a correction), unlike the answered_after_reveal task above.
        status, summary = self._get(f"/api/sets/{self.set_id}/summary")
        self.assertEqual(status, 200)
        graded_task_ids = {r["task_id"] for r in summary["answers"]
                            if r["task_id"] == task_id and not r.get("answered_after_reveal")}
        self.assertEqual(graded_task_ids, {task_id})


class TestDifferQuestionReformulated(unittest.TestCase):
    """Owner feedback #3 (issue #57 follow-up): "не совсем понятно что значит
    звучат ли они одинаково, голоса, я так понял совсем разные" -- the old
    differ-task wording mentioned "тон голоса" and offered "звучат
    одинаково" as an option, inviting exactly that (unanswerable, since
    voices are never pinned) reading. The new wording must not repeat it,
    and must stay internally consistent (options include the literal
    correct_answer string)."""

    def test_differ_question_does_not_mention_voice_tone_identity(self):
        self.assertNotIn("тон голоса", catalog.DIFFER_QUESTION)
        self.assertNotIn("звучат одинаково", catalog.DIFFER_QUESTION)
        for opt in catalog.DIFFER_OPTIONS:
            self.assertNotIn("звучат одинаково", opt)

    def test_differ_question_is_about_delivery_not_voice_identity(self):
        self.assertIn("одач", catalog.DIFFER_QUESTION.lower())  # "подача" (delivery)

    def test_differ_task_correct_answer_is_one_of_its_own_options(self):
        task = catalog._differ_task(
            "t1", REPO_ROOT / "output/m4t0_sadness.wav", REPO_ROOT / "output/m4t0_neutral.wav",
            "emotion:sadness", {},
        )
        self.assertIn(task["hidden"]["correct_answer"], task["options"])

    @unittest.skipUnless(OUTPUT_PRESENT, "output/m4_boundary_check/ audio not present in this checkout")
    def test_boundary_check_task_correct_answer_is_one_of_its_own_options(self):
        doc = catalog.build_boundary_check_set()
        if doc is None:
            self.skipTest("output/m4_boundary_check/ clips not present")
        task = doc["tasks"][0]
        self.assertIn(task["hidden"]["correct_answer"], task["options"])
        self.assertNotIn("тон голоса", task["question"])


class TestEmotionMatchedTextBuilder(unittest.TestCase):
    """Requirement 2 (issue #57 follow-up, owner feedback #2): the
    emotion-matched-text set is scaffolding only -- no audio has been
    generated for it yet (see docs/research/audiobook/
    m4-emotion-matched-texts.md), so this test proves the schema/build logic
    against a synthetic fixture (tiny tone WAVs, not real speech) rather than
    against real generated audio that doesn't exist."""

    MANIFEST_DIR = REPO_ROOT / "output" / "m4_emotion_matched_text"

    def tearDown(self):
        # Clean up the real (gitignored) output/ directory this test creates.
        if self.MANIFEST_DIR.is_dir():
            for f in self.MANIFEST_DIR.glob("*"):
                f.unlink()
            self.MANIFEST_DIR.rmdir()

    def test_returns_none_when_directory_absent(self):
        self.assertFalse(self.MANIFEST_DIR.is_dir())
        self.assertIsNone(catalog.build_emotion_matched_text_set())

    def test_returns_none_when_manifest_missing(self):
        self.MANIFEST_DIR.mkdir(parents=True)
        self.assertIsNone(catalog.build_emotion_matched_text_set())

    def test_builds_tasks_from_synthetic_fixture_and_marks_composed_text(self):
        self.MANIFEST_DIR.mkdir(parents=True)
        tagged = self.MANIFEST_DIR / "sadness_tagged.wav"
        plain = self.MANIFEST_DIR / "sadness_plain.wav"
        _write_tone_wav(tagged, 120, dur_s=0.2)
        _write_tone_wav(plain, 120, dur_s=0.2)
        manifest = [{
            "emotion": "sadness",
            "text": "Поэтому он был очень опечален.",
            "source": "chapter-e0-narration.txt §3 (sb-1-19)",
            "tagged_clip": "output/m4_emotion_matched_text/sadness_tagged.wav",
            "plain_clip": "output/m4_emotion_matched_text/sadness_plain.wav",
        }, {
            "emotion": "surprise",
            "text": "Он замер на месте — такого поворота событий никто не ожидал.",
            "source": "[СОЧИНЕНО]",
            "tagged_clip": "output/m4_emotion_matched_text/sadness_tagged.wav",  # reuse fixture file
            "plain_clip": "output/m4_emotion_matched_text/sadness_plain.wav",
        }]
        (self.MANIFEST_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        doc = catalog.build_emotion_matched_text_set()
        self.assertIsNotNone(doc)
        self.assertEqual(doc["id"], "emotion_matched_text")
        self.assertEqual(len(doc["tasks"]), 2)
        by_id = {t["id"]: t for t in doc["tasks"]}

        sadness_task = by_id["emo-matched-sadness"]
        self.assertFalse(sadness_task["hidden"]["text_composed"])
        self.assertIn("sb-1-19", sadness_task["hidden"]["text_source"])
        self.assertEqual(sadness_task["hidden"]["matched_text"], "Поэтому он был очень опечален.")

        surprise_task = by_id["emo-matched-surprise"]
        self.assertTrue(surprise_task["hidden"]["text_composed"])
        self.assertEqual(surprise_task["hidden"]["text_source"], "[СОЧИНЕНО]")

        # Correct_answer must be a real option (same contract as the other
        # differ-style tasks -- compute_matches_expected() relies on this).
        for task in doc["tasks"]:
            self.assertIn(task["hidden"]["correct_answer"], task["options"])

    def test_entries_with_missing_clip_files_are_skipped_not_crashed(self):
        self.MANIFEST_DIR.mkdir(parents=True)
        manifest = [{
            "emotion": "fear", "text": "...", "source": "[СОЧИНЕНО]",
            "tagged_clip": "output/m4_emotion_matched_text/does_not_exist_tagged.wav",
            "plain_clip": "output/m4_emotion_matched_text/does_not_exist_plain.wav",
        }]
        (self.MANIFEST_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.assertIsNone(catalog.build_emotion_matched_text_set())


@unittest.skipUnless(NUMPY_PRESENT, "numpy not available in this interpreter (pitch.py needs it via "
                                      "docs/research/audiobook/m4_prosody_metrics.py)")
class TestPitchModule(unittest.TestCase):
    """Requirement 1 (issue #57 follow-up, owner feedback #1): pitch-aware
    pairing. Uses synthetic pure-tone WAVs (known F0 by construction) so
    these tests don't depend on real generated speech."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.low1 = cls.tmp_dir / "low1.wav"
        cls.low2 = cls.tmp_dir / "low2.wav"
        cls.high1 = cls.tmp_dir / "high1.wav"
        cls.high2 = cls.tmp_dir / "high2.wav"
        _write_tone_wav(cls.low1, 110)
        _write_tone_wav(cls.low2, 120)
        _write_tone_wav(cls.high1, 220)
        _write_tone_wav(cls.high2, 230)

    def test_median_f0_hz_recovers_synthetic_tone_frequency(self):
        f0 = pitch.median_f0_hz(self.low1)
        self.assertIsNotNone(f0)
        self.assertAlmostEqual(f0, 110, delta=5)

    def test_semitone_diff_is_symmetric_and_zero_for_equal_pitch(self):
        self.assertEqual(pitch.semitone_diff(150.0, 150.0), 0.0)
        self.assertAlmostEqual(pitch.semitone_diff(110, 220), 12.0, delta=0.5)  # one octave
        self.assertAlmostEqual(pitch.semitone_diff(220, 110), pitch.semitone_diff(110, 220))

    def test_otsu_threshold_none_for_fewer_than_two_distinct_values(self):
        self.assertIsNone(pitch.otsu_threshold([]))
        self.assertIsNone(pitch.otsu_threshold([140.0]))
        self.assertIsNone(pitch.otsu_threshold([140.0, 140.0]))

    def test_otsu_threshold_splits_two_synthetic_clusters(self):
        values = [111.9, 122.1, 222.2, 231.9]  # measured medians, see below
        th = pitch.otsu_threshold(values)
        self.assertGreater(th, 122.1)
        self.assertLess(th, 222.2)

    def test_build_threshold_report_falls_back_for_tiny_corpus(self):
        report = pitch.build_threshold_report([140.0, 145.0])
        self.assertEqual(report["method"].startswith("fallback"), True)
        self.assertEqual(report["threshold_hz"], pitch._FALLBACK_THRESHOLD_HZ)

    def test_build_threshold_report_otsu_for_real_corpus(self):
        index = pitch.build_f0_index([self.low1, self.low2, self.high1, self.high2])
        report = pitch.build_threshold_report(list(index.values()))
        self.assertEqual(report["n"], 4)
        self.assertTrue(report["method"].startswith("otsu"))
        self.assertEqual(report["low_cluster_n"], 2)
        self.assertEqual(report["high_cluster_n"], 2)

    def test_pitch_gate_same_cluster_true_cross_cluster_false(self):
        f0_low1 = pitch.median_f0_hz(self.low1)
        f0_low2 = pitch.median_f0_hz(self.low2)
        f0_high1 = pitch.median_f0_hz(self.high1)
        threshold = 170.0  # between the low (~110-120) and high (~220-230) clusters
        self.assertTrue(pitch.pitch_gate(f0_low1, f0_low2, threshold))
        self.assertFalse(pitch.pitch_gate(f0_low1, f0_high1, threshold))

    def test_pitch_gate_unknown_pitch_is_never_comparable(self):
        self.assertFalse(pitch.pitch_gate(None, 150.0, 170.0))
        self.assertFalse(pitch.pitch_gate(150.0, None, 170.0))
        self.assertFalse(pitch.pitch_gate(None, None, 170.0))

    def test_annotate_pitch_warnings_flags_cross_cluster_pair_not_same_cluster(self):
        close_task = {
            "id": "close", "type": "pair_compare", "answer_kind": "which",
            "clips": {"A": str(self.low1), "B": str(self.low2)},
        }
        far_task = {
            "id": "far", "type": "pair_compare", "answer_kind": "which",
            "clips": {"A": str(self.low1), "B": str(self.high1)},
        }
        single_task = {  # not a comparison type -- must be left untouched
            "id": "solo", "type": "single_rating", "clips": {"A": str(self.low1)},
        }
        docs = [{"id": "synthetic", "tasks": [close_task, far_task, single_task]}]
        pitch.annotate_pitch_warnings(docs)
        self.assertIsNone(close_task["pitch_warning"])
        self.assertIsNotNone(far_task["pitch_warning"])
        self.assertEqual(far_task["pitch_warning"]["pairs"][0]["a"], "A")
        self.assertEqual(far_task["pitch_warning"]["pairs"][0]["b"], "B")
        self.assertNotIn("pitch_warning", single_task)


@unittest.skipUnless(OUTPUT_PRESENT and NUMPY_PRESENT,
                      "needs both real output/ audio fixtures and numpy")
class TestPitchWarningSummaryIntegration(unittest.TestCase):
    """A pitch-mismatched pair must stay visible/answerable but be excluded
    from the graded/differ_pairs pass-fail gate in the summary (owner
    feedback #1) -- mirrors how answered_after_reveal is already excluded.
    Forces pitch_warning on a real (already-loaded) task rather than relying
    on which real clips happen to be far apart in pitch, so this test does
    not get flaky if the underlying audio or threshold changes."""

    @classmethod
    def setUpClass(cls):
        cls._orig_results_dir = server.RESULTS_DIR
        cls.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = cls.tmp_dir
        cls.httpd = server.ThreadingHTTPServer((server.HOST, 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.set_id = "final_intonation"
        assert cls.set_id in server.TASK_SETS
        cls.task_id = "final-boundary-continuing"
        task = server.TASK_SETS[cls.set_id].task(cls.task_id)
        assert task is not None
        task["pitch_warning"] = {
            "reason": "test-forced mismatch", "threshold_hz": 165.0,
            "pairs": [{"a": "A", "b": "B", "f0_a_hz": 90.0, "f0_b_hz": 210.0, "semitone_diff": 14.7}],
        }

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        server.RESULTS_DIR = cls._orig_results_dir
        task = server.TASK_SETS[cls.set_id].task(cls.task_id)
        if task is not None:
            task["pitch_warning"] = None

    def _get(self, path):
        conn = http.client.HTTPConnection(server.HOST, self.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def _post_json(self, path, obj):
        conn = http.client.HTTPConnection(server.HOST, self.port, timeout=5)
        try:
            data = json.dumps(obj).encode("utf-8")
            conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def test_task_view_exposes_pitch_warning_before_answering(self):
        status, detail = self._get(f"/api/sets/{self.set_id}/task/{self.task_id}")
        self.assertEqual(status, 200)
        self.assertIsNotNone(detail["task"]["pitch_warning"])

    def test_pitch_mismatched_answer_excluded_from_graded_and_differ_buckets(self):
        status, ans = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": self.task_id, "answer_label": "Утверждение (точка)", "listen_ms": 100,
        })
        self.assertEqual(status, 200)
        self.assertEqual(server.load_answers(self.set_id)[self.task_id]["pitch_warning"]["threshold_hz"], 165.0)

        status, summary = self._get(f"/api/sets/{self.set_id}/summary")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(summary["pitch_unreliable_total"], 1)
        graded_ids = {r["task_id"] for r in summary["answers"]
                      if r["task_id"] == self.task_id and not r.get("pitch_warning")
                      and r.get("correct_answer") is not None}
        self.assertEqual(graded_ids, set(), "a pitch-mismatched task must not count toward graded_total")


class TestVoiceCastingBuilder(unittest.TestCase):
    """Requirement (issue #57/#118 follow-up, owner: "давай добавим тогда в
    инструмент и отбор голосов из этих 70 сегментов"): voice casting is not
    a blind task -- gender/age/name, no ground truth."""

    def test_returns_none_when_manifest_absent(self):
        orig_root = catalog.REPO_ROOT
        catalog.REPO_ROOT = Path(tempfile.mkdtemp())
        try:
            self.assertIsNone(catalog.build_voice_casting_set())
        finally:
            catalog.REPO_ROOT = orig_root

    @unittest.skipUnless(CHAPTER114E0_PRESENT, "output/chapter-114-e0/manifest.json not present in this checkout")
    def test_builds_one_task_per_segment_with_transcript_and_no_ground_truth(self):
        doc = catalog.build_voice_casting_set()
        self.assertIsNotNone(doc)
        self.assertEqual(doc["id"], "voice_casting_chapter114e0")
        self.assertEqual(len(doc["tasks"]), 70)
        for task in doc["tasks"]:
            self.assertEqual(task["type"], "voice_casting")
            self.assertEqual(task["answer_kind"], "voice_cast")
            self.assertNotIn("correct_answer", task["hidden"])
            self.assertTrue(task["hidden"]["segment_text"])
            self.assertTrue(Path(REPO_ROOT / task["clips"]["A"]).is_file())
        ids = [t["id"] for t in doc["tasks"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate task ids")


class TestVoiceCastingBackwardCompat(unittest.TestCase):
    """Owner: "нужно не потерять то что уже помелили" -- the owner already
    fully cast all 70 chapter-114-e0 segments (78 answers.jsonl lines) under
    the OLD schema, with no pleasantness/room_feel/measured_features keys.
    Adding those fields must not break reading, grading, or editing that
    real data."""

    def setUp(self):
        self._orig_results_dir = server.RESULTS_DIR
        self.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = self.tmp_dir

    def tearDown(self):
        server.RESULTS_DIR = self._orig_results_dir

    def _old_shaped_record(self, task_id="voice-cast-00", name=None, selected=False):
        # Exact shape of a real pre-this-change line (no pleasantness,
        # room_feel, or measured_features keys at all).
        return {
            "task_id": task_id, "set_id": "voice_casting_chapter114e0",
            "type": "voice_casting", "answer_kind": "voice_cast",
            "question": "Сегмент 1 из 70...",
            "answer_label": f"male/middle" + (f" → «{name}»" if selected else ""),
            "answer_role": None, "gender": "male", "age_bucket": "middle",
            "selected": selected, "name": name, "measured_f0_hz": 163.3,
            "note": "", "listen_ms": 26465, "timestamp": "2026-08-25T17:18:05+0300",
            "hidden": {"segment_index": 0, "output_path": "segment_x.wav",
                       "segment_text": "...", "manifest_speaker": "narrator"},
            "correct_answer": None, "matches_expected": None, "skipped_prior": False,
            "pitch_warning": None,
        }

    def test_old_shaped_record_reads_without_error(self):
        server.append_answer("voice_casting_chapter114e0", self._old_shaped_record())
        answers = server.load_answers("voice_casting_chapter114e0")
        rec = answers["voice-cast-00"]
        self.assertIsNone(rec.get("pleasantness"))
        self.assertIsNone(rec.get("room_feel"))
        self.assertIsNone(rec.get("measured_features"))

    def test_old_named_selected_record_is_still_selected_and_in_the_roster(self):
        server.append_answer("voice_casting_chapter114e0",
                              self._old_shaped_record(task_id="voice-cast-02", name="чтец", selected=True))
        answers = server.load_answers("voice_casting_chapter114e0")
        rec = answers["voice-cast-02"]
        self.assertTrue(rec["selected"])
        self.assertEqual(rec["name"], "чтец")

    def test_appending_pleasantness_only_on_top_of_an_old_record_preserves_gender_age_name(self):
        server.append_answer("voice_casting_chapter114e0",
                              self._old_shaped_record(task_id="voice-cast-02", name="чтец", selected=True))
        # Owner comes back later, adds only a pleasantness rating, resending
        # the same gender/age/name from the old record (client always sends
        # the full form -- see submitVoiceCast()/renderVoiceCastForm()
        # prefill in app.js) plus the new field.
        new_rec = self._old_shaped_record(task_id="voice-cast-02", name="чтец", selected=True)
        new_rec["pleasantness"] = "5"
        server.append_answer("voice_casting_chapter114e0", new_rec)

        latest = server.load_answers("voice_casting_chapter114e0")["voice-cast-02"]
        self.assertEqual(latest["name"], "чтец")
        self.assertEqual(latest["gender"], "male")
        self.assertEqual(latest["pleasantness"], "5")

        history = server.load_answer_history("voice_casting_chapter114e0")
        task_history = [r for r in history if r["task_id"] == "voice-cast-02"]
        self.assertEqual(len(task_history), 2, "the original old-shaped answer must survive on disk")


@unittest.skipUnless(CHAPTER114E0_PRESENT and NUMPY_PRESENT,
                      "needs both output/chapter-114-e0/ and numpy")
class TestVoiceCastingOverHTTP(unittest.TestCase):
    """End-to-end: casting a segment round-trips gender/age/name/selected,
    is excluded from every blind-gate statistic, and a real HTTP client
    sees the transcript and measured F0 up front (not blind)."""

    @classmethod
    def setUpClass(cls):
        cls._orig_results_dir = server.RESULTS_DIR
        cls.tmp_dir = Path(tempfile.mkdtemp())
        server.RESULTS_DIR = cls.tmp_dir
        cls.httpd = server.ThreadingHTTPServer((server.HOST, 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.set_id = "voice_casting_chapter114e0"
        assert cls.set_id in server.TASK_SETS, "voice_casting set failed to load"

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
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def _post_json(self, path, obj):
        conn = http.client.HTTPConnection(server.HOST, self.port, timeout=5)
        try:
            data = json.dumps(obj).encode("utf-8")
            conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def test_task_view_is_not_blind(self):
        status, detail = self._get(f"/api/sets/{self.set_id}/task/voice-cast-00")
        self.assertEqual(status, 200)
        task = detail["task"]
        self.assertEqual(task["response_mode"], "voice_cast")
        self.assertTrue(task["transcript"])  # shown up front, no reveal step

    def test_invalid_gender_rejected(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-01", "gender": "robot", "age_bucket": "middle",
            "selected": False, "listen_ms": 0,
        })
        self.assertEqual(status, 400)

    def test_empty_name_is_valid_and_means_not_selected(self):
        # Issue #57/#118 follow-up: no separate "selected" checkbox anymore
        # -- a blank name is a perfectly valid answer, it just isn't a cast
        # decision. Must NOT be rejected.
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-01", "gender": "male", "age_bucket": "middle",
            "name": "", "listen_ms": 0,
        })
        self.assertEqual(status, 200)
        rec = server.load_answers(self.set_id)["voice-cast-01"]
        self.assertFalse(rec["selected"])
        self.assertIsNone(rec["name"])

    def test_typed_name_alone_implies_selected_no_checkbox_needed(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-04", "gender": "male", "age_bucket": "young",
            "name": "arjuna", "listen_ms": 0,
        })
        self.assertEqual(status, 200)
        rec = server.load_answers(self.set_id)["voice-cast-04"]
        self.assertTrue(rec["selected"])
        self.assertEqual(rec["name"], "arjuna")

    def test_pleasantness_and_room_feel_are_optional(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-05", "gender": "female", "age_bucket": "old",
            "name": "", "listen_ms": 0,
        })
        self.assertEqual(status, 200)
        rec = server.load_answers(self.set_id)["voice-cast-05"]
        self.assertIsNone(rec["pleasantness"])
        self.assertIsNone(rec["room_feel"])

    def test_invalid_pleasantness_rejected(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-06", "gender": "male", "age_bucket": "young",
            "name": "", "pleasantness": "11", "listen_ms": 0,
        })
        self.assertEqual(status, 400)

    def test_invalid_room_feel_rejected(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-07", "gender": "male", "age_bucket": "young",
            "name": "", "room_feel": "cathedral", "listen_ms": 0,
        })
        self.assertEqual(status, 400)

    def test_pleasantness_scale_round_trips_and_is_shown_in_answer_label(self):
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-08", "gender": "female", "age_bucket": "middle",
            "name": "", "pleasantness": "5", "room_feel": "dry", "listen_ms": 0,
        })
        self.assertEqual(status, 200)
        rec = server.load_answers(self.set_id)["voice-cast-08"]
        self.assertEqual(rec["pleasantness"], "5")
        self.assertEqual(rec["room_feel"], "dry")
        self.assertIn("5", rec["answer_label"])

    def test_adding_pleasantness_to_an_already_cast_segment_does_not_require_redoing_it(self):
        """Owner: "нужно не потерять то что уже помелили" -- filling in
        gender/age/name once, then coming back later to add ONLY the new
        pleasantness field, must work without resupplying anything else."""
        task_id = "voice-cast-09"
        self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "gender": "male", "age_bucket": "old",
            "name": "narrator2", "listen_ms": 0,
        })
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "gender": "male", "age_bucket": "old",
            "name": "narrator2", "pleasantness": "4", "listen_ms": 0,
        })
        self.assertEqual(status, 200)
        rec = server.load_answers(self.set_id)[task_id]
        self.assertEqual(rec["name"], "narrator2")  # unchanged, not lost
        self.assertEqual(rec["pleasantness"], "4")  # newly added

    def test_measured_features_are_exposed_in_task_view_and_baked_into_the_record(self):
        status, detail = self._get(f"/api/sets/{self.set_id}/task/voice-cast-10")
        self.assertEqual(status, 200)
        # None only if numpy genuinely unavailable in this interpreter --
        # this test class already requires NUMPY_PRESENT.
        self.assertIsNotNone(detail["task"]["measured_features"])
        self.assertIn("reverb_tail_ms", detail["task"]["measured_features"])

        self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-10", "gender": "male", "age_bucket": "middle",
            "name": "", "listen_ms": 0,
        })
        rec = server.load_answers(self.set_id)["voice-cast-10"]
        self.assertIsNotNone(rec["measured_features"])

    def test_cast_roster_lists_named_voices_in_task_list(self):
        self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-11", "gender": "female", "age_bucket": "young",
            "name": "sita2", "listen_ms": 0,
        })
        status, tasks = self._get(f"/api/sets/{self.set_id}/tasks")
        self.assertEqual(status, 200)
        row = next(t for t in tasks["tasks"] if t["id"] == "voice-cast-11")
        self.assertEqual(row["name"], "sita2")

    def test_cast_and_name_round_trips_and_is_machine_readable(self):
        status, ans = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": "voice-cast-02", "gender": "female", "age_bucket": "young",
            "selected": True, "name": "sita", "note": "звонкий, чёткий", "listen_ms": 5000,
        })
        self.assertEqual(status, 200)

        rec = server.load_answers(self.set_id)["voice-cast-02"]
        self.assertEqual(rec["gender"], "female")
        self.assertEqual(rec["age_bucket"], "young")
        self.assertTrue(rec["selected"])
        self.assertEqual(rec["name"], "sita")
        self.assertEqual(rec["note"], "звонкий, чёткий")
        # This IS the casting result the engine reads -- output_path/segment_text
        # must be present and machine-usable as register_voice() inputs.
        self.assertTrue(rec["hidden"]["output_path"])
        self.assertTrue(rec["hidden"]["segment_text"])
        self.assertIsNone(rec["correct_answer"])
        self.assertIsNone(rec["matches_expected"])

        status, detail = self._get(f"/api/sets/{self.set_id}/task/voice-cast-02")
        self.assertEqual(status, 200)
        self.assertEqual(detail["previous_answer"]["name"], "sita")

    def test_recast_does_not_pollute_blind_gate_stats(self):
        task_id = "voice-cast-03"
        self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "gender": "male", "age_bucket": "old",
            "selected": False, "listen_ms": 0,
        })
        # Owner changes their mind on a re-listen.
        status, _ = self._post_json(f"/api/sets/{self.set_id}/answer", {
            "task_id": task_id, "gender": "male", "age_bucket": "middle",
            "selected": True, "name": "narrator", "listen_ms": 0,
        })
        self.assertEqual(status, 200)

        status, summary = self._get(f"/api/sets/{self.set_id}/summary")
        self.assertEqual(status, 200)
        # Voice casting must never appear in the blind-gate buckets, cast or
        # not, corrected or not.
        self.assertEqual(summary["graded_total"], 0)
        self.assertEqual(summary["differ_pairs_total"], 0)
        self.assertEqual(summary["pitch_unreliable_total"], 0)
        self.assertEqual(summary["answered_after_reveal"], 0,
                          "a voice-casting recast must never be counted as a blind-gate correction")
        self.assertGreaterEqual(summary["cast_total"], 1)
        self.assertGreaterEqual(summary["cast_selected_total"], 1)
        # The task's own revision history is still honestly tracked, just
        # not folded into the blind "answered_after_reveal" bucket meaning.
        latest = server.load_answers(self.set_id)[task_id]
        self.assertTrue(latest["is_correction"])


if __name__ == "__main__":
    unittest.main()
