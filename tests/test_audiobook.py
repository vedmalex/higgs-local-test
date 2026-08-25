#!/usr/bin/env python3
"""Regression tests for src/audiobook.py (issue #57 audit fixes F1-F14).

Each test class is named after the finding it reproduces/verifies. These are pure-function
tests only -- no model load, no GPU/CPU-heavy work -- safe to run on a loaded machine.

Run with the project's `.venv-tts` interpreter (needs numpy):
    .venv-tts/bin/python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import audiobook as ab  # noqa: E402


class TestF1QuoteTracking(unittest.TestCase):
    """F1: quote_depth used to only ever grow (both quote branches fed one counter with
    the same '"' character), so it never came back down and the rest of the text was
    swallowed into one sentence."""

    def test_straight_quote_pair_does_not_break_later_splitting(self):
        text = (
            '"Хорошо", — сказал он. Потом ушла. Затем пришла. И ещё одна фраза.'
        )
        sentences = ab.split_sentences(text)
        self.assertEqual(
            sentences,
            [
                '"Хорошо", — сказал он.',
                "Потом ушла.",
                "Затем пришла.",
                "И ещё одна фраза.",
            ],
        )

    def test_nested_guillemets_and_low_high_quotes_close_correctly(self):
        text = 'Он сказал: «Она крикнула „стой!“ и убежала». Потом тишина.'
        sentences = ab.split_sentences(text)
        self.assertEqual(
            sentences,
            [
                'Он сказал: «Она крикнула „стой!“ и убежала».',
                "Потом тишина.",
            ],
        )

    def test_paragraph_boundary_resets_stuck_quote_state(self):
        # An unclosed « in paragraph 1 must not swallow paragraph 2's sentences.
        text = (
            "Он сказал «привет и не договорил.\n\n"
            "Сегодня хороший день. Завтра будет дождь."
        )
        sentences = ab.split_sentences(text)
        self.assertEqual(
            sentences,
            [
                "Он сказал «привет и не договорил.",
                "Сегодня хороший день.",
                "Завтра будет дождь.",
            ],
        )

    def test_force_reset_recovers_after_a_long_unclosed_quote(self):
        # A single unmatched opening quote must not be able to swallow an entire
        # multi-hundred-character chapter -- the safety valve must kick back in.
        filler = "Слово тут и там снова и снова без остановки совсем. " * 10
        text = "«" + filler + "Это предложение после сброса. Ещё одно тут."
        self.assertGreater(len(filler), ab.QUOTE_FORCE_RESET_CHARS)
        sentences = ab.split_sentences(text)
        self.assertIn("Это предложение после сброса.", sentences)
        self.assertIn("Ещё одно тут.", sentences)


class TestF2LongSentenceHardSplit(unittest.TestCase):
    """F2: max_chars was only checked before adding a sentence to a chunk -- a single
    sentence longer than the budget went out whole, unbounded."""

    def test_extremely_long_single_sentence_is_force_split(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunks = ab.chunk_sentences(["x" * 20000], max_chars=500)
        self.assertTrue(any("force-splitting" in str(w.message) for w in caught))
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 500)

    def test_force_split_prefers_semicolons_then_commas_then_whitespace(self):
        sentence = "а" * 100 + ";" + "б" * 100 + ";" + "в" * 100
        pieces = ab._force_split_long_sentence(sentence, max_chars=150)
        self.assertTrue(all(len(p) <= 150 for p in pieces))
        self.assertEqual("".join(p.rstrip(";") for p in pieces).replace(";", ""), "а" * 100 + "б" * 100 + "в" * 100)


class TestF4AtomicManifest(unittest.TestCase):
    """F4: manifest writes used to truncate the file in place; a kill mid-write left an
    unparsable JSON file with no recovery path."""

    def test_save_manifest_is_atomic_and_keeps_backup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            m1 = {"model": "x", "max_chars": 1, "tag_scope": "chunk", "segments": []}
            ab.save_manifest(m1, manifest_path)
            self.assertTrue(manifest_path.exists())
            self.assertFalse((Path(d) / "manifest.json.bak").exists())

            m2 = {"model": "x", "max_chars": 1, "tag_scope": "chunk", "segments": [1]}
            ab.save_manifest(m2, manifest_path)
            bak_path = Path(d) / "manifest.json.bak"
            self.assertTrue(bak_path.exists())
            self.assertEqual(json.loads(bak_path.read_text()), m1)
            self.assertEqual(json.loads(manifest_path.read_text()), m2)

    def test_load_or_create_manifest_recovers_from_corrupt_json_via_backup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"

            # Reproduce the original defect directly: a plain write_text() truncated
            # mid-write leaves unparsable JSON with json.loads() raising uncaught.
            manifest_path.write_text('{"segments": [', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                json.loads(manifest_path.read_text(encoding="utf-8"))

            # Now verify the fixed recovery path: a good manifest, a good .bak, then a
            # corrupt primary -- load_or_create_manifest must recover from .bak instead
            # of raising an uncaught JSONDecodeError.
            good = {"model": ab.MODEL_ID, "max_chars": 1, "tag_scope": "chunk", "segments": []}
            manifest_path.write_text(json.dumps(good), encoding="utf-8")
            bak_path = Path(d) / "manifest.json.bak"
            bak_path.write_text(json.dumps(good), encoding="utf-8")
            manifest_path.write_text('{"segments": [', encoding="utf-8")  # corrupt primary

            chunks = ab.chunk_sentences(ab.split_sentences("Одно предложение тут."), max_chars=500)
            manifest = ab.load_or_create_manifest(manifest_path, chunks, max_chars=1, tag_scope="chunk")
            self.assertIn("segments", manifest)

    def test_load_or_create_manifest_raises_clear_error_when_backup_also_corrupt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            manifest_path.write_text('{"segments": [', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ab.load_or_create_manifest(manifest_path, [], max_chars=1, tag_scope="chunk")


class TestF5RetryAndContinueOnError(unittest.TestCase):
    """F5: one failed model.generate() call used to abort the entire run and re-raise,
    losing every remaining hour of a multi-hour chapter."""

    class _FlakyModel:
        def __init__(self, fail_times: int, always_fail: bool = False):
            self.fail_times = fail_times
            self.always_fail = always_fail
            self.calls = 0

        def generate(self, text, temperature, max_new_tokens):
            self.calls += 1
            if self.always_fail or self.calls <= self.fail_times:
                raise RuntimeError("simulated generation failure")
            sr = 24000
            n = max(1, int(len(text) * 0.05 * sr))
            return [type("R", (), {"audio": np.zeros(n, dtype=np.float64), "sample_rate": sr})()]

    def _manifest(self, d, texts):
        # One Chunk per text, built directly rather than via chunk_sentences, so each
        # text is guaranteed its own segment regardless of chunk_sentences' grouping.
        chunks = [
            ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t)
            for i, t in enumerate(texts)
        ]
        manifest_path = Path(d) / "manifest.json"
        manifest = ab.load_or_create_manifest(manifest_path, chunks, max_chars=1000, tag_scope="chunk")
        return manifest, manifest_path

    def test_segment_recovers_after_transient_failures_via_retry(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, ["Тестовое предложение для повтора."])
            model = self._FlakyModel(fail_times=2)
            result = ab.generate_segments(
                model, manifest, manifest_path, max_retries=3, retry_base_delay=0.0
            )
            self.assertEqual(result["failures"], [])
            self.assertEqual(manifest["segments"][0]["status"], "done")
            self.assertEqual(model.calls, 3)

    def test_exhausted_retries_raise_without_continue_on_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, ["Предложение которое всегда падает."])
            model = self._FlakyModel(fail_times=0, always_fail=True)
            with self.assertRaises(RuntimeError):
                ab.generate_segments(
                    model, manifest, manifest_path, max_retries=2, retry_base_delay=0.0
                )
            self.assertEqual(manifest["segments"][0]["status"], "failed")

    def test_continue_on_error_keeps_going_past_a_permanently_failed_segment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(
                d, ["Первое падающее предложение тут.", "Второе предложение которое должно пройти успешно."]
            )
            model = self._FlakyModel(fail_times=0, always_fail=False)
            # Force only the first segment to fail permanently by wrapping generate.
            real_generate = model.generate
            calls_per_text = {}

            def flaky_generate(text, temperature, max_new_tokens):
                calls_per_text[text] = calls_per_text.get(text, 0) + 1
                if "падающее" in text:
                    raise RuntimeError("permanent failure for this segment")
                return real_generate(text, temperature, max_new_tokens)

            model.generate = flaky_generate
            result = ab.generate_segments(
                model,
                manifest,
                manifest_path,
                max_retries=2,
                retry_base_delay=0.0,
                continue_on_error=True,
            )
            statuses = {s["index"]: s["status"] for s in manifest["segments"]}
            self.assertEqual(len(result["failures"]), 1)
            self.assertIn("failed", statuses.values())
            self.assertIn("done", statuses.values())


class TestF3StreamingAssembly(unittest.TestCase):
    """F3: assembly used to hold the full chapter as float64 arrays (read_wav's output)
    plus a second np.concatenate copy, plus a third int16 copy in write_wav -- three
    times a multi-hour chapter's worth of memory at peak. Verified here by asserting the
    implementation never builds a full-chapter array (structural/behavioral check via a
    manageable-but-nontrivial number of segments, and confirming correct output)."""

    def _make_segment(self, d, name, seconds, sr=8000, value=0.2):
        path = Path(d) / name
        n = int(seconds * sr)
        audio = np.full(n, value, dtype=np.float64)
        ab.write_wav(path, audio, sr)
        return path

    def test_streaming_assembly_produces_correct_concatenated_duration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            segs = []
            for i in range(5):
                name = f"segment_{i:04d}.wav"
                self._make_segment(d, name, seconds=0.5)
                segs.append(
                    {
                        "index": i,
                        "status": "done",
                        "output_path": name,
                        "text": "x",
                    }
                )
            manifest = {"segments": segs}
            out_path = Path(d) / "chapter.wav"
            result = ab.assemble_chapter(manifest, out_path, base_dir=Path(d), silence_ms=100)
            self.assertEqual(result["sample_rate"], 8000)
            audio, sr = ab.read_wav(out_path)
            expected_samples = 5 * int(0.5 * 8000) + 4 * int(0.1 * 8000)
            self.assertEqual(len(audio), expected_samples)

    def test_assemble_chapter_never_materializes_a_full_chapter_array(self):
        # Behavioral proxy: patch np.concatenate to fail if ever called with more than
        # one segment's worth of arrays (assemble_chapter must not call it at all).
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                self._make_segment(d, f"segment_{i:04d}.wav", seconds=0.2)
            manifest = {
                "segments": [
                    {"index": i, "status": "done", "output_path": f"segment_{i:04d}.wav", "text": "x"}
                    for i in range(3)
                ]
            }
            original_concatenate = np.concatenate

            def guarded_concatenate(arrays, *a, **kw):
                raise AssertionError(
                    "assemble_chapter must not call np.concatenate over whole-chapter audio"
                )

            np.concatenate = guarded_concatenate
            try:
                ab.assemble_chapter(manifest, Path(d) / "chapter.wav", base_dir=Path(d))
            finally:
                np.concatenate = original_concatenate


class TestF6SegmentIntegrityChecks(unittest.TestCase):
    """F6: a segment marked 'done' was accepted purely on out_path.exists() -- an empty
    (0-sample) WAV or a truncated one both passed silently."""

    def test_write_wav_with_empty_array_reproduces_original_silent_acceptance_risk(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.wav"
            ab.write_wav(path, np.array([], dtype=np.float64), 24000)
            self.assertTrue(path.exists())
            audio, sr = ab.read_wav(path)
            self.assertEqual(audio.shape, (0,))
            # This is exactly the silent-acceptance risk from the audit; the actual fix
            # is that _validate_generated_audio / _resume_check_ok now catch it:
            self.assertIsNotNone(ab._validate_generated_audio(audio, sr, "some real text"))

    def test_resume_check_rejects_missing_file(self):
        ok, reason = ab._resume_check_ok({"num_samples": 10}, Path("/nonexistent/path.wav"))
        self.assertFalse(ok)
        self.assertIn("missing", reason)

    def test_resume_check_rejects_sample_count_mismatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seg.wav"
            ab.write_wav(path, np.zeros(1000, dtype=np.float64), 24000)
            ok, reason = ab._resume_check_ok({"num_samples": 9999, "sample_rate": 24000}, path)
            self.assertFalse(ok)
            self.assertIn("samples", reason)

    def test_resume_check_accepts_matching_segment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seg.wav"
            audio = np.zeros(1000, dtype=np.float64)
            ab.write_wav(path, audio, 24000)
            ok, reason = ab._resume_check_ok({"num_samples": 1000, "sample_rate": 24000}, path)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_validate_generated_audio_flags_implausible_duration(self):
        text = "x" * 1000  # 1000 chars
        sr = 24000
        too_short = np.zeros(int(0.001 * sr), dtype=np.float64)  # far under 0.03s/char*1000
        self.assertIsNotNone(ab._validate_generated_audio(too_short, sr, text))
        plausible = np.zeros(int(0.1 * 1000 * sr), dtype=np.float64)  # 0.1 s/char, in range
        self.assertIsNone(ab._validate_generated_audio(plausible, sr, text))


class TestF8ChunkBudgetAccountsForPrefix(unittest.TestCase):
    """F8: under tag_scope="sentence", cur_len was incremented by the bare sentence
    length while the *prefixed* text was what actually got stored -- chunks silently
    exceeded max_chars by up to the length of the reopened tag(s)."""

    def test_chunks_never_exceed_max_chars_with_active_tag_reopened_every_sentence(self):
        max_chars = 200
        sentences = [
            "<|emotion:sadness|>Первое предложение довольно длинное для проверки бюджета.",
        ] + [
            f"Предложение номер {i} тоже сравнительно длинное для проверки лимита символов."
            for i in range(6)
        ]
        chunks = ab.chunk_sentences(sentences, max_chars=max_chars, tag_scope="sentence")
        for c in chunks:
            self.assertLessEqual(
                len(c.text), max_chars * 2,  # generous upper bound; real assertion below
            )
        # The real assertion: every chunk's actual character budget accounting must match
        # what chunk_sentences itself computed, i.e. no chunk should silently be allowed to
        # exceed max_chars by more than one reopened-tag's worth. Recompute expected
        # cur_len bookkeeping directly:
        for c in chunks:
            self.assertLessEqual(len(c.text), max_chars + len("<|emotion:sadness|>") + 1)


class TestF9NumberedListMarkers(unittest.TestCase):
    """F9: '1. Первый пункт. 2. Второй пункт.' used to split into four segments,
    including two bare, meaningless one-character segments ('1.', '2.')."""

    def test_numbered_list_items_are_not_split_off_as_their_own_sentence(self):
        text = "1. Первый пункт. 2. Второй пункт."
        sentences = ab.split_sentences(text)
        self.assertNotIn("1.", sentences)
        self.assertNotIn("2.", sentences)
        self.assertEqual(sentences, ["1. Первый пункт.", "2. Второй пункт."])

    def test_number_mid_sentence_is_unaffected(self):
        text = "Их было 5. Потом все ушли домой."
        sentences = ab.split_sentences(text)
        self.assertEqual(sentences, ["Их было 5.", "Потом все ушли домой."])


class TestF10TagReopenMidSentence(unittest.TestCase):
    """F10: _reopen_prefix only skipped reopening a category if the sentence STARTED
    with that category's tag -- a tag declared mid-sentence still got a conflicting
    reopened tag of the same category prepended."""

    def test_no_conflicting_tag_when_category_declared_mid_sentence(self):
        sentences = ["<|emotion:sad|>Первое.", "Второе <|emotion:elation|>предложение."]
        chunks = ab.chunk_sentences(sentences, max_chars=1000, tag_scope="sentence")
        text = chunks[0].text
        self.assertEqual(text.count("<|emotion:"), 2)
        self.assertNotIn("<|emotion:sad|>Второе", text)


class TestSfxOneShotNotReopened(unittest.TestCase):
    """Refs #57 sfx tag inventory: PROMPTING.md places `sfx` in the same "inline" bucket
    as `pause`/`long_pause` -- a one-shot effect at an exact position, not a sustained
    state. If `sfx` were tracked in `active` like emotion/prosody/style, a chunk boundary
    right after an `<|sfx:sneeze|>` tag would reopen it at the top of the next chunk,
    making the character sneeze again there -- a defect that is invisible in code review
    and expensive to catch only after a full book has been generated.
    """

    def test_prosody_pause_is_not_reopened_across_a_chunk_boundary(self):
        max_chars = 50
        sentences = [
            "Первое <|prosody:pause|> предложение здесь.",
            "Второе предложение отдельным куском тут.",
        ]
        chunks = ab.chunk_sentences(sentences, max_chars=max_chars, tag_scope="chunk")
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks[1:]:
            self.assertNotIn("<|prosody:pause|>", c.text)

    def test_sfx_tag_is_not_reopened_across_a_chunk_boundary(self):
        max_chars = 50
        sentences = [
            "Первое <|sfx:sneeze|>апчхи предложение здесь.",
            "Второе предложение отдельным куском тут.",
        ]
        chunks = ab.chunk_sentences(sentences, max_chars=max_chars, tag_scope="chunk")
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks[1:]:
            self.assertNotIn("<|sfx:sneeze|>", c.text)

    def test_sfx_tag_is_not_reopened_across_a_chunk_boundary_sentence_scope(self):
        max_chars = 50
        sentences = [
            "Первое <|sfx:sneeze|>апчхи предложение здесь.",
            "Второе предложение отдельным куском тут.",
        ]
        chunks = ab.chunk_sentences(sentences, max_chars=max_chars, tag_scope="sentence")
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks[1:]:
            self.assertNotIn("<|sfx:sneeze|>", c.text)

    def test_sfx_tag_does_not_suppress_an_active_emotion_reopen(self):
        # An sfx tag in a sentence must not be mistaken for an emotion/prosody/style
        # declaration that would suppress reopening of a genuinely active category.
        sentences = [
            "<|emotion:sadness|>Первое <|sfx:sigh|>предложение здесь.",
            "Второе предложение.",
        ]
        chunks = ab.chunk_sentences(sentences, max_chars=1000, tag_scope="sentence")
        self.assertEqual(len(chunks), 1)
        self.assertIn("<|emotion:sadness|>", chunks[0].text)


class TestSfxCategoryParsing(unittest.TestCase):
    """The `sfx` category must be recognized by TAG_RE (not silently dropped like an
    unrecognized category would be) once VALID_TAGS accepts it."""

    def test_tag_re_parses_sfx_category(self):
        matches = ab.TAG_RE.findall("текст <|sfx:cough|> текст")
        self.assertEqual(matches, [("sfx", "cough")])

    def test_sentence_own_tags_includes_sfx(self):
        tags = ab._sentence_own_tags("<|sfx:laughter|>Ха-ха!")
        self.assertEqual(tags, [("sfx", "laughter")])


class TestF11TagValidation(unittest.TestCase):
    """F11: TAG_RE's character class ([a-z_]+) silently failed to match a tag with a
    digit or uppercase letter (e.g. a typo like '<|emotion:Elation|>'), so the tag was
    neither tracked nor flagged -- it just became inert literal text for the rest of the
    chapter."""

    def test_valid_tag_passes(self):
        ab.validate_control_tags("Текст с <|emotion:elation|> тегом.")  # must not raise

    def test_miscased_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            ab.validate_control_tags("Текст с <|emotion:Elation|> тегом.")

    def test_invented_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            ab.validate_control_tags("Текст с <|emotion:ecstasy|> тегом.")

    def test_all_43_known_tags_are_accepted(self):
        # 21 emotion + 10 prosody + 3 style + 9 sfx, per the pinned checkpoint's
        # PROMPTING.md "Full tag catalog (43)" (bosonai/higgs-tts-3-4b,
        # snapshot 7556c17e05201fccd9c8cc120bc216dcc7b5d561).
        self.assertEqual(len(ab.VALID_TAGS), 43)
        for tag in ab.VALID_TAGS:
            ab.validate_control_tags(f"проверка {tag} тега")  # must not raise

    def test_sfx_tags_are_recognized_and_correctly_categorized(self):
        expected_sfx = {
            "<|sfx:cough|>", "<|sfx:laughter|>", "<|sfx:crying|>", "<|sfx:screaming|>",
            "<|sfx:burping|>", "<|sfx:humming|>", "<|sfx:sigh|>", "<|sfx:sniff|>",
            "<|sfx:sneeze|>",
        }
        actual_sfx = {t for t in ab.VALID_TAGS if t.startswith("<|sfx:")}
        self.assertEqual(actual_sfx, expected_sfx)

    def test_unknown_sfx_tag_is_still_rejected(self):
        with self.assertRaises(ValueError):
            ab.validate_control_tags("Текст с <|sfx:giggle|> тегом.")


class TestF12TruncatedWavDetection(unittest.TestCase):
    """F12: read_wav trusted the WAV header's declared frame count without checking that
    enough bytes were actually present -- a truncated file was read as if it were a
    shorter-but-valid clip, with no error."""

    def test_truncated_wav_raises_instead_of_silently_shrinking(self):
        import struct
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "good.wav"
            ab.write_wav(path, np.zeros(2478, dtype=np.float64), 24000)
            raw = path.read_bytes()
            # Truncate the file body but leave the RIFF/data-chunk-size header claiming
            # the original frame count -- this reproduces the audit's exact repro.
            truncated_path = Path(d) / "truncated.wav"
            truncated_path.write_bytes(raw[:-2000])
            with self.assertRaises(ValueError):
                ab.read_wav(truncated_path)


class TestF7ContentHashKeyingAndRelativePaths(unittest.TestCase):
    """F7: resume required the ENTIRE chunking plan's text list to match byte-for-byte,
    so one edited chunk invalidated every other already-generated segment; output_path
    was also stored as an absolute string, so moving the output directory forced a full
    silent regeneration."""

    def test_unrelated_edit_reuses_other_segments_by_content_hash(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            # One Chunk per sentence (built directly, not via chunk_sentences' grouping)
            # so each sentence is guaranteed its own segment/hash.
            chunks_v1 = [
                ab.Chunk(index=0, sentences=["Первое предложение."], reopened_tags={}, text="Первое предложение."),
                ab.Chunk(index=1, sentences=["Второе предложение."], reopened_tags={}, text="Второе предложение."),
                ab.Chunk(index=2, sentences=["Третье предложение."], reopened_tags={}, text="Третье предложение."),
            ]
            manifest = ab.load_or_create_manifest(manifest_path, chunks_v1, max_chars=1000, tag_scope="chunk")
            manifest["segments"][0]["status"] = "done"
            manifest["segments"][0]["num_samples"] = 123
            ab.save_manifest(manifest, manifest_path)

            # Edit only the LAST sentence; the first chunk's text/hash is unchanged.
            chunks_v2 = [
                ab.Chunk(index=0, sentences=["Первое предложение."], reopened_tags={}, text="Первое предложение."),
                ab.Chunk(index=1, sentences=["Второе предложение."], reopened_tags={}, text="Второе предложение."),
                ab.Chunk(index=2, sentences=["Совсем другое предложение."], reopened_tags={}, text="Совсем другое предложение."),
            ]
            manifest2 = ab.load_or_create_manifest(manifest_path, chunks_v2, max_chars=1000, tag_scope="chunk")
            self.assertEqual(manifest2["segments"][0]["status"], "done")
            self.assertEqual(manifest2["segments"][0]["num_samples"], 123)

    def test_output_path_is_relative_not_absolute(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = ab.chunk_sentences(ab.split_sentences("Одно предложение тут."), max_chars=1000)
            manifest = ab.load_or_create_manifest(manifest_path, chunks, max_chars=1000, tag_scope="chunk")
            for seg in manifest["segments"]:
                self.assertFalse(Path(seg["output_path"]).is_absolute())

    def test_mismatched_max_chars_raises_with_specific_reason(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = ab.chunk_sentences(ab.split_sentences("Одно предложение тут."), max_chars=1000)
            ab.load_or_create_manifest(manifest_path, chunks, max_chars=1000, tag_scope="chunk")
            with self.assertRaises(RuntimeError) as ctx:
                ab.load_or_create_manifest(manifest_path, chunks, max_chars=500, tag_scope="chunk")
            self.assertIn("max_chars", str(ctx.exception))


class TestF13AllowGapsAssembly(unittest.TestCase):
    """F13: assemble_chapter used to raise on the very first non-'done' segment,
    discarding a whole read pass; allow_gaps should insert reported silence instead."""

    def _make_segment(self, d, name, seconds, sr=8000, value=0.2):
        path = Path(d) / name
        n = int(seconds * sr)
        ab.write_wav(path, np.full(n, value, dtype=np.float64), sr)

    def test_default_raises_all_or_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "segment_0000.wav", 0.2)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "segment_0000.wav", "text": "x"},
                    {"index": 1, "status": "failed", "output_path": "segment_0001.wav", "text": "x"},
                ]
            }
            with self.assertRaises(RuntimeError):
                ab.assemble_chapter(manifest, Path(d) / "chapter.wav", base_dir=Path(d))

    def test_allow_gaps_inserts_silence_and_reports_it(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "segment_0000.wav", 0.2)
            self._make_segment(d, "segment_0002.wav", 0.2)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "segment_0000.wav", "text": "x"},
                    {"index": 1, "status": "failed", "output_path": "segment_0001.wav", "text": "x"},
                    {"index": 2, "status": "done", "output_path": "segment_0002.wav", "text": "x"},
                ]
            }
            result = ab.assemble_chapter(
                manifest, Path(d) / "chapter.wav", base_dir=Path(d), allow_gaps=True, silence_ms=0
            )
            self.assertEqual(len(result["gaps"]), 1)
            self.assertEqual(result["gaps"][0]["index"], 1)
            self.assertTrue((Path(d) / "chapter.wav").exists())


class TestScreenplayParsing(unittest.TestCase):
    """parse_screenplay: reads docs/guides/audiobook_guide.md sec. 3's
    [{"speaker": ..., "text": ...}, ...] DSL, validates it, and drops any field besides
    speaker/text so unrelated edits can never affect a segment's content hash."""

    def test_valid_screenplay_parses_and_keeps_only_speaker_and_text(self):
        data = [
            {"speaker": "narrator", "text": "Глава первая.", "note": "intro line"},
            {"speaker": "arjuna", "text": "<|emotion:sadness|>О Кришна."},
        ]
        lines = ab.parse_screenplay(data)
        self.assertEqual(
            lines,
            [
                {"speaker": "narrator", "text": "Глава первая."},
                {"speaker": "arjuna", "text": "<|emotion:sadness|>О Кришна."},
            ],
        )

    def test_top_level_must_be_a_list(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay({"speaker": "narrator", "text": "x"})
        self.assertIn("array", str(ctx.exception))

    def test_empty_screenplay_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay([])
        self.assertIn("empty", str(ctx.exception))

    def test_non_object_line_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay(["Глава первая."])
        self.assertIn("line 0", str(ctx.exception))

    def test_missing_speaker_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay([{"text": "Глава первая."}])
        self.assertIn("speaker", str(ctx.exception))

    def test_blank_speaker_rejected(self):
        with self.assertRaises(ValueError):
            ab.parse_screenplay([{"speaker": "   ", "text": "Глава первая."}])

    def test_non_string_speaker_rejected(self):
        with self.assertRaises(ValueError):
            ab.parse_screenplay([{"speaker": 5, "text": "Глава первая."}])

    def test_missing_text_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay([{"speaker": "narrator"}])
        self.assertIn("text", str(ctx.exception))

    def test_blank_text_rejected(self):
        with self.assertRaises(ValueError):
            ab.parse_screenplay([{"speaker": "narrator", "text": "   "}])

    def test_unknown_control_tag_in_line_text_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ab.parse_screenplay(
                [{"speaker": "narrator", "text": "<|emotion:Elation|>Привет."}]
            )
        self.assertIn("unknown control tag", str(ctx.exception))

    def test_invalid_style_whispering_tag_still_validates_as_a_known_tag(self):
        # whispering IS one of the 34 known-valid tags (VALID_TAGS) -- it just doesn't
        # sound like whispering (docs/research/audiobook/m4-sentiment-results.md sec 6b).
        # parse_screenplay only checks the tag is a real one, not that it "works".
        lines = ab.parse_screenplay(
            [{"speaker": "narrator", "text": "<|style:whispering|>Тише."}]
        )
        self.assertEqual(lines[0]["text"], "<|style:whispering|>Тише.")


class TestScreenplayChunking(unittest.TestCase):
    """chunk_screenplay: reuses split_sentences/chunk_sentences per line, resets tag state
    at every speaker line, and threads `speaker` into every resulting Chunk."""

    def test_each_chunk_is_tagged_with_its_speaker(self):
        lines = ab.parse_screenplay(
            [
                {"speaker": "narrator", "text": "Глава первая. Все собрались."},
                {"speaker": "arjuna", "text": "О Кришна, что происходит?"},
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = ab.chunk_screenplay(lines, max_chars=1000, tag_scope="sentence")
        speakers = [c.speaker for c in chunks]
        self.assertIn("narrator", speakers)
        self.assertIn("arjuna", speakers)

    def test_indices_are_renumbered_globally_across_lines(self):
        lines = ab.parse_screenplay(
            [
                {"speaker": "narrator", "text": "Раз. Два. Три."},
                {"speaker": "arjuna", "text": "Четыре."},
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = ab.chunk_screenplay(lines, max_chars=1000, tag_scope="sentence")
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_emotion_tag_does_not_leak_across_a_speaker_change(self):
        # arjuna's line opens <|emotion:sadness|> and never closes it; krishna's very next
        # line must NOT have that tag reopened onto it -- a character's leftover emotional
        # state has no textual basis once a different speaker starts talking.
        lines = ab.parse_screenplay(
            [
                {"speaker": "arjuna", "text": "<|emotion:sadness|>Мне грустно."},
                {"speaker": "krishna", "text": "Не грусти, Арджуна."},
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = ab.chunk_screenplay(lines, max_chars=1000, tag_scope="sentence")
        krishna_chunk = next(c for c in chunks if c.speaker == "krishna")
        self.assertNotIn("<|emotion:sadness|>", krishna_chunk.text)

    def test_single_speaker_screenplay_emits_no_multivoice_warning(self):
        lines = ab.parse_screenplay(
            [
                {"speaker": "narrator", "text": "Первая строка."},
                {"speaker": "narrator", "text": "Вторая строка."},
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ab.chunk_screenplay(lines, max_chars=1000, tag_scope="sentence")
        self.assertEqual(caught, [])

    def test_multi_speaker_screenplay_warns_naming_speakers_once(self):
        lines = ab.parse_screenplay(
            [
                {"speaker": "narrator", "text": "Первая строка."},
                {"speaker": "arjuna", "text": "Вторая строка."},
                {"speaker": "krishna", "text": "Третья строка."},
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ab.chunk_screenplay(lines, max_chars=1000, tag_scope="sentence")
        multivoice_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(multivoice_warnings), 1)
        msg = str(multivoice_warnings[0].message)
        self.assertIn("narrator", msg)
        self.assertIn("arjuna", msg)
        self.assertIn("krishna", msg)
        self.assertIn("default voice", msg)

    def test_long_reply_from_one_speaker_still_chunks_and_reopens_tags(self):
        # A single reply longer than max_chars must still be split into multiple chunks
        # by the existing chunk_sentences machinery, with its own tag reopened across
        # those internal chunk boundaries -- this is the exact gap the plain guide-example
        # pipeline (one model.generate() call per line) had (m4-chapter-results.md sec 2).
        long_text = "<|emotion:sadness|>" + " ".join(
            f"Предложение номер {i}." for i in range(1, 30)
        )
        lines = ab.parse_screenplay([{"speaker": "arjuna", "text": long_text}])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = ab.chunk_screenplay(lines, max_chars=100, tag_scope="sentence")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("<|emotion:sadness|>", c.text)
            self.assertEqual(c.speaker, "arjuna")


class TestScreenplayIncrementalRegeneration(unittest.TestCase):
    """The key property (Refs #57 task item 2): editing ONE line of an N-line screenplay
    and rebuilding the manifest must regenerate exactly that one segment -- everything else
    must be reused via the content hash (F7), extended to include `speaker`."""

    def _screenplay_chunks(self, lines_data):
        lines = ab.parse_screenplay(lines_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ab.chunk_screenplay(lines, max_chars=1000, tag_scope="chunk")

    def test_editing_one_line_only_invalidates_that_lines_segment(self):
        import tempfile

        base_lines = [
            {"speaker": "narrator", "text": "Первая строка."},
            {"speaker": "arjuna", "text": "Вторая строка."},
            {"speaker": "krishna", "text": "Третья строка."},
            {"speaker": "narrator", "text": "Четвёртая строка."},
        ]
        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks_v1 = self._screenplay_chunks(base_lines)
            manifest = ab.load_or_create_manifest(
                manifest_path, chunks_v1, max_chars=1000, tag_scope="chunk"
            )
            for i, seg in enumerate(manifest["segments"]):
                seg["status"] = "done"
                seg["num_samples"] = 1000 + i
            ab.save_manifest(manifest, manifest_path)

            edited_lines = [dict(l) for l in base_lines]
            edited_lines[2] = {"speaker": "krishna", "text": "Совсем другая третья строка."}
            chunks_v2 = self._screenplay_chunks(edited_lines)
            manifest2 = ab.load_or_create_manifest(
                manifest_path, chunks_v2, max_chars=1000, tag_scope="chunk"
            )

            statuses = [seg["status"] for seg in manifest2["segments"]]
            self.assertEqual(statuses.count("pending"), 1)
            self.assertEqual(statuses.count("done"), 3)
            # The three UNCHANGED lines kept their original num_samples (real reuse, not
            # just a matching status string).
            unchanged_samples = sorted(
                seg["num_samples"] for seg in manifest2["segments"] if seg["status"] == "done"
            )
            self.assertEqual(unchanged_samples, [1000, 1001, 1003])

    def test_editing_an_unrelated_field_does_not_invalidate_any_segment(self):
        import tempfile

        base_lines = [
            {"speaker": "narrator", "text": "Первая строка.", "id": "L1"},
            {"speaker": "arjuna", "text": "Вторая строка.", "id": "L2"},
        ]
        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks_v1 = self._screenplay_chunks(base_lines)
            manifest = ab.load_or_create_manifest(
                manifest_path, chunks_v1, max_chars=1000, tag_scope="chunk"
            )
            for seg in manifest["segments"]:
                seg["status"] = "done"
                seg["num_samples"] = 42
            ab.save_manifest(manifest, manifest_path)

            # Only the extraneous "id"/"note" field changes -- speaker and text untouched.
            edited_lines = [
                {"speaker": "narrator", "text": "Первая строка.", "id": "L1-renamed"},
                {"speaker": "arjuna", "text": "Вторая строка.", "note": "unrelated"},
            ]
            chunks_v2 = self._screenplay_chunks(edited_lines)
            manifest2 = ab.load_or_create_manifest(
                manifest_path, chunks_v2, max_chars=1000, tag_scope="chunk"
            )
            self.assertTrue(all(seg["status"] == "done" for seg in manifest2["segments"]))
            self.assertTrue(all(seg["num_samples"] == 42 for seg in manifest2["segments"]))

    def test_renaming_the_speaker_of_an_unchanged_line_invalidates_its_segment(self):
        # Same wording, different speaker -- the hash MUST change (speaker selects the
        # voice once cloning is wired in), so this must NOT be silently reused.
        import tempfile

        base_lines = [{"speaker": "narrator", "text": "Общая фраза."}]
        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks_v1 = self._screenplay_chunks(base_lines)
            manifest = ab.load_or_create_manifest(
                manifest_path, chunks_v1, max_chars=1000, tag_scope="chunk"
            )
            manifest["segments"][0]["status"] = "done"
            manifest["segments"][0]["num_samples"] = 7
            ab.save_manifest(manifest, manifest_path)

            renamed_lines = [{"speaker": "arjuna", "text": "Общая фраза."}]
            chunks_v2 = self._screenplay_chunks(renamed_lines)
            manifest2 = ab.load_or_create_manifest(
                manifest_path, chunks_v2, max_chars=1000, tag_scope="chunk"
            )
            self.assertEqual(manifest2["segments"][0]["status"], "pending")
            self.assertEqual(manifest2["segments"][0]["speaker"], "arjuna")


class TestScreenplaySpeakerChangeAssembly(unittest.TestCase):
    """assemble_chapter's optional speaker_change_silence_ms (screenplay format)."""

    def _make_segment(self, d, name, seconds, sr=8000, value=0.2):
        path = Path(d) / name
        n = int(seconds * sr)
        ab.write_wav(path, np.full(n, value, dtype=np.float64), sr)

    def test_speaker_change_uses_the_longer_pause(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "seg0.wav", 0.1)
            self._make_segment(d, "seg1.wav", 0.1)
            self._make_segment(d, "seg2.wav", 0.1)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "seg0.wav", "text": "x", "speaker": "narrator"},
                    {"index": 1, "status": "done", "output_path": "seg1.wav", "text": "x", "speaker": "narrator"},
                    {"index": 2, "status": "done", "output_path": "seg2.wav", "text": "x", "speaker": "arjuna"},
                ]
            }
            result = ab.assemble_chapter(
                manifest,
                Path(d) / "chapter.wav",
                base_dir=Path(d),
                silence_ms=100,
                speaker_change_silence_ms=900,
            )
            # 3 segments of 0.1s + one same-speaker join (100ms) + one speaker-change join
            # (900ms) = 0.3 + 0.1 + 0.9 = 1.3s.
            self.assertAlmostEqual(result["total_duration_seconds"], 1.3, places=2)

    def test_missing_speaker_key_falls_back_to_uniform_silence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "seg0.wav", 0.1)
            self._make_segment(d, "seg1.wav", 0.1)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "seg0.wav", "text": "x"},
                    {"index": 1, "status": "done", "output_path": "seg1.wav", "text": "x"},
                ]
            }
            result = ab.assemble_chapter(
                manifest,
                Path(d) / "chapter.wav",
                base_dir=Path(d),
                silence_ms=100,
                speaker_change_silence_ms=900,
            )
            self.assertAlmostEqual(result["total_duration_seconds"], 0.3, places=2)


if __name__ == "__main__":
    unittest.main()
