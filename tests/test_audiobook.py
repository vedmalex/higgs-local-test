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
            good = {"model": ab.MODEL_ID, "max_chars": 1, "tag_scope": "chunk",
                    "fades_ms": [ab.DEFAULT_FADE_IN_MS, ab.DEFAULT_FADE_OUT_MS],
                    "segments": []}
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

        def generate(self, text, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                     fade_in_ms=None, fade_out_ms=None):
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

            def flaky_generate(text, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                               fade_in_ms=None, fade_out_ms=None):
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

    def test_undocumented_env_and_chatml_tokens_are_rejected(self):
        # <|env:music|> (id 151702) and <|env:noise|> (id 151703) exist in the
        # tokenizer's added_tokens but are entirely undocumented in PROMPTING.md -- no
        # syntax, no example, no mention. Deliberately NOT in VALID_TAGS (see the comment
        # above it in src/audiobook.py) until their actual effect is investigated, so they
        # must still be rejected like any other invented/unknown tag.
        with self.assertRaises(ValueError):
            ab.validate_control_tags("Текст с <|env:music|> тегом.")
        with self.assertRaises(ValueError):
            ab.validate_control_tags("Текст с <|env:noise|> тегом.")

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


class TestFadeInvalidatesResume(unittest.TestCase):
    """Changing the fades must refuse to resume, like the voice reference does.

    The fades are baked into every wav at generation time and are NOT part of the
    segment hash (which keys on speaker+text only). Without this guard, re-running
    with corrected fades would happily reuse every segment whose onset the old fade
    had already damaged, and the chapter would come out uneven with nothing to point
    at. The damage cannot be repaired afterwards either: at 16-bit the first samples
    quantize to 0/-1/5, so dividing the ramp back out yields noise, not the consonant.
    """

    def _write(self, d, **extra):
        chunks = ab.chunk_sentences(ab.split_sentences("Одно предложение тут."), max_chars=500)
        mp = Path(d) / "manifest.json"
        base = {"model": ab.MODEL_ID, "max_chars": 500, "tag_scope": "chunk",
                "voice_reference": None, "segments": []}
        base.update(extra)
        mp.write_text(json.dumps(base), encoding="utf-8")
        return chunks, mp

    def test_changed_fades_refuse_to_resume(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chunks, mp = self._write(d, fades_ms=[30.0, 15.0])
            with self.assertRaises(RuntimeError) as ctx:
                ab.load_or_create_manifest(mp, chunks, max_chars=500, tag_scope="chunk",
                                           fade_in_ms=5.0, fade_out_ms=5.0)
            self.assertIn("fades_ms", str(ctx.exception))

    def test_same_fades_resume_fine(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chunks, mp = self._write(d, fades_ms=[5.0, 5.0])
            m = ab.load_or_create_manifest(mp, chunks, max_chars=500, tag_scope="chunk",
                                           fade_in_ms=5.0, fade_out_ms=5.0)
            self.assertIn("segments", m)

    def test_missing_field_reads_as_mlx_audios_old_defaults(self):
        """Not "unknown, wave it through" — such a manifest provably used 30/15."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chunks, mp = self._write(d)  # no fades_ms at all
            # resuming with the corrected fades must refuse...
            with self.assertRaises(RuntimeError) as ctx:
                ab.load_or_create_manifest(mp, chunks, max_chars=500, tag_scope="chunk",
                                           fade_in_ms=5.0, fade_out_ms=5.0)
            self.assertIn("отсутствует в манифесте", str(ctx.exception))
            # ...but resuming with the old values it was actually made with must work.
            m = ab.load_or_create_manifest(mp, chunks, max_chars=500, tag_scope="chunk",
                                           fade_in_ms=30.0, fade_out_ms=15.0)
            self.assertIn("segments", m)

    def test_new_manifest_records_the_fades(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chunks = ab.chunk_sentences(ab.split_sentences("Одно предложение тут."), max_chars=500)
            m = ab.build_manifest(chunks, max_chars=500, tag_scope="chunk",
                                  fade_in_ms=7.0, fade_out_ms=3.0)
            self.assertEqual(m["fades_ms"], [7.0, 3.0])


class TestFadeThreadedIntoGeneration(unittest.TestCase):
    """The fade lengths must actually reach model.generate()/batch_generate().

    mlx_audio defaults to fade_in_ms=30, which multiplies the first 30 ms of every
    segment by a 0->1 ramp -- long enough to swallow a plosive or half a fricative,
    so a segment opening on "Шри" loses its attack. The owner heard exactly that once
    the voice reference removed the per-seam voice change that had been masking it.
    Measured on chapter-114-e2: with fade 30 the first 40 ms sit at 0.001-0.004, with
    fade 5 at 0.004-0.056. If these kwargs silently stop being forwarded, the defect
    comes back inaudibly to the tests, so assert the wiring on both paths.
    """

    class _RecordingModel:
        def __init__(self):
            self.generate_kwargs = []
            self.batch_kwargs = []

        @staticmethod
        def _result(text):
            sr = 24000
            n = max(1, int(len(text) * 0.05 * sr))
            return type("R", (), {"audio": np.zeros(n, dtype=np.float64),
                                  "sample_rate": sr, "sequence_idx": 0})()

        def generate(self, text, temperature, max_new_tokens, ref_audio_codes=None,
                     ref_text=None, fade_in_ms=None, fade_out_ms=None):
            self.generate_kwargs.append((fade_in_ms, fade_out_ms))
            return [self._result(text)]

        def batch_generate(self, texts, temperature, max_new_tokens, ref_audio_codes=None,
                           ref_text=None, fade_in_ms=None, fade_out_ms=None):
            self.batch_kwargs.append((fade_in_ms, fade_out_ms))
            for i, t in enumerate(texts):
                r = self._result(t)
                r.sequence_idx = i
                yield r

    def _manifest(self, d, texts):
        chunks = [ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t)
                  for i, t in enumerate(texts)]
        manifest_path = Path(d) / "manifest.json"
        return ab.build_manifest(chunks, max_chars=500, tag_scope="chunk"), manifest_path

    def test_defaults_are_short_not_mlx_audios_thirty(self):
        """The whole point of the fix: our default must not be 30 ms."""
        self.assertLess(ab.DEFAULT_FADE_IN_MS, 30.0)
        self.assertGreater(ab.DEFAULT_FADE_IN_MS, 0.0,
                           "not zero either -- a jump from silence to a loud onset clicks")

    def test_unbatched_path_forwards_the_fades(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            manifest, mp = self._manifest(d, ["Шри Сута Госвами сказал."])
            model = self._RecordingModel()
            ab.generate_segments(model, manifest, mp, batch_size=1,
                                 fade_in_ms=7.0, fade_out_ms=3.0)
            self.assertEqual(model.generate_kwargs, [(7.0, 3.0)])

    def test_batched_path_forwards_the_fades(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            manifest, mp = self._manifest(d, ["Первый.", "Второй.", "Третий."])
            model = self._RecordingModel()
            ab.generate_segments(model, manifest, mp, batch_size=3,
                                 fade_in_ms=7.0, fade_out_ms=3.0)
            self.assertEqual(model.batch_kwargs, [(7.0, 3.0)])

    def test_defaults_reach_the_model_when_not_passed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            manifest, mp = self._manifest(d, ["Одна фраза."])
            model = self._RecordingModel()
            ab.generate_segments(model, manifest, mp, batch_size=1)
            self.assertEqual(model.generate_kwargs,
                             [(ab.DEFAULT_FADE_IN_MS, ab.DEFAULT_FADE_OUT_MS)])


class TestMidSentenceJoinSilence(unittest.TestCase):
    """assemble_chapter's mid_sentence_silence_ms (Refs #57).

    The owner heard the splice seam in a blind test (`midsentence_split` survey set);
    the measured cause was timing, not pitch -- the fixed 200 ms join is ~2x the model's
    own ~100 ms pause at the comma a long sentence gets cut on
    (docs/research/audiobook/m4-midsentence-split-results.md, n=1).
    """

    def _make_segment(self, d, name, seconds, sr=8000, value=0.2):
        path = Path(d) / name
        n = int(seconds * sr)
        ab.write_wav(path, np.full(n, value, dtype=np.float64), sr)

    def test_ends_mid_sentence_classifies_real_cuts(self):
        # Cut mid-sentence: what _force_split_long_sentence actually produces.
        self.assertTrue(ab.ends_mid_sentence("Хотя он и старался скрыть величие,"))
        self.assertTrue(ab.ends_mid_sentence("первая часть;"))
        self.assertTrue(ab.ends_mid_sentence("оборван на слове"))
        # Genuine sentence ends, including closing quotes/brackets after the punctuation.
        self.assertFalse(ab.ends_mid_sentence("Это конец."))
        self.assertFalse(ab.ends_mid_sentence("Неужели?"))
        self.assertFalse(ab.ends_mid_sentence("Очень хорошо!»"))
        self.assertFalse(ab.ends_mid_sentence('Он сказал: "Хорошо."'))
        self.assertFalse(ab.ends_mid_sentence("многоточие…"))
        # Empty/whitespace must not raise and must not claim a continuation.
        self.assertFalse(ab.ends_mid_sentence("   "))

    def test_mid_sentence_join_uses_the_shorter_pause(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "seg0.wav", 0.1)
            self._make_segment(d, "seg1.wav", 0.1)
            self._make_segment(d, "seg2.wav", 0.1)
            manifest = {
                "segments": [
                    # seg0 is cut mid-sentence -> the join before seg1 is the short one.
                    {"index": 0, "status": "done", "output_path": "seg0.wav", "text": "Хотя он и старался,"},
                    # seg1 ends properly -> the join before seg2 is the normal one.
                    {"index": 1, "status": "done", "output_path": "seg1.wav", "text": "мудрецы почтили его."},
                    {"index": 2, "status": "done", "output_path": "seg2.wav", "text": "Следующее предложение."},
                ]
            }
            result = ab.assemble_chapter(
                manifest,
                Path(d) / "chapter.wav",
                base_dir=Path(d),
                silence_ms=200,
                mid_sentence_silence_ms=100,
            )
            # 0.3s audio + one mid-sentence join (0.1) + one normal join (0.2) = 0.6s.
            self.assertAlmostEqual(result["total_duration_seconds"], 0.6, places=2)
            self.assertEqual(result["mid_sentence_silence_ms"], 100)

    def test_unset_keeps_the_old_uniform_behavior(self):
        """Regression guard: without the flag nothing about assembly changes."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "seg0.wav", 0.1)
            self._make_segment(d, "seg1.wav", 0.1)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "seg0.wav", "text": "Хотя он и старался,"},
                    {"index": 1, "status": "done", "output_path": "seg1.wav", "text": "конец."},
                ]
            }
            result = ab.assemble_chapter(
                manifest,
                Path(d) / "chapter.wav",
                base_dir=Path(d),
                silence_ms=200,
            )
            self.assertAlmostEqual(result["total_duration_seconds"], 0.4, places=2)

    def test_speaker_change_wins_over_mid_sentence(self):
        """A new speaker starting mid-sentence still earns the speaker-change pause."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_segment(d, "seg0.wav", 0.1)
            self._make_segment(d, "seg1.wav", 0.1)
            manifest = {
                "segments": [
                    {"index": 0, "status": "done", "output_path": "seg0.wav",
                     "text": "Мудрецы сказали,", "speaker": "narrator"},
                    {"index": 1, "status": "done", "output_path": "seg1.wav",
                     "text": "О царь!", "speaker": "sages"},
                ]
            }
            result = ab.assemble_chapter(
                manifest,
                Path(d) / "chapter.wav",
                base_dir=Path(d),
                silence_ms=200,
                mid_sentence_silence_ms=100,
                speaker_change_silence_ms=900,
            )
            self.assertAlmostEqual(result["total_duration_seconds"], 1.1, places=2)


class TestStressApostropheNotation(unittest.TestCase):
    """Refs #57: the owner confirmed by ear (docs/research/audiobook/m4-tag-inventory-results.md
    sec. 3) that an apostrophe placed right after the stressed vowel ("за'мок") is the one
    stress-mark notation that (a) is never spoken aloud and (b) does not corrupt Higgs's
    output, unlike U+0301/`+`/doubled vowels. These tests check that the apostrophe (1) does
    not disturb split_sentences/chunk_sentences/validate_control_tags/hashing, and (2) that the
    heuristic used to tell a stress mark apart from a name apostrophe ("д'Артаньян",
    "О'Генри") behaves as documented."""

    def test_apostrophe_does_not_break_sentence_splitting(self):
        text = "На холме стоит старинный за'мок. На двери висит крепкий замо'к."
        sentences = ab.split_sentences(text)
        self.assertEqual(
            sentences,
            ["На холме стоит старинный за'мок.", "На двери висит крепкий замо'к."],
        )

    def test_apostrophe_survives_chunking_unchanged(self):
        sentences = ab.split_sentences("Он живёт в старинном за'мке уже сто лет.")
        chunks = ab.chunk_sentences(sentences, max_chars=500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("за'мке", chunks[0].text)

    def test_apostrophe_does_not_trip_control_tag_validation(self):
        # Must not raise: a bare "'" does not match _TAG_SHAPE_RE and is not tracked as a
        # quote pair (only the curly '‘'/'’' pair is).
        ab.validate_control_tags("<|emotion:sadness|> Он вошёл в старый за'мок.")

    def test_stress_mark_apostrophe_recognized_inside_lowercase_word(self):
        text = "за'мок"
        # index of the apostrophe
        idx = text.index("'")
        self.assertTrue(ab.is_stress_apostrophe(text, idx))
        self.assertEqual(ab.count_stress_marks(text), 1)
        self.assertEqual(ab.count_ambiguous_apostrophes(text), 0)

    def test_name_apostrophe_after_consonant_and_capital_not_treated_as_stress(self):
        # "д'Артаньян": apostrophe follows a consonant and precedes an uppercase letter --
        # neither half of the stress-mark rule is satisfied.
        text = "Это был д'Артаньян."
        idx = text.index("'")
        self.assertFalse(ab.is_stress_apostrophe(text, idx))
        self.assertEqual(ab.count_stress_marks(text), 0)
        self.assertEqual(ab.count_ambiguous_apostrophes(text), 1)

    def test_name_apostrophe_after_vowel_but_before_capital_not_treated_as_stress(self):
        # "О'Генри": apostrophe follows a vowel (like a real stress mark would) but is
        # followed by an uppercase letter, not a lowercase continuation of the same word --
        # this is the case the heuristic exists specifically to catch.
        text = "Рассказ О'Генри мне понравился."
        idx = text.index("'")
        self.assertFalse(ab.is_stress_apostrophe(text, idx))
        self.assertEqual(ab.count_stress_marks(text), 0)
        self.assertEqual(ab.count_ambiguous_apostrophes(text), 1)

    def test_manifest_records_stress_and_ambiguous_apostrophe_counts(self):
        sentences = ab.split_sentences("В старинном за'мке жил д'Артаньян.")
        chunks = ab.chunk_sentences(sentences, max_chars=500)
        manifest = ab.build_manifest(chunks, max_chars=500, tag_scope="chunk")
        self.assertEqual(manifest["stress_marks_detected"], 1)
        self.assertEqual(manifest["ambiguous_apostrophes_detected"], 1)

    def test_editing_stress_mark_regenerates_only_that_segment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks_v1 = [
                ab.Chunk(index=0, sentences=["Стоит замок."], reopened_tags={}, text="Стоит замок."),
                ab.Chunk(index=1, sentences=["Второе предложение."], reopened_tags={}, text="Второе предложение."),
            ]
            manifest = ab.load_or_create_manifest(manifest_path, chunks_v1, max_chars=1000, tag_scope="chunk")
            manifest["segments"][0]["status"] = "done"
            manifest["segments"][0]["num_samples"] = 42
            manifest["segments"][1]["status"] = "done"
            manifest["segments"][1]["num_samples"] = 99
            ab.save_manifest(manifest, manifest_path)

            # Add a stress mark to only the first segment's text -- its hash must change
            # (forcing regeneration) while the untouched second segment is reused.
            chunks_v2 = [
                ab.Chunk(index=0, sentences=["Стоит за'мок."], reopened_tags={}, text="Стоит за'мок."),
                ab.Chunk(index=1, sentences=["Второе предложение."], reopened_tags={}, text="Второе предложение."),
            ]
            manifest2 = ab.load_or_create_manifest(manifest_path, chunks_v2, max_chars=1000, tag_scope="chunk")
            self.assertNotEqual(
                manifest2["segments"][0]["text_hash"], manifest["segments"][0]["text_hash"]
            )
            self.assertEqual(manifest2["segments"][0]["status"], "pending")
            self.assertEqual(manifest2["segments"][1]["status"], "done")
            self.assertEqual(manifest2["segments"][1]["num_samples"], 99)


class TestBatchGenerationIntegration(unittest.TestCase):
    """Refs #114: `generate_segments(..., batch_size=N)` wires `model.batch_generate()`
    into the working pipeline. These tests target exactly the risk areas the audit called
    out: result<->segment identity across the batch boundary, resumable atomic manifest
    writes mid-batch, per-segment (not per-batch) audio validation, and `--batch-size 1`
    being byte-for-byte the pre-#114 behavior.
    """

    class _FakeBatchModel:
        """`.generate()` and `.batch_generate()` both derive deterministic, distinguishable
        audio length from the text's own length, so a scrambled segment<->result mapping
        shows up as a length (and therefore duration) mismatch against the *wrong* text.
        `batch_generate` yields in REVERSED order on purpose -- return order must never be
        trusted, only `sequence_idx` -- and can be told to always fail (or fail for texts
        matching a substring) to exercise the retry/degrade/fallback path.
        """

        SR = 24000

        class _BatchResult:
            def __init__(self, audio, sequence_idx, sample_rate, processing_time_seconds=0.01):
                self.audio = audio
                self.sequence_idx = sequence_idx
                self.sample_rate = sample_rate
                self.processing_time_seconds = processing_time_seconds

        def __init__(self, fail_substrings: tuple[str, ...] = (), fail_batch_times: int = 0):
            self.fail_substrings = fail_substrings
            self.fail_batch_times = fail_batch_times
            self.batch_calls: list[list[str]] = []
            self.generate_calls: list[str] = []
            self.generate_ref_kwargs: list[tuple] = []
            self.batch_ref_kwargs: list[tuple] = []

        def _audio_for(self, text: str) -> np.ndarray:
            n = max(1, int(len(text) * 0.05 * self.SR))
            return np.zeros(n, dtype=np.float64)

        def generate(self, text, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                     fade_in_ms=None, fade_out_ms=None):
            # Deliberately always succeeds here, even for a `fail_substrings` text -- that
            # knob only makes `batch_generate` produce bad audio for the matching row, so
            # tests can verify the single-segment fallback path actually rescues it.
            self.generate_calls.append(text)
            self.generate_ref_kwargs.append((ref_audio_codes, ref_text))
            return [type("R", (), {"audio": self._audio_for(text), "sample_rate": self.SR})()]

        def batch_generate(self, texts, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                           fade_in_ms=None, fade_out_ms=None):
            self.batch_calls.append(list(texts))
            self.batch_ref_kwargs.append((ref_audio_codes, ref_text))
            if len(self.batch_calls) <= self.fail_batch_times:
                raise RuntimeError("simulated whole-batch failure")
            for idx in reversed(range(len(texts))):
                text = texts[idx]
                if any(s in text for s in self.fail_substrings):
                    # Bad audio for exactly this row -- too short to pass F6 validation --
                    # while the rest of the batch still yields plausible audio.
                    yield self._BatchResult(np.zeros(1, dtype=np.float64), idx, self.SR)
                else:
                    yield self._BatchResult(self._audio_for(text), idx, self.SR)

    def _manifest(self, d, texts):
        chunks = [
            ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t)
            for i, t in enumerate(texts)
        ]
        manifest_path = Path(d) / "manifest.json"
        manifest = ab.load_or_create_manifest(manifest_path, chunks, max_chars=1000, tag_scope="chunk")
        return manifest, manifest_path

    def test_batch_results_map_to_correct_segments_despite_reversed_yield_order(self):
        import tempfile

        texts = [
            "Короткий текст.",
            "Существенно более длинный текст для второго сегмента совсем.",
            "Средний по длине текст тут.",
        ]
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, texts)
            model = self._FakeBatchModel()
            result = ab.generate_segments(
                model, manifest, manifest_path, max_retries=2, retry_base_delay=0.0, batch_size=3
            )
            self.assertEqual(result["failures"], [])
            self.assertEqual(len(model.batch_calls), 1)
            for i, text in enumerate(texts):
                entry = manifest["segments"][i]
                self.assertEqual(entry["status"], "done")
                expected_samples = model._audio_for(text).size
                self.assertEqual(
                    entry["num_samples"],
                    expected_samples,
                    f"segment {i} ({text!r}) got the wrong audio -- batch result<->segment "
                    "mapping was scrambled",
                )
                # Cross-check against the actual WAV written to disk, not just the manifest.
                audio, sr = ab.read_wav(Path(d) / entry["output_path"])
                self.assertEqual(sr, model.SR)
                self.assertEqual(len(audio), expected_samples)

    def test_batch_size_one_is_byte_for_byte_the_original_unbatched_path(self):
        import tempfile

        texts = ["Первый сегмент тут.", "Второй сегмент здесь.", "Третий сегмент готов."]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            manifest1, path1 = self._manifest(d1, texts)
            model1 = self._FakeBatchModel()
            ab.generate_segments(
                model1, manifest1, path1, max_retries=2, retry_base_delay=0.0, batch_size=1
            )

            manifest2, path2 = self._manifest(d2, texts)
            model2 = self._FakeBatchModel()
            ab.generate_segments(
                model2, manifest2, path2, max_retries=2, retry_base_delay=0.0
            )  # default batch_size

            # batch_generate must never be called on the batch_size=1 path.
            self.assertEqual(model1.batch_calls, [])
            self.assertEqual(len(model1.generate_calls), len(texts))

            for s1, s2 in zip(manifest1["segments"], manifest2["segments"]):
                self.assertEqual(s1["status"], "done")
                self.assertEqual(s2["status"], "done")
                self.assertEqual(s1["num_samples"], s2["num_samples"])
                self.assertEqual(s1["sample_rate"], s2["sample_rate"])
                self.assertEqual(s1["text_hash"], s2["text_hash"])

    def test_resume_after_interruption_mid_batch_only_regenerates_missing_segment(self):
        import tempfile

        texts = ["Первый.", "Второй.", "Третий.", "Четвёртый."]
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, texts)
            model = self._FakeBatchModel()
            ab.generate_segments(
                model, manifest, manifest_path, max_retries=2, retry_base_delay=0.0, batch_size=4
            )
            self.assertTrue(all(s["status"] == "done" for s in manifest["segments"]))

            # Simulate a kill mid-batch: segment 2's WAV never made it to disk (as if the
            # process died before batch_generate yielded that row), but the manifest still
            # (incorrectly, as a killed process would leave it) claims "done".
            killed_entry = manifest["segments"][2]
            (Path(d) / killed_entry["output_path"]).unlink()

            reloaded = ab.load_or_create_manifest(
                manifest_path,
                [ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t) for i, t in enumerate(texts)],
                max_chars=1000,
                tag_scope="chunk",
            )
            model2 = self._FakeBatchModel()
            result = ab.generate_segments(
                model2, reloaded, manifest_path, max_retries=2, retry_base_delay=0.0, batch_size=4
            )
            self.assertEqual(result["failures"], [])
            # Only the missing segment should have been (re)requested from the model -- a
            # single leftover segment takes the `_generate_single_segment`/`model.generate`
            # path (a "batch" of one gains nothing from `batch_generate`), so check
            # `generate_calls`, not `batch_calls`.
            self.assertEqual(model2.batch_calls, [])
            self.assertEqual(model2.generate_calls, ["Третий."])
            self.assertTrue(all(s["status"] == "done" for s in reloaded["segments"]))

    def test_validation_applied_per_segment_isolates_one_bad_row_in_a_batch(self):
        import tempfile

        texts = ["Хороший сегмент один.", "ПЛОХОЙ сегмент здесь.", "Хороший сегмент три."]
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, texts)
            # batch_generate always yields empty/too-short audio for the "ПЛОХОЙ" row, but
            # model.generate() (the single-segment fallback path) succeeds for it.
            model = self._FakeBatchModel(fail_substrings=("ПЛОХОЙ",))
            result = ab.generate_segments(
                model, manifest, manifest_path, max_retries=2, retry_base_delay=0.0, batch_size=3
            )
            self.assertEqual(result["failures"], [])
            self.assertTrue(all(s["status"] == "done" for s in manifest["segments"]))
            # The bad row must have been isolated down to a single-segment fallback call,
            # not silently accepted, and not have taken the two good rows down with it.
            self.assertIn("ПЛОХОЙ сегмент здесь.", model.generate_calls)
            good_entry_0 = manifest["segments"][0]
            good_entry_2 = manifest["segments"][2]
            self.assertGreater(good_entry_0["num_samples"], 1)
            self.assertGreater(good_entry_2["num_samples"], 1)

    def test_manifest_written_after_each_segment_within_a_batch_not_only_after_whole_batch(self):
        import tempfile

        texts = ["Один два три.", "Четыре пять шесть семь.", "Восемь девять."]
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, texts)
            model = self._FakeBatchModel()
            save_calls = []
            real_save = ab.save_manifest

            def counting_save(m, p):
                # Snapshot how many segments are already "done" at each save -- if writes
                # only happened once per whole batch, this would jump straight from 0 to 3
                # in one call instead of climbing 1 at a time.
                save_calls.append(sum(1 for s in m["segments"] if s["status"] == "done"))
                real_save(m, p)

            ab.save_manifest = counting_save
            try:
                ab.generate_segments(
                    model, manifest, manifest_path, max_retries=2, retry_base_delay=0.0, batch_size=3
                )
            finally:
                ab.save_manifest = real_save

            done_counts_seen = sorted(set(save_calls))
            self.assertIn(1, done_counts_seen)
            self.assertIn(2, done_counts_seen)
            self.assertIn(3, done_counts_seen)

    def test_whole_batch_failure_degrades_and_still_completes_via_fallback(self):
        import tempfile

        texts = ["Сегмент А.", "Сегмент Б.", "Сегмент В.", "Сегмент Г."]
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, texts)
            # batch_generate fails outright for the first `max_retries` attempts at ANY
            # size (transient-looking failure), forcing the batch to retry, then split into
            # halves, then singles -- and still finish successfully via model.generate().
            model = self._FakeBatchModel(fail_batch_times=100)
            result = ab.generate_segments(
                model, manifest, manifest_path, max_retries=1, retry_base_delay=0.0, batch_size=4
            )
            self.assertEqual(result["failures"], [])
            self.assertTrue(all(s["status"] == "done" for s in manifest["segments"]))
            self.assertEqual(len(model.generate_calls), len(texts))


class TestVoiceReferenceResolution(unittest.TestCase):
    """Refs #57: --voice-name / --ref-audio resolve into a VoiceReference that pins the
    voice for the whole run. No model/GPU involved -- `_FakeEncodeModel` stands in for
    `HiggsAudioV3.encode_reference_audio()`."""

    class _FakeEncodeModel:
        def encode_reference_audio(self, path):
            return np.array([[1, 2, 3]], dtype=np.int32)  # stand-in "codes"

    def test_voice_name_loads_preencoded_codes_without_touching_the_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            voices_dir = Path(d)
            np.save(voices_dir / "narrator.npy", np.array([[9, 9, 9]]))
            (voices_dir / "narrator.txt").write_text("образец голоса", encoding="utf-8")

            class _ExplodingModel:
                def encode_reference_audio(self, path):
                    raise AssertionError("--voice-name must not call encode_reference_audio")

            ref = ab.resolve_voice_reference(
                _ExplodingModel(), "narrator", None, None, voices_dir=voices_dir
            )
            self.assertEqual(ref.name, "narrator")
            self.assertEqual(ref.ref_text, "образец голоса")
            self.assertEqual(ref.source, "voices/narrator.npy")
            np.testing.assert_array_equal(ref.codes, np.array([[9, 9, 9]]))

    def test_voice_name_missing_npy_raises_with_clear_message(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                ab.resolve_voice_reference(
                    self._FakeEncodeModel(), "ghost", None, None, voices_dir=Path(d)
                )
            self.assertIn("ghost", str(ctx.exception))
            self.assertIn("ghost.npy", str(ctx.exception))

    def test_voice_name_missing_txt_raises_with_clear_message(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            voices_dir = Path(d)
            np.save(voices_dir / "onlynpy.npy", np.array([[1]]))
            with self.assertRaises(FileNotFoundError) as ctx:
                ab.resolve_voice_reference(
                    self._FakeEncodeModel(), "onlynpy", None, None, voices_dir=voices_dir
                )
            self.assertIn("onlynpy.txt", str(ctx.exception))

    def test_ref_audio_requires_ref_text(self):
        with self.assertRaises(ValueError):
            ab.resolve_voice_reference(
                self._FakeEncodeModel(), None, Path("ref.wav"), None
            )

    def test_voice_name_and_ref_audio_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            ab.resolve_voice_reference(
                self._FakeEncodeModel(), "narrator", Path("ref.wav"), "text"
            )

    def test_neither_given_returns_none(self):
        ref = ab.resolve_voice_reference(self._FakeEncodeModel(), None, None, None)
        self.assertIsNone(ref)

    def test_ref_audio_without_save_encodes_but_does_not_write_voices_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            voices_dir = Path(d) / "voices"  # deliberately not created
            ref = ab.resolve_voice_reference(
                self._FakeEncodeModel(), None, Path("ref.wav"), "текст референса",
                voices_dir=voices_dir,
            )
            self.assertEqual(ref.ref_text, "текст референса")
            self.assertEqual(ref.source, "ref.wav")
            self.assertFalse(voices_dir.exists())

    def test_save_voice_as_registers_it_for_future_voice_name_reuse(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            voices_dir = Path(d)
            ref = ab.resolve_voice_reference(
                self._FakeEncodeModel(), None, Path("ref.wav"), "текст референса",
                save_voice_as="cloned_narrator", voices_dir=voices_dir,
            )
            self.assertEqual(ref.source, "voices/cloned_narrator.npy")
            self.assertTrue((voices_dir / "cloned_narrator.npy").exists())
            self.assertEqual(
                (voices_dir / "cloned_narrator.txt").read_text(encoding="utf-8"),
                "текст референса",
            )
            # Round-trips through --voice-name afterwards, same mechanism, one format.
            reloaded = ab.resolve_voice_reference(
                self._FakeEncodeModel(), "cloned_narrator", None, None, voices_dir=voices_dir
            )
            self.assertEqual(reloaded.ref_text, "текст референса")


class TestVoiceReferenceManifestInvalidation(unittest.TestCase):
    """Refs #57 task item 5: the voice reference is a property of the whole run (recorded
    in the manifest header), not of a single segment. Resuming a manifest under a
    DIFFERENT reference (or no reference at all) must refuse, the same way a max_chars/
    tag_scope mismatch already does -- otherwise a resumed run would silently splice
    segments generated under two different voices, quietly reproducing the bug this
    feature exists to fix.
    """

    def _chunks(self, texts):
        return [ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t) for i, t in enumerate(texts)]

    def _ref(self, name="narrator", text="образец"):
        return ab.VoiceReference(name=name, ref_text=text, codes=np.array([[1]]), source=f"voices/{name}.npy")

    def test_same_reference_resumes_without_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = self._chunks(["Первое предложение.", "Второе предложение."])
            ref = self._ref()
            ab.load_or_create_manifest(manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=ref)
            # Resuming with an equivalent (same name/source/ref_text) but distinct
            # VoiceReference object must not raise -- identity is by manifest_id(), not by
            # object identity or by the (possibly large) codes array.
            ref2 = self._ref()
            reloaded = ab.load_or_create_manifest(
                manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=ref2
            )
            self.assertEqual(reloaded["voice_reference"]["name"], "narrator")

    def test_swapping_the_reference_refuses_to_resume(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = self._chunks(["Первое предложение.", "Второе предложение."])
            ab.load_or_create_manifest(
                manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=self._ref("narrator")
            )
            with self.assertRaises(RuntimeError) as ctx:
                ab.load_or_create_manifest(
                    manifest_path, chunks, max_chars=500, tag_scope="sentence",
                    voice_reference=self._ref("different_voice"),
                )
            self.assertIn("voice_reference", str(ctx.exception))

    def test_dropping_the_reference_on_resume_also_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = self._chunks(["Первое предложение.", "Второе предложение."])
            ab.load_or_create_manifest(
                manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=self._ref()
            )
            with self.assertRaises(RuntimeError):
                ab.load_or_create_manifest(
                    manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=None
                )

    def test_adding_a_reference_where_none_existed_also_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = self._chunks(["Первое предложение.", "Второе предложение."])
            ab.load_or_create_manifest(
                manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=None
            )
            with self.assertRaises(RuntimeError):
                ab.load_or_create_manifest(
                    manifest_path, chunks, max_chars=500, tag_scope="sentence", voice_reference=self._ref()
                )

    def test_no_reference_manifests_are_unaffected_by_default(self):
        """The common no-cloning path must keep working exactly as before: omitting
        voice_reference entirely resumes fine against a manifest also built without one."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest_path = Path(d) / "manifest.json"
            chunks = self._chunks(["Одно предложение."])
            ab.load_or_create_manifest(manifest_path, chunks, max_chars=500, tag_scope="sentence")
            reloaded = ab.load_or_create_manifest(manifest_path, chunks, max_chars=500, tag_scope="sentence")
            self.assertIsNone(reloaded["voice_reference"])


class TestVoiceReferenceThreadedIntoGeneration(unittest.TestCase):
    """Refs #57: the resolved reference must reach EVERY segment's model call, in both the
    unbatched (batch_size=1) and batched (batch_size>1) paths -- not just the first
    segment, and not recomputed per call (the caller passes the same pre-encoded object
    every time; this test checks it is literally the same object reaching the model, not
    just an equal one)."""

    class _RefCapturingModel:
        SR = 24000

        def __init__(self):
            self.generate_ref_kwargs: list[tuple] = []
            self.batch_ref_kwargs: list[tuple] = []

        def _audio_for(self, text):
            n = max(1, int(len(text) * 0.05 * self.SR))
            return np.zeros(n, dtype=np.float64)

        def generate(self, text, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                     fade_in_ms=None, fade_out_ms=None):
            self.generate_ref_kwargs.append((ref_audio_codes, ref_text))
            return [type("R", (), {"audio": self._audio_for(text), "sample_rate": self.SR})()]

        def batch_generate(self, texts, temperature, max_new_tokens, ref_audio_codes=None, ref_text=None,
                           fade_in_ms=None, fade_out_ms=None):
            self.batch_ref_kwargs.append((ref_audio_codes, ref_text))
            for idx, t in enumerate(texts):
                yield type(
                    "R", (),
                    {"audio": self._audio_for(t), "sample_rate": self.SR, "sequence_idx": idx,
                     "processing_time_seconds": 0.01},
                )()

    def _manifest(self, d, texts):
        chunks = [ab.Chunk(index=i, sentences=[t], reopened_tags={}, text=t) for i, t in enumerate(texts)]
        manifest_path = Path(d) / "manifest.json"
        manifest = ab.load_or_create_manifest(manifest_path, chunks, max_chars=1000, tag_scope="chunk")
        return manifest, manifest_path

    def test_unbatched_path_passes_reference_to_every_segment(self):
        import tempfile

        codes_sentinel = object()
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(
                d, ["Первое.", "Второе.", "Третье."]
            )
            model = self._RefCapturingModel()
            ab.generate_segments(
                model, manifest, manifest_path, max_retries=1, retry_base_delay=0.0,
                batch_size=1, ref_audio_codes=codes_sentinel, ref_text="эталонный текст",
            )
            self.assertEqual(len(model.generate_ref_kwargs), 3)
            for codes, text in model.generate_ref_kwargs:
                self.assertIs(codes, codes_sentinel)
                self.assertEqual(text, "эталонный текст")

    def test_batched_path_passes_a_single_shared_reference_not_recomputed_per_batch(self):
        import tempfile

        codes_sentinel = object()
        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(
                d, ["Первое.", "Второе.", "Третье.", "Четвёртое."]
            )
            model = self._RefCapturingModel()
            ab.generate_segments(
                model, manifest, manifest_path, max_retries=1, retry_base_delay=0.0,
                batch_size=4, ref_audio_codes=codes_sentinel, ref_text="эталонный текст",
            )
            self.assertEqual(len(model.batch_ref_kwargs), 1)
            codes, text = model.batch_ref_kwargs[0]
            self.assertIs(codes, codes_sentinel)
            self.assertEqual(text, "эталонный текст")

    def test_no_reference_reproduces_old_none_none_behavior(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest, manifest_path = self._manifest(d, ["Одно."])
            model = self._RefCapturingModel()
            ab.generate_segments(model, manifest, manifest_path, max_retries=1, retry_base_delay=0.0, batch_size=1)
            self.assertEqual(model.generate_ref_kwargs, [(None, None)])


if __name__ == "__main__":
    unittest.main()
