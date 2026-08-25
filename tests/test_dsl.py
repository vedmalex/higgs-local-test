#!/usr/bin/env python3
"""Tests for the audiobook markup DSL (issue #114, stage E1): scripts/audiobook_dsl.py
and its `.abs` -> screenplay-JSON compiler, canonical round-trip, coverage, and budget
checks.

Run with the project's `.venv-tts` interpreter (needs numpy, for src/audiobook.py):
    .venv-tts/bin/python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audiobook_dsl as dsl  # noqa: E402
import audiobook as ab  # noqa: E402

E0_ABS = ROOT / "samples" / "audiobook" / "prepared" / "chapter-e0-narration.abs"
E0_TXT = ROOT / "samples" / "audiobook" / "prepared" / "chapter-e0-narration.txt"
E0_MANIFEST = ROOT / "output" / "chapter-114-e0" / "manifest.json"


class TestBlockParsing(unittest.TestCase):
    """All four directive types + all three content block types."""

    def test_prose_block_default_speaker_is_narrator(self):
        doc = dsl.parse_dsl("#prose\nПривет мир.\n")
        self.assertEqual(len(doc.blocks), 1)
        b = doc.blocks[0]
        self.assertEqual(b.unit, "prose")
        self.assertEqual(b.speaker, "narrator")
        self.assertEqual(b.attrs, {})

    def test_say_requires_speaker(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#say\nтекст\n")

    def test_say_parses_speaker_and_attrs(self):
        doc = dsl.parse_dsl("#say maya emotion=sadness\nО сын Кунти.\n")
        b = doc.blocks[0]
        self.assertEqual(b.speaker, "maya")
        self.assertEqual(b.attrs, {"emotion": "sadness"})

    def test_recite_speaker_optional_defaults_to_narrator(self):
        doc = dsl.parse_dsl("#recite\nСтрока один\nСтрока два\n")
        self.assertEqual(doc.blocks[0].speaker, "narrator")

    def test_recite_with_explicit_speaker(self):
        doc = dsl.parse_dsl("#recite arjuna\nСтрока один\nСтрока два\n")
        self.assertEqual(doc.blocks[0].speaker, "arjuna")

    def test_note_is_dropped_and_not_a_block(self):
        doc = dsl.parse_dsl("#prose\nТекст.\n\n#note сверить эпитет\n")
        self.assertEqual(len(doc.blocks), 1)
        self.assertEqual(doc.notes, [(4, "сверить эпитет")])

    def test_chapter_and_scene_are_metadata_not_blocks(self):
        doc = dsl.parse_dsl(
            "#chapter 2.1 Название\n#scene Дворец собраний\n\n#prose\nТекст.\n"
        )
        self.assertEqual(doc.chapter, "2.1 Название")
        self.assertEqual(len(doc.blocks), 1)
        self.assertEqual(doc.blocks[0].scene, "Дворец собраний")

    def test_scene_changes_mid_document_apply_only_after(self):
        doc = dsl.parse_dsl(
            "#scene Первая\n\n#prose\nА.\n\n#scene Вторая\n\n#prose\nБ.\n"
        )
        self.assertEqual(doc.blocks[0].scene, "Первая")
        self.assertEqual(doc.blocks[1].scene, "Вторая")

    def test_missing_blank_line_between_blocks_is_an_error_with_line_number(self):
        text = "#prose\nТекст один.\n#prose\nТекст два.\n"
        with self.assertRaises(dsl.DslError) as ctx:
            dsl.parse_dsl(text)
        self.assertEqual(ctx.exception.src_line, 3)

    def test_empty_document_is_an_error(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("\n\n")

    def test_text_outside_any_block_is_an_error(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("Просто текст без директивы.\n")

    def test_unknown_directive_is_an_error(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#verse\nтекст\n")

    def test_prose_rejects_a_speaker_token(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#prose narrator\nтекст\n")


class TestAttributeCompilation(unittest.TestCase):
    def test_attrs_compile_to_leading_tags_in_fixed_order(self):
        doc = dsl.parse_dsl("#say maya style=whispering emotion=sadness\nТекст.\n")
        segs = dsl.compile_document(doc)
        self.assertEqual(segs[0]["text"], "<|emotion:sadness|><|style:whispering|>Текст.")

    def test_unknown_tag_value_raises_with_line_number(self):
        doc = dsl.parse_dsl("#say maya emotion=notatag\nТекст.\n")
        with self.assertRaises(dsl.DslError) as ctx:
            dsl.compile_document(doc)
        self.assertEqual(ctx.exception.src_line, 1)

    def test_unknown_attribute_key_raises(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#say maya volume=loud\nТекст.\n")

    def test_repeated_attribute_key_raises(self):
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#say maya emotion=sadness emotion=anger\nТекст.\n")

    def test_env_tag_rejected_as_attribute(self):
        # env:* is not a real category in the attribute grammar (emotion/prosody/style
        # only), so this is caught as an unknown attribute key -- still a hard error.
        with self.assertRaises(dsl.DslError):
            dsl.parse_dsl("#say maya env=music\nТекст.\n")

    def test_raw_env_tag_inline_is_a_compile_error(self):
        doc = dsl.parse_dsl("#prose\nТекст <|env:music|> продолжение.\n")
        with self.assertRaises(dsl.DslError) as ctx:
            dsl.compile_document(doc)
        self.assertIn("env:*/chatml", str(ctx.exception))

    def test_raw_chatml_tag_inline_is_a_compile_error(self):
        doc = dsl.parse_dsl("#prose\nТекст <|chatml|> продолжение.\n")
        with self.assertRaises(dsl.DslError):
            dsl.compile_document(doc)

    def test_raw_valid_tag_inline_passes_through(self):
        doc = dsl.parse_dsl("#prose\nТекст <|emotion:awe|> продолжение.\n")
        segs = dsl.compile_document(doc)
        self.assertIn("<|emotion:awe|>", segs[0]["text"])


class TestPauseSugar(unittest.TestCase):
    def test_pause_sugar_compiles_to_tag(self):
        doc = dsl.parse_dsl("#prose\nТекст [пауза] продолжение.\n")
        segs = dsl.compile_document(doc)
        self.assertEqual(segs[0]["text"], "Текст <|prosody:pause|> продолжение.")

    def test_long_pause_sugar_compiles_to_tag(self):
        doc = dsl.parse_dsl("#prose\nТекст [долгая пауза] продолжение.\n")
        segs = dsl.compile_document(doc)
        self.assertEqual(segs[0]["text"], "Текст <|prosody:long_pause|> продолжение.")

    def test_unrecognized_bracket_marker_is_an_error(self):
        doc = dsl.parse_dsl("#prose\nТекст [неизвестно] продолжение.\n")
        with self.assertRaises(dsl.DslError):
            dsl.compile_document(doc)


class TestReciteCompilation(unittest.TestCase):
    def test_pause_at_end_of_each_line_long_pause_at_end_of_stanza(self):
        doc = dsl.parse_dsl(
            "#recite\nСтрока один\nСтрока два\nСтрока три\n"
        )
        segs = dsl.compile_document(doc)
        self.assertEqual(
            segs[0]["text"],
            "Строка один<|prosody:pause|> Строка два<|prosody:pause|> "
            "Строка три<|prosody:long_pause|>",
        )

    def test_trailing_backslash_suppresses_auto_pause(self):
        doc = dsl.parse_dsl("#recite\nСтрока один\\\nСтрока два\n")
        segs = dsl.compile_document(doc)
        self.assertEqual(
            segs[0]["text"], "Строка один Строка два<|prosody:long_pause|>"
        )

    def test_explicit_long_pause_sugar_mid_stanza_not_doubled(self):
        doc = dsl.parse_dsl(
            "#recite\nСтрока один [долгая пауза]\nСтрока два\n"
        )
        segs = dsl.compile_document(doc)
        self.assertEqual(
            segs[0]["text"],
            "Строка один <|prosody:long_pause|> Строка два<|prosody:long_pause|>",
        )

    def test_recite_rejects_speed_attribute(self):
        with self.assertRaises(dsl.DslError):
            doc = dsl.parse_dsl("#recite arjuna prosody=speed_slow\nСтрока один\nСтрока два\n")
            dsl.compile_document(doc)

    def test_recite_rejects_inline_speed_tag(self):
        doc = dsl.parse_dsl(
            "#recite\nСтрока <|prosody:speed_slow|> один\nСтрока два\n"
        )
        with self.assertRaises(dsl.DslError):
            dsl.compile_document(doc)

    def test_contradictory_suppress_and_explicit_pause_is_an_error(self):
        doc = dsl.parse_dsl(
            "#recite\nСтрока один [долгая пауза]\\\nСтрока два\n"
        )
        with self.assertRaises(dsl.DslError):
            dsl.compile_document(doc)

    def test_recite_lines_glued_into_one_json_sentence(self):
        doc = dsl.parse_dsl("#recite\nА\nБ\nВ\n")
        segs = dsl.compile_document(doc)
        self.assertEqual(len(segs), 1)
        sentences = ab.split_sentences(segs[0]["text"])
        # split_sentences must not carve the glued stanza back into micro-sentences.
        self.assertEqual(len(sentences), 1)


class TestSpeakerAndAttributesSurviveChunkBoundaries(unittest.TestCase):
    """Owner's listening-test finding (Э0 chapter.wav): a #say reply must keep its
    speaker AND its attribute across engine chunk boundaries, not just in its first
    chunk. chunk_screenplay already reopens per-sentence state and copies `speaker`
    onto every chunk cut from one line -- this test exercises that through the DSL
    compiler, not just the raw engine function, so a regression in either layer is
    caught."""

    def test_long_say_reply_keeps_speaker_and_emotion_in_every_chunk(self):
        long_reply = (
            "О главный среди святых царей династии Панду, строго следующих Самому "
            "Господу Шри Кришне! Нет ничего удивительного, что ты отказываешься от "
            "своего трона, украшенного шлемами многих царей, ради того, чтобы обрести "
            "возможность вечно общаться с Личностью Бога. Мы будем ждать здесь, пока "
            "самый выдающийся преданный Господа, Махараджа Парикшит, не вернется на "
            "верховную планету, совершенно свободную от всех видов материальной "
            "скверны и скорби."
        )
        abs_text = (
            "#prose\n"
            "Все великие мудрецы выразили свое одобрение.\n\n"
            f"#say mudretsy emotion=contentment\n{long_reply}\n"
        )
        doc, segs = dsl.compile_source(abs_text)
        lines = ab.parse_screenplay(
            [{"speaker": s["speaker"], "text": s["text"]} for s in segs]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = ab.chunk_screenplay(lines, max_chars=200, tag_scope="sentence")

        say_chunks = [c for c in chunks if c.speaker == "mudretsy"]
        self.assertGreater(
            len(say_chunks), 1, "test reply must actually need more than one chunk"
        )
        for c in say_chunks:
            self.assertEqual(c.speaker, "mudretsy")
            self.assertIn("<|emotion:contentment|>", c.text)

    def test_prose_to_say_transition_changes_speaker_field(self):
        """assemble_chapter's speaker_change_silence_ms branch keys off this field
        differing between consecutive segments -- verify the DSL actually produces
        that difference across a #prose -> #say boundary."""
        abs_text = "#prose\nПовествование.\n\n#say maya\nРеплика.\n"
        _doc, segs = dsl.compile_source(abs_text)
        speakers = [s["speaker"] for s in segs]
        self.assertEqual(speakers, ["narrator", "maya"])


class TestOwnerDecisionsE0Followup(unittest.TestCase):
    """dsl-spec.md sec. 2.8/2.9, sec. 7: owner decisions on the two Э0 listening-test
    open questions (resolved 2026-08-25, Refs #114) -- attribution clauses stay with the
    narrator, and a short embedded quote gets an audible pause via the existing inline
    sugar, not a new block type."""

    def test_attribution_clause_stays_in_prose_reply_starts_say(self):
        """dsl-spec.md sec. 2.8 worked example, the owner's own sentence."""
        abs_text = (
            "#prose\nМудрецы сказали:\n\n"
            "#say mudretsy\nО главный среди святых царей династии Панду.\n"
        )
        _doc, segs = dsl.compile_source(abs_text)
        self.assertEqual(segs[0]["speaker"], "narrator")
        self.assertEqual(segs[0]["text"], "Мудрецы сказали:")
        self.assertEqual(segs[1]["speaker"], "mudretsy")
        self.assertEqual(
            segs[1]["text"], "О главный среди святых царей династии Панду."
        )

    def test_short_quote_pause_uses_existing_sugar_no_new_block_type(self):
        """dsl-spec.md sec. 2.9: the owner's exact flagged sentence, wrapped in the
        existing [пауза] sugar on both sides of the quote, inside one #prose block --
        no #say, no speaker change, no new syntax."""
        abs_text = (
            "#prose\n"
            "Все великие мудрецы, собравшиеся там, восторженно приняли решение "
            "Махараджи Парикшита и выразили свое одобрение словами: [пауза] "
            "«Очень хорошо!» [пауза]\n"
        )
        _doc, segs = dsl.compile_source(abs_text)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["speaker"], "narrator")
        self.assertEqual(
            segs[0]["text"],
            "Все великие мудрецы, собравшиеся там, восторженно приняли решение "
            "Махараджи Парикшита и выразили свое одобрение словами: "
            "<|prosody:pause|> «Очень хорошо!» <|prosody:pause|>",
        )

    def test_short_quote_pause_canonical_round_trip(self):
        abs_text = (
            "#prose\nСлова: [пауза] «Очень хорошо!» [пауза] и тишина.\n"
        )
        canon = dsl.strip_markup(abs_text)
        self.assertEqual(canon, "Слова: «Очень хорошо!» и тишина.")

    def test_short_quote_pause_passes_coverage_and_lint(self):
        abs_text = (
            "#prose\nСлова: [пауза] «Очень хорошо!» [пауза] и тишина.\n"
        )
        self.assertEqual(dsl.collect_lint_issues(abs_text), [])
        _doc, segs = dsl.compile_source(abs_text)
        canon = dsl.strip_markup(abs_text)

        def norm(t):
            import re

            t = dsl.TAG_SPAN_RE.sub(" ", t)
            t = dsl._strip_stress_apostrophes(t)
            return re.sub(r"\s+", " ", t).strip().split(" ")

        self.assertEqual(norm(canon), norm("\n\n".join(s["text"] for s in segs)))


class TestCanonicalRoundTrip(unittest.TestCase):
    def test_simple_document_round_trips_to_plain_text(self):
        abs_text = (
            "#chapter Заголовок\n#scene Комната\n\n"
            "#prose\nТогда могу'щественный Майя обратился к Арджуне.\n\n"
            "#say maya emotion=sadness\nО сын Кунти. [пауза] Позволь мне помочь.\n\n"
            "#note проверить позже\n"
        )
        canon = dsl.strip_markup(abs_text)
        self.assertEqual(
            canon,
            "Тогда могущественный Майя обратился к Арджуне.\n\n"
            "О сын Кунти. Позволь мне помочь.",
        )

    def test_recite_backslash_and_pauses_stripped(self):
        abs_text = "#recite\nСтрока один\\\nСтрока два [долгая пауза]\n"
        canon = dsl.strip_markup(abs_text)
        self.assertEqual(canon, "Строка один\nСтрока два")

    def test_stress_apostrophe_stripped_by_reused_heuristic(self):
        abs_text = "#prose\nза'мок стоит на холме.\n"
        canon = dsl.strip_markup(abs_text)
        self.assertEqual(canon, "замок стоит на холме.")
        # Same heuristic as the engine -- confirm they agree, not just this module.
        self.assertTrue(ab.is_stress_apostrophe("за'мок", 2))

    def test_round_trip_fails_loudly_on_edited_text(self):
        """A markup author silently rewriting a sentence must fail the invariant."""
        abs_text = "#prose\nОригинальный текст главы.\n"
        canon = dsl.strip_markup(abs_text)
        tampered = "Подделанный текст главы."
        self.assertNotEqual(canon, tampered)

    def test_round_trip_fails_on_lowercase_both_sides_name_apostrophe(self):
        """Documented blind spot of is_stress_apostrophe (src/audiobook.py): a name
        apostrophe with lowercase letters on both sides is indistinguishable from a
        real stress mark, so it gets stripped here too -- the round trip must then
        NOT equal a canonical text that still has that apostrophe, surfacing the
        ambiguity loudly instead of silently mismarking a name."""
        # "невеста'льба" is a nonsense stand-in built only to have lowercase-lowercase
        # around the apostrophe, exactly the shape the heuristic cannot resolve.
        abs_text = "#prose\nэто невеста'льба и ничего больше.\n"
        canon = dsl.strip_markup(abs_text)
        original_with_apostrophe = "это невеста'льба и ничего больше."
        self.assertNotEqual(canon, original_with_apostrophe)
        self.assertEqual(canon, "это невестальба и ничего больше.")


class TestCoverage(unittest.TestCase):
    def test_full_coverage_no_remainder(self):
        abs_text = "#prose\nПервый абзац.\n\n#say maya\nВторой абзац реплики.\n"
        _doc, segs = dsl.compile_source(abs_text)
        canon = dsl.strip_markup(abs_text)

        def norm(t):
            import re

            t = dsl.TAG_SPAN_RE.sub(" ", t)
            t = dsl._strip_stress_apostrophes(t)
            return re.sub(r"\s+", " ", t).strip().split(" ")

        self.assertEqual(norm(canon), norm("\n\n".join(s["text"] for s in segs)))

    def test_dropped_paragraph_is_detected(self):
        """Simulates the suno-music-producer failure mode: a fragment silently
        missing from the compiled JSON while the .abs file itself looks fine on a
        cursory read. Here we simulate a compiler that drops a block."""
        abs_text = "#prose\nПервый абзац с важными словами.\n\n#prose\nВторой абзац.\n"
        _doc, segs = dsl.compile_source(abs_text)
        canon = dsl.strip_markup(abs_text)
        # Drop the first segment, as a buggy compiler might.
        broken_segs = segs[1:]

        def norm(t):
            import re

            t = dsl.TAG_SPAN_RE.sub(" ", t)
            return re.sub(r"\s+", " ", t).strip().split(" ")

        canon_words = norm(canon)
        recovered_words = norm("\n\n".join(s["text"] for s in broken_segs))
        self.assertNotEqual(canon_words, recovered_words)
        missing = set(canon_words) - set(recovered_words)
        self.assertIn("Первый", missing)


class TestLint(unittest.TestCase):
    def test_valid_document_has_no_issues(self):
        abs_text = "#prose\nТекст без проблем.\n"
        self.assertEqual(dsl.collect_lint_issues(abs_text), [])

    def test_unknown_tag_reported_with_line_number(self):
        abs_text = "#say maya emotion=notreal\nТекст.\n"
        issues = dsl.collect_lint_issues(abs_text)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].src_line, 1)

    def test_multiple_issues_all_collected_not_fail_fast(self):
        abs_text = (
            "#say maya emotion=notreal\nТекст один.\n\n"
            "#recite arjuna prosody=speed_slow\nСтрока один\nСтрока два\n"
        )
        issues = dsl.collect_lint_issues(abs_text)
        self.assertEqual(len(issues), 2)

    def test_stray_backslash_outside_recite_flagged(self):
        abs_text = "#prose\nТекст с обратным слэшем\\\n"
        issues = dsl.collect_lint_issues(abs_text)
        self.assertEqual(len(issues), 1)
        self.assertIn("only #recite", issues[0].message)


class TestBudget(unittest.TestCase):
    def test_estimate_scales_with_character_count(self):
        segs = [
            {"speaker": "narrator", "text": "а" * 100, "unit": "prose", "scene": None, "src_line": 1}
        ]
        report = dsl.estimate_budget(segs)
        self.assertEqual(report["total_chars"], 100)
        self.assertAlmostEqual(
            report["estimated_generation_seconds_no_batch"],
            100 * dsl.SECONDS_PER_CHAR * dsl.RTF_NO_BATCH,
            places=1,
        )

    def test_short_recite_segment_flagged(self):
        segs = [
            {"speaker": "narrator", "text": "Коротко.", "unit": "recite", "scene": None, "src_line": 1}
        ]
        report = dsl.estimate_budget(segs, min_recite_chars=40)
        self.assertEqual(len(report["short_recite_segments"]), 1)

    def test_long_recite_segment_not_flagged(self):
        segs = [
            {
                "speaker": "narrator",
                "text": "а" * 60,
                "unit": "recite",
                "scene": None,
                "src_line": 1,
            }
        ]
        report = dsl.estimate_budget(segs, min_recite_chars=40)
        self.assertEqual(report["short_recite_segments"], [])


@unittest.skipUnless(
    E0_ABS.exists() and E0_TXT.exists() and E0_MANIFEST.exists(),
    "Э0 fixtures/manifest not present in this checkout",
)
class TestE0HashReproduction(unittest.TestCase):
    """Stage E1 acceptance check (issue #114): does compiling the Э0 chapter's own
    narration text through the DSL, then through the *unmodified* engine
    (`parse_screenplay`/`chunk_screenplay`), reproduce the Э0 manifest's per-segment
    content hashes?

    Measured, honest answer: NO, not fully -- see the class docstring below the
    assertions for why, and docs/research/audiobook/e1-dsl-hash-reproduction.md for the
    full investigation. This test pins the actual measured overlap so a future change
    cannot silently regress it further without the test noticing; it is not a "success"
    assertion.
    """

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(E0_MANIFEST.read_text(encoding="utf-8"))
        cls.abs_text = E0_ABS.read_text(encoding="utf-8")

    def test_canonical_and_coverage_pass_for_the_e0_fixture(self):
        canon = dsl.strip_markup(self.abs_text)
        expected = E0_TXT.read_text(encoding="utf-8")
        expected_cmp = expected if expected.endswith("\n") else expected + "\n"
        canon_cmp = canon if canon.endswith("\n") else canon + "\n"
        self.assertEqual(canon_cmp, expected_cmp)

    def test_hash_overlap_with_e0_manifest_is_partial_not_total(self):
        """Root cause (measured, not guessed): the Э0 chapter was generated by feeding
        the WHOLE chapter text as one continuous stream to `chunk_sentences`, which
        freely packs sentences across the original paragraph boundaries up to
        `max_chars`. The DSL's grammar closes a block on a blank line, so a chapter
        with blank-line-separated paragraphs necessarily compiles to one screenplay
        JSON line per paragraph; `chunk_screenplay` (unchanged, by design -- see its
        docstring) chunks each JSON line independently and never merges two lines'
        sentences, even when both are short. 16 of the Э0 manifest's 70 segments were
        confirmed (by locating each chunk's sentences back in the source text) to span
        two original paragraphs -- exactly the kind of merge `chunk_screenplay`'s
        per-line design cannot reproduce. The cascading effect of a merge point
        shifting every following sentence-grouping inside that paragraph brings the
        actual measured hash overlap down to 26/70 (not just the 16 crossing chunks).

        This is a structural mismatch between two fixed, unchangeable pieces (the DSL's
        blank-line-closes-block grammar, and chunk_screenplay's per-line independence)
        -- not a bug in this compiler. See dsl-spec.md and the issue #114 comment for
        the full writeup."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc, segs = dsl.compile_source(self.abs_text)
            lines = ab.parse_screenplay(
                [{"speaker": s["speaker"], "text": s["text"]} for s in segs]
            )
            chunks = ab.chunk_screenplay(lines, max_chars=500, tag_scope="sentence")

        dsl_hashes = {ab._text_hash(ab._segment_hash_input(c)) for c in chunks}
        manifest_hashes = {s["text_hash"] for s in self.manifest["segments"]}

        overlap = dsl_hashes & manifest_hashes
        self.assertEqual(len(self.manifest["segments"]), 70)
        self.assertEqual(len(overlap), 26)


if __name__ == "__main__":
    unittest.main()
