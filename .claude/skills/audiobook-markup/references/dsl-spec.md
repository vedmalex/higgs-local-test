# Audiobook markup DSL v0.1 (issue #114, stage E1)

Normative specification for the `.abs` chapter source format and its compilation into
the existing screenplay JSON that `src/audiobook.py --screenplay-file` already reads.
**The engine (`src/audiobook.py`) is not modified by this DSL in any way.** Everything
below is implemented in `scripts/audiobook_dsl.py` plus its four CLI entry points
(`compile_dsl.py`, `lint_dsl.py`, `check_canonical.py`, `check_coverage.py`) and
`scripts/check_budget.py`.

## 1. Why a JSON hash contract, not a new engine feature

`_segment_hash_input` in `src/audiobook.py` is `speaker + "\x1f" + text` — nothing else.
The engine also documents, and relies on, "extra JSON keys are read but ignored"
(`parse_screenplay` keeps only `speaker`/`text`). That is a ready-made extension slot:
structural metadata (which scene a line belongs to, what kind of block it was, which
source line it came from) can ride along in the JSON without ever touching what gets
hashed or spoken. Editing a scene label or fixing a typo in a `#note` never changes a
segment's hash and therefore never forces a needless regeneration; changing what should
sound different (the words, the speaker, an attribute) always does, because it always
changes `text` or `speaker`.

## 2. File format

One file per chapter, extension `.abs` (a**ttributed** **s**cript). Plain UTF-8 text.

### 2.1 Directive lines

A line whose first character is `#` is a directive. Everything else is body text
belonging to the block currently open. A blank line closes the current block.

| Directive | Body? | Meaning |
|---|---|---|
| `#chapter <title>` | no | Informational chapter title. Never enters the JSON output; purely for a human reading the `.abs` file. |
| `#scene <label>` | no | Sets the "current scene" for every block that follows, until the next `#scene`. Written into each following segment's `scene` key (ignored by the engine, useful for a human/tool browsing the chapter). |
| `#note <text>` | no | A markup editor's comment (e.g. "verify this epithet against ШБ"). Dropped entirely — never in the JSON, never in the canonical text. |
| `#prose` | yes | Narration, speaker fixed to `narrator`. Takes no speaker token, only optional `key=value` attributes. |
| `#say <speaker> [attrs]` | yes | A line of dialogue. `<speaker>` is a single token (no spaces), required. |
| `#recite [<speaker>] [attrs]` | yes | Verse. `<speaker>` is optional (defaults to `narrator` — most verse in this project's source material is narrated, not spoken in character). |

`#chapter`/`#scene`/`#note` never open a block: they are single-line, standalone
directives that take effect immediately and do not consume following body lines.
`#prose`/`#say`/`#recite` open a block whose body is every non-blank line up to (but not
including) the next blank line or end of file. A directive line appearing before that
blank line is a structural error (a missing blank line between two blocks), reported
with its own line number.

### 2.2 Attributes

`emotion=`, `prosody=`, `style=` — at most one of each per block header, `key=value`,
no spaces around `=`. Compiled into leading `<|category:value|>` tags, always emitted in
a fixed order (`emotion`, then `prosody`, then `style`) regardless of the order they were
typed in, so the compiled text is deterministic. Every value is checked against
`VALID_TAGS` in `src/audiobook.py` — imported, never duplicated here — so a typo or an
invented tag fails to compile with the offending source line named. `<|env:*|>` and
`<|chatml|>` (present in the tokenizer's `added_tokens` but undocumented in PROMPTING.md)
are explicitly rejected as compile errors, both as attribute values and as raw inline
tags, even though the engine would eventually reject them too — this catches it before a
multi-hour run, with a line number.

### 2.3 No state inheritance between blocks — by design

There is no syntax for "same speaker/attributes as the previous block." Every block that
needs an attribute repeats it in its own header. This is deliberate, not a missing
feature: it makes the DSL's semantics match `chunk_screenplay`'s exactly — that function
already resets emotion/prosody/style state at the start of every screenplay line (a
different speaker's leftover emotional state has no textual basis), so a DSL that
pretended to inherit state across blocks would be lying about what the engine actually
does.

**Within one block**, state absolutely does carry forward — across as many
generation-call chunks as the block's text needs (`chunk_screenplay` reopens the
active tag before every sentence under the default `tag_scope="sentence"`, and the
`speaker` value is copied onto every chunk cut from that one line). This was verified
directly, not assumed: compiling a `#say` block long enough to be split into three
generation chunks under a small `max_chars` produces three `Chunk`s, all with
`speaker="mudretsy"`, and all three still carrying the leading `<|emotion:contentment|>`
tag — not just the first chunk. **A DSL author does not need to repeat an attribute
mid-reply for it to survive a chunk boundary** — that reopening is `chunk_screenplay`'s
job and already happens automatically. What the author must not do is expect one
`#say`/`#recite` block's attributes to still be active in the *next* block; that reset is
also automatic and also by design (§2.3 above).

### 2.4 Speaker changes and the join pause

Every block's compiled JSON line carries the block's `speaker`. `assemble_chapter`
(unchanged) already reads a `speaker_change_silence_ms` option: when a segment's speaker
differs from the previous segment's, that longer pause is used instead of the normal
`--silence-ms`. Because a `#prose` → `#say` transition genuinely changes `speaker` from
`narrator` to the character's name in the compiled JSON, passing
`--speaker-change-silence-ms` at assembly time *does* produce a longer pause exactly at
that transition — this was checked against `assemble_chapter`'s source, not assumed.
**This is opt-in, not automatic**: without `--speaker-change-silence-ms` on the
`audiobook.py` invocation, every join uses the same `--silence-ms`, narration-to-dialogue
included. A chapter's assembly command needs to pass that flag deliberately if a longer
narration→dialogue pause is wanted.

### 2.5 Inline text and pause sugar

Raw `<|category:tag|>` spans may be written directly in body text and pass through
unchanged (subject to the same `VALID_TAGS` check as attributes). Two pause sugars are
recognized, exact spelling, and nowhere else does `[` or `]` appear in body text — any
other bracketed span is a compile error (most likely a typo of one of these two, or a
leftover markup marker that should have been removed before this text reached the DSL):

| Sugar | Compiles to |
|---|---|
| `[пауза]` | `<|prosody:pause|>` |
| `[долгая пауза]` | `<|prosody:long_pause|>` |

### 2.6 Stress notation

An apostrophe placed directly after the stressed vowel (`за'мок`) is the one
owner-verified stress notation (see `docs/guides/audiobook_guide.md` sec. 5b). The DSL
does not transform it in any way — it reaches the compiled JSON text exactly as typed,
because the engine's own sentence-splitting/tagging code already leaves a bare `'`
untouched. The DSL's only involvement with stress apostrophes is at the *canonical*
layer (§4): `strip_markup` removes exactly the apostrophes `ab.is_stress_apostrophe`
recognizes, using that same function, so the two layers can never disagree about which
apostrophe is a stress mark.

### 2.7 `#recite`: pauses carry the rhythm, not tempo

Per-line rhythm, not per-line tempo: the compiler appends `<|prosody:pause|>` to the end
of every line in a `#recite` block except the last, which gets `<|prosody:long_pause|>`
instead (a stanza's last line is both "end of line" and "end of strophe" — the stronger
pause wins, it is not stacked on top of the ordinary one). `<|prosody:speed_*|>` tags are
rejected inside `#recite`, both as a `prosody=` attribute and as a raw inline tag — the
project owner confirmed by ear that the model's tempo tags are not reliably slower/faster
in a way that helps verse, while pauses are one-shot and positionally safe across chunk
boundaries (`docs/guides/audiobook_guide.md` sec. 5a.1).

All lines of one `#recite` block are joined with a single space into **one** JSON `text`
value (one sentence, as far as `split_sentences` is concerned) specifically so
`split_sentences`/`chunk_sentences` never carve a four-to-six-word verse line into its
own micro-segment — a segment that short risks both the runtime duration-plausibility
floor (`ab.MIN_SECONDS_PER_CHAR`/`MAX_SECONDS_PER_CHAR` in `_validate_generated_audio`)
and an audibly abrupt intonation cutoff.

A line can suppress its automatic pause with a trailing `\`:

```
#recite
Первая строка без паузы\
Вторая строка с обычной паузой
Последняя строка строфы
```

A line can force a stronger pause explicitly instead of accepting the automatic one —
write the tag directly at the line's end (`строка [долгая пауза]` or the raw tag). When a
line already ends with an explicit pause tag, the compiler does not add a second one.
Combining a trailing `\` with an explicit pause tag on the same line is a compile error
(contradictory: one says "no pause here," the other says "pause here").

## 3. Compilation output

Each block compiles to one JSON object:

```json
{"speaker": "narrator", "text": "Тогда могу'щественный Майя...", "unit": "prose", "scene": "Дворец собраний", "src_line": 12}
```

`speaker`/`text` are exactly what `parse_screenplay` keeps and what
`_segment_hash_input` hashes. `unit` (`"prose"`/`"say"`/`"recite"`), `scene`, and
`src_line` are additional keys the current engine reads and silently ignores — adding
them changes nothing about what `chunk_screenplay`, hashing, or generation do with a
line.

## 4. The canonical-text invariant

`strip_markup(source)` (in `scripts/audiobook_dsl.py`, exercised by
`scripts/check_canonical.py`) removes every directive line, all attribute-derived and
raw inline tags, pause sugar, `#recite`'s `\` suppression marker, and every stress
apostrophe (`ab.is_stress_apostrophe`) from a `.abs` file, and must reproduce the
original narration text **byte for byte**. Two things this catches that a purely
structural check would not:

1. **A silently rewritten sentence.** If the canonical text this produces doesn't match
   the chapter's known-good source text, someone changed the words while marking the
   chapter up, not just its performance — the round-trip fails loudly instead of shipping
   a rewritten sentence as if it were the original.
2. **`is_stress_apostrophe`'s own documented blind spot.** A name apostrophe with
   lowercase letters on both sides (a case the heuristic cannot tell apart from a real
   stress mark — see `src/audiobook.py`'s comment above `RUSSIAN_VOWELS_LOWER`) gets
   *stripped* by this same check, so the round-trip fails and a previously silent
   ambiguity becomes a loud, specific one, naming the exact source line.

## 5. Validation contours

Each is a standalone script with a meaningful exit code (0 = pass):

1. **`scripts/check_canonical.py`** — the byte-for-byte invariant above (§4).
2. **`scripts/check_coverage.py`** — completeness. There is no "skipped" category left in
   this DSL (`#verse`/`#gloss`/`skip` were removed — see §6), so this check is strict:
   every canonical paragraph must be represented in the compiled JSON, and after
   removing every covered paragraph's content the remainder must be empty. This is aimed
   specifically at the *compiler's* own completeness (a bug that silently drops a block
   while building JSON from parsed blocks), independently of `check_canonical.py`
   catching the same class of problem at the `.abs`-authoring level — see the suno-music-
   producer skill's `check-stress.py`, which this pattern is carried over from, and its
   documented failure mode (fragments silently dropped while an authenticity check stayed
   green).
3. **`scripts/lint_dsl.py`** — structure, the 43-tag catalog (`env:*`/`chatml` rejected),
   attribute syntax, and `#recite`'s pause-or-suppression rule, every finding reported
   with a source line number, and not fail-fast (one run reports everything wrong with a
   chapter). "Attributes repeated in every block" is not a runtime check here because it
   cannot fail: the grammar has no inheritance syntax to omit them with (§2.3).
4. **`scripts/check_budget.py`** — a character-count-based time estimate using the
   measured RTF constants from `docs/research/audiobook/m4-full-chapter-results.md`
   (`RTF_NO_BATCH = 3.83`, `RTF_BATCH8 = 1.08`) and an empirical seconds-per-character
   ratio computed from the Э0 manifest (`output/chapter-114-e0/manifest.json`:
   `0.0630 s/char`). Also flags any `#recite`-derived segment under
   `DEFAULT_MIN_RECITE_CHARS` (40, a documented heuristic, not a measurement) as still
   risking the short-segment failure mode §2.7 describes despite stanza-gluing.

## 6. What was in an earlier draft and is deliberately not here

`#verse`/`#gloss` block types and a `skip` mechanism were designed in an earlier revision
of this project's plan and removed before implementation: the project owner clarified
that this project's books go through a separate audiobook-preparation pass before
reaching a markup author (see `samples/audiobook/prepare.py` — it already drops
`## Пословный перевод` word-by-word glosses and navigation links), so material that
should never be narrated does not reach `.abs` authoring in the first place. There is no
"skipped" category in this DSL's coverage model as a result — §5's completeness check is
strict, not "strict except for a skip list."

## 7. Open questions for the project owner (not decided here)

These surfaced from listening to the Э0 chapter output
(`output/chapter-114-e0/chapter.wav`) against `samples/audiobook/prepared/sb-1-19.txt`,
which was marked up minimally (no speaker/attribute structure at all, to measure the
pipeline) and as a result reads dialogue and narration in one continuous voice with no
pause between them. The DSL mechanics needed to fix this already exist (`#say`, `#prose`
→ `#say` speaker changes, `--speaker-change-silence-ms`) and were verified working in
this stage (§2.3, §2.4) — what is genuinely undecided is *where to draw the block
boundary* in real source text, which is an editorial call, not a mechanical one:

1. **Does the attribution clause belong to the narrator or to the reply?** Concretely,
   in `Мудрецы сказали: О главный среди святых царей династии Панду, строго следующих
   Самому Господу Шри Кришне!..."`, should `#prose` end (and `#say` begin) before or
   after "Мудрецы сказали:"? Ending `#prose` before it makes the attribution itself
   spoken by the narrator (natural for an audiobook, and it gets the narrator's own
   voice/pace); folding it into the `#say` block makes it part of what the character
   line looks like on the page. Both are mechanically valid `.abs`; nothing in this
   stage's spec or code picks one. **Needs an owner decision before any real chapter is
   marked up**, since it changes where every such clause is split throughout a book.
2. **Short quotes embedded inside narration** — e.g. `...выразили свое одобрение
   словами: «Очень хорошо!»` — are one or two words, not a real turn of dialogue. Forcing
   a `#say` block around something this short seems disproportionate (and this DSL just
   deliberately shed two block types — `#verse`/`#gloss` — rather than gain one without
   need). Whether the DSL should grow a lighter marker for an inline quoted aside, or
   whether this is squarely the audiobook-preparation pass's job (upstream of `.abs`
   authoring, like the gloss/navigation stripping `prepare.py` already does), or whether
   it should simply be left as narrator prose and accepted as a minor imperfection, is an
   open question this stage does not resolve.
