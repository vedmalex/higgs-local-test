# M4: emotion-matched texts for the sentiment survey (issue #57 follow-up)

Owner feedback while blind-testing `make sentiment-survey`: "текст идет нейтральный и можно
различить только нотки в голосе, но помогало бы разобраться само выражение" — on a neutral
text, the tag has nothing to work with content-wise, only tone; a text whose own content
matches the emotion would help the ear.

This document does two things: (1) records the measurement-design decision so a future change
doesn't accidentally collapse two different questions into one, and (2) curates one short text
per `emotion:*` tag for the actual audio generation this enables — a step this task did **not**
perform (see "What's still open" below).

## What each variant now measures — read this before generating anything

The existing tag-vs-neutral comparison (`catalog.build_catalog_sets()`, `_differ_task`) holds
**text fixed** (the shared neutral baseline sentence) and varies only the **tag**. A "yes, the
delivery differs" answer is therefore attributable to the tag alone.

Naively replacing the neutral text with an emotion-matched one *and* keeping the same
tag-vs-neutral-baseline structure would vary **two** things at once (text content and tag), and
an "it differs" answer would become unattributable — the owner would be hearing "the content is
different" as much as "the tag is different," with no way to tell which. That is a strictly
worse experiment, not a strictly better one, even though it sounds like an upgrade.

The design implemented here (`catalog.build_emotion_matched_text_set()`) instead holds the
**emotion-matched text fixed** and varies only the **tag** (present vs. absent), mirroring the
existing methodology on a different, non-neutral text:

| Set | Text | Tag | What a "differs" answer means |
|---|---|---|---|
| `unheard_sfx_env` / `catalog_remaining` / `disputed_tags` (existing) | neutral, fixed | present vs. absent | the tag alone changed the delivery |
| `emotion_matched_text` (new, this task) | emotion-matched, fixed | present vs. absent | the tag added something **on top of** what the matching content already conveys |

These two numbers are **not interchangeable** and must never be reported as a single "does the
tag work" verdict. A tag could show a strong difference on the neutral text and a weak one on
the matched text (the matched text alone already carries most of the emotional coloring, so the
tag has less headroom to add) — that would be a *real and useful* finding about where the tag's
effect actually lives, not a contradiction to explain away. The summary/`answers.md` for this
set is explicit that it's a separate, non-substitutable measurement (see
`docs/guides/sentiment_survey_guide.md`, "Текст под эмоцию").

A third combination — emotion-matched text *with* the tag vs. the *original neutral-text*
baseline — was deliberately not built: it would vary both text and tag from the neutral
baseline at once, collecting an answer with no way to attribute it to either factor. Not
generating it is a design choice, not an oversight.

## What's still open (this task did not generate audio)

This task ran under a standing no-generation constraint (the sentiment-survey work in this
repo runs against already-generated files; it does not invoke the TTS model). So this document
delivers:

- the curated text per emotion below (what the owner asked to have saved for docs/skills),
- the manifest schema and auto-discovery builder (`catalog.build_emotion_matched_text_set()`,
  covered by `tests/test_sentiment_survey.py::TestEmotionMatchedTextBuilder` against a synthetic
  fixture, not real speech),

but **not** the actual audio. Someone still needs to, for each emotion below:

1. Generate `<emotion>_tagged.wav` — the matched text below with `<|emotion:<name>|>`.
2. Generate `<emotion>_plain.wav` — the same text, no emotion tag.
3. Drop both into `output/m4_emotion_matched_text/` and add an entry to
   `output/m4_emotion_matched_text/manifest.json`:

```json
{
  "emotion": "sadness",
  "text": "Поэтому он был очень опечален.",
  "source": "chapter-e0-narration.txt §3 (sb-1-19 para 3)",
  "tagged_clip": "output/m4_emotion_matched_text/sadness_tagged.wav",
  "plain_clip": "output/m4_emotion_matched_text/sadness_plain.wav"
}
```

The moment that manifest exists with real files, `make sentiment-survey` picks the
`emotion_matched_text` set up automatically — same auto-discovery convention as every other
`output/`-scanned set (no manual task-list editing).

## Curated text, one per `emotion:*` tag

Source material: `samples/audiobook/prepared/` (Śrīmad-Bhāgavatam 1.1.1, 1.16, 1.19,
chapter-e0-narration.txt — 1.19 and chapter-e0-narration.txt are the same underlying chapter
text). 14 of 21 are verbatim quotes from that material (cited); 7 have no naturally
emotion-matched sentence in the available samples and are composed short lines, marked
**[СОЧИНЕНО]** so nobody mistakes them for scripture text. Composed lines keep the register
plain/neutral in *content* (per the owner's own instruction — "сочини короткую нейтральную по
содержанию, но окрашенную по смыслу"), colored only by wording, not by dramatic incident.

| Emotion | Text | Source |
|---|---|---|
| `emotion:affection` | «Своими сладостными улыбками, в которых сквозила любовь, взглядом, полным нежности, и сердечными обращениями Он мог победить серьезность и страстный гнев таких Своих возлюбленных, как Сатьябхама.» | sb-1-16.txt §13 |
| `emotion:amusement` | «Он не смог сдержать улыбку — ситуация вышла на редкость забавной.» | **[СОЧИНЕНО]** |
| `emotion:anger` | «Он сжал кулаки и почувствовал, как гнев поднимается в груди.» | **[СОЧИНЕНО]** |
| `emotion:arousal` | «Сердце забилось быстрее, всё тело напряглось в ожидании.» | **[СОЧИНЕНО]** |
| `emotion:awe` | «О святой! О великий мистик! Как атеист не может находиться в присутствии Личности Бога, так одного твоего присутствия достаточно, чтобы немедленно уничтожить все неодолимые грехи человека.» | chapter-e0-narration.txt / sb-1-19.txt §19 |
| `emotion:bitterness` | «Он вспомнил обещанное и понял, что его снова обманули.» | **[СОЧИНЕНО]** |
| `emotion:confusion` | «Сейчас так называемые правители, сбитые с толку под влиянием века Кали, привели государственные дела в беспорядок.» | sb-1-16.txt §13 |
| `emotion:contemplation` | «Для философского ума естественно стремление постичь источник творения. Когда в ночном небе мы видим звезды, мы, конечно, задумываемся над тем, кто их населяет.» | sb-1-1-1.txt §7 |
| `emotion:contentment` | «Царь, очень довольный теми, кто воспевал их славу, в великом удовлетворении открывал глаза.» | sb-1-16.txt §9 |
| `emotion:determination` | «Царь, достойный потомок Пандавов, решился раз и навсегда обосноваться на берегу Ганги, чтобы поститься до самой смерти.» | chapter-e0-narration.txt / sb-1-19.txt §5 (trimmed) |
| `emotion:disgust` | «От одного вида этого зрелища к горлу подступила тошнота.» | **[СОЧИНЕНО]** |
| `emotion:elation` | «Все полубоги с высших планет восславили действия царя и в радости осыпали землю цветами и ударили в небесные барабаны.» | chapter-e0-narration.txt / sb-1-19.txt §11 |
| `emotion:enthusiasm` | «Все великие мудрецы, собравшиеся там, восторженно приняли решение Махараджи Парикшита и выразили свое одобрение словами: «Очень хорошо!».» | chapter-e0-narration.txt / sb-1-19.txt §11 |
| `emotion:fear` | «Пока он так раскаивался, до него дошло известие о том, что по проклятию сына мудреца он должен умереть от укуса летучего змея.» | chapter-e0-narration.txt / sb-1-19.txt §3 |
| `emotion:helplessness` | «Я потерял три ноги и теперь стою на одной.» | sb-1-16.txt §11 |
| `emotion:longing` | «Кто, скажи, сможет вынести боль разлуки с Верховной Личностью Бога?» | sb-1-16.txt §13 |
| `emotion:pride` | «Он поднял голову и произнёс это с гордостью, как о деле всей своей жизни.» | **[СОЧИНЕНО]** |
| `emotion:relief` | «Царь воспринял это как добрую весть, потому что это могло помочь ему обрести безразличие ко всему мирскому.» | chapter-e0-narration.txt / sb-1-19.txt §3 |
| `emotion:sadness` | «Поэтому он был очень опечален.» | chapter-e0-narration.txt / sb-1-19.txt §3 |
| `emotion:shame` | «Царь Махараджа Парикшит чувствовал, что его поступок по отношению к безупречному и могущественному брахману был отвратительным и варварским.» | chapter-e0-narration.txt / sb-1-19.txt §3 |
| `emotion:surprise` | «Он замер на месте — такого поворота событий никто не ожидал.» | **[СОЧИНЕНО]** |

Notes on the quoted lines:
- Trimmed quotes keep original wording and punctuation; only leading/trailing context was cut,
  never words inside the kept sentence.
- Several rows share the same source paragraph (chapter 19's king's-grief passage) because that
  passage happens to carry several distinct emotional beats in sequence (shame → fear → relief →
  sadness) — reusing one paragraph for four different tags is intentional, not an oversight; each
  extracted sentence stands on its own and is not reused verbatim across rows.
- `determination`'s row is trimmed from a longer sentence ("...ганги, чтобы поститься до самой
  смерти и предаться лотосным стопам Господа Кришны, который один способен даровать
  освобождение...") down to a self-contained clause; the trimmed tail is doctrinal content
  irrelevant to the determination reading and was cut for length, not to change the meaning.
