"""Auto-discover generated audio in output/ and build dynamic task sets from it.

Owner's instruction (issue #57 expansion): the app must scan directories and
build tasks itself, not require every new generation run to be hand-listed in
a JSON file. This module is the scanner. It never compares clips from
different generation runs against each other (different text/params make
that comparison meaningless) — each function below stays inside one run's own
directory and, where the run has its own local neutral baseline file, pairs
against that local baseline rather than a baseline from another run.

Produces "task set documents" shaped exactly like the hand-written JSON files
in task_sets/ (same schema fed to TaskSet()), so server.py can treat
JSON-loaded and scan-generated sets identically.
"""
from __future__ import annotations

import json
from pathlib import Path

from tag_reference import CATEGORY_RU, parse_tag_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
TAG_REF_PATH = REPO_ROOT / "docs" / "guides" / "tag_reference.md"

# Group B/C in the M4-T5 objective triage — the tags the owner explicitly
# called out as "spisornye" (disputed) and wants checked before anything else
# already-confirmed. Derived at runtime from tag_reference.md, not hardcoded,
# so a future re-triage in the doc automatically reprioritizes this set too.
DISPUTED_GROUPS = {"B", "C"}

# sfx/env tags deserve a specific "does it sound like X, and is it in the
# right place" question rather than a generic "do these differ" — this is
# the one bit of per-tag phrasing kept hand-written (short Russian nouns,
# not derivable from the tag name string itself).
SFX_ENV_NOUN = {
    "sfx:cough": "кашель",
    "sfx:laughter": "смех",
    "sfx:crying": "плач",
    "sfx:screaming": "крик",
    "sfx:burping": "отрыжка",
    "sfx:humming": "мычание/напевание",
    "sfx:sigh": "вздох",
    "sfx:sniff": "шмыганье носом",
    "sfx:sneeze": "чихание",
    "env:music": "музыкальный фон",
    "env:noise": "фоновый шум",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))



# Owner feedback (issue #57 follow-up, item 3): "не совсем понятно что значит
# звучат ли они одинаково, голоса, я так понял совсем разные -- они похоже не
# будут звучать же". The old wording ("Отличаются ли эти два клипа по
# звучанию (интонация, тон голоса, манера)?") invited exactly this reading --
# mentioning "тон голоса" made it sound like a voice-IDENTITY question, and
# the voices ARE always different (Higgs pins no seed/reference across
# calls -- see pitch.py's docstring). Voice identity is a question numbers
# already answer (measured F0), never something worth asking the ear. The
# question below asks only about the DELIVERY -- intonation/character/pace --
# and says so explicitly, so "of course the voices sound different" stops
# being a valid reason to distrust the question.
DIFFER_QUESTION = (
    "Отличается ли ПОДАЧА речи в этих двух клипах — интонация, эмоциональная "
    "окраска, темп? (Сами голоса почти наверняка будут разными — это не "
    "проверяется на слух, это отдельно измерено по высоте тона; вопрос только "
    "про характер речи.)"
)
DIFFER_OPTIONS = ["Да, подача отличается", "Нет, подача такая же", "Не уверен(а)"]


def _differ_task(task_id: str, tagged_path: Path, baseline_path: Path, tag_key: str,
                  tag_facts: dict, prior_verdict: str | None = None) -> dict:
    category = tag_key.split(":", 1)[0]
    category_ru = CATEGORY_RU.get(category, category)
    task = {
        "id": task_id,
        "type": "pair_compare",
        "answer_kind": "differ",
        "question": DIFFER_QUESTION,
        "options": list(DIFFER_OPTIONS),
        "clips": {"A": _rel(tagged_path), "B": _rel(baseline_path)},
        "hidden": {
            "A": {"tag": tag_key, "category": category_ru},
            "B": {"tag": "neutral (baseline)"},
            "correct_answer": DIFFER_OPTIONS[0],
        },
    }
    if tag_facts:
        task["hidden"]["A"]["doc_status"] = tag_facts.get("status_text")
        task["hidden"]["A"]["doc_group"] = tag_facts.get("group")
        task["hidden"]["A"]["tokenizer_id"] = tag_facts.get("id")
    if prior_verdict:
        task["prior_verdict"] = prior_verdict
    return task


def _sfx_env_task(task_id: str, path: Path, tag_key: str, tag_facts: dict) -> dict:
    noun = SFX_ENV_NOUN.get(tag_key, tag_key)
    is_env = tag_key.startswith("env:")
    if is_env:
        question = f"Слышен ли в этом клипе {noun}, и не испортил ли он разборчивость речи?"
        options = [
            f"Да, {noun} слышен, речь не пострадала",
            f"{noun[0].upper()}{noun[1:]} слышен, но речь стала хуже разборчива",
            f"{noun[0].upper()}{noun[1:]} не слышно",
            "Не уверен(а)",
        ]
    else:
        question = (
            f"Слышен ли в этом клипе {noun}, и появляется ли он между 1-м и 2-м предложением "
            f"(не портя саму речь)?"
        )
        options = [
            f"Да, {noun} слышен(а) и на своём месте",
            "Что-то слышно, но не похоже или не на месте",
            "Ничего не слышно / речь испорчена",
            "Не уверен(а)",
        ]
    task = {
        "id": task_id,
        "type": "single_rating",
        "question": question,
        "options": options,
        "clips": {"A": _rel(path)},
        "hidden": {"A": {"tag": tag_key}},
    }
    if tag_facts:
        task["hidden"]["A"]["doc_status"] = tag_facts.get("status_text")
        task["hidden"]["A"]["tokenizer_id"] = tag_facts.get("id")
    return task


def _catalog_tags(dir_path: Path) -> list[tuple[str, Path]]:
    """List (tag_key, path) for output/m4_tag_catalog/tag_*.wav files."""
    out = []
    for f in sorted(dir_path.glob("tag_*.wav")):
        stem = f.stem[len("tag_"):]
        category, _, name = stem.partition("_")
        out.append((f"{category}:{name}", f))
    return out


def build_catalog_sets() -> list[dict]:
    """output/m4_tag_catalog/ — 46 clips, one run, comparable to each other."""
    d = REPO_ROOT / "output" / "m4_tag_catalog"
    if not d.is_dir():
        return []
    baseline = d / "neutral_baseline.wav"
    if not baseline.is_file():
        return []
    tag_facts_all = parse_tag_reference(TAG_REF_PATH)

    unheard_tasks, disputed_tasks, rest_tasks = [], [], []
    for tag_key, path in _catalog_tags(d):
        category = tag_key.split(":", 1)[0]
        facts = tag_facts_all.get(tag_key, {})
        prior = None
        if facts.get("confirmed"):
            prior = (
                "По этому тегу уже есть более ранний устный вердикт владельца, подтверждённый "
                "лично (docs/guides/tag_reference.md). Подробности откроются после ответа."
            )
        elif facts.get("group") == "A" and category == "emotion":
            # tag_reference.md §0: owner gave a collective (not per-tag) verdict
            # specifically for the 21 emotion:* tags ("работают, используйте свободно"),
            # based on the individually re-verified sadness/elation pair. Do not extend
            # this claim to prosody/style group-A tags below — those are only an
            # objective-metrics bucket, never an owner verdict.
            prior = (
                "Эта эмоция входит в группу A объективной триажи M4-T5 и в число 21 emotion:*, "
                "которые владелец уже обобщённо подтвердил как различимые (docs/guides/"
                "tag_reference.md §0), опираясь на индивидуально перепроверенную пару "
                "sadness/elation. Подробности откроются после ответа."
            )
        # Group-A prosody/style tags get no skip option: a strong objective
        # signal (F0/tempo/energy) is not the same as an owner verdict, and
        # falsely offering "already known, skip" for those would misrepresent
        # what has actually been confirmed by ear.
        if category in ("sfx", "env"):
            task = _sfx_env_task(f"catalog-{tag_key.replace(':', '-')}", path, tag_key, facts)
            unheard_tasks.append(task)
            continue
        task = _differ_task(f"catalog-{tag_key.replace(':', '-')}", path, baseline, tag_key, facts, prior)
        if facts.get("group") in DISPUTED_GROUPS:
            disputed_tasks.append(task)
        else:
            rest_tasks.append(task)

    sets = []
    if unheard_tasks:
        sets.append({
            "id": "unheard_sfx_env",
            "title": "Непроверенное: sfx + env (первое прослушивание)",
            "priority": 1,
            "description": (
                "output/m4_tag_catalog/ — 9 sfx-тегов и 2 env-тега, ни разу не прослушанные "
                "человеком (issues #116, #119; docs/guides/tag_reference.md §2.4/§3). "
                "Автообнаружено сканированием output/m4_tag_catalog/tag_sfx_*.wav и tag_env_*.wav."
            ),
            "tasks": unheard_tasks,
        })
    if disputed_tasks:
        sets.append({
            "id": "disputed_tags",
            "title": "Спорные по метрикам (группы B/C объективной триажи)",
            "priority": 2,
            "description": (
                "output/m4_tag_catalog/ — теги, чей объективный сигнал (F0/темп/паузы) в M4-T5 "
                "оказался слабым, неоднозначным или направленным не в ту сторону. Автоматически "
                "отобраны из docs/guides/tag_reference.md по статусу «Группа B/C»."
            ),
            "tasks": disputed_tasks,
        })
    if rest_tasks:
        sets.append({
            "id": "catalog_remaining",
            "title": "Остальные теги каталога (группа A / стиль / просодия)",
            "priority": 5,
            "description": (
                "output/m4_tag_catalog/ — теги с сильным объективным сигналом или уже частично "
                "подтверждённые владельцем (группа A). Не обязательно переслушивать — можно "
                "нажать «Пропустить», если внизу вопроса есть пометка о более раннем вердикте."
            ),
            "tasks": rest_tasks,
        })
    return sets


_M4_TAGS_SUBDIRS = {
    "01_emotion": "emotion",
    "02_prosody": "prosody",
    "03_style": "style",
}


def build_m4_tags_second_run_set() -> dict | None:
    """output/m4_tags/{01_emotion,02_prosody,03_style}/ — a different run than
    m4_tag_catalog/ (different text/params per docs/guides/tag_reference.md's
    own framing note), so it gets its own set and pairs against each
    subdirectory's own zz_neutral_baseline.wav, never against m4_tag_catalog's."""
    root = REPO_ROOT / "output" / "m4_tags"
    if not root.is_dir():
        return None
    tasks = []
    for subdir, category in _M4_TAGS_SUBDIRS.items():
        d = root / subdir
        if not d.is_dir():
            continue
        baseline = d / "zz_neutral_baseline.wav"
        if not baseline.is_file():
            continue
        prefix = category + "_"
        for f in sorted(d.glob("*.wav")):
            if f.name == "zz_neutral_baseline.wav":
                continue
            name = f.stem[len(prefix):] if f.stem.startswith(prefix) else f.stem
            tag_key = f"{category}:{name}"
            task_id = f"m4tags-{subdir}-{name}"
            tasks.append(_differ_task(task_id, f, baseline, tag_key, {}))
    if not tasks:
        return None
    return {
        "id": "m4_tags_second_run",
        "title": "Второй независимый прогон (output/m4_tags/, другой текст)",
        "priority": 5,
        "description": (
            "output/m4_tags/01_emotion, 02_prosody, 03_style — независимый более ранний прогон "
            "(PR #108), другой текст/параметры, поэтому сравнивается только с собственной "
            "локальной нейтралью каждой подпапки (zz_neutral_baseline.wav), не с "
            "m4_tag_catalog/. Полезно как второе независимое подтверждение, не как замена "
            "первого набора."
        ),
        "tasks": tasks,
    }


def build_m4t0_set() -> dict | None:
    """output/m4t0_*.wav — the very first blind-sentiment probe (M4-T0)."""
    d = REPO_ROOT / "output"
    baseline = d / "m4t0_neutral.wav"
    if not baseline.is_file():
        return None
    names = {
        "sadness": "emotion:sadness",
        "elation": "emotion:elation",
        "whispering": "style:whispering",
        "speed_slow": "prosody:speed_slow",
    }
    tasks = []
    for short, tag_key in names.items():
        f = d / f"m4t0_{short}.wav"
        if not f.is_file():
            continue
        prior = (
            "Это самая первая проверка сентимента (M4-T0/M4-T5) — по ней уже есть записанный "
            "устный вердикт владельца. Подробности откроются после ответа."
        )
        tasks.append(_differ_task(f"m4t0-{short}", f, baseline, tag_key, {}, prior))
    if not tasks:
        return None
    return {
        "id": "m4t0_original_probe",
        "title": "Самая первая проверка (M4-T0, до всех остальных прогонов)",
        "priority": 5,
        "description": (
            "output/m4t0_*.wav — 5 клипов первой в проекте слепой проверки сентимента "
            "(sadness/elation/whispering/speed_slow против m4t0_neutral.wav). Автообнаружено; "
            "вердикты по ним уже есть, но переслушать можно."
        ),
        "tasks": tasks,
    }


def build_boundary_check_set() -> dict | None:
    """output/m4_boundary_check/ — does a sentence-level tag re-open across a
    chunk boundary that it shouldn't (or fail to, when it should)?"""
    d = REPO_ROOT / "output" / "m4_boundary_check"
    reopen = d / "chunk2_reopen.wav"
    noreopen = d / "chunk2_noreopen.wav"
    if not (reopen.is_file() and noreopen.is_file()):
        return None
    task = {
        "id": "boundary-reopen-vs-noreopen",
        "type": "pair_compare",
        "answer_kind": "differ",
        "question": (
            "Оба клипа — второй фрагмент (chunk) одного и того же текста, разбитого на части. "
            "В одном тег эмоции на границе куска переоткрывается, в другом — нет. "
            "Отличается ли подача речи (интонация, эмоциональная окраска) между ними? "
            "(Сами голоса могут отличаться и сами по себе — это отдельно измерено по высоте "
            "тона, вопрос только про характер речи.)"
        ),
        "options": list(DIFFER_OPTIONS),
        "clips": {"A": _rel(reopen), "B": _rel(noreopen)},
        "hidden": {
            "A": {"variant": "tag reopened at chunk boundary"},
            "B": {"variant": "tag NOT reopened at chunk boundary"},
            "correct_answer": DIFFER_OPTIONS[0],
        },
    }
    return {
        "id": "boundary_check",
        "title": "Граница чанка: переоткрывается ли тег",
        "priority": 3,
        "description": (
            "output/m4_boundary_check/ — проверка риска, важного для сегментации главы на куски: "
            "сохраняется ли эмоция на второй части текста, если тег не повторён в начале куска."
        ),
        "tasks": [task],
    }


def build_emotion_matched_text_set() -> dict | None:
    """output/m4_emotion_matched_text/manifest.json — owner feedback #2
    (issue #57 follow-up): "текст идет нейтральный и можно различить только
    нотки в голосе, но помогало бы разобраться само выражение". The existing
    tag-vs-neutral-baseline comparison (build_catalog_sets()) keeps text
    FIXED (neutral) and varies only the tag, so a "yes it differs" answer is
    attributable to the tag alone. Changing the text AND keeping the tag
    fixed would confound two variables in the opposite pair, so this builder
    keeps the SAME (emotion-matched) text fixed across its own pair and
    varies only the tag -- "does the tag add anything ON TOP OF what the
    matching text already conveys" -- rather than comparing across two
    different texts, which would be unattributable.

    Not run automatically: no audio has been generated for this set yet (see
    docs/research/audiobook/m4-emotion-matched-texts.md for the curated
    per-emotion text, ready for a future generation pass). Returns None,
    same as every other builder here, when its output/ directory or manifest
    is absent -- this function exists so the set is picked up automatically
    the moment someone runs that generation, with zero code changes.

    manifest.json schema (list of objects), one per emotion:
      {"emotion": "sadness",                     # matches an emotion:* tag
       "text": "...",                             # the emotion-matched text used
       "source": "chapter-e0-narration.txt §3"     # or "[СОЧИНЕНО]" if composed
                 or "[СОЧИНЕНО]",
       "tagged_clip": "output/m4_emotion_matched_text/sadness_tagged.wav",
       "plain_clip": "output/m4_emotion_matched_text/sadness_plain.wav"}
    `tagged_clip` = matched text + <|emotion:sadness|>; `plain_clip` = the
    SAME matched text, no tag. The comparison isolates the tag's marginal
    effect on top of content that already carries the emotion.
    """
    d = REPO_ROOT / "output" / "m4_emotion_matched_text"
    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    tasks = []
    for entry in entries:
        emotion = entry.get("emotion")
        tagged = d / Path(entry.get("tagged_clip", "")).name if entry.get("tagged_clip") else None
        plain = d / Path(entry.get("plain_clip", "")).name if entry.get("plain_clip") else None
        if not emotion or not tagged or not plain or not tagged.is_file() or not plain.is_file():
            continue
        composed = entry.get("source") == "[СОЧИНЕНО]"
        task = {
            "id": f"emo-matched-{emotion}",
            "type": "pair_compare",
            "answer_kind": "differ",
            "question": (
                f"Текст этого клипа подобран под эмоцию «{emotion}» (не нейтральный). "
                "Отличается ли подача — интонация, эмоциональная окраска — между вариантом "
                "с тегом эмоции и без него на ОДНОМ И ТОМ ЖЕ тексте? (Сами голоса могут "
                "отличаться сами по себе — это не проверяется на слух.)"
            ),
            "options": list(DIFFER_OPTIONS),
            "clips": {"A": _rel(tagged), "B": _rel(plain)},
            "hidden": {
                "A": {"tag": f"emotion:{emotion}", "text_variant": "matched+tag"},
                "B": {"tag": "matched text, no emotion tag", "text_variant": "matched, plain"},
                "matched_text": entry.get("text", ""),
                "text_source": entry.get("source", ""),
                "text_composed": composed,
                "correct_answer": DIFFER_OPTIONS[0],
            },
        }
        tasks.append(task)
    if not tasks:
        return None
    return {
        "id": "emotion_matched_text",
        "title": "Эмоция на подходящем тексте (а не на нейтральном)",
        "priority": 4,
        "description": (
            "output/m4_emotion_matched_text/ — та же эмоция, но на тексте, чьё СОДЕРЖАНИЕ "
            "уже окрашено под неё (не нейтральный текст), с тегом и без. Измеряет, добавляет "
            "ли тег что-то СВЕРХ того, что уже несёт сам текст — это ДРУГОЕ измерение, чем "
            "тег-против-нейтрали на нейтральном тексте (docs/guides/sentiment_survey_guide.md, "
            "«Текст под эмоцию»); результаты этих двух наборов не взаимозаменяемы."
        ),
        "tasks": tasks,
    }


def build_all_dynamic_sets() -> list[dict]:
    sets = list(build_catalog_sets())
    for builder in (build_m4_tags_second_run_set, build_m4t0_set, build_boundary_check_set,
                     build_emotion_matched_text_set):
        doc = builder()
        if doc:
            sets.append(doc)
    return sets
