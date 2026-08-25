#!/usr/bin/env python3
"""Local blind-listening survey app for Higgs sentiment-tag verification (issue #57).

Stdlib-only (http.server + json), no framework. Serves a small single-page UI that:
  - plays one task's clip(s) with tag identity hidden (opaque clip ids, no filenames
    leaked to the browser),
  - lets the owner answer without knowing which variant/tag they're hearing,
  - reveals the hidden metadata only after the answer is submitted,
  - writes every answer to disk immediately (atomic temp-file + os.replace), so an
    interrupted session never corrupts the results file and can be resumed later,
  - lets the owner navigate back/forward through already-answered tasks (or jump to
    any task in a full list with per-task state) and resubmit a corrected answer,
  - keeps every answer ever submitted in `answers.jsonl` (a correction is a new,
    separately timestamped line, not an in-place overwrite) while grading and the
    "next unanswered" flow always resolve to the *latest* answer per task,
  - honestly flags a resubmitted answer as `answered_after_reveal: true`, since the
    first answer's reveal already showed the hidden metadata — it can't be blind
    the second time, and the M4 gate counts it separately rather than pretending
    otherwise,
  - lets the owner attach an optional free-text note to any task (what they heard
    that no answer option captured), editable the same way as a correction (a new
    append-only revision) but without affecting is_correction/answered_after_reveal
    when only the note changes and the answer itself didn't -- see _answer_value().

Run: `python3 src/sentiment_survey/server.py` (see `make sentiment-survey`).
Results: `output/sentiment_survey_results/<set_id>/answers.jsonl` (machine-readable,
one JSON object per line) and `answers.md` (human-readable, regenerated after every
answer) — both under `output/`, which is gitignored.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import socketserver
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog  # noqa: E402  (needs sys.path tweak above; same-directory module)
import pitch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_SETS_DIR = Path(__file__).resolve().parent / "task_sets"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = REPO_ROOT / "output" / "sentiment_survey_results"

HOST = "127.0.0.1"
DEFAULT_PORT = 8877

VALID_ANSWER_KINDS = {"differ", "which", "which_matches_ref"}
VALID_TYPES = {"pair_compare", "single_rating", "triple_compare"}

# Sentinel answer text for "I already know the verdict from a previous session
# (docs/guides/tag_reference.md, m4-sentiment-results.md, ...) — don't make me
# re-listen." Never reveals the tag itself; the actual prior verdict text is
# only sent to the browser inside the post-answer reveal payload.
SKIP_LABEL = "Пропустить — уже есть более ранний вердикт"


# --------------------------------------------------------------------------- #
# Task set loading
# --------------------------------------------------------------------------- #

class TaskSet:
    def __init__(self, doc: dict, source_path: Path | str):
        self.id = doc["id"]
        self.title = doc.get("title", self.id)
        self.description = doc.get("description", "")
        self.priority = int(doc.get("priority", 5))
        self.tasks = doc["tasks"]
        self.source_path = source_path
        self._tasks_by_id = {t["id"]: t for t in self.tasks}
        self._validate()
        # Per-process randomized slot order, stable until server restart.
        self._slot_order: dict[str, list[str]] = {}
        self._rng = random.Random()
        for task in self.tasks:
            keys = list(task["clips"].keys())
            if task["type"] == "triple_compare":
                non_ref = [k for k in keys if k != "REF"]
                self._rng.shuffle(non_ref)
                order = (["REF"] if "REF" in keys else []) + non_ref
            else:
                order = list(keys)
                self._rng.shuffle(order)
            self._slot_order[task["id"]] = order

    def _validate(self):
        seen = set()
        for t in self.tasks:
            if t["id"] in seen:
                raise ValueError(f"{self.source_path}: duplicate task id {t['id']!r}")
            seen.add(t["id"])
            if t["type"] not in VALID_TYPES:
                raise ValueError(f"{self.source_path}: task {t['id']!r} has unknown type {t['type']!r}")
            if "clips" not in t or not t["clips"]:
                raise ValueError(f"{self.source_path}: task {t['id']!r} has no clips")
            for role, rel_path in t["clips"].items():
                abs_path = (REPO_ROOT / rel_path).resolve()
                if REPO_ROOT not in abs_path.parents and abs_path != REPO_ROOT:
                    raise ValueError(f"{self.source_path}: task {t['id']!r} clip {role!r} escapes repo root")
                if not abs_path.is_file():
                    raise ValueError(f"{self.source_path}: task {t['id']!r} clip {role!r} missing on disk: {abs_path}")

    def task(self, task_id: str) -> dict | None:
        return self._tasks_by_id.get(task_id)

    def slot_order(self, task_id: str) -> list[str]:
        return self._slot_order[task_id]


def _annotate_pitch_warnings_safely(docs: list[dict]) -> None:
    """Best-effort pitch-pairing annotation (issue #57 follow-up, owner
    feedback #1): mutates `docs` in place via pitch.annotate_pitch_warnings().
    Reuses docs/research/audiobook/m4_prosody_metrics.py, which needs numpy
    -- present in .venv-tts (what `make sentiment-survey` prefers) but not
    guaranteed for a bare `python3` fallback run. Missing numpy (or any
    other measurement failure) must never take the whole survey app down:
    log a warning and leave every task's pitch_warning unset, same as
    before this feature existed."""
    try:
        report = pitch.annotate_pitch_warnings(docs)
        print(
            f"Pitch-pairing threshold: {report.get('threshold_hz')} Hz "
            f"({report.get('method')}, n={report.get('n')})",
            file=sys.stderr,
        )
    except Exception as exc:  # ImportError (no numpy), bad wav, etc.
        print(f"warning: pitch-pairing analysis skipped ({exc})", file=sys.stderr)


def load_task_sets() -> dict[str, TaskSet]:
    """Hand-written JSON sets (task_sets/*.json) plus auto-discovered sets
    scanned live from output/ (catalog.build_all_dynamic_sets()). A dynamic
    set with the same id as a JSON one is skipped with a warning — JSON wins,
    since it means someone deliberately curated that set by hand."""
    json_docs = []
    for path in sorted(TASK_SETS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            json_docs.append((path, json.load(fh)))
    dynamic_docs = catalog.build_all_dynamic_sets()

    # Pitch-pair every comparison task across ALL loaded sets in one pass,
    # before constructing TaskSet objects, so JSON-curated sets (e.g.
    # emotion_vs_emotion.json) get the same treatment as auto-discovered
    # ones, and the corpus-wide threshold is computed once from everything.
    _annotate_pitch_warnings_safely([doc for _, doc in json_docs] + dynamic_docs)

    sets: dict[str, TaskSet] = {}
    for path, doc in json_docs:
        ts = TaskSet(doc, path)
        sets[ts.id] = ts

    for doc in dynamic_docs:
        if doc["id"] in sets:
            print(f"warning: dynamic set {doc['id']!r} shadowed by task_sets/*.json, skipping", file=sys.stderr)
            continue
        try:
            sets[doc["id"]] = TaskSet(doc, f"<dynamic:{doc['id']}>")
        except ValueError as exc:
            print(f"warning: skipping dynamic set {doc['id']!r}: {exc}", file=sys.stderr)
    return sets


TASK_SETS = load_task_sets()


# --------------------------------------------------------------------------- #
# Opaque clip id mapping (never leak real filenames/tags to the browser)
# --------------------------------------------------------------------------- #

CLIP_INDEX: dict[str, Path] = {}


def register_clip(rel_path: str) -> str:
    abs_path = (REPO_ROOT / rel_path).resolve()
    opaque = hashlib.sha1(str(abs_path).encode("utf-8")).hexdigest()[:20]
    CLIP_INDEX[opaque] = abs_path
    return opaque


for _ts in TASK_SETS.values():
    for _t in _ts.tasks:
        for _rel in _t["clips"].values():
            register_clip(_rel)


# --------------------------------------------------------------------------- #
# Results storage (atomic, incremental, resumable)
# --------------------------------------------------------------------------- #

def results_paths(set_id: str) -> tuple[Path, Path]:
    d = RESULTS_DIR / set_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "answers.jsonl", d / "answers.md"


def load_answer_history(set_id: str) -> list[dict]:
    """Every answer record ever written for this set, in file (= chronological)
    order. A task answered more than once (a correction) appears more than
    once here — each line is kept forever, nothing is overwritten in place.
    Old records written before the correction feature (issue #57 follow-up)
    have no `revision`/`is_correction`/`answered_after_reveal` keys; callers
    must use `.get()` with a default rather than assume they're present."""
    jsonl_path, _ = results_paths(set_id)
    history: list[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                history.append(json.loads(line))
    return history


def load_answers(set_id: str) -> dict[str, dict]:
    """Current (latest) answer per task_id — corrections shadow the record(s)
    they replace, but never erase them from the JSONL history."""
    answers: dict[str, dict] = {}
    for rec in load_answer_history(set_id):
        answers[rec["task_id"]] = rec
    return answers


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _answer_value(rec: dict) -> tuple:
    """The part of a record that constitutes "the actual decision" — used to
    tell a genuine correction (owner picked a different option) apart from a
    same-answer resubmission (owner only added or edited a free-text note).
    Deliberately excludes `note`: writing down an observation about what was
    heard doesn't change what was answered, so it must not affect blindness
    bookkeeping (issue #57 follow-up, "заметка слепоту не ломает")."""
    return (rec.get("answer_label"), rec.get("answer_role"))


def append_answer(set_id: str, record: dict) -> dict[str, dict]:
    """Append one answer to the permanent history and return the up-to-date
    "latest answer per task" map.

    History is never rewritten or dropped: if the owner already answered this
    task_id, `record` is a new line, but it is only stamped as a correction
    (revision > 1, `is_correction: true`) if the actual answer (option/role)
    differs from the immediately preceding revision — see `_answer_value()`.
    This is deliberate (issue #57 follow-up) — a changed mind is itself a
    signal worth keeping, not noise to overwrite. Grading and the resumable
    "next task" flow both use `load_answers()`, which always resolves to the
    latest revision per task, so a correction is what counts for the gate.

    A genuine correction (`is_correction`) also always means
    `answered_after_reveal = True`: the first answer's reveal (see
    `reveal_for_task`) already showed this task's hidden metadata, so
    changing the answer afterwards cannot be blind anymore. Recorded
    explicitly and honestly rather than silently re-blinding the UI, so the
    M4 gate can (and does, see `_handle_summary`) treat post-reveal answers
    as a separate, non-blind bucket. Resubmitting the *same* answer just to
    add or edit a free-text `note`, by contrast, does not touch this
    bookkeeping at all: the underlying decision was never revisited, only
    described in more detail, so it keeps the blind/non-blind status of
    whichever revision the answer itself last actually changed at.
    """
    jsonl_path, md_path = results_paths(set_id)
    history = load_answer_history(set_id)
    prior_for_task = [r for r in history if r["task_id"] == record["task_id"]]
    prior_count = len(prior_for_task)
    if prior_count == 0:
        record["revision"] = 1
        record["is_correction"] = False
        record["answered_after_reveal"] = False
        record["replaces_revision"] = None
    else:
        prev = prior_for_task[-1]
        answer_changed = _answer_value(record) != _answer_value(prev)
        record["revision"] = prior_count + 1
        record["is_correction"] = answer_changed
        # A note-only resubmission inherits the blind status the answer
        # already had (it didn't change), rather than being force-flipped
        # to non-blind just for arriving in a later revision.
        record["answered_after_reveal"] = answer_changed or bool(prev.get("answered_after_reveal"))
        record["replaces_revision"] = prior_count
    history.append(record)

    # Rewrite the whole JSONL from the in-memory history list: small files
    # (tens to a couple hundred lines even with corrections), so a full
    # atomic rewrite per answer is simpler and safer than a bare append
    # (which is not itself atomic against a mid-write crash).
    lines = [json.dumps(r, ensure_ascii=False) for r in history]
    atomic_write(jsonl_path, "\n".join(lines) + "\n" if lines else "")

    answers = {}
    for rec in history:
        answers[rec["task_id"]] = rec
    atomic_write(md_path, render_markdown(set_id, answers))
    return answers


def render_markdown(set_id: str, answers: dict[str, dict]) -> str:
    """Render the human-readable summary from the *latest* answer per task
    (`answers`, as returned by load_answers()/append_answer()) — corrections
    are what count here and for grading. The full history (including
    superseded answers) always stays in answers.jsonl, never in this file."""
    ts = TASK_SETS.get(set_id)
    title = ts.title if ts else set_id
    total = len(ts.tasks) if ts else len(answers)
    corrections = sum(1 for r in answers.values() if r.get("is_correction"))
    lines = [f"# {title}", "", f"Отвечено {len(answers)} из {total}.", ""]
    if corrections:
        lines.append(
            f"Из них исправлено {corrections}: показан последний ответ, "
            f"прежние варианты сохранены в `answers.jsonl` (не теряются)."
        )
        lines.append("")
    differed, not_differed, correct, incorrect, skipped, pitch_unreliable = 0, 0, 0, 0, 0, 0
    for rec in answers.values():
        if rec.get("skipped_prior"):
            skipped += 1
            continue
        if rec.get("pitch_warning"):
            # Voices not close enough in pitch (issue #57 follow-up, owner
            # feedback #1) -- kept out of the pass/fail tallies below, same
            # as a "already known" skip, but for a different reason.
            pitch_unreliable += 1
            continue
        exp = rec.get("correct_answer")
        gradable = record_is_still_gradable(ts, rec)
        if gradable:
            if rec.get("matches_expected"):
                correct += 1
            else:
                incorrect += 1
        if rec.get("type") == "pair_compare" and rec.get("answer_kind") == "differ":
            if rec.get("matches_expected"):
                differed += 1
            elif gradable:
                not_differed += 1
    if correct or incorrect:
        lines.append(f"**Совпало с ожиданием (свежие ответы, без пропущенных и недостоверных по высоте голоса): {correct} из {correct + incorrect}.**")
        lines.append("")
    if skipped:
        lines.append(f"**Пропущено как уже подтверждённое ранее: {skipped}.**")
        lines.append("")
    if pitch_unreliable:
        lines.append(
            f"**Недостоверно по высоте голоса (пара не прошла порог pitch-pairing, "
            f"см. `docs/guides/sentiment_survey_guide.md`): {pitch_unreliable}.**"
        )
        lines.append("")
    lines.append("| Задание | Тип | Ответ | Ожидалось | Совпало | Время прослушивания | Отметка времени |")
    lines.append("|---|---|---|---|---|---|---|")
    for tid, rec in answers.items():
        exp = rec.get("correct_answer") or ""
        gradable = record_is_still_gradable(ts, rec)
        if rec.get("skipped_prior"):
            match = "пропущено"
        elif rec.get("pitch_warning"):
            match = "недостоверно (высота голоса)"
        elif exp and not gradable:
            # The task_sets definition no longer claims a (or the same)
            # correct_answer for this task_id -- e.g. issue #57's
            # final-boundary-continuing, whose question turned out to have
            # no gradable correct answer. The historical record and its
            # own matches_expected are left exactly as answered; only the
            # display/tally treats it as retracted.
            match = "снято (не оценивается)"
        else:
            match = "" if exp == "" else ("да" if rec.get("matches_expected") else "нет")
        answer_display = rec.get('answer_label', rec.get('answer', ''))
        if rec.get('is_correction'):
            answer_display += f" _(испр., см. ред. {rec.get('replaces_revision')})_"
        lines.append(
            f"| `{tid}` | {rec.get('type', '')} | {answer_display} "
            f"| {exp} | {match} | {rec.get('listen_ms', 0) / 1000:.1f} с | {rec.get('timestamp', '')} |"
        )
    lines.append("")
    lines.append("## Раскрытые метаданные по заданиям")
    lines.append("")
    for tid, rec in answers.items():
        lines.append(f"### `{tid}`")
        lines.append("")
        lines.append(f"- Вопрос: {rec.get('question', '')}")
        lines.append(f"- Ответ владельца: **{rec.get('answer_label', rec.get('answer', ''))}**")
        lines.append(f"- Скрытые метаданные: `{json.dumps(rec.get('hidden', {}), ensure_ascii=False)}`")
        note = (rec.get("note") or "").strip()
        if note:
            lines.append("- Заметка владельца:")
            lines.append("")
            for note_line in note.splitlines() or [note]:
                lines.append(f"  > {note_line}")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Task view construction (what the browser is allowed to see before answering)
# --------------------------------------------------------------------------- #

def build_task_view(ts: TaskSet, task: dict) -> dict:
    order = ts.slot_order(task["id"])
    non_ref_count = sum(1 for role in order if role != "REF")
    slots = []
    seen_non_ref = 0
    for role in order:
        rel = task["clips"][role]
        opaque = register_clip(rel)
        if role == "REF":
            label = "Эталон"
        elif non_ref_count == 1:
            label = "Клип"
        else:
            seen_non_ref += 1
            label = f"Клип {seen_non_ref}"
        slots.append({"role": role, "label": label, "clip_url": f"/clip/{opaque}"})

    has_prior = bool(task.get("prior_verdict"))
    view = {
        "id": task["id"],
        "type": task["type"],
        "question": task["question"],
        "slots": slots,
        "has_prior_verdict": has_prior,
        # Voices are never pinned across generations, so a pitch mismatch is
        # not tag identity -- safe to reveal before answering, unlike
        # `hidden` (issue #57 follow-up, owner feedback #1). None when the
        # task's clips are close enough in pitch, or pitch analysis wasn't
        # available for this run (see _annotate_pitch_warnings_safely()).
        "pitch_warning": task.get("pitch_warning"),
    }
    answer_kind = task.get("answer_kind")
    if answer_kind == "differ":
        view["response_mode"] = "fixed_options"
        view["options"] = list(task["options"])
    elif answer_kind in ("which", "which_matches_ref"):
        view["response_mode"] = "choose_clip"
        non_ref_labels = [s["label"] for s in slots if s["role"] != "REF"]
        view["options"] = non_ref_labels + ["Не могу сказать"]
    else:
        view["response_mode"] = "fixed_options"
        view["options"] = list(task.get("options", ["Да", "Нет", "Не уверен(а)"]))
    if has_prior:
        view["options"] = view["options"] + [SKIP_LABEL]
    return view


def live_correct_answer(ts: "TaskSet | None", task_id: str) -> object:
    """The CURRENT expected answer for `task_id` per task_sets/*.json (or
    the dynamically-built catalog), as opposed to whatever a historical
    answers.jsonl record baked into its own `correct_answer` field at
    answer time.

    Grading (render_markdown(), _handle_summary()) uses this instead of a
    record's baked value, and only trusts the record's own
    `matches_expected` verdict when the record's baked correct_answer still
    agrees with the live one (`record_is_still_gradable()` below). This is
    the honest fix for issue #57's `final-boundary-continuing`: that task's
    expected answer turned out to be unanswerable (a grammatically complete
    sentence has no correct "is the thought finished?" verdict -- see
    m4-tag-inventory-results.md sec. 5 item 6), so its task_sets JSON entry
    now carries `correct_answer: null`. The owner's already-recorded answer
    in answers.jsonl is append-only history and is deliberately left
    untouched (it was the CORRECT listening call); this function is what
    keeps that historical record from being counted as a miss in any
    future summary, without rewriting or deleting anything in the JSONL.
    """
    if ts is None:
        return None
    task = ts.task(task_id)
    if task is None:
        return None
    return (task.get("hidden") or {}).get("correct_answer")


def record_is_still_gradable(ts: "TaskSet | None", rec: dict) -> bool:
    """True if `rec` should count toward the correct/incorrect tally: the
    task_sets definition still claims a correct_answer for this task_id,
    AND it is the same expectation the record was judged against when it
    was answered (so an expectation that has since changed value, not just
    been retracted to null, is also treated as ungradable rather than
    silently re-scored against a different answer than the owner saw)."""
    live = live_correct_answer(ts, rec.get("task_id"))
    return live is not None and rec.get("correct_answer") == live


def reveal_for_task(task: dict) -> dict:
    hidden = task.get("hidden", {})
    return {
        "hidden": hidden,
        "clips": task["clips"],
        "correct_answer": hidden.get("correct_answer"),
        "prior_verdict": task.get("prior_verdict"),
    }


def compute_matches_expected(task: dict, ts: TaskSet, answer_role_or_text: str, chosen_label: str) -> tuple[bool | None, str | None]:
    """Return (matches_expected, expected_display) using hidden.correct_answer.

    For 'which'/'which_matches_ref' answer kinds, correct_answer is a role key
    (e.g. "A"); we compare against the role the owner actually picked.
    For 'differ'/fixed_options, correct_answer is the literal expected option text.
    """
    hidden = task.get("hidden", {})
    expected = hidden.get("correct_answer")
    if expected is None:
        return None, None
    answer_kind = task.get("answer_kind")
    if answer_kind in ("which", "which_matches_ref"):
        return (answer_role_or_text == expected), expected
    return (chosen_label == expected), expected


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "SentimentSurvey/1.0"

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json({"error": message}, status)

    def _send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_audio(self, path: Path):
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                start_s, end_s = range_header[len("bytes="):].split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
            except ValueError:
                start, end = 0, size - 1
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Disposition", 'inline; filename="clip.wav"')
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                self.wfile.write(fh.read(length))
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", 'inline; filename="clip.wav"')
            self.end_headers()
            with path.open("rb") as fh:
                self.wfile.write(fh.read())

    # ---- routing ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/" or path == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/app.js":
                self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            elif path == "/style.css":
                self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
            elif path == "/api/sets":
                self._handle_list_sets()
            elif path.startswith("/api/sets/") and path.endswith("/next"):
                set_id = path[len("/api/sets/"):-len("/next")]
                self._handle_next(set_id)
            elif path.startswith("/api/sets/") and path.endswith("/summary"):
                set_id = path[len("/api/sets/"):-len("/summary")]
                self._handle_summary(set_id)
            elif path.startswith("/api/sets/") and path.endswith("/tasks"):
                set_id = path[len("/api/sets/"):-len("/tasks")]
                self._handle_task_list(set_id)
            elif "/task/" in path and path.startswith("/api/sets/"):
                rest = path[len("/api/sets/"):]
                set_id, task_id = rest.split("/task/", 1)
                self._handle_task_detail(unquote(set_id), unquote(task_id))
            elif path.startswith("/clip/"):
                opaque = path[len("/clip/"):]
                self._handle_clip(opaque)
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except BrokenPipeError:
            pass
        except Exception as exc:  # keep the server alive across bad requests
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/sets/") and path.endswith("/answer"):
                set_id = path[len("/api/sets/"):-len("/answer")]
                self._handle_answer(set_id)
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # ---- handlers ----

    def _handle_list_sets(self):
        out = []
        for ts in TASK_SETS.values():
            answers = load_answers(ts.id)
            out.append({
                "id": ts.id,
                "title": ts.title,
                "description": ts.description,
                "priority": ts.priority,
                "total": len(ts.tasks),
                "answered": len(answers),
            })
        out.sort(key=lambda s: (s["priority"], s["id"]))
        self._send_json({"sets": out})

    def _handle_next(self, set_id: str):
        ts = TASK_SETS.get(set_id)
        if ts is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown set {set_id!r}")
            return
        answers = load_answers(set_id)
        remaining = [t for t in ts.tasks if t["id"] not in answers]
        if not remaining:
            self._send_json({"done": True, "answered": len(answers), "total": len(ts.tasks)})
            return
        task = remaining[0]
        view = build_task_view(ts, task)
        self._send_json({
            "done": False,
            "answered": len(answers),
            "total": len(ts.tasks),
            "task": view,
        })

    def _handle_task_list(self, set_id: str):
        """Full ordered task inventory with per-task state, for the sidebar
        that lets the owner see where they are across all N tasks and jump
        to any one of them directly (issue #57 follow-up, requirement 4)."""
        ts = TASK_SETS.get(set_id)
        if ts is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown set {set_id!r}")
            return
        answers = load_answers(set_id)
        items = []
        for idx, task in enumerate(ts.tasks):
            rec = answers.get(task["id"])
            items.append({
                "index": idx,
                "id": task["id"],
                "question": task["question"],
                "answered": rec is not None,
                "is_correction": bool(rec.get("is_correction")) if rec else False,
                "answered_after_reveal": bool(rec.get("answered_after_reveal")) if rec else False,
                "matches_expected": rec.get("matches_expected") if rec else None,
                "skipped_prior": bool(rec.get("skipped_prior")) if rec else False,
                "has_note": bool((rec.get("note") or "").strip()) if rec else False,
            })
        self._send_json({
            "set_id": set_id,
            "title": ts.title,
            "total": len(ts.tasks),
            "answered": len(answers),
            "tasks": items,
        })

    def _handle_task_detail(self, set_id: str, task_id: str):
        """One task by id (not just 'the next unanswered one'), so the
        browser can navigate back/forward and jump to any task. If the task
        was already answered, the reveal and the previous answer are
        returned immediately — the labels were revealed the moment the
        first answer was submitted, so there is nothing left to hide, and
        pretending otherwise would just be lying to the owner about their
        own past session (issue #57 follow-up, requirements 1 and 3)."""
        ts = TASK_SETS.get(set_id)
        if ts is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown set {set_id!r}")
            return
        task = ts.task(task_id)
        if task is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown task {task_id!r} in set {set_id!r}")
            return
        answers = load_answers(set_id)
        idx = next((i for i, t in enumerate(ts.tasks) if t["id"] == task_id), None)
        prev_id = ts.tasks[idx - 1]["id"] if idx is not None and idx > 0 else None
        next_id = ts.tasks[idx + 1]["id"] if idx is not None and idx + 1 < len(ts.tasks) else None

        view = build_task_view(ts, task)
        prior_answer = answers.get(task_id)
        payload = {
            "index": idx,
            "total": len(ts.tasks),
            "task": view,
            "prev_id": prev_id,
            "next_id": next_id,
            "previous_answer": prior_answer,
            "reveal": reveal_for_task(task) if prior_answer is not None else None,
        }
        self._send_json(payload)

    def _handle_clip(self, opaque: str):
        path = CLIP_INDEX.get(opaque)
        if path is None or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "unknown clip")
            return
        self._send_audio(path)

    def _handle_answer(self, set_id: str):
        ts = TASK_SETS.get(set_id)
        if ts is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown set {set_id!r}")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return

        task_id = body.get("task_id")
        task = ts.task(task_id) if task_id else None
        if task is None:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "unknown or missing task_id")
            return

        chosen_label = body.get("answer_label", "")
        chosen_role = body.get("answer_role")  # only meaningful for choose_clip mode
        listen_ms = int(body.get("listen_ms", 0) or 0)
        # Free-text observation, optional. Never required to submit an
        # answer, and adding/editing it alone does not touch is_correction /
        # answered_after_reveal — see _answer_value()/append_answer().
        note = str(body.get("note") or "").strip()
        skipped_prior = chosen_label == SKIP_LABEL

        if skipped_prior:
            matches, expected_display = None, task.get("hidden", {}).get("correct_answer")
        else:
            matches, expected_display = compute_matches_expected(
                task, ts, chosen_role if chosen_role else chosen_label, chosen_label
            )

        record = {
            "task_id": task_id,
            "set_id": set_id,
            "type": task["type"],
            "answer_kind": task.get("answer_kind"),
            "question": task["question"],
            "answer_label": chosen_label,
            "answer_role": chosen_role,
            "note": note,
            "listen_ms": listen_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hidden": task.get("hidden", {}),
            "correct_answer": expected_display,
            "matches_expected": matches,
            "skipped_prior": skipped_prior,
            # Baked in at answer time (like hidden/correct_answer) so the
            # historical record reflects what was actually known then, even
            # if a later pitch-cache refresh changes the live task view.
            "pitch_warning": task.get("pitch_warning"),
        }
        answers = append_answer(set_id, record)

        reveal = reveal_for_task(task)
        self._send_json({
            "ok": True,
            "reveal": reveal,
            "matches_expected": matches,
            "answered": len(answers),
            "total": len(ts.tasks),
        })

    def _handle_summary(self, set_id: str):
        ts = TASK_SETS.get(set_id)
        if ts is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown set {set_id!r}")
            return
        answers = load_answers(set_id)
        total = len(ts.tasks)
        fresh = [r for r in answers.values() if not r.get("skipped_prior")]
        skipped = len(answers) - len(fresh)
        # Corrections resubmitted after the first answer's reveal are no
        # longer blind (see append_answer's docstring) — the M4 gate must
        # count them separately, not silently fold them into the blind tally.
        blind = [r for r in fresh if not r.get("answered_after_reveal")]
        non_blind = [r for r in fresh if r.get("answered_after_reveal")]
        # Pitch-mismatched pairs (owner feedback #1, issue #57 follow-up):
        # the two clips being compared are not close enough in pitch for an
        # emotion/tag verdict to be trustworthy (voice register alone can
        # dominate the perceived difference). Never dropped from the answer
        # list, but excluded from the graded/differ_pairs pass-fail gate,
        # same treatment as answered_after_reveal.
        pitch_unreliable = [r for r in blind if r.get("pitch_warning")]
        reliable = [r for r in blind if not r.get("pitch_warning")]
        # Uses the LIVE task_sets definition (record_is_still_gradable()),
        # not just the record's own baked correct_answer -- so a task whose
        # expected answer has since been retracted (issue #57's
        # final-boundary-continuing turned out unanswerable, see
        # m4-tag-inventory-results.md sec. 5 item 6) stops counting toward
        # the gate without editing or dropping its historical record.
        graded = [r for r in reliable if record_is_still_gradable(ts, r)]
        correct = sum(1 for r in graded if r.get("matches_expected"))
        differ_pairs = [
            r for r in reliable
            if r.get("type") == "pair_compare" and r.get("answer_kind") == "differ"
        ]
        differed = sum(1 for r in differ_pairs if r.get("matches_expected"))
        self._send_json({
            "set_id": set_id,
            "title": ts.title,
            "total": total,
            "answered": len(answers),
            "skipped_prior": skipped,
            "answered_after_reveal": len(non_blind),
            "pitch_unreliable_total": len(pitch_unreliable),
            "graded_total": len(graded),
            "graded_correct": correct,
            "differ_pairs_total": len(differ_pairs),
            "differ_pairs_distinguished": differed,
            "gate_threshold_note": "M4-план требует >= 8 слепых пар и провал при >6 неразличённых из 8 (docs/research/audiobook/m4-plan.md, §2). Ответы, данные после раскрытия меток (исправления) или на парах с несовпадающей высотой голоса, в эту статистику не входят.",
            "answers": list(answers.values()),
        })


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    url = f"http://{HOST}:{port}/"
    print(f"Sentiment survey app: {len(TASK_SETS)} task set(s) loaded: {', '.join(TASK_SETS)}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Serving on {url} — Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
