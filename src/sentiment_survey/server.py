#!/usr/bin/env python3
"""Local blind-listening survey app for Higgs sentiment-tag verification (issue #57).

Stdlib-only (http.server + json), no framework. Serves a small single-page UI that:
  - plays one task's clip(s) with tag identity hidden (opaque clip ids, no filenames
    leaked to the browser),
  - lets the owner answer without knowing which variant/tag they're hearing,
  - reveals the hidden metadata only after the answer is submitted,
  - writes every answer to disk immediately (atomic temp-file + os.replace), so an
    interrupted session never corrupts the results file and can be resumed later.

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
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog  # noqa: E402  (needs sys.path tweak above; same-directory module)

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


def load_task_sets() -> dict[str, TaskSet]:
    """Hand-written JSON sets (task_sets/*.json) plus auto-discovered sets
    scanned live from output/ (catalog.build_all_dynamic_sets()). A dynamic
    set with the same id as a JSON one is skipped with a warning — JSON wins,
    since it means someone deliberately curated that set by hand."""
    sets: dict[str, TaskSet] = {}
    for path in sorted(TASK_SETS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        ts = TaskSet(doc, path)
        sets[ts.id] = ts

    for doc in catalog.build_all_dynamic_sets():
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


def load_answers(set_id: str) -> dict[str, dict]:
    jsonl_path, _ = results_paths(set_id)
    answers: dict[str, dict] = {}
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answers[rec["task_id"]] = rec
    return answers


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_answer(set_id: str, record: dict) -> dict[str, dict]:
    """Record one answer atomically and return the full up-to-date answer map."""
    jsonl_path, md_path = results_paths(set_id)
    answers = load_answers(set_id)
    answers[record["task_id"]] = record

    # Rewrite the whole JSONL from the in-memory map: small files (tens of
    # entries), so a full atomic rewrite per answer is simpler and safer than
    # a bare append (which is not itself atomic against a mid-write crash).
    lines = [json.dumps(answers[tid], ensure_ascii=False) for tid in answers]
    atomic_write(jsonl_path, "\n".join(lines) + "\n" if lines else "")

    atomic_write(md_path, render_markdown(set_id, answers))
    return answers


def render_markdown(set_id: str, answers: dict[str, dict]) -> str:
    ts = TASK_SETS.get(set_id)
    title = ts.title if ts else set_id
    total = len(ts.tasks) if ts else len(answers)
    lines = [f"# {title}", "", f"Отвечено {len(answers)} из {total}.", ""]
    differed, not_differed, correct, incorrect, skipped = 0, 0, 0, 0, 0
    for rec in answers.values():
        if rec.get("skipped_prior"):
            skipped += 1
            continue
        exp = rec.get("correct_answer")
        if exp is not None:
            if rec.get("matches_expected"):
                correct += 1
            else:
                incorrect += 1
        if rec.get("type") == "pair_compare" and rec.get("answer_kind") == "differ":
            if rec.get("matches_expected"):
                differed += 1
            elif exp is not None:
                not_differed += 1
    if correct or incorrect:
        lines.append(f"**Совпало с ожиданием (свежие ответы, без пропущенных): {correct} из {correct + incorrect}.**")
        lines.append("")
    if skipped:
        lines.append(f"**Пропущено как уже подтверждённое ранее: {skipped}.**")
        lines.append("")
    lines.append("| Задание | Тип | Ответ | Ожидалось | Совпало | Время прослушивания | Отметка времени |")
    lines.append("|---|---|---|---|---|---|---|")
    for tid, rec in answers.items():
        exp = rec.get("correct_answer") or ""
        if rec.get("skipped_prior"):
            match = "пропущено"
        else:
            match = "" if exp == "" else ("да" if rec.get("matches_expected") else "нет")
        lines.append(
            f"| `{tid}` | {rec.get('type', '')} | {rec.get('answer_label', rec.get('answer', ''))} "
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
            "listen_ms": listen_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hidden": task.get("hidden", {}),
            "correct_answer": expected_display,
            "matches_expected": matches,
            "skipped_prior": skipped_prior,
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
        graded = [r for r in fresh if r.get("correct_answer") is not None]
        correct = sum(1 for r in graded if r.get("matches_expected"))
        differ_pairs = [
            r for r in fresh
            if r.get("type") == "pair_compare" and r.get("answer_kind") == "differ"
        ]
        differed = sum(1 for r in differ_pairs if r.get("matches_expected"))
        self._send_json({
            "set_id": set_id,
            "title": ts.title,
            "total": total,
            "answered": len(answers),
            "skipped_prior": skipped,
            "graded_total": len(graded),
            "graded_correct": correct,
            "differ_pairs_total": len(differ_pairs),
            "differ_pairs_distinguished": differed,
            "gate_threshold_note": "M4-план требует >= 8 слепых пар и провал при >6 неразличённых из 8 (docs/research/audiobook/m4-plan.md, §2).",
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
