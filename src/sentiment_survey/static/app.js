"use strict";

const state = {
  currentSetId: null,
  taskList: [],        // [{index, id, question, answered, is_correction, answered_after_reveal, matches_expected, skipped_prior}]
  currentIndex: -1,
  currentDetail: null,  // last /task/<id> response
  editMode: false,      // true = show the answer form again for an already-answered task
  listenMs: {},          // clip_url -> accumulated ms
  playStart: {},         // clip_url -> timestamp when play started
  audioEls: [],
};

const el = (id) => document.getElementById(id);

function show(sectionId) {
  for (const id of ["view-sets", "view-task", "view-done"]) {
    el(id).hidden = id !== sectionId;
  }
  el("home-btn").hidden = sectionId === "view-sets";
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str == null ? "" : str;
  return d.innerHTML;
}

// ---------------------------------------------------------------- sets list

async function loadSets() {
  show("view-sets");
  const list = el("sets-list");
  list.innerHTML = "<p>Загрузка…</p>";
  const { sets } = await api("/api/sets");
  if (!sets.length) {
    list.innerHTML = "<p>Наборы заданий не найдены. Проверьте output/ и src/sentiment_survey/task_sets/.</p>";
    return;
  }
  list.innerHTML = "";
  for (const s of sets) {
    const card = document.createElement("div");
    card.className = "set-card" + (s.answered >= s.total ? " done" : "");
    const pct = s.total ? Math.round((100 * s.answered) / s.total) : 0;
    card.innerHTML = `
      <h3>${escapeHtml(s.title)}</h3>
      <p>${escapeHtml(s.description || "")}</p>
      <p class="set-progress">${s.answered} / ${s.total} (${pct}%)${s.answered >= s.total ? " — пройдено, можно переслушать и исправить" : ""}</p>
    `;
    card.addEventListener("click", () => openSet(s.id));
    list.appendChild(card);
  }
}

// ---------------------------------------------------------------- task flow

async function openSet(setId) {
  state.currentSetId = setId;
  state.editMode = false;
  show("view-task");
  await refreshTaskList();
  if (!state.taskList.length) return;
  const firstUnanswered = state.taskList.findIndex((t) => !t.answered);
  await goToIndex(firstUnanswered >= 0 ? firstUnanswered : 0);
}

async function refreshTaskList() {
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/tasks`);
  state.taskList = data.tasks;
  updateProgress(data.answered, data.total);
  renderSidebar();
  return data;
}

function renderSidebar() {
  const box = el("task-sidebar-list");
  box.innerHTML = "";
  state.taskList.forEach((t, idx) => {
    const row = document.createElement("div");
    let cls = "sidebar-row";
    if (idx === state.currentIndex) cls += " current";
    if (t.skipped_prior) cls += " skipped";
    else if (t.answered_after_reveal) cls += " non-blind";
    else if (t.is_correction) cls += " corrected";
    else if (t.answered) cls += " answered";
    row.className = cls;
    const mark = t.skipped_prior ? "–" : t.answered ? "✓" : "·";
    const noteMark = t.has_note ? '<span class="row-note" title="Есть заметка">📝</span>' : "";
    row.innerHTML = `<span class="row-mark">${mark}</span><span class="row-idx">${idx + 1}.</span>`
      + `<span class="row-q">${escapeHtml(truncate(t.question, 60))}</span>${noteMark}`;
    row.title = t.answered
      ? (t.is_correction ? "Отвечено (исправлено) — нажмите, чтобы открыть" : "Отвечено — нажмите, чтобы открыть")
      : "Не отвечено — нажмите, чтобы открыть";
    row.addEventListener("click", () => goToIndex(idx));
    box.appendChild(row);
  });
}

function truncate(str, n) {
  if (!str) return "";
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

async function goToIndex(idx) {
  if (idx < 0 || idx >= state.taskList.length) return;
  state.currentIndex = idx;
  state.editMode = false;
  const taskId = state.taskList[idx].id;
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/task/${encodeURIComponent(taskId)}`);
  state.currentDetail = data;
  renderSidebar();
  renderNavBar();
  renderCurrentTask();
}

function renderNavBar() {
  const d = state.currentDetail;
  el("nav-position").textContent = `${d.index + 1} / ${d.total}`;
  el("prev-task-btn").disabled = d.prev_id == null;
  el("next-task-btn").disabled = d.next_id == null;
}

function updateProgress(answered, total) {
  el("progress-text").textContent = `${answered} / ${total}`;
  el("progress-fill").style.width = total ? `${(100 * answered) / total}%` : "0%";
}

function renderCurrentTask() {
  const d = state.currentDetail;
  el("reveal-card").hidden = true;
  if (d.previous_answer && !state.editMode) {
    el("task-card").hidden = true;
    renderAnswered(d);
  } else {
    el("answered-card").hidden = true;
    renderTaskForm(d.task);
  }
}

function renderAnswered(d) {
  const card = el("answered-card");
  card.hidden = false;
  const rec = d.previous_answer;
  const flag = el("answered-flag");
  if (rec.answered_after_reveal) {
    flag.textContent = "Этот ответ дан ПОСЛЕ раскрытия меток (исправление) — не считается слепым в итогах.";
    flag.className = "answered-flag non-blind";
  } else if (rec.skipped_prior) {
    flag.textContent = "Пропущено — уже был более ранний вердикт.";
    flag.className = "answered-flag";
  } else {
    flag.textContent = "Отвечено вслепую.";
    flag.className = "answered-flag";
  }
  let html = `<p><strong>Вопрос:</strong> ${escapeHtml(d.task.question)}</p>`;
  html += `<p><strong>Ваш ответ:</strong> ${escapeHtml(rec.answer_label)}</p>`;
  if (rec.correct_answer != null) {
    html += `<p><strong>Ожидалось:</strong> ${escapeHtml(rec.correct_answer)} — `
      + (rec.matches_expected ? `<span class="match-ok">совпало</span>` : `<span class="match-bad">не совпало</span>`) + `</p>`;
  }
  if (d.reveal && d.reveal.prior_verdict) {
    html += `<div class="prior-verdict"><strong>Более ранний вердикт:</strong> ${escapeHtml(d.reveal.prior_verdict)}</div>`;
  }
  if (d.reveal) {
    html += "<dl>";
    for (const [role, meta] of Object.entries(d.reveal.hidden || {})) {
      if (role === "correct_answer") continue;
      html += `<dt>${escapeHtml(role)}</dt><dd>${escapeHtml(JSON.stringify(meta))}</dd>`;
    }
    html += "</dl>";
  }
  el("answered-body").innerHTML = html;
  el("answered-note").value = rec.note || "";
}

async function saveNoteOnly() {
  const d = state.currentDetail;
  const rec = d.previous_answer;
  if (!rec) return;
  const body = {
    task_id: d.task.id,
    answer_label: rec.answer_label,
    listen_ms: 0,
    note: el("answered-note").value,
  };
  if (rec.answer_role != null) body.answer_role = rec.answer_role;
  await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  // Same answer, just a note change — stay on this task and refresh in
  // place rather than showing the reveal card (nothing new was revealed).
  await refreshTaskList();
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/task/${encodeURIComponent(d.task.id)}`);
  state.currentDetail = data;
  renderCurrentTask();
}

function renderTaskForm(task) {
  const card = el("task-card");
  card.hidden = false;
  state.listenMs = {};
  state.playStart = {};
  state.audioEls = [];

  el("task-question").textContent = task.question;
  el("prior-hint").hidden = !task.has_prior_verdict || state.editMode;
  // Carry the previous note forward into the answer-form textarea so
  // correcting the answer doesn't silently drop what was already written.
  const priorNote = (state.currentDetail.previous_answer && state.currentDetail.previous_answer.note) || "";
  el("task-note").value = priorNote;

  const slotsEl = el("task-slots");
  slotsEl.innerHTML = "";
  for (const slot of task.slots) {
    const div = document.createElement("div");
    div.className = "slot";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = slot.clip_url;
    trackListening(audio, slot.clip_url);
    div.innerHTML = `<h4>${escapeHtml(slot.label)}</h4>`;
    div.appendChild(audio);
    slotsEl.appendChild(div);
    state.audioEls.push(audio);
  }

  const optsEl = el("task-options");
  optsEl.innerHTML = "";
  task.options.forEach((opt, idx) => {
    const btn = document.createElement("button");
    const isSkip = opt.startsWith("Пропустить");
    if (isSkip) btn.classList.add("skip-option");
    const keyNum = idx + 1;
    btn.innerHTML = keyNum <= 9 ? `<span class="key-hint">${keyNum}</span>${escapeHtml(opt)}` : escapeHtml(opt);
    btn.addEventListener("click", () => submitAnswer(task, opt));
    optsEl.appendChild(btn);
  });
}

function trackListening(audio, key) {
  state.listenMs[key] = state.listenMs[key] || 0;
  audio.addEventListener("play", () => { state.playStart[key] = performance.now(); });
  const stop = () => {
    if (state.playStart[key] != null) {
      state.listenMs[key] += performance.now() - state.playStart[key];
      state.playStart[key] = null;
    }
  };
  audio.addEventListener("pause", stop);
  audio.addEventListener("ended", stop);
}

function totalListenMs() {
  let total = 0;
  const now = performance.now();
  for (const key of Object.keys(state.listenMs)) {
    total += state.listenMs[key];
    if (state.playStart[key] != null) total += now - state.playStart[key];
  }
  return Math.round(total);
}

function labelToRole(task, label) {
  const slot = task.slots.find((s) => s.label === label);
  return slot ? slot.role : null;
}

async function submitAnswer(task, optionLabel) {
  for (const a of state.audioEls) a.pause();
  const isSkip = optionLabel.startsWith("Пропустить");
  const body = {
    task_id: task.id,
    answer_label: optionLabel,
    listen_ms: totalListenMs(),
    note: el("task-note").value,
  };
  if (task.response_mode === "choose_clip" && !isSkip) {
    body.answer_role = labelToRole(task, optionLabel);
  }
  const result = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  updateProgress(result.answered, result.total);
  const isLastTask = state.currentDetail.next_id == null;
  await refreshTaskList();
  renderReveal(result.reveal, result.matches_expected, isSkip, isLastTask, body.note);
}

function renderReveal(reveal, matches, wasSkip, isLastTask, note) {
  el("task-card").hidden = true;
  el("answered-card").hidden = true;
  const revealCard = el("reveal-card");
  revealCard.hidden = false;
  const body = el("reveal-body");
  let html = "<dl>";
  for (const [role, meta] of Object.entries(reveal.hidden)) {
    if (role === "correct_answer") continue;
    html += `<dt>${escapeHtml(role)}</dt><dd>${escapeHtml(JSON.stringify(meta))}</dd>`;
  }
  html += "</dl>";
  if (!wasSkip && matches !== null && matches !== undefined) {
    html += matches
      ? `<p class="match-ok">Совпало с ожидаемым результатом.</p>`
      : `<p class="match-bad">НЕ совпало с ожидаемым результатом (ожидалось: ${escapeHtml(String(reveal.correct_answer))}).</p>`;
  }
  if (reveal.prior_verdict) {
    html += `<div class="prior-verdict"><strong>Более ранний вердикт:</strong> ${escapeHtml(reveal.prior_verdict)}</div>`;
  }
  if (note && note.trim()) {
    html += `<p><strong>Ваша заметка:</strong> «${escapeHtml(note.trim())}»</p>`;
  }
  body.innerHTML = html;
  el("next-btn").textContent = isLastTask ? "Итоги набора →" : "Далее →";
  el("next-btn").onclick = () => (isLastTask ? showSummary() : advanceAfterReveal());
}

async function advanceAfterReveal() {
  // Move forward by position (next_id from the task detail we already have)
  // rather than re-deriving "next unanswered" — this also works cleanly
  // right after a correction to an earlier task.
  const nextId = state.currentDetail.next_id;
  if (nextId == null) {
    await showSummary();
    return;
  }
  const nextIdx = state.taskList.findIndex((t) => t.id === nextId);
  await goToIndex(nextIdx >= 0 ? nextIdx : state.currentIndex);
}

// ---------------------------------------------------------------- summary

async function showSummary() {
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/summary`);
  show("view-done");
  const body = el("summary-body");
  let html = `<p>${data.answered} из ${data.total} заданий отвечено`;
  if (data.skipped_prior) html += `, из них ${data.skipped_prior} пропущено как уже известное`;
  html += `.</p>`;
  if (data.answered_after_reveal) {
    html += `<p class="hint">Из них ${data.answered_after_reveal} — исправления, данные уже после раскрытия меток; в статистику слепых ответов ниже они не входят.</p>`;
  }
  if (data.differ_pairs_total) {
    html += `<p><strong>Различил / не различил (для гейта сентимента):</strong> ${data.differ_pairs_distinguished} из ${data.differ_pairs_total} пар опознаны как различающиеся.</p>`;
    html += `<p class="hint">${escapeHtml(data.gate_threshold_note)}</p>`;
  }
  if (data.graded_total) {
    html += `<p><strong>Совпало с ожиданием:</strong> ${data.graded_correct} из ${data.graded_total}.</p>`;
  }
  body.innerHTML = html;
}

// ---------------------------------------------------------------- wiring

el("home-btn").addEventListener("click", () => {
  state.currentSetId = null;
  loadSets();
});
el("back-to-sets-btn").addEventListener("click", () => loadSets());
// next-btn's click handler is (re)assigned per-reveal in renderReveal(),
// since it toggles between "advance" and "show summary" depending on
// whether the just-answered task was the last one.
el("prev-task-btn").addEventListener("click", () => goToIndex(state.currentIndex - 1));
el("next-task-btn").addEventListener("click", () => goToIndex(state.currentIndex + 1));
el("edit-answer-btn").addEventListener("click", () => {
  state.editMode = true;
  renderCurrentTask();
});
el("save-note-btn").addEventListener("click", () => saveNoteOnly());
el("sidebar-toggle").addEventListener("click", () => {
  const sidebar = el("task-sidebar");
  const collapsed = sidebar.classList.toggle("collapsed");
  el("sidebar-toggle").textContent = collapsed ? "Показать" : "Скрыть";
});

document.addEventListener("keydown", (e) => {
  if (el("view-task").hidden) return;
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;

  if (!el("reveal-card").hidden) {
    if (e.key === "Enter") { e.preventDefault(); el("next-btn").click(); }
    return;
  }
  if (e.key === "ArrowLeft") { e.preventDefault(); goToIndex(state.currentIndex - 1); return; }
  if (e.key === "ArrowRight") { e.preventDefault(); goToIndex(state.currentIndex + 1); return; }
  if (e.key.toLowerCase() === "r" && state.audioEls.length) {
    state.audioEls[0].currentTime = 0;
    state.audioEls[0].play();
    return;
  }
  if (!el("task-card").hidden) {
    const n = parseInt(e.key, 10);
    if (!Number.isNaN(n)) {
      const buttons = el("task-options").querySelectorAll("button");
      if (buttons[n - 1]) buttons[n - 1].click();
    }
  }
});

loadSets();
