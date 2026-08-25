"use strict";

const state = {
  currentSetId: null,
  currentTask: null,
  listenMs: {},       // clip_url -> accumulated ms
  playStart: {},       // clip_url -> timestamp when play started
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
      <p class="set-progress">${s.answered} / ${s.total} (${pct}%)${s.answered >= s.total ? " — пройдено, можно переслушать" : ""}</p>
    `;
    card.addEventListener("click", () => openSet(s.id));
    list.appendChild(card);
  }
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

// ---------------------------------------------------------------- task flow

async function openSet(setId) {
  state.currentSetId = setId;
  show("view-task");
  await loadNextTask();
}

async function loadNextTask() {
  el("reveal-card").hidden = true;
  el("task-card").hidden = false;
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/next`);
  updateProgress(data.answered, data.total);
  if (data.done) {
    await showSummary();
    return;
  }
  renderTask(data.task);
}

function updateProgress(answered, total) {
  el("progress-text").textContent = `${answered} / ${total}`;
  el("progress-fill").style.width = total ? `${(100 * answered) / total}%` : "0%";
}

function renderTask(task) {
  state.currentTask = task;
  state.listenMs = {};
  state.playStart = {};
  state.audioEls = [];

  el("task-question").textContent = task.question;
  el("prior-hint").hidden = !task.has_prior_verdict;

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
  renderReveal(result.reveal, result.matches_expected, isSkip);
}

function renderReveal(reveal, matches, wasSkip) {
  el("task-card").hidden = true;
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
  body.innerHTML = html;
}

// ---------------------------------------------------------------- summary

async function showSummary() {
  const data = await api(`/api/sets/${encodeURIComponent(state.currentSetId)}/summary`);
  show("view-done");
  const body = el("summary-body");
  let html = `<p>${data.answered} из ${data.total} заданий отвечено`;
  if (data.skipped_prior) html += `, из них ${data.skipped_prior} пропущено как уже известное`;
  html += `.</p>`;
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
el("next-btn").addEventListener("click", () => loadNextTask());

document.addEventListener("keydown", (e) => {
  if (!el("view-task").hidden) {
    if (!el("reveal-card").hidden) {
      if (e.key === "Enter") { e.preventDefault(); loadNextTask(); }
      return;
    }
    if (e.key.toLowerCase() === "r" && state.audioEls.length) {
      state.audioEls[0].currentTime = 0;
      state.audioEls[0].play();
      return;
    }
    const n = parseInt(e.key, 10);
    if (!Number.isNaN(n)) {
      const buttons = el("task-options").querySelectorAll("button");
      if (buttons[n - 1]) buttons[n - 1].click();
    }
  }
});

loadSets();
