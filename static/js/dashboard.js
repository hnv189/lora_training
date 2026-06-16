/* LoRA Training Dashboard — front-end logic.
 * Fetches analysis from the Flask API and renders Chart.js visualisations.
 */

const C = {
  accent: "#5b9dff",
  green: "#3ecf8e",
  amber: "#f5b14c",
  red: "#ff6b6b",
  grid: "rgba(255,255,255,0.06)",
  muted: "#8a97ab",
};
Chart.defaults.color = C.muted;
Chart.defaults.font.family = "-apple-system, Segoe UI, Roboto, sans-serif";
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const charts = {};
const loaded = {};

function fmt(n, d = 3) {
  return n === null || n === undefined ? "—" : Number(n).toFixed(d);
}
function statusClass(s) {
  return s === "healthy" ? "good" : s === "warning" ? "warn" : "bad";
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

/* ---------------- Tabs ---------------- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    const id = tab.dataset.tab;
    document.getElementById("tab-" + id).classList.add("active");
    load(id);
  });
});

function load(tab) {
  const map = {
    train: loadTrain,
    loss: loadMetrics,
    dynamics: loadMetrics,
    audit: loadAudit,
    prepare: loadPrepare,
    eval: loadEval,
    models: loadModels,
    config: loadConfig,
  };
  (map[tab] || (() => {}))();
}

/* ---------------- Training pipeline ---------------- */
let trainPoll = null;

async function loadTrain(force) {
  if (loaded.train && !force) return;
  loaded.train = true;
  const env = await getJSON("/api/train/environment");
  document.getElementById("envBadges").innerHTML = `
    <span class="env-badge ${env.gpu_available ? "on" : "off"}">GPU ${env.gpu_available ? "detected" : "none"}</span>
    <span class="env-badge ${env.training_deps_available ? "on" : "off"}">ML deps ${env.training_deps_available ? "ready" : "missing"}</span>
    <span class="env-badge">auto → ${env.resolved_auto_mode}</span>`;
  if (!env.gpu_available) {
    document.getElementById("cfgMode").value = "simulate";
    document.getElementById("trainNote").textContent =
      "No GPU here, so runs use simulation mode — it writes a real trainer_state.json " +
      "incrementally so the live loss chart and analysis are fully exercised. On your " +
      "RTX 2000 Ada box (with requirements-train.txt installed) pick Real to fine-tune Qwen3-8B.";
  }
  refreshLocalModels();
  refreshTrainStatus();
}

document.getElementById("startBtn").addEventListener("click", async () => {
  const config = {
    mode: document.getElementById("cfgMode").value,
    base_model: document.getElementById("cfgBaseModel").value,
    epochs: parseInt(document.getElementById("cfgEpochs").value, 10),
    rank: parseInt(document.getElementById("cfgRank").value, 10),
    learning_rate: parseFloat(document.getElementById("cfgLr").value),
    step_delay: parseFloat(document.getElementById("cfgDelay").value),
  };
  const r = await getJSON("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (r.error) { alert(r.error); return; }
  startPolling();
});

document.getElementById("stopBtn").addEventListener("click", async () => {
  await fetch("/api/train/stop", { method: "POST" });
  refreshTrainStatus();
});

function startPolling() {
  if (trainPoll) clearInterval(trainPoll);
  refreshTrainStatus();
  trainPoll = setInterval(refreshTrainStatus, 1200);
}

async function refreshTrainStatus() {
  const d = await getJSON("/api/train/status");
  if (d.error) return;
  const job = d.job || { status: "idle" };
  const running = job.status === "running";

  const pill = document.getElementById("statusPill");
  pill.className = "status-pill " + job.status;
  pill.textContent = job.status;

  document.getElementById("statusMeta").textContent = job.mode
    ? `mode: ${job.mode}${job.config ? ` · ${job.config.epochs} epochs · rank ${job.config.rank}` : ""}` +
      (job.config && job.config.base_model ? ` · base: ${job.config.base_model}` : "")
    : "";

  document.getElementById("startBtn").disabled = running;
  document.getElementById("stopBtn").disabled = !running;

  const m = d.metrics;
  const bar = document.getElementById("progressBar");
  if (m) {
    renderTrainCards(m.summary);
    renderLiveLoss(m.series);
    // Progress = elapsed epochs / configured epochs.
    const epochs = job.config ? job.config.epochs : m.summary.total_epochs;
    const frac = epochs ? m.summary.total_epochs / epochs : 0;
    bar.style.width = (running ? Math.min(99, Math.round(frac * 100)) : 100) + "%";
  } else {
    bar.style.width = "0%";
  }

  const logEl = document.getElementById("trainLog");
  logEl.textContent = d.log || "Waiting for log output…";
  logEl.scrollTop = logEl.scrollHeight;

  if (!running && trainPoll) {
    clearInterval(trainPoll);
    trainPoll = null;
    loaded.metrics = false; // refresh Loss Analysis tab on next visit
  }
}

function renderTrainCards(s) {
  const cards = [
    { label: "Step", value: s.total_steps, cls: "" },
    { label: "Epoch", value: fmt(s.total_epochs, 2), cls: "" },
    { label: "Train loss", value: fmt(s.final_train_loss), cls: "good" },
    { label: "Eval loss", value: fmt(s.final_eval_loss), cls: "good" },
    { label: "Best eval", value: fmt(s.best_eval_loss), cls: "good" },
  ];
  document.getElementById("trainCards").innerHTML = cards.map((c) => `
    <div class="card"><div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div></div>`).join("");
}

function renderLiveLoss(series) {
  const train = series.train_loss.map((p) => ({ x: p.epoch, y: p.value }));
  const evalp = series.eval_loss.map((p) => ({ x: p.epoch, y: p.value }));
  lineChart("liveLossChart", [
    { label: "Train loss", data: train, borderColor: C.accent, backgroundColor: "transparent" },
    { label: "Eval loss", data: evalp, borderColor: C.amber, backgroundColor: "transparent",
      pointRadius: 4, pointBackgroundColor: C.amber },
  ], "epoch");
}

/* ---------------- Loss + Dynamics ---------------- */
async function loadMetrics(force) {
  if (loaded.metrics && !force) return;
  const data = await getJSON("/api/metrics");
  if (data.error) return;
  loaded.metrics = true;

  document.getElementById("sourcePill").textContent =
    data.source === "uploaded" ? "uploaded run" : "sample run";

  renderLossCards(data.summary, data.best_checkpoint);
  renderLossChart(data.series);
  renderGapChart(data.series.loss_gap);
  renderHealth("lossHealth", data.health.filter((h) =>
    ["Train loss", "Eval loss", "Train/Eval gap"].includes(h.metric)));

  renderGradChart(data.series.grad_norm);
  renderLrChart(data.series.learning_rate);
  renderHealth("dynamicsHealth", data.health.filter((h) => h.metric === "Gradient norm"));
}

function renderLossCards(s, best) {
  const overfit = s.final_eval_loss && s.best_eval_loss &&
    s.final_eval_loss - s.best_eval_loss > 0.03;
  const cards = [
    { label: "Final Train Loss", value: fmt(s.final_train_loss), cls: "good" },
    { label: "Final Eval Loss", value: fmt(s.final_eval_loss), cls: overfit ? "warn" : "good" },
    { label: "Best Eval Loss", value: fmt(s.best_eval_loss), cls: "good",
      delta: best ? `epoch ${best.epoch}` : "" },
    { label: "Peak Grad Norm", value: fmt(s.peak_grad_norm, 2),
      cls: s.peak_grad_norm > 2 ? "bad" : "good" },
    { label: "Epochs", value: fmt(s.total_epochs, 0), cls: "" },
    { label: "Steps", value: s.total_steps, cls: "" },
  ];
  document.getElementById("lossCards").innerHTML = cards.map((c) => `
    <div class="card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      ${c.delta ? `<div class="delta ${c.cls}">${c.delta}</div>` : ""}
    </div>`).join("");
}

function lineChart(id, datasets, xLabel) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "linear", title: { display: true, text: xLabel }, grid: { color: C.grid } },
        y: { grid: { color: C.grid } },
      },
      elements: { point: { radius: 0 }, line: { tension: 0.3, borderWidth: 2 } },
    },
  });
}

function renderLossChart(series) {
  const train = series.train_loss.map((p) => ({ x: p.epoch, y: p.value }));
  const evalp = series.eval_loss.map((p) => ({ x: p.epoch, y: p.value }));
  lineChart("lossChart", [
    { label: "Train loss", data: train, borderColor: C.accent,
      backgroundColor: "transparent" },
    { label: "Eval loss", data: evalp, borderColor: C.amber,
      backgroundColor: "transparent",
      pointRadius: 4, pointBackgroundColor: C.amber },
  ], "epoch");
}

function renderGapChart(gap) {
  const data = gap.map((p) => ({ x: p.epoch, y: Math.abs(p.value) }));
  if (charts.gapChart) charts.gapChart.destroy();
  charts.gapChart = new Chart(document.getElementById("gapChart"), {
    type: "bar",
    data: {
      datasets: [{
        label: "|train − eval|", data,
        backgroundColor: data.map((p) =>
          p.y > 0.5 ? C.red : p.y > 0.3 ? C.amber : C.green),
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "epoch" }, grid: { color: C.grid } },
        y: { grid: { color: C.grid },
          suggestedMax: 0.6,
          // 0.5 overfit line via annotation-free approach
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderGradChart(grad) {
  const data = grad.map((p) => ({ x: p.step, y: p.value }));
  lineChart("gradChart", [{
    label: "grad norm", data, borderColor: C.green, backgroundColor: "transparent",
  }], "step");
  charts.gradChart.options.scales.y.suggestedMax = 0.6;
  charts.gradChart.update();
}

function renderLrChart(lr) {
  const data = lr.map((p) => ({ x: p.step, y: p.value }));
  lineChart("lrChart", [{
    label: "learning rate", data, borderColor: C.accent, backgroundColor: "transparent",
  }], "step");
}

function renderHealth(elId, items) {
  document.getElementById(elId).innerHTML = items.map((h) => `
    <div class="health-item ${h.status}">
      <span class="dot"></span>
      <span class="metric">${h.metric}</span>
      <span class="msg">${h.message}</span>
    </div>`).join("");
}

/* Upload handling */
document.getElementById("metricsUpload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const data = await getJSON("/api/metrics/upload", { method: "POST", body: fd });
  if (data.error) { alert("Upload error: " + data.error); return; }
  loaded.metrics = false;
  loadMetrics(true);
});
document.getElementById("resetMetrics").addEventListener("click", async () => {
  await fetch("/api/metrics/reset", { method: "POST" });
  loaded.metrics = false;
  loadMetrics(true);
});

/* ---------------- Data Audit ---------------- */
async function loadAudit() {
  if (loaded.audit) return;
  const d = await getJSON("/api/audit");
  if (d.error) return;
  loaded.audit = true;

  const ts = d.token_stats || {};
  const readyMap = { good: "good", viable: "warn", collect_more: "bad", no_data: "bad" };
  const cards = [
    { label: "Total Pairs", value: d.pair_count, cls: readyMap[d.readiness] },
    { label: "Readiness", value: d.readiness.replace("_", " "), cls: readyMap[d.readiness], sm: true },
    { label: "Median Tokens", value: ts.median ?? "—", cls: "" },
    { label: "P95 Tokens", value: ts.p95 ?? "—", cls: "" },
    { label: "Rec. max_seq_length", value: ts.recommended_max_seq_length ?? "—", cls: "good" },
    { label: "Balanced?", value: d.categories.balanced ? "yes" : "no",
      cls: d.categories.balanced ? "good" : "warn", sm: true },
  ];
  document.getElementById("auditCards").innerHTML = cards.map((c) => `
    <div class="card"><div class="label">${c.label}</div>
      <div class="value ${c.sm ? "sm" : ""} ${c.cls}">${c.value}</div></div>`).join("");

  renderTokenChart(ts.histogram || []);
  renderCatChart(d.categories.categories || []);
  renderFingerprint(d.fingerprint);
}

function renderTokenChart(hist) {
  if (charts.tokenChart) charts.tokenChart.destroy();
  charts.tokenChart = new Chart(document.getElementById("tokenChart"), {
    type: "bar",
    data: {
      labels: hist.map((h) => h.range),
      datasets: [{ label: "pairs", data: hist.map((h) => h.count), backgroundColor: C.accent }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { grid: { display: false } }, y: { grid: { color: C.grid } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderCatChart(cats) {
  if (charts.catChart) charts.catChart.destroy();
  charts.catChart = new Chart(document.getElementById("catChart"), {
    type: "doughnut",
    data: {
      labels: cats.map((c) => c.category),
      datasets: [{
        data: cats.map((c) => c.count),
        backgroundColor: [C.accent, C.green, C.amber, C.red, "#9b7bff", "#4ec8d8"],
      }],
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } } },
  });
}

function renderFingerprint(fp) {
  const items = [
    ["Naming convention", fp.naming_convention],
    ["Error handling style", fp.error_handling_style],
    ["Uses docstrings", fp.uses_docstrings ? "yes" : "no"],
    ["Uses inline comments", fp.uses_inline_comments ? "yes" : "no"],
    ["Avg docstrings/impl", fp.avg_docstrings_per_impl],
  ];
  document.getElementById("fingerprint").innerHTML = `
    <h3>Behavioural Fingerprint <span class="hint">your evaluation checklist</span></h3>
    <div class="fp-grid">
      ${items.map(([k, v]) => `<div class="fp-item"><span class="k">${k}:</span><span class="v">${v}</span></div>`).join("")}
    </div>
    <ul class="checklist">
      ${fp.eval_checklist.map((c) => `<li>${c}</li>`).join("")}
    </ul>`;
}

/* ---------------- Data Prep ---------------- */
async function loadPrepare() {
  if (loaded.prepare) return;
  const d = await getJSON("/api/prepare");
  if (d.error) return;
  loaded.prepare = true;

  const cards = [
    { label: "Total Samples", value: d.total, cls: "" },
    { label: "Kept (quality)", value: d.kept, cls: "good" },
    { label: "Dropped", value: d.dropped, cls: d.dropped > 0 ? "warn" : "good" },
    { label: "Reasoning Traces", value: d.has_reasoning ? "present" : "missing",
      cls: d.has_reasoning ? "good" : "bad", sm: true },
  ];
  document.getElementById("prepCards").innerHTML = cards.map((c) => `
    <div class="card"><div class="label">${c.label}</div>
      <div class="value ${c.sm ? "sm" : ""} ${c.cls}">${c.value}</div></div>`).join("");

  const dd = document.getElementById("droppedDetail");
  dd.innerHTML = d.dropped_detail.length
    ? `<h3 class="block-title">Dropped samples</h3><ul>${d.dropped_detail.map((x) =>
        `<li><code>${x.name}</code> — ${x.reason}</li>`).join("")}</ul>`
    : "";
  document.getElementById("promptPreview").textContent = d.example_prompt;
}

/* ---------------- Behaviour Eval ---------------- */
async function loadEval() {
  if (loaded.eval) return;
  const d = await getJSON("/api/eval");
  if (d.error) return;
  loaded.eval = true;

  const cards = [
    { label: "Avg Checklist", value: fmt(d.avg_checklist, 2),
      cls: d.checklist_target_met ? "good" : "warn", delta: "target >0.75" },
    { label: "Avg CodeBLEU", value: fmt(d.avg_codebleu, 2),
      cls: d.codebleu_target_met ? "good" : "warn", delta: "target >0.60" },
    { label: "OOD Pass Rate", value: d.ood_pass_rate === null ? "—" : fmt(d.ood_pass_rate, 2),
      cls: (d.ood_pass_rate ?? 0) >= 0.75 ? "good" : "warn", delta: `${d.ood_total} OOD defs` },
    { label: "Samples", value: d.count, cls: "" },
  ];
  document.getElementById("evalCards").innerHTML = cards.map((c) => `
    <div class="card"><div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      ${c.delta ? `<div class="delta ${c.cls}">${c.delta}</div>` : ""}</div>`).join("");

  renderEvalChart(d.samples);
  renderEvalTable(d.samples);
  renderFailTable(d.failure_modes);
}

function renderEvalChart(samples) {
  if (charts.evalChart) charts.evalChart.destroy();
  charts.evalChart = new Chart(document.getElementById("evalChart"), {
    type: "bar",
    data: {
      labels: samples.map((s) => s.name),
      datasets: [
        { label: "Checklist avg", data: samples.map((s) => s.checklist_avg), backgroundColor: C.accent },
        { label: "CodeBLEU", data: samples.map((s) => s.codebleu), backgroundColor: C.green },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { suggestedMax: 1, grid: { color: C.grid } }, x: { grid: { display: false } } },
    },
  });
}

function renderEvalTable(samples) {
  document.getElementById("evalTable").innerHTML = `
    <thead><tr><th>Sample</th><th>Checklist</th><th>CodeBLEU</th><th>Type</th><th>Result</th></tr></thead>
    <tbody>${samples.map((s) => `
      <tr>
        <td>${s.name}</td>
        <td>${fmt(s.checklist_avg, 2)}</td>
        <td>${fmt(s.codebleu, 2)}</td>
        <td>${s.ood ? '<span class="badge ood">OOD</span>' : "in-dist"}</td>
        <td><span class="badge ${s.pass ? "pass" : "fail"}">${s.pass ? "pass" : "review"}</span></td>
      </tr>`).join("")}</tbody>`;
}

function renderFailTable(modes) {
  document.getElementById("failTable").innerHTML = `
    <thead><tr><th>Symptom</th><th>Likely cause</th><th>Fix</th></tr></thead>
    <tbody>${modes.map((m) => `
      <tr><td>${m.symptom}</td><td>${m.cause}</td><td>${m.fix}</td></tr>`).join("")}</tbody>`;
}

/* ---------------- Config ---------------- */
async function loadConfig() {
  if (loaded.config) return;
  const d = await getJSON("/api/config");
  loaded.config = true;
  document.getElementById("configView").innerHTML =
    `<pre>${JSON.stringify(d, null, 2)}</pre>`;
}

/* ---------------- Models ---------------- */
async function loadModels() {
  refreshLocalModels();
}

document.getElementById("modelSearchBtn").addEventListener("click", () => searchModels());
document.getElementById("modelSearchQuery").addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchModels();
});

async function searchModels() {
  const q = document.getElementById("modelSearchQuery").value.trim();
  const table = document.getElementById("modelSearchTable");
  if (!q) { table.innerHTML = ""; return; }
  table.innerHTML = `<tbody><tr><td>Searching…</td></tr></tbody>`;
  const d = await getJSON(`/api/models/search?q=${encodeURIComponent(q)}`);
  if (d.error) { table.innerHTML = `<tbody><tr><td>${d.error}</td></tr></tbody>`; return; }
  if (!d.results.length) {
    table.innerHTML = `<tbody><tr><td>No results with a supported format (.safetensors / .gguf).</td></tr></tbody>`;
    return;
  }
  table.innerHTML = `
    <thead><tr><th>Repo ID</th><th>Format</th><th>Downloads</th><th>Likes</th><th></th></tr></thead>
    <tbody>${d.results.map((m) => `
      <tr>
        <td>${m.repo_id}</td>
        <td>${m.extensions.join(", ")}</td>
        <td>${m.downloads ?? "—"}</td>
        <td>${m.likes ?? "—"}</td>
        <td>${m.local
          ? `<span class="env-badge on">downloaded</span>`
          : `<button class="ghost search-fetch-btn" data-repo="${m.repo_id}">⬇ Download</button>`}</td>
      </tr>`).join("")}</tbody>`;
  table.querySelectorAll(".search-fetch-btn").forEach((btn) => {
    btn.addEventListener("click", () => fetchModel(btn.dataset.repo, btn));
  });
}

let fetchModelPoll = null;

async function fetchModel(repo_id, triggerBtn) {
  const body = {
    repo_id,
    revision: document.getElementById("fetchRevision").value.trim(),
    token: document.getElementById("fetchToken").value.trim(),
  };
  const r = await getJSON("/api/models/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.error) { document.getElementById("fetchNote").textContent = r.error; return; }
  document.getElementById("fetchModelBtn").disabled = true;
  if (triggerBtn) { triggerBtn.disabled = true; triggerBtn.textContent = "Downloading…"; }
  if (fetchModelPoll) clearInterval(fetchModelPoll);
  fetchModelPoll = setInterval(refreshFetchStatus, 1500);
  refreshFetchStatus();
}

document.getElementById("fetchModelBtn").addEventListener("click", () => {
  const repo_id = document.getElementById("fetchRepoId").value.trim();
  if (!repo_id) { alert("Enter a repo id, e.g. Qwen/Qwen3-8B"); return; }
  fetchModel(repo_id);
});

async function refreshFetchStatus() {
  const d = await getJSON("/api/models/fetch/status");
  const note = document.getElementById("fetchNote");
  if (d.status === "running") {
    note.textContent = `Downloading ${d.repo_id}…`;
  } else if (d.status === "completed") {
    note.textContent = `Downloaded ${d.repo_id} → ${d.local_dir}`;
    clearInterval(fetchModelPoll);
    document.getElementById("fetchModelBtn").disabled = false;
    refreshLocalModels();
    document.querySelectorAll(".search-fetch-btn").forEach((btn) => {
      if (btn.dataset.repo === d.repo_id) {
        btn.outerHTML = `<span class="env-badge on">downloaded</span>`;
      }
    });
  } else if (d.status === "failed") {
    note.textContent = `Failed: ${d.error}`;
    clearInterval(fetchModelPoll);
    document.getElementById("fetchModelBtn").disabled = false;
  }
}

async function refreshLocalModels() {
  const d = await getJSON("/api/models");
  const table = document.getElementById("localModelsTable");
  if (!d.deps_available) {
    table.innerHTML = `<tbody><tr><td>huggingface_hub not installed — run: pip install -r requirements.txt</td></tr></tbody>`;
  } else if (!d.local.length) {
    table.innerHTML = `<tbody><tr><td>No models downloaded yet.</td></tr></tbody>`;
  } else {
    table.innerHTML = `
      <thead><tr><th>Repo ID</th><th>Local path</th></tr></thead>
      <tbody>${d.local.map((m) => `<tr><td>${m.repo_id}</td><td>${m.path}</td></tr>`).join("")}</tbody>`;
  }
  updateBaseModelOptions(d.local || []);
}

function updateBaseModelOptions(localModels) {
  const select = document.getElementById("cfgBaseModel");
  const current = select.value;
  select.innerHTML = `<option value="">Default (Qwen/Qwen3-8B from Hub)</option>` +
    localModels.map((m) => `<option value="${m.path}">${m.repo_id}</option>`).join("");
  if (localModels.some((m) => m.path === current)) select.value = current;
}

/* Initial load — Train is the default tab. Resume polling if a run is live. */
(async function init() {
  await loadTrain();
  const d = await getJSON("/api/train/status");
  if (d.job && d.job.status === "running") startPolling();
})();
