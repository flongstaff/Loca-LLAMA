import { api } from "./api.js";
import { escapeHtml } from "./utils.js";
import { drawBarChart, drawLineChart } from "./chart.js";

let benchRuntimes = [];
let benchJobId = null;
let benchPollTimer = null;
let streamSource = null;
let sweepJobId = null;
let sweepPollTimer = null;
let lastBenchResult = null;
let lastSweepResult = null;
let lastSweepAggregates = null;

async function detectRuntimes() {
  const btn = document.getElementById("detect-runtimes-btn");
  const runtimeSelect = document.getElementById("bench-runtime-select");
  const statusDiv = document.getElementById("bench-status");

  btn.disabled = true;
  btn.textContent = "Detecting…";
  statusDiv.innerHTML = '<p class="loading">Probing local runtimes…</p>';

  try {
    const data = await api.get("/runtime/status");
    benchRuntimes = data.runtimes;

    if (data.count === 0) {
      statusDiv.innerHTML = '<p class="placeholder">No runtimes detected. Start LM Studio or llama.cpp server first.</p>';
      runtimeSelect.disabled = true;
      return;
    }

    runtimeSelect.innerHTML = '<option value="">Select runtime…</option>';
    benchRuntimes.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = `${r.name} (${r.models.length} model${r.models.length !== 1 ? "s" : ""})`;
      runtimeSelect.appendChild(opt);
    });
    runtimeSelect.disabled = false;
    statusDiv.innerHTML = `<p class="text-muted">${data.count} runtime${data.count !== 1 ? "s" : ""} detected.</p>`;
  } catch (err) {
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Detect Runtimes";
  }
}

function onRuntimeChange() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelSelect = document.getElementById("bench-model-select");
  const startBtn = document.getElementById("bench-start-btn");
  const streamBtn = document.getElementById("bench-stream-btn");

  if (!runtimeName) {
    modelSelect.innerHTML = '<option value="">Select runtime first…</option>';
    modelSelect.disabled = true;
    startBtn.disabled = true;
    streamBtn.disabled = true;
    return;
  }

  const rt = benchRuntimes.find((r) => r.name === runtimeName);
  if (!rt || rt.models.length === 0) {
    modelSelect.innerHTML = '<option value="">No models loaded</option>';
    modelSelect.disabled = true;
    startBtn.disabled = true;
    streamBtn.disabled = true;
    return;
  }

  modelSelect.innerHTML = "";
  rt.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });
  modelSelect.disabled = false;
  startBtn.disabled = false;
  streamBtn.disabled = false;

  // Populate sweep model checkboxes
  populateSweepModels(rt.models);
}

async function startBenchmark() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelId = document.getElementById("bench-model-select").value;
  const promptType = document.getElementById("bench-prompt-select").value;
  const numRuns = parseInt(document.getElementById("bench-runs-input").value, 10) || 3;

  if (!runtimeName || !modelId) return;

  const startBtn = document.getElementById("bench-start-btn");
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-results");

  startBtn.disabled = true;
  startBtn.textContent = "Running…";
  resultsDiv.innerHTML = "";
  statusDiv.innerHTML = '<p class="loading">Starting benchmark…</p>';

  const body = {
    runtime_name: runtimeName,
    model_id: modelId,
    prompt_type: promptType,
    num_runs: numRuns,
  };
  if (promptType === "custom") {
    body.custom_prompt = document.getElementById("bench-custom-prompt-text").value;
  }

  try {
    const data = await api.post("/benchmark/start", body);

    benchJobId = data.job_id;
    pollBenchmarkStatus();
  } catch (err) {
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
    startBtn.disabled = false;
    startBtn.textContent = "Start Benchmark";
  }
}

function pollBenchmarkStatus() {
  if (!benchJobId) return;

  clearTimeout(benchPollTimer);

  const poll = async () => {
    try {
      const data = await api.get(`/benchmark/${benchJobId}`);
      renderBenchmarkStatus(data);

      if (data.status === "running") {
        benchPollTimer = setTimeout(poll, 1000);
      } else {
        document.getElementById("bench-start-btn").disabled = false;
        document.getElementById("bench-start-btn").textContent = "Start Benchmark";
      }
    } catch (err) {
      document.getElementById("bench-status").innerHTML =
        `<p class="error-message">${escapeHtml(err.message)}</p>`;
      document.getElementById("bench-start-btn").disabled = false;
      document.getElementById("bench-start-btn").textContent = "Start Benchmark";
    }
  };

  poll();
}

function renderBenchmarkStatus(data) {
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-results");

  if (data.status === "running") {
    const pct = data.progress ? Math.round((data.progress.current_run / data.progress.total_runs) * 100) : 0;
    statusDiv.innerHTML = `
      <div class="bench-progress">
        <span>Run ${data.progress?.current_run || 0} of ${data.progress?.total_runs || "?"}</span>
        <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
      </div>`;
    return;
  }

  if (data.status === "error") {
    statusDiv.innerHTML = `<p class="error-message">Benchmark failed: ${escapeHtml(data.error || "Unknown error")}</p>`;
    resultsDiv.innerHTML = "";
    return;
  }

  // Complete
  statusDiv.innerHTML = '<p class="text-accent">Benchmark complete.</p>';

  let html = "";

  // Aggregate summary
  if (data.aggregate) {
    const a = data.aggregate;
    html += `
      <div class="detail-panel detail-panel--visible mb-4">
        <h3>Summary (${a.runs} run${a.runs !== 1 ? "s" : ""}, warmup skipped)</h3>
        <div class="detail-grid">
          <div class="detail-item"><span class="label">Avg tok/s</span><span class="value">${a.avg_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Min tok/s</span><span class="value">${a.min_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Max tok/s</span><span class="value">${a.max_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Avg Prefill tok/s</span><span class="value">${a.avg_prefill_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Avg TTFT</span><span class="value">${a.avg_ttft_ms} ms</span></div>
          <div class="detail-item"><span class="label">Avg Total</span><span class="value">${a.avg_total_ms} ms</span></div>
          <div class="detail-item"><span class="label">Tokens Generated</span><span class="value">${a.total_tokens_generated}</span></div>
        </div>
      </div>`;
  }

  // Per-run table
  if (data.runs && data.runs.length > 0) {
    const rows = data.runs
      .map(
        (r) => `<tr class="${r.success ? "" : "error-row"}">
          <td class="num">${r.run_number}</td>
          <td>${r.success ? "Pass" : "Fail"}</td>
          <td class="num">${r.tokens_per_second}</td>
          <td class="num">${r.prompt_tokens_per_second}</td>
          <td class="num">${r.time_to_first_token_ms}</td>
          <td class="num">${r.total_time_ms}</td>
          <td class="num">${r.prompt_tokens}</td>
          <td class="num">${r.generated_tokens}</td>
        </tr>`
      )
      .join("");

    html += `
      <table>
        <thead><tr>
          <th>Run</th><th>Status</th><th>tok/s</th><th>Prefill tok/s</th>
          <th>TTFT (ms)</th><th>Total (ms)</th><th>Prompt Tokens</th><th>Generated</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // Export button
  html += `<button class="btn btn-export mt-3" id="bench-export-btn">Export JSON</button>`;

  // Per-run chart canvas
  if (data.runs && data.runs.length > 1) {
    html += `<div class="chart-container mt-4"><canvas id="bench-runs-chart"></canvas></div>`;
  }

  resultsDiv.innerHTML = html;
  lastBenchResult = data;

  // Draw per-run line chart
  if (data.runs && data.runs.length > 1) {
    const chartData = data.runs
      .filter((r) => r.success)
      .map((r) => ({ label: `#${r.run_number}`, value: r.tokens_per_second }));
    const canvas = document.getElementById("bench-runs-chart");
    if (canvas && chartData.length > 1) {
      drawLineChart(canvas, chartData, { title: "tok/s per Run", unit: "", height: 180, showDots: true });
    }
  }

  // Wire export button
  document.getElementById("bench-export-btn").addEventListener("click", () => exportJson(lastBenchResult, "benchmark"));
}

function onPromptTypeChange() {
  const promptType = document.getElementById("bench-prompt-select").value;
  const customDiv = document.getElementById("bench-custom-prompt");
  customDiv.classList.toggle("hidden", promptType !== "custom");
}

function startStream() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelId = document.getElementById("bench-model-select").value;
  const promptType = document.getElementById("bench-prompt-select").value;

  if (!runtimeName || !modelId) return;

  const streamBtn = document.getElementById("bench-stream-btn");
  const startBtn = document.getElementById("bench-start-btn");
  const statusDiv = document.getElementById("bench-status");
  const streamOutput = document.getElementById("bench-stream-output");
  const streamText = document.getElementById("bench-stream-text");

  // Reset stream display
  streamText.textContent = "";
  document.getElementById("stream-token-count").textContent = "0";
  document.getElementById("stream-tok-sec").textContent = "—";
  document.getElementById("stream-ttft").textContent = "—";
  document.getElementById("stream-elapsed").textContent = "—";
  streamOutput.classList.remove("hidden");
  document.getElementById("bench-results").innerHTML = "";

  streamBtn.disabled = true;
  startBtn.disabled = true;
  streamBtn.textContent = "Streaming…";
  statusDiv.innerHTML = '<p class="loading">Streaming tokens…</p>';

  // Build SSE URL
  const params = new URLSearchParams({
    runtime_name: runtimeName,
    model_id: modelId,
    prompt_type: promptType,
  });
  if (promptType === "custom") {
    params.set("custom_prompt", document.getElementById("bench-custom-prompt-text").value);
  }

  // Close any previous connection
  if (streamSource) {
    streamSource.close();
    streamSource = null;
  }

  const es = new EventSource(`/api/benchmark/stream?${params}`);
  streamSource = es;

  es.addEventListener("token", (e) => {
    const data = JSON.parse(e.data);
    streamText.textContent += data.text;
    document.getElementById("stream-token-count").textContent = data.token_count;
    // Auto-scroll
    streamText.scrollTop = streamText.scrollHeight;
  });

  es.addEventListener("metrics", (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("stream-tok-sec").textContent = data.tok_per_sec;
    document.getElementById("stream-ttft").textContent = `${data.ttft_ms} ms`;
    document.getElementById("stream-elapsed").textContent = `${(data.elapsed_ms / 1000).toFixed(1)}s`;
  });

  es.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("stream-tok-sec").textContent = data.tok_per_sec;
    document.getElementById("stream-ttft").textContent = `${data.ttft_ms} ms`;
    document.getElementById("stream-elapsed").textContent = `${(data.elapsed_ms / 1000).toFixed(1)}s`;
    document.getElementById("stream-token-count").textContent = data.tokens;
    statusDiv.innerHTML = '<p class="text-accent">Stream complete.</p>';
    cleanupStream();
  });

  es.addEventListener("error", (e) => {
    let msg = "Stream connection lost";
    try {
      const data = JSON.parse(e.data);
      msg = data.message || msg;
    } catch {
      // not a JSON error event — connection error
    }
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(msg)}</p>`;
    cleanupStream();
  });
}

function cleanupStream() {
  if (streamSource) {
    streamSource.close();
    streamSource = null;
  }
  const streamBtn = document.getElementById("bench-stream-btn");
  const startBtn = document.getElementById("bench-start-btn");
  streamBtn.disabled = false;
  startBtn.disabled = false;
  streamBtn.textContent = "Stream";
}

// ── Sweep Mode ─────────────────────────────────────────────────────────

function onSweepToggle() {
  const checked = document.getElementById("bench-sweep-toggle").checked;
  document.getElementById("bench-sweep-controls").classList.toggle("hidden", !checked);
}

function populateSweepModels(models) {
  const container = document.getElementById("bench-sweep-model-list");
  if (!models || models.length === 0) {
    container.innerHTML = '<p class="placeholder">No models loaded in this runtime.</p>';
    document.getElementById("bench-sweep-start-btn").disabled = true;
    return;
  }

  container.innerHTML = "";
  models.forEach((m) => {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = m;
    cb.addEventListener("change", () => {
      label.classList.toggle("checked", cb.checked);
      updateSweepStartBtn();
    });
    label.appendChild(cb);
    label.appendChild(document.createTextNode(m));
    container.appendChild(label);
  });
  updateSweepStartBtn();
}

function getSelectedSweepModels() {
  const checkboxes = document.querySelectorAll("#bench-sweep-model-list input[type=checkbox]:checked");
  return Array.from(checkboxes).map((cb) => cb.value);
}

function updateSweepStartBtn() {
  const selected = getSelectedSweepModels();
  document.getElementById("bench-sweep-start-btn").disabled = selected.length < 2;
}

async function startSweep() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelIds = getSelectedSweepModels();
  const promptType = document.getElementById("bench-prompt-select").value;
  const numRuns = parseInt(document.getElementById("bench-runs-input").value, 10) || 3;

  if (!runtimeName || modelIds.length < 2) return;

  const sweepBtn = document.getElementById("bench-sweep-start-btn");
  const statusDiv = document.getElementById("bench-status");
  const sweepResults = document.getElementById("bench-sweep-results");

  sweepBtn.disabled = true;
  sweepBtn.textContent = "Sweeping…";
  sweepResults.innerHTML = "";
  document.getElementById("bench-results").innerHTML = "";
  statusDiv.innerHTML = '<p class="loading">Starting sweep…</p>';

  const body = {
    runtime_name: runtimeName,
    model_ids: modelIds,
    prompt_type: promptType,
    num_runs: numRuns,
  };
  if (promptType === "custom") {
    body.custom_prompt = document.getElementById("bench-custom-prompt-text").value;
  }

  try {
    const data = await api.post("/benchmark/sweep", body);
    sweepJobId = data.job_id;
    pollSweepStatus();
  } catch (err) {
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
    sweepBtn.disabled = false;
    sweepBtn.textContent = "Start Sweep";
  }
}

function pollSweepStatus() {
  if (!sweepJobId) return;
  clearTimeout(sweepPollTimer);

  const poll = async () => {
    try {
      const data = await api.get(`/benchmark/sweep/${sweepJobId}`);
      renderSweepStatus(data);

      if (data.status === "running") {
        sweepPollTimer = setTimeout(poll, 1000);
      } else {
        const btn = document.getElementById("bench-sweep-start-btn");
        btn.disabled = false;
        btn.textContent = "Start Sweep";
        updateSweepStartBtn();
      }
    } catch (err) {
      document.getElementById("bench-status").innerHTML =
        `<p class="error-message">${escapeHtml(err.message)}</p>`;
      const btn = document.getElementById("bench-sweep-start-btn");
      btn.disabled = false;
      btn.textContent = "Start Sweep";
      updateSweepStartBtn();
    }
  };

  poll();
}

function renderSweepStatus(data) {
  const statusDiv = document.getElementById("bench-status");
  const sweepResults = document.getElementById("bench-sweep-results");

  if (data.status === "running") {
    const p = data.progress;
    const comboPct = p ? Math.round((p.current_combo / p.total_combos) * 100) : 0;
    statusDiv.innerHTML = `
      <div class="bench-progress">
        <span>Model ${p?.current_combo || 0} of ${p?.total_combos || "?"} — run ${p?.current_run_in_combo || 0}/${p?.total_runs_per_combo || "?"}</span>
        <div class="progress-bar"><div class="progress-fill" style="width:${comboPct}%"></div></div>
      </div>`;
    return;
  }

  if (data.status === "error") {
    statusDiv.innerHTML = `<p class="error-message">Sweep failed: ${escapeHtml(data.error || "Unknown error")}</p>`;
    sweepResults.innerHTML = "";
    return;
  }

  // Complete — render comparison table
  statusDiv.innerHTML = '<p class="text-accent">Sweep complete.</p>';

  if (!data.combo_results || data.combo_results.length === 0) {
    sweepResults.innerHTML = '<p class="placeholder">No results.</p>';
    return;
  }

  // Find best values for highlighting
  const aggregates = data.combo_results
    .filter((cr) => cr.aggregate && cr.aggregate.runs > 0)
    .map((cr) => ({ model_id: cr.model_id, ...cr.aggregate }));

  let bestTokSec = 0;
  let bestPrefill = 0;
  let bestTtft = Infinity;

  for (const a of aggregates) {
    if (a.avg_tok_per_sec > bestTokSec) bestTokSec = a.avg_tok_per_sec;
    if (a.avg_prefill_tok_per_sec > bestPrefill) bestPrefill = a.avg_prefill_tok_per_sec;
    if (a.avg_ttft_ms < bestTtft) bestTtft = a.avg_ttft_ms;
  }

  const rows = data.combo_results
    .map((cr) => {
      const a = cr.aggregate;
      if (!a || a.runs === 0) {
        return `<tr><td>${escapeHtml(cr.model_id)}</td><td colspan="6" class="text-danger">Failed</td></tr>`;
      }
      const cls = (val, best, lower) => {
        const isBest = lower ? val <= best : val >= best;
        return isBest ? "num best-value" : "num";
      };
      return `<tr>
        <td>${escapeHtml(cr.model_id)}</td>
        <td class="${cls(a.avg_tok_per_sec, bestTokSec)}">${a.avg_tok_per_sec}</td>
        <td class="${cls(a.min_tok_per_sec, bestTokSec)}">${a.min_tok_per_sec}</td>
        <td class="${cls(a.max_tok_per_sec, bestTokSec)}">${a.max_tok_per_sec}</td>
        <td class="${cls(a.avg_prefill_tok_per_sec, bestPrefill)}">${a.avg_prefill_tok_per_sec}</td>
        <td class="${cls(a.avg_ttft_ms, bestTtft, true)}">${a.avg_ttft_ms} ms</td>
        <td class="num">${a.runs}</td>
      </tr>`;
    })
    .join("");

  sweepResults.innerHTML = `
    <div class="sweep-comparison">
      <h3>Sweep Comparison</h3>
      <table>
        <thead><tr>
          <th>Model</th><th>Avg tok/s</th><th>Min tok/s</th><th>Max tok/s</th>
          <th>Prefill tok/s</th><th>Avg TTFT</th><th>Runs</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <button class="btn btn-export mt-3" id="sweep-export-btn">Export JSON</button>
      <div class="chart-container mt-4"><canvas id="bench-sweep-chart"></canvas></div>
    </div>`;

  lastSweepResult = data;
  lastSweepAggregates = aggregates;

  // Draw sweep bar chart comparing tok/s across models
  if (aggregates.length > 0) {
    const chartData = aggregates.map((a) => ({
      label: a.model_id.split("/").pop(),
      value: a.avg_tok_per_sec,
    }));
    const canvas = document.getElementById("bench-sweep-chart");
    if (canvas) {
      drawBarChart(canvas, chartData, { title: "Avg tok/s by Model", unit: "", height: 250 });
    }
  }

  // Wire export button
  document.getElementById("sweep-export-btn").addEventListener("click", () => exportJson(lastSweepResult, "sweep"));
}

function exportJson(data, prefix) {
  if (!data) return;
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${prefix}-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function reRenderBenchCharts() {
  // Re-render bench line chart
  if (lastBenchResult && lastBenchResult.runs && lastBenchResult.runs.length > 1) {
    const chartData = lastBenchResult.runs
      .filter((r) => r.success)
      .map((r) => ({ label: `#${r.run_number}`, value: r.tokens_per_second }));
    const canvas = document.getElementById("bench-runs-chart");
    if (canvas && chartData.length > 1) {
      drawLineChart(canvas, chartData, { title: "tok/s per Run", unit: "", height: 180, showDots: true });
    }
  }
  // Re-render sweep bar chart
  if (lastSweepAggregates && lastSweepAggregates.length > 0) {
    const chartData = lastSweepAggregates.map((a) => ({
      label: a.model_id.split("/").pop(),
      value: a.avg_tok_per_sec,
    }));
    const canvas = document.getElementById("bench-sweep-chart");
    if (canvas) {
      drawBarChart(canvas, chartData, { title: "Avg tok/s by Model", unit: "", height: 250 });
    }
  }
}

export function initBenchmark() {
  document.getElementById("detect-runtimes-btn").addEventListener("click", detectRuntimes);
  document.getElementById("bench-runtime-select").addEventListener("change", onRuntimeChange);
  document.getElementById("bench-start-btn").addEventListener("click", startBenchmark);
  document.getElementById("bench-stream-btn").addEventListener("click", startStream);
  document.getElementById("bench-prompt-select").addEventListener("change", onPromptTypeChange);
  document.getElementById("bench-sweep-toggle").addEventListener("change", onSweepToggle);
  document.getElementById("bench-sweep-start-btn").addEventListener("click", startSweep);

  document.addEventListener("themechange", reRenderBenchCharts);
}
