import { api } from "./api.js";
import { escapeHtml, copyToClipboard } from "./utils.js";
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
let throughputJobId = null;
let throughputPollTimer = null;
let compareJobId = null;
let comparePollTimer = null;

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

  // Enable throughput button
  updateThroughputBtn();

  // Populate compare runtime dropdowns
  populateCompareRuntimes();
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
          <div class="detail-item"><span class="label">Median tok/s</span><span class="value">${a.median_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">P95 tok/s</span><span class="value">${a.p95_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Min tok/s</span><span class="value">${a.min_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Max tok/s</span><span class="value">${a.max_tok_per_sec}</span></div>
          <div class="detail-item"><span class="label">Std Dev</span><span class="value">${a.stddev_tok_per_sec}</span></div>
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

  // Export and copy buttons
  html += `<div class="mt-3 results-actions">
    <button class="btn btn-copy" id="bench-copy-btn" title="Copy summary to clipboard">Copy Results</button>
    <button class="btn btn-export" id="bench-export-btn">Export JSON</button>
    <button class="btn btn-export ml-2" id="bench-export-html-btn">Export HTML Report</button>
  </div>`;

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

  // Wire export and copy buttons
  document.getElementById("bench-export-btn").addEventListener("click", () => exportJson(lastBenchResult, "benchmark"));
  document.getElementById("bench-export-html-btn").addEventListener("click", exportHtmlReport);
  document.getElementById("bench-copy-btn").addEventListener("click", (e) => {
    const text = formatBenchmarkText(data);
    copyToClipboard(text, e.currentTarget);
  });
}

async function exportHtmlReport() {
  if (!benchJobId) return;
  try {
    const resp = await fetch(`/api/benchmark/${benchJobId}/report`);
    if (!resp.ok) throw new Error("Report generation failed");
    const html = await resp.text();
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `benchmark-report-${Date.now()}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Failed to export HTML report: " + err.message);
  }
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
      <div class="mt-3 results-actions">
        <button class="btn btn-copy" id="sweep-copy-btn" title="Copy comparison to clipboard">Copy Results</button>
        <button class="btn btn-export" id="sweep-export-btn">Export JSON</button>
      </div>
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

  // Wire export and copy buttons
  document.getElementById("sweep-export-btn").addEventListener("click", () => exportJson(lastSweepResult, "sweep"));
  document.getElementById("sweep-copy-btn").addEventListener("click", (e) => {
    const text = formatSweepText(data);
    copyToClipboard(text, e.currentTarget);
  });
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

// ── Throughput Test ─────────────────────────────────────────────────────

function onThroughputToggle() {
  const checked = document.getElementById("bench-throughput-toggle").checked;
  document.getElementById("bench-throughput-controls").classList.toggle("hidden", !checked);
}

function updateThroughputBtn() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelId = document.getElementById("bench-model-select").value;
  const btn = document.getElementById("bench-throughput-start-btn");
  btn.disabled = !runtimeName || !modelId;
}

async function startThroughput() {
  const runtimeName = document.getElementById("bench-runtime-select").value;
  const modelId = document.getElementById("bench-model-select").value;
  const concurrency = parseInt(document.getElementById("bench-concurrency-select").value, 10);
  const totalRequests = parseInt(document.getElementById("bench-total-requests").value, 10) || 8;

  if (!runtimeName || !modelId) return;

  const btn = document.getElementById("bench-throughput-start-btn");
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-throughput-results");

  btn.disabled = true;
  btn.textContent = "Running…";
  resultsDiv.innerHTML = "";
  statusDiv.innerHTML = '<p class="loading">Starting throughput test…</p>';

  try {
    const data = await api.post("/benchmark/throughput", {
      runtime_name: runtimeName,
      model_id: modelId,
      concurrency,
      total_requests: totalRequests,
    });
    throughputJobId = data.job_id;
    pollThroughputStatus();
  } catch (err) {
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
    btn.disabled = false;
    btn.textContent = "Start Throughput Test";
  }
}

function pollThroughputStatus() {
  if (!throughputJobId) return;
  clearTimeout(throughputPollTimer);

  const poll = async () => {
    try {
      const data = await api.get(`/benchmark/throughput/${throughputJobId}`);
      renderThroughputStatus(data);
      if (data.status === "running") {
        throughputPollTimer = setTimeout(poll, 1000);
      } else {
        const btn = document.getElementById("bench-throughput-start-btn");
        btn.disabled = false;
        btn.textContent = "Start Throughput Test";
        updateThroughputBtn();
      }
    } catch (err) {
      document.getElementById("bench-status").innerHTML =
        `<p class="error-message">${escapeHtml(err.message)}</p>`;
      const btn = document.getElementById("bench-throughput-start-btn");
      btn.disabled = false;
      btn.textContent = "Start Throughput Test";
    }
  };
  poll();
}

function renderThroughputStatus(data) {
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-throughput-results");

  if (data.status === "running") {
    statusDiv.innerHTML = '<p class="loading">Throughput test in progress…</p>';
    return;
  }

  if (data.status === "error") {
    statusDiv.innerHTML = `<p class="error-message">Throughput test failed: ${escapeHtml(data.error || "Unknown error")}</p>`;
    return;
  }

  statusDiv.innerHTML = '<p class="text-accent">Throughput test complete.</p>';

  resultsDiv.innerHTML = `
    <div class="detail-panel detail-panel--visible mb-4">
      <div class="detail-header-row">
        <h3>Throughput Results (${data.concurrency} concurrent, ${data.total_requests} requests)</h3>
        <button class="btn btn-copy btn-copy-sm" id="throughput-copy-btn" title="Copy throughput results">Copy</button>
      </div>
      <div class="detail-grid">
        <div class="detail-item"><span class="label">Aggregate Throughput</span><span class="value font-bold">${data.throughput_tps} tok/s</span></div>
        <div class="detail-item"><span class="label">Total Tokens</span><span class="value">${data.total_tokens}</span></div>
        <div class="detail-item"><span class="label">Elapsed</span><span class="value">${data.elapsed_seconds}s</span></div>
        <div class="detail-item"><span class="label">Success / Failed</span><span class="value">${data.successful_requests} / ${data.failed_requests}</span></div>
        <div class="detail-item"><span class="label">Avg Latency</span><span class="value">${data.avg_latency_ms} ms</span></div>
        <div class="detail-item"><span class="label">Min Latency</span><span class="value">${data.min_latency_ms} ms</span></div>
        <div class="detail-item"><span class="label">Max Latency</span><span class="value">${data.max_latency_ms} ms</span></div>
        <div class="detail-item"><span class="label">Error Rate</span><span class="value">${(data.error_rate * 100).toFixed(1)}%</span></div>
      </div>
    </div>`;

  document.getElementById("throughput-copy-btn").addEventListener("click", (e) => {
    const text = formatThroughputText(data);
    copyToClipboard(text, e.currentTarget);
  });

  // Per-request table
  if (data.per_request && data.per_request.length > 0) {
    const rows = data.per_request.map(r => `
      <tr class="${r.success ? '' : 'error-row'}">
        <td class="num">${r.request_id}</td>
        <td>${r.success ? 'Pass' : 'Fail'}</td>
        <td class="num">${r.tokens_generated}</td>
        <td class="num">${r.elapsed_ms}</td>
        <td class="num">${r.tokens_per_second}</td>
      </tr>`).join("");
    resultsDiv.innerHTML += `
      <table>
        <thead><tr><th>Request</th><th>Status</th><th>Tokens</th><th>Latency (ms)</th><th>tok/s</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }
}

// ── Compare Mode ───────────────────────────────────────────────────────

function onCompareToggle() {
  const checked = document.getElementById("bench-compare-toggle").checked;
  document.getElementById("bench-compare-controls").classList.toggle("hidden", !checked);
}

function populateCompareRuntimes() {
  const selA = document.getElementById("bench-compare-runtime-a");
  const selB = document.getElementById("bench-compare-runtime-b");

  if (benchRuntimes.length < 2) {
    selA.innerHTML = '<option value="">Need 2+ runtimes…</option>';
    selB.innerHTML = '<option value="">Need 2+ runtimes…</option>';
    selA.disabled = true;
    selB.disabled = true;
    document.getElementById("bench-compare-start-btn").disabled = true;
    return;
  }

  [selA, selB].forEach((sel, idx) => {
    sel.innerHTML = '<option value="">Select…</option>';
    benchRuntimes.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = r.name;
      sel.appendChild(opt);
    });
    sel.disabled = false;
    // Pre-select different runtimes
    if (benchRuntimes.length >= 2) {
      sel.value = benchRuntimes[idx % benchRuntimes.length].name;
    }
  });
  updateCompareBtn();
}

function updateCompareBtn() {
  const a = document.getElementById("bench-compare-runtime-a").value;
  const b = document.getElementById("bench-compare-runtime-b").value;
  const modelId = document.getElementById("bench-model-select").value;
  document.getElementById("bench-compare-start-btn").disabled = !a || !b || a === b || !modelId;
}

async function startCompare() {
  const runtimeA = document.getElementById("bench-compare-runtime-a").value;
  const runtimeB = document.getElementById("bench-compare-runtime-b").value;
  const modelId = document.getElementById("bench-model-select").value;
  const promptType = document.getElementById("bench-prompt-select").value;
  const numRuns = parseInt(document.getElementById("bench-runs-input").value, 10) || 3;

  if (!runtimeA || !runtimeB || runtimeA === runtimeB || !modelId) return;

  const btn = document.getElementById("bench-compare-start-btn");
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-compare-results");

  btn.disabled = true;
  btn.textContent = "Comparing…";
  resultsDiv.innerHTML = "";
  statusDiv.innerHTML = '<p class="loading">Running comparison benchmarks…</p>';

  const body = {
    runtime_a: runtimeA,
    runtime_b: runtimeB,
    model_id: modelId,
    prompt_type: promptType,
    num_runs: numRuns,
  };
  if (promptType === "custom") {
    body.custom_prompt = document.getElementById("bench-custom-prompt-text").value;
  }

  try {
    const data = await api.post("/benchmark/compare", body);
    compareJobId = data.job_id;
    pollCompareStatus();
  } catch (err) {
    statusDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
    btn.disabled = false;
    btn.textContent = "Compare Runtimes";
  }
}

function pollCompareStatus() {
  if (!compareJobId) return;
  clearTimeout(comparePollTimer);

  const poll = async () => {
    try {
      const data = await api.get(`/benchmark/compare/${compareJobId}`);
      renderCompareStatus(data);
      if (data.status === "running") {
        comparePollTimer = setTimeout(poll, 1500);
      } else {
        const btn = document.getElementById("bench-compare-start-btn");
        btn.disabled = false;
        btn.textContent = "Compare Runtimes";
        updateCompareBtn();
      }
    } catch (err) {
      document.getElementById("bench-status").innerHTML =
        `<p class="error-message">${escapeHtml(err.message)}</p>`;
      const btn = document.getElementById("bench-compare-start-btn");
      btn.disabled = false;
      btn.textContent = "Compare Runtimes";
    }
  };
  poll();
}

function renderCompareStatus(data) {
  const statusDiv = document.getElementById("bench-status");
  const resultsDiv = document.getElementById("bench-compare-results");

  if (data.status === "running") {
    statusDiv.innerHTML = '<p class="loading">Running comparison…</p>';
    return;
  }

  if (data.status === "error") {
    statusDiv.innerHTML = `<p class="error-message">Compare failed: ${escapeHtml(data.error || "Unknown")}</p>`;
    return;
  }

  statusDiv.innerHTML = '<p class="text-accent">Comparison complete.</p>';

  if (!data.results || data.results.length < 2) {
    resultsDiv.innerHTML = '<p class="placeholder">No comparison results.</p>';
    return;
  }

  const speedupHtml = data.speedup_pct != null
    ? `<div class="detail-panel detail-panel--visible mb-4">
        <h3>${escapeHtml(data.faster_runtime)} is ${data.speedup_pct}% faster</h3>
      </div>`
    : "";

  const rows = data.results.map(cr => {
    const a = cr.aggregate;
    if (!a || a.runs === 0) {
      return `<tr><td>${escapeHtml(cr.runtime_name)}</td><td colspan="8" class="text-danger">Failed</td></tr>`;
    }
    return `<tr>
      <td>${escapeHtml(cr.runtime_name)}</td>
      <td class="num">${a.avg_tok_per_sec}</td>
      <td class="num">${a.median_tok_per_sec}</td>
      <td class="num">${a.p95_tok_per_sec}</td>
      <td class="num">${a.min_tok_per_sec}</td>
      <td class="num">${a.max_tok_per_sec}</td>
      <td class="num">${a.avg_prefill_tok_per_sec}</td>
      <td class="num">${a.avg_ttft_ms} ms</td>
      <td class="num">${a.runs}</td>
    </tr>`;
  }).join("");

  resultsDiv.innerHTML = `
    ${speedupHtml}
    <table>
      <thead><tr>
        <th>Runtime</th><th>Avg tok/s</th><th>Median</th><th>P95</th>
        <th>Min</th><th>Max</th><th>Prefill tok/s</th><th>Avg TTFT</th><th>Runs</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="mt-3 results-actions">
      <button class="btn btn-copy" id="compare-copy-btn" title="Copy comparison to clipboard">Copy Results</button>
    </div>
    <div class="chart-container mt-4"><canvas id="bench-compare-chart"></canvas></div>`;

  document.getElementById("compare-copy-btn").addEventListener("click", (e) => {
    const text = formatCompareText(data);
    copyToClipboard(text, e.currentTarget);
  });

  // Draw comparison bar chart
  const chartData = data.results
    .filter(cr => cr.aggregate && cr.aggregate.runs > 0)
    .map(cr => ({ label: cr.runtime_name, value: cr.aggregate.avg_tok_per_sec }));
  const canvas = document.getElementById("bench-compare-chart");
  if (canvas && chartData.length > 0) {
    drawBarChart(canvas, chartData, { title: "Avg tok/s by Runtime", unit: "", height: 200 });
  }
}

// ── Copy-to-clipboard text formatters ─────────────────────────────────

function formatBenchmarkText(data) {
  const lines = ["Loca-LLAMA Benchmark Results", "=".repeat(40)];

  if (data.aggregate) {
    const a = data.aggregate;
    lines.push(
      `Runs: ${a.runs} (warmup skipped)`,
      "",
      "Key Metrics:",
      `  Avg tok/s:         ${a.avg_tok_per_sec}`,
      `  Median tok/s:      ${a.median_tok_per_sec}`,
      `  P95 tok/s:         ${a.p95_tok_per_sec}`,
      `  Min / Max tok/s:   ${a.min_tok_per_sec} / ${a.max_tok_per_sec}`,
      `  Std Dev:           ${a.stddev_tok_per_sec}`,
      `  Avg Prefill tok/s: ${a.avg_prefill_tok_per_sec}`,
      `  Avg TTFT:          ${a.avg_ttft_ms} ms`,
      `  Avg Total:         ${a.avg_total_ms} ms`,
      `  Tokens Generated:  ${a.total_tokens_generated}`,
    );
  }

  if (data.runs && data.runs.length > 0) {
    lines.push("", "Per-Run Results:", "Run  Status  tok/s    Prefill  TTFT(ms)  Total(ms)");
    lines.push("-".repeat(55));
    data.runs.forEach((r) => {
      lines.push(
        `${String(r.run_number).padStart(3)}  ${(r.success ? "Pass" : "Fail").padEnd(6)}  ` +
        `${String(r.tokens_per_second).padStart(7)}  ${String(r.prompt_tokens_per_second).padStart(7)}  ` +
        `${String(r.time_to_first_token_ms).padStart(8)}  ${String(r.total_time_ms).padStart(9)}`
      );
    });
  }

  return lines.join("\n");
}

function formatSweepText(data) {
  const lines = ["Loca-LLAMA Sweep Comparison", "=".repeat(50)];

  if (!data.combo_results || data.combo_results.length === 0) {
    lines.push("No results.");
    return lines.join("\n");
  }

  const nameW = Math.max(5, ...data.combo_results.map((cr) => cr.model_id.length));
  const hdr = `${"Model".padEnd(nameW)}  Avg tok/s  Min  Max  Prefill  TTFT     Runs`;
  lines.push(hdr, "-".repeat(hdr.length));

  data.combo_results.forEach((cr) => {
    const a = cr.aggregate;
    if (!a || a.runs === 0) {
      lines.push(`${cr.model_id.padEnd(nameW)}  FAILED`);
      return;
    }
    lines.push(
      `${cr.model_id.padEnd(nameW)}  ` +
      `${String(a.avg_tok_per_sec).padStart(9)}  ` +
      `${String(a.min_tok_per_sec).padStart(3)}  ` +
      `${String(a.max_tok_per_sec).padStart(3)}  ` +
      `${String(a.avg_prefill_tok_per_sec).padStart(7)}  ` +
      `${(a.avg_ttft_ms + " ms").padStart(7)}  ` +
      `${String(a.runs).padStart(4)}`
    );
  });

  return lines.join("\n");
}

function formatThroughputText(data) {
  return [
    "Loca-LLAMA Throughput Test",
    "=".repeat(40),
    `Concurrency:   ${data.concurrency}`,
    `Total Requests: ${data.total_requests}`,
    "",
    "Results:",
    `  Throughput:    ${data.throughput_tps} tok/s`,
    `  Total Tokens:  ${data.total_tokens}`,
    `  Elapsed:       ${data.elapsed_seconds}s`,
    `  Success/Fail:  ${data.successful_requests} / ${data.failed_requests}`,
    `  Avg Latency:   ${data.avg_latency_ms} ms`,
    `  Min Latency:   ${data.min_latency_ms} ms`,
    `  Max Latency:   ${data.max_latency_ms} ms`,
    `  Error Rate:    ${(data.error_rate * 100).toFixed(1)}%`,
  ].join("\n");
}

function formatCompareText(data) {
  const lines = ["Loca-LLAMA Runtime Comparison", "=".repeat(50)];

  if (data.speedup_pct != null) {
    lines.push(`Winner: ${data.faster_runtime} (+${data.speedup_pct}% faster)`, "");
  }

  if (!data.results || data.results.length === 0) {
    lines.push("No results.");
    return lines.join("\n");
  }

  const nameW = Math.max(7, ...data.results.map((cr) => cr.runtime_name.length));
  const hdr = `${"Runtime".padEnd(nameW)}  Avg tok/s  Median  P95   Min   Max   Prefill  TTFT     Runs`;
  lines.push(hdr, "-".repeat(hdr.length));

  data.results.forEach((cr) => {
    const a = cr.aggregate;
    if (!a || a.runs === 0) {
      lines.push(`${cr.runtime_name.padEnd(nameW)}  FAILED`);
      return;
    }
    lines.push(
      `${cr.runtime_name.padEnd(nameW)}  ` +
      `${String(a.avg_tok_per_sec).padStart(9)}  ` +
      `${String(a.median_tok_per_sec).padStart(6)}  ` +
      `${String(a.p95_tok_per_sec).padStart(4)}  ` +
      `${String(a.min_tok_per_sec).padStart(4)}  ` +
      `${String(a.max_tok_per_sec).padStart(4)}  ` +
      `${String(a.avg_prefill_tok_per_sec).padStart(7)}  ` +
      `${(a.avg_ttft_ms + " ms").padStart(7)}  ` +
      `${String(a.runs).padStart(4)}`
    );
  });

  return lines.join("\n");
}

export function initBenchmark() {
  document.getElementById("detect-runtimes-btn").addEventListener("click", detectRuntimes);
  document.getElementById("bench-runtime-select").addEventListener("change", onRuntimeChange);
  document.getElementById("bench-start-btn").addEventListener("click", startBenchmark);
  document.getElementById("bench-stream-btn").addEventListener("click", startStream);
  document.getElementById("bench-prompt-select").addEventListener("change", onPromptTypeChange);
  document.getElementById("bench-sweep-toggle").addEventListener("change", onSweepToggle);
  document.getElementById("bench-sweep-start-btn").addEventListener("click", startSweep);

  // Throughput controls
  document.getElementById("bench-throughput-toggle").addEventListener("change", onThroughputToggle);
  document.getElementById("bench-throughput-start-btn").addEventListener("click", startThroughput);

  // Compare controls
  document.getElementById("bench-compare-toggle").addEventListener("change", onCompareToggle);
  document.getElementById("bench-compare-start-btn").addEventListener("click", startCompare);
  document.getElementById("bench-compare-runtime-a").addEventListener("change", updateCompareBtn);
  document.getElementById("bench-compare-runtime-b").addEventListener("change", updateCompareBtn);

  document.addEventListener("themechange", reRenderBenchCharts);
}
