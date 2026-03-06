import { api } from "./api.js";
import { escapeHtml } from "./utils.js";

let benchRuntimes = [];
let benchJobId = null;
let benchPollTimer = null;
let streamSource = null;

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
    statusDiv.innerHTML = `<p style="color:var(--text-muted)">${data.count} runtime${data.count !== 1 ? "s" : ""} detected.</p>`;
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
  statusDiv.innerHTML = '<p style="color:var(--accent)">Benchmark complete.</p>';

  let html = "";

  // Aggregate summary
  if (data.aggregate) {
    const a = data.aggregate;
    html += `
      <div class="detail-panel" style="display:block;margin-bottom:1rem;">
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
          <td>${r.run_number}</td>
          <td>${r.success ? "Pass" : "Fail"}</td>
          <td>${r.tokens_per_second}</td>
          <td>${r.prompt_tokens_per_second}</td>
          <td>${r.time_to_first_token_ms}</td>
          <td>${r.total_time_ms}</td>
          <td>${r.prompt_tokens}</td>
          <td>${r.generated_tokens}</td>
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

  resultsDiv.innerHTML = html;
}

function onPromptTypeChange() {
  const promptType = document.getElementById("bench-prompt-select").value;
  const customDiv = document.getElementById("bench-custom-prompt");
  customDiv.style.display = promptType === "custom" ? "block" : "none";
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
  streamOutput.style.display = "block";
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
    statusDiv.innerHTML = '<p style="color:var(--accent)">Stream complete.</p>';
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

export function initBenchmark() {
  document.getElementById("detect-runtimes-btn").addEventListener("click", detectRuntimes);
  document.getElementById("bench-runtime-select").addEventListener("change", onRuntimeChange);
  document.getElementById("bench-start-btn").addEventListener("click", startBenchmark);
  document.getElementById("bench-stream-btn").addEventListener("click", startStream);
  document.getElementById("bench-prompt-select").addEventListener("change", onPromptTypeChange);
}
