import { api } from "./api.js";
import { escapeHtml } from "./utils.js";

let macData = null;

export async function loadMyMac() {
  const container = document.getElementById("mymac-content");
  const memSection = document.getElementById("mymac-memory");

  if (!container || !memSection) {
    console.error("My Mac: missing DOM elements", { container, memSection });
    return;
  }

  container.innerHTML = '<p class="loading">Detecting hardware…</p>';
  memSection.innerHTML = "";

  try {
    const detect = await api.get("/hardware/detect");
    if (!detect.detected || !detect.name) {
      container.innerHTML = '<p class="error-message">Could not detect Apple Silicon hardware. This feature requires a Mac with Apple Silicon.</p>';
      return;
    }

    const [spec, mem] = await Promise.all([
      api.get(`/hardware/${encodeURIComponent(detect.name)}`),
      api.get("/memory/current"),
    ]);

    macData = { detect, spec, mem };
    renderHardware(spec);
    renderMemory(mem);
    startMemoryPoll();
  } catch (err) {
    container.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

function renderHardware(spec) {
  const container = document.getElementById("mymac-content");
  container.innerHTML = `
    <div class="mymac-header">
      <h3>${escapeHtml(spec.chip)}</h3>
      <span class="badge comfortable">${escapeHtml(spec.name)}</span>
    </div>
    <div class="detail-grid">
      <div class="detail-item"><span class="label">Chip</span><span class="value">${escapeHtml(spec.chip)}</span></div>
      <div class="detail-item"><span class="label">CPU Cores</span><span class="value">${spec.cpu_cores}</span></div>
      <div class="detail-item"><span class="label">GPU Cores</span><span class="value">${spec.gpu_cores}</span></div>
      <div class="detail-item"><span class="label">Neural Engine</span><span class="value">${spec.neural_engine_cores} cores</span></div>
      <div class="detail-item"><span class="label">Total Memory</span><span class="value">${spec.memory_gb} GB</span></div>
      <div class="detail-item"><span class="label">Usable for LLMs</span><span class="value">${spec.usable_memory_gb.toFixed(1)} GB</span></div>
      <div class="detail-item"><span class="label">Memory Bandwidth</span><span class="value">${spec.memory_bandwidth_gbs} GB/s</span></div>
      <div class="detail-item"><span class="label">GPU TFLOPS</span><span class="value">${spec.gpu_tflops}</span></div>
    </div>`;
}

function renderMemory(data) {
  const section = document.getElementById("mymac-memory");
  const pct = Math.min(100, Math.max(0, data.usage_pct));

  const pressureClass = data.pressure === "critical" ? "pressure-critical"
    : data.pressure === "warn" ? "pressure-warn"
    : "pressure-normal";

  section.innerHTML = `
    <h3>Live Memory</h3>
    <div class="memory-gauge">
      <div class="memory-bar">
        <div class="memory-fill ${pressureClass}" style="width:${pct}%"></div>
      </div>
      <div class="memory-gauge-label">${data.used_gb.toFixed(1)} / ${data.total_gb.toFixed(1)} GB (${pct.toFixed(0)}%)</div>
    </div>
    <div class="detail-grid">
      <div class="detail-item"><span class="label">Used</span><span class="value">${data.used_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">Free</span><span class="value">${data.free_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">Pressure</span><span class="value"><span class="pressure-badge ${pressureClass}">${escapeHtml(data.pressure)}</span></span></div>
    </div>`;
}

let mymacPollTimer = null;

function startMemoryPoll() {
  stopMemoryPoll();
  mymacPollTimer = setInterval(async () => {
    const tab = document.getElementById("tab-mymac");
    if (!tab || !tab.classList.contains("active")) {
      stopMemoryPoll();
      return;
    }
    try {
      const mem = await api.get("/memory/current");
      renderMemory(mem);
    } catch {
      // silent — keep showing last data
    }
  }, 3000);
}

export function stopMemoryPoll() {
  clearInterval(mymacPollTimer);
  mymacPollTimer = null;
}

export function initMyMac() {
  // Loads on-demand when tab is switched to
}
