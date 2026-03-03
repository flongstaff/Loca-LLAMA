import { api } from "./api.js";
import { escapeHtml } from "./utils.js";
import { setupCanvas, getThemeColors, drawGuideLines, drawAreaLine, drawTimeLabels } from "./chart.js";

const MAX_MEMORY_SLOTS = 60;
let memoryPollTimer = null;
let memoryHistory = [];

export function stopMemoryPolling() {
  clearTimeout(memoryPollTimer);
  memoryPollTimer = null;
}

export async function loadMemory() {
  stopMemoryPolling();
  memoryHistory = [];

  const poll = async () => {
    try {
      const data = await api.get("/memory/current");
      memoryHistory.push(data);
      if (memoryHistory.length > MAX_MEMORY_SLOTS) memoryHistory.shift();
      renderMemoryGauge(data);
      renderMemoryChart();
    } catch (err) {
      console.error("Memory poll failed:", err);
      if (memoryHistory.length === 0) {
        document.getElementById("memory-gauge-label").textContent = "Failed to load memory data";
        return;
      }
    }

    // Only schedule next poll if memory tab is still active
    const memTab = document.getElementById("tab-memory");
    if (memTab && memTab.classList.contains("active")) {
      memoryPollTimer = setTimeout(poll, 2000);
    }
  };

  poll();
  loadMemoryReport();
}

function renderMemoryGauge(data) {
  const fill = document.getElementById("memory-fill");
  const label = document.getElementById("memory-gauge-label");
  const pct = Math.min(100, Math.max(0, data.usage_pct));

  fill.style.width = `${pct}%`;

  // Color transitions: green -> yellow -> red
  if (data.pressure === "critical") {
    fill.className = "memory-fill pressure-critical";
  } else if (data.pressure === "warn") {
    fill.className = "memory-fill pressure-warn";
  } else {
    fill.className = "memory-fill pressure-normal";
  }

  label.textContent = `${data.used_gb.toFixed(1)} / ${data.total_gb.toFixed(1)} GB (${pct.toFixed(0)}%)`;

  // Update stat values
  document.getElementById("mem-used").textContent = `${data.used_gb.toFixed(2)} GB`;
  document.getElementById("mem-free").textContent = `${data.free_gb.toFixed(2)} GB`;
  document.getElementById("mem-total").textContent = `${data.total_gb.toFixed(1)} GB`;
  document.getElementById("mem-pct").textContent = `${pct.toFixed(1)}%`;

  const pressureEl = document.getElementById("mem-pressure");
  const badgeClass = data.pressure === "critical" ? "pressure-critical"
    : data.pressure === "warn" ? "pressure-warn"
    : "pressure-normal";
  pressureEl.innerHTML = `<span class="pressure-badge ${badgeClass}">${escapeHtml(data.pressure)}</span>`;
}

function renderMemoryChart() {
  const canvas = document.getElementById("memory-chart");
  if (!canvas || memoryHistory.length === 0) return;

  const { ctx, displayWidth, displayHeight } = setupCanvas(canvas, 200);

  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const w = displayWidth - pad.left - pad.right;
  const h = displayHeight - pad.top - pad.bottom;

  // Clear
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  const totalGb = memoryHistory[memoryHistory.length - 1].total_gb;
  const maxY = totalGb;

  // Guide lines
  drawGuideLines(ctx, pad, w, h, maxY, "G");

  if (memoryHistory.length < 2) return;

  // Plot usage line + area
  const colors = getThemeColors();
  const values = memoryHistory.map((s) => s.used_gb);
  drawAreaLine(ctx, pad, w, h, maxY, values, MAX_MEMORY_SLOTS, colors.accent);

  // Time labels
  const timestamps = memoryHistory.map((s) => s.timestamp);
  const formatTime = (ts) => {
    const d = new Date(ts * 1000);
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`;
  };
  drawTimeLabels(ctx, pad, w, h, timestamps, MAX_MEMORY_SLOTS, formatTime);
}

async function loadMemoryReport() {
  const reportDiv = document.getElementById("memory-report");

  try {
    const data = await api.get("/memory/report");

    reportDiv.innerHTML = `
      <h3>Memory Report</h3>
      <div class="detail-grid">
        <div class="detail-item">
          <span class="label">Peak Used</span>
          <span class="value">${data.peak_used_gb.toFixed(2)} GB (${data.peak_pct.toFixed(1)}%)</span>
        </div>
        <div class="detail-item">
          <span class="label">Baseline Used</span>
          <span class="value">${data.baseline_used_gb.toFixed(2)} GB (${data.baseline_pct.toFixed(1)}%)</span>
        </div>
        <div class="detail-item">
          <span class="label">Delta</span>
          <span class="value">${data.delta_gb.toFixed(2)} GB</span>
        </div>
        <div class="detail-item">
          <span class="label">Total Memory</span>
          <span class="value">${data.total_gb.toFixed(1)} GB</span>
        </div>
        <div class="detail-item">
          <span class="label">Monitoring Duration</span>
          <span class="value">${data.duration_sec.toFixed(0)}s (${data.sample_count} samples)</span>
        </div>
      </div>`;
  } catch (err) {
    reportDiv.innerHTML = `<h3>Memory Report</h3><p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

export function initMemory() {
  // Memory tab initializes on-demand when the tab is switched to
}
