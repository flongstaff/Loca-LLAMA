import { api } from "./api.js";
import { escapeHtml, tierToCssClass } from "./utils.js";
import { drawBarChart } from "./chart.js";

let debounceTimer = null;
let calculatorModels = [];

function contextFromRange(val) {
  return Math.round(Math.pow(2, val));
}

function formatContextLabel(ctx) {
  return ctx >= 1024 ? `${(ctx / 1024).toFixed(0)}K` : String(ctx);
}

async function loadCalculatorModels() {
  try {
    const data = await api.get("/calculator/models");
    calculatorModels = data.models;

    const select = document.getElementById("calc-model-select");
    select.innerHTML = '<option value="">Custom parameters…</option>';
    data.models.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `${m.name} (${m.params_billion}B, ${m.num_layers}L)`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load calculator models:", err);
  }
}

function fillFromModel(index) {
  const m = calculatorModels[index];
  if (!m) return;

  document.getElementById("calc-params").value = m.params_billion;
  document.getElementById("calc-layers").value = m.num_layers;
  document.getElementById("calc-kv-heads").value = m.num_kv_heads;
  document.getElementById("calc-head-dim").value = m.head_dim;

  // Set context to default_context_length via log scale
  const logVal = Math.log2(m.default_context_length);
  const range = document.getElementById("calc-context-range");
  range.value = Math.max(7, Math.min(18, logVal));
  document.getElementById("calc-context-label").textContent =
    formatContextLabel(contextFromRange(range.value));

  scheduleEstimate();
}

function scheduleEstimate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runEstimate, 150);
}

async function runEstimate() {
  const params = {
    params_billion: parseFloat(document.getElementById("calc-params").value),
    bits_per_weight: parseFloat(document.getElementById("calc-bpw").value),
    num_layers: parseInt(document.getElementById("calc-layers").value, 10),
    num_kv_heads: parseInt(document.getElementById("calc-kv-heads").value, 10),
    head_dim: parseInt(document.getElementById("calc-head-dim").value, 10),
    context_length: contextFromRange(
      parseFloat(document.getElementById("calc-context-range").value)
    ),
    kv_bits: parseInt(document.getElementById("calc-kv-bits").value, 10),
  };

  // Basic client-side validation
  if (Object.values(params).some((v) => isNaN(v))) {
    return;
  }

  try {
    const data = await api.post("/calculator/estimate", params);
    renderBreakdown(data);
    renderBreakdownChart(data);
    renderHardwareTable(data.compatible_hardware);
  } catch (err) {
    document.getElementById("calc-total").textContent = "Error";
    document.getElementById("calc-hardware-table").innerHTML =
      `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

function renderBreakdown(data) {
  document.getElementById("calc-model-size").textContent =
    `${data.model_size_gb.toFixed(2)} GB`;
  document.getElementById("calc-kv-cache").textContent =
    `${data.kv_cache_gb.toFixed(2)} GB`;
  document.getElementById("calc-overhead").textContent =
    `${data.overhead_gb.toFixed(2)} GB`;
  document.getElementById("calc-total").textContent =
    `${data.total_memory_gb.toFixed(2)} GB`;
  document.getElementById("calc-disk-size").textContent =
    `${data.on_disk_size_gb.toFixed(2)} GB`;
}

function renderBreakdownChart(data) {
  const canvas = document.getElementById("calc-breakdown-chart");
  if (!canvas) return;

  // Stacked bar for VRAM breakdown + individual bars for compatible hardware
  const chartData = [
    {
      label: "Required",
      segments: [
        { label: "Model", value: data.model_size_gb, color: "#6c8cff" },
        { label: "KV Cache", value: data.kv_cache_gb, color: "#4caf50" },
        { label: "Overhead", value: data.overhead_gb, color: "#ff9800" },
      ],
    },
  ];

  // Add top compatible hardware entries for comparison
  if (data.compatible_hardware && data.compatible_hardware.length > 0) {
    const top = data.compatible_hardware.slice(0, 5);
    for (const hw of top) {
      chartData.push({
        label: hw.name,
        value: hw.memory_gb,
        color: hw.headroom_gb >= 8 ? "#4caf50" : hw.headroom_gb >= 0 ? "#ff9800" : "#f44336",
      });
    }
  }

  drawBarChart(canvas, chartData, {
    title: "Memory Breakdown vs Hardware",
    unit: " GB",
    height: 200,
  });
}

function renderHardwareTable(hardware) {
  const container = document.getElementById("calc-hardware-table");

  if (!hardware || hardware.length === 0) {
    container.innerHTML =
      '<p class="placeholder">No compatible hardware found for this configuration.</p>';
    return;
  }

  const rows = hardware
    .map(
      (hw) => `<tr>
        <td>${escapeHtml(hw.name)}</td>
        <td class="num">${hw.memory_gb} GB</td>
        <td><span class="badge ${tierToCssClass(hw.tier)}">${escapeHtml(hw.tier_label)}</span></td>
        <td class="num">${hw.headroom_gb >= 0 ? "+" : ""}${hw.headroom_gb.toFixed(1)} GB</td>
        <td class="num">${hw.estimated_tok_per_sec != null ? hw.estimated_tok_per_sec.toFixed(1) : "—"}</td>
      </tr>`
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Hardware</th>
          <th>Memory</th>
          <th>Tier</th>
          <th>Headroom</th>
          <th>Est. tok/s</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

export function initCalculator() {
  loadCalculatorModels();

  // Auto-fill from model dropdown
  document.getElementById("calc-model-select").addEventListener("change", (e) => {
    if (e.target.value) {
      fillFromModel(parseInt(e.target.value, 10));
    }
  });

  // Context range slider label update
  const ctxRange = document.getElementById("calc-context-range");
  ctxRange.addEventListener("input", () => {
    document.getElementById("calc-context-label").textContent =
      formatContextLabel(contextFromRange(parseFloat(ctxRange.value)));
    scheduleEstimate();
  });

  // Attach debounced estimate to all numeric inputs
  const inputIds = [
    "calc-params",
    "calc-bpw",
    "calc-layers",
    "calc-kv-heads",
    "calc-head-dim",
  ];
  inputIds.forEach((id) => {
    document.getElementById(id).addEventListener("input", scheduleEstimate);
  });

  // KV bits select
  document.getElementById("calc-kv-bits").addEventListener("change", scheduleEstimate);

  // Re-render chart on theme change
  document.addEventListener("themechange", () => scheduleEstimate());

  // Run initial estimate with default values
  runEstimate();
}
