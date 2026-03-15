import { api } from "./api.js";
import { escapeHtml, tierToCssClass, detectHardware } from "./utils.js";

let recResults = [];
let recSortCol = "rank";
let recSortAsc = true;
let recSelectedModel = null;

const TIER_ORDER = { full_gpu: 0, comfortable: 1, tight_fit: 2, partial: 3, wont_fit: 4 };

async function loadRecommendDropdowns() {
  try {
    const hwData = await api.get("/hardware");

    const hwSelect = document.getElementById("rec-hw-select");
    hwSelect.innerHTML = '<option value="">Select hardware…</option>';
    hwData.hardware.forEach((hw) => {
      const opt = document.createElement("option");
      opt.value = hw.name;
      opt.textContent = `${hw.name} (${hw.memory_gb}GB, ${hw.memory_bandwidth_gbs} GB/s)`;
      hwSelect.appendChild(opt);
    });

    const recBtn = document.getElementById("rec-analyze-btn");
    hwSelect.addEventListener("change", () => {
      recBtn.disabled = !hwSelect.value;
    });

    recBtn.addEventListener("click", runRecommend);

    // Auto-detect hardware after dropdown is populated
    const detectBtn = document.getElementById("rec-detect-hw-btn");
    const feedbackEl = document.getElementById("rec-detect-hw-feedback");
    detectHardware(hwSelect, detectBtn, feedbackEl);
  } catch (err) {
    console.error("Failed to load recommend dropdowns:", err);
    const container = document.getElementById("rec-results");
    container.innerHTML = "";
    const p = document.createElement("p");
    p.className = "error-message";
    p.textContent = `Failed to load dropdowns: ${err.message}`;
    container.appendChild(p);
  }
}

async function runRecommend() {
  const hwName = document.getElementById("rec-hw-select").value;
  const useCase = document.getElementById("rec-usecase-select").value;
  if (!hwName) return;

  const container = document.getElementById("rec-results");
  const detail = document.getElementById("rec-detail");
  container.innerHTML = '<p class="loading">Finding best models…</p>';
  detail.classList.add("hidden");
  recSelectedModel = null;

  try {
    const data = await api.post("/recommend", {
      hardware_name: hwName,
      use_case: useCase,
    });

    recResults = data.recommendations;
    renderRecSummary(data);
    renderRecTable();
  } catch (err) {
    container.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

function renderRecSummary(data) {
  const div = document.getElementById("rec-summary");
  if (data.count === 0) {
    div.innerHTML = '<p class="text-muted">No models fit this hardware for the selected use case.</p>';
    return;
  }
  div.innerHTML = `
    <div class="tier-summary">
      <span class="text-muted">${data.count} recommendations for <strong>${escapeHtml(data.hardware)}</strong> (${escapeHtml(data.use_case)})</span>
    </div>`;
}

function renderRecTable() {
  const container = document.getElementById("rec-results");
  if (recResults.length === 0) {
    container.innerHTML = '<p class="placeholder">No recommendations found.</p>';
    return;
  }

  const sorted = [...recResults].sort((a, b) => {
    let va, vb;
    if (recSortCol === "tier") {
      va = TIER_ORDER[a.tier] ?? 5;
      vb = TIER_ORDER[b.tier] ?? 5;
    } else if (recSortCol === "model_name") {
      va = a.model_name.toLowerCase();
      vb = b.model_name.toLowerCase();
    } else {
      va = a[recSortCol] ?? 0;
      vb = b[recSortCol] ?? 0;
    }
    if (va < vb) return recSortAsc ? -1 : 1;
    if (va > vb) return recSortAsc ? 1 : -1;
    return 0;
  });

  const columns = [
    { key: "rank", label: "#" },
    { key: "model_name", label: "Model" },
    { key: "quant_name", label: "Quant" },
    { key: "total_memory_gb", label: "Memory (GB)" },
    { key: "memory_utilization_pct", label: "Util %" },
    { key: "tier", label: "Tier" },
    { key: "estimated_tok_per_sec", label: "Est. tok/s" },
    { key: "max_context_k", label: "Max Ctx" },
  ];

  const ths = columns
    .map((col) => {
      let cls = "sortable";
      if (recSortCol === col.key) cls += recSortAsc ? " sort-asc" : " sort-desc";
      return `<th class="${cls}" data-sort="${col.key}">${col.label}</th>`;
    })
    .join("");

  const rows = sorted
    .map(
      (r) => `<tr data-model="${escapeHtml(r.model_name)}" class="cursor-pointer${recSelectedModel === r.model_name ? " active-row" : ""}">
        <td class="num">${r.rank}</td>
        <td>${escapeHtml(r.model_name)}</td>
        <td>${escapeHtml(r.quant_name)}</td>
        <td class="num">${r.total_memory_gb.toFixed(1)}</td>
        <td class="num">${r.memory_utilization_pct.toFixed(0)}%</td>
        <td><span class="badge ${tierToCssClass(r.tier)}">${escapeHtml(r.tier_label)}</span></td>
        <td class="num">${r.estimated_tok_per_sec != null ? r.estimated_tok_per_sec.toFixed(1) : "—"}</td>
        <td class="num">${escapeHtml(r.max_context_k)}</td>
      </tr>`
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead><tr>${ths}</tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  container.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (recSortCol === col) {
        recSortAsc = !recSortAsc;
      } else {
        recSortCol = col;
        recSortAsc = true;
      }
      renderRecTable();
    });
  });

  container.querySelectorAll("tr[data-model]").forEach((tr) => {
    tr.addEventListener("click", () => {
      const name = tr.dataset.model;
      if (recSelectedModel === name) {
        recSelectedModel = null;
        document.getElementById("rec-detail").classList.add("hidden");
        renderRecTable();
      } else {
        recSelectedModel = name;
        showRecDetail(name);
        renderRecTable();
      }
    });
  });
}

function showRecDetail(modelName) {
  const detail = document.getElementById("rec-detail");
  const result = recResults.find((r) => r.model_name === modelName);
  if (!result) return;

  detail.classList.remove("hidden");
  detail.innerHTML = `
    <h3>${escapeHtml(result.model_name)} <span class="badge ${tierToCssClass(result.tier)}">${escapeHtml(result.tier_label)}</span></h3>
    <p class="text-muted mb-2">Recommended quantization: <strong>${escapeHtml(result.quant_name)}</strong></p>
    <div class="detail-grid">
      <div class="detail-item"><span class="label">Model Size</span><span class="value">${result.model_size_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">KV Cache</span><span class="value">${result.kv_cache_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">Overhead</span><span class="value">${result.overhead_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">Total Memory</span><span class="value">${result.total_memory_gb.toFixed(2)} GB</span></div>
      <div class="detail-item"><span class="label">Available</span><span class="value">${result.available_memory_gb.toFixed(1)} GB</span></div>
      <div class="detail-item"><span class="label">Headroom</span><span class="value">${result.headroom_gb.toFixed(1)} GB</span></div>
      <div class="detail-item"><span class="label">Utilization</span><span class="value">${result.memory_utilization_pct.toFixed(1)}%</span></div>
      <div class="detail-item"><span class="label">Est. Speed</span><span class="value">${result.estimated_tok_per_sec != null ? result.estimated_tok_per_sec.toFixed(1) + " tok/s" : "—"}</span></div>
      <div class="detail-item"><span class="label">GPU Layers</span><span class="value">${result.gpu_layers != null ? result.gpu_layers + " / " + result.total_layers : "—"}</span></div>
      <div class="detail-item"><span class="label">GPU Offload</span><span class="value">${result.offload_pct != null ? result.offload_pct.toFixed(0) + "%" : "—"}</span></div>
      <div class="detail-item"><span class="label">Context</span><span class="value">${(result.context_length / 1024).toFixed(0)}K</span></div>
      <div class="detail-item"><span class="label">Max Context</span><span class="value">${escapeHtml(result.max_context_k)}</span></div>
    </div>`;
}

export function initRecommend() {
  loadRecommendDropdowns();

  const hwSelect = document.getElementById("rec-hw-select");
  const detectBtn = document.getElementById("rec-detect-hw-btn");
  const feedbackEl = document.getElementById("rec-detect-hw-feedback");
  detectBtn.addEventListener("click", () => detectHardware(hwSelect, detectBtn, feedbackEl));

  document.getElementById("rec-usecase-select").addEventListener("change", () => {
    if (document.getElementById("rec-hw-select").value && recResults.length > 0) runRecommend();
  });
}
