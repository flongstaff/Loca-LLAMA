import { api } from "./api.js";
import { escapeHtml, tierToCssClass } from "./utils.js";

let compatResults = [];
let compatSortCol = "tier";
let compatSortAsc = true;

const TIER_ORDER = { full_gpu: 0, comfortable: 1, tight_fit: 2, partial: 3, wont_fit: 4 };

async function loadCompatibilityDropdowns() {
  try {
    const [hwData, quantData, modelData] = await Promise.all([
      api.get("/hardware"),
      api.get("/quantizations"),
      api.get("/models"),
    ]);

    const hwSelect = document.getElementById("hw-select");
    hwSelect.innerHTML = '<option value="">Select hardware…</option>';
    hwData.hardware.forEach((hw) => {
      const opt = document.createElement("option");
      opt.value = hw.name;
      opt.textContent = `${hw.name} (${hw.memory_gb}GB, ${hw.memory_bandwidth_gbs} GB/s)`;
      hwSelect.appendChild(opt);
    });

    const quantSelect = document.getElementById("quant-select");
    quantSelect.innerHTML = '<option value="">Select quantization…</option>';
    quantData.formats.forEach((q) => {
      const opt = document.createElement("option");
      opt.value = q.name;
      const rec = quantData.recommended.includes(q.name) ? " ★" : "";
      opt.textContent = `${q.name} (${q.bits_per_weight} bpw)${rec}`;
      quantSelect.appendChild(opt);
    });

    const familySelect = document.getElementById("compat-family-select");
    familySelect.innerHTML = '<option value="">All Families</option>';
    modelData.families.forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f;
      familySelect.appendChild(opt);
    });

    const analyzeBtn = document.getElementById("analyze-btn");
    const checkReady = () => {
      analyzeBtn.disabled = !hwSelect.value || !quantSelect.value;
    };
    hwSelect.addEventListener("change", checkReady);
    quantSelect.addEventListener("change", checkReady);

    analyzeBtn.addEventListener("click", runAnalysis);
  } catch (err) {
    console.error("Failed to load compatibility dropdowns:", err);
    const resultsEl = document.getElementById("compat-results");
    resultsEl.innerHTML = "";
    const p = document.createElement("p");
    p.className = "error-message";
    p.textContent = `Failed to load dropdowns: ${err.message} `;
    const retryLink = document.createElement("span");
    retryLink.className = "retry-link";
    retryLink.textContent = "Retry";
    retryLink.addEventListener("click", loadCompatibilityDropdowns);
    p.appendChild(retryLink);
    resultsEl.appendChild(p);
  }
}

async function runAnalysis() {
  const hwName = document.getElementById("hw-select").value;
  const quantName = document.getElementById("quant-select").value;
  const family = document.getElementById("compat-family-select").value || undefined;
  const onlyFits = document.getElementById("only-fits-check").checked;

  if (!hwName || !quantName) return;

  const resultsDiv = document.getElementById("compat-results");
  const summaryDiv = document.getElementById("compat-summary");
  const detailDiv = document.getElementById("compat-detail");

  resultsDiv.innerHTML = '<p class="loading">Analyzing…</p>';
  summaryDiv.innerHTML = "";
  detailDiv.style.display = "none";

  try {
    const data = await api.post("/analyze/all", {
      hardware_name: hwName,
      quant_names: [quantName],
      family: family,
      only_fits: onlyFits,
      include_partial: true,
    });

    compatResults = data.results;
    renderTierSummary(data.summary, data.count);
    renderCompatTable();
  } catch (err) {
    resultsDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

function renderTierSummary(summary, count) {
  const div = document.getElementById("compat-summary");
  const tiers = [
    { key: "full_gpu", label: "Full GPU", css: "full-gpu" },
    { key: "comfortable", label: "Comfortable", css: "comfortable" },
    { key: "tight_fit", label: "Tight Fit", css: "tight-fit" },
    { key: "partial_offload", label: "Partial Offload", css: "partial-offload" },
    { key: "wont_fit", label: "Won't Fit", css: "wont-fit" },
  ];

  div.innerHTML = `
    <div class="tier-summary">
      <span style="color:var(--text-muted)">${count} results:</span>
      ${tiers
        .filter((t) => summary[t.key] > 0)
        .map(
          (t) => `<span class="tier-count">
            <span class="tier-dot ${t.css}"></span>
            ${summary[t.key]} ${t.label}
          </span>`
        )
        .join("")}
    </div>`;
}

function renderCompatTable() {
  const container = document.getElementById("compat-results");
  if (compatResults.length === 0) {
    container.innerHTML = '<p class="placeholder">No results found.</p>';
    return;
  }

  const sorted = [...compatResults].sort((a, b) => {
    let va, vb;
    if (compatSortCol === "tier") {
      va = TIER_ORDER[a.tier] ?? 5;
      vb = TIER_ORDER[b.tier] ?? 5;
    } else if (compatSortCol === "model_name") {
      va = a.model_name.toLowerCase();
      vb = b.model_name.toLowerCase();
    } else {
      va = a[compatSortCol] ?? 0;
      vb = b[compatSortCol] ?? 0;
    }
    if (va < vb) return compatSortAsc ? -1 : 1;
    if (va > vb) return compatSortAsc ? 1 : -1;
    return 0;
  });

  const columns = [
    { key: "model_name", label: "Model" },
    { key: "total_memory_gb", label: "Total Mem (GB)" },
    { key: "headroom_gb", label: "Headroom (GB)" },
    { key: "memory_utilization_pct", label: "Util %" },
    { key: "tier", label: "Tier" },
    { key: "estimated_tok_per_sec", label: "Est. tok/s" },
  ];

  const ths = columns
    .map((col) => {
      let cls = "sortable";
      if (compatSortCol === col.key) cls += compatSortAsc ? " sort-asc" : " sort-desc";
      return `<th class="${cls}" data-sort="${col.key}">${col.label}</th>`;
    })
    .join("");

  const rows = sorted
    .map(
      (r) => `<tr data-model="${escapeHtml(r.model_name)}" style="cursor:pointer">
        <td>${escapeHtml(r.model_name)}</td>
        <td class="num">${r.total_memory_gb.toFixed(1)}</td>
        <td class="num">${r.headroom_gb.toFixed(1)}</td>
        <td class="num">${r.memory_utilization_pct.toFixed(0)}%</td>
        <td><span class="badge ${tierToCssClass(r.tier)}">${escapeHtml(r.tier_label)}</span></td>
        <td class="num">${r.estimated_tok_per_sec != null ? r.estimated_tok_per_sec.toFixed(1) : "—"}</td>
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
      if (compatSortCol === col) {
        compatSortAsc = !compatSortAsc;
      } else {
        compatSortCol = col;
        compatSortAsc = true;
      }
      renderCompatTable();
    });
  });

  container.querySelectorAll("tr[data-model]").forEach((tr) => {
    tr.addEventListener("click", () => showDetail(tr.dataset.model));
  });
}

async function showDetail(modelName) {
  const detail = document.getElementById("compat-detail");
  const result = compatResults.find((r) => r.model_name === modelName);
  if (!result) return;

  const hwName = document.getElementById("hw-select").value;
  const quantName = document.getElementById("quant-select").value;

  let maxCtx = "—";
  try {
    const mc = await api.post("/analyze/max-context", {
      hardware_name: hwName,
      model_name: modelName,
      quant_name: quantName,
    });
    maxCtx = mc.max_context_k;
  } catch {
    /* ignore */
  }

  detail.style.display = "block";
  detail.innerHTML = `
    <h3>${escapeHtml(result.model_name)} <span class="badge ${tierToCssClass(result.tier)}">${escapeHtml(result.tier_label)}</span></h3>
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
      <div class="detail-item"><span class="label">Max Context</span><span class="value">${maxCtx}</span></div>
    </div>`;
}

export function initCompat() {
  loadCompatibilityDropdowns();

  document.getElementById("only-fits-check").addEventListener("change", () => {
    if (compatResults.length > 0) runAnalysis();
  });

  document.getElementById("compat-family-select").addEventListener("change", () => {
    if (document.getElementById("hw-select").value) runAnalysis();
  });
}
