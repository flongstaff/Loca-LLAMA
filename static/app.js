/* Loca-LLAMA Webapp */
"use strict";

// ── API Wrapper ─────────────────────────────────────────────────────────────

const api = {
  async get(path) {
    const res = await fetch(`/api${path}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async post(path, body) {
    const res = await fetch(`/api${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
};

// ── Tab Switching ───────────────────────────────────────────────────────────

function switchTab(tabName) {
  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-content").forEach((section) => {
    section.classList.toggle("active", section.id === `tab-${tabName}`);
  });
}

document.querySelectorAll("[data-tab]").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

// ── Compatibility Tab (Phase 2: dropdowns only, analysis in Phase 3) ────────

async function loadCompatibilityDropdowns() {
  try {
    const [hwData, quantData] = await Promise.all([
      api.get("/hardware"),
      api.get("/quantizations"),
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

    // Enable analyze button when both selected (wired to action in Phase 3)
    const analyzeBtn = document.getElementById("analyze-btn");
    const checkReady = () => {
      analyzeBtn.disabled = !hwSelect.value || !quantSelect.value;
    };
    hwSelect.addEventListener("change", checkReady);
    quantSelect.addEventListener("change", checkReady);
  } catch (err) {
    console.error("Failed to load compatibility dropdowns:", err);
  }
}

// ── Models Tab ──────────────────────────────────────────────────────────────

let allFamilies = [];

async function loadModels(family) {
  try {
    const query = family ? `?family=${encodeURIComponent(family)}` : "";
    const data = await api.get(`/models${query}`);

    // Populate family filter on first load
    const familySelect = document.getElementById("family-select");
    if (allFamilies.length === 0 && data.families.length > 0) {
      allFamilies = data.families;
      familySelect.innerHTML = '<option value="">All Families</option>';
      allFamilies.forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        familySelect.appendChild(opt);
      });
    }

    const container = document.getElementById("models-table");
    if (data.count === 0) {
      container.innerHTML = '<p class="placeholder">No models found.</p>';
      return;
    }

    const rows = data.models
      .map(
        (m) => `<tr>
        <td>${escapeHtml(m.name)}</td>
        <td>${escapeHtml(m.family)}</td>
        <td>${m.params_billion}B</td>
        <td>${(m.default_context_length / 1024).toFixed(0)}K</td>
        <td>${(m.max_context_length / 1024).toFixed(0)}K</td>
        <td>${m.num_layers}</td>
        <td>${escapeHtml(m.license)}</td>
      </tr>`
      )
      .join("");

    container.innerHTML = `
      <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:0.5rem;">
        ${data.count} models${family ? ` in ${escapeHtml(family)}` : ""}
      </p>
      <table>
        <thead><tr>
          <th>Model</th><th>Family</th><th>Params</th>
          <th>Default Ctx</th><th>Max Ctx</th><th>Layers</th><th>License</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (err) {
    const container = document.getElementById("models-table");
    container.innerHTML = `<p class="placeholder" style="color:var(--danger)">Failed to load models: ${escapeHtml(err.message)}</p>`;
  }
}

// ── Placeholder Tab Renderers ───────────────────────────────────────────────

async function loadLocalModels() {
  // Will be implemented in Phase 4
}

async function loadHub() {
  // Will be implemented in Phase 4
}

async function loadBenchmark() {
  // Will be implemented in Phase 5
}

async function loadMemory() {
  // Will be implemented in Phase 6
}

// ── Init ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  console.log("Loca-LLAMA webapp loaded");

  // Load data for dropdowns
  loadCompatibilityDropdowns();
  loadModels();

  // Family filter
  document.getElementById("family-select").addEventListener("change", (e) => {
    loadModels(e.target.value || undefined);
  });
});
