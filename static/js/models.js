import { api } from "./api.js";
import { escapeHtml } from "./utils.js";

let allFamilies = [];

async function loadModels(family) {
  const container = document.getElementById("models-table");
  container.innerHTML = '<p class="loading">Loading models…</p>';

  try {
    const query = family ? `?family=${encodeURIComponent(family)}` : "";
    const data = await api.get(`/models${query}`);

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

    if (data.count === 0) {
      container.innerHTML = '<p class="placeholder">No models found.</p>';
      return;
    }

    const rows = data.models
      .map(
        (m) => `<tr class="clickable-row" data-model="${escapeHtml(m.name)}" role="button" tabindex="0" aria-expanded="false">
        <td>${escapeHtml(m.name)}</td>
        <td>${escapeHtml(m.family)}</td>
        <td class="num">${m.params_billion}B</td>
        <td class="num">${(m.default_context_length / 1024).toFixed(0)}K</td>
        <td class="num">${(m.max_context_length / 1024).toFixed(0)}K</td>
        <td class="num">${m.num_layers}</td>
        <td>${escapeHtml(m.license)}</td>
      </tr>`
      )
      .join("");

    container.innerHTML = `
      <p class="text-muted text-sm mb-2">
        ${data.count} models${family ? ` in ${escapeHtml(family)}` : ""} — click a row for details
      </p>
      <table>
        <thead><tr>
          <th>Model</th><th>Family</th><th>Params</th>
          <th>Default Ctx</th><th>Max Ctx</th><th>Layers</th><th>License</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    container.querySelectorAll("tr.clickable-row").forEach((tr) => {
      const handler = () => showModelDetail(tr.dataset.model);
      tr.addEventListener("click", handler);
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handler(); }
      });
    });
  } catch (err) {
    container.innerHTML = `<p class="error-message">Failed to load models: ${escapeHtml(err.message)}</p>`;
  }
}

async function showModelDetail(modelName) {
  const detail = document.getElementById("model-detail");
  detail.classList.remove("hidden");
  detail.innerHTML = '<p class="loading">Loading details…</p>';

  document.querySelectorAll("#models-table tr.clickable-row").forEach((tr) => {
    tr.classList.toggle("active-row", tr.dataset.model === modelName);
    tr.setAttribute("aria-expanded", tr.dataset.model === modelName ? "true" : "false");
  });
  try {
    const m = await api.get(`/models/${encodeURIComponent(modelName)}`);
    detail.innerHTML = `
      <h3>${escapeHtml(m.name)}</h3>
      <div class="detail-grid">
        <div class="detail-item"><span class="label">Family</span><span class="value">${escapeHtml(m.family)}</span></div>
        <div class="detail-item"><span class="label">Parameters</span><span class="value">${m.params_billion}B</span></div>
        <div class="detail-item"><span class="label">Default Context</span><span class="value">${(m.default_context_length / 1024).toFixed(0)}K</span></div>
        <div class="detail-item"><span class="label">Max Context</span><span class="value">${(m.max_context_length / 1024).toFixed(0)}K</span></div>
        <div class="detail-item"><span class="label">Layers</span><span class="value">${m.num_layers}</span></div>
        <div class="detail-item"><span class="label">KV Heads</span><span class="value">${m.num_kv_heads}</span></div>
        <div class="detail-item"><span class="label">Head Dim</span><span class="value">${m.head_dim}</span></div>
        <div class="detail-item"><span class="label">License</span><span class="value">${escapeHtml(m.license)}</span></div>
      </div>`;
  } catch (err) {
    detail.innerHTML = `<p class="error-message">Failed to load model details: ${escapeHtml(err.message)}</p>`;
  }
}

export function initModels() {
  loadModels();

  document.getElementById("family-select").addEventListener("change", (e) => {
    loadModels(e.target.value || undefined);
  });
}
