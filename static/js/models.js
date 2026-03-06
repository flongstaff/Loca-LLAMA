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
    container.innerHTML = `<p class="error-message">Failed to load models: ${escapeHtml(err.message)}</p>`;
  }
}

export function initModels() {
  loadModels();

  document.getElementById("family-select").addEventListener("change", (e) => {
    loadModels(e.target.value || undefined);
  });
}
