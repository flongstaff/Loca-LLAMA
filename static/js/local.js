import { api } from "./api.js";
import { escapeHtml, formatSizeGb } from "./utils.js";

async function scanLocalModels(customDir) {
  const resultsDiv = document.getElementById("scan-results");
  const summaryDiv = document.getElementById("scan-summary");

  resultsDiv.innerHTML = '<p class="loading">Scanning…</p>';
  summaryDiv.innerHTML = "";

  try {
    const query = customDir ? `?custom_dir=${encodeURIComponent(customDir)}` : "";
    const data = await api.get(`/scanner/local${query}`);

    if (data.count === 0) {
      resultsDiv.innerHTML = '<p class="placeholder">No models found. Try scanning a custom directory.</p>';
      return;
    }

    const sourceEntries = Object.entries(data.sources)
      .map(([src, n]) => `<span class="tier-count">${n} ${escapeHtml(src)}</span>`)
      .join("");

    summaryDiv.innerHTML = `
      <div class="tier-summary">
        <span style="color:var(--text-muted)">${data.count} models (${formatSizeGb(data.total_size_gb)} total):</span>
        ${sourceEntries}
      </div>`;

    const rows = data.models
      .map(
        (m) => `<tr>
          <td>${escapeHtml(m.name)}</td>
          <td>${formatSizeGb(m.size_gb)}</td>
          <td>${escapeHtml(m.format)}</td>
          <td>${escapeHtml(m.source)}</td>
          <td>${m.quant ? escapeHtml(m.quant) : "—"}</td>
          <td>${m.family ? escapeHtml(m.family) : "—"}</td>
        </tr>`
      )
      .join("");

    resultsDiv.innerHTML = `
      <table>
        <thead><tr>
          <th>Name</th><th>Size</th><th>Format</th>
          <th>Source</th><th>Quant</th><th>Family</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (err) {
    resultsDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

export function initLocal() {
  document.getElementById("scan-btn").addEventListener("click", () => scanLocalModels());
  document.getElementById("scan-custom-btn").addEventListener("click", () => {
    const dir = document.getElementById("custom-dir-input").value.trim();
    if (dir) scanLocalModels(dir);
  });
}
