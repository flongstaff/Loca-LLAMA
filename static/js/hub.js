import { api } from "./api.js";
import { escapeHtml, formatSizeGb, formatDownloads } from "./utils.js";

let hubDebounceTimer = null;

async function searchHub() {
  const query = document.getElementById("hub-search-input").value.trim();
  if (!query) return;

  const format = document.getElementById("hub-format-select").value || undefined;
  const sort = document.getElementById("hub-sort-select").value;
  const resultsDiv = document.getElementById("hub-results");
  const filesDiv = document.getElementById("hub-files");

  resultsDiv.innerHTML = '<p class="loading">Searching…</p>';
  filesDiv.style.display = "none";

  try {
    let url = `/hub/search?query=${encodeURIComponent(query)}&sort=${sort}&limit=30`;
    if (format) url += `&format=${format}`;
    const data = await api.get(url);

    if (data.count === 0) {
      resultsDiv.innerHTML = '<p class="placeholder">No models found.</p>';
      return;
    }

    const rows = data.results
      .map(
        (m) => `<tr data-repo="${escapeHtml(m.repo_id)}" style="cursor:pointer">
          <td>${escapeHtml(m.repo_id)}</td>
          <td class="num">${formatDownloads(m.downloads)}</td>
          <td class="num">${m.likes}</td>
          <td>${m.is_gguf ? '<span class="badge comfortable">GGUF</span>' : ""}${m.is_mlx ? '<span class="badge full-gpu">MLX</span>' : ""}</td>
          <td>${m.pipeline_tag ? escapeHtml(m.pipeline_tag) : "—"}</td>
        </tr>`
      )
      .join("");

    resultsDiv.innerHTML = `
      <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:0.5rem;">
        ${data.count} results for "${escapeHtml(query)}"
      </p>
      <table>
        <thead><tr>
          <th>Repository</th><th>Downloads</th><th>Likes</th>
          <th>Format</th><th>Pipeline</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    resultsDiv.querySelectorAll("tr[data-repo]").forEach((tr) => {
      tr.addEventListener("click", () => showRepoFiles(tr.dataset.repo));
    });
  } catch (err) {
    resultsDiv.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

async function showRepoFiles(repoId) {
  const filesDiv = document.getElementById("hub-files");
  filesDiv.style.display = "block";
  filesDiv.innerHTML = `<h3>${escapeHtml(repoId)}</h3><p class="loading">Loading files…</p>`;

  try {
    const data = await api.get(`/hub/files/${repoId}`);

    if (data.files.length === 0) {
      filesDiv.innerHTML = `<h3>${escapeHtml(repoId)}</h3><p class="placeholder">No files found.</p>`;
      return;
    }

    const rows = data.files
      .map(
        (f) => `<tr>
          <td>${escapeHtml(f.filename)}</td>
          <td class="num">${f.size > 0 ? formatSizeGb(f.size / (1024 * 1024 * 1024)) : "—"}</td>
        </tr>`
      )
      .join("");

    filesDiv.innerHTML = `
      <h3>${escapeHtml(repoId)} <span style="color:var(--text-muted);font-size:0.85rem">(${data.files.length} files)</span></h3>
      <table>
        <thead><tr><th>Filename</th><th>Size</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (err) {
    filesDiv.innerHTML = `<h3>${escapeHtml(repoId)}</h3><p class="error-message">${escapeHtml(err.message)}</p>`;
  }
}

export function initHub() {
  document.getElementById("hub-search-input").addEventListener("input", () => {
    clearTimeout(hubDebounceTimer);
    hubDebounceTimer = setTimeout(searchHub, 300);
  });
  document.getElementById("hub-format-select").addEventListener("change", () => {
    if (document.getElementById("hub-search-input").value.trim()) searchHub();
  });
  document.getElementById("hub-sort-select").addEventListener("change", () => {
    if (document.getElementById("hub-search-input").value.trim()) searchHub();
  });
}
