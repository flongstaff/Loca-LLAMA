export function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

export function tierToCssClass(tier) {
  const map = {
    full_gpu: "full-gpu",
    comfortable: "comfortable",
    tight_fit: "tight-fit",
    partial: "partial-offload",
    wont_fit: "wont-fit",
  };
  return map[tier] || "wont-fit";
}

export function formatSizeGb(gb) {
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(gb * 1024).toFixed(0)} MB`;
}

export function formatDownloads(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
