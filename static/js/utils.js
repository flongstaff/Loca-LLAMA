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

/**
 * Call GET /api/hardware/detect and auto-select the matching dropdown option.
 * @param {HTMLSelectElement} hwSelect - the hardware dropdown
 * @param {HTMLButtonElement} btn - the "Detect My Mac" button
 * @param {HTMLElement} feedbackEl - inline span for status messages
 */
export async function detectHardware(hwSelect, btn, feedbackEl) {
  btn.disabled = true;
  btn.textContent = "Detecting…";
  btn.setAttribute("aria-busy", "true");
  feedbackEl.textContent = "";

  try {
    const resp = await fetch("/api/hardware/detect");
    const data = await resp.json();

    if (data.detected && data.name) {
      // Find and select the matching option
      const options = Array.from(hwSelect.options);
      const match = options.find((o) => o.value === data.name);
      if (match) {
        hwSelect.value = data.name;
        hwSelect.dispatchEvent(new Event("change"));
      }
      feedbackEl.textContent = `Detected: ${data.name}`;
      feedbackEl.className = "detect-feedback detect-success";
      setTimeout(() => { feedbackEl.textContent = ""; }, 3000);
    } else {
      feedbackEl.textContent = "Detection unavailable — select manually";
      feedbackEl.className = "detect-feedback detect-muted";
      setTimeout(() => { feedbackEl.textContent = ""; }, 5000);
    }
  } catch {
    feedbackEl.textContent = "Detection unavailable — select manually";
    feedbackEl.className = "detect-feedback detect-muted";
    setTimeout(() => { feedbackEl.textContent = ""; }, 5000);
  }

  btn.disabled = false;
  btn.textContent = "Detect My Mac";
  btn.removeAttribute("aria-busy");
}

export function formatDownloads(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/**
 * Copy text to clipboard and show brief visual feedback on the trigger button.
 * @param {string} text - The text to copy
 * @param {HTMLButtonElement} btn - The button that triggered the copy
 */
export async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Fallback for insecure contexts (localhost without HTTPS)
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
  if (btn) {
    const original = btn.textContent;
    btn.textContent = "Copied!";
    btn.classList.add("copy-success");
    setTimeout(() => {
      btn.textContent = original;
      btn.classList.remove("copy-success");
    }, 1500);
  }
}
