const STORAGE_KEY = "loca-llama-theme";

const getSystemTheme = () =>
  window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";

const applyTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
};

const safeGetItem = (key) => {
  try { return localStorage.getItem(key); } catch { return null; }
};

const safeSetItem = (key, value) => {
  try { localStorage.setItem(key, value); } catch { /* private browsing */ }
};

export const initTheme = () => {
  const saved = safeGetItem(STORAGE_KEY);
  const theme = saved || getSystemTheme();
  applyTheme(theme);

  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const current = document.documentElement.dataset.theme;
      const next = current === "dark" ? "light" : "dark";
      safeSetItem(STORAGE_KEY, next);
      applyTheme(next);
    });
  }

  // Listen for OS theme changes when no user preference is saved
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
    if (!safeGetItem(STORAGE_KEY)) {
      applyTheme(e.matches ? "light" : "dark");
    }
  });
};
