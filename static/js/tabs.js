import { stopMemoryPolling, loadMemory } from "./memory.js";
import { stopMemoryPoll as stopMyMacPoll, loadMyMac } from "./mymac.js";

function switchTab(tabName) {
  stopMemoryPolling();
  stopMyMacPoll();
  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-content").forEach((section) => {
    section.classList.toggle("active", section.id === `tab-${tabName}`);
  });
  if (tabName === "memory") loadMemory();
  if (tabName === "mymac") loadMyMac();
}

export function initTabs() {
  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
}
