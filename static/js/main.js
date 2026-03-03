import { initTabs } from "./tabs.js";
import { initCompat } from "./compat.js";
import { initModels } from "./models.js";
import { initLocal } from "./local.js";
import { initHub } from "./hub.js";
import { initBenchmark } from "./benchmark.js";
import { initMemory } from "./memory.js";

document.addEventListener("DOMContentLoaded", () => {
  console.log("Loca-LLAMA webapp loaded");

  initTabs();
  initCompat();
  initModels();
  initLocal();
  initHub();
  initBenchmark();
  initMemory();
});
