import { initTheme } from "./theme.js";
import { initTabs } from "./tabs.js";
import { initCompat } from "./compat.js";
import { initModels } from "./models.js";
import { initLocal } from "./local.js";
import { initHub } from "./hub.js";
import { initBenchmark } from "./benchmark.js";
import { initMemory } from "./memory.js";
import { initCalculator } from "./calculator.js";
import { initRecommend } from "./recommend.js";

document.addEventListener("DOMContentLoaded", () => {
  console.log("Loca-LLAMA webapp loaded");

  initTheme();
  initTabs();
  initCompat();
  initModels();
  initLocal();
  initHub();
  initBenchmark();
  initMemory();
  initCalculator();
  initRecommend();
});
