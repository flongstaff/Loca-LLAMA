# Loca-LLAMA v2 Webapp -- UX Brief

**Date**: 2026-03-02
**Author**: UX Designer Agent
**Status**: Draft
**Inputs**: pm.md (Product Requirements), eng.md (Technical Design), RESEARCH.md

---

## 1. User Stories with Acceptance Criteria

### Persona 1: Mac Developer (Primary)

#### US-1.1: Hardware Compatibility Check (P0)

**As a** Mac Developer, **I want to** select my Apple Silicon hardware and instantly see which LLMs are compatible, **so that** I avoid downloading models that won't run on my machine.

**Acceptance Criteria**:
- Hardware selector dropdown is populated with all 44 Apple Silicon configurations on page load.
- Selecting a hardware config triggers an analysis and displays results within 500ms.
- Results table shows model name, family, quantization, total memory, headroom, tier, and estimated tokens/sec.
- Rows are color-coded by compatibility tier with text labels (not color alone).
- WONT_FIT models are hidden by default; a "Show all" toggle reveals them.
- Results persist when switching tabs and returning (no re-fetch required).

#### US-1.2: Model Detail Drill-Down (P0)

**As a** Mac Developer, **I want to** click a model in the compatibility table and see a detailed memory breakdown, **so that** I understand how memory is allocated across model weights, KV cache, and overhead.

**Acceptance Criteria**:
- Clicking a table row opens an in-page detail panel (no full page navigation).
- Detail panel displays: memory breakdown visualization (model size, KV cache, overhead, headroom), quantization comparison table (all 13 formats), max context per quantization, and recommended configuration.
- A "Back to results" button returns to the compatibility table without re-fetching.
- The detail panel is keyboard-accessible (Escape closes it, Tab cycles through interactive elements).

#### US-1.3: Quantization Comparison (P1)

**As a** Mac Developer, **I want to** compare Q4_K_M vs Q6_K vs Q8_0 for the same model on my hardware, **so that** I can evaluate the trade-off between quality and speed.

**Acceptance Criteria**:
- The model detail view shows all 13 quantization formats in a comparison table.
- Each row shows: format name, bits per weight, model size (GB), total memory (GB), tier, estimated tokens/sec.
- Rows are sorted by bits per weight ascending (smallest first).
- Rows that exceed available memory are visually de-emphasized (lower opacity or strikethrough).
- The recommended quantization for the selected hardware is highlighted with a badge.

#### US-1.4: Side-by-Side Filter and Sort (P1)

**As a** Mac Developer, **I want to** filter the compatibility table by model family, tier, and quantization, **so that** I can narrow down options to what is most relevant.

**Acceptance Criteria**:
- Family dropdown populated from the models API (e.g., Llama, Qwen, DeepSeek).
- Tier filter checkboxes: FULL_GPU, COMFORTABLE, TIGHT_FIT, PARTIAL_OFFLOAD (each independently toggleable).
- Quantization multi-select filter (defaults to recommended: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16).
- Table is sortable by clicking any column header. Active sort column shows a directional arrow indicator.
- Filter and sort states are preserved when switching tabs and returning.

---

### Persona 2: Data Scientist / ML Engineer

#### US-2.1: Browse All Models (P0)

**As a** Data Scientist, **I want to** browse all 49+ known models with metadata, **so that** I can evaluate options without memorizing CLI commands.

**Acceptance Criteria**:
- Models tab displays all models in a searchable, filterable table.
- Columns: name, family, params (B), default context length, max context length, license.
- Family filter pills allow quick filtering (clicking a pill toggles that family).
- A text search field filters by model name (client-side, instant).
- Clicking a model row shows a detail card with full specs and template info (if available).

#### US-2.2: Find Largest Compatible Model (P1)

**As a** Data Scientist, **I want to** sort compatible models by parameter count descending, **so that** I can find the largest model that fits comfortably on my hardware.

**Acceptance Criteria**:
- Compatibility table is sortable by any column, including parameter count and total memory.
- Sorting by headroom ascending shows the tightest fits first; sorting descending shows the most comfortable fits.
- The tier filter defaults to showing FULL_GPU and COMFORTABLE only (user can expand).
- A summary line above the table reads: "X of Y models fit on [hardware name]" with tier breakdown.

#### US-2.3: Context Length Impact (P2)

**As a** Data Scientist, **I want to** adjust context length and see how it affects compatibility, **so that** I can find the right balance between context window and model size.

**Acceptance Criteria**:
- A context length control (slider or number input) is available on the Compatibility tab.
- Changing context length triggers a re-analysis with the new value.
- The compatibility table updates to reflect changed memory estimates and tiers.
- The context length control shows common presets: 2K, 4K, 8K, 16K, 32K, 64K, 128K.

---

### Persona 3: Power User / Runtime Comparer

#### US-3.1: Detect Running Runtimes (P0)

**As a** Power User, **I want to** see which runtimes (LM Studio, llama.cpp) are currently running and what models they have loaded, **so that** I know what is available for benchmarking.

**Acceptance Criteria**:
- Benchmark tab auto-detects runtimes on open (calls `GET /api/runtime/status`).
- Each runtime shows: name, running status (green dot / red dot), URL, loaded models.
- A "Refresh" button re-checks runtime status.
- If no runtimes are detected, an informative empty state is displayed with instructions.

#### US-3.2: Run Benchmark (P0)

**As a** Power User, **I want to** run a benchmark against a loaded model and see tokens/sec results, **so that** I can compare runtime performance.

**Acceptance Criteria**:
- Benchmark form includes: runtime selector, model selector (from loaded models), prompt type, duration, and number of runs.
- Clicking "Start Benchmark" submits the job and shows a progress indicator.
- Progress updates every 2 seconds via polling (shows current run number / total).
- On completion, results display: tokens/sec (mean, min, max), prompt eval speed, time to first token.
- On error, a clear message explains the failure with actionable guidance.
- Only one benchmark can run at a time; the "Start Benchmark" button is disabled during execution.

#### US-3.3: Monitor Memory During Benchmark (P1)

**As a** Power User, **I want to** see real-time memory usage during a benchmark, **so that** I can detect memory pressure and swapping.

**Acceptance Criteria**:
- Memory tab shows a live gauge (polls every 2 seconds).
- Gauge displays: used memory / total memory, percentage, and pressure level.
- A simple history chart shows the last 60 readings (~2 minutes of data).
- Pressure levels are color-coded: normal (green), warn (yellow), critical (red).

---

### Persona 4: Model Explorer

#### US-4.1: Scan Local Models (P0)

**As a** Model Explorer, **I want to** scan my filesystem for downloaded models, **so that** I can see what I already have and check their compatibility.

**Acceptance Criteria**:
- "Scan" button triggers a filesystem scan of default paths (LM Studio, llama.cpp, HuggingFace cache).
- Results show: model name, file size (GB), format badge (GGUF/MLX/SafeTensors), source badge, detected quantization, detected family.
- A summary bar shows: total count, total size, breakdown by source.
- "Scan Custom Directory" allows entering a custom path for scanning.
- Loading state is shown during scan (may take up to 5 seconds).

#### US-4.2: Search HuggingFace (P0)

**As a** Model Explorer, **I want to** search HuggingFace for GGUF and MLX models, **so that** I can discover new models and assess their feasibility.

**Acceptance Criteria**:
- Search input field with debounced input (300ms delay before API call).
- Format filter: All, GGUF only, MLX only.
- Sort options: downloads (default), likes, recently modified.
- Results display: repo ID, author, download count, like count, format badges.
- Each result has a "Check compatibility" action that cross-references with the user's selected hardware.
- Network errors display a clear message: "Unable to reach HuggingFace. Check your internet connection."

#### US-4.3: View Recommended Settings (P1)

**As a** Model Explorer, **I want to** see recommended settings for a model (quantization, context length, temperature), **so that** I can configure my runtime optimally.

**Acceptance Criteria**:
- Model detail view includes a "Recommended Settings" section when a template exists for the model family.
- Settings shown: recommended quantization per hardware tier, context length, temperature, and other generation parameters.
- LM Studio preset export and llama.cpp command generation are available as copy-to-clipboard actions.
- If no template exists for the model, a message reads: "No preset available for this model family."

---

## 2. User Flows

### Flow 1: Check Hardware Compatibility

**Goal**: Determine which LLMs run on the user's hardware.

1. User opens `http://localhost:8000/` -- the Compatibility tab is active by default.
2. The hardware selector dropdown loads all 44 configurations from the API.
3. User selects their hardware (e.g., "M4 Pro 48GB") from the dropdown.
4. A loading spinner appears in the results area.
5. The API returns compatibility data; the results table renders with rows color-coded by tier.
6. Above the table, a summary reads: "38 of 49 models fit on M4 Pro 48GB (38 Full GPU, 22 Comfortable, 8 Tight Fit)."
7. User uses filter controls (family dropdown, tier checkboxes, quant selector) to narrow results.
8. User clicks a column header to sort (e.g., "Total Memory" ascending to see lightest models first).
9. User clicks a model row (e.g., "Qwen 2.5 32B -- Q4_K_M").
10. A detail panel slides in from the right (or expands below the row) showing:
    - Memory breakdown bar (model weights | KV cache | overhead | headroom).
    - Quantization comparison table (all 13 formats).
    - Recommended settings (if template exists).
    - Copy-to-clipboard llama.cpp command.
11. User clicks "Back to results" or presses Escape to close the detail panel.

### Flow 2: Browse and Filter Models

**Goal**: Explore the full model database, find models by family or size.

1. User clicks the "Models" tab in the navigation.
2. A table of all 49+ models loads instantly (in-memory data, no loading state needed).
3. User clicks a family filter pill (e.g., "Qwen") -- the table filters to Qwen models only.
4. User types "32B" in the search field -- the table further filters to models containing "32B".
5. User clicks a model row to expand a detail card.
6. The detail card shows: model specs (params, layers, KV heads, head dim), license, max context length.
7. If a template exists, recommended settings are displayed.
8. User clicks "Check compatibility" -- the app switches to the Compatibility tab with this model pre-selected as a filter.

### Flow 3: Scan Local Models

**Goal**: Discover models already downloaded on the user's machine.

1. User clicks the "Local Models" tab.
2. An empty state displays: "Click Scan to discover models on your machine."
3. User clicks the "Scan" button.
4. A loading spinner displays with text: "Scanning default locations..."
5. After 1-3 seconds, results appear in a table/card layout.
6. A summary bar shows: "12 models found (145.2 GB total) -- LM Studio: 8, HuggingFace: 3, MLX: 1."
7. User reviews the list, noting format badges (GGUF, MLX, SafeTensors) and source badges.
8. Optionally, user clicks "Scan Custom Directory," enters a path, and triggers a targeted scan.
9. New results merge with (or replace) the existing results.

### Flow 4: Search HuggingFace

**Goal**: Find new models on HuggingFace and check if they fit.

1. User clicks the "HuggingFace" tab.
2. A search bar with format filter (All / GGUF / MLX) and sort selector is displayed.
3. User types "deepseek reasoning gguf" in the search field.
4. After 300ms of no typing, the API call fires. A spinner appears below the search bar.
5. Results render as cards: repo ID, author, download count, like count, tags.
6. User clicks a format filter to narrow to "GGUF only."
7. User clicks "Check compatibility" on a result card.
8. The app navigates to the Compatibility tab, pre-filtered to that model's family (if in the database), or shows a message: "This model is not in the Loca-LLAMA database. Estimated size: X GB based on parameters."
9. If the user wants more info, they click a result card to expand file listing and HuggingFace config details.

### Flow 5: Run a Benchmark

**Goal**: Benchmark a model on a runtime and see performance numbers.

1. User clicks the "Benchmark" tab.
2. The app auto-detects runtimes by calling `GET /api/runtime/status`.
3. A status panel shows detected runtimes with status indicators (green = running, red = not detected).
4. If no runtimes are detected, the empty state reads: "No runtimes detected. Start LM Studio or llama.cpp to run benchmarks." A "Refresh" button allows re-checking.
5. If runtimes are detected, a benchmark form appears:
   - Runtime selector (dropdown: LM Studio, llama.cpp).
   - Model selector (populated from loaded models in the selected runtime).
   - Prompt type selector (default, coding, reasoning, creative).
   - Duration slider (5-300 seconds, default 30).
   - Number of runs (1-10, default 3).
6. User fills in the form and clicks "Start Benchmark."
7. The "Start Benchmark" button disables and shows "Running..."
8. A progress indicator appears: "Run 1 of 3..." then "Run 2 of 3..." (polling every 2 seconds).
9. On completion, results render below the form:
   - Aggregate: mean/min/max tokens/sec, avg prompt eval speed, avg time to first token.
   - Per-run breakdown table.
10. On error, a message displays: "Benchmark failed: [reason]. Ensure the model is loaded in [runtime]."

### Flow 6: Monitor Memory

**Goal**: View real-time memory usage of the system.

1. User clicks the "Memory" tab.
2. The tab immediately begins polling `GET /api/memory/current` every 2 seconds.
3. A gauge displays: used memory / total memory (e.g., "28.5 GB / 48.0 GB -- 59.4%").
4. A pressure badge shows the current level: "Normal" (green), "Warn" (yellow), or "Critical" (red).
5. Below the gauge, a simple line chart shows the last 60 readings (~2 minutes of history).
6. The chart auto-scrolls as new readings arrive, dropping the oldest reading when full.
7. When the user navigates away from the Memory tab, polling stops. When they return, polling resumes.
8. A "Full Report" section below the chart shows peak memory, baseline memory, delta, and monitoring duration.

---

## 3. States per View

### 3.1 Compatibility Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial** | Hardware selector dropdown (populated). No results table. Prompt text: "Select your hardware to check model compatibility." |
| **Loading** | Spinner in the results area. Filters and selector remain interactive but dimmed. Text: "Analyzing compatibility..." |
| **Success** | Summary line ("X of Y models fit...") + results table with color-coded rows. Filters active. |
| **Empty** | Table with zero rows (all filtered out). Message: "No models match the current filters. Try adjusting the family, tier, or quantization filters." |
| **Error** | Red banner above results area: "Failed to load compatibility data. [error detail]. Try selecting your hardware again." |
| **Detail Open** | Detail panel visible (memory breakdown, quant table, settings). Results table remains visible but de-emphasized in background. |

### 3.2 Models Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial/Success** | Full model table loaded (in-memory data, no loading state needed). Family filter pills and search field. |
| **Empty (filtered)** | Table with zero rows. Message: "No models match '[search term]' in the [family] family." |
| **Error** | Red banner: "Failed to load model data. Please refresh the page." (Unlikely since data is in-memory.) |
| **Detail Open** | Model detail card expanded below or beside the selected row. Card shows specs, template info, and "Check compatibility" action. |

### 3.3 Local Models Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial** | Large "Scan" button centered. Instructional text: "Click Scan to discover LLM models on your machine. Scans LM Studio, llama.cpp, and HuggingFace default directories." |
| **Loading** | Spinner with text: "Scanning local directories..." The Scan button is disabled. |
| **Success** | Summary bar (count, total size, source breakdown) + results table/cards. "Scan Again" and "Scan Custom Directory" buttons visible. |
| **Empty** | Message with icon: "No models found. Download models through LM Studio or HuggingFace to get started." Links to LM Studio docs and HuggingFace. |
| **Error** | Red banner: "Scan failed: [error detail]. Check that the default model directories are accessible." |
| **Custom Dir Error** | Inline error below path input: "Directory not found: [path]. Enter a valid directory path." |

### 3.4 HuggingFace Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial** | Search bar, format filter, sort selector. Prompt text: "Search HuggingFace for GGUF and MLX models." No results. |
| **Loading** | Spinner below search bar. Search field remains editable (new input cancels pending request). Text: "Searching HuggingFace..." |
| **Success** | Result cards with repo ID, author, download count, like count, format badges. Count indicator: "Showing 20 results for '[query]'." |
| **Empty** | Message: "No models found for '[query]'. Try a different search term or change the format filter." |
| **Error (network)** | Yellow banner: "Unable to reach HuggingFace. Check your internet connection and try again." Retry button. |
| **Error (rate limit)** | Yellow banner: "HuggingFace search is temporarily unavailable. Please wait a moment and try again." |
| **Detail Open** | Expanded card showing file listing and HuggingFace config details. |

### 3.5 Benchmark Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial (detecting)** | Spinner with text: "Detecting runtimes..." |
| **No runtimes** | Info panel: "No runtimes detected. Start LM Studio or llama.cpp to run benchmarks." Refresh button. Icon of disconnected server. |
| **Runtimes detected** | Status panel showing runtimes with green/red indicators. Benchmark form below with all fields. |
| **Running** | Progress indicator: "Running benchmark... Run 2 of 3." The form is disabled. A "Cancel" option is available if supported. |
| **Complete** | Aggregate results card (mean/min/max tok/s, prefill speed, TTFT) + per-run table. "Run Again" button re-enables the form. |
| **Error** | Red banner: "Benchmark failed: [reason]. Ensure the model is loaded in [runtime]." The form re-enables for retry. |

### 3.6 Memory Tab

| State | What the User Sees |
|-------|--------------------|
| **Initial** | Spinner: "Connecting to memory monitor..." |
| **Active** | Live gauge (used / total / percentage), pressure badge, history chart (auto-updating every 2s). Full report section below. |
| **Error** | Yellow banner: "Memory monitor unavailable. This feature requires macOS." Gauge shows "-- / -- GB." |
| **Stale** | If polling fails 3 times consecutively: "Memory data may be stale. Last updated X seconds ago." Retry button. |

---

## 4. Component Inventory

### 4.1 Reusable Components (Cross-Tab)

| Component | Description | Used In | New / Reusable |
|-----------|-------------|---------|----------------|
| **HardwareSelector** | Dropdown populated from `GET /api/hardware`. Stores selected hardware in app state for cross-tab use. | Compatibility, Model Detail | New (used in 2+ contexts) |
| **TierBadge** | Colored badge with text label and icon for compatibility tiers. Colors: green (FULL_GPU), blue (COMFORTABLE), yellow (TIGHT_FIT), orange (PARTIAL_OFFLOAD), red (WONT_FIT). Always includes text label + optional icon for accessibility. | Compatibility, Model Detail, Local Models | New (critical reusable) |
| **LoadingSpinner** | Centered spinner with optional text message. Renders as a CSS animation (no image dependency). | All tabs | New (global utility) |
| **ErrorBanner** | Dismissible banner at top of content area. Variants: error (red), warning (yellow), info (blue). Contains message text and optional retry button. | All tabs | New (global utility) |
| **EmptyState** | Centered message with optional icon, heading, body text, and action button. Used when a query returns zero results or a feature has not been activated. | All tabs | New (global utility) |
| **DataTable** | Sortable, filterable HTML table. Supports: click-to-sort column headers with directional arrows, row click handlers for detail drill-down, color-coded rows via CSS class, sticky header on scroll. | Compatibility, Models, Local Models, Benchmark results | New (most complex reusable) |
| **FilterControls** | Container for filter pills, dropdowns, checkboxes, and search inputs. Renders above a DataTable. | Compatibility, Models, HuggingFace | New (layout component) |
| **FormatBadge** | Small label badge for file formats: GGUF (blue), MLX (purple), SafeTensors (green). | Local Models, HuggingFace | New (simple reusable) |
| **SourceBadge** | Small label badge for model sources: LM Studio, llama.cpp, HuggingFace, MLX Community. | Local Models | New (simple reusable) |
| **CopyButton** | Button that copies text to clipboard and shows a brief "Copied!" confirmation. | Model Detail (llama.cpp command, LM Studio preset) | New (utility) |
| **TabNavigation** | Tab bar with `<button>` elements. Manages active state, keyboard navigation (arrow keys between tabs), and ARIA roles. | Global (header) | New (global structure) |
| **SummaryBar** | Horizontal bar showing aggregate stats (e.g., "12 models, 145 GB, by source"). | Compatibility, Local Models | New (layout component) |

### 4.2 Tab-Specific Components

#### Compatibility Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **CompatibilityForm** | Composite: HardwareSelector + context length control + "Only fits" toggle + "Show all" toggle. | New (tab-specific) |
| **ContextLengthControl** | Slider or number input with preset buttons (2K, 4K, 8K, 16K, 32K, 64K, 128K). | New (tab-specific) |
| **ModelDetailPanel** | Expandable panel: memory breakdown bar, quant comparison table, recommended settings, CLI commands. | New (tab-specific) |
| **MemoryBreakdownBar** | Stacked horizontal bar chart showing model size, KV cache, overhead, and headroom as proportional segments. Pure CSS (no canvas). | New (tab-specific) |
| **TierSummary** | Compact summary: "38 Full GPU, 22 Comfortable, 8 Tight Fit, 12 Partial, 62 Won't Fit." Each count paired with its TierBadge. | New (tab-specific) |

#### Models Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **FamilyFilterPills** | Row of toggleable pill buttons, one per model family. Multiple can be active. | New (tab-specific) |
| **ModelDetailCard** | Expandable card with model specs, template info, "Check compatibility" action. | New (tab-specific) |

#### Local Models Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **ScanButton** | Primary action button with scan icon. Disables during scan, re-enables after. | New (tab-specific) |
| **CustomDirInput** | Text input for custom directory path + "Scan" submit button. Inline validation. | New (tab-specific) |

#### HuggingFace Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **SearchInput** | Text input with debounce (300ms). Fires API call after user stops typing. Shows inline spinner during search. | New (tab-specific) |
| **ResultCard** | Card layout for a single HuggingFace search result. Shows repo ID, author, stats, format badges, and "Check compatibility" action. | New (tab-specific) |
| **FileListPanel** | Expandable section showing files in a HuggingFace repo (name + size). | New (tab-specific) |

#### Benchmark Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **RuntimeStatusPanel** | Status display for detected runtimes. Green/red dot indicators, model lists. | New (tab-specific) |
| **BenchmarkForm** | Composite form: runtime selector, model selector, prompt type, duration slider, num runs. | New (tab-specific) |
| **ProgressIndicator** | Animated progress bar or step counter: "Run 2 of 3." Polling-driven updates. | New (tab-specific) |
| **BenchmarkResults** | Aggregate summary card + per-run breakdown table. | New (tab-specific) |

#### Memory Tab
| Component | Description | New / Reusable |
|-----------|-------------|----------------|
| **MemoryGauge** | Circular or bar gauge showing used / total memory + percentage. Animated on updates. | New (tab-specific) |
| **PressureBadge** | Badge showing memory pressure level (normal/warn/critical) with color. | New (tab-specific, could be reusable) |
| **MemoryChart** | Simple line chart (HTML canvas) showing last 60 memory readings. Auto-scrolls as new data arrives. | New (tab-specific) |
| **MemoryReport** | Summary card: peak, baseline, delta, duration, sample count. | New (tab-specific) |

### 4.3 Component Notes

- All components are vanilla JS functions that create and return DOM elements. No framework, no web components.
- Pattern: `function renderTierBadge(tier) { ... return element; }`
- Components are defined in `app.js`. If `app.js` exceeds ~1500 lines, split into per-tab modules loaded via `<script>` tags (no ES modules/import maps required for this scope).
- CSS for all components lives in `style.css` using BEM-like naming: `.tier-badge`, `.tier-badge--full-gpu`, `.data-table__header`, etc.

---

## 5. Accessibility

### 5.1 Keyboard Navigation

| Element | Keyboard Behavior |
|---------|-------------------|
| **Tab navigation bar** | Arrow Left/Right moves focus between tabs. Enter/Space activates the focused tab. Tab key moves focus into the active tab's content. |
| **Hardware selector dropdown** | Standard `<select>` keyboard behavior (arrow keys, type-ahead). |
| **Data tables** | Tab key moves through sortable column headers. Enter on a header sorts by that column. Tab continues into table rows. Enter on a row opens the detail view. |
| **Filter controls** | Tab key cycles through filter elements. Checkboxes toggle with Space. Dropdowns use arrow keys. |
| **Detail panel** | Focus is trapped within the panel when open. Escape closes the panel and returns focus to the originating row. Tab cycles through interactive elements within the panel. |
| **Modal/overlay** | If any overlay is used, focus is trapped within it. Escape dismisses. Focus returns to the trigger element on close. |
| **Buttons** | All buttons are `<button>` elements (not `<div>` or `<span>`), ensuring native keyboard support. |

### 5.2 ARIA Attributes

| Pattern | ARIA Implementation |
|---------|---------------------|
| **Tab bar** | `role="tablist"` on container. Each tab: `role="tab"`, `aria-selected="true/false"`, `aria-controls="tab-panel-id"`. Each panel: `role="tabpanel"`, `aria-labelledby="tab-id"`. |
| **Sortable table headers** | `aria-sort="ascending"`, `aria-sort="descending"`, or `aria-sort="none"`. `role="columnheader"` on `<th>`. |
| **Data tables** | `<caption>` describing the table (e.g., "Model compatibility results for M4 Pro 48GB"). `<th scope="col">` for column headers. `<th scope="row">` for model name column (first column). |
| **Loading states** | `aria-live="polite"` on the loading region. `aria-busy="true"` while loading. Screen reader announcement: "Loading compatibility data." |
| **Error banners** | `role="alert"` for immediate announcement. `aria-live="assertive"` for critical errors. |
| **Filter controls** | `<label>` elements associated with every `<input>`, `<select>`, and `<fieldset>`. `aria-label` on icon-only buttons. |
| **Detail panel** | `role="dialog"` with `aria-labelledby` pointing to the panel heading. `aria-modal="true"` if focus-trapped. |
| **Tier badges** | `aria-label` that reads the full tier name (e.g., `aria-label="Full GPU - runs entirely in GPU memory"`). |

### 5.3 Color-Independent Tier Encoding

Every compatibility tier is communicated through **three independent channels** so that no information is lost for users with color vision deficiency:

| Tier | Color | Text Label | Icon/Symbol |
|------|-------|------------|-------------|
| FULL_GPU | `#22c55e` (green) | "Full GPU" | Solid circle or check mark |
| COMFORTABLE | `#3b82f6` (blue) | "Comfortable" | Three-quarter circle or thumbs up |
| TIGHT_FIT | `#eab308` (yellow) | "Tight Fit" | Half circle or warning triangle |
| PARTIAL_OFFLOAD | `#f97316` (orange) | "Partial Offload" | Quarter circle or arrow down |
| WONT_FIT | `#6b7280` (gray/red) | "Won't Fit" | Empty circle or X mark |

The text label is always visible (not hidden behind a tooltip). The icon provides a shape-based cue that works independently of color.

### 5.4 Screen Reader Support

- All tables include `<caption>` elements describing the data and current filter context.
- Form inputs have associated `<label>` elements (no placeholder-only labels).
- Dynamic content updates use `aria-live` regions to announce changes without requiring focus shift.
- The memory gauge provides a text alternative: "Memory usage: 28.5 GB of 48.0 GB (59.4%), pressure: normal."
- Loading spinners include `aria-label="Loading"` and are announced via `aria-live="polite"`.
- Sort state changes are announced: "Table sorted by Total Memory, ascending."

### 5.5 Focus Management

| Scenario | Focus Behavior |
|----------|----------------|
| **Tab switch** | Focus moves to the first interactive element inside the newly activated tab panel. |
| **Detail panel opens** | Focus moves to the panel heading or close button. Original table row is stored for focus return. |
| **Detail panel closes** | Focus returns to the originating table row. |
| **Error banner appears** | Focus is not moved (error is announced via `aria-live`). User can Tab to the dismiss/retry button. |
| **Filter change** | Focus stays on the filter control. The table updates in the background (announced via `aria-live`). |
| **Scan complete** | Focus stays on the scan button. Results are announced: "Scan complete. 12 models found." |
| **Benchmark complete** | Focus moves to the results section. Announcement: "Benchmark complete. Average 12.1 tokens per second." |

---

## 6. Responsive Behavior

### 6.1 Viewport Requirements

- **Minimum supported viewport**: 768px wide (standard laptop).
- **Primary design target**: 1024px-1440px (13"-16" MacBook display).
- **Maximum stretch**: 1920px+ (external monitor). Content max-width capped at 1400px, centered.
- **Below 768px**: Not supported. A dismissible banner reads: "Loca-LLAMA is designed for laptop and desktop screens. For the best experience, use a screen at least 768px wide."

### 6.2 Breakpoints

| Breakpoint | Range | Layout Behavior |
|------------|-------|-----------------|
| **Small laptop** | 768px - 1023px | Single-column layout. Tables scroll horizontally inside a container. Detail panels render below content (not as side panels). Filter controls stack vertically. |
| **Standard laptop** | 1024px - 1439px | Primary design. Two-column where appropriate (e.g., results table + detail panel side by side). Filters render inline. |
| **External monitor** | 1440px+ | Same as standard, with more breathing room. Content max-width 1400px, centered with padding. Data tables can show more columns without truncation. |

### 6.3 Table Behavior

Tables are the primary data display. Their behavior adapts to viewport width:

| Viewport | Behavior |
|----------|----------|
| **1024px+** | Full table with all columns visible. Sortable headers. Row hover highlight. |
| **768px - 1023px** | Horizontal scroll enabled on the table container. A subtle shadow on the right edge indicates more content. Priority columns (Model Name, Tier, Total Memory) remain visible; secondary columns (Family, Headroom, Speed) scroll into view. Alternatively, the table collapses into a card layout (one card per model) if the table is too wide. |

### 6.4 Component-Specific Responsive Rules

| Component | 1024px+ | 768px-1023px |
|-----------|---------|--------------|
| **Tab navigation** | Horizontal tab bar with text labels. | Horizontal tab bar with abbreviated labels or icons + text. If tabs overflow, allow horizontal scroll on the tab bar. |
| **Compatibility form** | Inline: hardware selector, context control, and toggles in a single row. | Stacked: each control on its own row. |
| **Model detail panel** | Side panel (right side, ~40% width) alongside results table. | Full-width panel below the results table. |
| **HuggingFace result cards** | 2-column grid. | Single-column stack. |
| **Memory gauge + chart** | Gauge and chart side by side. | Gauge above, chart below (stacked). |
| **Benchmark form** | Two-column form layout. | Single-column form layout. |
| **Filter controls** | Inline (pills + dropdowns in a row). | Wrapped or collapsible ("Filters" toggle that expands a panel). |

### 6.5 Typography and Spacing

- Base font size: 16px (1rem). No reduction below 14px at any viewport.
- Line height: 1.5 for body text, 1.2 for headings.
- Table cell padding: 12px at 1024px+, 8px at 768px-1023px.
- Touch targets (buttons, clickable rows): minimum 44px height.
- CSS custom properties for spacing scale: `--space-xs: 4px`, `--space-sm: 8px`, `--space-md: 16px`, `--space-lg: 24px`, `--space-xl: 32px`.

### 6.6 Dark Mode

- Dark mode supported via `@media (prefers-color-scheme: dark)`.
- All colors defined as CSS custom properties in `:root` and overridden in the dark media query.
- Tier colors adjusted for dark backgrounds (slightly desaturated, higher luminance) to maintain WCAG AA contrast ratio (4.5:1 for text, 3:1 for large text/UI components).
- No manual dark mode toggle (follows system preference). Can be added later if requested.

---

*Generated with Claude Code on 2026-03-02. Based on pm.md (Product Requirements) and eng.md (Technical Design).*
