# Implementation Plan: Loca-LLAMA v2 Webapp

## Overview

Convert the Loca-LLAMA CLI/TUI codebase into a full-featured local webapp by adding a thin FastAPI API layer over the existing 13 core modules, a vanilla HTML/CSS/JS frontend, and a comprehensive pytest test suite. The plan is organized into 7 phases matching the engineering milestones, with 42 tasks totaling an estimated 11 implementation days.

## Reference Documents

- **Feature Request**: `rpi/loca-llama-v2/REQUEST.md`
- **Research**: `rpi/loca-llama-v2/research/RESEARCH.md`
- **Product Requirements**: `rpi/loca-llama-v2/plan/pm.md`
- **Technical Design**: `rpi/loca-llama-v2/plan/eng.md`

## Verification Commands (All Phases)

```bash
pip install -e ".[web,dev]"                      # Install with all deps
python -m pytest tests/ -v                       # Run all tests
python -m pytest tests/api/ -v                   # Run API tests only
python -m pytest tests/ --cov=loca_llama -v      # With coverage
uvicorn loca_llama.api.app:app --reload          # Dev server
```

---

## Phase 1: Foundation

**Goal**: FastAPI app skeleton builds, starts, and serves a static HTML page. All infrastructure is in place for subsequent phases to add route implementations incrementally.

**Acceptance**: `pip install -e ".[web,dev]"` succeeds. `uvicorn loca_llama.api.app:app` starts on port 8000 and serves `index.html`. `/docs` shows Swagger UI with stubbed route groups.

- [x] **Task 1.1**: Update `pyproject.toml` with optional dependencies and entry point
  - Files: `pyproject.toml`
  - Details: Add `[project.optional-dependencies]` sections for `web` (fastapi>=0.115.0, uvicorn[standard]>=0.34.0) and `dev` (pytest>=8.0, pytest-asyncio>=0.24, httpx>=0.28, coverage>=7.0). Add `loca-llama-web` entry point under `[project.scripts]`. Add `[tool.pytest.ini_options]` with `asyncio_mode = "auto"`, `testpaths = ["tests"]`, `addopts = "-v --tb=short -x"`.
  - Complexity: **Low**
  - Dependencies: None

- [x] **Task 1.2**: Create API package structure and app factory
  - Files: `loca_llama/api/__init__.py`, `loca_llama/api/app.py`
  - Details: `__init__.py` exposes `create_app`. `app.py` implements `create_app()` factory with lifespan context manager (placeholder for MemoryMonitor), mounts `StaticFiles` at `/` from `static/` directory, registers all route groups under `/api`. Include `serve()` function for the `loca-llama-web` CLI entry point. Create module-level `app = create_app()` for `uvicorn` discovery.
  - Complexity: **Medium**
  - Dependencies: Task 1.1

- [x] **Task 1.3**: Create AppState and dependency injection
  - Files: `loca_llama/api/state.py`, `loca_llama/api/dependencies.py`
  - Details: `state.py` defines `BenchmarkJob` dataclass and `AppState` class (memory_monitor placeholder, benchmark_jobs dict, cleanup_old_jobs method). `dependencies.py` defines `get_state()` dependency that returns the singleton AppState.
  - Complexity: **Low**
  - Dependencies: Task 1.2

- [x] **Task 1.4**: Create all Pydantic request/response schemas
  - Files: `loca_llama/api/schemas.py`
  - Details: Define all Pydantic models referenced in eng.md Section 2: `HardwareResponse`, `HardwareListResponse`, `ModelResponse`, `ModelListResponse`, `QuantResponse`, `QuantListResponse`, `AnalyzeRequest`, `AnalyzeResponse`, `AnalyzeAllRequest`, `AnalyzeAllResponse`, `MaxContextRequest`, `MaxContextResponse`, `TemplateResponse`, `TemplateListResponse`, `LMStudioPresetRequest`, `LMStudioPresetResponse`, `LlamaCppCommandRequest`, `LlamaCppCommandResponse`, `ScannerResponse`, `HubSearchResponse`, `HubFilesResponse`, `HubConfigResponse`, `RuntimeStatusResponse`, `BenchmarkStartRequest`, `BenchmarkStartResponse`, `BenchmarkStatusResponse`, `BenchmarkPromptsResponse`, `MemoryCurrentResponse`, `MemoryReportResponse`. Include `@classmethod` factory methods (e.g., `from_estimate`, `from_spec`) where appropriate for converting core dataclasses to Pydantic models.
  - Complexity: **High**
  - Dependencies: None (references core module types but does not import at module level)

- [x] **Task 1.5**: Create route stubs with APIRouter
  - Files: `loca_llama/api/routes/__init__.py`, `loca_llama/api/routes/hardware.py`, `loca_llama/api/routes/models.py`, `loca_llama/api/routes/quantization.py`, `loca_llama/api/routes/analysis.py`, `loca_llama/api/routes/templates.py`, `loca_llama/api/routes/scanner.py`, `loca_llama/api/routes/hub.py`, `loca_llama/api/routes/benchmark.py`, `loca_llama/api/routes/memory.py`, `loca_llama/api/routes/runtime.py`
  - Details: Each route file creates an `APIRouter` with appropriate prefix and tags. Stub handlers return `501 Not Implemented`. `routes/__init__.py` aggregates all routers into a single list for `app.py` to include. Follow the route handler template from eng.md Section 7.
  - Complexity: **Medium**
  - Dependencies: Task 1.4

- [x] **Task 1.6**: Create minimal static frontend shell
  - Files: `static/index.html`, `static/style.css`, `static/app.js`
  - Details: `index.html` with semantic HTML structure (header, nav with 6 tab buttons, main with 6 tab-content sections, footer). `style.css` with CSS custom properties for theming, tab layout, basic table styles. `app.js` with API wrapper (`api.get()`, `api.post()`), tab switching logic, and placeholder render functions for each tab. Follow the HTML/JS patterns from eng.md Section 5.
  - Complexity: **Medium**
  - Dependencies: None

- [x] **Task 1.7**: Create test infrastructure
  - Files: `tests/__init__.py`, `tests/conftest.py`, `tests/api/__init__.py`, `tests/api/conftest.py`
  - Details: `tests/conftest.py` with shared fixtures: `app` (create_app), `client` (async httpx.AsyncClient with ASGITransport), `m4_pro_48` (MacSpec fixture), `qwen_32b` (LLMModel fixture), `q4km` (QuantFormat fixture). `tests/api/conftest.py` with API-specific fixtures if needed. Follow fixture patterns from eng.md Section 6.
  - Complexity: **Low**
  - Dependencies: Task 1.2

**Phase 1 Verification**:
```bash
pip install -e ".[web,dev]"
uvicorn loca_llama.api.app:app --host 127.0.0.1 --port 8000  # Confirm starts, serves HTML
curl http://127.0.0.1:8000/                                    # Returns index.html
curl http://127.0.0.1:8000/docs                                # Returns Swagger UI
curl http://127.0.0.1:8000/api/hardware                        # Returns 501 (stub)
```

---

## Phase 2: Data Endpoints

**Goal**: All read-only, non-blocking endpoints are implemented and tested. The frontend can populate hardware dropdowns, model lists, quantization selectors, and template displays.

**Acceptance**: `python -m pytest tests/api/test_hardware_routes.py tests/api/test_model_routes.py tests/api/test_quant_routes.py tests/api/test_template_routes.py -v` passes. Frontend dropdowns populate from live API.

- [x] **Task 2.1**: Implement hardware endpoints
  - Files: `loca_llama/api/routes/hardware.py`
  - Details: `GET /api/hardware` returns all specs from `hardware.APPLE_SILICON_SPECS` converted to `HardwareListResponse`. `GET /api/hardware/{name}` returns single spec or 404. Both call directly into the hardware module (no I/O, in-memory data).
  - Complexity: **Low**
  - Dependencies: Task 1.4, Task 1.5

- [x] **Task 2.2**: Implement model listing endpoint
  - Files: `loca_llama/api/routes/models.py`
  - Details: `GET /api/models` returns all models from `models.MODELS` with optional `family` query param filter. Returns `ModelListResponse` with count and families list.
  - Complexity: **Low**
  - Dependencies: Task 1.4, Task 1.5

- [x] **Task 2.3**: Implement quantization endpoint
  - Files: `loca_llama/api/routes/quantization.py`
  - Details: `GET /api/quantizations` returns all formats from `quantization.QUANT_FORMATS` plus `RECOMMENDED_FORMATS`. Returns `QuantListResponse`.
  - Complexity: **Low**
  - Dependencies: Task 1.4, Task 1.5

- [x] **Task 2.4**: Implement template endpoints
  - Files: `loca_llama/api/routes/templates.py`
  - Details: `GET /api/templates` returns all templates. `GET /api/templates/match/{model_name}` finds best match via `templates.get_template()` or 404. `POST /api/templates/lm-studio-preset` generates preset via `templates.get_lm_studio_preset()`. `POST /api/templates/llama-cpp-command` generates CLI command via `templates.get_llama_cpp_command()`.
  - Complexity: **Medium**
  - Dependencies: Task 1.4, Task 1.5

- [x] **Task 2.5**: Write tests for data endpoints
  - Files: `tests/api/test_hardware_routes.py`, `tests/api/test_model_routes.py`, `tests/api/test_quant_routes.py`, `tests/api/test_template_routes.py`
  - Details: Test each endpoint for 200 response with valid input, correct response shape, 404 for missing resources, 422 for invalid params. Verify hardware count matches `len(APPLE_SILICON_SPECS)`, model count matches `len(MODELS)`, quant count matches `len(QUANT_FORMATS)`. Test template match for known model name and 404 for unknown.
  - Complexity: **Medium**
  - Dependencies: Task 1.7, Tasks 2.1-2.4

- [x] **Task 2.6**: Wire frontend dropdowns to data endpoints
  - Files: `static/app.js`, `static/index.html`
  - Details: On page load, fetch `/api/hardware` and populate hardware selector dropdown. Fetch `/api/quantizations` and render checkboxes with recommended formats pre-selected. Fetch `/api/models` and render model table in Models tab with family filter pills. Fetch `/api/templates` and render template list in Compatibility tab detail view.
  - Complexity: **Medium**
  - Dependencies: Task 1.6, Tasks 2.1-2.4

**Phase 2 Verification**:
```bash
python -m pytest tests/api/test_hardware_routes.py tests/api/test_model_routes.py tests/api/test_quant_routes.py tests/api/test_template_routes.py -v
uvicorn loca_llama.api.app:app --reload  # Open browser, verify dropdowns populate
```

---

## Phase 3: Analysis Endpoints

**Goal**: The core compatibility analysis engine is exposed via API. Users can analyze a single model, all models, or find max context length. The Compatibility tab renders results in a sortable, filterable table.

**Acceptance**: `python -m pytest tests/test_analyzer.py tests/api/test_analysis_routes.py -v` passes. Compatibility tab shows results with tier badges, and analysis output matches CLI for the same inputs.

- [x] **Task 3.1**: Implement single analysis endpoint
  - Files: `loca_llama/api/routes/analysis.py`
  - Details: `POST /api/analyze` accepts `AnalyzeRequest`, looks up hardware/model/quant by name, calls `analyzer.analyze_model()`, returns `AnalyzeResponse`. Returns 400 if hardware, model, or quant name not found.
  - Complexity: **Medium**
  - Dependencies: Task 1.4, Task 1.5

- [x] **Task 3.2**: Implement bulk analysis and max-context endpoints
  - Files: `loca_llama/api/routes/analysis.py`
  - Details: `POST /api/analyze/all` accepts `AnalyzeAllRequest`, iterates models (filtered by family if provided), calls `analyzer.analyze_model()` for each model+quant combo, filters by `only_fits` and `include_partial`, returns `AnalyzeAllResponse` with summary counts per tier. `POST /api/analyze/max-context` accepts `MaxContextRequest`, calls `analyzer.max_context_for_model()`, returns `MaxContextResponse`.
  - Complexity: **Medium**
  - Dependencies: Task 3.1

- [x] **Task 3.3**: Write unit tests for analyzer functions
  - Files: `tests/test_analyzer.py`
  - Details: Test `estimate_model_size_gb()` for known model sizes. Test `estimate_kv_cache_gb()` for specific architectures. Test `compute_tier()` at boundary conditions (75%, 90%, 100%, 150% utilization). Test `analyze_model()` for M4 Pro 48GB + Qwen 32B + Q4_K_M and verify against known values. Test `max_context_for_model()` returns reasonable values.
  - Complexity: **Medium**
  - Dependencies: Task 1.7

- [x] **Task 3.4**: Write integration tests for analysis routes
  - Files: `tests/api/test_analysis_routes.py`
  - Details: Test `POST /api/analyze` with valid input returns 200 with correct fields. Test with invalid hardware name returns 400. Test `POST /api/analyze/all` returns results and summary. Test `only_fits=true` filters out WONT_FIT results. Test `POST /api/analyze/max-context` returns positive context length. Verify API output matches direct `analyzer.analyze_model()` call for same inputs.
  - Complexity: **Medium**
  - Dependencies: Task 1.7, Tasks 3.1-3.2

- [x] **Task 3.5**: Build Compatibility tab in frontend
  - Files: `static/app.js`, `static/index.html`, `static/style.css`
  - Details: Hardware selector triggers `POST /api/analyze/all`. Results table with columns: Model Name, Family, Quant, Total Memory (GB), Headroom (GB), Tier, Est. Speed (tok/s). Rows color-coded by tier (green=FULL_GPU, blue=COMFORTABLE, yellow=TIGHT_FIT, orange=PARTIAL_OFFLOAD, red=WONT_FIT). Client-side column sorting. Family filter dropdown. Tier filter checkboxes. "Only show models that fit" toggle (default on, hides WONT_FIT). Click row to expand detail panel showing template, max context, memory breakdown. Context length slider.
  - Complexity: **High**
  - Dependencies: Task 2.6, Tasks 3.1-3.2

**Phase 3 Verification**:
```bash
python -m pytest tests/test_analyzer.py tests/api/test_analysis_routes.py -v
# Manual: Open browser, select "M4 Pro 48GB", verify results match CLI output:
# python -m loca_llama --hardware "M4 Pro 48GB" --quant Q4_K_M
```

---

## Phase 4: Discovery Endpoints

**Goal**: Local model scanning and HuggingFace search are exposed via async API endpoints. The Local Models and HuggingFace tabs are functional with loading states and error handling.

**Acceptance**: `python -m pytest tests/api/test_scanner_routes.py tests/api/test_hub_routes.py -v` passes. Local scan finds models on disk. HuggingFace search returns results.

- [ ] **Task 4.1**: Implement local scanner endpoint
  - Files: `loca_llama/api/routes/scanner.py`
  - Details: `GET /api/scanner/local` with optional `custom_dir` query param. Uses `asyncio.to_thread(scanner.scan_all)` for non-blocking filesystem I/O. If `custom_dir` provided, also scans that directory via `scanner.scan_custom_dir()`. Returns `ScannerResponse` with models, count, total_size_gb, and source breakdown.
  - Complexity: **Medium**
  - Dependencies: Task 1.3, Task 1.4

- [ ] **Task 4.2**: Implement HuggingFace hub endpoints
  - Files: `loca_llama/api/routes/hub.py`
  - Details: `GET /api/hub/search` with `query` (required), `limit`, `sort`, `format` params. Uses `asyncio.to_thread(hub.search_huggingface)` or the format-specific search functions. `GET /api/hub/files/{repo_id:path}` gets file listing via `asyncio.to_thread(hub.get_model_files)`. `GET /api/hub/config/{repo_id:path}` fetches HF config via `asyncio.to_thread(hf_templates.fetch_hf_model_config)`. Network errors return 502.
  - Complexity: **Medium**
  - Dependencies: Task 1.3, Task 1.4

- [ ] **Task 4.3**: Write tests for discovery endpoints (mocked)
  - Files: `tests/api/test_scanner_routes.py`, `tests/api/test_hub_routes.py`
  - Details: Mock `scanner.scan_all()` to return synthetic model list. Verify response shape, count, total_size_gb calculation. Mock `hub.search_huggingface()` to return synthetic results. Test limit and format filters. Mock `hf_templates.fetch_hf_model_config()` for config endpoint. Test 502 response when mock raises network error.
  - Complexity: **Medium**
  - Dependencies: Task 1.7, Tasks 4.1-4.2

- [ ] **Task 4.4**: Build Local Models tab
  - Files: `static/app.js`, `static/index.html`, `static/style.css`
  - Details: "Scan Local Models" button triggers `GET /api/scanner/local`. Loading spinner during scan. Results table: name, size (GB), format badge, source badge, quant, family. Summary bar: total count, total size, counts by source. "Scan Custom Directory" input + button. Empty state message when no models found. Error state for scan failures.
  - Complexity: **Medium**
  - Dependencies: Task 1.6, Task 4.1

- [ ] **Task 4.5**: Build HuggingFace tab
  - Files: `static/app.js`, `static/index.html`, `static/style.css`
  - Details: Search input with 300ms debounce. Format filter (All / GGUF / MLX). Sort selector (downloads / likes / lastModified). Results cards: repo_id, author, downloads, likes, tags, format badges. Click result to show file listing from `/api/hub/files/{repo_id}`. "Check compatibility" action that navigates to Compatibility tab with the model. Network error state. Loading indicator during search.
  - Complexity: **Medium**
  - Dependencies: Task 1.6, Task 4.2

**Phase 4 Verification**:
```bash
python -m pytest tests/api/test_scanner_routes.py tests/api/test_hub_routes.py -v
# Manual: Open Local Models tab, click Scan, verify models found
# Manual: Open HuggingFace tab, search "qwen gguf", verify results
```

---

## Phase 5: Benchmark & Runtime

**Goal**: Runtime detection, benchmark execution with background tasks, and polling-based progress updates are functional. The Benchmark tab provides a complete workflow from runtime detection to results display.

**Acceptance**: `python -m pytest tests/api/test_benchmark_routes.py tests/api/test_runtime_routes.py -v` passes. Benchmark job lifecycle (start -> poll progress -> complete/error) works end-to-end.

- [ ] **Task 5.1**: Implement runtime status endpoint
  - Files: `loca_llama/api/routes/runtime.py`
  - Details: `GET /api/runtime/status` uses `asyncio.to_thread(benchmark.detect_all_runtimes)` to probe LM Studio (port 1234) and llama.cpp (ports 8080, 8081). Returns `RuntimeStatusResponse` with list of detected runtimes, each including name, URL, loaded models, and version.
  - Complexity: **Low**
  - Dependencies: Task 1.3, Task 1.4

- [ ] **Task 5.2**: Implement benchmark prompts endpoint
  - Files: `loca_llama/api/routes/benchmark.py`
  - Details: `GET /api/benchmark/prompts` returns available prompt types from `benchmark.BENCH_PROMPTS`. Returns `BenchmarkPromptsResponse`.
  - Complexity: **Low**
  - Dependencies: Task 1.4

- [ ] **Task 5.3**: Implement benchmark start and polling endpoints
  - Files: `loca_llama/api/routes/benchmark.py`, `loca_llama/api/state.py`
  - Details: `POST /api/benchmark/start` validates runtime exists, creates a `BenchmarkJob` in AppState, launches `asyncio.create_task()` wrapping `asyncio.to_thread(run_benchmark_suite)` with progress callback. Returns job_id immediately. `GET /api/benchmark/{job_id}` returns current status (running with progress, complete with results + aggregate, error with message, or 404 for unknown job_id). Implement `_run_benchmark_background()` async helper as specified in eng.md Section 4.
  - Complexity: **High**
  - Dependencies: Task 1.3, Task 1.4, Task 5.1

- [ ] **Task 5.4**: Write tests for benchmark and runtime endpoints (mocked)
  - Files: `tests/api/test_benchmark_routes.py`, `tests/api/test_runtime_routes.py`
  - Details: Mock `benchmark.detect_all_runtimes()` to return synthetic runtime info. Test runtime status returns correct shape. Mock `benchmark.run_benchmark_suite()` to return synthetic results after short delay. Test benchmark start returns job_id. Test polling returns "running" then "complete". Test 404 for nonexistent job_id. Test 400 when runtime not found. Test prompts endpoint returns dict.
  - Complexity: **High**
  - Dependencies: Task 1.7, Tasks 5.1-5.3

- [ ] **Task 5.5**: Build Benchmark tab with polling UI
  - Files: `static/app.js`, `static/index.html`, `static/style.css`
  - Details: On tab open, auto-detect runtimes via `GET /api/runtime/status` with status indicators. "Start Benchmark" form: runtime selector, model selector (populated from detected runtimes), prompt type selector, num_runs slider, max_tokens input. Submit triggers `POST /api/benchmark/start`. Progress bar polls `GET /api/benchmark/{job_id}` every 2 seconds (clearInterval on complete/error). Results table: run#, tok/s, prefill tok/s, TTFT, total time. Aggregate summary section. Error state with runtime-specific message. Empty state when no runtimes detected.
  - Complexity: **High**
  - Dependencies: Task 1.6, Tasks 5.1-5.3

**Phase 5 Verification**:
```bash
python -m pytest tests/api/test_benchmark_routes.py tests/api/test_runtime_routes.py -v
# Manual (requires LM Studio or llama.cpp running):
# 1. Open Benchmark tab, verify runtimes detected
# 2. Start benchmark, verify progress updates
# 3. Verify results display on completion
```

---

## Phase 6: Memory Monitor

**Goal**: The MemoryMonitor is wired into FastAPI's lifespan, memory endpoints return live readings, and the Memory tab shows a live-updating gauge with history.

**Acceptance**: `python -m pytest tests/api/test_memory_routes.py -v` passes. Memory tab shows live readings that update every 2 seconds.

- [ ] **Task 6.1**: Wire MemoryMonitor into lifespan
  - Files: `loca_llama/api/app.py`, `loca_llama/api/state.py`
  - Details: Update the lifespan context manager in `app.py` to call `app_state.memory_monitor.start()` on startup and `app_state.memory_monitor.stop()` on shutdown. Add structured logging for start/stop events. Ensure `AppState.__init__` creates the monitor with `interval=1.0`.
  - Complexity: **Low**
  - Dependencies: Task 1.2, Task 1.3

- [ ] **Task 6.2**: Implement memory endpoints
  - Files: `loca_llama/api/routes/memory.py`
  - Details: `GET /api/memory/current` calls `state.memory_monitor.get_current()` and returns `MemoryCurrentResponse` (used_gb, free_gb, total_gb, usage_pct, pressure). `GET /api/memory/report` calls `state.memory_monitor._build_report()` and returns `MemoryReportResponse` (peak, baseline, delta, duration, sample_count). Both use dependency injection to access AppState.
  - Complexity: **Low**
  - Dependencies: Task 1.4, Task 6.1

- [ ] **Task 6.3**: Write tests for memory endpoints (mocked)
  - Files: `tests/api/test_memory_routes.py`
  - Details: Mock `memory_monitor.get_memory_sample()` to return deterministic values (e.g., 28.5 GB used, 19.5 GB free, 48.0 GB total). Test `/api/memory/current` returns correct shape with plausible values. Test `/api/memory/report` returns report with peak >= baseline. Test memory values are within expected ranges (0 < used < total).
  - Complexity: **Medium**
  - Dependencies: Task 1.7, Task 6.2

- [ ] **Task 6.4**: Build Memory tab with live polling and gauge
  - Files: `static/app.js`, `static/index.html`, `static/style.css`
  - Details: Live memory gauge that polls `GET /api/memory/current` every 2 seconds when the Memory tab is active (stop polling when switching away). Visual bar showing used/total with pressure badge (normal=green, warn=yellow, critical=red). Memory history chart: simple canvas-based line chart showing last 60 readings (~2 minutes). Display: total memory, used, free, percentage, pressure level. Report section showing peak, baseline, delta from `/api/memory/report`.
  - Complexity: **Medium**
  - Dependencies: Task 1.6, Task 6.2

**Phase 6 Verification**:
```bash
python -m pytest tests/api/test_memory_routes.py -v
# Manual: Open Memory tab, verify gauge updates every 2 seconds
# Manual: Verify pressure badge reflects actual system memory state
```

---

## Phase 7: Polish

**Goal**: Production-quality error handling, loading states, responsive layout, dark mode, and full test coverage. The webapp is ready for use.

**Acceptance**: `python -m pytest tests/ --cov=loca_llama -v` passes with >= 75% overall coverage and >= 85% API route coverage. All error states render correctly. Dark mode works. Layout is usable at 768px width.

- [ ] **Task 7.1**: Error handling audit for all routes
  - Files: All files in `loca_llama/api/routes/`
  - Details: Verify every route handler follows the try/except pattern from eng.md Section 7: catch KeyError -> 404, ValueError -> 400, HTTPException -> re-raise, Exception -> 500 with generic message. Add structured logging (Python `logging` module) for all caught exceptions with context (endpoint, input params, error type). Verify no internal paths or stack traces leak to clients.
  - Complexity: **Medium**
  - Dependencies: Tasks 2.1-6.2

- [ ] **Task 7.2**: Frontend error states and loading indicators
  - Files: `static/app.js`, `static/style.css`
  - Details: Every API call shows a loading spinner during fetch and displays a user-friendly error message on failure. Specific error messages: "Unable to reach HuggingFace" for hub endpoints, "No runtimes detected" for benchmark, "Memory monitoring requires macOS" for memory on non-Mac. Network failures show generic "Unable to connect to server" message. Empty states for all tabs (no models found, no results, no templates).
  - Complexity: **Medium**
  - Dependencies: Tasks 2.6, 3.5, 4.4, 4.5, 5.5, 6.4

- [ ] **Task 7.3**: Responsive CSS for 768px+ screens
  - Files: `static/style.css`
  - Details: Media queries for tablet (768px) and desktop (1024px+). Results table scrolls horizontally on narrow screens. Tab navigation wraps on smaller screens. Form controls stack vertically on narrow screens. Memory gauge adapts to available width. Minimum font size 14px for readability.
  - Complexity: **Medium**
  - Dependencies: Task 1.6

- [ ] **Task 7.4**: Dark mode
  - Files: `static/style.css`
  - Details: Use `@media (prefers-color-scheme: dark)` to override CSS custom properties. Dark background, light text, adjusted tier badge colors for dark backgrounds. Ensure sufficient contrast ratios (WCAG AA). Test in Safari, Chrome, Firefox dark modes.
  - Complexity: **Low**
  - Dependencies: Task 7.3

- [ ] **Task 7.5**: Fill test coverage gaps to 75%+
  - Files: All test files in `tests/` and `tests/api/`
  - Details: Run `python -m pytest tests/ --cov=loca_llama --cov-report=term-missing -v` to identify uncovered lines. Add tests for: edge cases in analysis (zero-layer models, extreme context lengths), hardware data integrity (no duplicate names, all fields non-null), model data integrity (all families have at least one model), schema validation (invalid types, missing fields return 422), error paths in all routes (malformed input, nonexistent resources).
  - Complexity: **High**
  - Dependencies: All previous test tasks

- [ ] **Task 7.6**: Write unit tests for core modules (hardware, models, quantization)
  - Files: `tests/test_hardware.py`, `tests/test_models.py`, `tests/test_quantization.py`
  - Details: Test data integrity: no duplicate names in `APPLE_SILICON_SPECS`, `MODELS`, or `QUANT_FORMATS`. All hardware specs have positive values for cpu_cores, gpu_cores, memory_gb, memory_bandwidth_gbs. All models have positive params_billion and valid context lengths. All quant formats have bits_per_weight between 2.0 and 16.0. `RECOMMENDED_FORMATS` are a subset of `QUANT_FORMATS` keys.
  - Complexity: **Low**
  - Dependencies: Task 1.7

- [ ] **Task 7.7**: Final integration verification
  - Files: None (verification only)
  - Details: Cross-interface comparison: run same analysis via CLI (`python -m loca_llama --hardware "M4 Pro 48GB" --quant Q4_K_M`) and webapp API (`POST /api/analyze/all`), verify identical results. Test all 6 frontend tabs render without JS errors. Verify `loca-llama-web` entry point starts and serves the app. Verify OpenAPI docs at `/docs`. Run full test suite with coverage report.
  - Complexity: **Low**
  - Dependencies: All previous tasks

**Phase 7 Verification**:
```bash
python -m pytest tests/ --cov=loca_llama --cov-report=term-missing -v  # >= 75% coverage
python -m pytest tests/api/ --cov=loca_llama/api -v                     # >= 85% route coverage
loca-llama-web  # Starts on port 8000, all tabs functional
```

---

## Task Dependencies

### Critical Path

```
Task 1.1 (pyproject.toml)
  -> Task 1.2 (app factory)
    -> Task 1.3 (state + DI)
    -> Task 1.5 (route stubs) [also depends on Task 1.4]
      -> Tasks 2.1-2.4 (data endpoints)
        -> Task 3.1 (single analysis)
          -> Task 3.2 (bulk analysis + max-context)
            -> Task 5.3 (benchmark start/poll) [also depends on Task 5.1]
              -> Task 6.1 (lifespan) -> Task 6.2 (memory endpoints)
```

### Cross-Phase Dependencies

| Task | Depends On | Reason |
|------|-----------|--------|
| Task 1.2 (app factory) | Task 1.1 (pyproject.toml) | Needs FastAPI installed |
| Task 1.3 (state + DI) | Task 1.2 (app factory) | AppState referenced by lifespan |
| Task 1.5 (route stubs) | Task 1.4 (schemas) | Stubs reference response types |
| Task 1.7 (test infra) | Task 1.2 (app factory) | Fixtures need create_app |
| Tasks 2.1-2.4 (data endpoints) | Task 1.5 (route stubs) | Replace stub implementations |
| Task 2.5 (data tests) | Task 1.7 + Tasks 2.1-2.4 | Tests need fixtures + implementations |
| Task 2.6 (frontend dropdowns) | Task 1.6 + Tasks 2.1-2.4 | JS needs live API endpoints |
| Task 3.1 (single analysis) | Tasks 2.1-2.3 | Analysis looks up hardware/model/quant |
| Task 3.5 (Compatibility tab) | Task 2.6 + Tasks 3.1-3.2 | Builds on dropdowns + analysis API |
| Tasks 4.1-4.2 (discovery) | Task 1.3 (state) | Uses asyncio.to_thread pattern |
| Tasks 4.4-4.5 (discovery tabs) | Task 1.6 + Tasks 4.1-4.2 | JS needs live API endpoints |
| Task 5.3 (benchmark start) | Task 5.1 (runtime status) | Validates runtime before starting |
| Task 5.5 (Benchmark tab) | Tasks 5.1-5.3 | Needs all benchmark APIs |
| Task 6.1 (lifespan) | Task 1.2 + Task 1.3 | Wires monitor into app lifecycle |
| Task 6.2 (memory endpoints) | Task 6.1 | Needs running monitor |
| Task 6.4 (Memory tab) | Task 6.2 | Needs live memory API |
| Task 7.1 (error audit) | All route implementations | Audits all routes |
| Task 7.2 (frontend errors) | All frontend tabs | Adds error states to all views |
| Task 7.5 (coverage gaps) | All tests | Fills gaps identified by coverage |

---

## Parallelization Opportunities

### Within Phase 1
- **Task 1.4** (schemas) and **Task 1.6** (static frontend) and **Task 1.7** (test infra) have no dependencies on each other and can run in parallel.
- **Task 1.1** must complete first, then **Task 1.2**, then **Task 1.3** and **Task 1.5** can run in parallel.

### Within Phase 2
- **Tasks 2.1-2.4** (all data endpoints) can be implemented in parallel since they are independent route files.
- **Task 2.5** (tests) and **Task 2.6** (frontend wiring) can run in parallel once endpoints are complete.

### Within Phase 3
- **Task 3.3** (analyzer unit tests) can run in parallel with **Task 3.1** (they test the same module but one tests the core, the other tests the route).
- **Task 3.5** (Compatibility tab) can start once Tasks 3.1-3.2 are done, in parallel with **Task 3.4** (integration tests).

### Within Phase 4
- **Task 4.1** (scanner endpoint) and **Task 4.2** (hub endpoints) are independent and can run in parallel.
- **Task 4.4** (Local Models tab) and **Task 4.5** (HuggingFace tab) are independent and can run in parallel.
- **Task 4.3** (tests) can run in parallel with frontend tasks once endpoints are done.

### Within Phase 5
- **Task 5.1** (runtime) and **Task 5.2** (prompts) can run in parallel.
- **Task 5.4** (tests) and **Task 5.5** (Benchmark tab) can run in parallel once Task 5.3 is done.

### Within Phase 6
- **Task 6.3** (tests) and **Task 6.4** (Memory tab) can run in parallel once Task 6.2 is done.

### Within Phase 7
- **Task 7.1** (error audit), **Task 7.3** (responsive CSS), **Task 7.4** (dark mode), and **Task 7.6** (core unit tests) can all run in parallel.
- **Task 7.2** (frontend errors) depends on Task 7.1 completing.
- **Task 7.5** (coverage gaps) runs last to identify remaining holes.

### Cross-Phase Parallelism
- **Test writing** (Tasks 2.5, 3.3, 3.4, 4.3, 5.4, 6.3, 7.6) can be assigned to a dedicated test-writer agent working in parallel with implementation, provided the schemas (Task 1.4) and test infrastructure (Task 1.7) are complete.
- **Frontend work** (Tasks 2.6, 3.5, 4.4, 4.5, 5.5, 6.4) can be batched to a frontend-focused agent once the corresponding API endpoints are implemented.

---

## Agent Assignments

| Task | Agent | Parallel? | Notes |
|------|-------|-----------|-------|
| **Phase 1** | | | |
| 1.1: pyproject.toml | senior-software-engineer | No | Must complete first |
| 1.2: App factory | senior-software-engineer | No | Depends on 1.1 |
| 1.3: State + DI | senior-software-engineer | Yes (with 1.4, 1.6, 1.7) | After 1.2 |
| 1.4: Pydantic schemas | senior-software-engineer | Yes (with 1.3, 1.6, 1.7) | No dependencies |
| 1.5: Route stubs | senior-software-engineer | No | Depends on 1.4 |
| 1.6: Static frontend shell | senior-software-engineer | Yes (with 1.3, 1.4, 1.7) | No dependencies |
| 1.7: Test infrastructure | test-writer | Yes (with 1.3, 1.4, 1.6) | After 1.2 |
| **Phase 2** | | | |
| 2.1: Hardware endpoints | senior-software-engineer | Yes (with 2.2, 2.3, 2.4) | After 1.5 |
| 2.2: Models endpoint | senior-software-engineer | Yes (with 2.1, 2.3, 2.4) | After 1.5 |
| 2.3: Quant endpoint | senior-software-engineer | Yes (with 2.1, 2.2, 2.4) | After 1.5 |
| 2.4: Template endpoints | senior-software-engineer | Yes (with 2.1, 2.2, 2.3) | After 1.5 |
| 2.5: Data endpoint tests | test-writer | Yes (with 2.6) | After 2.1-2.4 |
| 2.6: Frontend dropdowns | senior-software-engineer | Yes (with 2.5) | After 2.1-2.4 |
| **Phase 3** | | | |
| 3.1: Single analysis | senior-software-engineer | Yes (with 3.3) | After 2.1-2.3 |
| 3.2: Bulk analysis + max-ctx | senior-software-engineer | No | After 3.1 |
| 3.3: Analyzer unit tests | test-writer | Yes (with 3.1) | After 1.7 |
| 3.4: Analysis route tests | test-writer | Yes (with 3.5) | After 3.2 |
| 3.5: Compatibility tab | senior-software-engineer | Yes (with 3.4) | After 2.6 + 3.2 |
| **Phase 4** | | | |
| 4.1: Scanner endpoint | senior-software-engineer | Yes (with 4.2) | After 1.3 |
| 4.2: Hub endpoints | senior-software-engineer | Yes (with 4.1) | After 1.3 |
| 4.3: Discovery tests | test-writer | Yes (with 4.4, 4.5) | After 4.1 + 4.2 |
| 4.4: Local Models tab | senior-software-engineer | Yes (with 4.3, 4.5) | After 4.1 |
| 4.5: HuggingFace tab | senior-software-engineer | Yes (with 4.3, 4.4) | After 4.2 |
| **Phase 5** | | | |
| 5.1: Runtime status | senior-software-engineer | Yes (with 5.2) | After 1.3 |
| 5.2: Benchmark prompts | senior-software-engineer | Yes (with 5.1) | After 1.4 |
| 5.3: Benchmark start/poll | senior-software-engineer | No | After 5.1 |
| 5.4: Benchmark/runtime tests | test-writer | Yes (with 5.5) | After 5.3 |
| 5.5: Benchmark tab | senior-software-engineer | Yes (with 5.4) | After 5.3 |
| **Phase 6** | | | |
| 6.1: Lifespan wiring | senior-software-engineer | No | After 1.2 + 1.3 |
| 6.2: Memory endpoints | senior-software-engineer | No | After 6.1 |
| 6.3: Memory tests | test-writer | Yes (with 6.4) | After 6.2 |
| 6.4: Memory tab | senior-software-engineer | Yes (with 6.3) | After 6.2 |
| **Phase 7** | | | |
| 7.1: Error handling audit | senior-software-engineer | Yes (with 7.3, 7.4, 7.6) | After all routes |
| 7.2: Frontend error states | senior-software-engineer | No | After 7.1 |
| 7.3: Responsive CSS | senior-software-engineer | Yes (with 7.1, 7.4, 7.6) | After 1.6 |
| 7.4: Dark mode | senior-software-engineer | Yes (with 7.1, 7.3, 7.6) | After 7.3 |
| 7.5: Coverage gap fill | test-writer | No | After all other tests |
| 7.6: Core module unit tests | test-writer | Yes (with 7.1, 7.3, 7.4) | After 1.7 |
| 7.7: Final verification | senior-software-engineer | No | After all tasks |

---

## Summary

| Phase | Tasks | Est. Complexity | Key Deliverables |
|-------|-------|----------------|------------------|
| 1: Foundation | 7 | Medium | App skeleton, schemas, stubs, test infra, static shell |
| 2: Data Endpoints | 6 | Low-Medium | Hardware, models, quants, templates APIs + tests + frontend |
| 3: Analysis | 5 | Medium-High | Analyze, analyze/all, max-context APIs + Compatibility tab |
| 4: Discovery | 5 | Medium | Scanner, hub search APIs + Local Models + HuggingFace tabs |
| 5: Benchmark | 5 | High | Runtime detection, async benchmarks + Benchmark tab |
| 6: Memory | 4 | Low-Medium | Lifespan monitor, memory APIs + Memory tab |
| 7: Polish | 7 | Medium | Error handling, loading states, responsive, dark mode, coverage |
| **Total** | **39** | | **21 API endpoints, 6 frontend tabs, 75%+ test coverage** |
