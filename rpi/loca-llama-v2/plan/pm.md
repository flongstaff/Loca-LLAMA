# Loca-LLAMA v2 Webapp — Product Requirements Document

**Date**: 2026-03-02
**Author**: Product Manager Agent
**Status**: Draft
**Verdict**: CONDITIONAL GO (from Research)

---

## 1. Context & Why Now

### Problem Statement

Loca-LLAMA is a mature Python CLI/TUI tool that solves a real problem for Apple Silicon Mac users: determining which LLMs will run on their hardware *before* downloading 10GB+ model files. The tool provides VRAM estimation, compatibility tiering, runtime comparison, benchmarking, and model discovery across 49 LLM definitions and 44 Apple Silicon hardware configurations.

However, the tool is currently accessible only through terminal interfaces (CLI and TUI), limiting its audience to developers comfortable with command-line tools. Non-CLI users — including data scientists who primarily work in notebooks/GUIs, and technical managers evaluating hardware investments — cannot easily use the tool.

### Opportunity

Adding a local webapp interface (FastAPI + vanilla HTML/CSS/JS) makes the tool accessible to a broader audience without compromising the CLI/TUI experience. The existing codebase is architecturally clean: 13 modules with zero external dependencies in core logic, all functions return serializable data structures, and no TODOs or FIXMEs exist. This means the webapp is purely additive — API routes wrap existing functions, and the frontend renders the same data the CLI already computes.

### Why Now

1. **Codebase maturity**: v1 is 90%+ feature-complete with stable, well-structured code. Adding a web layer now avoids the risk of refactoring a less mature codebase.
2. **Apple Silicon adoption**: M4 family (including M4 Ultra 256GB) is now in the hardware database, covering the full current product line. The market of developers running local LLMs on Apple Silicon continues to grow.
3. **FastAPI ecosystem**: Python 3.11+ async/await patterns and FastAPI's zero-boilerplate design make the web integration straightforward — all core logic is directly callable from async route handlers.
4. **Zero-dependency core preserved**: The web framework is additive (optional dependency), not a refactor of the core. CLI/TUI continue to work without installing FastAPI.

### Assumptions

- Users run the webapp on their local Mac (localhost:8000). No cloud deployment is planned or supported.
- Users have Python 3.11+ installed (same requirement as CLI).
- LM Studio and/or llama.cpp are already installed and configured for benchmarking and runtime comparison features. The webapp does not install or manage these runtimes.
- The existing 49 model definitions and 44 hardware configs are sufficient for the initial release. New models can be added to the data layer without webapp changes.

---

## 2. Users & Jobs-to-be-Done

### Persona 1: Mac Developer (Primary)

**Profile**: Software engineer with an Apple Silicon Mac (M2-M4 family, 16-64GB). Uses LLMs for coding assistance, local inference for privacy, or hobby projects. Comfortable with CLI but prefers visual tools for comparison workflows.

**Jobs-to-be-Done**:
- "Tell me which models fit my Mac so I don't waste time downloading incompatible ones."
- "Show me how much memory a specific model + quantization combo will use, and how fast it will run."
- "Compare Q4_K_M vs Q6_K for the same model — what do I trade off in quality vs. speed?"

**Pain Points**:
- CLI output is ephemeral — can't easily compare multiple configurations side by side.
- TUI navigation requires learning keybindings; non-obvious for first-time users.
- No way to share analysis results with team members (terminal output is not linkable).

### Persona 2: Data Scientist / ML Engineer

**Profile**: Evaluates models for on-device inference. Works in Jupyter notebooks and web-based tools. May not have CLI proficiency. Needs to compare multiple models against hardware constraints.

**Jobs-to-be-Done**:
- "I need to find the largest model that fits comfortably on my Mac with enough headroom for 32K context."
- "Show me all models from the Qwen family that run at full GPU speed on my hardware."
- "I want to search HuggingFace for GGUF models and immediately see if they'll fit."

**Pain Points**:
- CLI commands require memorizing flags and hardware config names.
- No filtering or sorting by family, size, or compatibility tier in the TUI.
- HuggingFace search results in the TUI don't automatically cross-reference with local hardware analysis.

### Persona 3: Power User / Runtime Comparer

**Profile**: Experienced LLM user who runs both LM Studio and llama.cpp. Wants to benchmark the same model on both runtimes and pick the faster one.

**Jobs-to-be-Done**:
- "Run the same prompt through LM Studio and llama.cpp and show me tokens/sec side by side."
- "Monitor memory usage during a benchmark to see if swapping is occurring."
- "Find my locally downloaded models and check which ones I should re-quantize for better performance."

**Pain Points**:
- Benchmark output in the TUI disappears when navigating away.
- No persistent benchmark history — must re-run to compare.
- Memory monitoring is a background thread with no visual representation.

### Persona 4: Model Explorer

**Profile**: Browses HuggingFace for new models, especially GGUF and MLX variants. Wants to discover models and immediately assess feasibility on their hardware.

**Jobs-to-be-Done**:
- "Search HuggingFace for 'deepseek reasoning gguf' and see which results fit my Mac."
- "Browse my locally downloaded models and see their compatibility at a glance."
- "Find recommended settings (quantization, context length, temperature) for a model I'm interested in."

**Pain Points**:
- HuggingFace search in CLI requires exact syntax; no type-ahead or fuzzy search.
- No visual indication of model size relative to available memory.
- Template recommendations exist but are buried in CLI subcommands.

---

## 3. Success Metrics

### Leading Indicators (Measurable During Development)

| Metric | Target | Measurement |
|--------|--------|-------------|
| API endpoint coverage | 100% of core CLI features exposed as API routes | Count of routes vs. CLI commands |
| Feature parity | 80%+ of CLI/TUI features accessible via webapp | Feature checklist comparison |
| API response time (p95) | < 200ms for read endpoints (compatibility, models, templates) | FastAPI request logging |
| Test coverage for API routes | >= 80% line coverage in `loca_llama/api/` | pytest-cov report |
| Overall test coverage | >= 70% line coverage | pytest-cov report |
| Build verification | `pip install -e .[web]` succeeds on Python 3.11, 3.12, 3.13 | CI matrix |

### Lagging Indicators (Post-Release)

| Metric | Target | Measurement |
|--------|--------|-------------|
| CLI backward compatibility | Zero regressions — all existing CLI commands produce identical output | Manual + automated regression |
| VRAM estimation accuracy | Webapp and CLI return identical values for same input | Cross-interface comparison test |
| Benchmark consistency | Webapp benchmark results within 5% of CLI benchmark results for same model/runtime | Side-by-side benchmark runs |
| Webapp page load time | < 1 second for initial load (static files from localhost) | Browser dev tools |
| Time to first useful result | < 10 seconds from opening webapp to seeing compatibility results | User timing |

---

## 4. Functional Requirements

### FR-1: Hardware Selection API

**Description**: Expose the hardware database via API so the webapp can present a hardware selector.

**Acceptance Criteria**:
- `GET /api/hardware` returns a JSON array of all 44 Apple Silicon configurations.
- Each entry includes: `name` (string), `chip` (string), `cpu_cores` (int), `gpu_cores` (int), `memory_gb` (int), `memory_bandwidth_gbs` (float), `usable_memory_gb` (float).
- Response is sorted by chip family (M1, M2, M3, M4), then by memory size ascending.
- Response time < 50ms (in-memory data, no I/O).

### FR-2: Model Compatibility Check API

**Description**: Expose the core VRAM analysis engine via API. Given a hardware config and optional filters, return compatibility results for all models.

**Acceptance Criteria**:
- `POST /api/check` accepts: `hardware_name` (required string), `context_length` (optional int), `quant_formats` (optional string array), `family_filter` (optional string), `include_partial` (optional bool, default false).
- Returns a JSON array of `ModelEstimate` objects, each containing: model name, family, params (billions), quantization, context length, model size (GB), KV cache (GB), overhead (GB), total memory (GB), headroom (GB), memory utilization (%), compatibility tier, estimated tokens/sec, GPU layers, offload percentage.
- Results match the output of `analyzer.analyze_all()` for the same inputs — verified by unit test.
- Invalid `hardware_name` returns HTTP 400 with `{"detail": "Hardware '<name>' not found. Use GET /api/hardware for valid options."}`.
- Response time < 500ms for full model database (49 models x 5 recommended quant formats = 245 combinations).

### FR-3: Single Model Detail API

**Description**: Deep analysis of one specific model, including memory breakdown across all quantization formats and a context scaling table.

**Acceptance Criteria**:
- `GET /api/model/{model_name}?hardware={hardware_name}` returns detailed analysis.
- Response includes: model metadata (name, family, params, layers, KV heads, head dim, license), per-quantization analysis (all 13 formats with VRAM, tier, estimated speed), maximum context length per quantization format, and recommended quantization for the given hardware.
- `model_name` not found returns HTTP 404.
- `hardware` query param is required; omitting it returns HTTP 422 (Pydantic validation).

### FR-4: Model Listing API

**Description**: List all known models with metadata for the model browser.

**Acceptance Criteria**:
- `GET /api/models` returns all 49 models with: name, family, params (billions), default context length, max context length, license.
- Supports optional query params: `family` (filter by family name), `min_params` / `max_params` (filter by size range).
- Response includes `count` field with total number of results.
- Response time < 50ms.

### FR-5: Quantization Formats API

**Description**: Expose the quantization format reference data.

**Acceptance Criteria**:
- `GET /api/quantizations` returns all 13 GGUF quantization formats.
- Each entry includes: `name`, `bits_per_weight`, `quality_rating`, `description`.
- Also returns the `recommended_formats` array (`["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]`).

### FR-6: Local Model Scanner API

**Description**: Scan the user's filesystem for downloaded models (LM Studio, llama.cpp, HuggingFace cache).

**Acceptance Criteria**:
- `GET /api/scan` scans all default paths (LM Studio: 2 paths, llama.cpp: 2 paths, HuggingFace: 1 path) and returns found models.
- Each model includes: name, path (string), size (GB), format (gguf/mlx/safetensors), source (lm-studio/llama.cpp/huggingface/mlx-community), detected quantization, detected family, repo ID (if available).
- `POST /api/scan` with `{"directory": "/path/to/custom"}` scans a custom directory.
- Custom directory path must exist; non-existent path returns HTTP 400.
- Results are deduplicated by file path and sorted by size descending.
- Response time < 5 seconds for directories with up to 100 model files.

### FR-7: HuggingFace Search API

**Description**: Search HuggingFace Hub for GGUF and MLX models.

**Acceptance Criteria**:
- `GET /api/hf-search?query={query}` searches HuggingFace with the given query string.
- Supports optional params: `limit` (int, default 20, max 50), `sort` (downloads|likes|lastModified), `format` (gguf|mlx|all, default all).
- Each result includes: repo_id, name, author, downloads (formatted), likes, tags, is_mlx, is_gguf, last_modified.
- Network errors return HTTP 502 with `{"detail": "Unable to reach HuggingFace API"}`.
- Request to HuggingFace times out after 15 seconds.

### FR-8: Runtime Status API

**Description**: Check connectivity to LM Studio and llama.cpp runtimes.

**Acceptance Criteria**:
- `GET /api/runtimes` returns status of each runtime: `{"lm_studio": {"running": true, "models": ["model-id"]}, "llama_cpp": {"running": false, "models": []}}`.
- LM Studio checked at `http://127.0.0.1:1234/v1/models`.
- llama.cpp checked at ports 8080, 8081 (skipping 8000 to avoid conflict with webapp).
- Each runtime check times out after 3 seconds.
- Total endpoint response time < 10 seconds (worst case: both runtimes unreachable, 3s timeout each plus port scanning).

### FR-9: Benchmark Execution API (Async)

**Description**: Run performance benchmarks against loaded models in LM Studio or llama.cpp. Benchmarks are long-running (30s+) and must not block the server.

**Acceptance Criteria**:
- `POST /api/benchmark` accepts: `model_id` (required), `runtime` (auto|lm-studio|llama.cpp, default auto), `duration_seconds` (int, 5-300, default 30), `num_runs` (int, 1-10, default 3).
- Returns immediately with `{"job_id": "<uuid>", "status": "queued"}`.
- `GET /api/benchmark/{job_id}` returns current status: `queued`, `running`, `complete`, or `error`.
- On completion, result includes: tokens per second (mean, min, max across runs), prompt eval speed, total tokens generated, runtime used, memory stats during benchmark.
- On error, result includes: `{"status": "error", "error": "<user-safe message>"}`.
- Benchmark runs in a background task (FastAPI BackgroundTasks or asyncio.to_thread), not blocking the event loop.
- Job results are stored in-memory (lost on server restart — acceptable for local tool).
- Requesting a non-existent job_id returns HTTP 404.

### FR-10: Memory Monitor API

**Description**: Expose real-time memory usage from the background memory monitor.

**Acceptance Criteria**:
- `GET /api/memory` returns current memory statistics: total memory (GB), used memory (GB), free memory (GB), swap used (GB), memory pressure level.
- Memory monitor starts on FastAPI app startup and stops on shutdown (via lifespan context manager).
- Data sourced from macOS `vm_stat` command (same as existing `memory_monitor.py`).
- Endpoint returns the most recent sample (not a new sample on each request).
- Response time < 50ms.

### FR-11: Model Templates API

**Description**: Expose recommended configurations and settings templates per model family.

**Acceptance Criteria**:
- `GET /api/templates` returns all available model templates.
- `GET /api/templates/{family}` returns templates for a specific model family (e.g., "Llama", "Qwen").
- Each template includes: model family, recommended quantization per hardware tier, recommended context length, recommended temperature, system prompt suggestion, and llama.cpp/LM Studio launch commands.
- Family not found returns HTTP 404.

### FR-12: Webapp Frontend — Hardware & Compatibility View

**Description**: Single-page web application served from FastAPI's static file mount. The primary view allows selecting hardware and viewing model compatibility.

**Acceptance Criteria**:
- Webapp loads at `http://localhost:8000/` and displays a hardware selector dropdown populated from `GET /api/hardware`.
- Selecting hardware triggers `POST /api/check` and displays results in a table.
- Table columns: Model Name, Family, Quantization, Total Memory (GB), Headroom (GB), Tier, Est. Speed (tok/s).
- Rows are color-coded by compatibility tier: FULL_GPU (green), COMFORTABLE (blue), TIGHT_FIT (yellow), PARTIAL_OFFLOAD (orange), WONT_FIT (red/grayed out).
- Table is sortable by any column (client-side sort).
- Filter controls: family dropdown, tier filter checkboxes, quantization multi-select.
- WONT_FIT models are hidden by default; a "Show all" toggle reveals them.
- Page is functional with JavaScript disabled (progressive enhancement): static HTML renders, but interactivity requires JS.

### FR-13: Webapp Frontend — Model Detail View

**Description**: Detailed analysis view for a single model, accessible by clicking a model row in the compatibility table.

**Acceptance Criteria**:
- Clicking a model row navigates to a detail panel (in-page, not a new page).
- Detail panel shows: memory breakdown bar chart (model weights vs. KV cache vs. overhead vs. headroom), quantization comparison table (all 13 formats with size, tier, speed), context scaling information (max context per quantization), and recommended configuration.
- A "Back to results" button returns to the compatibility table without re-fetching data.

### FR-14: Webapp Frontend — Local Model Scanner View

**Description**: View for scanning and displaying locally downloaded models.

**Acceptance Criteria**:
- "Scan Local Models" button triggers `GET /api/scan` and displays results.
- Results shown in a card or table layout with: model name, size (GB), format badge (GGUF/MLX/SafeTensors), source badge (LM Studio/llama.cpp/HuggingFace), detected quantization, detected family.
- "Scan Custom Directory" allows entering a path and scanning via `POST /api/scan`.
- Loading state shown during scan (spinner or progress indicator).
- Empty state: "No models found. Download models through LM Studio or llama.cpp."

### FR-15: Webapp Frontend — HuggingFace Search View

**Description**: Search interface for browsing HuggingFace models.

**Acceptance Criteria**:
- Search input field with debounced input (300ms delay before API call).
- Results displayed in a list with: repo ID, author, download count, like count, format badges (GGUF/MLX).
- Sort selector: downloads (default), likes, recently modified.
- Format filter: All, GGUF only, MLX only.
- Each result has a "Check compatibility" action that navigates to the model detail view with the user's selected hardware.
- Network error state: "Unable to reach HuggingFace. Check your internet connection."

### FR-16: Webapp Frontend — Benchmark View

**Description**: Interface for running and viewing benchmark results.

**Acceptance Criteria**:
- Shows detected runtimes (from `GET /api/runtimes`) with status indicators.
- "Start Benchmark" form with: model selector (from loaded models in detected runtimes), runtime selector, duration slider (5-300 seconds), number of runs (1-10).
- After submission, shows a progress indicator with polling (`GET /api/benchmark/{job_id}` every 2 seconds).
- On completion, displays: tokens/sec (mean/min/max), a bar chart comparing runs, memory usage during benchmark.
- Error state: "Benchmark failed: <reason>. Ensure the model is loaded in <runtime>."
- If no runtimes are detected: "No runtimes detected. Start LM Studio or llama.cpp to run benchmarks."

### FR-17: FastAPI App Initialization & Lifecycle

**Description**: Proper FastAPI application setup with lifespan management for background services.

**Acceptance Criteria**:
- `create_app()` factory function returns configured FastAPI instance.
- App uses lifespan context manager to start/stop memory monitor.
- All routers registered with `/api` prefix and appropriate tags.
- Static files served from `static/` directory at root mount (`/`).
- App binds to `127.0.0.1:8000` (localhost only, not `0.0.0.0`).
- Running `python -m loca_llama.api.app` or `uvicorn loca_llama.api.app:app` starts the server.
- OpenAPI docs accessible at `/docs` (Swagger UI) and `/redoc`.

### FR-18: CLI Entry Point for Webapp

**Description**: Add a CLI command to start the webapp.

**Acceptance Criteria**:
- `loca-llama-web` command starts the webapp server (registered in `pyproject.toml` scripts).
- Accepts `--port` flag (default 8000) and `--host` flag (default 127.0.0.1).
- Prints "Loca-LLAMA webapp running at http://127.0.0.1:8000" on startup.
- Graceful shutdown on Ctrl+C (SIGINT).
- If FastAPI/uvicorn not installed, prints: "Webapp requires additional dependencies. Install with: pip install loca-llama[web]" and exits with code 1.

### FR-19: Optional Dependency Management

**Description**: FastAPI and uvicorn are optional dependencies, not required for CLI/TUI usage.

**Acceptance Criteria**:
- `pyproject.toml` declares web dependencies as extras: `[project.optional-dependencies] web = ["fastapi>=0.109", "uvicorn>=0.27"]`.
- `pip install loca-llama` installs CLI/TUI only (zero external dependencies).
- `pip install loca-llama[web]` installs CLI/TUI plus FastAPI and uvicorn.
- Core modules (`analyzer.py`, `hardware.py`, `models.py`, etc.) have zero imports from FastAPI or uvicorn.

---

## 5. Non-Functional Requirements

### NFR-1: Performance

- **API read endpoints** (hardware, models, quantizations, templates): p95 response time < 100ms. These serve in-memory data with no I/O.
- **Compatibility analysis** (`POST /api/check` with full model database): p95 response time < 500ms.
- **Local model scan** (`GET /api/scan`): response time < 5 seconds for up to 100 model files across all default paths.
- **HuggingFace search**: response time bounded by upstream API (15-second timeout). Webapp shows loading state immediately.
- **Frontend initial load**: < 1 second for all static assets from localhost (HTML + CSS + JS, no build step, no bundler).
- **Concurrent requests**: Server handles at least 10 concurrent API requests without degradation (single user, but browser may issue parallel requests for different data).

### NFR-2: Security

- **Localhost binding only**: Server binds to `127.0.0.1`, never `0.0.0.0`. This is enforced in the startup configuration, not just documented.
- **No authentication**: Acceptable because the tool is local-only. Document this explicitly in README.
- **No file writes**: API endpoints read filesystem data (model scanning) but never write, create, delete, or modify files.
- **Input validation**: All user inputs validated via Pydantic models before processing. Path inputs for custom directory scan validated to exist and be a directory (no path traversal).
- **No secrets**: No API keys, tokens, or credentials stored or transmitted. HuggingFace API is used without authentication (public endpoints only).
- **Error sanitization**: Internal errors (stack traces, file paths, function names) never exposed in API responses. All 500 errors return generic `{"detail": "Internal server error"}`.
- **Subprocess safety**: llama.cpp server start command uses explicit argument list (no shell=True). Model paths are validated before passing to subprocess.

### NFR-3: Accessibility

- **Semantic HTML**: Webapp uses semantic elements (`<nav>`, `<main>`, `<table>`, `<form>`, `<button>`) not generic `<div>` soup.
- **Keyboard navigation**: All interactive elements (dropdowns, buttons, table sort headers) are keyboard-accessible.
- **Color-independent encoding**: Compatibility tiers use both color AND text labels/icons — not color alone.
- **Screen reader support**: Tables have `<caption>` and `<th scope>` attributes. Form inputs have associated `<label>` elements.
- **Responsive design**: Webapp is usable on screens 768px wide and larger (laptop minimum). Not required to work on mobile.

### NFR-4: Observability

- **Structured logging**: API requests logged with: endpoint, method, status code, response time (ms). Use Python `logging` module, not `print()`.
- **Error logging**: All caught exceptions logged with context (endpoint, input parameters, error type) at ERROR level.
- **Startup logging**: Server startup logs: Python version, FastAPI version, bound address, number of hardware configs loaded, number of models loaded.
- **Memory monitor logging**: Log memory sampling start/stop events at INFO level.

### NFR-5: Compatibility

- **Python versions**: 3.11, 3.12, 3.13.
- **macOS versions**: macOS 13 (Ventura) and later. Memory monitoring depends on `vm_stat` command (present in all macOS versions).
- **Browsers**: Latest versions of Safari, Chrome, Firefox. No IE11 support.
- **Apple Silicon**: All M1 through M4 Ultra configurations. Not tested or supported on Intel Macs (memory monitoring and bandwidth estimates are Apple Silicon-specific).

---

## 6. Scope

### In Scope

- FastAPI web server with RESTful JSON API (all routes listed in FR-1 through FR-11)
- Single-page webapp frontend with vanilla HTML/CSS/JS (FR-12 through FR-16)
- FastAPI app lifecycle management with memory monitor (FR-17)
- CLI entry point for starting webapp (FR-18)
- Optional dependency management in pyproject.toml (FR-19)
- pytest test suite for API routes (80%+ coverage) and core logic (70%+ coverage)
- OpenAPI documentation auto-generated by FastAPI (`/docs`, `/redoc`)
- GitHub Actions CI workflow for Python 3.11+ on macOS

### Explicitly Out of Scope

- **Ollama support**: Per original requirements. No Ollama runtime connector, no Ollama model format support.
- **Cloud deployment**: No Docker, no Kubernetes, no cloud hosting. Local-only tool.
- **Authentication / authorization**: No users, no sessions, no tokens. Localhost access is implicit trust.
- **Remote model serving**: No model download/install functionality. Users download models through LM Studio or direct HuggingFace CLI.
- **Database / persistent storage**: All data in-memory. No SQLite, no file-based storage. Benchmark results are ephemeral.
- **Real-time streaming**: No WebSocket or SSE for live benchmark streaming. Polling-based updates are sufficient for a single-user local tool.
- **Model execution**: The webapp does not run LLM inference directly. It connects to LM Studio or llama.cpp which do the actual inference.
- **Mobile support**: Not designed for screens below 768px.
- **Package publishing**: No PyPI release in this phase. Install from source (`pip install -e .[web]`).
- **Frontend build system**: No webpack, no Vite, no npm. Vanilla HTML/CSS/JS served directly.
- **MLX runtime connector**: MLX models are scanned and displayed, but there is no MLX runtime connector for benchmarking. Only LM Studio and llama.cpp are supported as runtimes.

---

## 7. Rollout Plan

### Phase 1: API Foundation (Days 1-2)

**Goal**: All read-only API endpoints working and tested.

**Deliverables**:
1. FastAPI app skeleton (`loca_llama/api/app.py`, router structure)
2. Pydantic request/response models (`loca_llama/api/models.py`)
3. Read-only routes: `GET /api/hardware`, `GET /api/models`, `GET /api/model/{name}`, `GET /api/quantizations`, `GET /api/templates`
4. `pyproject.toml` updated with optional `[web]` dependencies
5. pytest configuration and fixtures in `tests/conftest.py`
6. Unit tests for all Phase 1 routes

**Exit Criteria**: `python -m pytest tests/api/ -v` passes. All Phase 1 routes return correct JSON validated against Pydantic models.

### Phase 2: Analysis & Discovery APIs (Days 3-4)

**Goal**: Core analysis, scanning, and search endpoints.

**Deliverables**:
1. `POST /api/check` — compatibility analysis route
2. `GET /api/scan` and `POST /api/scan` — local model scanner routes
3. `GET /api/hf-search` — HuggingFace search route
4. `GET /api/runtimes` — runtime detection route
5. `GET /api/memory` — memory monitor route with lifespan management
6. Integration tests for all Phase 2 routes

**Exit Criteria**: Full API passes tests. Compatibility analysis results verified against CLI output for 3 hardware configs (M1 8GB, M4 Pro 48GB, M4 Ultra 256GB).

### Phase 3: Benchmark API (Day 5)

**Goal**: Async benchmark execution with job polling.

**Deliverables**:
1. `POST /api/benchmark` — async benchmark submission
2. `GET /api/benchmark/{job_id}` — result polling
3. Background task integration with `asyncio.to_thread()`
4. Tests with mocked benchmark execution

**Exit Criteria**: Benchmark job lifecycle (queued -> running -> complete/error) verified through API tests. Background execution confirmed to not block the event loop.

### Phase 4: Frontend (Days 6-9)

**Goal**: Functional single-page webapp.

**Deliverables**:
1. `static/index.html` — main page layout with navigation
2. `static/style.css` — styling with tier color coding and responsive layout
3. `static/app.js` — API client, rendering logic, interactivity
4. Hardware selection and compatibility table view (FR-12)
5. Model detail view (FR-13)
6. Local model scanner view (FR-14)
7. HuggingFace search view (FR-15)
8. Benchmark view (FR-16)

**Exit Criteria**: All views render correctly in Safari, Chrome, and Firefox. Keyboard navigation works for all interactive elements. Color-independent tier indicators verified.

### Phase 5: Integration & Polish (Days 10-11)

**Goal**: CLI entry point, CI, documentation, and polish.

**Deliverables**:
1. `loca-llama-web` CLI entry point (FR-18)
2. GitHub Actions workflow for Python 3.11/3.12/3.13 on macOS
3. README.md updated with webapp usage instructions
4. Structured logging added to all routes
5. Error handling audit — all routes verified for consistent error shapes
6. Cross-interface verification: CLI vs. webapp output comparison for 5 model/hardware combinations

**Exit Criteria**: `pip install -e .[web]` followed by `loca-llama-web` starts the server. CI pipeline green. README includes webapp quickstart.

---

## 8. Risks & Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Memory monitor is macOS-specific** (`vm_stat` command). Non-macOS users get no memory data. | Low (tool is macOS-only by design) | Low | Return `{"error": "Memory monitoring requires macOS"}` on non-macOS. Document in README. |
| **llama.cpp port conflict**: Webapp defaults to port 8000; llama.cpp may also use port 8000. | Medium | Medium | Webapp default port is 8000; llama.cpp default is 8080. Skip port 8000 when scanning for llama.cpp runtimes (FR-8). Document port configuration. |
| **HuggingFace API rate limiting**: Unauthenticated API calls may be rate-limited during heavy search usage. | Medium | Low | 15-second timeout already in place. Return HTTP 502 with clear message. Consider client-side caching of search results (5-minute TTL). |
| **Large model databases**: If model count grows significantly (500+), `POST /api/check` performance may degrade. | Low (currently 49 models) | Low | Analysis is O(models x quant_formats). At 500 models x 13 formats = 6500 computations — still sub-second with pure math. Monitor and add pagination if needed. |
| **Benchmark state loss on restart**: In-memory job storage means benchmark results disappear on server restart. | Medium | Low | Document behavior. Acceptable for local tool where server restarts are intentional. |
| **No test infrastructure exists today**: Building test suite from scratch adds time and introduces risk of low-quality tests. | Medium | Medium | Follow the testing strategy established in `.claude/rules/testing-strategy.md`. Start with route-level integration tests (highest value), add unit tests for analyzer functions. Use conftest fixtures to avoid duplication. |
| **Frontend complexity creep**: Vanilla JS for a multi-view SPA may become unwieldy without a framework. | Medium | Medium | Keep views simple — tab-based navigation, server-side data logic. Each view is a function that renders into a container div. No client-side routing. If complexity exceeds ~1500 LOC in app.js, consider splitting into per-view modules. |

### Open Questions

1. **Should the webapp auto-detect the user's hardware?** macOS `sysctl` can detect the chip and memory configuration. This would eliminate the need for manual hardware selection. Decision needed: auto-detect with manual override, or manual-only?

2. **Should benchmark results persist to a JSON file?** Currently planned as in-memory only. A simple `~/.loca-llama/benchmarks.json` file could provide benchmark history across server restarts. Cost: minor file I/O. Benefit: comparative analysis over time.

3. **Should the webapp support dark mode?** Many developer tools support dark mode. Cost: additional CSS (prefers-color-scheme media query). Benefit: better developer experience. Recommendation: add in Phase 4 with minimal effort.

4. **Should the OpenAPI docs be exposed in production?** FastAPI auto-generates `/docs` and `/redoc`. These are useful for debugging but may confuse non-technical users. Decision: always show (local tool, no security concern) or hide behind a `--debug` flag?

5. **What is the maximum number of concurrent benchmark jobs?** Current design allows unlimited background tasks. Should there be a limit (e.g., 1 active benchmark at a time) to prevent resource exhaustion? Recommendation: limit to 1 concurrent benchmark since benchmarks need exclusive hardware access for accurate results.

6. **Should the webapp include a "compare" feature?** Allow selecting 2-3 model/quantization combinations and viewing a side-by-side comparison table. Not in the original request but would add significant value for the primary user persona. Decision: include in Phase 4 or defer to v2.1?

---

*Generated with Claude Code on 2026-03-02. Based on research verdict: CONDITIONAL GO.*
