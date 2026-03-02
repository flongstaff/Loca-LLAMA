# Implementation Record: Loca-LLAMA v2 Webapp

**Feature**: loca-llama-v2
**Started**: 2026-03-02
**Status**: IN_PROGRESS

**Reference**:
- Plan: `rpi/loca-llama-v2/plan/PLAN.md`
- Technical Design: `rpi/loca-llama-v2/plan/eng.md`
- Product Requirements: `rpi/loca-llama-v2/plan/pm.md`

**Verification Commands**:
```bash
.venv/bin/pip install -e ".[web,dev]"
.venv/bin/python -m pytest tests/ -v
.venv/bin/uvicorn loca_llama.api.app:app --reload
```

---

## Phase 1: Foundation

**Date**: 2026-03-02
**Verdict**: PASS

### Deliverables
- [x] Task 1.1: pyproject.toml — web/dev optional deps, entry point, pytest config
- [x] Task 1.2: App factory — create_app(), lifespan, StaticFiles mount, serve() CLI
- [x] Task 1.3: State + DI — AppState, BenchmarkJob, get_state() dependency
- [x] Task 1.4: Pydantic schemas — ~28 request/response models in schemas.py
- [x] Task 1.5: Route stubs — 10 route files, 21 endpoints returning 501
- [x] Task 1.6: Static frontend shell — 6-tab SPA (index.html, style.css, app.js)
- [x] Task 1.7: Test infrastructure — conftest.py with shared fixtures

### Files Created
| File | Lines |
|------|-------|
| `loca_llama/api/__init__.py` | 4 |
| `loca_llama/api/app.py` | 75 |
| `loca_llama/api/state.py` | ~60 |
| `loca_llama/api/dependencies.py` | ~20 |
| `loca_llama/api/schemas.py` | ~338 |
| `loca_llama/api/routes/__init__.py` | ~30 |
| `loca_llama/api/routes/hardware.py` | ~25 |
| `loca_llama/api/routes/models.py` | ~15 |
| `loca_llama/api/routes/quantization.py` | ~15 |
| `loca_llama/api/routes/analysis.py` | ~30 |
| `loca_llama/api/routes/templates.py` | ~35 |
| `loca_llama/api/routes/scanner.py` | ~15 |
| `loca_llama/api/routes/hub.py` | ~30 |
| `loca_llama/api/routes/benchmark.py` | ~30 |
| `loca_llama/api/routes/memory.py` | ~20 |
| `loca_llama/api/routes/runtime.py` | ~15 |
| `static/index.html` | ~150 |
| `static/style.css` | ~200 |
| `static/app.js` | ~150 |
| `tests/__init__.py` | 1 |
| `tests/conftest.py` | 39 |
| `tests/api/__init__.py` | 1 |
| `tests/api/conftest.py` | 2 |

### Verification
- Install: PASS (`.venv/bin/pip install -e ".[web,dev]"`)
- App starts: PASS (21 endpoints + static mount registered)
- Stubs: PASS (all return 501)
- Static files: PASS (/ returns index.html with "Loca-LLAMA")
- Swagger: PASS (/docs returns 200)
- Pytest infra: PASS (exit code 5 — no tests yet, expected)

### Notes
- Homebrew Python 3.14 requires venv (PEP 668). Created `.venv/` at project root.
- macOS lacks GNU `timeout` command. Used Python script for integration tests.

---

## Phase 2: Data Endpoints

**Date**: 2026-03-02
**Verdict**: PASS

### Deliverables
- [x] Task 2.1: Hardware endpoints — GET list (44 specs), GET by name with 404
- [x] Task 2.2: Models endpoint — GET list (49 models) with optional ?family= filter
- [x] Task 2.3: Quantization endpoint — GET list (13 formats) with recommended list
- [x] Task 2.4: Template endpoints — GET list, GET match, POST lm-studio-preset, POST llama-cpp-command
- [x] Task 2.5: Tests — 15 tests across 4 test files, all passing
- [x] Task 2.6: Frontend wiring — Models table with family filter, Compatibility dropdowns

### Files Changed
| File | Change | Lines |
|------|--------|-------|
| `loca_llama/api/routes/hardware.py` | rewritten (stub → impl) | 55 |
| `loca_llama/api/routes/models.py` | rewritten (stub → impl) | 40 |
| `loca_llama/api/routes/quantization.py` | rewritten (stub → impl) | 30 |
| `loca_llama/api/routes/templates.py` | rewritten (stub → impl) | 80 |
| `tests/api/test_hardware_routes.py` | new | 37 |
| `tests/api/test_model_routes.py` | new | 38 |
| `tests/api/test_quant_routes.py` | new | 29 |
| `tests/api/test_template_routes.py` | new | 65 |
| `static/index.html` | modified (dropdowns + table containers) | 75 |
| `static/app.js` | rewritten (live data loading) | 145 |
| `static/style.css` | modified (controls layout) | 231 |

### Verification
- Tests: PASS (15/15 in 0.13s)
- App starts: PASS (26 routes registered)
- Endpoints: All 4 data routes return real data; remaining 7 routes still 501

---

## Phase 3: Analysis Endpoints

**Date**: 2026-03-02
**Verdict**: PASS

### Deliverables
- [x] Task 3.1: Single analysis endpoint — POST /api/analyze with hardware/model/quant lookup + 400 errors
- [x] Task 3.2: Bulk analysis + max-context — POST /api/analyze/all with tier summary, POST /api/analyze/max-context
- [x] Task 3.3: Unit tests — 16 tests for estimate_model_size_gb, estimate_kv_cache_gb, estimate_overhead_gb, compute_tier, analyze_model, max_context_for_model
- [x] Task 3.4: Integration tests — 11 tests for analysis API routes (valid/invalid inputs, filters, core match)
- [x] Task 3.5: Compatibility tab — sortable results table, tier badges, detail panel with memory breakdown + max context

### Files Changed
| File | Change | Lines |
|------|--------|-------|
| `loca_llama/api/routes/analysis.py` | rewritten (stub → impl) | 127 |
| `tests/test_analyzer.py` | new | 137 |
| `tests/api/test_analysis_routes.py` | new | 153 |
| `static/app.js` | rewritten (analysis tab + sorting) | 392 |
| `static/index.html` | modified (family filter, checkbox, containers) | +8 |
| `static/style.css` | modified (badges, summary, detail panel) | +121 |

### Verification
- Tests: PASS (42/42 in 0.25s — 27 new + 15 existing)
- Build: PASS (app starts with all routes registered)
- Analysis API matches core analyzer output for same inputs

### Notes
- ModelEstimate → AnalyzeResponse mapping: `tier.value` for tier string, `rating` for tier_label
- Rounding: 2 decimals for GB, 1 for percentages/tok/s, 0 for context K values
- Frontend tier sorting uses TIER_ORDER map for correct ordering
- Detail panel makes secondary API call to /api/analyze/max-context

---

## Phase 4: Discovery Endpoints

**Date**: 2026-03-02
**Verdict**: PASS

### Deliverables
- [x] Task 4.1: Scanner endpoint — GET /api/scanner/local with optional custom_dir, source breakdown, total_size_gb
- [x] Task 4.2: Hub endpoints — GET /api/hub/search (gguf/mlx/all dispatch), GET /api/hub/files/{repo_id}, GET /api/hub/config/{repo_id}
- [x] Task 4.3: Mocked tests — 6 scanner + 12 hub tests with unittest.mock.patch at route level
- [x] Task 4.4: Local Models tab — scan button, custom dir input, results table with Name/Size/Format/Source/Quant/Family
- [x] Task 4.5: HuggingFace tab — debounced search, format/sort filters, file listing panel

### Files Changed
| File | Change | Lines |
|------|--------|-------|
| `loca_llama/api/routes/scanner.py` | rewritten (stub → impl) | 59 |
| `loca_llama/api/routes/hub.py` | rewritten (stub → impl) | 108 |
| `tests/api/test_scanner_routes.py` | new | 112 |
| `tests/api/test_hub_routes.py` | new | 193 |
| `static/index.html` | modified (Local Models + HuggingFace tabs) | +20 |
| `static/app.js` | modified (scanLocalModels, searchHub, showRepoFiles) | +170 |

### Verification
- Tests: PASS (60/60 in 0.45s — 18 new + 42 existing)
- Build: PASS (app loads with 26 routes)

### Code Review
- APPROVED WITH FIXES APPLIED
- Fixed: showRepoFiles URL encoding bug (critical), generic error messages in 500/502 responses (security), type hint on _model_to_response
- Deferred to Phase 7: `format` builtin shadowing, `sort` Literal validation, scan button disable-while-running

### Commit
`0aa4c18` — Implement discovery endpoints with scanner, hub, and frontend tabs

### Notes
- All blocking I/O wrapped in asyncio.to_thread() (filesystem scan, network requests)
- LocalModel.path is Path → needs str() for Pydantic serialization
- Hub format param dispatches to search_gguf_models/search_mlx_models/search_huggingface
- HubFileResponse.size defaults None→0 in route
- 300ms debounce on HuggingFace search input
- Error details sanitized: 500/502 return generic messages, raw exceptions logged server-side only

---

## Phase 5: Benchmark & Runtime

**Date**: 2026-03-02
**Verdict**: PASS

### Deliverables
- [x] Task 5.1: Runtime status endpoint — GET /api/runtime/status via asyncio.to_thread(detect_all_runtimes)
- [x] Task 5.2: Benchmark prompts endpoint — GET /api/benchmark/prompts returning BENCH_PROMPTS dict
- [x] Task 5.3: Benchmark start + polling — POST /api/benchmark/start (validates runtime/model/prompt, launches asyncio.create_task), GET /api/benchmark/{job_id} (progress/results/error)
- [x] Task 5.4: Mocked tests — 5 runtime + 11 benchmark tests with unittest.mock.patch
- [x] Task 5.5: Benchmark tab — detect runtimes, select runtime/model, configure prompt/runs, start benchmark, poll progress bar, display aggregate + per-run results

### Files Changed
| File | Change | Lines |
|------|--------|-------|
| `loca_llama/api/routes/runtime.py` | rewritten (stub → impl) | 39 |
| `loca_llama/api/routes/benchmark.py` | rewritten (stub → impl) | 180 |
| `tests/api/test_runtime_routes.py` | new | 83 |
| `tests/api/test_benchmark_routes.py` | new | 247 |
| `static/index.html` | modified (benchmark tab controls) | +24 |
| `static/app.js` | modified (benchmark tab JS) | +213 |
| `static/style.css` | modified (progress bar CSS) | +30 |

### Verification
- Tests: PASS (76/76 in 1.49s — 16 new + 60 existing)
- Build: PASS (app loads with 26 routes)

### Code Review
- APPROVED WITH FIXES APPLIED
- Fixed: raw exception leak in background task error (critical — sanitized to generic message), thread-safe progress updates via call_soon_threadsafe (critical — data race), stored create_task reference (warning)
- Deferred: 400 error messages echo user input (acceptable for localhost tool), "Run 0 of N" UX, schema upper bounds on max_tokens/context_length

### Commit
`fc2fbe0` — Implement benchmark & runtime endpoints with thread-safe progress tracking

### Notes
- Background task pattern: asyncio.create_task() + asyncio.to_thread(run_benchmark_suite) with progress_callback
- Progress callback uses loop.call_soon_threadsafe() for thread-safe attribute writes from worker thread
- Error messages sanitized in both sync (HTTPException) and async (background task) error paths
- Task reference stored on job object to prevent garbage collection

---
