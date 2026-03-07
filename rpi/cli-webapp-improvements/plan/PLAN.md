# Implementation Roadmap: CLI & Webapp Improvements

**Project**: Loca-LLAMA v2
**Date**: 2026-03-06
**Status**: Ready for implementation
**Research Verdict**: GO (High Confidence)
**Scope**: 8 features, ~600 LOC (production + tests), zero new dependencies

---

## Overview

Eight improvements to close feature parity gaps between CLI and webapp, reduce friction for the most common workflow ("will this model fit my Mac?"), and improve error handling. All changes reuse existing business logic with zero new dependencies.

**Implementation order**: 1 -> 5 -> 8 -> 6 -> 7 -> 3 -> 4 -> 2 (smallest/most independent first, largest feature last)

| # | Feature | Priority | Complexity | Est. LOC |
|---|---------|----------|------------|----------|
| 1 | CLI Auto-detect Hardware | P0 | Low | ~30 |
| 2 | Recommend API + Webapp Tab | P1 | Medium | ~180 |
| 3 | Webapp Auto-detect Hardware | P1 | Medium | ~70 |
| 4 | Model Detail Click | P2 | Medium | ~80 |
| 5 | Better Benchmark Error Messages | P2 | Low | ~60 |
| 6 | CLI Calc Subcommand | P3 | Medium | ~55 |
| 7 | CLI Memory Subcommand | P3 | Low | ~35 |
| 8 | CLI Recommend Rich Formatting | P0 | Low | ~25 |

---

## Phase 1: CLI Foundation (Features 1, 5, 8)

**Goal**: Fix CLI friction points -- auto-detect hardware, benchmark error messages, recommend formatting.
**Dependencies**: None (all three tasks are independent and can be implemented in parallel).

### Task 1.1: CLI Auto-detect Hardware (Feature 1)

Make `--hw` optional on all CLI subcommands that require it by falling back to `detect_mac()`.

**Files Modified**: `loca_llama/cli.py`

- [ ] Add `resolve_hw_or_detect()` helper function that accepts `hw_name: str | None`
  - If `hw_name` is provided, use it (backward compatible)
  - If `hw_name` is `None`, call `detect_mac()` from `hardware.py`
  - On successful detection, print `Detected: {key}` in green to stderr
  - On failed detection (`None` return), print error and `sys.exit(1)`:
    `"Could not detect hardware. Use --hw to specify manually. Run 'loca-llama list-hw' for options."`
- [ ] Change `--hw` from `required=True` to `required=False, default=None` in `build_parser()` for subcommands: `check`, `detail`, `max-context`, `recommend`
- [ ] Update `cmd_check()`, `cmd_detail()`, `cmd_max_context()`, `cmd_recommend()` to call `resolve_hw_or_detect(args.hw)` instead of using `args.hw` directly
- [ ] Verify backward compatibility: `--hw "M4 Pro 48GB"` still takes precedence

**Acceptance Criteria**: AC-1.1 through AC-1.5 from PRD (FR-1).
**Estimated LOC**: ~30

### Task 1.2: Better Benchmark Error Messages (Feature 5)

Replace broad `except Exception` in `benchmark_openai_api()` with specific handlers that produce user-friendly messages.

**Files Modified**: `loca_llama/benchmark.py`, `loca_llama/api/routes/benchmark.py`

- [ ] Create `format_benchmark_error(exc: Exception, runtime_name: str, url: str) -> str` function in `benchmark.py`
  - `ConnectionRefusedError` / `urllib.error.URLError [Errno 61]` -> `"Cannot connect to {runtime_name} at {url}. Is the server running?"`
  - `urllib.error.HTTPError` 4xx -> `"{runtime_name} rejected the request: {status_code} {reason}"`
  - `urllib.error.HTTPError` 5xx -> `"{runtime_name} internal error ({status_code}). Try restarting the server."`
  - `socket.timeout` -> `"Request to {runtime_name} timed out after {timeout}s. The model may be too large or the server overloaded."`
  - `json.JSONDecodeError` -> `"Invalid response from {runtime_name}. The server may be misconfigured."`
  - Generic `Exception` as final fallback -> `"An unexpected error occurred."`
- [ ] Replace broad `except Exception` in `benchmark_openai_api()` (line ~163) with specific handlers using this function
- [ ] Log raw exception at DEBUG level for troubleshooting; never surface to user
- [ ] Update API benchmark route to return structured error: `{ "detail": str, "user_message": str, "technical_detail": str }`
- [ ] Update webapp `benchmark.js` to render `user_message` as primary text + expandable `<details>` for `technical_detail`

**Acceptance Criteria**: AC-6.1 through AC-6.6 from PRD (FR-6).
**Estimated LOC**: ~60

### Task 1.3: CLI Recommend Rich Formatting (Feature 8)

Apply the same ANSI formatting used by `check` and `detail` commands to the `recommend` output.

**Files Modified**: `loca_llama/cli.py` (`cmd_recommend()` function)

- [ ] Add colored tier badges using existing `color_rating()` or tier-to-color mapping (FULL_GPU=green, COMFORTABLE=green, TIGHT_FIT=yellow, PARTIAL_OFFLOAD=yellow, WONT_FIT=red)
- [ ] Add memory utilization bar using existing `bar()` function next to memory fraction
- [ ] Add tok/s column with colored thresholds using existing `format_tok_s()` (green >=30, yellow >=10, red <10)
- [ ] Maintain consistent table alignment with other CLI commands

**Acceptance Criteria**: AC-2.1 through AC-2.4 from PRD (FR-2).
**Estimated LOC**: ~25

### Phase 1 Verification

```bash
python -m pytest tests/ -v
loca-llama check                          # Should auto-detect hardware
loca-llama check --hw "M4 Pro 48GB"       # Backward compat, explicit flag
loca-llama recommend --hw "M4 Pro 48GB"   # Rich formatted output
```

---

## Phase 2: New CLI Subcommands (Features 6, 7)

**Goal**: Add VRAM calculator and memory monitoring to the CLI for feature parity with webapp.
**Dependencies**: Phase 1 should be complete (build_parser changes settled), but Features 6 and 7 are independent of each other.

### Task 2.1: CLI Calc Subcommand (Feature 6)

Add `loca-llama calc` for VRAM estimation from the CLI, mirroring the webapp Calculator tab.

**Files Modified**: `loca_llama/cli.py`

- [ ] Add `calc` subcommand to `build_parser()` with arguments:
  - `--model` (str, optional): Model name (substring match via existing `resolve_model`)
  - `--params` (float, optional): Parameter count in billions (for custom models)
  - `--bpw` (float, optional): Bits per weight (for custom models)
  - `--quant` (str, default `"Q4_K_M"`): Quantization format (used with `--model`)
  - `--context` (int, default `4096`): Context length
  - `--layers` (int, optional): Number of layers (for custom models)
  - `--kv-heads` (int, optional): Number of KV heads (for custom models)
  - `--head-dim` (int, default `128`): Head dimension (for custom models)
  - `--hw` (str, optional): Hardware for fit assessment (uses auto-detect if omitted)
- [ ] Implement `cmd_calc(args)` function:
  - When `--model` is provided: look up model params, call `estimate_model_size()` and `estimate_kv_cache_size()` from `analyzer.py`
  - When `--params`/`--bpw` are provided: use manual parameters for custom model estimation
  - When both `--model` and manual params: `--model` takes precedence, warn that manual params are ignored
  - Display: model size, KV cache, overhead, total memory
  - Display: compatible hardware list sorted by memory, with fit tier (when `--hw` is available)
- [ ] Reuse existing formatting functions: `print_header()`, `bar()`, `color_rating()`

**Acceptance Criteria**: AC-7.1 through AC-7.5 from PRD (FR-7).
**Estimated LOC**: ~55

### Task 2.2: CLI Memory Subcommand (Feature 7)

Add `loca-llama memory` for quick memory usage snapshot.

**Files Modified**: `loca_llama/cli.py`

- [ ] Add `memory` subcommand to `build_parser()` (no required arguments)
- [ ] Implement `cmd_memory(args)` function:
  - Check `sys.platform == "darwin"`; if not, print `"Memory monitoring requires macOS."` and `sys.exit(1)`
  - Call `get_memory_sample()` from `memory_monitor.py` for a single snapshot
  - Display: used GB, free GB, total GB
  - Display: usage percentage with colored bar via `bar()` (green <60%, yellow <80%, red >=80%)
  - Display: memory pressure level (normal / warn / critical)
- [ ] Use ANSI color helpers for pressure level coloring

**Acceptance Criteria**: AC-8.1 through AC-8.5 from PRD (FR-8).
**Estimated LOC**: ~35

### Phase 2 Verification

```bash
python -m pytest tests/ -v
loca-llama calc --model "Qwen 2.5 32B" --quant Q4_K_M
loca-llama calc --params 7 --bpw 4.5 --context 8192 --layers 32 --kv-heads 8
loca-llama memory
loca-llama --help                          # Verify new subcommands appear
```

---

## Phase 3: Webapp Enhancements (Features 3, 4)

**Goal**: Add hardware auto-detect and model detail drill-down to the webapp.
**Dependencies**: Feature 3 follows the pattern of Feature 1 but is fully independent. Features 3 and 4 can be implemented in parallel.

### Task 3.1: Webapp Auto-detect Hardware (Feature 3)

Add `GET /api/hardware/detect` endpoint and "Detect My Mac" button to webapp.

**Backend Files Modified**: `loca_llama/api/routes/hardware.py`, `loca_llama/api/schemas.py`

- [ ] Add `HardwareDetectResponse` Pydantic schema:
  - `detected: bool`
  - `name: str | None`
  - `chip: str | None`
  - `memory_gb: int | None`
  - `reason: str | None` (populated on failure)
- [ ] Add `GET /api/hardware/detect` endpoint in `hardware.py`
  - MUST be registered BEFORE the `GET /{name}` route to avoid path conflict
  - Call `detect_mac()` from `hardware.py`
  - Return 200 with `detected=false` on non-Apple (not 404)
  - Return 200 with `detected=true` + hardware details on success
  - Response time target: <100ms

**Frontend Files Modified**: `static/index.html`, `static/js/compat.js`, `static/js/utils.js`, `static/style.css`

- [ ] Add "Detect My Mac" button in the Compatibility tab controls row (next to hardware dropdown)
  - Use `<button class="btn btn-secondary" id="detect-hw-btn" aria-label="Detect my Mac hardware automatically">Detect My Mac</button>`
- [ ] Add `detectHardware()` function in `utils.js` (shared between compat.js and recommend.js)
  - On click: disable button, set text to "Detecting...", set `aria-busy="true"`
  - Call `GET /api/hardware/detect`
  - On success: auto-select matching dropdown option, show green inline feedback "Detected: {name}" for 3s
  - On failure: show muted inline feedback "Detection unavailable -- select manually" for 5s
  - Re-enable button after response
- [ ] Add CSS: `.btn-secondary` class, `.detect-feedback` fade animation, feedback text inline `<span role="status" aria-live="polite">`

**Acceptance Criteria**: AC-4.1 through AC-4.6 from PRD (FR-4).
**Estimated LOC**: ~70

### Task 3.2: Model Detail Click (Feature 4)

Add click-to-detail interaction on model table rows in the Models tab.

**Backend Files Modified**: `loca_llama/api/routes/models.py`, `loca_llama/api/schemas.py`

- [ ] Add `GET /api/models/{name}` endpoint returning full model details:
  - `name`, `family`, `params_billion`, `default_context_length`, `max_context_length`
  - `num_layers`, `num_kv_heads`, `head_dim`, `vocab_size`, `license`, `architecture`
  - Return 404 with `{ "detail": "Model not found" }` for unknown models

**Frontend Files Modified**: `static/js/models.js`, `static/index.html`, `static/style.css`

- [ ] Add `#model-detail` panel div to the Models tab section in `index.html` (below the table)
  - Use same `.detail-panel` + `.detail-grid` pattern as `#compat-detail`
- [ ] Add click handler to model table rows in `models.js`:
  - Clicking a row: call `GET /api/models/{name}`, render detail panel below table
  - Clicking a different row: update detail panel in-place
  - Clicking the same row: toggle detail panel closed
  - Follow `showDetail()` pattern from `compat.js` (lines 206-243)
- [ ] Add CSS: `.active-row` style (`tr.active-row td { background: var(--bg-hover); }`), hover state (`cursor: pointer`, subtle background)
- [ ] Add `role="button"`, `aria-expanded`, `tabindex="0"` to clickable rows
- [ ] Add `aria-live="polite"` to detail panel container

**Acceptance Criteria**: AC-5.1 through AC-5.6 from PRD (FR-5).
**Estimated LOC**: ~80

### Phase 3 Verification

```bash
python -m pytest tests/ -v
python -m loca_llama.api.app              # Start webapp
# Manual: Click "Detect My Mac" on Compatibility tab
# Manual: Click a model row in the Models tab, verify detail panel
```

---

## Phase 4: Recommend Feature (Feature 2)

**Goal**: Full recommend pipeline -- extract shared algorithm, API endpoint, webapp tab.
**Dependencies**: Task 4.1 (API) is independent. Task 4.2 (webapp tab) depends on 4.1. Both benefit from Feature 3 (auto-detect button) being complete.

### Task 4.1: Recommend API Endpoint

Extract the recommend algorithm from `cmd_recommend()` into a shared function and expose via API.

**Files Modified**: `loca_llama/analyzer.py`, `loca_llama/cli.py`, `loca_llama/api/schemas.py`, `loca_llama/api/routes/__init__.py`
**Files Created**: `loca_llama/api/routes/recommend.py`

- [x] Extract recommend algorithm from `cmd_recommend()` in `cli.py` into a reusable function in `analyzer.py`:
  - `recommend_models(hw_key: str, use_case: str = "general", top_n: int = 8) -> list[dict]`
  - Algorithm: iterate quant formats in quality order (`Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q8_0`, `Q3_K_L`), call `analyze_all()` per quant, filter models at <=90% memory utilization, deduplicate by model name (best quant wins), sort by params descending, return top N
- [x] Refactor `cmd_recommend()` in `cli.py` to call the shared `recommend_models()` function
- [x] Add Pydantic schemas in `schemas.py`:
  - `RecommendRequest`: `hardware_name: str`, `use_case: str` (validated against enum `["general", "coding", "reasoning", "small", "large-context"]`)
  - `RecommendItem`: extends existing fields with `rank: int`, `max_context_k: int`
  - `RecommendResponse`: `recommendations: list[RecommendItem]`, `count: int`, `hardware: str`, `use_case: str`
- [x] Create `api/routes/recommend.py`:
  - `POST /api/recommend` -- validate input, call `recommend_models()`, return `RecommendResponse`
  - Invalid `hardware_name` returns HTTP 400
  - Invalid `use_case` returns HTTP 400
  - Empty results returns 200 with empty list (not 404)
- [x] Register router in `api/routes/__init__.py` via `all_routers` list

**Acceptance Criteria**: AC-3.1 through AC-3.4, AC-3.7 from PRD (FR-3).
**Estimated LOC**: ~100

### Task 4.2: Recommend Webapp Tab

Add the Recommend tab to the webapp frontend.

**Files Created**: `static/js/recommend.js`
**Files Modified**: `static/index.html`, `static/js/main.js`, `static/js/tabs.js`

- [x] Add "Recommend" tab button to `<nav>` in `index.html` (position: between Calculator and Memory, or at end)
- [x] Add `<section id="tab-recommend" class="tab-content">` with:
  - Controls row: hardware dropdown, "Detect My Mac" button, use-case dropdown (`General`, `Coding`, `Reasoning`, `Small Models`, `Large Context`), "Get Recommendations" button
  - Results area: `<div id="recommend-results">` (initially empty/placeholder)
  - Detail panel: `<div id="recommend-detail" class="detail-panel">` (hidden by default)
- [x] Create `static/js/recommend.js` module:
  - `initRecommend()`: populate hardware dropdown (same data as Compatibility), wire event handlers
  - `runRecommendations()`: call `POST /api/recommend`, render results table with rank, model name, quant, tier badge, memory %, memory bar, est. tok/s
  - `showRecommendDetail(item)`: render detail panel below table using `detail-grid` pattern (model size, KV cache, overhead, total memory, available, headroom, utilization, speed, GPU layers, max context)
  - Row click: toggle detail panel (same behavior as compat tab)
  - Handle loading, error, and empty states using existing CSS classes (`.loading`, `.error-message`, `.placeholder`)
  - Wire "Detect My Mac" button using shared `detectHardware()` from `utils.js` (Feature 3)
  - "Get Recommendations" button disabled until hardware is selected
- [x] Register tab in `main.js` and `tabs.js`

**Acceptance Criteria**: AC-3.5, AC-3.6 from PRD (FR-3).
**Estimated LOC**: ~80

### Phase 4 Verification

```bash
python -m pytest tests/ -v
python -m loca_llama.api.app              # Start webapp
# Manual: Open Recommend tab, select hardware, pick use case, click "Get Recommendations"
# Manual: Click a recommendation row, verify detail panel
# Cross-validate: Run `loca-llama recommend --hw "M4 Pro 48GB"` and compare results with webapp
```

---

## Phase 5: Testing

**Goal**: Comprehensive test coverage for all new features.
**Dependencies**: Tests can be written alongside each phase (parallel), but this phase ensures completeness.

### Task 5.1: CLI Tests

**Files Created/Modified**: `tests/test_cli.py`

- [x] Test `resolve_hw_or_detect()`:
  - With explicit `--hw` value: returns that value
  - With `None` on Apple Silicon: calls `detect_mac()`, returns detected key
  - With `None` on non-Mac: prints error, exits with code 1
  - Mock `detect_mac()` for deterministic results
- [x] Test `cmd_calc()`:
  - With `--model` flag: produces VRAM breakdown output
  - With `--params`/`--bpw` flags: produces custom model output
  - With both: `--model` wins, warning printed
  - Invalid model name: appropriate error
- [x] Test `cmd_memory()`:
  - On macOS: produces memory stats output
  - On non-macOS: prints error, exits with code 1
  - Mock `get_memory_sample()` for deterministic results
- [x] Test `cmd_recommend()` formatting:
  - Output contains ANSI color codes (tier badges, memory bars)
  - Verify `bar()` and `color_rating()` are used

**Estimated LOC**: ~60

### Task 5.2: API Tests

**Files Created**: `tests/api/test_recommend.py`
**Files Modified**: `tests/api/test_hardware.py`

- [x] Test `POST /api/recommend`:
  - Valid hardware + valid use case: returns recommendations with correct schema
  - Invalid hardware name: returns HTTP 400
  - Invalid use case: returns HTTP 400
  - Empty results case (tiny hardware + large-context use case): returns 200 with empty list
  - Verify recommendation count <= top_n
  - Verify recommendations are sorted by params descending
  - Verify no duplicate model names
- [x] Test `GET /api/hardware/detect`:
  - Mock `detect_mac()` returning a valid spec: returns `detected=true` with hardware details
  - Mock `detect_mac()` returning `None`: returns `detected=false` with reason
  - Verify response time semantics (no heavy computation)
- [x] Test `GET /api/models/{name}`:
  - Valid model name: returns full details
  - Invalid model name: returns HTTP 404

**Estimated LOC**: ~80

### Task 5.3: Benchmark Error Tests

**Files Created/Modified**: `tests/test_benchmark.py`

- [x] Test `format_benchmark_error()`:
  - `ConnectionRefusedError` -> connection refused message
  - `urllib.error.HTTPError` with 400 status -> rejected request message
  - `urllib.error.HTTPError` with 500 status -> internal error message
  - `socket.timeout` -> timeout message
  - `json.JSONDecodeError` -> invalid response message
  - Generic `Exception` -> unexpected error message
- [x] Mock urllib responses to simulate each error type

**Estimated LOC**: ~40

### Phase 5 Verification

```bash
python -m pytest tests/ -v --cov=loca_llama --cov-report=term-missing
# Target: 80%+ coverage for api/ routes, 70%+ overall
```

---

## Task Dependencies

```
Phase 1 (independent, all parallel):
  Task 1.1 (CLI auto-detect) ─────────────┐
  Task 1.2 (benchmark errors) ────────────┤
  Task 1.3 (recommend format) ────────────┘
                                            │
Phase 2 (after Phase 1, parallel):          v
  Task 2.1 (CLI calc) ───────────────────┐
  Task 2.2 (CLI memory) ────────────────┘
                                            │
Phase 3 (independent of Phase 2, parallel): v
  Task 3.1 (webapp detect) ─────────────┐
  Task 3.2 (model detail) ──────────────┘
                                            │
Phase 4 (sequential):                       v
  Task 4.1 (recommend API) ──────> Task 4.2 (recommend tab)
                                            │
Phase 5 (parallel with each phase):         v
  Task 5.1 (CLI tests)
  Task 5.2 (API tests)
  Task 5.3 (benchmark tests)
```

---

## Agent Assignments

| Task | Agent | Parallel? |
|------|-------|-----------|
| Task 1.1-1.3 | senior-software-engineer | Yes (all 3 in parallel) |
| Task 2.1-2.2 | senior-software-engineer | Yes (parallel) |
| Task 3.1-3.2 | fullstack-feature-builder | Yes (parallel) |
| Task 4.1 | senior-software-engineer | No (sequential) |
| Task 4.2 | fullstack-feature-builder | After 4.1 |
| Task 5.1-5.3 | test-writer | Parallel with each phase |

---

## Files Summary

### New Files (2)

| File | Purpose |
|------|---------|
| `loca_llama/api/routes/recommend.py` | POST /api/recommend endpoint |
| `static/js/recommend.js` | Recommend tab frontend module |

### Modified Files (9+)

| File | Changes |
|------|---------|
| `loca_llama/cli.py` | Auto-detect fallback, recommend formatting, `calc` subcommand, `memory` subcommand |
| `loca_llama/benchmark.py` | `format_benchmark_error()` function, specific exception handlers |
| `loca_llama/analyzer.py` | Extract `recommend_models()` shared function |
| `loca_llama/api/schemas.py` | `HardwareDetectResponse`, `RecommendRequest`, `RecommendResponse`, `RecommendItem` |
| `loca_llama/api/routes/hardware.py` | `GET /api/hardware/detect` endpoint |
| `loca_llama/api/routes/models.py` | `GET /api/models/{name}` detail endpoint |
| `loca_llama/api/routes/benchmark.py` | Structured error responses |
| `loca_llama/api/routes/__init__.py` | Register recommend router |
| `static/index.html` | Recommend tab, detect button, model detail panel |
| `static/style.css` | `.btn-secondary`, `.detect-feedback`, `.active-row`, `.error-details` |
| `static/js/compat.js` | Wire detect button |
| `static/js/models.js` | Row click handler, detail panel rendering |
| `static/js/benchmark.js` | Friendly error messages + expandable details |
| `static/js/utils.js` | Shared `detectHardware()` function |
| `static/js/main.js` | Import and init recommend module |
| `static/js/tabs.js` | Register recommend tab |

### Test Files (3)

| File | Coverage |
|------|----------|
| `tests/test_cli.py` | Auto-detect, calc, memory, recommend formatting |
| `tests/api/test_recommend.py` | POST /api/recommend |
| `tests/test_benchmark.py` | `format_benchmark_error()` |
| `tests/api/test_hardware.py` (extend) | GET /api/hardware/detect |

---

## Estimates

| Metric | Value |
|--------|-------|
| **Total production LOC** | ~400 |
| **Total test LOC** | ~180 |
| **New files** | 2 production + 1-3 test files |
| **Modified files** | ~16 |
| **Phases** | 5 |
| **Overall complexity** | Medium (each individual task is Low-Medium) |
| **New dependencies** | Zero |
| **Breaking changes** | Zero |

---

## Verification Commands

After each phase:

```bash
python -m pytest tests/ -v                  # Run all tests
python -m loca_llama.cli --help             # Verify CLI subcommands
python -m loca_llama.api.app                # Start webapp, manual check
```

Phase-specific:

| Phase | Command | Expected Result |
|-------|---------|-----------------|
| 1 | `loca-llama check` (no --hw) | Auto-detects hardware, prints "Detected: ..." |
| 1 | `loca-llama recommend --hw "M4 Pro 48GB"` | Rich formatted output with bars and colors |
| 2 | `loca-llama calc --model "Qwen 2.5 32B" --quant Q4_K_M` | VRAM breakdown table |
| 2 | `loca-llama calc --params 7 --bpw 4.5 --context 8192 --layers 32 --kv-heads 8` | Custom model estimate |
| 2 | `loca-llama memory` | Memory stats with colored bar |
| 3 | Webapp: click "Detect My Mac" button | Hardware dropdown auto-selects |
| 3 | Webapp: click model row in Models tab | Detail panel appears below table |
| 4 | Webapp: open Recommend tab, select hardware + use case | Recommendations table renders |
| 5 | `python -m pytest tests/ -v --cov=loca_llama` | 80%+ coverage for api/ routes |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `detect_mac()` returns `None` on unrecognized Apple Silicon variant | Low | Medium | Already handles variant keys; add test with mocked `sysctl` output |
| Recommend algorithm diverges between CLI and API after extraction | Low | High | Single shared function in `analyzer.py`; add cross-validation test |
| "Detect My Mac" fails when server and browser on different machines | Medium | Low | Expected behavior for localhost tool; return `detected=false` with explanation |
| Tab count growing to 8 causes nav overflow on mobile | Low | Low | Existing horizontal scroll handles this; 480px media query reduces padding |
| Memory subcommand fails on non-macOS | Medium | Low | Check `sys.platform == "darwin"`, exit cleanly with explanatory message |

---

## Open Questions (Resolved)

1. **Should Recommend tab auto-detect on load?** -- Yes, auto-detect on first load, let user override. Show subtle "Auto-detected: {name}" label.
2. **Should `calc` support `--hw` for fit assessment?** -- Yes, when `--hw` is provided or auto-detected, show fit tier and memory bar.
3. **Should recommend algorithm be configurable (`--top N`, `--min-quant`)?** -- Deferred to future iteration. Fixed algorithm covers 90% of use cases.
4. **Should model detail include VRAM per quantization?** -- No. Show static metadata only. VRAM estimates require hardware context.
