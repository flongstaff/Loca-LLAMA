# Loca-LLAMA v2 — Complete LLM Mac Compatibility Tool

## Project Overview

Loca-LLAMA is a comprehensive local LLM management and analysis tool for Apple Silicon Macs. It helps developers and data scientists determine which LLMs they can run on their hardware, with detailed VRAM estimations, runtime comparisons, and performance benchmarking.

**Core Value**: Solve the "will this model fit?" problem before downloading 10GB+ model files.

## Architecture & Scope

This is a **Python 3.11+ CLI/TUI/Webapp project** with three interfaces:
- **CLI** (`loca-llama` command): Non-interactive analysis and queries
- **TUI** (`loca-llama-ui` command): Interactive terminal UI with navigation and browsing
- **Webapp** (new): Local web interface for accessibility (FastAPI + vanilla HTML/CSS/JS)

**Key Constraint**: Core analysis logic uses **zero external dependencies** (pure stdlib). Web framework (FastAPI) is UI layer only, justified by scope addition.

## Tech Stack

- **Language**: Python 3.11+ (type hints required)
- **Package Manager**: pip / setuptools
- **Build System**: setuptools (pyproject.toml)
- **CLI Framework**: Pure stdlib (argparse)
- **TUI Framework**: Pure stdlib (curses)
- **Webapp Framework**: FastAPI + uvicorn (async-native, modern Python patterns)
- **Testing**: pytest (to be added)
- **Deployment**: Local-only (localhost binding, no network exposure)

## Core Modules (13 total, zero TODOs/FIXMEs)

| Module | Purpose | Reusable for Webapp |
|--------|---------|---------------------|
| `analyzer.py` | VRAM estimation & compatibility logic | ✅ Yes (core business logic) |
| `hardware.py` | Apple Silicon specs database (33 configs) | ✅ Yes (data layer) |
| `models.py` | LLM model definitions (50+) | ✅ Yes (data layer) |
| `quantization.py` | 13 GGUF quantization formats | ✅ Yes (data layer) |
| `runtime.py` | LM Studio & llama.cpp connectors | ✅ Yes (async-ready) |
| `memory_monitor.py` | Background memory sampling (macOS vm_stat) | ✅ Yes (needs lifecycle mgmt) |
| `templates.py` | Command templates & presets | ✅ Yes (data layer) |
| `hf_templates.py` | HuggingFace live config fetching | ✅ Yes (already async-capable) |
| `scanner.py` | Local model detection | ✅ Yes (directory scanning) |
| `hub.py` | HuggingFace API integration | ✅ Yes (search, metadata) |
| `benchmark.py` | Runtime performance benchmarking | ✅ Yes (needs async wrapping) |
| `cli.py` | CLI entry point | ⚠️ Separate from webapp routes |
| `interactive.py` | TUI entry point (48 KB) | ⚠️ Separate from webapp routes |

## Entry Points

```toml
[project.scripts]
loca-llama = "loca_llama.cli:main"
loca-llama-ui = "loca_llama.interactive:main_interactive"
# New: web server will use `loca-llama-web` or FastAPI run command
```

## Build & Development Commands

```bash
# Install in editable mode
pip install -e .

# Run CLI
loca-llama --help
loca-llama check --hardware "M4 Pro" --vram 48

# Run TUI
loca-llama-ui

# Run webapp (after implementation)
python -m loca_llama.api.app  # or: uvicorn loca_llama.api.app:app --reload

# Testing
python -m pytest tests/ -v          # Run all tests
python -m pytest tests/api/ -v      # Run webapp tests only

# Type checking
python -m mypy loca_llama/          # When added

# Linting
python -m ruff check loca_llama/    # When added
```

## Directory Structure

```
loca-llama/
├── loca_llama/                    # Main package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                     # CLI entry point
│   ├── interactive.py             # TUI entry point
│   ├── api/                       # NEW: Webapp layer
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app + lifespan setup
│   │   ├── models.py              # Pydantic request/response models
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── compatibility.py   # GET /api/check, /api/models
│   │       ├── benchmark.py       # POST /api/benchmark
│   │       ├── scanner.py         # GET /api/scan, /api/hf-search
│   │       └── templates.py       # GET /api/templates, /api/model/{id}
│   ├── analyzer.py                # Core VRAM estimation (reused by API)
│   ├── hardware.py                # Hardware specs database
│   ├── models.py                  # Model definitions
│   ├── quantization.py            # Quantization formats
│   ├── scanner.py                 # Local model detection
│   ├── hub.py                     # HuggingFace API
│   ├── benchmark.py               # Benchmarking logic
│   ├── runtime.py                 # Runtime connectors (LM Studio, llama.cpp)
│   ├── memory_monitor.py          # Memory tracking
│   ├── templates.py               # Templates & presets
│   └── hf_templates.py            # HF config parsing
├── static/                         # NEW: Webapp frontend files
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                          # NEW: Test suite
│   ├── __init__.py
│   ├── test_analyzer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_compatibility.py
│   │   ├── test_benchmark.py
│   │   └── test_scanner.py
│   └── conftest.py                # pytest fixtures
├── .claude/                        # Claude Code config
│   ├── CLAUDE.md                   # This file
│   ├── settings.json
│   └── rules/
│       ├── python-async.md
│       ├── fastapi-patterns.md
│       └── testing-strategy.md
├── rpi/                            # RPI documentation
│   └── loca-llama-v2/
│       ├── REQUEST.md              # Feature request
│       ├── research/
│       │   └── RESEARCH.md         # Research verdict: CONDITIONAL GO
│       └── plan/
│           └── PLAN.md             # Detailed implementation plan (next)
├── .github/workflows/              # GitHub Actions
├── README.md
├── pyproject.toml                  # Build config (update with FastAPI dep)
└── LICENSE (MIT)
```

## Key Architecture Decisions

**1. Core Logic → API Routes Pattern**
- All business logic (analyzer, hardware, models, etc.) stays in their original modules
- API routes import these modules and call functions, returning JSON responses
- CLI/TUI unchanged — they can also call these same modules
- **Benefit**: Single source of truth, no duplication, backward compatible

**2. FastAPI for Webapp**
- Modern async/await patterns match Python 3.11+ conventions
- Built-in OpenAPI documentation and request validation (Pydantic)
- Easy background task integration for long-running benchmarks
- Lightweight ASGI server (uvicorn), zero extra boilerplate vs Flask
- **Local-only**: Bind to localhost:8000, no network exposure

**3. Memory Monitor Lifecycle Management**
- Background thread sampling (macOS `vm_stat`) needs proper startup/shutdown
- Use FastAPI lifespan context manager (`@app.lifespan`)
- Start on app startup, clean up on shutdown
- **Prevents**: Memory leaks, orphaned processes

**4. Benchmarking Async Integration**
- Currently synchronous in TUI (blocks UI for 30s+)
- Webapp uses FastAPI background tasks or async subprocess
- Return job ID immediately, client polls `/api/benchmark/{job_id}` for results
- **UX**: Non-blocking, stream results as they arrive

## Code Standards

@imports: /Users/flong/.claude/rules/python-standards.md

**Python-Specific**:
- Type hints on all function signatures (not optional)
- Use `pathlib.Path` over `os.path`
- Prefer f-strings for formatting
- Dataclasses or Pydantic models for structured data
- Async/await with try/except for async operations
- Follow existing project conventions (FastAPI, pytest)

**For Webapp**:
- FastAPI route handlers must wrap in try/catch
- Validate request body/params at boundary before processing
- Return consistent error shape: `{ error: string, details?: unknown }`
- Never expose internal error messages or stack traces to clients
- Use dependency injection for shared instances (runtime connectors, memory monitor)

**For Testing**:
- Unit tests for all api/ routes
- E2E tests for critical workflows (analyze → benchmark → results)
- Fixtures for hardware specs, models, mock runtimes
- No hardcoded test data duplicating production constants

## Dependencies

**Production (Current)**:
- None — pure Python stdlib

**Production (After Webapp)**:
- `fastapi` (web framework)
- `uvicorn` (ASGI server)
- `pydantic` (included with FastAPI, request validation)
- Mark in pyproject.toml as optional: `extras = { web = ["fastapi", "uvicorn"] }`

**Development (To Add)**:
- `pytest` (testing framework)
- `pytest-asyncio` (async test support)
- `ruff` (linting, formatting)
- `mypy` (type checking, optional)

## Feature Scope

**Existing (v1, 90% complete)**:
- ✅ Model compatibility analysis with VRAM tiers
- ✅ Hardware database (33 Apple Silicon configs)
- ✅ Model templates with recommended settings
- ✅ Runtime connectors (LM Studio, llama.cpp)
- ✅ Memory monitoring
- ✅ HuggingFace integration (live config fetching)
- ✅ Benchmarking suite
- ✅ Interactive TUI

**New (Webapp, Conditional GO)**:
- 🆕 FastAPI web server (local-only)
- 🆕 Single-page webapp (HTML/CSS/JS)
- 🆕 Reusable API layer
- 🆕 Test suite (pytest)

**Explicitly NOT Included**:
- ❌ Ollama support (per requirements)
- ❌ Cloud deployment
- ❌ Network authentication
- ❌ Remote model serving

## Testing Strategy

- **Unit**: Test individual analyzer functions, data transformations
- **Integration**: Test API routes calling existing business logic
- **E2E**: Critical user workflows (check compatibility → view models → benchmark)
- **Coverage Target**: 80%+ for api/ routes, 70%+ overall
- **Framework**: pytest + pytest-asyncio for async route testing

## CI/CD

**Current**: GitHub Actions configured for Node/Docker (irrelevant for Python)

**To Add**:
- Python 3.11+ test matrix on macOS (M-series runners when available)
- Linting (ruff), type checking (mypy)
- Build verification (pip install -e . succeeds)

## Success Criteria (From RESEARCH.md)

- ✅ Existing CLI remains fully functional (backward compatible)
- 🆕 Webapp serves all core features (compatibility check, benchmarking, discovery)
- 🆕 Local-only deployment (localhost:8000, no cloud)
- 🆕 VRAM estimation, templates, benchmark results identical between CLI and webapp
- 🆕 Single-page app provides 80%+ feature parity with CLI/TUI
- 🆕 Code structure enables easy addition of MLX/Ollama connectors

## Conditions for Proceeding

1. ✅ Use FastAPI + uvicorn (add to optional dependencies)
2. ✅ Refactor into `api/` subpackage to avoid coupling
3. ✅ Add pytest test suite for webapp routes
4. ⚠️ Test on M1/M4 hardware before release (memory monitoring is platform-specific)

---

**Last Updated**: 2026-03-02
**Status**: Initialization phase (CLAUDE.md creation)
**Next Step**: Run `/rpi:plan "loca-llama-v2"` for detailed implementation planning
