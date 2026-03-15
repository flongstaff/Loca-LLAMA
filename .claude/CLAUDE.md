# Loca-LLAMA

Local LLM compatibility tool for Apple Silicon. Determines which models fit your hardware with VRAM estimation, runtime comparison, and benchmarking.

## Tech Stack
- **Language**: Python 3.11+ (type hints required)
- **Core**: Zero external dependencies (pure stdlib)
- **Webapp**: FastAPI + uvicorn (optional, `pip install -e ".[web]"`)
- **Testing**: pytest + pytest-asyncio
- **Interfaces**: CLI (`loca-llama`), TUI (`loca-llama-ui`), Webapp

## Commands
```bash
pip install -e .                              # Install editable
loca-llama --help                             # CLI
loca-llama-ui                                 # TUI
uvicorn loca_llama.api.app:app --reload       # Webapp
python -m pytest tests/ -v                    # Tests
```

## Architecture
- Core modules (analyzer, hardware, models, quantization, runtime, scanner, hub, benchmark, templates, memory_monitor) are shared across all interfaces
- API routes (`loca_llama/api/routes/`) import core modules and return JSON
- CLI/TUI unchanged — backward compatible
- Benchmarks use background tasks (non-blocking)
- Memory monitor uses FastAPI lifespan for lifecycle

## Conventions
- All business logic in core modules, not in routes
- FastAPI handlers: `async def`, try/except, no stack traces to client
- Pydantic models for request/response validation
- Local-only deployment (localhost binding)

@rules/fastapi-patterns.md
@rules/python-async.md
@rules/testing-strategy.md
