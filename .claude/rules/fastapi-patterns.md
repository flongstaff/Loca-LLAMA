# FastAPI Patterns for Loca-LLAMA Webapp

Applies to: `loca_llama/api/**/*.py`

## Project Structure

```
loca_llama/api/
├── __init__.py
├── app.py                 # FastAPI app initialization + lifespan
├── models.py              # Pydantic request/response models
└── routes/
    ├── __init__.py
    ├── compatibility.py   # GET /api/check, /api/models, /api/model/{id}
    ├── benchmark.py       # POST /api/benchmark, GET /api/benchmark/{job_id}
    ├── scanner.py         # GET /api/scan, /api/hf-search
    └── templates.py       # GET /api/templates
```

## App Initialization Pattern

**`loca_llama/api/app.py`**:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from loca_llama.memory_monitor import MemoryMonitor
from loca_llama.api.routes import compatibility, benchmark, scanner, templates

# Global instance shared across routes
monitor = MemoryMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    monitor.start()
    print("Memory monitor started")
    yield
    # Shutdown
    monitor.stop()
    print("Memory monitor stopped")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Loca-LLAMA API",
        description="Local LLM management and benchmarking tool",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(compatibility.router, prefix="/api", tags=["compatibility"])
    app.include_router(benchmark.router, prefix="/api", tags=["benchmark"])
    app.include_router(scanner.router, prefix="/api", tags=["scanner"])
    app.include_router(templates.router, prefix="/api", tags=["templates"])

    # Serve static frontend
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app

app = create_app()
```

## Route Handler Pattern

**`loca_llama/api/routes/compatibility.py`**:

```python
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any
from loca_llama.analyzer import analyze_compatibility
from loca_llama.hardware import HARDWARE_DB
from loca_llama.models import MODEL_DB

router = APIRouter()

class CompatibilityRequest(BaseModel):
    hardware_name: str
    vram_gb: int

class CompatibilityResponse(BaseModel):
    compatible_models: list[str]
    partially_usable: list[str]
    incompatible: list[str]

@router.post("/check")
async def check_compatibility(req: CompatibilityRequest) -> CompatibilityResponse:
    """Check which models are compatible with user's hardware."""
    try:
        # Validate hardware exists
        if req.hardware_name not in HARDWARE_DB:
            raise HTTPException(
                status_code=400,
                detail=f"Hardware '{req.hardware_name}' not found"
            )

        # Call core analysis logic
        result = analyze_compatibility(req.hardware_name, req.vram_gb)

        return CompatibilityResponse(
            compatible_models=result["full_gpu"],
            partially_usable=result["partial"],
            incompatible=result["wont_fit"],
        )
    except HTTPException:
        raise  # Re-raise client errors
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List all known models with metadata."""
    try:
        models = [
            {
                "id": model.id,
                "family": model.family,
                "size_b": model.size_billions,
                "context": model.default_context,
                "quantizations": model.supported_quantizations,
            }
            for model in MODEL_DB.values()
        ]
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list models")

@router.get("/model/{model_id}")
async def get_model_details(model_id: str) -> dict[str, Any]:
    """Get detailed info about a specific model."""
    try:
        if model_id not in MODEL_DB:
            raise HTTPException(status_code=404, detail="Model not found")

        model = MODEL_DB[model_id]
        return {
            "id": model.id,
            "family": model.family,
            "size_b": model.size_billions,
            "vram_by_quantization": model.vram_estimates,
            "recommended_context": model.default_context,
            "supported_quantizations": model.supported_quantizations,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Error Response Pattern

All errors must return consistent shape:

```python
# ✅ Correct
raise HTTPException(
    status_code=400,
    detail="Invalid hardware name"
)

# Response: {"detail": "Invalid hardware name"}

# ❌ Wrong
raise HTTPException(
    status_code=500,
    detail=f"Error in analyze_compatibility: {e}",  # Exposes internal function names
)
```

## Dependency Injection for Shared State

```python
from fastapi import Depends, FastAPI

class AppState:
    memory_monitor = MemoryMonitor()
    runtime_connectors = {}

app_state = AppState()

async def get_app_state() -> AppState:
    return app_state

@router.get("/memory")
async def get_memory_status(state: AppState = Depends(get_app_state)) -> dict:
    """Get current memory usage from shared monitor."""
    return state.memory_monitor.current_stats()
```

## Background Task Pattern

```python
import uuid
from fastapi import BackgroundTasks

BENCHMARK_RESULTS = {}

@router.post("/benchmark")
async def start_benchmark(
    model_id: str,
    background_tasks: BackgroundTasks
) -> dict[str, str]:
    """Start benchmarking a model (non-blocking)."""
    job_id = str(uuid.uuid4())

    # Validate model exists
    if model_id not in MODEL_DB:
        raise HTTPException(status_code=404, detail="Model not found")

    # Add task to background queue
    background_tasks.add_task(run_benchmark, model_id, job_id)

    return {"job_id": job_id, "status": "queued"}

def run_benchmark(model_id: str, job_id: str) -> None:
    """Long-running benchmark (blocks background thread, not request)."""
    try:
        from loca_llama.benchmark import benchmark
        result = benchmark(model_id)
        BENCHMARK_RESULTS[job_id] = {"status": "complete", "result": result}
    except Exception as e:
        BENCHMARK_RESULTS[job_id] = {"status": "error", "error": str(e)}

@router.get("/benchmark/{job_id}")
async def get_benchmark_result(job_id: str) -> dict:
    """Poll for benchmark results."""
    if job_id not in BENCHMARK_RESULTS:
        raise HTTPException(status_code=404, detail="Job not found")

    return BENCHMARK_RESULTS[job_id]
```

## Validation Pattern

Always validate at route boundary:

```python
from pydantic import BaseModel, Field, validator

class BenchmarkRequest(BaseModel):
    model_id: str = Field(..., min_length=1, description="Model identifier")
    runtime: str = Field(default="auto", description="Runtime: lm-studio, llama.cpp, or auto")
    duration_seconds: int = Field(default=30, ge=5, le=300, description="Benchmark duration")

    @validator("runtime")
    def validate_runtime(cls, v):
        if v not in ["auto", "lm-studio", "llama.cpp"]:
            raise ValueError("Invalid runtime")
        return v

# FastAPI automatically validates request before handler is called
@router.post("/benchmark")
async def benchmark(req: BenchmarkRequest) -> dict:
    # req is guaranteed to have valid fields
    pass
```

## Key Rules

1. **Routers**: One module per endpoint group (compatibility, benchmark, scanner, templates)
2. **Pydantic Models**: All request/response types defined explicitly
3. **Error Handling**: Wrap all handlers in try/except, return HTTPException for client errors
4. **Validation**: Use Pydantic Field validators, validate at boundary
5. **Async**: All handlers are `async def`
6. **Shared State**: Use dependency injection for memory monitor, runtime connectors
7. **Long Operations**: Use background tasks or asyncio.create_task()
8. **Logging**: Always log exceptions with context (endpoint, user input, error details)
9. **No Stack Traces**: Never expose internal error messages to client
10. **Lifespan**: Use context manager for startup/shutdown
