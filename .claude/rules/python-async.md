# Python Async/Await Patterns for Loca-LLAMA Webapp

Applies to: `loca_llama/api/**/*.py`

## FastAPI Route Handlers

All FastAPI route handlers must be async functions:

```python
from fastapi import FastAPI, HTTPException
from typing import Any

app = FastAPI()

# ✅ Correct
@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    try:
        # Sync code (library calls) runs fine in async context
        from loca_llama.models import MODEL_DB
        return {"models": list(MODEL_DB.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ❌ Wrong
@app.get("/api/models")
def get_models():  # Missing async
    return {"models": []}
```

## Wrapping Blocking Operations

Long-running I/O or CPU-bound work must use `asyncio.to_thread()` or background tasks:

```python
import asyncio

# ✅ Correct: Benchmark in background
@app.post("/api/benchmark")
async def benchmark_model(model_id: str) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    # Launch in background, return immediately
    asyncio.create_task(run_benchmark(model_id, job_id))
    return {"job_id": job_id}

async def run_benchmark(model_id: str, job_id: str) -> None:
    try:
        result = await asyncio.to_thread(benchmark, model_id)
        # Store result for client polling
        BENCHMARK_RESULTS[job_id] = result
    except Exception as e:
        BENCHMARK_RESULTS[job_id] = {"error": str(e)}

# ❌ Wrong: Blocks entire server
@app.post("/api/benchmark")
async def benchmark_model(model_id: str) -> dict:
    return await benchmark(model_id)  # 30s+ blocks all requests
```

## Memory Monitor Lifecycle

Use FastAPI lifespan context manager to manage background threads:

```python
from contextlib import asynccontextmanager
from loca_llama.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    monitor.start()
    yield
    # Shutdown
    monitor.stop()

app = FastAPI(lifespan=lifespan)
```

## Exception Handling

Always wrap async operations in try/except:

```python
# ✅ Correct
@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str) -> dict:
    try:
        info = await asyncio.to_thread(fetch_model_info, model_id)
        return info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fetch model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ❌ Wrong: Unhandled exception crashes request
@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str) -> dict:
    return await asyncio.to_thread(fetch_model_info, model_id)
```

## Testing Async Routes

Use pytest-asyncio:

```python
import pytest
from httpx import AsyncClient
from loca_llama.api.app import app

@pytest.mark.asyncio
async def test_get_models():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/models")
        assert response.status_code == 200
        assert "models" in response.json()
```

## Key Rules

1. All FastAPI handlers are `async def`
2. Blocking operations use `asyncio.to_thread()` or background tasks
3. Never await on synchronous functions; use `to_thread()` instead
4. Memory monitor and background services managed via lifespan context
5. Every async operation wrapped in try/except with proper error handling
6. No nested event loops — use `asyncio.create_task()` for concurrent work
