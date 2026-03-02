# Loca-LLAMA v2 Webapp -- Technical Design Document

## 1. High-Level Architecture

### System Diagram

```
Browser (localhost:8000)
  |
  |  HTTP (JSON + static files)
  v
+------------------------------------------------------+
|  FastAPI Application  (loca_llama/api/app.py)        |
|                                                       |
|  Lifespan: MemoryMonitor start/stop                  |
|  AppState: monitor, benchmark jobs, connectors       |
|                                                       |
|  +------------------+  +------------------+          |
|  | /api/compat/*    |  | /api/benchmark/* |          |
|  | /api/hardware/*  |  | /api/memory/*    |          |
|  | /api/models/*    |  | /api/scanner/*   |          |
|  | /api/templates/* |  | /api/hub/*       |          |
|  | /api/runtime/*   |  |                  |          |
|  +--------+---------+  +--------+---------+          |
|           |                      |                    |
|  +--------v----------------------v---------+         |
|  |       Core Modules (unchanged)          |         |
|  |  analyzer  hardware  models  quant      |         |
|  |  templates  benchmark  scanner  hub     |         |
|  |  runtime  memory_monitor  hf_templates  |         |
|  +-----------------------------------------+         |
|                                                       |
|  StaticFiles: /static/  (index.html, style.css, app.js)
+------------------------------------------------------+
```

### Data Flow

1. **Frontend** makes `fetch()` calls to `/api/*` endpoints
2. **Route handlers** (async) validate input via Pydantic models
3. **Route handlers** call **core module** functions (pure stdlib)
4. For blocking I/O (benchmark, scanner, hub, hf_templates), handlers use `asyncio.to_thread()` or background tasks
5. **Core modules** return stdlib dataclasses
6. **Route handlers** convert dataclasses to Pydantic response models and return JSON
7. **Frontend** renders the JSON response into the DOM

### Key Principle: Thin API Layer

The API layer (`loca_llama/api/`) is strictly a translation layer. It:
- Validates input (Pydantic)
- Calls core functions (no business logic in routes)
- Converts dataclass results to JSON-serializable Pydantic models
- Handles errors at the boundary

No core module is modified. The CLI, TUI, and webapp all call the same functions.

---

## 2. API Contracts

### 2.1 Error Response Shape

All errors follow FastAPI's `HTTPException` format:

```json
{
  "detail": "Human-readable error message"
}
```

For validation errors (422), FastAPI auto-generates:

```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 2.2 Hardware Endpoints

#### `GET /api/hardware`

List all Apple Silicon configurations.

**Response** `200`:
```json
{
  "hardware": [
    {
      "name": "M4 Pro 48GB",
      "chip": "M4 Pro",
      "cpu_cores": 12,
      "gpu_cores": 16,
      "neural_engine_cores": 16,
      "memory_gb": 48,
      "memory_bandwidth_gbs": 273.0,
      "gpu_tflops": 7.5,
      "usable_memory_gb": 44.0
    }
  ],
  "count": 33
}
```

Source: `hardware.APPLE_SILICON_SPECS`

#### `GET /api/hardware/{name}`

Get a single hardware spec by name.

**Response** `200`: Single hardware object (same shape as list item above).
**Response** `404`: `{"detail": "Hardware 'X' not found"}`

---

### 2.3 Model Endpoints

#### `GET /api/models`

List all known LLM models.

**Query params**:
- `family` (optional, str): Filter by family (e.g., "Llama", "Qwen")

**Response** `200`:
```json
{
  "models": [
    {
      "name": "Qwen 2.5 32B",
      "family": "Qwen",
      "params_billion": 32.76,
      "default_context_length": 4096,
      "max_context_length": 131072,
      "num_layers": 64,
      "num_kv_heads": 8,
      "head_dim": 128,
      "license": "Apache 2.0"
    }
  ],
  "count": 50,
  "families": ["Command", "CodeLlama", "DeepSeek", "Falcon", "Gemma", "Llama", "Mistral", "Mixtral", "Nous", "Phi", "Qwen", "StarCoder", "Yi"]
}
```

Source: `models.MODELS`

---

### 2.4 Quantization Endpoints

#### `GET /api/quantizations`

List all quantization formats.

**Response** `200`:
```json
{
  "formats": [
    {
      "name": "Q4_K_M",
      "bits_per_weight": 4.85,
      "quality_rating": "Good",
      "description": "4-bit quantization medium (k-quant). Recommended default for most users."
    }
  ],
  "recommended": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]
}
```

Source: `quantization.QUANT_FORMATS`, `quantization.RECOMMENDED_FORMATS`

---

### 2.5 Compatibility Analysis Endpoints

#### `POST /api/analyze`

Analyze a single model + quant + context combination against hardware.

**Request body**:
```json
{
  "hardware_name": "M4 Pro 48GB",
  "model_name": "Qwen 2.5 32B",
  "quant_name": "Q4_K_M",
  "context_length": null
}
```

- `hardware_name` (required, str): Key from `APPLE_SILICON_SPECS`
- `model_name` (required, str): `LLMModel.name` value
- `quant_name` (required, str): Key from `QUANT_FORMATS`
- `context_length` (optional, int): Override context; null uses model default

**Response** `200`:
```json
{
  "model_name": "Qwen 2.5 32B",
  "quant_name": "Q4_K_M",
  "context_length": 4096,
  "model_size_gb": 14.85,
  "kv_cache_gb": 0.25,
  "overhead_gb": 1.99,
  "total_memory_gb": 17.09,
  "available_memory_gb": 44.0,
  "headroom_gb": 26.91,
  "fits_in_memory": true,
  "tier": "full_gpu",
  "tier_label": "Full GPU",
  "memory_utilization_pct": 38.8,
  "estimated_tok_per_sec": 15.9,
  "gpu_layers": 64,
  "total_layers": 64,
  "offload_pct": 100.0
}
```

**Response** `400`: Hardware or model or quant not found.

Source: `analyzer.analyze_model()`

#### `POST /api/analyze/all`

Analyze all models (or a filtered subset) against hardware.

**Request body**:
```json
{
  "hardware_name": "M4 Pro 48GB",
  "quant_names": ["Q4_K_M", "Q6_K"],
  "context_length": null,
  "only_fits": true,
  "include_partial": false,
  "family": null
}
```

- `hardware_name` (required, str)
- `quant_names` (optional, list[str]): Defaults to `RECOMMENDED_FORMATS`
- `context_length` (optional, int)
- `only_fits` (optional, bool, default false)
- `include_partial` (optional, bool, default false)
- `family` (optional, str): Filter models by family

**Response** `200`:
```json
{
  "results": [
    { /* same shape as single analyze response */ }
  ],
  "count": 142,
  "hardware": "M4 Pro 48GB",
  "summary": {
    "full_gpu": 38,
    "comfortable": 22,
    "tight_fit": 8,
    "partial_offload": 12,
    "wont_fit": 62
  }
}
```

Source: `analyzer.analyze_all()`

#### `POST /api/analyze/max-context`

Find maximum context length for a model + quant on given hardware.

**Request body**:
```json
{
  "hardware_name": "M4 Pro 48GB",
  "model_name": "Qwen 2.5 32B",
  "quant_name": "Q4_K_M"
}
```

**Response** `200`:
```json
{
  "model_name": "Qwen 2.5 32B",
  "quant_name": "Q4_K_M",
  "max_context_length": 94208,
  "max_context_k": "92K"
}
```

Source: `analyzer.max_context_for_model()`

---

### 2.6 Template Endpoints

#### `GET /api/templates`

List all model templates.

**Response** `200`:
```json
{
  "templates": [
    {
      "model_pattern": "Qwen 2.5 32B",
      "family": "Qwen",
      "quant_24gb": "Q4_K_M",
      "quant_48gb": "Q6_K",
      "quant_64gb": "Q8_0",
      "recommended_ctx": 8192,
      "max_practical_ctx": 65536,
      "temperature": 0.7,
      "top_p": 0.8,
      "top_k": 20,
      "repeat_penalty": 1.05,
      "min_p": 0.05,
      "system_prompt": "You are Qwen...",
      "chat_template": "chatml",
      "llama_cpp_flags": [],
      "notes": "Best bang-for-buck on 48GB...",
      "bench_tok_s_q4": 12.0,
      "bench_tok_s_q8": 7.5,
      "bench_prefill_q4": 150.0
    }
  ],
  "count": 22
}
```

Source: `templates.TEMPLATES`

#### `GET /api/templates/match/{model_name}`

Find the best matching template for a model name.

**Response** `200`: Single template object.
**Response** `404`: `{"detail": "No template found for 'X'"}`

Source: `templates.get_template()`

#### `POST /api/templates/lm-studio-preset`

Generate an LM Studio preset for a model.

**Request body**:
```json
{
  "model_name": "Qwen 2.5 32B"
}
```

**Response** `200`:
```json
{
  "name": "Loca-LLAMA: Qwen 2.5 32B",
  "inference_params": { "temperature": 0.7, "top_p": 0.8, "..." : "..." },
  "context_length": 8192,
  "system_prompt": "You are Qwen..."
}
```

Source: `templates.get_lm_studio_preset()`

#### `POST /api/templates/llama-cpp-command`

Generate a llama.cpp CLI command.

**Request body**:
```json
{
  "model_name": "Qwen 2.5 32B",
  "model_path": "/path/to/model.gguf",
  "context_length": null,
  "n_gpu_layers": -1
}
```

**Response** `200`:
```json
{
  "command": "llama-cli \\\n  -m /path/to/model.gguf \\\n  -c 8192 ..."
}
```

Source: `templates.get_llama_cpp_command()`

---

### 2.7 Scanner Endpoints

#### `GET /api/scanner/local`

Scan for locally downloaded models.

**Query params**:
- `custom_dir` (optional, str): Additional directory to scan

**Response** `200`:
```json
{
  "models": [
    {
      "name": "Qwen2.5-32B-Instruct-Q6_K",
      "path": "/Users/user/.cache/lm-studio/models/.../model.gguf",
      "size_gb": 23.8,
      "format": "gguf",
      "source": "lm-studio",
      "quant": "Q6_K",
      "family": "Qwen",
      "repo_id": "Qwen/Qwen2.5-32B-Instruct-GGUF"
    }
  ],
  "count": 12,
  "total_size_gb": 145.2,
  "sources": {"lm-studio": 8, "huggingface": 3, "mlx-community": 1}
}
```

**Async strategy**: `asyncio.to_thread(scanner.scan_all)` (blocking filesystem I/O)

Source: `scanner.scan_all()`, `scanner.scan_custom_dir()`

---

### 2.8 Hub (HuggingFace Search) Endpoints

#### `GET /api/hub/search`

Search HuggingFace for models.

**Query params**:
- `query` (required, str): Search query
- `limit` (optional, int, default 20, max 50)
- `sort` (optional, str, default "downloads"): "downloads", "likes", "lastModified"
- `format` (optional, str): "gguf", "mlx", or null for all

**Response** `200`:
```json
{
  "results": [
    {
      "repo_id": "Qwen/Qwen2.5-32B-Instruct-GGUF",
      "name": "Qwen2.5-32B-Instruct-GGUF",
      "author": "Qwen",
      "downloads": 150000,
      "likes": 85,
      "tags": ["gguf", "text-generation", "qwen2"],
      "pipeline_tag": "text-generation",
      "is_mlx": false,
      "is_gguf": true,
      "last_modified": "2025-01-15T10:30:00Z"
    }
  ],
  "count": 20,
  "query": "qwen 32b"
}
```

**Async strategy**: `asyncio.to_thread(hub.search_huggingface)` (blocking HTTP)

Source: `hub.search_huggingface()`, `hub.search_gguf_models()`, `hub.search_mlx_models()`

#### `GET /api/hub/files/{repo_id:path}`

Get file listing for a HuggingFace repo.

**Response** `200`:
```json
{
  "repo_id": "Qwen/Qwen2.5-32B-Instruct-GGUF",
  "files": [
    {"filename": "qwen2.5-32b-instruct-q4_k_m.gguf", "size": 19500000000},
    {"filename": "qwen2.5-32b-instruct-q6_k.gguf", "size": 24800000000}
  ]
}
```

Source: `hub.get_model_files()`

#### `GET /api/hub/config/{repo_id:path}`

Fetch HuggingFace model configuration (config.json, generation_config, etc.).

**Query params**:
- `fetch_card` (optional, bool, default true)

**Response** `200`:
```json
{
  "repo_id": "Qwen/Qwen2.5-32B-Instruct",
  "model_type": "qwen2",
  "architecture": "Qwen2ForCausalLM",
  "num_layers": 64,
  "num_attention_heads": 40,
  "num_kv_heads": 8,
  "head_dim": 128,
  "hidden_size": 5120,
  "max_position_embeddings": 131072,
  "vocab_size": 152064,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": null,
  "chat_template": "{% for message in messages %}...",
  "license": "apache-2.0",
  "tags": ["text-generation", "qwen2"]
}
```

**Async strategy**: `asyncio.to_thread(hf_templates.fetch_hf_model_config)` (multiple sequential HTTP calls)

Source: `hf_templates.fetch_hf_model_config()`

---

### 2.9 Runtime Endpoints

#### `GET /api/runtime/status`

Check which runtimes are currently running.

**Response** `200`:
```json
{
  "runtimes": [
    {
      "name": "lm-studio",
      "url": "http://127.0.0.1:1234",
      "models": ["qwen2.5-32b-instruct"],
      "version": null
    },
    {
      "name": "llama.cpp-server",
      "url": "http://127.0.0.1:8080",
      "models": ["(loaded model)"],
      "version": null
    }
  ],
  "count": 2
}
```

**Async strategy**: `asyncio.to_thread(benchmark.detect_all_runtimes)` (blocking HTTP probes)

Source: `benchmark.detect_all_runtimes()`

---

### 2.10 Benchmark Endpoints

#### `POST /api/benchmark/start`

Start a benchmark suite (non-blocking). Returns a job ID for polling.

**Request body**:
```json
{
  "runtime_name": "lm-studio",
  "model_id": "qwen2.5-32b-instruct",
  "prompt_type": "default",
  "num_runs": 3,
  "max_tokens": 200,
  "context_length": 4096
}
```

- `runtime_name` (required): "lm-studio" or "llama.cpp-server"
- `model_id` (required): Model ID as reported by the runtime
- `prompt_type` (optional, default "default"): "default", "coding", "reasoning", "creative"
- `num_runs` (optional, default 3, min 1, max 10)
- `max_tokens` (optional, default 200)
- `context_length` (optional, default 4096)

**Response** `200`:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running"
}
```

**Response** `400`: Runtime not found or not running.

**Async strategy**: `asyncio.create_task()` wrapping `asyncio.to_thread(run_benchmark_suite)`. The task stores results in `AppState.benchmark_jobs`.

Source: `benchmark.run_benchmark_suite()`

#### `GET /api/benchmark/{job_id}`

Poll for benchmark results.

**Response** `200` (running):
```json
{
  "job_id": "a1b2c3d4...",
  "status": "running",
  "progress": {"current_run": 2, "total_runs": 3}
}
```

**Response** `200` (complete):
```json
{
  "job_id": "a1b2c3d4...",
  "status": "complete",
  "runs": [
    {
      "run_number": 1,
      "success": true,
      "tokens_per_second": 11.5,
      "prompt_tokens_per_second": 145.0,
      "time_to_first_token_ms": 280.0,
      "total_time_ms": 17400.0,
      "prompt_tokens": 42,
      "generated_tokens": 200
    }
  ],
  "aggregate": {
    "avg_tok_per_sec": 12.1,
    "min_tok_per_sec": 11.5,
    "max_tok_per_sec": 12.8,
    "avg_prefill_tok_per_sec": 148.0,
    "avg_ttft_ms": 275.0,
    "avg_total_ms": 16800.0,
    "total_tokens_generated": 400,
    "runs": 2
  }
}
```

**Response** `200` (error):
```json
{
  "job_id": "a1b2c3d4...",
  "status": "error",
  "error": "Runtime connection refused"
}
```

**Response** `404`: `{"detail": "Job not found"}`

Source: `benchmark.aggregate_results()`

#### `GET /api/benchmark/prompts`

List available benchmark prompt types.

**Response** `200`:
```json
{
  "prompts": {
    "default": "Write a detailed explanation of...",
    "coding": "Write a Python function that...",
    "reasoning": "A farmer has 17 sheep...",
    "creative": "Write a short story..."
  }
}
```

Source: `benchmark.BENCH_PROMPTS`

---

### 2.11 Memory Monitor Endpoints

#### `GET /api/memory/current`

Get current memory usage snapshot.

**Response** `200`:
```json
{
  "used_gb": 28.5,
  "free_gb": 19.5,
  "total_gb": 48.0,
  "usage_pct": 59.4,
  "pressure": "normal"
}
```

Source: `memory_monitor.MemoryMonitor.get_current()` (from lifespan-managed instance)

#### `GET /api/memory/report`

Get full memory report from monitoring period.

**Response** `200`:
```json
{
  "peak_used_gb": 35.2,
  "baseline_used_gb": 22.0,
  "delta_gb": 13.2,
  "total_gb": 48.0,
  "peak_pct": 73.3,
  "baseline_pct": 45.8,
  "duration_sec": 120.5,
  "sample_count": 241
}
```

Source: `MemoryMonitor._build_report()` (live, without stopping)

---

## 3. In-Memory State Management

No database. All state is in-memory, managed through an `AppState` class using FastAPI dependency injection.

### AppState Design

```python
# loca_llama/api/state.py
import asyncio
import uuid
from dataclasses import dataclass, field
from loca_llama.memory_monitor import MemoryMonitor
from loca_llama.benchmark import BenchmarkResult

@dataclass
class BenchmarkJob:
    """Tracks a running or completed benchmark."""
    job_id: str
    status: str  # "running", "complete", "error"
    runtime_name: str
    model_id: str
    num_runs: int
    current_run: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    error: str | None = None

class AppState:
    """Shared application state, created once at startup."""

    def __init__(self) -> None:
        self.memory_monitor = MemoryMonitor(interval=1.0)
        self.benchmark_jobs: dict[str, BenchmarkJob] = {}
        self._lock = asyncio.Lock()

    def create_benchmark_job(
        self, runtime_name: str, model_id: str, num_runs: int
    ) -> BenchmarkJob:
        job_id = str(uuid.uuid4())
        job = BenchmarkJob(
            job_id=job_id,
            status="running",
            runtime_name=runtime_name,
            model_id=model_id,
            num_runs=num_runs,
        )
        self.benchmark_jobs[job_id] = job
        return job

    def cleanup_old_jobs(self, max_jobs: int = 50) -> None:
        """Remove oldest completed jobs if over limit."""
        completed = [
            (k, v) for k, v in self.benchmark_jobs.items()
            if v.status in ("complete", "error")
        ]
        if len(completed) > max_jobs:
            for k, _ in completed[:len(completed) - max_jobs]:
                del self.benchmark_jobs[k]
```

### State Lifecycle

```python
# In app.py lifespan
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.memory_monitor.start()
    yield
    app_state.memory_monitor.stop()

# Dependency injection
async def get_state() -> AppState:
    return app_state
```

### Why Not a Database?

- This is a **local-only, single-user** tool
- Benchmark results are ephemeral (one session)
- Memory monitor data streams continuously
- No persistence requirement across restarts
- Simplicity: zero config, zero migrations

### Memory Bounds

- `benchmark_jobs` capped at 50 via `cleanup_old_jobs()`
- `MemoryMonitor.samples` grows at ~2/sec; for a 1-hour session that is ~7200 samples at ~100 bytes each = ~720 KB. Acceptable.
- Total state footprint: < 10 MB even under heavy use

---

## 4. Async Integration Strategy

### Problem

The 13 core modules are all synchronous. Several perform blocking I/O:
- **benchmark.py**: HTTP calls to runtimes (30s+ per suite)
- **scanner.py**: Filesystem traversal (< 1s usually, but can be slow with many models)
- **hub.py**: HTTP calls to HuggingFace API (2-15s)
- **hf_templates.py**: 4-5 sequential HTTP calls to HuggingFace (5-20s)
- **runtime.py**: HTTP probes to localhost ports (2-5s)
- **memory_monitor.py**: Already uses background thread (compatible)

### Strategy: `asyncio.to_thread()` for All Blocking Calls

```python
import asyncio

# Pattern for short blocking calls (< 5s)
result = await asyncio.to_thread(scanner.scan_all)

# Pattern for long blocking calls with progress
job = state.create_benchmark_job(...)
asyncio.create_task(_run_benchmark_in_background(state, job, runtime, ...))
```

### Classification of Each Module

| Module | Blocking? | Duration | Strategy |
|--------|-----------|----------|----------|
| `analyzer.py` | No (CPU-bound, fast) | < 10ms | Call directly in async handler |
| `hardware.py` | No (in-memory data) | < 1ms | Call directly |
| `models.py` | No (in-memory data) | < 1ms | Call directly |
| `quantization.py` | No (in-memory data) | < 1ms | Call directly |
| `templates.py` | No (in-memory data) | < 1ms | Call directly |
| `scanner.py` | Yes (filesystem I/O) | 0.1-5s | `asyncio.to_thread()` |
| `hub.py` | Yes (HTTP) | 2-15s | `asyncio.to_thread()` |
| `hf_templates.py` | Yes (HTTP x5) | 5-20s | `asyncio.to_thread()` |
| `benchmark.py` (detect) | Yes (HTTP probes) | 2-5s | `asyncio.to_thread()` |
| `benchmark.py` (suite) | Yes (HTTP, long) | 30-180s | `asyncio.create_task()` + `to_thread()` |
| `runtime.py` (detect) | Yes (HTTP probes) | 2-5s | `asyncio.to_thread()` |
| `memory_monitor.py` | Own thread | Continuous | Lifespan-managed; `get_current()` is thread-safe |

### Benchmark Background Task Pattern

```python
async def _run_benchmark_background(
    state: AppState,
    job: BenchmarkJob,
    runtime: RuntimeInfo,
    prompt_type: str,
    max_tokens: int,
    context_length: int,
) -> None:
    """Runs in a background asyncio task."""
    def progress_callback(current: int, total: int) -> None:
        job.current_run = current

    try:
        results = await asyncio.to_thread(
            run_benchmark_suite,
            runtime,
            job.model_id,
            prompt_type,
            job.num_runs,
            max_tokens,
            context_length,
            progress_callback,
        )
        job.results = results
        job.aggregate = aggregate_results(results)
        job.status = "complete"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
    finally:
        state.cleanup_old_jobs()
```

### Why Not `BackgroundTasks`?

FastAPI's `BackgroundTasks` run **after** the response is sent, in the same request lifecycle. For benchmark jobs that run 30-180s, `asyncio.create_task()` is better because:
- It is not tied to a request's lifecycle
- The task survives even if the client disconnects
- We can poll progress from separate requests

---

## 5. Frontend Architecture

### Page Structure

Single-page app with tab-based navigation. No framework, no build step.

```
static/
  index.html     # Shell: nav tabs + content container
  style.css      # Styles (CSS custom properties for theming)
  app.js         # All application logic (~800-1200 lines)
```

### HTML Structure

```html
<body>
  <header>
    <h1>Loca-LLAMA</h1>
    <nav>
      <button data-tab="compatibility" class="active">Compatibility</button>
      <button data-tab="models">Models</button>
      <button data-tab="local">Local Models</button>
      <button data-tab="hub">HuggingFace</button>
      <button data-tab="benchmark">Benchmark</button>
      <button data-tab="memory">Memory</button>
    </nav>
  </header>
  <main>
    <section id="tab-compatibility" class="tab-content active">...</section>
    <section id="tab-models" class="tab-content">...</section>
    <section id="tab-local" class="tab-content">...</section>
    <section id="tab-hub" class="tab-content">...</section>
    <section id="tab-benchmark" class="tab-content">...</section>
    <section id="tab-memory" class="tab-content">...</section>
  </main>
  <footer>Loca-LLAMA v0.1.0 | localhost:8000</footer>
</body>
```

### Tab Descriptions

1. **Compatibility** (default view)
   - Hardware selector dropdown (populated from `/api/hardware`)
   - Quant format checkboxes (populated from `/api/quantizations`)
   - Context length slider
   - "Only show models that fit" toggle
   - Results table: model name, quant, total VRAM, tier badge, tok/s estimate, headroom
   - Click row to expand details (template, commands)

2. **Models**
   - Searchable/filterable table of all 50+ models
   - Family filter pills
   - Click to show model detail card (specs, template if available, max context)

3. **Local Models**
   - "Scan" button triggers `/api/scanner/local`
   - Table: name, size, format, source, quant, family
   - Summary bar: total count, total size, by source

4. **HuggingFace**
   - Search input + format filter (GGUF / MLX / All)
   - Results cards: repo_id, downloads, likes, tags
   - Click to show file listing + HF config details

5. **Benchmark**
   - Detect running runtimes (auto on tab open)
   - Select runtime + model + prompt type
   - "Run Benchmark" button
   - Progress indicator (polling `/api/benchmark/{job_id}`)
   - Results table: run#, tok/s, prefill tok/s, TTFT, total time
   - Aggregate summary section

6. **Memory**
   - Live memory gauge (polls `/api/memory/current` every 2s)
   - Visual bar: used / total, pressure badge
   - Memory history chart (simple canvas-based, last 60 readings)

### JavaScript Patterns

```javascript
// API wrapper with error handling
const api = {
  async get(path) {
    const res = await fetch(`/api${path}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(`/api${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
};

// Tab switching
document.querySelectorAll("[data-tab]").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// Polling pattern for benchmarks
async function pollBenchmark(jobId) {
  const poll = setInterval(async () => {
    const data = await api.get(`/benchmark/${jobId}`);
    updateBenchmarkUI(data);
    if (data.status !== "running") {
      clearInterval(poll);
    }
  }, 2000);
}
```

### No Build Step

- Vanilla JS with `fetch()` for API calls
- CSS custom properties for theming (dark mode via `prefers-color-scheme`)
- No bundler, no transpiler, no npm
- Served directly by FastAPI's `StaticFiles`

---

## 6. Testing Strategy

### Test Directory Structure

```
tests/
  __init__.py
  conftest.py                    # Shared fixtures: app, client, sample data
  test_analyzer.py               # Unit: analyzer functions
  test_hardware.py               # Unit: hardware lookup
  test_models.py                 # Unit: model data integrity
  test_quantization.py           # Unit: quant format data
  api/
    __init__.py
    conftest.py                  # API-specific fixtures
    test_hardware_routes.py      # Integration: /api/hardware/*
    test_model_routes.py         # Integration: /api/models
    test_quant_routes.py         # Integration: /api/quantizations
    test_analysis_routes.py      # Integration: /api/analyze/*
    test_template_routes.py      # Integration: /api/templates/*
    test_scanner_routes.py       # Integration: /api/scanner/*
    test_hub_routes.py           # Integration: /api/hub/*  (mocked HTTP)
    test_benchmark_routes.py     # Integration: /api/benchmark/* (mocked runtime)
    test_memory_routes.py        # Integration: /api/memory/*
    test_runtime_routes.py       # Integration: /api/runtime/*
```

### Shared Fixtures (`tests/conftest.py`)

```python
import pytest
from httpx import ASGITransport, AsyncClient
from loca_llama.api.app import create_app
from loca_llama.hardware import MacSpec, APPLE_SILICON_SPECS
from loca_llama.models import LLMModel, MODELS
from loca_llama.quantization import QUANT_FORMATS

@pytest.fixture
def app():
    return create_app()

@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def m4_pro_48() -> MacSpec:
    return APPLE_SILICON_SPECS["M4 Pro 48GB"]

@pytest.fixture
def qwen_32b() -> LLMModel:
    return next(m for m in MODELS if m.name == "Qwen 2.5 32B")

@pytest.fixture
def q4km():
    return QUANT_FORMATS["Q4_K_M"]
```

### What to Test

**Unit Tests (core modules, no API)**:
- `analyzer.estimate_model_size_gb()`: Known model sizes
- `analyzer.estimate_kv_cache_gb()`: Known cache sizes for specific architectures
- `analyzer.compute_tier()`: Boundary conditions (exactly 75%, 90%, 100%, 150%)
- `analyzer.analyze_model()`: Full pipeline for known model+hardware combos
- `analyzer.max_context_for_model()`: Verify binary search correctness
- Hardware/model/quant data integrity: No duplicate names, all referenced quants exist

**API Integration Tests (route-level)**:
- All endpoints return 200 with valid input
- All endpoints return correct error codes with invalid input (400, 404, 422)
- Analyze endpoint math matches direct module call
- Scanner endpoint mocked to avoid filesystem dependency
- Hub endpoints mocked to avoid HuggingFace API dependency
- Benchmark start returns job_id; GET returns correct status progression
- Memory endpoint returns valid readings (mock `get_memory_sample` for CI)

**Mocking Boundaries**:
- Mock `scanner.scan_all()` (filesystem)
- Mock `hub.search_huggingface()` (external HTTP)
- Mock `hf_templates.fetch_hf_model_config()` (external HTTP)
- Mock `benchmark.detect_all_runtimes()` (localhost HTTP probes)
- Mock `benchmark.run_benchmark_suite()` (localhost HTTP + timing)
- Mock `memory_monitor.get_memory_sample()` (subprocess calls to vm_stat)

### Coverage Targets

| Layer | Target | Rationale |
|-------|--------|-----------|
| `loca_llama/api/routes/` | 85%+ | All routes must be tested |
| `loca_llama/api/models.py` | 90%+ | Schema validation is critical |
| `loca_llama/analyzer.py` | 80%+ | Core business logic |
| `loca_llama/hardware.py` | 70%+ | Mostly data, test integrity |
| `loca_llama/models.py` | 70%+ | Mostly data, test integrity |
| Overall | 75%+ | Balance between coverage and effort |

### pytest Configuration

```toml
# In pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --tb=short -x"
```

### Running Tests

```bash
python -m pytest tests/ -v                        # All
python -m pytest tests/test_analyzer.py -v        # Unit only
python -m pytest tests/api/ -v                    # API only
python -m pytest tests/ --cov=loca_llama -v       # With coverage
```

---

## 7. Implementation Approach

### File Organization

```
loca_llama/
  api/
    __init__.py
    app.py                  # FastAPI app factory + lifespan
    state.py                # AppState, BenchmarkJob
    schemas.py              # All Pydantic request/response models
    dependencies.py         # Dependency injection helpers (get_state)
    routes/
      __init__.py           # Router aggregation
      hardware.py           # /api/hardware/*
      models.py             # /api/models
      quantization.py       # /api/quantizations
      analysis.py           # /api/analyze/*
      templates.py          # /api/templates/*
      scanner.py            # /api/scanner/*
      hub.py                # /api/hub/*
      benchmark.py          # /api/benchmark/*
      memory.py             # /api/memory/*
      runtime.py            # /api/runtime/*
static/
  index.html
  style.css
  app.js
tests/
  __init__.py
  conftest.py
  test_analyzer.py
  test_hardware.py
  test_models.py
  api/
    __init__.py
    conftest.py
    test_hardware_routes.py
    test_model_routes.py
    test_analysis_routes.py
    test_template_routes.py
    test_scanner_routes.py
    test_hub_routes.py
    test_benchmark_routes.py
    test_memory_routes.py
    test_runtime_routes.py
```

### Why `schemas.py` Instead of `api/models.py`

The project already has `loca_llama/models.py` (LLMModel database). Using `api/models.py` for Pydantic schemas would cause confusing imports:

```python
from loca_llama.models import MODELS            # LLMModel list
from loca_llama.api.models import ModelResponse  # Pydantic schema
```

Renaming to `schemas.py` avoids ambiguity:

```python
from loca_llama.api.schemas import ModelResponse
```

### Implementation Order (Milestones)

**Milestone 1: Foundation** (API skeleton, builds, serves static)
1. Update `pyproject.toml` with optional `[web]` deps
2. Create `loca_llama/api/__init__.py`, `app.py`, `state.py`, `dependencies.py`
3. Create `loca_llama/api/schemas.py` (all Pydantic models)
4. Create route stubs with `APIRouter`
5. Create minimal `static/index.html`
6. Verify: `uvicorn loca_llama.api.app:app` starts and serves HTML

**Milestone 2: Data Endpoints** (no blocking I/O)
1. Implement `/api/hardware`, `/api/models`, `/api/quantizations`
2. Implement `/api/templates/*`
3. Add unit tests for data endpoints
4. Verify: Frontend can populate dropdowns

**Milestone 3: Analysis Endpoints** (core feature)
1. Implement `/api/analyze`, `/api/analyze/all`, `/api/analyze/max-context`
2. Add unit tests for analyzer functions
3. Add integration tests for analysis routes
4. Build Compatibility tab in frontend
5. Verify: Full analysis workflow works end-to-end

**Milestone 4: Discovery Endpoints** (blocking I/O, needs async wrapping)
1. Implement `/api/scanner/local` with `asyncio.to_thread()`
2. Implement `/api/hub/search`, `/api/hub/files`, `/api/hub/config` with `asyncio.to_thread()`
3. Add tests with mocked external calls
4. Build Local Models and HuggingFace tabs
5. Verify: Scanner finds local models; HF search returns results

**Milestone 5: Benchmark & Runtime** (background tasks, polling)
1. Implement `/api/runtime/status`
2. Implement `/api/benchmark/start`, `/api/benchmark/{job_id}`, `/api/benchmark/prompts`
3. Implement background task with progress tracking
4. Add tests with mocked runtimes
5. Build Benchmark tab with polling UI
6. Verify: Benchmark runs, progress updates, results display

**Milestone 6: Memory Monitor** (lifespan management)
1. Wire `MemoryMonitor` into lifespan context
2. Implement `/api/memory/current`, `/api/memory/report`
3. Build Memory tab with live polling + gauge
4. Add tests with mocked `get_memory_sample()`
5. Verify: Memory readings update live in browser

**Milestone 7: Polish**
1. Error handling audit (all routes have try/except)
2. Frontend error states and loading indicators
3. Responsive CSS
4. Dark mode
5. Full test suite passes with 75%+ coverage

### Patterns to Follow

**Route Handler Template**:
```python
@router.get("/endpoint")
async def handler_name(
    param: str = Query(..., description="..."),
    state: AppState = Depends(get_state),
) -> ResponseModel:
    try:
        # 1. Validate (Pydantic handles most of this)
        # 2. Call core module
        result = core_module.function(param)
        # 3. Convert to response
        return ResponseModel.from_core(result)
    except KeyError:
        raise HTTPException(status_code=404, detail="Resource not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Pydantic Model Template**:
```python
class AnalyzeResponse(BaseModel):
    model_name: str
    quant_name: str
    tier: str

    @classmethod
    def from_estimate(cls, est: ModelEstimate) -> "AnalyzeResponse":
        return cls(
            model_name=est.model.name,
            quant_name=est.quant.name,
            tier=est.tier.value,
        )
```

### Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
web = ["fastapi>=0.115.0", "uvicorn[standard]>=0.34.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "httpx>=0.28", "coverage>=7.0"]
```

### Entry Point

```toml
[project.scripts]
loca-llama = "loca_llama.cli:main"
loca-llama-ui = "loca_llama.interactive:main_interactive"
loca-llama-web = "loca_llama.api.app:serve"
```

With a `serve()` function in `app.py`:
```python
def serve() -> None:
    import uvicorn
    uvicorn.run(
        "loca_llama.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
```

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Benchmark blocks event loop | All requests hang during benchmark | `asyncio.create_task()` + `asyncio.to_thread()` |
| Memory monitor thread leak | Memory grows over long sessions | Lifespan context manager guarantees `stop()` |
| HuggingFace API rate limits | Hub search fails | Return cached results; 15s timeout; graceful error in UI |
| Large analyze/all response | Slow JSON serialization for 50 models x 5 quants | Optional `family` filter; frontend lazy-loads details |
| Scanner on large model dirs | Blocks for seconds | `asyncio.to_thread()` + loading indicator in UI |
| Port conflict with llama.cpp | Webapp default port 8000 conflicts with llama.cpp port 8000 | Check port availability at startup; use 8000 (llama.cpp defaults to 8080) |
| Frontend bundle size | N/A (vanilla JS) | No risk: zero dependencies, < 50 KB total |

---

## 9. What Is NOT Included

- **Authentication/authorization**: Local-only tool, no users
- **Database/ORM**: All in-memory, no persistence needed
- **WebSocket**: Polling is sufficient for benchmark progress and memory
- **Docker/containerization**: Local Mac tool, not deployable
- **CI pipeline changes**: Out of scope for this design (existing GitHub Actions are for Node/Docker)
- **Ollama support**: Explicitly excluded per requirements
- **Cloud deployment**: localhost:8000 only
