# Testing Strategy for Loca-LLAMA

Applies to: `tests/**/*.py`

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                # Shared fixtures
├── test_analyzer.py           # Unit tests for analyzer
├── test_hardware.py           # Unit tests for hardware module
├── api/
│   ├── __init__.py
│   ├── test_compatibility.py  # API integration tests
│   ├── test_benchmark.py
│   ├── test_scanner.py
│   └── test_templates.py
```

## Setup

**`tests/conftest.py`** — Shared fixtures:

```python
import pytest
from httpx import AsyncClient
from loca_llama.api.app import create_app
from loca_llama.hardware import Hardware
from loca_llama.models import Model
from loca_llama.quantization import QuantizationFormat

@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_app()

@pytest.fixture
async def client(app):
    """Async HTTP client for testing routes."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_hardware() -> Hardware:
    """Test hardware: M4 Pro with 48GB."""
    return Hardware(
        name="M4 Pro (test)",
        chip="Apple M4 Pro",
        cpu_cores=12,
        gpu_cores=16,
        unified_memory_gb=48,
        memory_type="LPDDR5",
    )

@pytest.fixture
def sample_model() -> Model:
    """Test model: Qwen2.5-32B."""
    return Model(
        id="qwen2.5-32b",
        family="Qwen2.5",
        size_billions=32,
        architecture="Transformer",
        context_length=131072,
        default_context=8192,
        vocab_size=151643,
        supported_quantizations=["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    )

@pytest.fixture
def sample_quantization() -> QuantizationFormat:
    """Test quantization format."""
    return QuantizationFormat(
        name="Q4_K_M",
        bits_per_weight=4.5,
        description="4-bit with K-quant medium, recommended balance",
    )
```

## Unit Tests

Test individual functions in isolation:

```python
# tests/test_analyzer.py
import pytest
from loca_llama.analyzer import (
    estimate_vram,
    calculate_kv_cache,
    categorize_fit,
)

def test_estimate_vram_32b_fp16():
    """VRAM for 32B model in FP16."""
    vram_gb = estimate_vram(size_billions=32, bits_per_weight=16)
    # 32e9 * 16 / 8 / 1024^3 ≈ 60 GB
    assert 59 < vram_gb < 61

def test_estimate_vram_7b_q4():
    """VRAM for 7B model in Q4."""
    vram_gb = estimate_vram(size_billions=7, bits_per_weight=4)
    # 7e9 * 4 / 8 / 1024^3 ≈ 3.4 GB
    assert 3.3 < vram_gb < 3.5

def test_calculate_kv_cache_context_8k():
    """KV cache for 8K context."""
    kv_gb = calculate_kv_cache(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_context=8192,
    )
    # 2 * 32 * 8 * 128 * 8192 * 8 / 1024^3 ≈ 0.5 GB
    assert 0.4 < kv_gb < 0.6

def test_categorize_fit_full_gpu(sample_hardware, sample_model):
    """Model fits entirely in GPU."""
    model_vram = 32  # GB
    category = categorize_fit(model_vram, sample_hardware.unified_memory_gb)
    assert category == "FULL_GPU"

def test_categorize_fit_wont_fit(sample_hardware):
    """Model too large for hardware."""
    model_vram = 100  # GB (more than M4 Pro 48GB)
    category = categorize_fit(model_vram, sample_hardware.unified_memory_gb)
    assert category == "WONT_FIT"
```

## API Integration Tests

Test routes with realistic inputs:

```python
# tests/api/test_compatibility.py
import pytest
from httpx import AsyncClient
from loca_llama.api.app import create_app

@pytest.mark.asyncio
async def test_check_compatibility_valid_hardware(client: AsyncClient):
    """Check compatibility with valid hardware."""
    response = await client.post(
        "/api/check",
        json={"hardware_name": "M4 Pro", "vram_gb": 48}
    )
    assert response.status_code == 200
    data = response.json()
    assert "compatible_models" in data
    assert "partially_usable" in data
    assert "incompatible" in data

@pytest.mark.asyncio
async def test_check_compatibility_invalid_hardware(client: AsyncClient):
    """Check compatibility with non-existent hardware."""
    response = await client.post(
        "/api/check",
        json={"hardware_name": "Unknown Hardware", "vram_gb": 48}
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    """Retrieve all models."""
    response = await client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data
    assert len(data["models"]) > 0
    # Verify model structure
    for model in data["models"]:
        assert "id" in model
        assert "family" in model
        assert "size_b" in model

@pytest.mark.asyncio
async def test_get_model_details_valid(client: AsyncClient):
    """Get details for a specific model."""
    response = await client.get("/api/model/qwen2.5-32b")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "qwen2.5-32b"
    assert data["size_b"] == 32
    assert "vram_by_quantization" in data

@pytest.mark.asyncio
async def test_get_model_details_not_found(client: AsyncClient):
    """Try to get details for non-existent model."""
    response = await client.get("/api/model/nonexistent-model")
    assert response.status_code == 404
```

## Async Test Pattern

Use pytest-asyncio for async routes:

```python
# tests/api/test_benchmark.py
import pytest
import uuid

@pytest.mark.asyncio
async def test_start_benchmark(client: AsyncClient):
    """Start a benchmark job."""
    response = await client.post(
        "/api/benchmark",
        json={"model_id": "qwen2.5-7b", "runtime": "auto"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert len(data["job_id"]) > 0  # Valid UUID

@pytest.mark.asyncio
async def test_start_benchmark_invalid_model(client: AsyncClient):
    """Start benchmark with invalid model."""
    response = await client.post(
        "/api/benchmark",
        json={"model_id": "nonexistent", "runtime": "auto"}
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_get_benchmark_result_not_found(client: AsyncClient):
    """Query result for non-existent job."""
    fake_job_id = str(uuid.uuid4())
    response = await client.get(f"/api/benchmark/{fake_job_id}")
    assert response.status_code == 404
```

## Mocking Pattern

Mock external dependencies, not business logic:

```python
# tests/api/test_hf_search.py
import pytest
from unittest.mock import patch
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_hf_search_with_mock(client: AsyncClient):
    """Test HF search with mocked API call."""
    mock_results = [
        {
            "id": "Qwen/Qwen2.5-32B-Instruct",
            "size": 32,
            "downloads": 10000,
        }
    ]

    with patch("loca_llama.hub.search_huggingface", return_value=mock_results):
        response = await client.get("/api/hf-search?query=qwen")
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/api/test_compatibility.py -v

# Run specific test
python -m pytest tests/api/test_compatibility.py::test_check_compatibility_valid_hardware -v

# Run with coverage
python -m pytest tests/ --cov=loca_llama --cov-report=html

# Run only async tests
python -m pytest tests/ -k "asyncio" -v
```

## pytest Configuration

**`pyproject.toml`**:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

## Coverage Target

- **API routes** (`loca_llama/api/`): 80%+ coverage
- **Core logic** (`analyzer`, `hardware`, `models`): 70%+ coverage
- **Overall**: 75%+ coverage

## Key Rules

1. **Fixtures Over Setup**: Use pytest fixtures for reusable test data
2. **One Assertion Per Focus**: Each test has one clear purpose
3. **Descriptive Names**: `test_check_compatibility_invalid_hardware` not `test_1`
4. **Async Tests**: Mark with `@pytest.mark.asyncio`, use `async def`
5. **Mock Boundaries**: Mock HTTP calls, not business logic
6. **No Hardcoded Data**: Import constants from `loca_llama.models`, not duplicated
7. **Cleanup**: Use `conftest.py` fixtures for teardown
8. **Isolation**: Each test should be independent (can run in any order)
9. **FastAPI Testing**: Use `AsyncClient` with `base_url` parameter
10. **Error Cases**: Test both success and failure paths
