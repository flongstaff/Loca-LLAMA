"""Integration tests for benchmark API routes (mocked runtimes)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from loca_llama.benchmark import BenchmarkResult, RuntimeInfo


MOCK_RUNTIMES = [
    RuntimeInfo(
        name="lm-studio",
        url="http://localhost:1234",
        models=["llama-3-8b-q4_k_m", "mistral-7b-q5_k_m"],
        version="0.3.5",
    ),
]

MOCK_RESULTS = [
    BenchmarkResult(
        model_name="llama-3-8b-q4_k_m",
        runtime="lm-studio",
        prompt_tokens=42,
        generated_tokens=100,
        prompt_eval_time_ms=150.0,
        eval_time_ms=2500.0,
        total_time_ms=2650.0,
        tokens_per_second=40.0,
        prompt_tokens_per_second=280.0,
        context_length=4096,
        success=True,
        run_number=1,
    ),
    BenchmarkResult(
        model_name="llama-3-8b-q4_k_m",
        runtime="lm-studio",
        prompt_tokens=42,
        generated_tokens=100,
        prompt_eval_time_ms=120.0,
        eval_time_ms=2300.0,
        total_time_ms=2420.0,
        tokens_per_second=43.5,
        prompt_tokens_per_second=350.0,
        context_length=4096,
        success=True,
        run_number=2,
    ),
]

MOCK_AGGREGATE = {
    "avg_tok_per_sec": 43.5,
    "min_tok_per_sec": 43.5,
    "max_tok_per_sec": 43.5,
    "avg_prefill_tok_per_sec": 350.0,
    "avg_ttft_ms": 120.0,
    "avg_total_ms": 2420.0,
    "total_tokens_generated": 100,
    "runs": 1,
}


# ── Prompts Tests ───────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_prompts(client):
    """GET /api/benchmark/prompts returns available prompt types."""
    resp = await client.get("/api/benchmark/prompts")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompts" in data
    assert "default" in data["prompts"]
    assert isinstance(data["prompts"]["default"], str)


@pytest.mark.anyio
async def test_list_prompts_has_all_types(client):
    """Prompts include default, coding, reasoning, creative."""
    resp = await client.get("/api/benchmark/prompts")
    prompts = resp.json()["prompts"]
    for key in ("default", "coding", "reasoning", "creative"):
        assert key in prompts


# ── Start Tests ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_suite", return_value=MOCK_RESULTS)
@patch("loca_llama.api.routes.benchmark.aggregate_results", return_value=MOCK_AGGREGATE)
async def test_start_benchmark(mock_agg, mock_run, mock_detect, client):
    """POST /api/benchmark/start creates a job."""
    resp = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "default",
        "num_runs": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"
    # Allow background task to complete
    await asyncio.sleep(0.1)


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_start_benchmark_unknown_runtime(mock_detect, client):
    """POST /api/benchmark/start returns 400 for unknown runtime."""
    resp = await client.post("/api/benchmark/start", json={
        "runtime_name": "nonexistent",
        "model_id": "some-model",
    })
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_start_benchmark_unknown_model(mock_detect, client):
    """POST /api/benchmark/start returns 400 for model not loaded in runtime."""
    resp = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "not-loaded-model",
    })
    assert resp.status_code == 400
    assert "not loaded" in resp.json()["detail"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_start_benchmark_unknown_prompt_type(mock_detect, client):
    """POST /api/benchmark/start returns 400 for unknown prompt_type."""
    resp = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "nonexistent",
    })
    assert resp.status_code == 400
    assert "Unknown prompt_type" in resp.json()["detail"]


@pytest.mark.anyio
@patch(
    "loca_llama.api.routes.benchmark.detect_all_runtimes",
    side_effect=Exception("Connection error"),
)
async def test_start_benchmark_runtime_detection_error(mock_detect, client):
    """POST /api/benchmark/start returns 500 on detection failure."""
    resp = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "some-model",
    })
    assert resp.status_code == 500
    assert "Runtime detection failed" in resp.json()["detail"]
    assert "Connection error" not in resp.json()["detail"]


# ── Status Polling Tests ────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_suite", return_value=MOCK_RESULTS)
@patch("loca_llama.api.routes.benchmark.aggregate_results", return_value=MOCK_AGGREGATE)
async def test_poll_completed_job(mock_agg, mock_run, mock_detect, client):
    """GET /api/benchmark/{job_id} returns completed results."""
    start = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "num_runs": 2,
    })
    job_id = start.json()["job_id"]
    # Wait for background task
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["progress"]["total_runs"] == 2
    assert len(data["runs"]) == 2
    assert data["aggregate"]["avg_tok_per_sec"] == 43.5
    assert data["aggregate"]["runs"] == 1


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_suite", return_value=MOCK_RESULTS)
@patch("loca_llama.api.routes.benchmark.aggregate_results", return_value=MOCK_AGGREGATE)
async def test_poll_job_run_fields(mock_agg, mock_run, mock_detect, client):
    """Benchmark run results include all expected fields."""
    start = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "num_runs": 2,
    })
    job_id = start.json()["job_id"]
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/{job_id}")
    run = resp.json()["runs"][0]
    assert run["run_number"] == 1
    assert run["success"] is True
    assert run["tokens_per_second"] == 40.0
    assert run["prompt_tokens_per_second"] == 280.0
    assert run["time_to_first_token_ms"] == 150.0
    assert run["total_time_ms"] == 2650.0
    assert run["prompt_tokens"] == 42
    assert run["generated_tokens"] == 100


@pytest.mark.anyio
async def test_poll_nonexistent_job(client):
    """GET /api/benchmark/{job_id} returns 404 for unknown ID."""
    resp = await client.get("/api/benchmark/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch(
    "loca_llama.api.routes.benchmark.run_benchmark_suite",
    side_effect=Exception("Runtime crashed"),
)
async def test_poll_failed_job(mock_run, mock_detect, client):
    """GET /api/benchmark/{job_id} shows error status on failure."""
    start = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "num_runs": 1,
    })
    job_id = start.json()["job_id"]
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"] is not None
