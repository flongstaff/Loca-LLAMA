"""Integration tests for benchmark API routes (mocked runtimes)."""

from __future__ import annotations

import asyncio
import json
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


# ── Stream Tests ───────────────────────────────────────────────────────────


def _mock_streaming_generator(tokens: list[str]):
    """Return a generator that yields (token_text, elapsed_ms) tuples."""
    def gen(*args, **kwargs):
        for i, tok in enumerate(tokens, 1):
            yield (tok, float(i * 100))
    return gen


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch(
    "loca_llama.api.routes.benchmark.benchmark_openai_api_streaming",
    side_effect=_mock_streaming_generator(["Hello", " world", "!"]),
)
async def test_stream_returns_sse_events(mock_stream, mock_detect, client):
    """GET /api/benchmark/stream returns SSE token and done events."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "default",
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events from response body
    lines = resp.text.strip().split("\n")
    events = []
    current_event = {}
    for line in lines:
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
            events.append(current_event)
            current_event = {}

    # Should have 3 token events + 1 done event
    token_events = [e for e in events if e.get("event") == "token"]
    done_events = [e for e in events if e.get("event") == "done"]
    assert len(token_events) == 3
    assert len(done_events) == 1
    assert token_events[0]["data"]["text"] == "Hello"
    assert done_events[0]["data"]["tokens"] == 3


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_stream_unknown_runtime_returns_400(mock_detect, client):
    """GET /api/benchmark/stream with unknown runtime returns 400."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "nonexistent-runtime",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "default",
    })
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_stream_unknown_model_returns_400(mock_detect, client):
    """GET /api/benchmark/stream with unknown model returns 400."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "nonexistent-model",
        "prompt_type": "default",
    })
    assert resp.status_code == 400
    assert "not loaded" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_stream_custom_prompt_required(client):
    """GET /api/benchmark/stream requires custom_prompt for prompt_type=custom."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "custom",
    })
    assert resp.status_code == 400
    assert "custom_prompt" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_stream_unknown_prompt_type_returns_400(client):
    """GET /api/benchmark/stream with invalid prompt_type returns 400."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "nonexistent_type",
    })
    assert resp.status_code == 400
    assert "unknown prompt_type" in resp.json()["detail"].lower()


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch(
    "loca_llama.api.routes.benchmark.benchmark_openai_api_streaming",
    side_effect=ConnectionError("Connection refused"),
)
async def test_stream_error_emits_sse_error_event(mock_stream, mock_detect, client):
    """GET /api/benchmark/stream emits error SSE event on streaming failure."""
    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "default",
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    events = []
    current_event = {}
    for line in lines:
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
            events.append(current_event)
            current_event = {}

    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "message" in error_events[0]["data"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.benchmark_openai_api_streaming")
async def test_stream_custom_prompt_forwarded(mock_stream, mock_detect, client):
    """GET /api/benchmark/stream forwards custom_prompt to the streaming function."""
    mock_stream.side_effect = _mock_streaming_generator(["ok"])

    resp = await client.get("/api/benchmark/stream", params={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "prompt_type": "custom",
        "custom_prompt": "  My custom prompt  ",
    })
    assert resp.status_code == 200

    # Verify the streaming function received the stripped custom prompt
    mock_stream.assert_called_once()
    call_args = mock_stream.call_args
    assert call_args[0][2] == "My custom prompt"  # 3rd positional arg = prompt


@pytest.mark.anyio
async def test_prompts_include_json_type(client):
    """GET /api/benchmark/prompts includes 'json' prompt type."""
    resp = await client.get("/api/benchmark/prompts")
    assert resp.status_code == 200
    assert "json" in resp.json()["prompts"]
