"""Tests verifying API response shapes for frontend chart rendering and JSON export.

Phase 4 charts and export consume these API endpoints:
- Calculator /estimate → stacked bar chart (model_size_gb, kv_cache_gb, overhead_gb)
- Benchmark /start + poll → line chart (per-run tok/s) + JSON export
- Benchmark /sweep + poll → bar chart (per-model avg tok/s) + JSON export
"""

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
        run_number=i,
    )
    for i in range(1, 4)
]

MOCK_AGGREGATE = {
    "avg_tok_per_sec": 40.0,
    "min_tok_per_sec": 38.0,
    "max_tok_per_sec": 42.0,
    "avg_prefill_tok_per_sec": 280.0,
    "avg_ttft_ms": 150.0,
    "avg_total_ms": 2650.0,
    "total_tokens_generated": 300,
    "runs": 3,
}


# ── Calculator chart data shape ────────────────────────────────────────────


@pytest.mark.anyio
async def test_calculator_estimate_has_chart_segments(client):
    """Estimate response includes all fields for stacked bar chart segments."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 7.0,
            "bits_per_weight": 4.85,
            "context_length": 8192,
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "kv_bits": 16,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # Stacked bar segments: model + kv_cache + overhead
    assert isinstance(data["model_size_gb"], float)
    assert isinstance(data["kv_cache_gb"], float)
    assert isinstance(data["overhead_gb"], float)
    assert data["model_size_gb"] > 0
    assert data["kv_cache_gb"] > 0
    assert data["overhead_gb"] > 0

    # Hardware comparison bars
    hw = data["compatible_hardware"]
    assert len(hw) > 0
    for entry in hw:
        assert isinstance(entry["memory_gb"], (int, float))
        assert isinstance(entry["headroom_gb"], (int, float))
        assert "name" in entry


# ── Benchmark line chart data shape ────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_suite", return_value=MOCK_RESULTS)
@patch("loca_llama.api.routes.benchmark.aggregate_results", return_value=MOCK_AGGREGATE)
async def test_benchmark_runs_have_line_chart_fields(mock_agg, mock_run, mock_det, client):
    """Completed benchmark runs include per-run tok/s for line chart rendering."""
    start = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "num_runs": 3,
    })
    job_id = start.json()["job_id"]
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/{job_id}")
    assert resp.status_code == 200
    data = resp.json()

    # Line chart needs: each run's tokens_per_second and run_number
    runs = data["runs"]
    assert len(runs) == 3
    for run in runs:
        assert "tokens_per_second" in run
        assert "run_number" in run
        assert isinstance(run["tokens_per_second"], (int, float))
        assert run["tokens_per_second"] > 0


# ── Benchmark JSON export data completeness ────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_suite", return_value=MOCK_RESULTS)
@patch("loca_llama.api.routes.benchmark.aggregate_results", return_value=MOCK_AGGREGATE)
async def test_benchmark_response_exportable(mock_agg, mock_run, mock_det, client):
    """Benchmark response includes all fields needed for JSON export."""
    start = await client.post("/api/benchmark/start", json={
        "runtime_name": "lm-studio",
        "model_id": "llama-3-8b-q4_k_m",
        "num_runs": 3,
    })
    job_id = start.json()["job_id"]
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/{job_id}")
    data = resp.json()

    # Export needs: status, aggregate, individual runs with timing
    assert data["status"] == "complete"
    assert "aggregate" in data
    agg = data["aggregate"]
    assert "avg_tok_per_sec" in agg
    assert "min_tok_per_sec" in agg
    assert "max_tok_per_sec" in agg
    assert "runs" in agg

    for run in data["runs"]:
        assert "total_time_ms" in run
        assert "generated_tokens" in run
        assert "tokens_per_second" in run
        assert "run_number" in run


# ── Sweep bar chart data shape ─────────────────────────────────────────────


MOCK_SWEEP_RESULTS = [
    {
        "model_id": "llama-3-8b-q4_k_m",
        "results": MOCK_RESULTS[:2],
        "aggregate": MOCK_AGGREGATE,
    },
    {
        "model_id": "mistral-7b-q5_k_m",
        "results": MOCK_RESULTS[:1],
        "aggregate": {**MOCK_AGGREGATE, "avg_tok_per_sec": 35.0},
    },
]


@pytest.mark.anyio
@patch("loca_llama.api.routes.benchmark.detect_all_runtimes", return_value=MOCK_RUNTIMES)
@patch("loca_llama.api.routes.benchmark.run_benchmark_sweep", return_value=MOCK_SWEEP_RESULTS)
async def test_sweep_results_have_bar_chart_fields(mock_sweep, mock_det, client):
    """Sweep results include per-model avg_tok_per_sec for bar chart comparison."""
    start = await client.post("/api/benchmark/sweep", json={
        "runtime_name": "lm-studio",
        "model_ids": ["llama-3-8b-q4_k_m", "mistral-7b-q5_k_m"],
        "num_runs": 2,
    })
    job_id = start.json()["job_id"]
    await asyncio.sleep(0.2)

    resp = await client.get(f"/api/benchmark/sweep/{job_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "complete"
    combos = data["combo_results"]
    assert len(combos) == 2

    # Bar chart needs: model_id + aggregate.avg_tok_per_sec per combo
    for combo in combos:
        assert "model_id" in combo
        assert "aggregate" in combo
        assert "avg_tok_per_sec" in combo["aggregate"]
        assert isinstance(combo["aggregate"]["avg_tok_per_sec"], (int, float))

    # Verify different models have different values (chart is meaningful)
    tok_values = [c["aggregate"]["avg_tok_per_sec"] for c in combos]
    assert tok_values[0] != tok_values[1]
