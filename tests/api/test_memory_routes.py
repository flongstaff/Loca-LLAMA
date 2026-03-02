"""Integration tests for memory monitor API routes (mocked monitor)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from loca_llama.memory_monitor import MemoryReport, MemorySample


# ── Shared test data ─────────────────────────────────────────────────────────

SAMPLE_A = MemorySample(
    timestamp=0.0,
    used_gb=10.0,
    free_gb=22.0,
    total_gb=32.0,
    pressure="normal",
)

SAMPLE_B = MemorySample(
    timestamp=1.0,
    used_gb=14.123456,
    free_gb=17.876544,
    total_gb=32.0,
    pressure="warn",
)

MOCK_REPORT = MemoryReport(
    samples=[SAMPLE_A, SAMPLE_B],
    peak_used_gb=14.123456,
    baseline_used_gb=10.0,
    total_gb=32.0,
    duration_sec=60.5,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_monitor(app):
    return app.state.app_state.memory_monitor


# ── GET /api/memory/current ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_memory_current_returns_sample(mock_monitor, client):
    """GET /api/memory/current returns correct shape and values when monitor has a sample."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_A)

    resp = await client.get("/api/memory/current")

    assert resp.status_code == 200
    data = resp.json()
    assert data["used_gb"] == 10.0
    assert data["free_gb"] == 22.0
    assert data["total_gb"] == 32.0
    assert data["pressure"] == "normal"


@pytest.mark.anyio
async def test_memory_current_503_when_no_samples(mock_monitor, client):
    """GET /api/memory/current returns 503 when monitor has no samples yet."""
    mock_monitor.get_current = MagicMock(return_value=None)

    resp = await client.get("/api/memory/current")

    assert resp.status_code == 503
    assert "no samples" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_memory_current_values_rounded(mock_monitor, client):
    """GET /api/memory/current returns GB values rounded to 2 decimals, pct to 1 decimal."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_B)

    resp = await client.get("/api/memory/current")

    assert resp.status_code == 200
    data = resp.json()
    assert data["used_gb"] == round(SAMPLE_B.used_gb, 2)
    assert data["free_gb"] == round(SAMPLE_B.free_gb, 2)
    assert data["total_gb"] == round(SAMPLE_B.total_gb, 2)
    assert data["usage_pct"] == round(SAMPLE_B.usage_pct, 1)


# ── GET /api/memory/history ───────────────────────────────────────────────────

@pytest.fixture
def _stub_history(mock_monitor):
    """Stub get_history and start_time for history tests."""
    mock_monitor._start_time = 1000.0
    mock_monitor.start_time  # ensure property works


@pytest.mark.anyio
async def test_memory_history_returns_samples(mock_monitor, client):
    """GET /api/memory/history returns all samples when under default limit."""
    mock_monitor.get_history = MagicMock(return_value=[SAMPLE_A, SAMPLE_B])
    mock_monitor._start_time = 1000.0

    resp = await client.get("/api/memory/history")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["samples"]) == 2


@pytest.mark.anyio
async def test_memory_history_respects_limit(mock_monitor, client):
    """GET /api/memory/history?limit=1 returns at most 1 sample."""
    mock_monitor.get_history = MagicMock(return_value=[SAMPLE_B])
    mock_monitor._start_time = 1000.0

    resp = await client.get("/api/memory/history?limit=1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert len(data["samples"]) == 1
    mock_monitor.get_history.assert_called_once_with(1)


@pytest.mark.anyio
async def test_memory_history_empty_when_no_samples(mock_monitor, client):
    """GET /api/memory/history returns empty list when monitor has no samples."""
    mock_monitor.get_history = MagicMock(return_value=[])
    mock_monitor._start_time = 1000.0

    resp = await client.get("/api/memory/history")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["samples"] == []


@pytest.mark.anyio
async def test_memory_history_timestamps_are_epoch(mock_monitor, client):
    """GET /api/memory/history converts relative timestamps to epoch seconds."""
    mock_monitor.get_history = MagicMock(return_value=[SAMPLE_A])
    mock_monitor._start_time = 1000.0

    resp = await client.get("/api/memory/history")

    data = resp.json()
    ts = data["samples"][0]["timestamp"]
    # Epoch timestamp should be much larger than the relative 0.0
    assert ts > 1_000_000_000, f"Expected epoch seconds, got {ts}"


# ── GET /api/memory/report ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_memory_report_returns_report(mock_monitor, client):
    """GET /api/memory/report returns the correct response shape."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_A)
    mock_monitor.get_report = MagicMock(return_value=MOCK_REPORT)

    resp = await client.get("/api/memory/report")

    assert resp.status_code == 200
    data = resp.json()
    for field in (
        "peak_used_gb",
        "baseline_used_gb",
        "delta_gb",
        "total_gb",
        "peak_pct",
        "baseline_pct",
        "duration_sec",
        "sample_count",
    ):
        assert field in data


@pytest.mark.anyio
async def test_memory_report_values_consistent(mock_monitor, client):
    """GET /api/memory/report: peak >= baseline and delta equals peak minus baseline."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_A)
    mock_monitor.get_report = MagicMock(return_value=MOCK_REPORT)

    resp = await client.get("/api/memory/report")

    data = resp.json()
    assert data["peak_used_gb"] >= data["baseline_used_gb"]
    expected_delta = round(MOCK_REPORT.delta_gb, 2)
    assert data["delta_gb"] == expected_delta


@pytest.mark.anyio
async def test_memory_report_sample_count(mock_monitor, client):
    """GET /api/memory/report sample_count matches the number of samples in the report."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_A)
    mock_monitor.get_report = MagicMock(return_value=MOCK_REPORT)

    resp = await client.get("/api/memory/report")

    data = resp.json()
    assert data["sample_count"] == len(MOCK_REPORT.samples)


@pytest.mark.anyio
async def test_memory_report_duration_rounded(mock_monitor, client):
    """GET /api/memory/report duration_sec is rounded to 1 decimal."""
    mock_monitor.get_current = MagicMock(return_value=SAMPLE_A)
    mock_monitor.get_report = MagicMock(return_value=MOCK_REPORT)

    resp = await client.get("/api/memory/report")

    data = resp.json()
    assert data["duration_sec"] == round(MOCK_REPORT.duration_sec, 1)


@pytest.mark.anyio
async def test_memory_report_503_when_no_samples(mock_monitor, client):
    """GET /api/memory/report returns 503 when monitor has no samples yet."""
    mock_monitor.get_current = MagicMock(return_value=None)

    resp = await client.get("/api/memory/report")

    assert resp.status_code == 503
    assert "no samples" in resp.json()["detail"].lower()
