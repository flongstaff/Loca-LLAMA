"""Integration tests for runtime API routes (mocked detection)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from loca_llama.benchmark import RuntimeInfo


MOCK_RUNTIMES = [
    RuntimeInfo(
        name="lm-studio",
        url="http://localhost:1234",
        models=["llama-3-8b-q4_k_m", "mistral-7b-q5_k_m"],
        version="0.3.5",
    ),
    RuntimeInfo(
        name="llama.cpp-server",
        url="http://localhost:8080",
        models=["phi-3-mini-q4_k_m"],
        version=None,
    ),
]


@pytest.mark.anyio
@patch("loca_llama.api.routes.runtime.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_runtime_status_returns_runtimes(mock_detect, client):
    """GET /api/runtime/status returns detected runtimes."""
    resp = await client.get("/api/runtime/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["runtimes"]) == 2


@pytest.mark.anyio
@patch("loca_llama.api.routes.runtime.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_runtime_status_fields(mock_detect, client):
    """Runtime response includes all expected fields."""
    resp = await client.get("/api/runtime/status")
    rt = resp.json()["runtimes"][0]
    assert rt["name"] == "lm-studio"
    assert rt["url"] == "http://localhost:1234"
    assert rt["models"] == ["llama-3-8b-q4_k_m", "mistral-7b-q5_k_m"]
    assert rt["version"] == "0.3.5"


@pytest.mark.anyio
@patch("loca_llama.api.routes.runtime.detect_all_runtimes", return_value=MOCK_RUNTIMES)
async def test_runtime_status_null_version(mock_detect, client):
    """Runtime with no version returns null."""
    resp = await client.get("/api/runtime/status")
    rt = resp.json()["runtimes"][1]
    assert rt["name"] == "llama.cpp-server"
    assert rt["version"] is None


@pytest.mark.anyio
@patch("loca_llama.api.routes.runtime.detect_all_runtimes", return_value=[])
async def test_runtime_status_empty(mock_detect, client):
    """GET /api/runtime/status with no runtimes returns empty list."""
    resp = await client.get("/api/runtime/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["runtimes"] == []


@pytest.mark.anyio
@patch(
    "loca_llama.api.routes.runtime.detect_all_runtimes",
    side_effect=Exception("Connection refused"),
)
async def test_runtime_status_error(mock_detect, client):
    """GET /api/runtime/status returns 500 on detection failure."""
    resp = await client.get("/api/runtime/status")
    assert resp.status_code == 500
    assert "Runtime detection failed" in resp.json()["detail"]
    assert "Connection refused" not in resp.json()["detail"]
