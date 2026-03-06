"""Integration tests for analysis API routes."""

from __future__ import annotations

import pytest

from loca_llama.analyzer import analyze_model
from loca_llama.hardware import APPLE_SILICON_SPECS
from loca_llama.models import MODELS
from loca_llama.quantization import QUANT_FORMATS


@pytest.mark.anyio
async def test_analyze_single(client):
    """POST /api/analyze returns valid analysis."""
    resp = await client.post("/api/analyze", json={
        "hardware_name": "M4 Pro 48GB",
        "model_name": "Qwen 2.5 32B",
        "quant_name": "Q4_K_M",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "Qwen 2.5 32B"
    assert data["quant_name"] == "Q4_K_M"
    assert data["total_memory_gb"] > 0
    assert data["tier"] in ("full_gpu", "comfortable", "tight_fit", "partial", "wont_fit")
    assert "tier_label" in data
    assert "memory_utilization_pct" in data


@pytest.mark.anyio
async def test_analyze_single_matches_core(client):
    """API output should match direct analyzer call for same inputs."""
    mac = APPLE_SILICON_SPECS["M4 Pro 48GB"]
    model = next(m for m in MODELS if m.name == "Qwen 2.5 32B")
    quant = QUANT_FORMATS["Q4_K_M"]
    est = analyze_model(mac, model, quant)

    resp = await client.post("/api/analyze", json={
        "hardware_name": "M4 Pro 48GB",
        "model_name": "Qwen 2.5 32B",
        "quant_name": "Q4_K_M",
    })
    data = resp.json()
    assert data["tier"] == est.tier.value
    assert abs(data["total_memory_gb"] - round(est.total_memory_gb, 2)) < 0.01


@pytest.mark.anyio
async def test_analyze_invalid_hardware(client):
    resp = await client.post("/api/analyze", json={
        "hardware_name": "M99 Ultra 1TB",
        "model_name": "Qwen 2.5 32B",
        "quant_name": "Q4_K_M",
    })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_analyze_invalid_model(client):
    resp = await client.post("/api/analyze", json={
        "hardware_name": "M4 Pro 48GB",
        "model_name": "Nonexistent Model",
        "quant_name": "Q4_K_M",
    })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_analyze_invalid_quant(client):
    resp = await client.post("/api/analyze", json={
        "hardware_name": "M4 Pro 48GB",
        "model_name": "Qwen 2.5 32B",
        "quant_name": "Q99_Z_Z",
    })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_analyze_all(client):
    """POST /api/analyze/all returns results with summary."""
    resp = await client.post("/api/analyze/all", json={
        "hardware_name": "M4 Pro 48GB",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    assert data["hardware"] == "M4 Pro 48GB"
    assert "summary" in data
    summary = data["summary"]
    total = sum(summary.values())
    assert total == data["count"]


@pytest.mark.anyio
async def test_analyze_all_with_family(client):
    """Filter by family reduces results."""
    all_resp = await client.post("/api/analyze/all", json={
        "hardware_name": "M4 Pro 48GB",
    })
    filtered_resp = await client.post("/api/analyze/all", json={
        "hardware_name": "M4 Pro 48GB",
        "family": "Qwen",
    })
    assert filtered_resp.status_code == 200
    assert filtered_resp.json()["count"] <= all_resp.json()["count"]


@pytest.mark.anyio
async def test_analyze_all_only_fits(client):
    """only_fits=true excludes WONT_FIT results."""
    resp = await client.post("/api/analyze/all", json={
        "hardware_name": "M4 Pro 48GB",
        "only_fits": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["wont_fit"] == 0
    for result in data["results"]:
        assert result["tier"] != "wont_fit"


@pytest.mark.anyio
async def test_analyze_all_invalid_hardware(client):
    resp = await client.post("/api/analyze/all", json={
        "hardware_name": "Nonexistent",
    })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_max_context(client):
    """POST /api/analyze/max-context returns positive value for small model."""
    resp = await client.post("/api/analyze/max-context", json={
        "hardware_name": "M4 Pro 48GB",
        "model_name": "Qwen 2.5 7B",
        "quant_name": "Q4_K_M",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["max_context_length"] > 0
    assert "K" in data["max_context_k"]


@pytest.mark.anyio
async def test_max_context_invalid_hardware(client):
    resp = await client.post("/api/analyze/max-context", json={
        "hardware_name": "Nonexistent",
        "model_name": "Qwen 2.5 7B",
        "quant_name": "Q4_K_M",
    })
    assert resp.status_code == 400
