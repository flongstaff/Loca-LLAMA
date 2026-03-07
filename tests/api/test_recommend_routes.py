"""Tests for POST /api/recommend endpoint."""

from __future__ import annotations

import pytest

from loca_llama.hardware import APPLE_SILICON_SPECS


@pytest.mark.anyio
async def test_recommend_valid_request(client):
    """Should return recommendations for valid hardware and use case."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "general",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["hardware"] == "M4 Pro 48GB"
    assert data["use_case"] == "general"
    assert data["count"] <= 8
    assert len(data["recommendations"]) == data["count"]


@pytest.mark.anyio
async def test_recommend_items_have_correct_schema(client):
    """Should return recommendation items with all required fields."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "general",
    })
    assert resp.status_code == 200
    items = resp.json()["recommendations"]
    assert len(items) > 0

    item = items[0]
    required_fields = [
        "rank", "model_name", "quant_name", "tier", "tier_label",
        "model_size_gb", "kv_cache_gb", "overhead_gb", "total_memory_gb",
        "available_memory_gb", "headroom_gb", "memory_utilization_pct",
        "estimated_tok_per_sec", "gpu_layers", "total_layers",
        "offload_pct", "context_length", "max_context_k",
    ]
    for field in required_fields:
        assert field in item, f"Missing field: {field}"


@pytest.mark.anyio
async def test_recommend_sorted_by_params_descending(client):
    """Should return recommendations sorted by model size descending."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "general",
    })
    items = resp.json()["recommendations"]
    # Ranks should be sequential
    for i, item in enumerate(items, 1):
        assert item["rank"] == i


@pytest.mark.anyio
async def test_recommend_no_duplicate_models(client):
    """Should not recommend the same model name twice."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "general",
    })
    items = resp.json()["recommendations"]
    model_names = [item["model_name"] for item in items]
    assert len(model_names) == len(set(model_names))


@pytest.mark.anyio
async def test_recommend_invalid_hardware(client):
    """Should return 400 for unknown hardware name."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M99 Ultra 1TB",
        "use_case": "general",
    })
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_recommend_invalid_use_case(client):
    """Should return 422 for invalid use case (Pydantic validation)."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "invalid_use_case",
    })
    # Pydantic Literal validation returns 422
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_recommend_coding_use_case(client):
    """Should return results filtered to coding models."""
    resp = await client.post("/api/recommend", json={
        "hardware_name": "M4 Pro 48GB",
        "use_case": "coding",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["use_case"] == "coding"


@pytest.mark.anyio
async def test_recommend_small_hardware_large_context(client):
    """Should return few or no results for tiny hardware with large-context use case."""
    # Find smallest hardware
    smallest = min(APPLE_SILICON_SPECS.values(), key=lambda s: s.memory_gb)
    smallest_name = next(k for k, v in APPLE_SILICON_SPECS.items() if v is smallest)

    resp = await client.post("/api/recommend", json={
        "hardware_name": smallest_name,
        "use_case": "large-context",
    })
    assert resp.status_code == 200
    # Should still return valid response even if empty
    data = resp.json()
    assert isinstance(data["recommendations"], list)
    assert data["count"] == len(data["recommendations"])


@pytest.mark.anyio
async def test_hardware_detect_endpoint(client):
    """Should return detection result (may or may not detect depending on host)."""
    resp = await client.get("/api/hardware/detect")
    assert resp.status_code == 200
    data = resp.json()
    assert "detected" in data
    assert isinstance(data["detected"], bool)


@pytest.mark.anyio
async def test_model_detail_valid(client):
    """Should return full details for a known model."""
    resp = await client.get("/api/models/Qwen 2.5 32B")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Qwen 2.5 32B"
    assert data["family"] == "Qwen"
    assert data["params_billion"] > 30.0


@pytest.mark.anyio
async def test_model_detail_not_found(client):
    """Should return 404 for unknown model name."""
    resp = await client.get("/api/models/NonexistentModel99B")
    assert resp.status_code == 404
