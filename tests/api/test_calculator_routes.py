"""Tests for calculator endpoints."""

from __future__ import annotations

import pytest

from loca_llama.models import MODELS


@pytest.mark.anyio
async def test_estimate_valid_request(client):
    """Should return memory breakdown for valid parameters."""
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
    assert "model_size_gb" in data
    assert "kv_cache_gb" in data
    assert "overhead_gb" in data
    assert "total_memory_gb" in data
    assert "on_disk_size_gb" in data
    assert "compatible_hardware" in data
    # Total should equal parts
    expected = data["model_size_gb"] + data["kv_cache_gb"] + data["overhead_gb"]
    assert abs(data["total_memory_gb"] - expected) < 0.01


@pytest.mark.anyio
async def test_estimate_returns_compatible_hardware(client):
    """A small model should have many compatible hardware entries."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 3.0,
            "bits_per_weight": 4.0,
            "context_length": 4096,
            "num_layers": 26,
            "num_kv_heads": 4,
            "head_dim": 128,
            "kv_bits": 16,
        },
    )
    assert resp.status_code == 200
    hw = resp.json()["compatible_hardware"]
    assert len(hw) > 0
    first = hw[0]
    assert "name" in first
    assert "memory_gb" in first
    assert "tier" in first
    assert "tier_label" in first
    assert "headroom_gb" in first


@pytest.mark.anyio
async def test_estimate_huge_model_few_hardware(client):
    """A very large model should have fewer compatible hardware entries."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 70.0,
            "bits_per_weight": 16.0,
            "context_length": 8192,
            "num_layers": 80,
            "num_kv_heads": 8,
            "head_dim": 128,
            "kv_bits": 16,
        },
    )
    assert resp.status_code == 200
    hw = resp.json()["compatible_hardware"]
    # 70B FP16 ~ 130GB+ — only 128GB+ Macs can handle this (partial offload or better)
    for h in hw:
        assert h["memory_gb"] >= 128


@pytest.mark.anyio
async def test_estimate_kv_bits_affects_cache(client):
    """Changing kv_bits should change the KV cache estimate."""
    base_params = {
        "params_billion": 7.0,
        "bits_per_weight": 4.85,
        "context_length": 8192,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    resp_16 = await client.post(
        "/api/calculator/estimate",
        json={**base_params, "kv_bits": 16},
    )
    resp_4 = await client.post(
        "/api/calculator/estimate",
        json={**base_params, "kv_bits": 4},
    )
    assert resp_16.status_code == 200
    assert resp_4.status_code == 200
    kv_16 = resp_16.json()["kv_cache_gb"]
    kv_4 = resp_4.json()["kv_cache_gb"]
    assert kv_16 > kv_4
    # FP16 should be 4x INT4
    assert abs(kv_16 / kv_4 - 4.0) < 0.01


@pytest.mark.anyio
async def test_estimate_invalid_params_below_min(client):
    """Should return 422 for params_billion below minimum."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 0.01,
            "bits_per_weight": 4.0,
            "context_length": 4096,
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_estimate_invalid_kv_bits(client):
    """Should return 422 for kv_bits not in [4, 8, 16]."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 7.0,
            "bits_per_weight": 4.0,
            "context_length": 4096,
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "kv_bits": 12,
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_list_calculator_models(client):
    """Should return all models with architecture parameters."""
    resp = await client.get("/api/calculator/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(MODELS)
    assert len(data["models"]) == data["count"]
    assert "families" in data
    assert len(data["families"]) > 0

    first = data["models"][0]
    assert "name" in first
    assert "family" in first
    assert "params_billion" in first
    assert "num_layers" in first
    assert "num_kv_heads" in first
    assert "head_dim" in first
    assert "default_context_length" in first
    assert "max_context_length" in first


@pytest.mark.anyio
async def test_estimate_hardware_sorted_by_memory(client):
    """Compatible hardware should be sorted by memory_gb ascending."""
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 7.0,
            "bits_per_weight": 4.85,
            "context_length": 4096,
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "kv_bits": 16,
        },
    )
    assert resp.status_code == 200
    hw = resp.json()["compatible_hardware"]
    mem_values = [h["memory_gb"] for h in hw]
    assert mem_values == sorted(mem_values)


@pytest.mark.anyio
async def test_estimate_tiers_are_valid(client):
    """All returned tiers should be recognized tier values."""
    valid_tiers = {"full_gpu", "comfortable", "tight_fit", "partial_offload"}
    resp = await client.post(
        "/api/calculator/estimate",
        json={
            "params_billion": 14.0,
            "bits_per_weight": 4.85,
            "context_length": 8192,
            "num_layers": 48,
            "num_kv_heads": 8,
            "head_dim": 128,
            "kv_bits": 16,
        },
    )
    assert resp.status_code == 200
    for hw in resp.json()["compatible_hardware"]:
        assert hw["tier"] in valid_tiers
