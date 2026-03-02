"""Tests for quantization endpoints."""

from __future__ import annotations

import pytest

from loca_llama.quantization import QUANT_FORMATS, RECOMMENDED_FORMATS


@pytest.mark.anyio
async def test_list_quantizations(client):
    """Should return all quant formats with recommended list."""
    resp = await client.get("/api/quantizations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["formats"]) == len(QUANT_FORMATS)
    assert data["recommended"] == RECOMMENDED_FORMATS

    first = data["formats"][0]
    assert "name" in first
    assert "bits_per_weight" in first
    assert "quality_rating" in first
    assert "description" in first


@pytest.mark.anyio
async def test_recommended_formats_subset(client):
    """Recommended formats should be a subset of all format names."""
    resp = await client.get("/api/quantizations")
    data = resp.json()
    all_names = {f["name"] for f in data["formats"]}
    for rec in data["recommended"]:
        assert rec in all_names
