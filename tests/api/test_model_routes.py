"""Tests for model database endpoints."""

from __future__ import annotations

import pytest

from loca_llama.models import MODELS


@pytest.mark.anyio
async def test_list_models(client):
    """Should return all models with families list."""
    resp = await client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(MODELS)
    assert len(data["models"]) == data["count"]
    assert len(data["families"]) > 0

    first = data["models"][0]
    assert "name" in first
    assert "family" in first
    assert "params_billion" in first
    assert "license" in first


@pytest.mark.anyio
async def test_list_models_filter_by_family(client):
    """Should filter models by family (case-insensitive)."""
    resp = await client.get("/api/models?family=llama")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    for m in data["models"]:
        assert m["family"].lower() == "llama"


@pytest.mark.anyio
async def test_list_models_filter_unknown_family(client):
    """Should return empty list for unknown family."""
    resp = await client.get("/api/models?family=nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["models"] == []
    # families list still includes all known families
    assert len(data["families"]) > 0
