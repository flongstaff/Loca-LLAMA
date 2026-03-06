"""Tests for hardware endpoints."""

from __future__ import annotations

import pytest

from loca_llama.hardware import APPLE_SILICON_SPECS


@pytest.mark.anyio
async def test_list_hardware(client):
    """Should return all hardware specs with correct count."""
    resp = await client.get("/api/hardware")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(APPLE_SILICON_SPECS)
    assert len(data["hardware"]) == data["count"]

    first = data["hardware"][0]
    assert "name" in first
    assert "chip" in first
    assert "memory_gb" in first
    assert "usable_memory_gb" in first


@pytest.mark.anyio
async def test_get_hardware_by_name(client):
    """Should return a single hardware spec by name."""
    resp = await client.get("/api/hardware/M4 Pro 48GB")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "M4 Pro 48GB"
    assert data["chip"] == "M4 Pro"
    assert data["memory_gb"] == 48
    assert data["usable_memory_gb"] == 44.0


@pytest.mark.anyio
async def test_get_hardware_not_found(client):
    """Should return 404 for unknown hardware."""
    resp = await client.get("/api/hardware/M99 Ultra 1TB")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
