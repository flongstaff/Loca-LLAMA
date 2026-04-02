"""Integration tests for scanner API routes (mocked filesystem)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from loca_llama.scanner import LocalModel


MOCK_MODELS = [
    LocalModel(
        name="llama-3-8b-q4_k_m.gguf",
        path=Path("/models/llama-3-8b-q4_k_m.gguf"),
        size_gb=4.58,
        format="gguf",
        source="lm-studio",
        quant="Q4_K_M",
        family="Llama",
        repo_id="meta-llama/Llama-3-8B-GGUF",
    ),
    LocalModel(
        name="mistral-7b-q5_k_m.gguf",
        path=Path("/models/mistral-7b-q5_k_m.gguf"),
        size_gb=5.13,
        format="gguf",
        source="llama.cpp",
        quant="Q5_K_M",
        family="Mistral",
    ),
    LocalModel(
        name="phi-3-mini",
        path=Path("/models/phi-3-mini"),
        size_gb=2.2,
        format="safetensors",
        source="huggingface",
        family="Phi",
    ),
]


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_all", return_value=MOCK_MODELS)
async def test_scan_local_returns_models(mock_scan, client):
    """GET /api/scanner/local returns scanned models."""
    resp = await client.get("/api/scanner/local")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["models"]) == 3
    assert data["total_size_gb"] == pytest.approx(11.91, abs=0.01)


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_all", return_value=MOCK_MODELS)
async def test_scan_local_sources_dict(mock_scan, client):
    """GET /api/scanner/local includes source breakdown."""
    resp = await client.get("/api/scanner/local")
    data = resp.json()
    assert data["sources"]["lm-studio"] == 1
    assert data["sources"]["llama.cpp"] == 1
    assert data["sources"]["huggingface"] == 1


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_all", return_value=MOCK_MODELS)
async def test_scan_local_model_fields(mock_scan, client):
    """Models include all expected fields with correct types."""
    resp = await client.get("/api/scanner/local")
    model = resp.json()["models"][0]
    assert model["name"] == "llama-3-8b-q4_k_m.gguf"
    assert model["path"] == "/models/llama-3-8b-q4_k_m.gguf"
    assert model["format"] == "gguf"
    assert model["source"] == "lm-studio"
    assert model["quant"] == "Q4_K_M"
    assert model["family"] == "Llama"


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_all", return_value=[])
async def test_scan_local_empty(mock_scan, client):
    """GET /api/scanner/local with no models returns empty list."""
    resp = await client.get("/api/scanner/local")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["models"] == []
    assert data["total_size_gb"] == 0.0


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_custom_dir", return_value=MOCK_MODELS[:1])
@patch("loca_llama.api.routes.scanner._validate_custom_dir")
async def test_scan_custom_dir(mock_validate, mock_scan, client):
    """GET /api/scanner/local?custom_dir= uses scan_custom_dir."""
    resp = await client.get("/api/scanner/local?custom_dir=/my/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    mock_scan.assert_called_once_with("/my/models")


@pytest.mark.anyio
async def test_scan_custom_dir_path_traversal(client):
    """GET /api/scanner/local rejects paths outside allowed directories."""
    resp = await client.get("/api/scanner/local?custom_dir=/etc/passwd")
    assert resp.status_code == 400
    assert "custom_dir must be under" in resp.json()["detail"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.scanner.scan_all", side_effect=PermissionError("denied"))
async def test_scan_local_error(mock_scan, client):
    """GET /api/scanner/local returns 500 on scan failure."""
    resp = await client.get("/api/scanner/local")
    assert resp.status_code == 500
    assert "Scan failed" in resp.json()["detail"]
    assert "denied" not in resp.json()["detail"]
