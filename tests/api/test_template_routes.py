"""Tests for template endpoints."""

from __future__ import annotations

import pytest

from loca_llama.templates import TEMPLATES


@pytest.mark.anyio
async def test_list_templates(client):
    """Should return all templates with correct count."""
    resp = await client.get("/api/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(TEMPLATES)
    assert len(data["templates"]) == data["count"]

    first = data["templates"][0]
    assert "model_pattern" in first
    assert "family" in first
    assert "temperature" in first


@pytest.mark.anyio
async def test_match_template(client):
    """Should find a matching template for a known model name."""
    resp = await client.get("/api/templates/match/llama-3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["family"].lower() == "llama"


@pytest.mark.anyio
async def test_match_template_not_found(client):
    """Should return 404 when no template matches."""
    resp = await client.get("/api/templates/match/totally-unknown-model-xyz")
    assert resp.status_code == 404
    assert "no template found" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_lm_studio_preset(client):
    """Should generate an LM Studio preset for a known model."""
    resp = await client.post("/api/templates/lm-studio-preset", json={"model_name": "llama-3"})
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "inference_params" in data
    assert isinstance(data["inference_params"], dict)


@pytest.mark.anyio
async def test_lm_studio_preset_not_found(client):
    """Should return 404 for unknown model."""
    resp = await client.post("/api/templates/lm-studio-preset", json={"model_name": "unknown-model-xyz"})
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_llama_cpp_command(client):
    """Should generate a llama.cpp CLI command."""
    resp = await client.post(
        "/api/templates/llama-cpp-command",
        json={"model_name": "llama-3", "model_path": "/models/llama3.gguf"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "command" in data
    assert "/models/llama3.gguf" in data["command"]


@pytest.mark.anyio
async def test_llama_cpp_command_not_found(client):
    """Should return 404 for unknown model."""
    resp = await client.post(
        "/api/templates/llama-cpp-command",
        json={"model_name": "unknown-model-xyz", "model_path": "/models/test.gguf"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_llama_cpp_command_with_overrides(client):
    """Should apply sampling overrides to generated command."""
    resp = await client.post(
        "/api/templates/llama-cpp-command",
        json={
            "model_name": "llama-3",
            "model_path": "/models/llama3.gguf",
            "sampling_overrides": {"temperature": 1.0, "top_p": 0.95},
        },
    )
    assert resp.status_code == 200
    cmd = resp.json()["command"]
    assert "--temp 1.0" in cmd
    assert "--top-p 0.95" in cmd


@pytest.mark.anyio
async def test_llama_cpp_command_has_modern_flags(client):
    """Generated command should use modern llama.cpp flags."""
    resp = await client.post(
        "/api/templates/llama-cpp-command",
        json={"model_name": "llama-3", "model_path": "/models/llama3.gguf"},
    )
    assert resp.status_code == 200
    cmd = resp.json()["command"]
    assert "--jinja" in cmd
    assert "--color" in cmd
    assert "-fa" in cmd
    assert " -i" not in cmd
