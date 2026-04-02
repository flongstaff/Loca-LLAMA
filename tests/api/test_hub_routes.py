"""Integration tests for hub API routes (mocked network)."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from loca_llama.hub import HubModel
from loca_llama.hf_templates import HFModelConfig


MOCK_HUB_MODELS = [
    HubModel(
        repo_id="TheBloke/Llama-2-7B-GGUF",
        name="Llama-2-7B-GGUF",
        author="TheBloke",
        downloads=500000,
        likes=1200,
        tags=["gguf", "text-generation", "llama"],
        pipeline_tag="text-generation",
        is_mlx=False,
        is_gguf=True,
        last_modified="2024-01-15T10:00:00Z",
    ),
    HubModel(
        repo_id="mlx-community/Mistral-7B-v0.3-4bit",
        name="Mistral-7B-v0.3-4bit",
        author="mlx-community",
        downloads=25000,
        likes=80,
        tags=["mlx", "text-generation"],
        pipeline_tag="text-generation",
        is_mlx=True,
        is_gguf=False,
    ),
]

MOCK_FILES = [
    {"filename": "llama-2-7b.Q4_K_M.gguf", "size": 4370000000},
    {"filename": "config.json", "size": 512},
    {"filename": "README.md", "size": None},
]

MOCK_CONFIG = HFModelConfig(
    repo_id="meta-llama/Llama-2-7B",
    model_type="llama",
    architecture="LlamaForCausalLM",
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=32,
    hidden_size=4096,
    max_position_embeddings=4096,
    vocab_size=32000,
    license="llama2",
    tags=["text-generation"],
)


# ── Search Tests ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.search_huggingface", return_value=MOCK_HUB_MODELS)
async def test_search_hub(mock_search, client):
    """GET /api/hub/search returns search results."""
    resp = await client.get("/api/hub/search?query=llama")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "llama"
    assert data["count"] == 2
    assert len(data["results"]) == 2


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.search_huggingface", return_value=MOCK_HUB_MODELS)
async def test_search_hub_model_fields(mock_search, client):
    """Search results include all expected fields."""
    resp = await client.get("/api/hub/search?query=llama")
    model = resp.json()["results"][0]
    assert model["repo_id"] == "TheBloke/Llama-2-7B-GGUF"
    assert model["author"] == "TheBloke"
    assert model["downloads"] == 500000
    assert model["is_gguf"] is True
    assert model["is_mlx"] is False


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.search_gguf_models", return_value=MOCK_HUB_MODELS[:1])
async def test_search_hub_gguf_filter(mock_search, client):
    """GET /api/hub/search?format=gguf uses GGUF-specific search."""
    resp = await client.get("/api/hub/search?query=llama&format=gguf")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    mock_search.assert_called_once()


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.search_mlx_models", return_value=MOCK_HUB_MODELS[1:])
async def test_search_hub_mlx_filter(mock_search, client):
    """GET /api/hub/search?format=mlx uses MLX-specific search."""
    resp = await client.get("/api/hub/search?query=mistral&format=mlx")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    mock_search.assert_called_once()


@pytest.mark.anyio
async def test_search_hub_requires_query(client):
    """GET /api/hub/search without query returns 422."""
    resp = await client.get("/api/hub/search")
    assert resp.status_code == 422


# ── Files Tests ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.get_model_files", return_value=MOCK_FILES)
async def test_get_files(mock_files, client):
    """GET /api/hub/files/{repo_id} returns file listing."""
    resp = await client.get("/api/hub/files/TheBloke/Llama-2-7B-GGUF")
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "TheBloke/Llama-2-7B-GGUF"
    assert len(data["files"]) == 3


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.get_model_files", return_value=MOCK_FILES)
async def test_get_files_null_size_defaults_to_zero(mock_files, client):
    """Files with null size get default of 0."""
    resp = await client.get("/api/hub/files/TheBloke/Llama-2-7B-GGUF")
    readme = next(f for f in resp.json()["files"] if f["filename"] == "README.md")
    assert readme["size"] == 0


# ── Config Tests ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.fetch_hf_model_config", return_value=MOCK_CONFIG)
async def test_get_config(mock_cfg, client):
    """GET /api/hub/config/{repo_id} returns model configuration."""
    resp = await client.get("/api/hub/config/meta-llama/Llama-2-7B")
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "meta-llama/Llama-2-7B"
    assert data["model_type"] == "llama"
    assert data["num_layers"] == 32
    assert data["hidden_size"] == 4096
    assert data["vocab_size"] == 32000


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.fetch_hf_model_config", return_value=MOCK_CONFIG)
async def test_get_config_fields(mock_cfg, client):
    """Config response includes architecture and license."""
    resp = await client.get("/api/hub/config/meta-llama/Llama-2-7B")
    data = resp.json()
    assert data["architecture"] == "LlamaForCausalLM"
    assert data["license"] == "llama2"
    assert data["tags"] == ["text-generation"]


# ── Error Tests ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.search_huggingface", side_effect=urllib.error.URLError("Network error"))
async def test_search_hub_network_error(mock_search, client):
    """GET /api/hub/search returns 502 on network failure."""
    resp = await client.get("/api/hub/search?query=llama")
    assert resp.status_code == 502
    assert "HuggingFace API error" in resp.json()["detail"]
    assert "Network error" not in resp.json()["detail"]


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.get_model_files", side_effect=urllib.error.URLError("Timeout"))
async def test_get_files_network_error(mock_files, client):
    """GET /api/hub/files returns 502 on network failure."""
    resp = await client.get("/api/hub/files/some/repo")
    assert resp.status_code == 502


@pytest.mark.anyio
@patch("loca_llama.api.routes.hub.fetch_hf_model_config", side_effect=urllib.error.URLError("404"))
async def test_get_config_network_error(mock_cfg, client):
    """GET /api/hub/config returns 502 on fetch failure."""
    resp = await client.get("/api/hub/config/some/repo")
    assert resp.status_code == 502
