"""Tests for generic Exception catch blocks (500 error paths) across all API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── hardware.py ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_hardware_returns_500_on_unexpected_error(client):
    """GET /api/hardware should return 500 when iterating APPLE_SILICON_SPECS raises."""
    with patch(
        "loca_llama.api.routes.hardware.APPLE_SILICON_SPECS",
        new_callable=lambda: type("BrokenDict", (), {"items": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))})
    ):
        resp = await client.get("/api/hardware")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to list hardware"


@pytest.mark.anyio
async def test_get_hardware_returns_500_on_unexpected_error(client):
    """GET /api/hardware/{name} should return 500 when spec access raises a non-KeyError."""
    broken = MagicMock()
    broken.__getitem__ = MagicMock(side_effect=RuntimeError("unexpected"))

    with patch("loca_llama.api.routes.hardware.APPLE_SILICON_SPECS", broken):
        resp = await client.get("/api/hardware/M4 Pro 48GB")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to get hardware"


# ── models.py ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_models_returns_500_on_unexpected_error(client):
    """GET /api/models should return 500 when MODELS raises unexpectedly."""
    with patch("loca_llama.api.routes.models.MODELS", new_callable=lambda: type("BadList", (), {"__iter__": lambda self: (_ for _ in ()).throw(RuntimeError("broken"))})):
        resp = await client.get("/api/models")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to list models"


# ── quantization.py ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_quantizations_returns_500_on_unexpected_error(client):
    """GET /api/quantizations should return 500 when QUANT_FORMATS raises unexpectedly."""
    broken = MagicMock()
    broken.values = MagicMock(side_effect=RuntimeError("broken quant"))

    with patch("loca_llama.api.routes.quantization.QUANT_FORMATS", broken):
        resp = await client.get("/api/quantizations")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to list quantizations"


# ── templates.py ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_templates_returns_500_on_unexpected_error(client):
    """GET /api/templates should return 500 when iterating TEMPLATES raises."""
    with patch("loca_llama.api.routes.templates.TEMPLATES", new_callable=lambda: type("BadList", (), {"__iter__": lambda self: (_ for _ in ()).throw(RuntimeError("broken templates"))})):
        resp = await client.get("/api/templates")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to list templates"


@pytest.mark.anyio
async def test_match_template_returns_500_on_unexpected_error(client):
    """GET /api/templates/match/{name} should return 500 when get_template raises."""
    with patch(
        "loca_llama.api.routes.templates.get_template",
        side_effect=RuntimeError("internal failure"),
    ):
        resp = await client.get("/api/templates/match/llama-3")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to match template"


@pytest.mark.anyio
async def test_lm_studio_preset_returns_500_on_unexpected_error(client):
    """POST /api/templates/lm-studio-preset should return 500 when get_lm_studio_preset raises."""
    with patch(
        "loca_llama.api.routes.templates.get_lm_studio_preset",
        side_effect=RuntimeError("preset exploded"),
    ):
        resp = await client.post(
            "/api/templates/lm-studio-preset",
            json={"model_name": "llama-3"},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to generate preset"


@pytest.mark.anyio
async def test_llama_cpp_command_returns_500_on_unexpected_error(client):
    """POST /api/templates/llama-cpp-command should return 500 when get_llama_cpp_command raises."""
    with patch(
        "loca_llama.api.routes.templates.get_llama_cpp_command",
        side_effect=RuntimeError("command exploded"),
    ):
        resp = await client.post(
            "/api/templates/llama-cpp-command",
            json={"model_name": "llama-3", "model_path": "/models/llama.gguf"},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to generate command"


# ── analysis.py ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_analyze_single_returns_500_on_unexpected_error(client):
    """POST /api/analyze should return 500 when analyze_model raises unexpectedly."""
    with patch(
        "loca_llama.api.routes.analysis.analyze_model",
        side_effect=RuntimeError("analyzer crash"),
    ):
        resp = await client.post(
            "/api/analyze",
            json={
                "hardware_name": "M4 Pro 48GB",
                "model_name": "Qwen 2.5 32B",
                "quant_name": "Q4_K_M",
            },
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to analyze model"


@pytest.mark.anyio
async def test_analyze_all_returns_500_on_unexpected_error(client):
    """POST /api/analyze/all should return 500 when analyze_all raises unexpectedly."""
    with patch(
        "loca_llama.api.routes.analysis.analyze_all",
        side_effect=RuntimeError("bulk crash"),
    ):
        resp = await client.post(
            "/api/analyze/all",
            json={"hardware_name": "M4 Pro 48GB"},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to analyze models"


@pytest.mark.anyio
async def test_find_max_context_returns_500_on_unexpected_error(client):
    """POST /api/analyze/max-context should return 500 when max_context_for_model raises."""
    with patch(
        "loca_llama.api.routes.analysis.max_context_for_model",
        side_effect=RuntimeError("max context crash"),
    ):
        resp = await client.post(
            "/api/analyze/max-context",
            json={
                "hardware_name": "M4 Pro 48GB",
                "model_name": "Qwen 2.5 32B",
                "quant_name": "Q4_K_M",
            },
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to find max context"


# ── memory.py ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_memory_current_returns_500_on_unexpected_error(app, client):
    """GET /api/memory/current should return 500 when get_current raises unexpectedly."""
    app.state.app_state.memory_monitor.get_current = MagicMock(
        side_effect=RuntimeError("monitor crash")
    )

    resp = await client.get("/api/memory/current")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to read memory status"


@pytest.mark.anyio
async def test_memory_history_returns_500_on_unexpected_error(app, client):
    """GET /api/memory/history should return 500 when get_history raises unexpectedly."""
    app.state.app_state.memory_monitor.get_history = MagicMock(
        side_effect=RuntimeError("history crash")
    )

    resp = await client.get("/api/memory/history")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to read memory history"


@pytest.mark.anyio
async def test_memory_report_returns_500_on_unexpected_error(app, client):
    """GET /api/memory/report should return 500 when get_report raises unexpectedly."""
    monitor = app.state.app_state.memory_monitor
    monitor.get_current = MagicMock(return_value=MagicMock())  # not None, so 503 guard passes
    monitor.get_report = MagicMock(side_effect=RuntimeError("report crash"))

    resp = await client.get("/api/memory/report")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to build memory report"
