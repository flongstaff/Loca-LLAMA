"""Shared test fixtures for Loca-LLAMA."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from loca_llama.api.app import create_app
from loca_llama.hardware import APPLE_SILICON_SPECS, MacSpec
from loca_llama.models import MODELS, LLMModel
from loca_llama.quantization import QUANT_FORMATS, QuantFormat


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost") as ac:
        yield ac


@pytest.fixture
def m4_pro_48() -> MacSpec:
    return APPLE_SILICON_SPECS["M4 Pro 48GB"]


@pytest.fixture
def qwen_32b() -> LLMModel:
    return next(m for m in MODELS if m.name == "Qwen 2.5 32B")


@pytest.fixture
def q4km() -> QuantFormat:
    return QUANT_FORMATS["Q4_K_M"]
