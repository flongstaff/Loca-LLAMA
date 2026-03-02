"""Model database endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from loca_llama.api.schemas import ModelListResponse, ModelResponse
from loca_llama.models import MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models(family: str | None = Query(None, description="Filter by model family")) -> ModelListResponse:
    """Return all LLM models, optionally filtered by family."""
    filtered = MODELS
    if family:
        filtered = [m for m in MODELS if m.family.lower() == family.lower()]

    items = [
        ModelResponse(
            name=m.name,
            family=m.family,
            params_billion=m.params_billion,
            default_context_length=m.default_context_length,
            max_context_length=m.max_context_length,
            num_layers=m.num_layers,
            num_kv_heads=m.num_kv_heads,
            head_dim=m.head_dim,
            license=m.license,
        )
        for m in filtered
    ]
    families = sorted({m.family for m in MODELS})
    return ModelListResponse(models=items, count=len(items), families=families)
