"""Model database endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from loca_llama.api.schemas import ModelDetailResponse, ModelListResponse, ModelResponse
from loca_llama.models import MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models(family: str | None = Query(None, description="Filter by model family")) -> ModelListResponse:
    """Return all LLM models, optionally filtered by family."""
    try:
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
    except Exception as e:
        logger.error("Failed to list models: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/{name}", response_model=ModelDetailResponse)
async def get_model_detail(name: str) -> ModelDetailResponse:
    """Return full details for a single model by name."""
    try:
        model = next((m for m in MODELS if m.name == name), None)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return ModelDetailResponse(
            name=model.name,
            family=model.family,
            params_billion=model.params_billion,
            default_context_length=model.default_context_length,
            max_context_length=model.max_context_length,
            num_layers=model.num_layers,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            license=model.license,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model '%s': %s", name, e)
        raise HTTPException(status_code=500, detail="Failed to get model details")

