"""Quantization format endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from loca_llama.api.schemas import QuantListResponse, QuantResponse
from loca_llama.quantization import QUANT_FORMATS, RECOMMENDED_FORMATS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quantizations", tags=["quantization"])


@router.get("", response_model=QuantListResponse)
async def list_quantizations() -> QuantListResponse:
    """Return all quantization formats with recommended list."""
    items = [
        QuantResponse(
            name=qf.name,
            bits_per_weight=qf.bits_per_weight,
            quality_rating=qf.quality_rating,
            description=qf.description,
        )
        for qf in QUANT_FORMATS.values()
    ]
    return QuantListResponse(formats=items, recommended=RECOMMENDED_FORMATS)
