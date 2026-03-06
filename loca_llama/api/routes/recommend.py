"""Recommendation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from loca_llama.analyzer import (
    CompatibilityTier,
    max_context_for_model,
    recommend_models,
    VALID_USE_CASES,
)
from loca_llama.api.schemas import (
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
)
from loca_llama.hardware import APPLE_SILICON_SPECS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])

_TIER_LABELS = {
    CompatibilityTier.FULL_GPU: "Full GPU",
    CompatibilityTier.COMFORTABLE: "Comfortable",
    CompatibilityTier.TIGHT_FIT: "Tight Fit",
    CompatibilityTier.PARTIAL_OFFLOAD: "Partial Offload",
    CompatibilityTier.WONT_FIT: "Won't Fit",
}


@router.post("", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest) -> RecommendResponse:
    """Get top model recommendations for the given hardware and use case."""
    try:
        mac = APPLE_SILICON_SPECS.get(req.hardware_name)
        if mac is None:
            raise HTTPException(
                status_code=400,
                detail=f"Hardware '{req.hardware_name}' not found",
            )

        if req.use_case not in VALID_USE_CASES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid use_case '{req.use_case}'",
            )

        results = recommend_models(mac, use_case=req.use_case, top_n=8)

        items: list[RecommendItem] = []
        for rank, est in enumerate(results, 1):
            max_ctx = max_context_for_model(mac, est.model, est.quant)
            max_ctx_k = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)
            items.append(
                RecommendItem(
                    rank=rank,
                    model_name=est.model.name,
                    quant_name=est.quant.name,
                    tier=est.tier.value,
                    tier_label=_TIER_LABELS.get(est.tier, "Unknown"),
                    model_size_gb=round(est.model_size_gb, 2),
                    kv_cache_gb=round(est.kv_cache_gb, 2),
                    overhead_gb=round(est.overhead_gb, 2),
                    total_memory_gb=round(est.total_memory_gb, 2),
                    available_memory_gb=round(est.available_memory_gb, 1),
                    headroom_gb=round(est.headroom_gb, 1),
                    memory_utilization_pct=round(est.memory_utilization_pct, 1),
                    estimated_tok_per_sec=(
                        round(est.estimated_tok_per_sec, 1)
                        if est.estimated_tok_per_sec is not None
                        else None
                    ),
                    gpu_layers=est.gpu_layers,
                    total_layers=est.total_layers,
                    offload_pct=(
                        round(est.offload_pct, 1)
                        if est.offload_pct is not None
                        else None
                    ),
                    context_length=est.context_length,
                    max_context_k=max_ctx_k,
                )
            )

        return RecommendResponse(
            recommendations=items,
            count=len(items),
            hardware=req.hardware_name,
            use_case=req.use_case,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recommend endpoint error")
        raise HTTPException(status_code=500, detail="Internal server error")
