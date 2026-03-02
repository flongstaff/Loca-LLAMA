"""Compatibility analysis endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from loca_llama.analyzer import CompatibilityTier, analyze_all, analyze_model, max_context_for_model
from loca_llama.api.schemas import (
    AnalyzeAllRequest,
    AnalyzeAllResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    MaxContextRequest,
    MaxContextResponse,
    TierSummary,
)
from loca_llama.hardware import APPLE_SILICON_SPECS
from loca_llama.models import MODELS
from loca_llama.quantization import QUANT_FORMATS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


def _lookup_hardware(name: str):
    """Look up hardware by name, raise 400 if not found."""
    try:
        return APPLE_SILICON_SPECS[name]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Hardware '{name}' not found")


def _lookup_model(name: str):
    """Look up model by name, raise 400 if not found."""
    match = next((m for m in MODELS if m.name == name), None)
    if match is None:
        raise HTTPException(status_code=400, detail=f"Model '{name}' not found")
    return match


def _lookup_quant(name: str):
    """Look up quantization format by name, raise 400 if not found."""
    try:
        return QUANT_FORMATS[name]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Quantization '{name}' not found")


def _estimate_to_response(est) -> AnalyzeResponse:
    """Convert a ModelEstimate dataclass to an AnalyzeResponse schema."""
    return AnalyzeResponse(
        model_name=est.model.name,
        quant_name=est.quant.name,
        context_length=est.context_length,
        model_size_gb=round(est.model_size_gb, 2),
        kv_cache_gb=round(est.kv_cache_gb, 2),
        overhead_gb=round(est.overhead_gb, 2),
        total_memory_gb=round(est.total_memory_gb, 2),
        available_memory_gb=round(est.available_memory_gb, 2),
        headroom_gb=round(est.headroom_gb, 2),
        fits_in_memory=est.fits_in_memory,
        tier=est.tier.value,
        tier_label=est.rating,
        memory_utilization_pct=round(est.memory_utilization_pct, 1),
        estimated_tok_per_sec=round(est.estimated_tok_per_sec, 1) if est.estimated_tok_per_sec else None,
        gpu_layers=est.gpu_layers,
        total_layers=est.total_layers,
        offload_pct=round(est.offload_pct, 1) if est.offload_pct else None,
    )


@router.post("", response_model=AnalyzeResponse)
async def analyze_single(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a single model/quant/context combo against hardware."""
    try:
        mac = _lookup_hardware(req.hardware_name)
        model = _lookup_model(req.model_name)
        quant = _lookup_quant(req.quant_name)

        est = analyze_model(mac, model, quant, req.context_length)
        return _estimate_to_response(est)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to analyze model")


@router.post("/all", response_model=AnalyzeAllResponse)
async def analyze_all_models(req: AnalyzeAllRequest) -> AnalyzeAllResponse:
    """Analyze all models against hardware, with optional filters."""
    try:
        mac = _lookup_hardware(req.hardware_name)

        models = MODELS
        if req.family:
            models = [m for m in MODELS if m.family.lower() == req.family.lower()]

        results = analyze_all(
            mac,
            models,
            quant_names=req.quant_names,
            context_length=req.context_length,
            only_fits=req.only_fits,
            include_partial=req.include_partial,
        )

        responses = [_estimate_to_response(est) for est in results]

        summary = TierSummary()
        for est in results:
            if est.tier == CompatibilityTier.FULL_GPU:
                summary.full_gpu += 1
            elif est.tier == CompatibilityTier.COMFORTABLE:
                summary.comfortable += 1
            elif est.tier == CompatibilityTier.TIGHT_FIT:
                summary.tight_fit += 1
            elif est.tier == CompatibilityTier.PARTIAL_OFFLOAD:
                summary.partial_offload += 1
            else:
                summary.wont_fit += 1

        return AnalyzeAllResponse(
            results=responses,
            count=len(responses),
            hardware=req.hardware_name,
            summary=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Bulk analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to analyze models")


@router.post("/max-context", response_model=MaxContextResponse)
async def find_max_context(req: MaxContextRequest) -> MaxContextResponse:
    """Find the maximum context length that fits in memory."""
    try:
        mac = _lookup_hardware(req.hardware_name)
        model = _lookup_model(req.model_name)
        quant = _lookup_quant(req.quant_name)

        max_ctx = max_context_for_model(mac, model, quant)
        max_k = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)

        return MaxContextResponse(
            model_name=req.model_name,
            quant_name=req.quant_name,
            max_context_length=max_ctx,
            max_context_k=max_k,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Max context search failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to find max context")
