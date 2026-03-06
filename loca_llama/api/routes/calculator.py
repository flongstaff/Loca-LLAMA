"""VRAM calculator endpoints — estimate memory from raw model parameters."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from loca_llama.analyzer import (
    CompatibilityTier,
    estimate_kv_cache_raw,
    estimate_model_size_gb,
    estimate_overhead_gb,
    estimate_partial_offload_speed,
    estimate_tokens_per_second,
)
from loca_llama.api.schemas import (
    CalculatorEstimateRequest,
    CalculatorEstimateResponse,
    CalculatorModelItem,
    CalculatorModelsResponse,
    HardwareCompatibilityItem,
)
from loca_llama.hardware import APPLE_SILICON_SPECS
from loca_llama.models import MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calculator", tags=["calculator"])


@router.post("/estimate", response_model=CalculatorEstimateResponse)
async def estimate_vram(req: CalculatorEstimateRequest) -> CalculatorEstimateResponse:
    """Estimate VRAM requirements from raw model parameters."""
    try:
        model_size = estimate_model_size_gb(req.params_billion, req.bits_per_weight)
        kv_cache = estimate_kv_cache_raw(
            req.num_layers, req.num_kv_heads, req.head_dim,
            req.context_length, req.kv_bits,
        )
        overhead = estimate_overhead_gb(model_size)
        total = model_size + kv_cache + overhead

        compatible: list[HardwareCompatibilityItem] = []
        for name, mac in sorted(APPLE_SILICON_SPECS.items(), key=lambda x: x[1].memory_gb):
            available = mac.usable_memory_gb
            pct = (total / available) * 100 if available > 0 else 999

            if pct <= 75:
                tier = CompatibilityTier.FULL_GPU
            elif pct <= 90:
                tier = CompatibilityTier.COMFORTABLE
            elif pct <= 100:
                tier = CompatibilityTier.TIGHT_FIT
            elif pct <= 150:
                tier = CompatibilityTier.PARTIAL_OFFLOAD
            else:
                continue  # WONT_FIT — skip

            if tier == CompatibilityTier.PARTIAL_OFFLOAD:
                offload_pct = (available / total) * 100 if total > 0 else 0
                tok_s = estimate_partial_offload_speed(mac, model_size, offload_pct)
            else:
                tok_s = estimate_tokens_per_second(mac, model_size, req.params_billion)

            compatible.append(HardwareCompatibilityItem(
                name=name,
                memory_gb=mac.memory_gb,
                tier=tier.value,
                tier_label={
                    CompatibilityTier.FULL_GPU: "Full GPU",
                    CompatibilityTier.COMFORTABLE: "Comfortable",
                    CompatibilityTier.TIGHT_FIT: "Tight Fit",
                    CompatibilityTier.PARTIAL_OFFLOAD: "Partial Offload",
                }[tier],
                headroom_gb=round(available - total, 2),
                estimated_tok_per_sec=round(tok_s, 1) if tok_s else None,
            ))

        return CalculatorEstimateResponse(
            model_size_gb=round(model_size, 2),
            kv_cache_gb=round(kv_cache, 2),
            overhead_gb=round(overhead, 2),
            total_memory_gb=round(total, 2),
            on_disk_size_gb=round(model_size, 2),
            compatible_hardware=compatible,
        )
    except Exception as e:
        logger.error("Calculator estimate failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to compute estimate")


@router.get("/models", response_model=CalculatorModelsResponse)
async def list_calculator_models() -> CalculatorModelsResponse:
    """Return model parameters for auto-fill in the calculator UI."""
    try:
        items = [
            CalculatorModelItem(
                name=m.name,
                family=m.family,
                params_billion=m.params_billion,
                num_layers=m.num_layers,
                num_kv_heads=m.num_kv_heads,
                head_dim=m.head_dim,
                default_context_length=m.default_context_length,
                max_context_length=m.max_context_length,
            )
            for m in MODELS
        ]
        families = sorted({m.family for m in MODELS})
        return CalculatorModelsResponse(models=items, count=len(items), families=families)
    except Exception as e:
        logger.error("Failed to list calculator models: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models")
