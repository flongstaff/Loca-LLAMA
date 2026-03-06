"""Runtime detection endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from loca_llama.api.schemas import RuntimeResponse, RuntimeStatusResponse
from loca_llama.benchmark import detect_all_runtimes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/status", response_model=RuntimeStatusResponse)
async def runtime_status() -> RuntimeStatusResponse:
    """Detect all running LLM runtimes (LM Studio, llama.cpp)."""
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception as e:
        logger.error("Runtime detection error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Runtime detection failed: check server logs for details",
        )

    results = [
        RuntimeResponse(
            name=r.name,
            url=r.url,
            models=r.models,
            version=r.version,
        )
        for r in runtimes
    ]

    return RuntimeStatusResponse(runtimes=results, count=len(results))
