"""Local model scanner endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from loca_llama.api.schemas import LocalModelResponse, ScannerResponse
from loca_llama.scanner import LocalModel, scan_all, scan_custom_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanner", tags=["scanner"])


def _model_to_response(m: LocalModel) -> LocalModelResponse:
    """Convert a LocalModel dataclass to a LocalModelResponse schema."""
    return LocalModelResponse(
        name=m.name,
        path=str(m.path),
        size_gb=round(m.size_gb, 2),
        format=m.format,
        source=m.source,
        quant=m.quant,
        family=m.family,
        repo_id=m.repo_id,
    )


@router.get("/local", response_model=ScannerResponse)
async def scan_local(
    custom_dir: str | None = Query(None, description="Custom directory to scan"),
) -> ScannerResponse:
    """Scan the local filesystem for downloaded LLM models."""
    try:
        if custom_dir:
            raw = await asyncio.to_thread(scan_custom_dir, custom_dir)
        else:
            raw = await asyncio.to_thread(scan_all)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Scanner error: %s", e)
        raise HTTPException(status_code=500, detail="Scan failed: check server logs for details")

    models = [_model_to_response(m) for m in raw]
    total_size = round(sum(m.size_gb for m in models), 2)
    sources: dict[str, int] = {}
    for m in models:
        sources[m.source] = sources.get(m.source, 0) + 1

    return ScannerResponse(
        models=models,
        count=len(models),
        total_size_gb=total_size,
        sources=sources,
    )
