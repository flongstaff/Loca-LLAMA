"""HuggingFace hub search endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from loca_llama.api.schemas import (
    HubConfigResponse,
    HubFileResponse,
    HubFilesResponse,
    HubModelResponse,
    HubSearchResponse,
)
from loca_llama.hf_templates import fetch_hf_model_config
from loca_llama.hub import get_model_files, search_gguf_models, search_huggingface, search_mlx_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hub", tags=["hub"])


@router.get("/search", response_model=HubSearchResponse)
async def search_hub(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    sort: str = Query("downloads", description="Sort by: downloads, likes, lastModified"),
    format: str | None = Query(None, description="Format filter: gguf, mlx, or None for all"),
) -> HubSearchResponse:
    """Search HuggingFace Hub for models."""
    try:
        if format == "gguf":
            raw = await asyncio.to_thread(search_gguf_models, query, limit)
        elif format == "mlx":
            raw = await asyncio.to_thread(search_mlx_models, query, limit)
        else:
            raw = await asyncio.to_thread(search_huggingface, query, limit, sort)
    except Exception as e:
        logger.error("Hub search error: %s", e)
        raise HTTPException(status_code=502, detail="HuggingFace API error: check server logs for details")

    results = [
        HubModelResponse(
            repo_id=m.repo_id,
            name=m.name,
            author=m.author,
            downloads=m.downloads,
            likes=m.likes,
            tags=m.tags,
            pipeline_tag=m.pipeline_tag,
            is_mlx=m.is_mlx,
            is_gguf=m.is_gguf,
            last_modified=m.last_modified,
        )
        for m in raw
    ]

    return HubSearchResponse(results=results, count=len(results), query=query)


@router.get("/files/{repo_id:path}", response_model=HubFilesResponse)
async def get_files(repo_id: str) -> HubFilesResponse:
    """Get files in a HuggingFace model repository."""
    try:
        raw = await asyncio.to_thread(get_model_files, repo_id)
    except Exception as e:
        logger.error("Hub files error for %s: %s", repo_id, e)
        raise HTTPException(status_code=502, detail="HuggingFace API error: check server logs for details")

    files = [
        HubFileResponse(filename=f["filename"], size=f.get("size") or 0)
        for f in raw
    ]

    return HubFilesResponse(repo_id=repo_id, files=files)


@router.get("/config/{repo_id:path}", response_model=HubConfigResponse)
async def get_config(repo_id: str) -> HubConfigResponse:
    """Fetch model configuration from HuggingFace."""
    try:
        cfg = await asyncio.to_thread(fetch_hf_model_config, repo_id)
    except Exception as e:
        logger.error("Hub config error for %s: %s", repo_id, e)
        raise HTTPException(status_code=502, detail="HuggingFace API error: check server logs for details")

    return HubConfigResponse(
        repo_id=cfg.repo_id,
        model_type=cfg.model_type,
        architecture=cfg.architecture,
        num_layers=cfg.num_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        hidden_size=cfg.hidden_size,
        max_position_embeddings=cfg.max_position_embeddings,
        vocab_size=cfg.vocab_size,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        chat_template=cfg.chat_template,
        license=cfg.license,
        tags=cfg.tags,
    )
