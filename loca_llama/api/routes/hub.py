"""HuggingFace hub search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/hub", tags=["hub"])


@router.get("/search")
async def search_hub():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/files/{repo_id:path}")
async def get_files(repo_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/config/{repo_id:path}")
async def get_config(repo_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")
