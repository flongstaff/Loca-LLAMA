"""Runtime detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/status")
async def runtime_status():
    raise HTTPException(status_code=501, detail="Not implemented")
