"""Memory monitor endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/current")
async def memory_current():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/report")
async def memory_report():
    raise HTTPException(status_code=501, detail="Not implemented")
