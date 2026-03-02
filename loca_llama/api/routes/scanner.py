"""Local model scanner endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/scanner", tags=["scanner"])


@router.get("/local")
async def scan_local():
    raise HTTPException(status_code=501, detail="Not implemented")
