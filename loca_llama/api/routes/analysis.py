"""Compatibility analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("")
async def analyze_single():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/all")
async def analyze_all():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/max-context")
async def max_context():
    raise HTTPException(status_code=501, detail="Not implemented")
