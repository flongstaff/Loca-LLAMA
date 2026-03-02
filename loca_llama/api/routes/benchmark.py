"""Benchmark execution endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/prompts")
async def list_prompts():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/start")
async def start_benchmark():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{job_id}")
async def get_benchmark_status(job_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")
