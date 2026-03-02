"""Memory monitor endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from loca_llama.api.dependencies import get_state
from loca_llama.api.schemas import (
    MemoryCurrentResponse,
    MemoryHistoryResponse,
    MemoryReportResponse,
    MemorySampleResponse,
)
from loca_llama.api.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


def _wall_clock_offset(start_time: float) -> float:
    """Compute offset to convert monotonic-relative timestamps to epoch seconds."""
    return time.time() - time.monotonic() + start_time


@router.get("/current", response_model=MemoryCurrentResponse)
async def memory_current(
    state: AppState = Depends(get_state),
) -> MemoryCurrentResponse:
    """Return the most recent memory sample."""
    try:
        sample = state.memory_monitor.get_current()
        if sample is None:
            raise HTTPException(
                status_code=503, detail="Memory monitor has no samples yet"
            )
        return MemoryCurrentResponse(
            used_gb=round(sample.used_gb, 2),
            free_gb=round(sample.free_gb, 2),
            total_gb=round(sample.total_gb, 2),
            usage_pct=round(sample.usage_pct, 1),
            pressure=sample.pressure,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory current failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read memory status")


@router.get("/history", response_model=MemoryHistoryResponse)
async def memory_history(
    limit: int = Query(default=60, ge=1, le=600),
    state: AppState = Depends(get_state),
) -> MemoryHistoryResponse:
    """Return recent memory samples for charting."""
    try:
        samples = state.memory_monitor.get_history(limit)
        offset = _wall_clock_offset(state.memory_monitor.start_time)
        return MemoryHistoryResponse(
            samples=[
                MemorySampleResponse(
                    timestamp=round(offset + s.timestamp, 2),
                    used_gb=round(s.used_gb, 2),
                    free_gb=round(s.free_gb, 2),
                    total_gb=round(s.total_gb, 2),
                    usage_pct=round(s.usage_pct, 1),
                    pressure=s.pressure,
                )
                for s in samples
            ],
            count=len(samples),
        )
    except Exception as e:
        logger.error("Memory history failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read memory history")


@router.get("/report", response_model=MemoryReportResponse)
async def memory_report(
    state: AppState = Depends(get_state),
) -> MemoryReportResponse:
    """Return aggregate memory report since monitor start."""
    try:
        sample = state.memory_monitor.get_current()
        if sample is None:
            raise HTTPException(
                status_code=503, detail="Memory monitor has no samples yet"
            )
        report = state.memory_monitor.get_report()
        return MemoryReportResponse(
            peak_used_gb=round(report.peak_used_gb, 2),
            baseline_used_gb=round(report.baseline_used_gb, 2),
            delta_gb=round(report.delta_gb, 2),
            total_gb=round(report.total_gb, 2),
            peak_pct=round(report.peak_pct, 1),
            baseline_pct=round(report.baseline_pct, 1),
            duration_sec=round(report.duration_sec, 1),
            sample_count=len(report.samples),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory report failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to build memory report")
