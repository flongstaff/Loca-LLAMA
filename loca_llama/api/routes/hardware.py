"""Hardware specification endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from loca_llama.api.schemas import HardwareDetectResponse, HardwareListResponse, HardwareResponse
from loca_llama.hardware import APPLE_SILICON_SPECS, detect_mac

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hardware", tags=["hardware"])


@router.get("", response_model=HardwareListResponse)
async def list_hardware() -> HardwareListResponse:
    """Return all Apple Silicon hardware specifications."""
    try:
        items = [
            HardwareResponse(
                name=name,
                chip=spec.chip,
                cpu_cores=spec.cpu_cores,
                gpu_cores=spec.gpu_cores,
                neural_engine_cores=spec.neural_engine_cores,
                memory_gb=spec.memory_gb,
                memory_bandwidth_gbs=spec.memory_bandwidth_gbs,
                gpu_tflops=spec.gpu_tflops,
                usable_memory_gb=spec.usable_memory_gb,
            )
            for name, spec in APPLE_SILICON_SPECS.items()
        ]
        return HardwareListResponse(hardware=items, count=len(items))
    except Exception as e:
        logger.error("Failed to list hardware: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list hardware")


@router.get("/detect", response_model=HardwareDetectResponse)
async def detect_hardware() -> HardwareDetectResponse:
    """Auto-detect the current Mac's Apple Silicon hardware."""
    try:
        result = detect_mac()
        if result is None:
            return HardwareDetectResponse(
                detected=False,
                reason="Could not detect Apple Silicon hardware. This feature requires a Mac with Apple Silicon.",
            )
        name, spec = result
        return HardwareDetectResponse(
            detected=True,
            name=name,
            chip=spec.chip,
            memory_gb=spec.memory_gb,
        )
    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        return HardwareDetectResponse(
            detected=False,
            reason="Hardware detection encountered an error.",
        )


@router.get("/{name}", response_model=HardwareResponse)
async def get_hardware(name: str) -> HardwareResponse:
    """Return a single hardware specification by name."""
    try:
        spec = APPLE_SILICON_SPECS[name]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hardware '{name}' not found")
    except Exception as e:
        logger.error("Failed to get hardware '%s': %s", name, e)
        raise HTTPException(status_code=500, detail="Failed to get hardware")
    return HardwareResponse(
        name=name,
        chip=spec.chip,
        cpu_cores=spec.cpu_cores,
        gpu_cores=spec.gpu_cores,
        neural_engine_cores=spec.neural_engine_cores,
        memory_gb=spec.memory_gb,
        memory_bandwidth_gbs=spec.memory_bandwidth_gbs,
        gpu_tflops=spec.gpu_tflops,
        usable_memory_gb=spec.usable_memory_gb,
    )
