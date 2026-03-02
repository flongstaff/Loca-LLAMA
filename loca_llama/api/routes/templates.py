"""Model template endpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from loca_llama.api.schemas import (
    LlamaCppCommandRequest,
    LlamaCppCommandResponse,
    LMStudioPresetRequest,
    LMStudioPresetResponse,
    TemplateListResponse,
    TemplateResponse,
)
from loca_llama.templates import (
    TEMPLATES,
    get_llama_cpp_command,
    get_lm_studio_preset,
    get_template,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["templates"])


def _template_to_response(t) -> TemplateResponse:
    """Convert a ModelTemplate dataclass to a TemplateResponse."""
    d = asdict(t)
    return TemplateResponse(**d)


@router.get("", response_model=TemplateListResponse)
async def list_templates() -> TemplateListResponse:
    """Return all model templates."""
    items = [_template_to_response(t) for t in TEMPLATES]
    return TemplateListResponse(templates=items, count=len(items))


@router.get("/match/{model_name}", response_model=TemplateResponse)
async def match_template(model_name: str) -> TemplateResponse:
    """Find the best matching template for a model name."""
    t = get_template(model_name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No template found for '{model_name}'")
    return _template_to_response(t)


@router.post("/lm-studio-preset", response_model=LMStudioPresetResponse)
async def lm_studio_preset(body: LMStudioPresetRequest) -> LMStudioPresetResponse:
    """Generate an LM Studio preset for a model."""
    t = get_template(body.model_name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No template found for '{body.model_name}'")
    preset = get_lm_studio_preset(t, body.model_name)
    return LMStudioPresetResponse(
        name=preset["name"],
        inference_params=preset["inference_params"],
        context_length=preset["context_length"],
        system_prompt=preset["system_prompt"],
    )


@router.post("/llama-cpp-command", response_model=LlamaCppCommandResponse)
async def llama_cpp_command(body: LlamaCppCommandRequest) -> LlamaCppCommandResponse:
    """Generate a llama.cpp CLI command for a model."""
    t = get_template(body.model_name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No template found for '{body.model_name}'")
    cmd = get_llama_cpp_command(
        t,
        model_path=body.model_path,
        context_length=body.context_length,
        n_gpu_layers=body.n_gpu_layers,
    )
    return LlamaCppCommandResponse(command=cmd)
