"""Route aggregation — collects all APIRouters for registration."""

from __future__ import annotations

from .analysis import router as analysis_router
from .benchmark import router as benchmark_router
from .hardware import router as hardware_router
from .hub import router as hub_router
from .memory import router as memory_router
from .models import router as models_router
from .quantization import router as quantization_router
from .runtime import router as runtime_router
from .scanner import router as scanner_router
from .templates import router as templates_router
from .calculator import router as calculator_router
from .recommend import router as recommend_router

all_routers = [
    hardware_router,
    models_router,
    quantization_router,
    analysis_router,
    templates_router,
    scanner_router,
    hub_router,
    benchmark_router,
    memory_router,
    runtime_router,
    calculator_router,
    recommend_router,
]
