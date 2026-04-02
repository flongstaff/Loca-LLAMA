"""FastAPI application factory for Loca-LLAMA web API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from loca_llama.api.dependencies import init_state
from loca_llama.api.routes import all_routers
from loca_llama.api.state import AppState

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — start/stop MemoryMonitor."""
    state: AppState = app.state.app_state
    logger.info("Starting MemoryMonitor (interval=%.1fs)", state.memory_monitor.interval)
    state.memory_monitor.start()
    try:
        yield
    finally:
        logger.info("Stopping MemoryMonitor")
        state.memory_monitor.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Loca-LLAMA",
        description="Local LLM Apple Mac Analyzer — Web API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Restrict incoming Host headers to localhost only
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1"],
    )

    # Add security response headers
    @application.middleware("http")
    async def add_security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    # Create shared state and wire up dependency injection
    state = AppState()
    application.state.app_state = state
    init_state(state)

    # Register all API route groups under /api
    for router in all_routers:
        application.include_router(router, prefix="/api")

    # Mount static files last so /api routes take priority
    if STATIC_DIR.is_dir():
        application.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return application


# Module-level app for `uvicorn loca_llama.api.app:app`
app = create_app()


def serve() -> None:
    """CLI entry point for `loca-llama-web`."""
    import uvicorn

    uvicorn.run(
        "loca_llama.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
