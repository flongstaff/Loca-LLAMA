"""FastAPI dependency injection helpers."""

from __future__ import annotations

from loca_llama.api.state import AppState

# Singleton — set by app.py at module import time
_app_state: AppState | None = None


def init_state(state: AppState) -> None:
    """Initialize the global app state (called once by create_app)."""
    global _app_state
    _app_state = state


async def get_state() -> AppState:
    """FastAPI dependency that provides the shared AppState."""
    if _app_state is None:
        raise RuntimeError("Application state not initialized")
    return _app_state
