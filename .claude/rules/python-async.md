# Python Async Patterns

Applies to: `loca_llama/api/**/*.py`

## Rules
1. All FastAPI handlers are `async def`
2. Blocking/CPU-bound work → `asyncio.to_thread()` or background tasks
3. Never `await` synchronous functions directly — use `to_thread()`
4. Memory monitor and background services managed via lifespan context
5. Every async operation wrapped in try/except with proper error handling
6. No nested event loops — use `asyncio.create_task()` for concurrent work
7. Test async routes with `pytest-asyncio` and `httpx.AsyncClient`
