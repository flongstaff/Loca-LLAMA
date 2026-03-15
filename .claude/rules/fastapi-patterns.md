# FastAPI Patterns

Applies to: `loca_llama/api/**/*.py`

## Structure
- One router per endpoint group in `loca_llama/api/routes/`
- App init + lifespan in `loca_llama/api/app.py`
- Pydantic models in `loca_llama/api/models.py`

## Rules
1. All handlers are `async def` with `APIRouter()`
2. Pydantic models for all request/response types
3. Wrap handlers in try/except — re-raise `HTTPException`, catch everything else as 500
4. Never expose internal error messages: `detail="Internal server error"` not `detail=str(e)`
5. Use `Depends()` for shared state (memory monitor, runtime connectors)
6. Long operations → `BackgroundTasks` or `asyncio.create_task()`, return job ID for polling
7. Lifespan context manager for startup/shutdown (memory monitor)
8. Validate at route boundary with Pydantic `Field` validators
