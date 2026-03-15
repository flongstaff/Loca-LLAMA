# Testing Strategy

Applies to: `tests/**/*.py`

## Structure
- `tests/conftest.py` — shared fixtures (app, client, sample data)
- `tests/test_*.py` — unit tests for core modules
- `tests/api/test_*.py` — API integration tests

## Rules
1. Fixtures over setup — use `conftest.py` for reusable test data
2. One assertion focus per test, descriptive names
3. Async tests: `@pytest.mark.asyncio`, `async def`, `AsyncClient`
4. Mock external dependencies (HTTP calls), not business logic
5. Import constants from `loca_llama.*`, don't hardcode duplicates
6. Test both success and failure paths
7. Each test independent — can run in any order

## Config (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --tb=short"
```

## Coverage Target
- API routes: 80%+, Core logic: 70%+, Overall: 75%+
