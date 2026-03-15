# Testing Guide

This guide explains how to run tests and verify the Loca-LLAMA functionality.

## Running Tests

### Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=loca_llama --cov-report=html
```

### Test Categories

```bash
# CLI tests
pytest tests/test_cli.py -v

# API route tests
pytest tests/api/ -v

# Analyzer tests
pytest tests/test_analyzer.py -v

# Benchmark tests
pytest tests/test_benchmark*.py -v
```

## Testing the Web Interface

### Start the Development Server

```bash
# Option 1: Using the CLI entry point
loca-llama-web

# Option 2: Using uvicorn directly (with auto-reload)
uvicorn loca_llama.api.app:app --host 127.0.0.1 --port 8000 --reload

# Option 3: Using Python module
python -m loca_llama.api.app
```

Then open http://localhost:8000 in your browser.

### Manual Testing Checklist

The web interface has 8 tabs. Here's what to verify in each:

1. **Compatibility Tab** - Select hardware, quantization, and analyze model compatibility
2. **Models Tab** - Browse all models in the database
3. **Local Models Tab** - Scan for locally downloaded GGUF/MLX files
4. **HuggingFace Tab** - Search for models on HuggingFace Hub
5. **Benchmark Tab** - Run inference benchmarks (requires LM Studio or llama.cpp running)
6. **Memory Tab** - View real-time memory monitoring
7. **Calculator Tab** - Estimate VRAM requirements for custom configurations
8. **Recommend Tab** - Get model recommendations based on hardware and use case

### Cross-Cutting Checks

- **Theme Toggle** - Switch between dark/light themes
- **Responsive Layout** - Test at different screen sizes (768px, 480px, 320px)
- **Console Errors** - Open DevTools and verify no JavaScript errors
- **Detect My Mac** - Auto-detect hardware (macOS only)

## API Endpoints

Full OpenAPI documentation is available at http://localhost:8000/docs when the server is running.

Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/hardware` | List all hardware configs |
| `GET /api/hardware/detect` | Auto-detect current Mac (macOS only) |
| `GET /api/models` | List all models |
| `POST /api/analyze` | Run compatibility analysis |
| `POST /api/recommend` | Get model recommendations |
| `POST /api/benchmark/start` | Start benchmark run |
| `GET /api/memory/current` | Current memory stats |

## Troubleshooting

- **Server won't start**: Ensure FastAPI and uvicorn are installed (`pip install -e ".[web]"`)
- **Blank page**: Check browser console for JavaScript errors
- **"Detect My Mac" fails**: Only works on macOS
- **Benchmark shows no runtimes**: You need LM Studio or llama.cpp running with a model loaded
- **HuggingFace search returns nothing**: Requires internet access

## Writing Tests

When adding new features:

1. Add tests in the appropriate `tests/` file
2. Use pytest fixtures where applicable
3. Follow existing test patterns in the codebase
4. Ensure tests are deterministic and don't require network access (unless testing network features)

Example test structure:

```python
def test_feature_name():
    # Arrange
    # Act
    result = some_function()
    # Assert
    assert result == expected
```

## CI/CD

Tests run automatically on:
- Pull requests
- Pushes to main branch

All tests must pass before merging.
