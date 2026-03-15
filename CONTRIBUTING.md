# Contributing to Loca-LLAMA

Thank you for your interest in contributing! This guide helps you get started.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/flongstaff/loca-llama.git
cd loca-llama

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Start development server (for web UI)
python -m loca_llama.api.app
```

## Project Structure

```
loca_llama/
├── analyzer.py      # Core memory/performance estimation
├── benchmark.py     # Inference benchmarking
├── cli.py          # CLI commands
├── hardware.py     # Apple Silicon database
├── hub.py          # HuggingFace integration
├── interactive.py  # Terminal UI
├── models.py       # LLM model database
├── quantization.py # Quantization formats
├── runtime.py      # Runtime detection
├── scanner.py      # Local model scanner
└── api/            # FastAPI web interface
```

## Adding New Models

1. Edit `loca_llama/models.py` and add to the `MODELS` dictionary
2. Include parameters, layers, attention heads, and context length
3. Run tests: `pytest tests/test_models.py`

## Adding New Hardware

1. Edit `loca_llama/hardware.py` and add to the `HARDWARE` dictionary
2. Include CPU cores, GPU cores, memory configs, and bandwidth
3. Run tests: `pytest tests/test_hardware.py`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=loca_llama

# Run specific test file
pytest tests/test_cli.py

# Run with verbose output
pytest -v
```

See [docs/TESTING.md](docs/TESTING.md) for detailed testing guidelines.

## Code Style

- Python 3.11+
- Type hints encouraged
- Docstrings for public functions
- Follow existing patterns in the codebase

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Write/update tests
5. Run tests: `pytest`
6. Commit with conventional commits: `feat: add new model`
7. Push and open a Pull Request

## Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

## Questions?

Feel free to open an issue for questions or suggestions.
