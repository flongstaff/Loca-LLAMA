# Loca-LLAMA Documentation

## For Users

- **[Main README](../README.md)** - Quick start and usage guide
- **[Testing Guide](TESTING.md)** - How to test the web interface and run tests

## For Contributors

- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to the project
- **[Testing Guide](TESTING.md)** - Running tests and CI/CD information

## API Documentation

When running the web server, full OpenAPI/Swagger documentation is available at:
- http://localhost:8000/docs - Interactive API documentation
- http://localhost:8000/redoc - ReDoc format

## Project Structure

```
loca-llama/
├── loca_llama/              # Core Python package
│   ├── analyzer.py          # Memory/performance estimation
│   ├── cli.py               # CLI commands
│   ├── hardware.py          # Apple Silicon database
│   ├── models.py            # LLM model database
│   ├── benchmark.py         # Speed benchmarks (tok/s, TTFT, percentiles)
│   ├── quality_bench.py     # 10 coding/reasoning quality tasks
│   ├── eval_benchmarks.py   # Standard evals (GSM8K, ARC, etc.)
│   ├── sql_bench.py         # 25 SQL generation questions
│   ├── throughput.py        # Concurrent throughput testing
│   ├── unified_report.py    # Unified HTML mega-report
│   ├── benchmark_results.py # Result storage + hardware detection
│   └── api/                 # FastAPI web interface
├── static/                  # Frontend assets (HTML, CSS, JS)
├── tests/                   # Test suite
├── docs/                    # Documentation
└── README.md                # Main documentation
```

## Additional Resources

- **HuggingFace GGUF models**: https://huggingface.co/models?library=gguf
- **llama.cpp documentation**: https://github.com/ggerganov/llama.cpp
- **LM Studio**: https://lmstudio.ai/
