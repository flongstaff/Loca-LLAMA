# CLAUDE.md

## Project Overview

loca-llama is a Python CLI/TUI/Web tool that analyzes which LLMs your Mac can run. Zero-dep core with optional FastAPI webapp layer. Python 3.11+.

## Commands

```bash
# Install
pip install -e ".[dev]"          # Dev install with test deps
pip install -e ".[web]"          # With FastAPI webapp

# Run
loca-llama                       # CLI
loca-llama-ui                    # Interactive TUI
loca-llama-web                   # FastAPI webapp

# Test
python3 -m pytest                # Run all tests
python3 -m pytest tests/ -v      # Verbose
python3 -m pytest --coverage     # With coverage
```

## Architecture

- `loca_llama/cli.py` — CLI entry point
- `loca_llama/interactive.py` — TUI entry point
- `loca_llama/api/` — FastAPI webapp
- `loca_llama/analyzer.py` — Hardware analysis
- `loca_llama/hardware.py` — Mac hardware detection
- `loca_llama/scanner.py` — Model scanning
- `loca_llama/models.py` — Data models
- **Benchmarking suite:**
  - `benchmark.py` — Speed benchmarks (tok/s, TTFT, percentiles)
  - `quality_bench.py` — 10 coding/reasoning tasks with code execution
  - `eval_benchmarks.py` — Standard evals (GSM8K, ARC, HellaSwag, IFEval, HumanEval, MMLU)
  - `sql_bench.py` — 25 SQL generation questions across 4 difficulty tiers
  - `throughput.py` — Concurrent load testing with ramped concurrency
  - `benchmark_results.py` — Unified result storage (~/.loca-llama/results/)
  - `benchmark_report.py` — HTML speed reports
  - `sql_bench_report.py` — HTML SQL heatmap reports
  - `unified_report.py` — Unified mega-report combining all benchmarks
  - `code_sandbox.py` — Safe code execution for quality/eval scoring
- `tests/` — pytest test suite
