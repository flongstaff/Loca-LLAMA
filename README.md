# Loca-LLAMA

**Local LLM Apple Mac Analyzer** — Find out which LLMs your Mac can actually run, how fast they'll go, and how good they really are.

Running LLMs locally on Apple Silicon means juggling memory limits, quantization trade-offs, and runtime differences — with no easy way to know what fits or performs well on *your* hardware. Loca-LLAMA solves this: it reads your Mac's specs, maps them against 50+ models and 13 quantization formats, and gives you concrete answers — what fits, how fast, and at what quality. Then it lets you benchmark everything end-to-end with speed tests, coding challenges, SQL generation, and standard evals, all combined into a single HTML scorecard you can share.

## Features

- **Hardware database** — All Apple Silicon chips from M1 to M4 Ultra with memory configs
- **50+ LLM models** — Llama, Mistral, Mixtral, Phi, Gemma, Qwen, DeepSeek, CodeLlama, Yi, StarCoder, and more
- **13 quantization formats** — Q2_K through FP16 with bits-per-weight and quality ratings
- **Memory analysis** — Accurate estimates for model weights + KV cache + overhead
- **Context length calculator** — Find the max context your hardware supports for any model
- **Performance estimates** — Token generation speed based on memory bandwidth
- **GPU & batch optimization** — Find optimal GPU layer allocation and batch sizes
- **Local model scanner** — Auto-detect models from LM Studio, llama.cpp, HuggingFace cache, and MLX
- **HuggingFace search** — Browse and search GGUF and MLX models online
- **Comprehensive benchmarking** — Speed (tok/s, TTFT, percentiles), quality (10 coding tasks), SQL generation (25 questions), standard evals (GSM8K, ARC, HellaSwag, IFEval, HumanEval, MMLU), concurrent throughput
- **Unified HTML reports** — Rich scorecard with leaderboard, heatmaps, radar charts, per-model detail cards
- **Three interfaces** — CLI commands, interactive terminal UI, and a web dashboard
- **Zero core dependencies** — Pure Python 3.11+, no pip installs needed for CLI/TUI

## Prerequisites

- **macOS** with Apple Silicon (M1, M2, M3, M4 series)
- **Python 3.11+**

## Quick Start

```bash
pip install -e .

# Auto-detects your hardware
loca-llama check

# Or specify hardware explicitly
loca-llama check --hw "M4 Pro 48GB"

# Interactive terminal UI
loca-llama-ui
```

## Installation

```bash
# Core CLI and TUI (no external dependencies)
pip install -e .

# With web interface (adds FastAPI + uvicorn)
pip install -e ".[web]"

# With dev tools (adds pytest, coverage, httpx)
pip install -e ".[dev]"
```

## CLI Commands

Hardware is auto-detected on Apple Silicon Macs. Use `--hw "M4 Pro 48GB"` to specify manually.

### Check model compatibility

```bash
# Show all models that fit at recommended quant levels
loca-llama check

# Filter by model family
loca-llama check --family Llama

# Check specific quant formats
loca-llama check --quant Q4_K_M Q8_0

# Show everything including models that don't fit
loca-llama check --all

# Check with a specific context length
loca-llama check --context 32768
```

### Detailed model analysis

```bash
loca-llama detail --model "Llama 3.1 70B" --quant Q4_K_M
```

### Find max context length

```bash
loca-llama max-context --model "Qwen 2.5 32B" --quant Q4_K_M
```

### Get recommendations

```bash
loca-llama recommend
loca-llama recommend --use-case coding
loca-llama recommend --use-case reasoning
```

Use cases: `general`, `coding`, `reasoning`, `small`, `large-context`.

### Calculate VRAM requirements

```bash
# For a known model
loca-llama calc --model "Llama 3.1 70B" --quant Q4_K_M --context 8192

# For a custom model
loca-llama calc --params 13 --bpw 4.85 --context 4096 --layers 40 --kv-heads 8
```

### Scan for local models

```bash
loca-llama scan                  # Scan default locations
loca-llama scan --dir ~/models   # Scan a custom directory
```

### GPU and batch optimization

```bash
# Optimal GPU layer allocation
loca-llama gpu-optimize --model "Llama 3.1 70B"
loca-llama gpu-optimize --model "Llama 3.1 70B" --compare   # Compare all quants

# Optimal batch size for throughput
loca-llama batch-optimize --model "Qwen 2.5 32B"
loca-llama batch-optimize --model "Qwen 2.5 32B" --preference high  # Max throughput
```

### Run benchmarks

```bash
# Speed benchmark (against running LM Studio, oMLX, or llama.cpp server)
loca-llama speed                             # Auto-detect runtime + model
loca-llama speed --sweep                     # Benchmark all loaded models
loca-llama speed --prompt coding --runs 5    # Coding prompt, 5 runs

# Quality benchmark (10 coding/reasoning tasks with code execution)
loca-llama quality                           # All loaded models
loca-llama quality --compare claude          # Compare local vs Claude

# SQL generation benchmark (25 questions, 4 difficulty tiers)
loca-llama sql                               # All loaded models
loca-llama sql --difficulty easy,medium       # Filter tiers
loca-llama sql --export html --output r.html  # HTML heatmap report

# Standard LLM evaluations (GSM8K, ARC, HellaSwag, IFEval, HumanEval, MMLU)
loca-llama eval                              # Run all benchmarks
loca-llama eval --bench gsm8k humaneval      # Specific benchmarks
loca-llama eval --samples 50                 # Quick run with fewer samples

# Concurrent throughput test
loca-llama throughput --concurrency 4 --requests 8

# Unified report (combines all saved benchmark results)
loca-llama report                            # Generate HTML scorecard
loca-llama report --models Qwen,Llama        # Filter models
loca-llama report --output scorecard.html    # Custom path

# View saved results
loca-llama results                           # All results
loca-llama results --type sql --compare      # Side-by-side comparison

# Live proxy monitor
loca-llama monitor                           # Monitor LLM API traffic
```

### Show memory status

```bash
loca-llama memory
```

### List available configs

```bash
loca-llama list-hw        # All Apple Silicon configs
loca-llama list-models    # All models in the database
loca-llama list-quants    # Quantization format explanations
```

## Interactive Terminal UI

Run `loca-llama-ui` for a menu-driven terminal interface:

1. **Check Model Compatibility** — See which models fit with colorized memory bars
2. **Scan Local Models** — Find GGUF/MLX models already downloaded on your Mac
3. **Search HuggingFace** — Browse and discover GGUF and MLX models
4. **Detailed Model Analysis** — Deep dive: memory breakdown, context scaling table, speed estimates
5. **Benchmark** — Run speed, quality, SQL, and evaluation benchmarks against local models

## Web Interface

The web dashboard provides a browser-based UI with tabs for compatibility analysis, model comparison, hardware specs, VRAM calculator, benchmarks, local model scanning, HuggingFace Hub search, memory monitoring, and recommendations.

```bash
# Install web dependencies
pip install -e ".[web]"

# Start the server
loca-llama-web

# Or run directly with uvicorn (with auto-reload for development)
uvicorn loca_llama.api.app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## How It Works

### Memory Estimation

```
Total Memory = Model Weights + KV Cache + Overhead

Model Weights = Parameters × Bits-per-Weight / 8
KV Cache      = 2 × Layers × KV_Heads × Head_Dim × Context_Length × 2 bytes
Overhead      = ~10% of model size + 0.5 GB
```

### Performance Estimation

LLM inference on Apple Silicon is memory-bandwidth-bound:

```
Tokens/sec ≈ Memory_Bandwidth / (Model_Size × 1.1)
```

For example, an M4 Pro with 48GB has 273 GB/s bandwidth — enough for ~17 tok/s on a Q4_K_M 70B model.

### Quantization Quick Reference

| Format  | Bits/Weight | Quality     | 7B Model Size |
|---------|------------|-------------|---------------|
| Q4_K_M  | 4.85       | Good        | ~4.3 GB       |
| Q5_K_M  | 5.69       | Very Good   | ~5.1 GB       |
| Q6_K    | 6.56       | Excellent   | ~5.8 GB       |
| Q8_0    | 8.50       | Excellent   | ~7.6 GB       |
| FP16    | 16.00      | Lossless    | ~14.3 GB      |

## Example Output: M4 Pro 48GB

With 48GB unified memory (~44GB usable), you could run:

| Model              | Quant   | Memory | Speed      | Max Context |
|--------------------|---------|--------|------------|-------------|
| Llama 3.3 70B      | Q4_K_M  | ~43 GB | ~6 tok/s   | ~8K         |
| Qwen 2.5 32B       | Q6_K    | ~22 GB | ~12 tok/s  | ~128K       |
| Qwen 2.5 14B       | Q8_0    | ~14 GB | ~19 tok/s  | ~128K       |
| DeepSeek-R1 14B    | Q5_K_M  | ~10 GB | ~26 tok/s  | ~128K       |
| Llama 3.1 8B       | Q8_0    | ~8 GB  | ~32 tok/s  | ~128K       |
| Phi-4 14B           | Q6_K    | ~11 GB | ~24 tok/s  | ~16K        |

## Acknowledgments & Inspirations

Loca-LLAMA's benchmark suite draws from several excellent open-source projects and community efforts:

- **[sql-benchmark.nicklothian.com](https://sql-benchmark.nicklothian.com/)** — The SQL generation benchmark design, heatmap report style, and agentic retry-loop approach are directly inspired by Nick Lothian's SQL benchmark for LLMs. The question format (schema + natural language → SQL → execute → verify) and difficulty-tiered scoring follow the same philosophy.
- **[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)** (HuggingFace) — The evaluation benchmark selection (GSM8K, ARC-Challenge, HellaSwag, MMLU) follows the leaderboard's standard eval suite, adapted for local inference with smaller sample sizes.
- **[GSM8K](https://github.com/openai/grade-school-math)** (OpenAI) — Grade school math reasoning benchmark.
- **[ARC-Challenge](https://allenai.org/data/arc)** (Allen AI) — Science reasoning multiple-choice benchmark.
- **[HellaSwag](https://rowanzellers.com/hellaswag/)** (Rowan Zellers et al.) — Commonsense reasoning completion benchmark.
- **[HumanEval](https://github.com/openai/human-eval)** (OpenAI) — Python code generation benchmark.
- **[IFEval](https://arxiv.org/abs/2311.07911)** (Google) — Instruction-following evaluation with verifiable constraints.
- **[MMLU](https://github.com/hendrycks/test)** (Dan Hendrycks et al.) — Massive multitask language understanding benchmark.
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** (EleutherAI) — Reference implementation for running standard LLM evals; informed our benchmark runner design.

The unified report's weighted scorecard approach (combining speed, quality, eval, SQL, and throughput metrics) is original to this project.

## Documentation

- **[Testing Guide](docs/TESTING.md)** — How to run and contribute tests
- **[Contributing](CONTRIBUTING.md)** — Development setup and contribution guidelines

## License

MIT
