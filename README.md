# Loca-LLAMA

**Local LLM Apple Mac Analyzer** — Find out which LLMs your Mac can run locally, how fast, and at what quality.

Loca-LLAMA analyzes your Apple Silicon Mac's specs (memory, bandwidth, GPU) and tells you exactly which LLM models fit, at which quantization levels, with what context length, and how fast they'll generate tokens.

## Features

- **Hardware database** — All Apple Silicon chips from M1 to M4 Ultra with memory configs
- **50+ LLM models** — Llama, Mistral, Mixtral, Phi, Gemma, Qwen, DeepSeek, CodeLlama, Yi, StarCoder, and more
- **13 quantization formats** — Q2_K through FP16 with bits-per-weight and quality ratings
- **Memory analysis** — Accurate estimates for model weights + KV cache + overhead
- **Context length calculator** — Find the max context your hardware supports for any model
- **Performance estimates** — Token generation speed based on memory bandwidth
- **Local model scanner** — Auto-detect models from LM Studio, llama.cpp, HuggingFace cache, and MLX
- **HuggingFace search** — Browse and search GGUF and MLX models online
- **Benchmarks** — Compare LM Studio vs llama.cpp performance side-by-side
- **Interactive UI** — Menu-driven terminal interface, or traditional CLI commands
- **Zero dependencies** — Pure Python 3.11+, no pip installs needed

## Quick Start

```bash
# Interactive mode (recommended)
python -m loca_llama

# Or install and use the CLI
pip install -e .
loca-llama-ui          # Interactive mode
loca-llama check --hw "M4 Pro 48GB"   # CLI mode
```

## CLI Commands

### Check which models fit your Mac

```bash
# Show all models that fit on M4 Pro 48GB at recommended quant levels
loca-llama check --hw "M4 Pro 48GB"

# Filter by model family
loca-llama check --hw "M4 Pro 48GB" --family Llama

# Check specific quant formats
loca-llama check --hw "M4 Pro 48GB" --quant Q4_K_M Q8_0

# Show everything including models that don't fit
loca-llama check --hw "M4 Pro 48GB" --all

# Check with a specific context length
loca-llama check --hw "M4 Pro 48GB" --context 32768
```

### Detailed analysis of a specific model

```bash
loca-llama detail --hw "M4 Pro 48GB" --model "Llama 3.1 70B" --quant Q4_K_M
```

### Find max context length

```bash
loca-llama max-context --hw "M4 Pro 48GB" --model "Qwen 2.5 32B" --quant Q4_K_M
```

### Get recommendations

```bash
loca-llama recommend --hw "M4 Pro 48GB"
loca-llama recommend --hw "M4 Pro 48GB" --use-case coding
loca-llama recommend --hw "M4 Pro 48GB" --use-case reasoning
```

### List available configs

```bash
loca-llama list-hw        # All Apple Silicon configs
loca-llama list-models    # All models in the database
loca-llama list-quants    # Quantization format explanations
```

## Interactive Mode

Run `python -m loca_llama` or `loca-llama-ui` to get the interactive menu:

1. **Check Model Compatibility** — See which models fit with colorized memory bars
2. **Scan Local Models** — Find GGUF/MLX models already downloaded on your Mac
3. **Search HuggingFace** — Browse and discover GGUF and MLX models
4. **Detailed Model Analysis** — Deep dive: memory breakdown, context scaling table, speed estimates
5. **Benchmark** — Run actual inference tests comparing LM Studio and llama.cpp

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

## License

MIT
