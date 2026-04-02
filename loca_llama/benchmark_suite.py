"""Benchmark suite for testing model performance across different configurations.

Based on:
- https://github.com/alexziskind1/llama-throughput-lab (throughput testing methodology)
- https://github.com/alexziskind1/draftbench (benchmarking framework)

Provides comprehensive benchmarking for:
- Throughput tests (tokens/second)
- Latency tests (time to first token)
- Memory usage monitoring
- Batch size sweeps
- Context length sweeps
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections.abc import Generator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Model settings
    model_path: str
    gpu_layers: int = -1  # -1 = auto, all = all layers
    context_length: int = 4096
    batch_size: int = 512
    ubatch_size: int = 1024

    # Generation settings
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40

    # Test settings
    num_runs: int = 3
    warmup_runs: int = 1
    prompt: str = ""

    # Server settings
    port: int = 8099
    threads: int = 0  # 0 = auto-detect


@dataclass
class SuiteBenchmarkResult:
    """Result of a single benchmark run."""

    # Configuration
    config_hash: str
    model_name: str
    gpu_layers: int
    batch_size: int
    context_length: int

    # Timing metrics
    time_to_first_token_ms: float
    prompt_eval_time_ms: float
    eval_time_ms: float
    total_time_ms: float

    # Token counts
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int

    # Performance
    tokens_per_second: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float

    # Memory (if available)
    gpu_memory_used_gb: float | None = None
    system_memory_used_gb: float | None = None

    # Additional metrics
    success: bool
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkSweepResult:
    """Results from sweeping through multiple configurations."""

    # Sweep parameters
    parameter_name: str  # "batch_size", "context_length", "gpu_layers"
    parameter_values: list
    results: list[SuiteBenchmarkResult]

    # Aggregations
    best_result: SuiteBenchmarkResult | None = None
    avg_throughput: float | None = None
    best_throughput: float | None = None
    min_throughput: float | None = None

    # Analysis
    recommendations: list[str] = field(default_factory=list)


def calculate_config_hash(config: BenchmarkConfig) -> str:
    """Generate a unique hash for the configuration."""
    return f"{config.model_path.split('/')[-1]}_gl{config.gpu_layers}_bs{config.batch_size}_ctx{config.context_length}"


def run_server(config: BenchmarkConfig) -> subprocess.Popen | None:
    """Start llama-server with the given configuration."""
    import shutil

    exe = shutil.which("llama-server") or shutil.which("server")
    if not exe:
        logger.error("llama-server not found in PATH")
        return None

    # Check port availability before starting
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", config.port)) == 0:
            logger.error("Port %d already in use", config.port)
            return None

    cmd = [
        exe,
        "-m", config.model_path,
        "--port", str(config.port),
        "-c", str(config.context_length),
        "-ngl", str(config.gpu_layers),
        "-b", str(config.batch_size),
        "--ubatch", str(config.ubatch_size),
        "--threads", str(config.threads) if config.threads > 0 else "",
        "--temp", str(config.temperature),
        "--top-p", str(config.top_p),
        "--top-k", str(config.top_k),
        "--no-mmap",  # Always mmap for consistency
    ]

    # Filter empty args
    cmd = [arg for arg in cmd if arg]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready
    for _ in range(60):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{config.port}/health", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return proc
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if proc.poll() is not None:
                logger.warning("Server process exited during startup")
                return None

    logger.warning("Server did not become ready within 60s")
    return None


def send_completion_request(
    port: int,
    prompt: str,
    max_tokens: int,
    timeout: int = 120,
) -> tuple[dict, float, bool]:
    """Send a completion request and measure timing.

    Returns: (response_dict, time_to_first_token_ms, success)
    """
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}

    start_time = time.perf_counter()
    ttft_ms = 0.0
    success = False

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ttft_ms = (time.perf_counter() - start_time) * 1000
            response = json.loads(resp.read().decode())
            success = True
            return response, ttft_ms, success
    except urllib.error.HTTPError as e:
        logger.debug("Completion HTTP error: %s", e.code)
        return {}, ttft_ms, False
    except (urllib.error.URLError, OSError) as e:
        logger.debug("Completion network error: %s", e)
        return {}, ttft_ms, False


def parse_llama_timings(output: str) -> dict:
    """Parse timing output from llama.cpp."""
    import re

    timings = {}

    # Modern format: [ Prompt: 132.3 t/s | Generation: 28.3 t/s ]
    m = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", output)
    if m:
        timings["prompt_tok_s"] = float(m.group(1))
        timings["gen_tok_s"] = float(m.group(2))

    # Legacy format
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        timings["prompt_eval_ms"] = float(m.group(1))
        timings["prompt_tokens"] = int(m.group(2))

    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        timings["eval_ms"] = float(m.group(1))
        timings["eval_tokens"] = int(m.group(2))

    return timings


def run_benchmark(config: BenchmarkConfig, prompt: str | None = None) -> SuiteBenchmarkResult:
    """Run a single benchmark test."""
    if not prompt:
        prompt = "Write a detailed explanation of how neural networks learn through backpropagation."

    config_hash = calculate_config_hash(config)
    model_name = config.model_path.split("/")[-1]

    # Start server
    proc = run_server(config)
    if not proc:
        return SuiteBenchmarkResult(
            config_hash=config_hash,
            model_name=model_name,
            gpu_layers=config.gpu_layers,
            batch_size=config.batch_size,
            context_length=config.context_length,
            time_to_first_token_ms=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            total_time_ms=0,
            prompt_tokens=0,
            generated_tokens=0,
            total_tokens=0,
            tokens_per_second=0,
            prompt_tokens_per_second=0,
            generation_tokens_per_second=0,
            success=False,
            error="Failed to start server",
        )

    try:
        # Send request
        response, ttft_ms, success = send_completion_request(
            config.port, prompt, config.max_tokens
        )

        if not success or not response:
            return SuiteBenchmarkResult(
                config_hash=config_hash,
                model_name=model_name,
                gpu_layers=config.gpu_layers,
                batch_size=config.batch_size,
                context_length=config.context_length,
                time_to_first_token_ms=ttft_ms,
                prompt_eval_time_ms=0,
                eval_time_ms=0,
                total_time_ms=0,
                prompt_tokens=0,
                generated_tokens=0,
                total_tokens=0,
                tokens_per_second=0,
                prompt_tokens_per_second=0,
                generation_tokens_per_second=0,
                success=False,
                error="Request failed",
            )

        # Parse response
        choices = response.get("choices", [])
        usage = response.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate timing — ttft already covers the full round-trip time for a
        # non-streaming request, so use it directly as total_time_ms.
        total_time_ms = ttft_ms if choices else 0

        # Get timings from response if available
        timings = response.get("timings", {})
        prompt_eval_ms = timings.get("prompt_eval_time_ms", 0)
        eval_ms = timings.get("eval_time_ms", 0)

        if not prompt_eval_ms and prompt_tokens > 0:
            prompt_eval_ms = ttft_ms * (prompt_tokens / max(completion_tokens, 1))

        if not eval_ms and completion_tokens > 0:
            eval_ms = total_time_ms - prompt_eval_ms if total_time_ms > prompt_eval_ms else ttft_ms

        # Calculate speeds
        tok_s = (completion_tokens / eval_ms * 1000) if eval_ms > 0 else 0
        prompt_tok_s = (prompt_tokens / prompt_eval_ms * 1000) if prompt_eval_ms > 0 else 0
        gen_tok_s = tok_s

        return SuiteBenchmarkResult(
            config_hash=config_hash,
            model_name=model_name,
            gpu_layers=config.gpu_layers,
            batch_size=config.batch_size,
            context_length=config.context_length,
            time_to_first_token_ms=ttft_ms,
            prompt_eval_time_ms=prompt_eval_ms,
            eval_time_ms=eval_ms,
            total_time_ms=total_time_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tok_s,
            prompt_tokens_per_second=prompt_tok_s,
            generation_tokens_per_second=gen_tok_s,
            success=True,
        )

    finally:
        # Stop server
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def run_benchmark_sweep(
    config: BenchmarkConfig,
    parameter_name: str,
    parameter_values: list,
    prompt: str | None = None,
) -> BenchmarkSweepResult:
    """Run a benchmark sweep across multiple parameter values.

    Args:
        config: Base configuration
        parameter_name: Parameter to sweep ("batch_size", "context_length", etc.)
        parameter_values: List of values to test
        prompt: Prompt to use (optional)

    Returns: BenchmarkSweepResult with all results
    """
    results = []

    for value in parameter_values:
        # Create modified config
        modified_config = BenchmarkConfig(
            model_path=config.model_path,
            gpu_layers=config.gpu_layers,
            context_length=config.context_length,
            batch_size=value if parameter_name == "batch_size" else config.batch_size,
            ubatch_size=value * 2 if parameter_name == "batch_size" else config.ubatch_size,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_runs=1,
            warmup_runs=0,
            prompt=prompt or config.prompt,
            port=config.port,
            threads=config.threads,
        )

        if parameter_name == "context_length":
            modified_config.context_length = value

        # Run benchmark
        result = run_benchmark(modified_config, prompt)
        results.append(result)

    # Calculate aggregations
    successful_results = [r for r in results if r.success]
    throughputs = [r.tokens_per_second for r in successful_results]

    best_result = max(successful_results, key=lambda r: r.tokens_per_second) if successful_results else None

    return BenchmarkSweepResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        results=results,
        best_result=best_result,
        avg_throughput=sum(throughputs) / len(throughputs) if throughputs else None,
        best_throughput=max(throughputs) if throughputs else None,
        min_throughput=min(throughputs) if throughputs else None,
        recommendations=[
            f"Best {parameter_name}: {best_result.batch_size if best_result else 'N/A'}",
            f"Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tok/s" if throughputs else "No successful runs",
        ],
    )


def run_full_benchmark_suite(
    model_path: str,
    gpu_layers: int = -1,
    batch_sizes: list[int] = None,
    context_lengths: list[int] = None,
    num_runs: int = 3,
) -> list[BenchmarkSweepResult]:
    """Run a full benchmark suite across multiple configurations.

    Args:
        model_path: Path to GGUF model
        gpu_layers: Number of GPU layers (-1 = auto)
        batch_sizes: Batch sizes to test (default: [256, 512, 1024, 2048])
        context_lengths: Context lengths to test (default: [512, 1024, 2048])
        num_runs: Number of runs per configuration

    Returns: List of BenchmarkSweepResult for each sweep
    """
    if batch_sizes is None:
        batch_sizes = [256, 512, 1024, 2048]

    if context_lengths is None:
        context_lengths = [512, 1024, 2048]

    base_config = BenchmarkConfig(
        model_path=model_path,
        gpu_layers=gpu_layers,
        context_length=2048,
        batch_size=512,
        max_tokens=200,
        num_runs=num_runs,
    )

    # Run batch size sweep
    batch_sweep = run_benchmark_sweep(
        base_config,
        "batch_size",
        batch_sizes,
    )

    # Run context length sweep
    context_sweep = run_benchmark_sweep(
        base_config,
        "context_length",
        context_lengths,
    )

    return [batch_sweep, context_sweep]
