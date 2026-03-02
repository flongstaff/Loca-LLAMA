"""Benchmark runner: test models on llama.cpp server and LM Studio."""

import json
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    runtime: str  # "llama.cpp" or "lm-studio"
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_time_ms: float  # Time to process prompt (prefill)
    eval_time_ms: float  # Time to generate tokens
    total_time_ms: float
    tokens_per_second: float  # Generation speed
    prompt_tokens_per_second: float  # Prefill speed
    context_length: int
    success: bool
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def time_to_first_token_ms(self) -> float:
        return self.prompt_eval_time_ms


BENCH_PROMPT = (
    "Write a detailed explanation of how neural networks learn through backpropagation. "
    "Include the mathematical foundations and provide a concrete example."
)

BENCH_MAX_TOKENS = 200


def detect_llama_cpp_server() -> str | None:
    """Check if llama.cpp server is running and return the base URL."""
    for port in [8080, 8081, 8000]:
        url = f"http://127.0.0.1:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None


def detect_lm_studio() -> str | None:
    """Check if LM Studio API server is running."""
    for port in [1234, 1235]:
        url = f"http://127.0.0.1:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None


def get_lm_studio_models(base_url: str) -> list[str]:
    """Get list of loaded models from LM Studio."""
    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def benchmark_openai_api(
    base_url: str,
    model_id: str,
    runtime_name: str,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
) -> BenchmarkResult:
    """Run benchmark against an OpenAI-compatible API (LM Studio or llama.cpp)."""
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    headers = {
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return BenchmarkResult(
            model_name=model_id,
            runtime=runtime_name,
            prompt_tokens=0,
            generated_tokens=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
            prompt_tokens_per_second=0,
            context_length=context_length,
            success=False,
            error=str(e),
        )

    total_ms = (time.perf_counter() - start) * 1000
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Try to get timing from response (llama.cpp provides this)
    timings = data.get("timings", {})
    prompt_eval_ms = timings.get("prompt_eval_time_ms") or timings.get("prompt_ms", 0)
    eval_ms = timings.get("eval_time_ms") or timings.get("predicted_ms", 0)

    if not prompt_eval_ms and not eval_ms:
        # Estimate from total time: assume 30% prompt, 70% generation
        prompt_eval_ms = total_ms * 0.3
        eval_ms = total_ms * 0.7

    tok_per_sec = (completion_tokens / eval_ms * 1000) if eval_ms > 0 else 0
    prompt_tok_per_sec = (prompt_tokens / prompt_eval_ms * 1000) if prompt_eval_ms > 0 else 0

    return BenchmarkResult(
        model_name=model_id,
        runtime=runtime_name,
        prompt_tokens=prompt_tokens,
        generated_tokens=completion_tokens,
        prompt_eval_time_ms=prompt_eval_ms,
        eval_time_ms=eval_ms,
        total_time_ms=total_ms,
        tokens_per_second=tok_per_sec,
        prompt_tokens_per_second=prompt_tok_per_sec,
        context_length=context_length,
        success=True,
        extra=timings,
    )


def benchmark_llama_cpp_native(
    model_path: str,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    n_gpu_layers: int = -1,
) -> BenchmarkResult:
    """Run benchmark using llama.cpp CLI directly (llama-cli or main)."""
    # Find llama.cpp executable
    exe = shutil.which("llama-cli") or shutil.which("llama-cpp") or shutil.which("main")
    if not exe:
        return BenchmarkResult(
            model_name=Path(model_path).stem,
            runtime="llama.cpp",
            prompt_tokens=0,
            generated_tokens=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
            prompt_tokens_per_second=0,
            context_length=context_length,
            success=False,
            error="llama.cpp CLI not found (tried: llama-cli, llama-cpp, main)",
        )

    cmd = [
        exe,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-c", str(context_length),
        "-ngl", str(n_gpu_layers),
        "--log-disable",
    ]

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = result.stderr + result.stdout
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            model_name=Path(model_path).stem,
            runtime="llama.cpp",
            prompt_tokens=0,
            generated_tokens=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
            prompt_tokens_per_second=0,
            context_length=context_length,
            success=False,
            error="Benchmark timed out (180s)",
        )
    except Exception as e:
        return BenchmarkResult(
            model_name=Path(model_path).stem,
            runtime="llama.cpp",
            prompt_tokens=0,
            generated_tokens=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
            prompt_tokens_per_second=0,
            context_length=context_length,
            success=False,
            error=str(e),
        )

    total_ms = (time.perf_counter() - start) * 1000

    # Parse llama.cpp timing output
    import re

    prompt_eval_ms = 0.0
    eval_ms = 0.0
    prompt_tokens = 0
    gen_tokens = 0

    # llama_print_timings: prompt eval time = X ms / Y tokens (Z tok/s)
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        prompt_eval_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))

    # llama_print_timings:        eval time = X ms / Y tokens (Z tok/s)
    m = re.search(r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        eval_ms = float(m.group(1))
        gen_tokens = int(m.group(2))

    tok_per_sec = (gen_tokens / eval_ms * 1000) if eval_ms > 0 else 0
    prompt_tok_per_sec = (prompt_tokens / prompt_eval_ms * 1000) if prompt_eval_ms > 0 else 0

    return BenchmarkResult(
        model_name=Path(model_path).stem,
        runtime="llama.cpp",
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        prompt_eval_time_ms=prompt_eval_ms,
        eval_time_ms=eval_ms,
        total_time_ms=total_ms,
        tokens_per_second=tok_per_sec,
        prompt_tokens_per_second=prompt_tok_per_sec,
        context_length=context_length,
        success=result.returncode == 0 and gen_tokens > 0,
        error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
    )


def compare_runtimes(
    model_path: str,
    lm_studio_url: str | None = None,
    llama_cpp_url: str | None = None,
) -> list[BenchmarkResult]:
    """Run the same model on both runtimes and compare."""
    results = []
    model_name = Path(model_path).stem

    # Try llama.cpp native (CLI)
    print(f"  Benchmarking with llama.cpp CLI...")
    r = benchmark_llama_cpp_native(model_path)
    results.append(r)

    # Try llama.cpp server
    if not llama_cpp_url:
        llama_cpp_url = detect_llama_cpp_server()
    if llama_cpp_url:
        print(f"  Benchmarking with llama.cpp server ({llama_cpp_url})...")
        r = benchmark_openai_api(llama_cpp_url, model_name, "llama.cpp-server")
        results.append(r)

    # Try LM Studio
    if not lm_studio_url:
        lm_studio_url = detect_lm_studio()
    if lm_studio_url:
        lm_models = get_lm_studio_models(lm_studio_url)
        if lm_models:
            print(f"  Benchmarking with LM Studio ({lm_studio_url})...")
            r = benchmark_openai_api(lm_studio_url, lm_models[0], "lm-studio")
            results.append(r)

    return results
