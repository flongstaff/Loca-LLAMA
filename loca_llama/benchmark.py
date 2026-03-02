"""Benchmark runner: test models on llama.cpp, LM Studio, and Ollama."""

import json
import re
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
    runtime: str  # "llama.cpp", "llama.cpp-server", "lm-studio", "ollama"
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_time_ms: float  # Time to process prompt (prefill)
    eval_time_ms: float  # Time to generate tokens
    total_time_ms: float
    tokens_per_second: float  # Generation speed
    prompt_tokens_per_second: float  # Prefill speed
    context_length: int
    success: bool
    run_number: int = 1
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def time_to_first_token_ms(self) -> float:
        return self.prompt_eval_time_ms


@dataclass
class RuntimeInfo:
    """Detected runtime with its URL and loaded models."""

    name: str  # "lm-studio", "llama.cpp-server", "ollama"
    url: str
    models: list[str]
    version: str | None = None


BENCH_PROMPTS = {
    "default": (
        "Write a detailed explanation of how neural networks learn through backpropagation. "
        "Include the mathematical foundations and provide a concrete example."
    ),
    "coding": (
        "Write a Python function that implements a binary search tree with insert, delete, "
        "and search operations. Include type hints and docstrings."
    ),
    "reasoning": (
        "A farmer has 17 sheep. All but 9 die. How many are left? "
        "Then explain: if you have a 3-gallon jug and a 5-gallon jug, "
        "how do you measure exactly 4 gallons? Show your step-by-step reasoning."
    ),
    "creative": (
        "Write a short story (3 paragraphs) about a robot who discovers it can dream. "
        "Use vivid imagery and emotional depth."
    ),
}

BENCH_MAX_TOKENS = 200


# ── Runtime Detection ────────────────────────────────────────────────────────

def detect_llama_cpp_server() -> RuntimeInfo | None:
    """Check if llama.cpp server is running and return info."""
    for port in [8080, 8081, 8000]:
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    # Try to get model info
                    models = []
                    try:
                        with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as r:
                            mdata = json.loads(r.read().decode())
                            models = [m["id"] for m in mdata.get("data", [])]
                    except Exception:
                        models = ["(loaded model)"]
                    return RuntimeInfo(
                        name="llama.cpp-server",
                        url=url,
                        models=models,
                    )
        except Exception:
            continue
    return None


def detect_lm_studio() -> RuntimeInfo | None:
    """Check if LM Studio API server is running."""
    for port in [1234, 1235]:
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                models = [m["id"] for m in data.get("data", [])]
                if models:
                    return RuntimeInfo(
                        name="lm-studio",
                        url=url,
                        models=models,
                    )
        except Exception:
            continue
    return None


def detect_ollama() -> RuntimeInfo | None:
    """Check if Ollama is running."""
    url = "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            # Get version
            version = None
            try:
                with urllib.request.urlopen(f"{url}/api/version", timeout=2) as vr:
                    vdata = json.loads(vr.read().decode())
                    version = vdata.get("version")
            except Exception:
                pass
            return RuntimeInfo(
                name="ollama",
                url=url,
                models=models,
                version=version,
            )
    except Exception:
        return None


def detect_all_runtimes() -> list[RuntimeInfo]:
    """Detect all running LLM runtimes."""
    runtimes = []
    for detector in [detect_lm_studio, detect_llama_cpp_server, detect_ollama]:
        info = detector()
        if info:
            runtimes.append(info)
    return runtimes


# ── Benchmarking ─────────────────────────────────────────────────────────────

def _make_fail_result(model_name: str, runtime: str, ctx: int, error: str, run: int = 1) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name, runtime=runtime,
        prompt_tokens=0, generated_tokens=0,
        prompt_eval_time_ms=0, eval_time_ms=0, total_time_ms=0,
        tokens_per_second=0, prompt_tokens_per_second=0,
        context_length=ctx, success=False, run_number=run, error=error,
    )


def benchmark_openai_api(
    base_url: str,
    model_id: str,
    runtime_name: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    run_number: int = 1,
) -> BenchmarkResult:
    """Run benchmark against an OpenAI-compatible API (LM Studio, llama.cpp, Ollama)."""
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}

    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return _make_fail_result(model_id, runtime_name, context_length, str(e), run_number)

    total_ms = (time.perf_counter() - start) * 1000
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Try to get timing from response (llama.cpp provides this)
    timings = data.get("timings", {})
    prompt_eval_ms = timings.get("prompt_eval_time_ms") or timings.get("prompt_ms", 0)
    eval_ms = timings.get("eval_time_ms") or timings.get("predicted_ms", 0)

    if not prompt_eval_ms and not eval_ms:
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
        run_number=run_number,
        extra=timings,
    )


def benchmark_ollama_api(
    model_name: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    run_number: int = 1,
) -> BenchmarkResult:
    """Run benchmark against Ollama's native API for precise timings."""
    url = "http://127.0.0.1:11434/api/generate"
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "num_ctx": context_length,
        },
    }).encode()

    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return _make_fail_result(model_name, "ollama", context_length, str(e), run_number)

    total_ms = (time.perf_counter() - start) * 1000

    # Ollama returns timings in nanoseconds
    prompt_eval_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    prompt_count = data.get("prompt_eval_count", 0)
    eval_count = data.get("eval_count", 0)

    prompt_eval_ms = prompt_eval_ns / 1_000_000
    eval_ms = eval_ns / 1_000_000

    tok_per_sec = (eval_count / eval_ms * 1000) if eval_ms > 0 else 0
    prompt_tok_per_sec = (prompt_count / prompt_eval_ms * 1000) if prompt_eval_ms > 0 else 0

    return BenchmarkResult(
        model_name=model_name,
        runtime="ollama",
        prompt_tokens=prompt_count,
        generated_tokens=eval_count,
        prompt_eval_time_ms=prompt_eval_ms,
        eval_time_ms=eval_ms,
        total_time_ms=total_ms,
        tokens_per_second=tok_per_sec,
        prompt_tokens_per_second=prompt_tok_per_sec,
        context_length=context_length,
        success=True,
        run_number=run_number,
        extra={
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
        },
    )


def benchmark_llama_cpp_native(
    model_path: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    n_gpu_layers: int = -1,
    run_number: int = 1,
) -> BenchmarkResult:
    """Run benchmark using llama.cpp CLI directly."""
    exe = shutil.which("llama-cli") or shutil.which("llama-cpp") or shutil.which("main")
    if not exe:
        return _make_fail_result(
            Path(model_path).stem, "llama.cpp-cli", context_length,
            "llama.cpp CLI not found (tried: llama-cli, llama-cpp, main)", run_number,
        )

    cmd = [
        exe, "-m", model_path, "-p", prompt,
        "-n", str(max_tokens), "-c", str(context_length),
        "-ngl", str(n_gpu_layers), "--log-disable",
    ]

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stderr + result.stdout
    except subprocess.TimeoutExpired:
        return _make_fail_result(Path(model_path).stem, "llama.cpp-cli", context_length, "Timed out (180s)", run_number)
    except Exception as e:
        return _make_fail_result(Path(model_path).stem, "llama.cpp-cli", context_length, str(e), run_number)

    total_ms = (time.perf_counter() - start) * 1000

    prompt_eval_ms = 0.0
    eval_ms = 0.0
    prompt_tokens = 0
    gen_tokens = 0

    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        prompt_eval_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))

    m = re.search(r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        eval_ms = float(m.group(1))
        gen_tokens = int(m.group(2))

    tok_per_sec = (gen_tokens / eval_ms * 1000) if eval_ms > 0 else 0
    prompt_tok_per_sec = (prompt_tokens / prompt_eval_ms * 1000) if prompt_eval_ms > 0 else 0

    return BenchmarkResult(
        model_name=Path(model_path).stem,
        runtime="llama.cpp-cli",
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        prompt_eval_time_ms=prompt_eval_ms,
        eval_time_ms=eval_ms,
        total_time_ms=total_ms,
        tokens_per_second=tok_per_sec,
        prompt_tokens_per_second=prompt_tok_per_sec,
        context_length=context_length,
        success=result.returncode == 0 and gen_tokens > 0,
        run_number=run_number,
        error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
    )


def run_benchmark_suite(
    runtime: RuntimeInfo,
    model_id: str,
    prompt_type: str = "default",
    num_runs: int = 3,
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    progress_callback=None,
) -> list[BenchmarkResult]:
    """Run a multi-round benchmark on a single runtime.

    Args:
        runtime: The runtime to benchmark against
        model_id: Model identifier for the runtime
        prompt_type: Key into BENCH_PROMPTS
        num_runs: Number of rounds (first is warmup)
        max_tokens: Max tokens to generate per run
        context_length: Context length to use
        progress_callback: Optional callable(run_number, total_runs) for progress updates
    """
    prompt = BENCH_PROMPTS.get(prompt_type, BENCH_PROMPTS["default"])
    results = []

    for i in range(1, num_runs + 1):
        if progress_callback:
            progress_callback(i, num_runs)

        if runtime.name == "ollama":
            r = benchmark_ollama_api(model_id, prompt, max_tokens, context_length, run_number=i)
        else:
            r = benchmark_openai_api(
                runtime.url, model_id, runtime.name, prompt, max_tokens, context_length, run_number=i,
            )
        results.append(r)

    return results


def aggregate_results(results: list[BenchmarkResult], skip_first: bool = True) -> dict:
    """Aggregate multiple benchmark runs, optionally skipping the warmup run."""
    successful = [r for r in results if r.success]
    if skip_first and len(successful) > 1:
        successful = successful[1:]  # Skip warmup

    if not successful:
        return {"success": False, "runs": 0}

    tok_speeds = [r.tokens_per_second for r in successful]
    prefill_speeds = [r.prompt_tokens_per_second for r in successful]
    ttfts = [r.time_to_first_token_ms for r in successful]

    return {
        "success": True,
        "runs": len(successful),
        "avg_tok_per_sec": sum(tok_speeds) / len(tok_speeds),
        "min_tok_per_sec": min(tok_speeds),
        "max_tok_per_sec": max(tok_speeds),
        "avg_prefill_tok_per_sec": sum(prefill_speeds) / len(prefill_speeds),
        "avg_ttft_ms": sum(ttfts) / len(ttfts),
        "avg_total_ms": sum(r.total_time_ms for r in successful) / len(successful),
        "total_tokens_generated": sum(r.generated_tokens for r in successful),
    }
