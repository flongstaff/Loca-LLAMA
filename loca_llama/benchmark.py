"""Benchmark runner: test models on LM Studio and llama.cpp."""

import json
import re
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    runtime: str  # "llama.cpp-server", "llama.cpp-cli", "lm-studio"
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_time_ms: float
    eval_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    prompt_tokens_per_second: float
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

    name: str  # "lm-studio", "llama.cpp-server"
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
    "json": (
        "Return a JSON object with exactly these keys: "
        '"name" (string), "version" (string), "features" (array of 5 strings), '
        '"config" (nested object with "debug": bool, "max_retries": int, "timeout_ms": int). '
        "Output only valid JSON, no markdown fences."
    ),
}

BENCH_MAX_TOKENS = 200


# ── Runtime Detection ────────────────────────────────────────────────────────

def detect_llama_cpp_server() -> RuntimeInfo | None:
    """Check if llama.cpp server is running."""
    for port in [8080, 8081, 8000]:
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    models = []
                    try:
                        with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as r:
                            mdata = json.loads(r.read().decode())
                            models = [m["id"] for m in mdata.get("data", [])]
                    except Exception:
                        models = ["(loaded model)"]
                    return RuntimeInfo(name="llama.cpp-server", url=url, models=models)
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
                    return RuntimeInfo(name="lm-studio", url=url, models=models)
        except Exception:
            continue
    return None


def detect_all_runtimes() -> list[RuntimeInfo]:
    """Detect all running LLM runtimes."""
    runtimes = []
    for detector in [detect_lm_studio, detect_llama_cpp_server]:
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
    """Run benchmark against an OpenAI-compatible API (LM Studio or llama.cpp)."""
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

    # llama.cpp provides precise timings
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


def benchmark_openai_api_streaming(
    base_url: str,
    model_id: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    timeout: int = 120,
) -> Generator[tuple[str, float], None, None]:
    """Stream tokens from an OpenAI-compatible API, yielding (token_text, elapsed_ms).

    Parses SSE lines in the format:
        data: {"choices":[{"delta":{"content":"token"}}]}
    Terminates on:
        data: [DONE]
    """
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            # Process complete lines
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data: "):
                    data_str = text[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        obj = json.loads(data_str)
                        delta = obj["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            elapsed = (time.perf_counter() - start) * 1000
                            yield (content, elapsed)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


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
    custom_prompt: str | None = None,
) -> list[BenchmarkResult]:
    """Run a multi-round benchmark. First run is warmup."""
    prompt = custom_prompt if custom_prompt else BENCH_PROMPTS.get(prompt_type, BENCH_PROMPTS["default"])
    results = []

    for i in range(1, num_runs + 1):
        if progress_callback:
            progress_callback(i, num_runs)
        r = benchmark_openai_api(
            runtime.url, model_id, runtime.name, prompt, max_tokens, context_length, run_number=i,
        )
        results.append(r)

    return results


def run_benchmark_sweep(
    runtime: RuntimeInfo,
    model_ids: list[str],
    prompt_type: str = "default",
    num_runs: int = 3,
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    combo_callback: object = None,
    run_callback: object = None,
    custom_prompt: str | None = None,
) -> list[dict]:
    """Run benchmark suite for multiple models sequentially.

    Returns list of dicts, one per model:
      {"model_id": str, "results": list[BenchmarkResult], "aggregate": dict}
    """
    combo_results: list[dict] = []

    for combo_idx, model_id in enumerate(model_ids):
        if combo_callback:
            combo_callback(combo_idx + 1, len(model_ids))

        results = run_benchmark_suite(
            runtime, model_id, prompt_type, num_runs, max_tokens,
            context_length, run_callback, custom_prompt,
        )
        agg = aggregate_results(results)
        combo_results.append({
            "model_id": model_id,
            "results": results,
            "aggregate": agg,
        })

    return combo_results


def aggregate_results(results: list[BenchmarkResult], skip_first: bool = True) -> dict:
    """Aggregate multiple benchmark runs, optionally skipping warmup."""
    successful = [r for r in results if r.success]
    if skip_first and len(successful) > 1:
        successful = successful[1:]

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
