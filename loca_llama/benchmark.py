"""Benchmark runner: test models on LM Studio and llama.cpp."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    preset_name: str | None = None  # LM Studio preset name (e.g., "35b thinking general")
    preset_config: dict | None = None  # Full preset configuration dict

    @property
    def time_to_first_token_ms(self) -> float:
        return self.prompt_eval_time_ms


@dataclass
class RuntimeInfo:
    """Detected runtime with its URL and loaded models."""

    name: str  # "lm-studio", "llama.cpp-server", "litellm-proxy"
    url: str
    models: list[str]
    version: str | None = None
    api_key: str | None = None


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

def detect_litellm() -> RuntimeInfo | None:
    """Check if LiteLLM proxy is running (routes to LM Studio or llama.cpp)."""
    url = "http://127.0.0.1:4000"
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            if data.get("healthy_endpoints") or data.get("status") == "healthy":
                models = []
                try:
                    with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as r:
                        mdata = json.loads(r.read().decode())
                        models = [m["id"] for m in mdata.get("data", [])]
                except Exception:
                    pass
                return RuntimeInfo(name="litellm-proxy", url=url, models=models, api_key="sk-local")
    except Exception:
        pass
    return None


def detect_omlx() -> RuntimeInfo | None:
    """Check if oMLX server is running."""
    url = os.environ.get("OMLX_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("OMLX_API_KEY", "9514")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "healthy":
                models = []
                try:
                    req = urllib.request.Request(
                        f"{url}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    with urllib.request.urlopen(req, timeout=2) as r:
                        mdata = json.loads(r.read().decode())
                        models = [m["id"] for m in mdata.get("data", [])]
                except Exception:
                    models = ["(omlx model)"]
                return RuntimeInfo(name="omlx", url=url, models=models, api_key=api_key)
    except Exception:
        pass
    return None


def detect_llama_cpp_server() -> RuntimeInfo | None:
    """Check if llama.cpp server is running."""
    for port in [8082, 8080, 8081]:
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


def get_lm_studio_preset_info(model_id: str, base_url: str = "http://127.0.0.1:1234") -> dict | None:
    """Fetch preset information from LM Studio for a loaded model.

    LM Studio stores presets per model. This attempts to fetch the current
    preset configuration for the specified model.

    Returns dict with preset info, or None if not available.
    """
    # LM Studio doesn't have a direct API for presets, but we can infer from
    # the model's loaded configuration and known preset patterns
    try:
        # Try to get model details which may include preset info
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            for model in data.get("data", []):
                if model["id"] == model_id:
                    preset_info = model.get("preset_info") or model.get("config", {})
                    if preset_info:
                        return preset_info
    except Exception:
        pass

    # Fallback: try common preset endpoints
    # LM Studio v0.2+ uses /api/presets endpoint
    preset_endpoints = [
        f"{base_url}/api/presets",
        f"{base_url}/api/model-presets",
        f"{base_url}/presets",
    ]

    for endpoint in preset_endpoints:
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as resp:
                presets_data = json.loads(resp.read().decode())
                # Look for preset matching this model
                if isinstance(presets_data, list):
                    for preset in presets_data:
                        if preset.get("modelId") == model_id or preset.get("model") == model_id:
                            return preset
                elif isinstance(presets_data, dict):
                    # Try to find preset by model_id
                    if presets_data.get("modelId") == model_id:
                        return presets_data
                    for key, value in presets_data.items():
                        if isinstance(value, dict) and value.get("modelId") == model_id:
                            return value
        except Exception:
            continue

    return None


def detect_all_runtimes() -> list[RuntimeInfo]:
    """Detect all running LLM runtimes.

    LiteLLM proxy is checked first — if it's running, it can route to
    whichever backend (LM Studio or llama.cpp) is up, so you only need
    one model loaded.  Direct backends are still detected for fallback.
    """
    runtimes = []
    for detector in [detect_omlx, detect_litellm, detect_lm_studio, detect_llama_cpp_server]:
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


def format_benchmark_error(exc: Exception, runtime_name: str, url: str) -> str:
    """Convert raw exceptions to user-friendly benchmark error messages."""
    if isinstance(exc, urllib.error.HTTPError):
        # Try to read error body for "Channel Error" (MLX model crash in LM Studio)
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if "channel" in body.lower() or "channel" in str(exc.reason).lower():
            return f"Model crashed in {runtime_name} (Channel Error). The model may be unstable — try a smaller quantization or different model."
        if exc.code >= 500:
            return f"{runtime_name} internal error ({exc.code}). Try restarting the server."
        return f"{runtime_name} rejected the request: {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (ConnectionRefusedError, OSError)):
            return f"Cannot connect to {runtime_name} at {url}. Is the server running?"
        reason_str = str(reason)
        if "connection reset" in reason_str.lower() or "broken pipe" in reason_str.lower():
            return f"Model crashed in {runtime_name} (connection reset). The model may be unstable."
        return f"Network error connecting to {runtime_name}: {reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return f"Request to {runtime_name} timed out. The model may be too large or the server overloaded."
    if isinstance(exc, json.JSONDecodeError):
        return f"Invalid response from {runtime_name}. The server may be misconfigured."
    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError, BrokenPipeError)):
        return f"Cannot connect to {runtime_name} at {url}. Is the server running?"
    # http.client.IncompleteRead — connection dropped mid-stream (model crashed)
    exc_name = type(exc).__name__
    if exc_name == "IncompleteRead" or "incomplete" in str(exc).lower():
        return f"Model crashed in {runtime_name} mid-generation (connection dropped). The model may be unstable."
    return f"Benchmark error: {exc_name}"


def _is_model_crash(error: str | None) -> bool:
    """Check if a benchmark error indicates a model crash (not transient)."""
    if not error:
        return False
    crash_indicators = ["crashed", "channel error", "connection reset", "unstable", "segmentation"]
    return any(ind in error.lower() for ind in crash_indicators)


def _wait_for_server_ready(base_url: str, timeout: int = 30) -> bool:
    """Wait for an OpenAI-compatible server to respond to /v1/models."""
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def benchmark_openai_api(
    base_url: str,
    model_id: str,
    runtime_name: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    run_number: int = 1,
    api_key: str | None = None,
    preset_name: str | None = None,
    preset_config: dict | None = None,
) -> BenchmarkResult:
    """Run benchmark against an OpenAI-compatible API (LM Studio, llama.cpp, or LiteLLM).

    Uses streaming to measure real TTFT (time to first token) and generation
    speed.  Falls back to server-provided timings when available (llama.cpp).
    """
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.perf_counter()
    token_count = 0
    ttft_ms = 0.0
    first_token_time = 0.0
    last_token_time = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    timings: dict = {}
    token_timestamps: list[float] = []  # elapsed ms for each token

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text or text.startswith(":"):
                    continue
                if not text.startswith("data: "):
                    continue
                data_str = text[6:]
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # Server timings / usage in final chunk (llama.cpp)
                if "timings" in obj:
                    timings = obj["timings"]
                if "usage" in obj:
                    u = obj["usage"]
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)
                # Content delta → token
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        elapsed = (time.perf_counter() - start) * 1000
                        token_count += 1
                        token_timestamps.append(elapsed)
                        if token_count == 1:
                            ttft_ms = elapsed
                            first_token_time = elapsed
                        last_token_time = elapsed
    except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, socket.gaierror, ConnectionResetError, BrokenPipeError, ConnectionRefusedError) as e:
        # Network errors - model crashed or server unreachable
        error_msg = format_benchmark_error(e, runtime_name, base_url)
        return _make_fail_result(model_id, runtime_name, context_length, error_msg, run_number)
    except Exception as e:
        # Unexpected errors
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        return _make_fail_result(model_id, runtime_name, context_length, error_msg, run_number)

    total_ms = (time.perf_counter() - start) * 1000
    if not completion_tokens:
        completion_tokens = token_count

    # Estimate prompt tokens if server didn't report them (LM Studio streaming)
    if not prompt_tokens:
        prompt_tokens = max(len(prompt) // 4, 1)

    # Prefer server-provided timings (llama.cpp gives precise values)
    prompt_eval_ms = timings.get("prompt_eval_time_ms") or timings.get("prompt_ms", 0)
    eval_ms = timings.get("eval_time_ms") or timings.get("predicted_ms", 0)

    if not prompt_eval_ms:
        # Use measured TTFT as prompt eval time
        prompt_eval_ms = ttft_ms if ttft_ms > 0 else total_ms * 0.3

    if not eval_ms:
        # Generation time = time between first and last token
        if token_count > 1 and last_token_time > first_token_time:
            eval_ms = last_token_time - first_token_time
        else:
            eval_ms = total_ms - prompt_eval_ms if prompt_eval_ms < total_ms else total_ms * 0.7

    # Tokens counted from stream deltas; gen speed uses (tokens - 1) for inter-token interval
    gen_tokens_for_speed = max(completion_tokens - 1, 1) if token_count > 1 else completion_tokens
    tok_per_sec = (gen_tokens_for_speed / eval_ms * 1000) if eval_ms > 0 else 0
    prompt_tok_per_sec = (prompt_tokens / prompt_eval_ms * 1000) if prompt_eval_ms > 0 and prompt_tokens > 0 else 0

    # Detect silent failures: no tokens generated means benchmark failed
    success = token_count > 0
    error = None
    if not success:
        error = "No tokens generated (model may have crashed or rejected request)"

    # Compute per-token latencies (inter-token intervals)
    per_token_latencies: list[float] = []
    if len(token_timestamps) > 1:
        per_token_latencies = [
            token_timestamps[i] - token_timestamps[i - 1]
            for i in range(1, len(token_timestamps))
        ]

    extra_data = dict(timings)
    if per_token_latencies:
        extra_data["per_token_latencies"] = per_token_latencies

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
        success=success,
        run_number=run_number,
        error=error,
        extra=extra_data,
        preset_name=preset_name,
        preset_config=preset_config,
    )


def benchmark_openai_api_streaming(
    base_url: str,
    model_id: str,
    prompt: str = BENCH_PROMPTS["default"],
    max_tokens: int = BENCH_MAX_TOKENS,
    timeout: int = 120,
    api_key: str | None = None,
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

    headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            text = raw_line.decode("utf-8", errors="replace").strip()
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
        "-ngl", str(n_gpu_layers), "-fa", "on",
        "--no-conversation",
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
    tok_per_sec = 0.0
    prompt_tok_per_sec = 0.0

    # --- Old format: llama_print_timings (builds before ~b8000) ---
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        prompt_eval_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))

    m = re.search(r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        eval_ms = float(m.group(1))
        gen_tokens = int(m.group(2))

    # --- New format: [ Prompt: 132.3 t/s | Generation: 28.3 t/s ] (builds b8000+) ---
    if gen_tokens == 0:
        m_new = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", output)
        if m_new:
            prompt_tok_per_sec = float(m_new.group(1))
            tok_per_sec = float(m_new.group(2))
            # Estimate tokens from total time and rates
            prompt_tokens = max(len(prompt) // 4, 1)
            gen_tokens = max_tokens
            if prompt_tok_per_sec > 0:
                prompt_eval_ms = (prompt_tokens / prompt_tok_per_sec) * 1000
            if tok_per_sec > 0:
                eval_ms = (gen_tokens / tok_per_sec) * 1000

    if tok_per_sec == 0 and eval_ms > 0:
        tok_per_sec = (gen_tokens / eval_ms * 1000)
    if prompt_tok_per_sec == 0 and prompt_eval_ms > 0:
        prompt_tok_per_sec = (prompt_tokens / prompt_eval_ms * 1000)

    parsed = gen_tokens > 0
    error = None
    if result.returncode != 0:
        error = f"Exit code: {result.returncode}"
    elif not parsed:
        error = "Could not parse timing output from llama-cli"

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
        success=result.returncode == 0 and parsed,
        run_number=run_number,
        error=error,
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
    preset_name: str | None = None,
    preset_config: dict | None = None,
) -> list[BenchmarkResult]:
    """Run a multi-round benchmark. First run is warmup.

    Aborts early if the model crashes repeatedly (e.g. MLX segfaults).
    After a crash, waits for the server to recover before retrying.
    """
    prompt = custom_prompt if custom_prompt else BENCH_PROMPTS.get(prompt_type, BENCH_PROMPTS["default"])
    results = []
    consecutive_crashes = 0
    max_consecutive_crashes = 2

    for i in range(1, num_runs + 1):
        if progress_callback:
            progress_callback(i, num_runs)

        # After a crash, wait for the server to reload the model
        if consecutive_crashes > 0:
            if not _wait_for_server_ready(runtime.url, timeout=30):
                results.append(_make_fail_result(
                    model_id, runtime.name, context_length,
                    "Server did not recover after model crash", i,
                ))
                break

        r = benchmark_openai_api(
            runtime.url, model_id, runtime.name, prompt, max_tokens, context_length,
            run_number=i, api_key=runtime.api_key,
            preset_name=preset_name, preset_config=preset_config,
        )
        results.append(r)

        if not r.success and _is_model_crash(r.error):
            consecutive_crashes += 1
            if consecutive_crashes >= max_consecutive_crashes:
                # Fill remaining runs with the same error so callers see full count
                for j in range(i + 1, num_runs + 1):
                    results.append(_make_fail_result(
                        model_id, runtime.name, context_length,
                        f"Skipped: model crashed {consecutive_crashes} times consecutively", j,
                    ))
                break
        else:
            consecutive_crashes = 0

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


def benchmark_with_server(
    model_path: str,
    prompt_type: str = "default",
    max_tokens: int = BENCH_MAX_TOKENS,
    context_length: int = 4096,
    n_gpu_layers: int = -1,
    num_runs: int = 3,
    port: int = 8099,
    progress_callback: object = None,
    custom_prompt: str | None = None,
) -> list[BenchmarkResult]:
    """Start a temporary llama-server, benchmark against it, then stop and unload.

    Ensures the model is loaded before benchmarking and fully unloaded
    (process terminated, memory freed) when done or on error.
    """
    model_name = Path(model_path).stem

    exe = shutil.which("llama-server") or shutil.which("server")
    if not exe:
        return [_make_fail_result(
            model_name, "llama.cpp-server", context_length,
            "llama-server not found in PATH", i,
        ) for i in range(1, num_runs + 1)]

    cmd = [
        exe, "-m", model_path,
        "--port", str(port),
        "-c", str(context_length),
        "-ngl", str(n_gpu_layers),
        "-fa", "on",
    ]

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to become healthy
        base_url = f"http://127.0.0.1:{port}"
        healthy = False
        for _ in range(60):
            time.sleep(1)
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        healthy = True
                        break
            except Exception:
                continue

        if not healthy:
            return [_make_fail_result(
                model_name, "llama.cpp-server", context_length,
                "Server failed to start or load model within 60s", i,
            ) for i in range(1, num_runs + 1)]

        # Discover model ID from the running server
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as resp:
                mdata = json.loads(resp.read().decode())
                models = [m["id"] for m in mdata.get("data", [])]
                model_id = models[0] if models else model_name
        except Exception:
            model_id = model_name

        runtime = RuntimeInfo(name="llama.cpp-server", url=base_url, models=[model_id])

        results = run_benchmark_suite(
            runtime, model_id, prompt_type, num_runs, max_tokens,
            context_length, progress_callback, custom_prompt,
        )
        return results

    finally:
        # Always terminate the server to unload the model and free memory
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (p / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def aggregate_results(results: list[BenchmarkResult], skip_first: bool = True) -> dict:
    """Aggregate multiple benchmark runs, optionally skipping warmup.

    Includes percentiles (p50/p90/p95/p99) for tok/s and TTFT,
    per-token latency distribution, and auto-warmup detection.
    """
    import statistics

    successful = [r for r in results if r.success]

    # Auto-warmup detection
    warmup_excluded = False
    if skip_first and len(successful) > 2:
        first_speed = successful[0].tokens_per_second
        rest_speeds = [r.tokens_per_second for r in successful[1:]]
        rest_mean = sum(rest_speeds) / len(rest_speeds) if rest_speeds else 0
        if rest_mean > 0 and first_speed < rest_mean * 0.8:
            warmup_excluded = True
        successful = successful[1:]
    elif skip_first and len(successful) > 1:
        successful = successful[1:]

    if not successful:
        return {"success": False, "runs": 0}

    tok_speeds = [r.tokens_per_second for r in successful]
    prefill_speeds = [r.prompt_tokens_per_second for r in successful]
    ttfts = [r.time_to_first_token_ms for r in successful]

    # Collect all per-token latencies across runs
    all_per_token: list[float] = []
    for r in successful:
        latencies = r.extra.get("per_token_latencies", [])
        if isinstance(latencies, list):
            all_per_token.extend(latencies)

    return {
        "success": True,
        "runs": len(successful),
        "warmup_excluded": warmup_excluded,
        # tok/s stats
        "avg_tok_per_sec": sum(tok_speeds) / len(tok_speeds),
        "min_tok_per_sec": min(tok_speeds),
        "max_tok_per_sec": max(tok_speeds),
        "median_tok_per_sec": statistics.median(tok_speeds),
        "stddev_tok_per_sec": statistics.stdev(tok_speeds) if len(tok_speeds) >= 2 else 0.0,
        # tok/s percentiles
        "p50_tok_per_sec": _percentile(tok_speeds, 50),
        "p90_tok_per_sec": _percentile(tok_speeds, 90),
        "p95_tok_per_sec": _percentile(tok_speeds, 95),
        "p99_tok_per_sec": _percentile(tok_speeds, 99),
        # TTFT percentiles
        "avg_ttft_ms": sum(ttfts) / len(ttfts),
        "p50_ttft_ms": _percentile(ttfts, 50),
        "p90_ttft_ms": _percentile(ttfts, 90),
        "p95_ttft_ms": _percentile(ttfts, 95),
        "p99_ttft_ms": _percentile(ttfts, 99),
        # Prefill + totals
        "avg_prefill_tok_per_sec": sum(prefill_speeds) / len(prefill_speeds),
        "avg_total_ms": sum(r.total_time_ms for r in successful) / len(successful),
        "total_tokens_generated": sum(r.generated_tokens for r in successful),
        # Per-token latency distribution
        "per_token_latency_p50_ms": _percentile(all_per_token, 50) if all_per_token else 0.0,
        "per_token_latency_p90_ms": _percentile(all_per_token, 90) if all_per_token else 0.0,
        "per_token_latency_p99_ms": _percentile(all_per_token, 99) if all_per_token else 0.0,
    }
