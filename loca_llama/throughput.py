"""Concurrent throughput testing for LLM inference servers.

Adapted from patterns in llama-throughput-lab: sends concurrent requests
via ThreadPoolExecutor to measure aggregate multi-user throughput.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestResult:
    """Result of a single concurrent request."""

    request_id: int
    success: bool
    tokens_generated: int = 0
    elapsed_ms: float = 0.0
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    error: str | None = None


@dataclass
class ThroughputResult:
    """Aggregate result of a concurrent throughput test."""

    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    elapsed_seconds: float
    throughput_tps: float  # aggregate tokens/sec across all concurrent requests
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_ttft_ms: float
    error_rate: float
    # Latency percentiles
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    per_request: list[RequestResult] = field(default_factory=list)


def _post_json_with_retry(
    url: str,
    payload: dict,
    headers: dict[str, str],
    max_retries: int = 2,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST JSON with exponential backoff retry on transient errors."""
    data = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
    return {}  # unreachable


def _single_request(
    request_id: int,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
) -> RequestResult:
    """Execute a single non-streaming inference request and measure timing."""
    url = f"{base_url}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        resp_data = _post_json_with_retry(url, payload, headers)
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = resp_data.get("usage", {})
        tokens = usage.get("completion_tokens", 0)

        # Fallback: estimate tokens from response content
        if tokens == 0:
            choices = resp_data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                tokens = max(len(content) // 4, 1) if content else 0

        tok_per_sec = (tokens / elapsed_ms * 1000) if elapsed_ms > 0 and tokens > 0 else 0.0

        return RequestResult(
            request_id=request_id,
            success=tokens > 0,
            tokens_generated=tokens,
            elapsed_ms=elapsed_ms,
            tokens_per_second=tok_per_sec,
            error="No tokens generated" if tokens == 0 else None,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            success=False,
            elapsed_ms=elapsed_ms,
            error=f"{type(e).__name__}: {e}",
        )


def run_throughput_test(
    base_url: str,
    model_id: str,
    concurrency: int = 4,
    total_requests: int = 8,
    prompt: str = "Write a brief explanation of how neural networks work.",
    max_tokens: int = 100,
    api_key: str | None = None,
    progress_callback: Any = None,
) -> ThroughputResult:
    """Run a concurrent throughput test against an OpenAI-compatible API.

    Sends `total_requests` requests with `concurrency` workers in parallel.
    Measures aggregate throughput (total tokens / wall-clock time).
    """
    results: list[RequestResult] = []
    completed = 0

    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _single_request, i, base_url, model_id, prompt, max_tokens, api_key
            ): i
            for i in range(total_requests)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, total_requests)

    overall_elapsed = time.perf_counter() - overall_start

    # Sort by request_id for consistent ordering
    results.sort(key=lambda r: r.request_id)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_tokens = sum(r.tokens_generated for r in successful)
    throughput = total_tokens / overall_elapsed if overall_elapsed > 0 else 0.0

    latencies = [r.elapsed_ms for r in successful] if successful else [0.0]
    sorted_latencies = sorted(latencies)

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        idx = (p / 100) * (len(vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])

    return ThroughputResult(
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_tokens=total_tokens,
        elapsed_seconds=overall_elapsed,
        throughput_tps=throughput,
        avg_latency_ms=sum(latencies) / len(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        avg_ttft_ms=0.0,  # non-streaming doesn't measure TTFT
        error_rate=len(failed) / total_requests if total_requests > 0 else 0.0,
        p50_latency_ms=_pct(sorted_latencies, 50),
        p90_latency_ms=_pct(sorted_latencies, 90),
        p99_latency_ms=_pct(sorted_latencies, 99),
        per_request=results,
    )


def run_throughput_ramp(
    base_url: str,
    model_id: str,
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 4,
    prompt: str = "Explain the concept of recursion in programming. Be concise.",
    max_tokens: int = 100,
    api_key: str | None = None,
    progress_callback: Any = None,
) -> list[ThroughputResult]:
    """Run throughput tests at increasing concurrency to find saturation point.

    Returns one ThroughputResult per concurrency level.
    """
    levels = concurrency_levels or [1, 2, 4, 8]
    results: list[ThroughputResult] = []

    for level in levels:
        print(f"  Concurrency {level}...", end=" ", flush=True)
        result = run_throughput_test(
            base_url=base_url,
            model_id=model_id,
            concurrency=level,
            total_requests=max(requests_per_level, level),
            prompt=prompt,
            max_tokens=max_tokens,
            api_key=api_key,
            progress_callback=progress_callback,
        )
        results.append(result)
        print(f"{result.throughput_tps:.1f} tok/s, "
              f"p50={result.p50_latency_ms:.0f}ms, "
              f"p90={result.p90_latency_ms:.0f}ms")

    return results
