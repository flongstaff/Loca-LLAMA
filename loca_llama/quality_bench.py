"""Quality benchmark: compare local models on coding tasks with automated scoring.

Tests coding correctness, instruction following, and reasoning quality.
Runs against any OpenAI-compatible API. Optionally compares with cloud models
via OpenRouter.

Usage (via CLI):
    loca-llama quality                          # All loaded models, local only
    loca-llama quality --model Qwen3.5-35B      # Single model
    loca-llama quality --compare pi             # Local vs Pi side-by-side
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .benchmark import detect_all_runtimes, RuntimeInfo


# --- Cloud Providers (via OpenRouter) ---
CLOUD_PROVIDERS: dict[str, dict[str, str]] = {
    "pi": {
        "base_url": "https://openrouter.ai/api",
        "model": "inflection/inflection-3-pi",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "claude": {
        "base_url": "https://openrouter.ai/api",
        "model": "anthropic/claude-sonnet-4-6-20250514",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


# --- Test Tasks ---
TASKS: list[dict[str, Any]] = [
    {
        "name": "fizzbuzz_variant",
        "category": "coding",
        "prompt": "Write a Python function `fizzbuzz(n: int) -> list[str]` that returns a list of strings from 1 to n. For multiples of 3 return 'Fizz', multiples of 5 return 'Buzz', multiples of both return 'FizzBuzz', otherwise the number as string. Include type hints.",
        "validation": {
            "must_contain": ["def fizzbuzz", "-> list[str]", "FizzBuzz", "Fizz", "Buzz"],
            "must_not_contain": ["import fizzbuzz"],
            "runnable": True,
            "test_code": textwrap.dedent("""\
                result = fizzbuzz(15)
                assert len(result) == 15, f"Expected 15 items, got {len(result)}"
                assert result[0] == "1", f"Expected '1', got {result[0]!r}"
                assert result[2] == "Fizz", f"Expected 'Fizz' at 3, got {result[2]!r}"
                assert result[4] == "Buzz", f"Expected 'Buzz' at 5, got {result[4]!r}"
                assert result[14] == "FizzBuzz", f"Expected 'FizzBuzz' at 15, got {result[14]!r}"
                assert result[6] == "7", f"Expected '7' at 7, got {result[6]!r}"
                print("PASS: fizzbuzz")
            """),
        },
    },
    {
        "name": "binary_search",
        "category": "coding",
        "prompt": "Write a Python function `binary_search(arr: list[int], target: int) -> int` that returns the index of target in a sorted array, or -1 if not found. Use iterative approach, not recursive. Include type hints.",
        "validation": {
            "must_contain": ["def binary_search", "-> int"],
            "must_not_contain": ["import bisect"],
            "runnable": True,
            "test_code": textwrap.dedent("""\
                assert binary_search([1, 3, 5, 7, 9], 5) == 2, "Should find 5 at index 2"
                assert binary_search([1, 3, 5, 7, 9], 1) == 0, "Should find 1 at index 0"
                assert binary_search([1, 3, 5, 7, 9], 9) == 4, "Should find 9 at index 4"
                assert binary_search([1, 3, 5, 7, 9], 4) == -1, "Should return -1 for missing"
                assert binary_search([], 1) == -1, "Should handle empty array"
                assert binary_search([1], 1) == 0, "Should handle single element"
                print("PASS: binary_search")
            """),
        },
    },
    {
        "name": "dataclass_model",
        "category": "coding",
        "prompt": "Write a Python dataclass `User` with fields: name (str), email (str), age (int, default 0). Add a method `is_adult() -> bool` that returns True if age >= 18. Add a `__post_init__` that validates email contains '@'. Raise ValueError if invalid.",
        "validation": {
            "must_contain": ["@dataclass", "class User", "is_adult", "__post_init__", "ValueError"],
            "must_not_contain": [],
            "runnable": True,
            "test_code": textwrap.dedent("""\
                u = User(name="Alice", email="alice@example.com", age=25)
                assert u.is_adult() is True, "25 should be adult"
                u2 = User(name="Bob", email="bob@example.com", age=17)
                assert u2.is_adult() is False, "17 should not be adult"
                u3 = User(name="Charlie", email="c@x.com")
                assert u3.age == 0, "Default age should be 0"
                try:
                    User(name="Bad", email="no-at-sign", age=20)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    pass
                print("PASS: dataclass_model")
            """),
        },
    },
    {
        "name": "error_handling",
        "category": "coding",
        "prompt": "Write a Python function `safe_divide(a: float, b: float) -> dict[str, Any]` that returns {'result': value, 'error': None} on success, or {'result': None, 'error': 'description'} on failure. Handle ZeroDivisionError and TypeError. Include type hints and `from typing import Any`.",
        "validation": {
            "must_contain": ["def safe_divide", "ZeroDivisionError", "TypeError", "-> dict"],
            "must_not_contain": [],
            "runnable": True,
            "test_code": textwrap.dedent("""\
                r = safe_divide(10, 2)
                assert r['result'] == 5.0, f"10/2 should be 5.0, got {r['result']}"
                assert r['error'] is None, "No error expected"
                r2 = safe_divide(10, 0)
                assert r2['result'] is None, "Division by zero should have no result"
                assert r2['error'] is not None, "Should have error message"
                r3 = safe_divide("a", 2)
                assert r3['result'] is None, "Type error should have no result"
                assert r3['error'] is not None, "Should have error message"
                print("PASS: error_handling")
            """),
        },
    },
    {
        "name": "reasoning_logic",
        "category": "reasoning",
        "prompt": "A farmer has 3 fields. Field A produces 20% more wheat than Field B. Field C produces 15% less than Field A. If Field B produces 500 kg, how much does each field produce and what is the total? Show your reasoning step by step, then give the final answer as a Python dict: {'A': ..., 'B': ..., 'C': ..., 'total': ...}",
        "validation": {
            "must_contain": ["600", "500", "510", "1610"],
            "must_not_contain": [],
            "runnable": False,
        },
    },
    {
        "name": "refactor_suggestion",
        "category": "code_review",
        "prompt": textwrap.dedent("""\
            Review this Python code and suggest improvements. List specific issues:

            ```python
            def process(data):
                result = []
                for i in range(len(data)):
                    if data[i] != None:
                        if type(data[i]) == str:
                            result.append(data[i].strip().lower())
                        elif type(data[i]) == int:
                            result.append(str(data[i]))
                result.sort()
                return result
            ```
        """),
        "validation": {
            "must_contain_any": [
                ["is not None", "isinstance", "type hint", "list comprehension", "enumerate"],
            ],
            "must_not_contain": [],
            "runnable": False,
        },
    },
]


@dataclass
class TaskResult:
    task_name: str
    model: str
    category: str
    response: str = ""
    contains_score: float = 0.0
    runnable_score: float = 0.0
    speed_tps: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    error: str = ""


# --- API Caller (stdlib-only, SSE streaming) ---

def call_openai_api(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    system_prompt: str = "You are a senior Python engineer. Write clean, correct, typed code. Be concise.",
) -> tuple[str, float, float, float]:
    """Call any OpenAI-compatible API with SSE streaming.

    Returns (response_text, tokens_per_second, ttft_ms, total_ms).
    Works with oMLX, LM Studio, OpenRouter, or any OpenAI-compatible endpoint.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
    }).encode()

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.monotonic()
    ttft = 0.0
    tokens_generated = 0
    response_text = ""

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft == 0.0:
                        ttft = (time.monotonic() - start) * 1000
                    response_text += content
                    tokens_generated += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    total_ms = (time.monotonic() - start) * 1000
    tps = tokens_generated / (total_ms / 1000) if total_ms > 0 else 0
    return response_text, tps, ttft, total_ms


# --- Code Extraction & Scoring ---

def extract_python_code(text: str) -> str:
    """Extract Python code from markdown fences, thinking tags, or raw text."""
    # Strip thinking tags (Qwen thinking mode wraps in <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown fence first
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)

    # Look for def/class/from/import at start of line
    lines = text.split("\n")
    code_lines: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "from ", "import ", "@")):
            in_code = True
        if in_code:
            if not stripped and code_lines:
                continue
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text


def score_task(task: dict[str, Any], response: str) -> tuple[float, float, str]:
    """Score response. Returns (contains_score, runnable_score, error)."""
    validation = task["validation"]

    must_contain = validation.get("must_contain", [])
    must_contain_any = validation.get("must_contain_any", [])
    must_not_contain = validation.get("must_not_contain", [])

    contains_hits = 0
    contains_total = len(must_contain) + len(must_contain_any)

    for s in must_contain:
        if s.lower() in response.lower():
            contains_hits += 1

    for group in must_contain_any:
        if any(s.lower() in response.lower() for s in group):
            contains_hits += 1

    for s in must_not_contain:
        if s.lower() in response.lower():
            contains_hits = max(0, contains_hits - 1)

    contains_score = contains_hits / contains_total if contains_total > 0 else 1.0

    if not validation.get("runnable", False):
        return contains_score, 0.0, ""

    test_code = validation.get("test_code", "")
    if not test_code:
        return contains_score, 0.0, ""

    code = extract_python_code(response)
    if not code:
        return contains_score, 0.0, "No code extracted"

    full_code = f"from dataclasses import dataclass\nfrom typing import Any\n\n{code}\n\n{test_code}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "PASS" in result.stdout:
            return contains_score, 1.0, ""
        err = result.stderr.strip().split("\n")[-1] if result.stderr else result.stdout.strip()
        return contains_score, 0.0, err[:200]
    except subprocess.TimeoutExpired:
        return contains_score, 0.0, "Timeout (10s)"
    except Exception as e:
        return contains_score, 0.0, str(e)[:200]


# --- Model Discovery ---

def get_models_from_api(base_url: str, api_key: str | None = None) -> list[str]:
    """Get available models from an OpenAI-compatible API."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


# --- Benchmark Runner ---

def run_quality_benchmark(
    base_url: str,
    models: list[str],
    api_key: str | None = None,
    runtime_name: str = "local",
) -> list[TaskResult]:
    """Run all quality tasks against specified models."""
    results: list[TaskResult] = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")

        for task in TASKS:
            print(f"  [{task['category']}] {task['name']}...", end=" ", flush=True)

            try:
                response, tps, ttft, total_ms = call_openai_api(
                    base_url, model, task["prompt"], api_key=api_key,
                )
                contains_score, runnable_score, error = score_task(task, response)

                r = TaskResult(
                    task_name=task["name"],
                    model=model,
                    category=task["category"],
                    response=response,
                    contains_score=contains_score,
                    runnable_score=runnable_score,
                    speed_tps=tps,
                    ttft_ms=ttft,
                    total_ms=total_ms,
                    error=error,
                )
                results.append(r)

                if task["validation"].get("runnable"):
                    status = "PASS" if runnable_score == 1.0 else "FAIL"
                    print(f"{status} contains={contains_score:.0%} run={runnable_score:.0%} {tps:.0f}t/s {ttft:.0f}ms TTFT")
                else:
                    status = "PASS" if contains_score >= 0.8 else "WARN" if contains_score >= 0.5 else "FAIL"
                    print(f"{status} contains={contains_score:.0%} {tps:.0f}t/s {ttft:.0f}ms TTFT")

                if error:
                    print(f"       error: {error}")

            except Exception as e:
                print(f"ERROR: {e}")
                results.append(TaskResult(
                    task_name=task["name"], model=model, category=task["category"], error=str(e),
                ))

    return results


def run_quality_comparison(
    local_base_url: str,
    local_model: str,
    cloud_provider: str,
    local_api_key: str | None = None,
) -> tuple[list[TaskResult], list[TaskResult]]:
    """Run quality benchmark on both local model and cloud provider.

    Returns (local_results, cloud_results).
    """
    provider = CLOUD_PROVIDERS.get(cloud_provider)
    if not provider:
        raise ValueError(f"Unknown cloud provider: {cloud_provider}. Options: {list(CLOUD_PROVIDERS.keys())}")

    cloud_api_key = os.environ.get(provider["api_key_env"])
    if not cloud_api_key:
        raise ValueError(
            f"Set {provider['api_key_env']} environment variable for {cloud_provider} comparison.\n"
            f"Get an API key at https://openrouter.ai/settings/keys"
        )

    # Warn about costs
    print(f"\n  Cloud comparison: {cloud_provider} ({provider['model']})")
    print(f"  API costs apply for cloud requests via OpenRouter.\n")

    # Run local
    print("--- LOCAL ---")
    local_results = run_quality_benchmark(
        local_base_url, [local_model], api_key=local_api_key, runtime_name="local",
    )

    # Run cloud
    print("\n--- CLOUD ---")
    cloud_results = run_quality_benchmark(
        provider["base_url"], [provider["model"]], api_key=cloud_api_key, runtime_name=cloud_provider,
    )

    return local_results, cloud_results


# --- Summary Printing ---

def print_quality_summary(results: list[TaskResult]) -> None:
    """Print quality benchmark summary table."""
    models = sorted(set(r.model for r in results))

    print(f"\n{'='*80}")
    print("  QUALITY BENCHMARK RESULTS")
    print(f"{'='*80}\n")

    header = f"{'Model':<45} {'Pass':>5} {'Contain':>8} {'Run':>5} {'TPS':>6} {'TTFT':>7}"
    print(header)
    print("-" * len(header))

    for model in models:
        mr = [r for r in results if r.model == model]
        runnable_tasks = [
            r for r in mr
            if any(t["name"] == r.task_name and t["validation"].get("runnable") for t in TASKS)
        ]

        pass_count = sum(1 for r in runnable_tasks if r.runnable_score == 1.0)
        total_runnable = len(runnable_tasks)
        avg_contains = sum(r.contains_score for r in mr) / len(mr) if mr else 0
        avg_run = sum(r.runnable_score for r in runnable_tasks) / len(runnable_tasks) if runnable_tasks else 0
        avg_tps = sum(r.speed_tps for r in mr if r.speed_tps > 0) / max(1, sum(1 for r in mr if r.speed_tps > 0))
        avg_ttft = sum(r.ttft_ms for r in mr if r.ttft_ms > 0) / max(1, sum(1 for r in mr if r.ttft_ms > 0))

        short_name = model[:43]
        print(f"{short_name:<45} {pass_count}/{total_runnable:>3} {avg_contains:>7.0%} {avg_run:>4.0%} {avg_tps:>5.0f} {avg_ttft:>6.0f}ms")

    # Per-task breakdown
    print(f"\n{'Task':<25} ", end="")
    for model in models:
        short = model[:18]
        print(f"{short:>20} ", end="")
    print()
    print("-" * (26 + 21 * len(models)))

    for task in TASKS:
        print(f"{task['name']:<25} ", end="")
        for model in models:
            r = next((r for r in results if r.model == model and r.task_name == task["name"]), None)
            if r is None:
                print(f"{'--':>20} ", end="")
            elif task["validation"].get("runnable"):
                status = "PASS" if r.runnable_score == 1.0 else "FAIL"
                print(f"{status} {r.speed_tps:>5.0f}t/s {r.ttft_ms:>5.0f}ms ", end="")
            else:
                status = "PASS" if r.contains_score >= 0.8 else "FAIL"
                print(f"{status} {r.speed_tps:>5.0f}t/s {r.ttft_ms:>5.0f}ms ", end="")
        print()


def results_to_record(
    results: list[TaskResult],
    runtime_name: str,
    cloud_provider: str = "",
    cloud_results: list[TaskResult] | None = None,
) -> "BenchmarkRecord":
    """Convert TaskResults to a BenchmarkRecord for saving."""
    from .benchmark_results import BenchmarkRecord

    if not results:
        return BenchmarkRecord(type="quality", model="unknown", runtime=runtime_name)

    model = results[0].model
    runnable_tasks = [
        r for r in results
        if any(t["name"] == r.task_name and t["validation"].get("runnable") for t in TASKS)
    ]
    pass_count = sum(1 for r in runnable_tasks if r.runnable_score == 1.0)
    total_runnable = len(runnable_tasks)
    avg_contains = sum(r.contains_score for r in results) / len(results) if results else 0
    avg_tps = sum(r.speed_tps for r in results if r.speed_tps > 0) / max(1, sum(1 for r in results if r.speed_tps > 0))

    task_details = [
        {
            "name": r.task_name,
            "category": r.category,
            "contains": r.contains_score,
            "runnable": r.runnable_score,
            "tps": r.speed_tps,
            "ttft_ms": r.ttft_ms,
            "error": r.error,
        }
        for r in results
    ]

    quality_scores: dict[str, Any] = {
        "pass_rate": pass_count / total_runnable if total_runnable else 0,
        "avg_contains": avg_contains,
        "pass_count": pass_count,
        "total_runnable": total_runnable,
        "tasks": task_details,
    }

    cloud_scores: dict[str, Any] = {}
    if cloud_results:
        cloud_runnable = [
            r for r in cloud_results
            if any(t["name"] == r.task_name and t["validation"].get("runnable") for t in TASKS)
        ]
        cloud_pass = sum(1 for r in cloud_runnable if r.runnable_score == 1.0)
        cloud_scores = {
            "model": cloud_results[0].model if cloud_results else "",
            "pass_rate": cloud_pass / len(cloud_runnable) if cloud_runnable else 0,
            "tasks": [
                {
                    "name": r.task_name,
                    "contains": r.contains_score,
                    "runnable": r.runnable_score,
                    "tps": r.speed_tps,
                    "ttft_ms": r.ttft_ms,
                }
                for r in cloud_results
            ],
        }

    return BenchmarkRecord(
        type="quality",
        model=model,
        runtime=runtime_name,
        tokens_per_second=avg_tps,
        quality_scores=quality_scores,
        cloud_provider=cloud_provider,
        cloud_scores=cloud_scores,
    )
