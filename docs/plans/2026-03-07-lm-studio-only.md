# LM Studio Only + Benchmark Fixes + Coding Tool Benchmarking

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify local LLM infrastructure to LM Studio as single backend, fix benchmark model reload bugs, improve crash handling, and add coding tool (opencode/aider/goose) benchmarking to loca-llama.

**Architecture:** LM Studio (:1234) becomes the only runtime backend. LiteLLM and llama.cpp code paths are deprioritized (kept but not default). Runtime detection order changes to LM Studio first. Benchmark suite gets per-run model availability checks and better crash recovery. A new coding tool benchmark module wraps opencode/aider/goose CLI invocations to measure coding task performance through local models.

**Tech Stack:** Python 3.11+, FastAPI, LM Studio API (OpenAI-compatible on :1234), subprocess for CLI tool invocation.

---

## Task 1: Reorder Runtime Detection — LM Studio First

**Files:**
- Modify: `loca_llama/benchmark.py:138-150` (`detect_all_runtimes`)
- Modify: `loca_llama/runtime.py:288-310` (`detect_all_connectors`)
- Test: `tests/test_runtime_benchmark.py`

**Step 1: Update `detect_all_runtimes()` in benchmark.py**

Change the detector order so LM Studio is checked first, LiteLLM second, llama.cpp last:

```python
def detect_all_runtimes() -> list[RuntimeInfo]:
    """Detect all running LLM runtimes.

    LM Studio is the primary backend. LiteLLM and llama.cpp are
    kept as fallbacks but deprioritized.
    """
    runtimes = []
    for detector in [detect_lm_studio, detect_litellm, detect_llama_cpp_server]:
        info = detector()
        if info:
            runtimes.append(info)
    return runtimes
```

**Step 2: Update `detect_all_connectors()` in runtime.py**

```python
def detect_all_connectors() -> dict[str, object]:
    """Detect and return all available runtime connectors.

    LM Studio is the primary backend. LiteLLM and llama.cpp are fallbacks.
    """
    connectors = {}

    lms = LMStudioConnector()
    if lms.is_running():
        connectors["lm-studio"] = lms

    litellm = LiteLLMConnector()
    if litellm.is_running():
        connectors["litellm"] = litellm

    for port in [8082, 8080, 8081, 8000]:
        lcp = LlamaCppConnector(f"http://127.0.0.1:{port}")
        if lcp.is_running():
            connectors[f"llama.cpp:{port}"] = lcp
            break

    return connectors
```

**Step 3: Run tests**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_runtime_benchmark.py -v -k "detect" 2>/dev/null || echo "No matching tests"`

**Step 4: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/benchmark.py loca_llama/runtime.py
git commit -m "refactor: prioritize LM Studio in runtime detection order

LM Studio is now checked first as the primary backend.
LiteLLM and llama.cpp kept as fallbacks but deprioritized."
```

---

## Task 2: Add Per-Run Model Availability Check to Benchmark Suite

**Files:**
- Modify: `loca_llama/benchmark.py:487-539` (`run_benchmark_suite`)
- Test: `tests/test_runtime_benchmark.py`

**Step 1: Write the failing test**

Add to `tests/test_runtime_benchmark.py`:

```python
def test_benchmark_suite_checks_model_between_runs(monkeypatch):
    """Benchmark suite should verify model is still loaded before each run."""
    call_count = {"check": 0, "bench": 0}

    def mock_check_model(base_url, model_id, timeout=5):
        call_count["check"] += 1
        return True

    def mock_benchmark(*args, **kwargs):
        call_count["bench"] += 1
        return BenchmarkResult(
            model_name="test", runtime="lm-studio",
            prompt_tokens=10, generated_tokens=50,
            prompt_eval_time_ms=100, eval_time_ms=500,
            total_time_ms=600, tokens_per_second=100,
            prompt_tokens_per_second=100, context_length=4096,
            success=True, run_number=call_count["bench"],
        )

    monkeypatch.setattr("loca_llama.benchmark._check_model_available", mock_check_model)
    monkeypatch.setattr("loca_llama.benchmark.benchmark_openai_api", mock_benchmark)

    rt = RuntimeInfo(name="lm-studio", url="http://127.0.0.1:1234", models=["test"])
    results = run_benchmark_suite(rt, "test", num_runs=3)

    assert call_count["check"] == 3  # checked before every run
    assert call_count["bench"] == 3
    assert len(results) == 3
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_runtime_benchmark.py::test_benchmark_suite_checks_model_between_runs -v`
Expected: FAIL — `_check_model_available` does not exist yet.

**Step 3: Implement `_check_model_available` and update `run_benchmark_suite`**

Add before `run_benchmark_suite` in `benchmark.py`:

```python
def _check_model_available(base_url: str, model_id: str, timeout: int = 5) -> bool:
    """Verify a specific model is still loaded on the server."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            loaded = [m["id"] for m in data.get("data", [])]
            return any(model_id in mid or mid in model_id for mid in loaded)
    except Exception:
        return False
```

Update `run_benchmark_suite` to check before each run:

```python
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
    """Run a multi-round benchmark. First run is warmup.

    Verifies model availability before each run to handle LM Studio
    unloading models between runs. Aborts early on repeated crashes.
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

        # Verify model is still loaded before each run
        if not _check_model_available(runtime.url, model_id):
            if not _wait_for_server_ready(runtime.url, timeout=15):
                results.append(_make_fail_result(
                    model_id, runtime.name, context_length,
                    "Model no longer loaded on server", i,
                ))
                break

        r = benchmark_openai_api(
            runtime.url, model_id, runtime.name, prompt, max_tokens, context_length,
            run_number=i, api_key=runtime.api_key,
        )
        results.append(r)

        if not r.success and _is_model_crash(r.error):
            consecutive_crashes += 1
            if consecutive_crashes >= max_consecutive_crashes:
                for j in range(i + 1, num_runs + 1):
                    results.append(_make_fail_result(
                        model_id, runtime.name, context_length,
                        f"Skipped: model crashed {consecutive_crashes} times consecutively", j,
                    ))
                break
        else:
            consecutive_crashes = 0

    return results
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_runtime_benchmark.py::test_benchmark_suite_checks_model_between_runs -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/ -v`

**Step 6: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/benchmark.py tests/test_runtime_benchmark.py
git commit -m "fix: add per-run model availability check in benchmark suite

Prevents benchmark failures when LM Studio unloads models between runs.
Checks /v1/models before each benchmark run and waits for recovery if
the model disappears."
```

---

## Task 3: Improve Crash/Segfault Detection and Recovery

**Files:**
- Modify: `loca_llama/benchmark.py:199-204` (`_is_model_crash`)
- Modify: `loca_llama/benchmark.py:165-196` (`format_benchmark_error`)
- Test: `tests/test_benchmark_errors.py`

**Step 1: Update `_is_model_crash` to catch more crash patterns**

```python
def _is_model_crash(error: str | None) -> bool:
    """Check if a benchmark error indicates a model crash (not transient)."""
    if not error:
        return False
    crash_indicators = [
        "crashed", "channel error", "connection reset", "unstable",
        "segmentation", "broken pipe", "connection dropped", "incomplete",
        "internal error",
    ]
    return any(ind in error.lower() for ind in crash_indicators)
```

**Step 2: Update `format_benchmark_error` for segfault detection**

Add after the `IncompleteRead` check (before the final return) in `format_benchmark_error`:

```python
    # Segfault — Python MLX worker crashed
    if "segmentation" in str(exc).lower() or "segfault" in str(exc).lower():
        return f"Model crashed in {runtime_name} (segfault). MLX models may be unstable at this quantization — try a lower bit-width."
```

**Step 3: Run error tests**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_benchmark_errors.py -v`

**Step 4: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/benchmark.py
git commit -m "fix: improve crash detection for MLX segfaults and connection drops

Adds segfault, broken pipe, incomplete read, and internal error
to crash indicators. Better error messages for MLX instability."
```

---

## Task 4: Update OpenCode Config — Remove llama.cpp Provider

**Files:**
- Modify: `~/.config/opencode/opencode.json`

**Step 1: Remove the `llamacpp` provider block**

Update `opencode.json` to only have the `lmstudio` provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio (local)",
      "options": {
        "baseURL": "http://127.0.0.1:1234/v1"
      },
      "models": {
        "qwen3.5-27b": {
          "name": "Qwen3.5-27B (LM Studio)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        },
        "qwen3.5-35b-a3b": {
          "name": "Qwen3.5-35B-A3B MoE (LM Studio)",
          "limit": {
            "context": 131072,
            "output": 8192
          }
        },
        "qwen3.5-9b": {
          "name": "Qwen3.5-9B (LM Studio)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        },
        "gpt-oss-20b": {
          "name": "GPT-OSS-20B (LM Studio)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  },
  "mcp": {
    "mcpjungle": {
      "type": "remote",
      "url": "http://localhost:8080/v0/groups/opencode-full/mcp",
      "enabled": true
    },
    "MCP_DOCKER": {
      "type": "local",
      "command": ["docker", "mcp", "gateway", "run"],
      "enabled": true
    }
  }
}
```

Also removes the duplicate `unsloth/qwen3.5-35b-a3b` model entry that was never needed.

**Step 2: Verify opencode starts**

Run: `opencode --version` (or launch and check it connects to LM Studio)

**Step 3: No git commit needed — config file is not in a repo**

---

## Task 5: Simplify .zshrc — Deprioritize llama.cpp and LiteLLM Helpers

**Files:**
- Modify: `~/.zshrc:49-168`

**Step 1: Add deprecation comments and simplify**

Keep the functions (they still work if needed) but:
1. Remove the auto-detection block that sets `ANTHROPIC_BASE_URL` when LiteLLM is running (lines 158-161) — this was overriding Claude Code's API key
2. Add comments marking llama.cpp and LiteLLM sections as "optional/legacy"
3. Keep `lms-load`, `lms-clean`, `gpu-lock/unlock` as primary helpers

Replace lines 49-168 with:

```bash
# =============================================================================
# Local LLMs — LM Studio is the primary backend (port 1234)
# =============================================================================
# Architecture:
#   All coding tools (opencode, aider, goose) → LM Studio (:1234)
#   Claude Code uses Anthropic API (cloud) by default
#
# Workflow:
#   1. Start LM Studio (load model via lms-load)
#   2. Use opencode/aider/goose with models served on :1234
#
# Models: qwen3.5-27b | qwen3.5-35b-a3b | qwen3.5-9b | gpt-oss-20b

# --- GPU memory management ---
gpu-lock()   { sudo sysctl iogpu.wired_limit_mb=40960; }
gpu-unlock() { sudo sysctl iogpu.wired_limit_mb=0; }

# --- LM Studio helpers (primary) ---
lms-load() {
  declare -A lms_models=(
    [qwen35b]="qwen3.5-35b-a3b:131072"
    [qwen27b]="qwen3.5-27b:32768"
    [qwen9b]="qwen3.5-9b:32768"
    [glm]="zai-org/glm-4.7-flash:32768"
    [oss]="openai/gpt-oss-20b:49152"
  )

  local entry="${lms_models[$1]}"
  if [[ -z "$entry" ]]; then
    echo "Usage: lms-load <qwen35b|qwen27b|qwen9b|glm|oss>"
    return 1
  fi

  lms unload --all
  local model="${entry%%:*}" ctx="${entry##*:}"
  echo "Loading $model (ctx: $ctx)..."
  lms load "$model" --context-length "$ctx" --gpu max -y
}

lms-clean() {
  lms unload --all
  gpu-unlock
  echo "GPU cleared, RAM unlocked."
}

# --- Legacy: LiteLLM proxy (only needed for Claude Code local models) ---
LITELLM_CONFIG="$HOME/Developer/litellm/config.yaml"
LITELLM_PORT=4000

litellm-start() {
  if curl -s "http://localhost:$LITELLM_PORT/health" >/dev/null 2>&1; then
    echo "LiteLLM already running on :$LITELLM_PORT"
    return 0
  fi
  echo "Starting LiteLLM proxy on :$LITELLM_PORT..."
  /usr/bin/nohup "$HOME/.local/bin/litellm" --config "$LITELLM_CONFIG" --port "$LITELLM_PORT" > /tmp/litellm.log 2>&1 &
  echo $! > /tmp/litellm.pid
  sleep 2
  if curl -s "http://localhost:$LITELLM_PORT/health" >/dev/null 2>&1; then
    echo "LiteLLM ready. Use: claude --model qwen3.5-27b"
  else
    echo "LiteLLM starting... check /tmp/litellm.log"
  fi
}

litellm-stop() {
  [[ -f /tmp/litellm.pid ]] && kill "$(cat /tmp/litellm.pid)" 2>/dev/null && rm -f /tmp/litellm.pid
  pkill -f "litellm.*--port $LITELLM_PORT" 2>/dev/null
  echo "LiteLLM stopped."
}

# --- Legacy: llama.cpp server (slower than LM Studio, keep for GGUF-only testing) ---
LLAMA_PORT=8082

llama-start() {
  local models_dir="$HOME/.lmstudio/models"
  declare -A model_map=(
    [qwen27b]="unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q4_K_XL.gguf"
    [qwen35b]="unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q6_K_S.gguf"
    [qwen9b]="unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf"
    [distill]="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf"
  )

  local key="${1:-qwen27b}"
  local path="${model_map[$key]}"
  if [[ -z "$path" ]]; then
    echo "Usage: llama-start <qwen27b|qwen35b|qwen9b|distill>"
    return 1
  fi

  pkill -f "llama-server.*--port $LLAMA_PORT" 2>/dev/null
  echo "Starting llama-server on :$LLAMA_PORT with $(basename "$path")..."
  /usr/bin/nohup llama-server \
    --model "$models_dir/$path" \
    --ctx-size "${2:-32768}" \
    --n-gpu-layers 99 \
    --port "$LLAMA_PORT" \
    --host 127.0.0.1 > /tmp/llama-server.log 2>&1 &
  echo $! > /tmp/llama-server.pid
  echo "Loading model... check /tmp/llama-server.log"
}

llama-stop() {
  [[ -f /tmp/llama-server.pid ]] && kill "$(cat /tmp/llama-server.pid)" 2>/dev/null && rm -f /tmp/llama-server.pid
  pkill -f "llama-server.*--port $LLAMA_PORT" 2>/dev/null
  echo "llama-server stopped."
}

# --- Claude Code shortcuts ---
alias claude-quick='claude --model haiku'
alias claude-qa='claude --model sonnet'
alias claude-cheap='claude --model haiku -p'
```

Key changes:
- Removed auto-`ANTHROPIC_BASE_URL` block (lines 158-161) — was hijacking Claude Code when LiteLLM happened to be running
- Removed `local-test` function (no longer needed)
- Reordered: LM Studio helpers first, legacy stuff after
- Added clear section comments

**Step 2: Source and verify**

Run: `source ~/.zshrc && type lms-load && type llama-start`

**Step 3: No git commit needed — dotfile not in a repo**

---

## Task 6: Create Coding Tool Benchmark Module

**Files:**
- Create: `loca_llama/coding_benchmark.py`
- Test: `tests/test_coding_benchmark.py`

**Step 1: Write the failing test**

Create `tests/test_coding_benchmark.py`:

```python
"""Tests for coding tool benchmark module."""
import pytest
from unittest.mock import patch, MagicMock
from loca_llama.coding_benchmark import (
    CodingTool,
    CodingBenchmarkResult,
    detect_coding_tools,
    benchmark_coding_tool,
    CODING_PROMPTS,
)


def test_coding_tool_dataclass():
    tool = CodingTool(name="aider", command="aider", version="0.50.0")
    assert tool.name == "aider"
    assert tool.command == "aider"


def test_coding_prompts_exist():
    assert "fizzbuzz" in CODING_PROMPTS
    assert "refactor" in CODING_PROMPTS
    assert len(CODING_PROMPTS) >= 2


def test_detect_coding_tools_finds_installed(monkeypatch):
    """Should detect tools that exist on PATH."""
    def mock_which(cmd):
        return f"/usr/local/bin/{cmd}" if cmd in ["aider", "opencode"] else None

    monkeypatch.setattr("shutil.which", mock_which)
    tools = detect_coding_tools()
    names = [t.name for t in tools]
    assert "aider" in names
    assert "opencode" in names
    assert "goose" not in names


def test_benchmark_coding_tool_returns_result(monkeypatch):
    """Should return a CodingBenchmarkResult from a successful run."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Done"
    mock_result.stderr = ""

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock_result,
    )

    tool = CodingTool(name="aider", command="aider")
    result = benchmark_coding_tool(tool, "fizzbuzz", timeout=10)
    assert isinstance(result, CodingBenchmarkResult)
    assert result.tool_name == "aider"
    assert result.success is True
    assert result.total_time_ms > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_coding_benchmark.py -v`
Expected: FAIL — module does not exist

**Step 3: Implement `coding_benchmark.py`**

Create `loca_llama/coding_benchmark.py`:

```python
"""Benchmark coding tools (aider, opencode, goose) against local LLMs."""

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodingTool:
    """A detected coding tool."""
    name: str
    command: str
    version: str | None = None


@dataclass
class CodingBenchmarkResult:
    """Result of benchmarking a coding tool on a task."""
    tool_name: str
    task_name: str
    success: bool
    total_time_ms: float
    output: str = ""
    error: str | None = None
    exit_code: int = 0
    extra: dict = field(default_factory=dict)


CODING_PROMPTS = {
    "fizzbuzz": {
        "description": "Write FizzBuzz in Python",
        "prompt": "Create a file called fizzbuzz.py that prints FizzBuzz from 1 to 100.",
        "validation": "fizzbuzz.py",
    },
    "refactor": {
        "description": "Refactor a function",
        "prompt": "Refactor the calculate function in calc.py to use match/case instead of if/elif.",
        "setup_file": "calc.py",
        "setup_content": (
            "def calculate(op, a, b):\n"
            "    if op == 'add':\n"
            "        return a + b\n"
            "    elif op == 'sub':\n"
            "        return a - b\n"
            "    elif op == 'mul':\n"
            "        return a * b\n"
            "    elif op == 'div':\n"
            "        return a / b if b != 0 else None\n"
            "    else:\n"
            "        return None\n"
        ),
        "validation": "calc.py",
    },
    "test_generation": {
        "description": "Generate unit tests",
        "prompt": "Write pytest tests for utils.py — cover all functions with edge cases.",
        "setup_file": "utils.py",
        "setup_content": (
            "def clamp(value: float, low: float, high: float) -> float:\n"
            "    return max(low, min(value, high))\n\n"
            "def slugify(text: str) -> str:\n"
            "    return text.lower().strip().replace(' ', '-')\n"
        ),
        "validation": "test_utils.py",
    },
}


def detect_coding_tools() -> list[CodingTool]:
    """Detect installed coding tools on PATH."""
    tools = []
    tool_defs = [
        ("aider", "aider"),
        ("opencode", "opencode"),
        ("goose", "goose"),
    ]

    for name, cmd in tool_defs:
        path = shutil.which(cmd)
        if path:
            version = _get_tool_version(cmd)
            tools.append(CodingTool(name=name, command=cmd, version=version))

    return tools


def _get_tool_version(cmd: str) -> str | None:
    """Try to get version string from a tool."""
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _build_tool_command(tool: CodingTool, prompt: str, workdir: Path) -> list[str]:
    """Build the CLI command for each coding tool."""
    if tool.name == "aider":
        return [
            tool.command,
            "--yes-always",
            "--no-auto-commits",
            "--no-git",
            "--message", prompt,
        ]
    elif tool.name == "opencode":
        return [
            tool.command,
            "-p", prompt,
        ]
    elif tool.name == "goose":
        return [
            tool.command,
            "run",
            "--text", prompt,
        ]
    else:
        return [tool.command, prompt]


def benchmark_coding_tool(
    tool: CodingTool,
    task_name: str,
    timeout: int = 120,
    model: str | None = None,
) -> CodingBenchmarkResult:
    """Run a coding tool on a benchmark task and measure performance.

    Creates a temporary directory, sets up any required files,
    invokes the tool, and measures wall-clock time.
    """
    task = CODING_PROMPTS.get(task_name)
    if not task:
        return CodingBenchmarkResult(
            tool_name=tool.name, task_name=task_name,
            success=False, total_time_ms=0,
            error=f"Unknown task: {task_name}",
        )

    with tempfile.TemporaryDirectory(prefix=f"loca-bench-{tool.name}-") as tmpdir:
        workdir = Path(tmpdir)

        # Set up any required files
        if "setup_file" in task and "setup_content" in task:
            (workdir / task["setup_file"]).write_text(task["setup_content"])

        cmd = _build_tool_command(tool, task["prompt"], workdir)

        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout, cwd=str(workdir),
            )
            total_ms = (time.perf_counter() - start) * 1000

            # Check if expected output file was created
            validation_file = task.get("validation")
            file_created = (workdir / validation_file).exists() if validation_file else True

            return CodingBenchmarkResult(
                tool_name=tool.name,
                task_name=task_name,
                success=result.returncode == 0 and file_created,
                total_time_ms=total_ms,
                output=result.stdout[:2000],
                error=result.stderr[:500] if result.returncode != 0 else None,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            total_ms = (time.perf_counter() - start) * 1000
            return CodingBenchmarkResult(
                tool_name=tool.name, task_name=task_name,
                success=False, total_time_ms=total_ms,
                error=f"Timed out after {timeout}s",
            )
        except Exception as e:
            total_ms = (time.perf_counter() - start) * 1000
            return CodingBenchmarkResult(
                tool_name=tool.name, task_name=task_name,
                success=False, total_time_ms=total_ms,
                error=str(e),
            )


def run_coding_benchmark_suite(
    tools: list[CodingTool] | None = None,
    tasks: list[str] | None = None,
    timeout: int = 120,
    progress_callback=None,
) -> list[CodingBenchmarkResult]:
    """Run all coding benchmarks across all detected tools.

    Returns a flat list of results (one per tool+task combination).
    """
    if tools is None:
        tools = detect_coding_tools()
    if tasks is None:
        tasks = list(CODING_PROMPTS.keys())

    results = []
    total = len(tools) * len(tasks)
    idx = 0

    for tool in tools:
        for task_name in tasks:
            idx += 1
            if progress_callback:
                progress_callback(idx, total, tool.name, task_name)

            result = benchmark_coding_tool(tool, task_name, timeout)
            results.append(result)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/test_coding_benchmark.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/ -v`

**Step 6: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/coding_benchmark.py tests/test_coding_benchmark.py
git commit -m "feat: add coding tool benchmark module

Supports benchmarking aider, opencode, and goose on standardized
coding tasks (fizzbuzz, refactor, test generation). Measures
wall-clock time, success rate, and validates output files."
```

---

## Task 7: Add Coding Tool Benchmark to TUI

**Files:**
- Modify: `loca_llama/interactive.py`

**Step 1: Add coding benchmark import and menu option**

Add import near top of `interactive.py`:

```python
from loca_llama.coding_benchmark import (
    detect_coding_tools,
    run_coding_benchmark_suite,
    CODING_PROMPTS,
)
```

Add a new function `_benchmark_coding_tools()` and wire it into the TUI menu where the regular benchmark option lives. The function should:
1. Call `detect_coding_tools()` and display found tools
2. Let user pick tasks (all or specific)
3. Run `run_coding_benchmark_suite()` with progress display
4. Show results table (tool, task, time, success/fail)

This is a larger change — implement after reading the full TUI menu structure to find the right insertion point.

**Step 2: Test manually**

Run: `cd /Users/flong/Developer/loca-llama && python -m loca_llama.interactive`

Navigate to the benchmark section and verify the coding tool option appears.

**Step 3: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/interactive.py
git commit -m "feat: add coding tool benchmarking to TUI

New menu option lets users benchmark aider, opencode, and goose
on standardized coding tasks with progress display and results table."
```

---

## Task 8: Add Coding Tool Benchmark API Route

**Files:**
- Create: `loca_llama/api/routes/coding_benchmark.py`
- Modify: `loca_llama/api/app.py` (register new router)
- Test: `tests/api/test_coding_benchmark_routes.py`

**Step 1: Write the failing test**

Create `tests/api/test_coding_benchmark_routes.py`:

```python
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from loca_llama.api.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_list_coding_tools(client):
    with patch("loca_llama.coding_benchmark.detect_coding_tools") as mock:
        mock.return_value = []
        response = await client.get("/api/coding-benchmark/tools")
        assert response.status_code == 200
        assert "tools" in response.json()


@pytest.mark.asyncio
async def test_list_coding_tasks(client):
    response = await client.get("/api/coding-benchmark/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/api/test_coding_benchmark_routes.py -v`
Expected: FAIL — route does not exist

**Step 3: Implement the route**

Create `loca_llama/api/routes/coding_benchmark.py`:

```python
"""API routes for coding tool benchmarking."""
import asyncio
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loca_llama.coding_benchmark import (
    detect_coding_tools,
    benchmark_coding_tool,
    CodingTool,
    CODING_PROMPTS,
)

router = APIRouter(prefix="/coding-benchmark", tags=["coding-benchmark"])

# In-memory job store
_jobs: dict[str, dict] = {}


class CodingBenchmarkRequest(BaseModel):
    tool_name: str = Field(..., description="Tool to benchmark: aider, opencode, or goose")
    task_name: str = Field(default="fizzbuzz", description="Task name from CODING_PROMPTS")
    timeout: int = Field(default=120, ge=10, le=600)


@router.get("/tools")
async def list_coding_tools() -> dict:
    """List detected coding tools."""
    try:
        tools = await asyncio.to_thread(detect_coding_tools)
        return {
            "tools": [
                {"name": t.name, "command": t.command, "version": t.version}
                for t in tools
            ]
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to detect coding tools")


@router.get("/tasks")
async def list_coding_tasks() -> dict:
    """List available benchmark tasks."""
    return {
        "tasks": [
            {"name": name, "description": task["description"]}
            for name, task in CODING_PROMPTS.items()
        ]
    }


@router.post("/start")
async def start_coding_benchmark(req: CodingBenchmarkRequest) -> dict:
    """Start a coding tool benchmark (non-blocking)."""
    tools = await asyncio.to_thread(detect_coding_tools)
    tool = next((t for t in tools if t.name == req.tool_name), None)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found on PATH")

    if req.task_name not in CODING_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task_name}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None}

    async def _run():
        try:
            result = await asyncio.to_thread(
                benchmark_coding_tool, tool, req.task_name, req.timeout
            )
            _jobs[job_id] = {
                "status": "complete",
                "result": {
                    "tool_name": result.tool_name,
                    "task_name": result.task_name,
                    "success": result.success,
                    "total_time_ms": result.total_time_ms,
                    "error": result.error,
                    "exit_code": result.exit_code,
                },
            }
        except Exception as e:
            _jobs[job_id] = {"status": "error", "error": str(e)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "running"}


@router.get("/{job_id}")
async def get_coding_benchmark_result(job_id: str) -> dict:
    """Poll for coding benchmark result."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]
```

Register in `app.py` — add import and `app.include_router(coding_benchmark.router, prefix="/api")`.

**Step 4: Run tests**

Run: `cd /Users/flong/Developer/loca-llama && python -m pytest tests/api/test_coding_benchmark_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/flong/Developer/loca-llama
git add loca_llama/api/routes/coding_benchmark.py tests/api/test_coding_benchmark_routes.py loca_llama/api/app.py
git commit -m "feat: add coding tool benchmark API routes

POST /api/coding-benchmark/start — run aider/opencode/goose benchmark
GET /api/coding-benchmark/tools — list detected tools
GET /api/coding-benchmark/tasks — list available tasks
GET /api/coding-benchmark/{job_id} — poll for results"
```

---

## Task 9: Update Memory File

**Files:**
- Modify: `~/.claude/projects/-Users-flong-Developer/memory/local-llm.md`

**Step 1: Update the local LLM memory to reflect new architecture**

Update the memory file to note:
- LM Studio is now the primary and only required backend
- LiteLLM and llama.cpp are legacy/optional
- Runtime detection order changed to LM Studio first
- `ANTHROPIC_BASE_URL` auto-detection removed from .zshrc
- Coding tool benchmarking added to loca-llama

**Step 2: No commit needed — memory files are not in a repo**

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Reorder runtime detection | benchmark.py, runtime.py |
| 2 | Per-run model availability check | benchmark.py |
| 3 | Better crash/segfault detection | benchmark.py |
| 4 | Simplify opencode config | opencode.json |
| 5 | Clean up .zshrc | .zshrc |
| 6 | Coding tool benchmark module | coding_benchmark.py |
| 7 | TUI integration | interactive.py |
| 8 | API routes for coding benchmarks | api/routes/coding_benchmark.py |
| 9 | Update memory | memory/local-llm.md |

Tasks 1-3 are bug fixes. Task 4-5 are config cleanup. Tasks 6-8 are the new feature. Task 9 is housekeeping.
