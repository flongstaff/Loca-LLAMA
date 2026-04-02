"""Unified benchmark results storage.

All benchmark results (speed, quality, monitor, eval, sql, throughput) are saved
to ~/.loca-llama/results/ as individual JSON files per run.
"""

from __future__ import annotations

import json
import platform
import re
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path.home() / ".loca-llama" / "results"

# ── Model Categorization ─────────────────────────────────────────────────

_CLOUD_RUNTIMES = {"openrouter", "litellm", "openai", "anthropic", "together"}

CATEGORY_ORDER = ["Cloud API", "Local (Large)", "Local (Medium)", "Local (Small)", "Local"]


def categorize_model(model: str, runtime: str) -> str:
    """Infer model category from runtime and model name.

    Returns one of: 'Cloud API', 'Local (Large)', 'Local (Medium)',
    'Local (Small)', or 'Local'.
    """
    if any(r in runtime.lower() for r in _CLOUD_RUNTIMES):
        return "Cloud API"
    match = re.search(r"(\d+)[Bb]", model)
    if match:
        params = int(match.group(1))
        if params >= 30:
            return "Local (Large)"
        if params >= 13:
            return "Local (Medium)"
        return "Local (Small)"
    return "Local"


def detect_hardware_string() -> str:
    """Auto-detect hardware as a human-readable string, e.g. 'M4 Pro 48GB'.

    Falls back to platform.machine() on non-Mac systems.
    """
    try:
        from .hardware import detect_mac
        result = detect_mac()
        if result:
            _key, spec = result
            return f"{spec.chip} {spec.memory_gb}GB"
    except Exception:
        pass

    # Fallback: try to read chip + memory directly via sysctl on macOS
    if platform.system() == "Darwin":
        try:
            import subprocess
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=2,
            ).strip()
            mem_bytes = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=2,
            ).strip()
            if brand.startswith("Apple "):
                chip = brand.removeprefix("Apple ")
                mem_gb = round(int(mem_bytes) / (1024 ** 3))
                return f"{chip} {mem_gb}GB"
        except Exception:
            pass

    return platform.machine()


@dataclass
class BenchmarkRecord:
    """A single benchmark result record."""

    type: str  # "speed", "quality", "monitor", "eval", "sql", "throughput"
    model: str
    runtime: str  # "lm-studio", "omlx", "llama.cpp-server", "openrouter", etc.
    timestamp: float = field(default_factory=time.time)

    # Hardware context
    hardware: str = ""  # e.g. "M4 Pro 48GB"

    # Speed metrics (speed + monitor types)
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0

    # Quality metrics (quality / eval / sql types)
    quality_scores: dict[str, Any] = field(default_factory=dict)
    # e.g. {"pass_rate": 0.75, "avg_contains": 0.9, "tasks": [...]}

    # Monitor metrics (monitor type)
    monitor_stats: dict[str, Any] = field(default_factory=dict)
    # e.g. {"total_requests": 12, "total_tokens": 4832, "session_duration_s": 300}

    # Throughput metrics (throughput type)
    throughput_stats: dict[str, Any] = field(default_factory=dict)
    # e.g. {"concurrency": 4, "throughput_tps": 120, "p50_latency_ms": 200, ...}

    # Speed percentile metrics
    speed_percentiles: dict[str, float] = field(default_factory=dict)
    # e.g. {"p50_tok_per_sec": 25, "p90_tok_per_sec": 28, "p50_ttft_ms": 120, ...}

    # Cloud comparison (quality --compare)
    cloud_provider: str = ""  # "pi", "claude", ""
    cloud_scores: dict[str, Any] = field(default_factory=dict)

    # Extra metadata
    # Conventions: extra["cost_cents"] = total cost in cents for the run
    #              extra["cost_per_1k_tokens"] = cost per 1k tokens
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.timestamp))
        safe_model = self.model.replace("/", "_").replace(" ", "_")[:60]
        return f"{ts}_{self.type}_{safe_model}.json"


def _ensure_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically via temp file + rename."""
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=".result_"
    )
    try:
        with open(fd, "w") as f:
            f.write(content)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def save_result(record: BenchmarkRecord) -> Path:
    """Save a benchmark record to ~/.loca-llama/results/.

    Uses atomic writes (temp + rename) to prevent corruption.
    Falls back to a temp directory if the primary path is not writable.
    """
    content = json.dumps(asdict(record), indent=2, default=str)
    try:
        out_dir = _ensure_dir()
        path = out_dir / record.filename
        _atomic_write(path, content)
        return path
    except (PermissionError, OSError) as e:
        fallback = Path(tempfile.gettempdir()) / "loca-llama-results"
        fallback.mkdir(parents=True, exist_ok=True)
        path = fallback / record.filename
        _atomic_write(path, content)
        print(f"  Warning: saved to {path} (primary dir not writable: {e})")
        return path


def load_results(
    type_filter: str | None = None,
    model_filter: str | None = None,
    limit: int = 50,
) -> list[BenchmarkRecord]:
    """Load saved results, newest first."""
    if not RESULTS_DIR.exists():
        return []

    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    records: list[BenchmarkRecord] = []

    for f in files:
        if len(records) >= limit:
            break
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if type_filter and data.get("type") != type_filter:
            continue
        if model_filter and model_filter.lower() not in data.get("model", "").lower():
            continue

        records.append(BenchmarkRecord(**{
            k: v for k, v in data.items()
            if k in BenchmarkRecord.__dataclass_fields__
        }))

    return records


def print_results_table(records: list[BenchmarkRecord]) -> None:
    """Print a summary table of benchmark records."""
    if not records:
        print("  No results found.")
        return

    # Header
    print(f"\n{'Type':<10} {'Model':<45} {'TPS':>7} {'TTFT':>8} {'Date':>12}")
    print("-" * 85)

    for r in records:
        date_str = time.strftime("%Y-%m-%d", time.localtime(r.timestamp))
        model_short = r.model[:43]

        if r.type == "quality":
            pass_rate = r.quality_scores.get("pass_rate", 0)
            print(f"{r.type:<10} {model_short:<45} {'pass=' + f'{pass_rate:.0%}':>7} {'':>8} {date_str:>12}")
        elif r.type == "sql":
            total_pass = r.quality_scores.get("total_pass", 0)
            total_q = r.quality_scores.get("total_questions", 0)
            score_str = f"{total_pass}/{total_q}" if total_q else "—"
            tps_str = f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "—"
            print(f"{r.type:<10} {model_short:<45} {score_str:>7} {tps_str:>8} {date_str:>12}")
        elif r.type == "eval":
            # Show average score across eval benchmarks
            scores = [v["score"] for v in r.quality_scores.values()
                      if isinstance(v, dict) and "score" in v]
            avg_score = sum(scores) / len(scores) if scores else 0
            n_bench = len(scores)
            print(f"{r.type:<10} {model_short:<45} {'avg=' + f'{avg_score:.0%}':>7} {f'{n_bench} bench':>8} {date_str:>12}")
        elif r.type == "throughput":
            tp_tps = r.throughput_stats.get("throughput_tps", r.tokens_per_second)
            conc = r.throughput_stats.get("concurrency", "")
            conc_str = f"c={conc}" if conc else "—"
            print(f"{r.type:<10} {model_short:<45} {tp_tps:>6.1f} {conc_str:>8} {date_str:>12}")
        elif r.type == "monitor":
            reqs = r.monitor_stats.get("total_requests", 0)
            avg_tps = r.monitor_stats.get("avg_tps", 0)
            print(f"{r.type:<10} {model_short:<45} {avg_tps:>6.1f} {'':>8} {date_str:>12}  [{reqs} reqs]")
        else:
            tps_str = f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "—"
            ttft_str = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "—"
            print(f"{r.type:<10} {model_short:<45} {tps_str:>7} {ttft_str:>8} {date_str:>12}")

    if records and records[0].cloud_provider:
        print(f"\n  (Cloud comparison: {records[0].cloud_provider})")


def print_comparison_table(records: list[BenchmarkRecord]) -> None:
    """Print side-by-side comparison of latest result per model."""
    if not records:
        print("  No results found.")
        return

    # Group by model, keep latest per model
    by_model: dict[str, BenchmarkRecord] = {}
    for r in records:
        if r.model not in by_model:
            by_model[r.model] = r

    print(f"\n{'Model':<45} {'Type':<10} {'TPS':>7} {'TTFT':>8} {'Quality':>8}")
    print("-" * 80)

    for model, r in by_model.items():
        model_short = model[:43]
        tps = f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "—"
        ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "—"
        quality = f"{r.quality_scores.get('pass_rate', 0):.0%}" if r.quality_scores else "—"
        print(f"{model_short:<45} {r.type:<10} {tps:>7} {ttft:>8} {quality:>8}")
