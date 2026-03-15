"""Unified benchmark results storage.

All benchmark results (speed, quality, monitor) are saved to ~/.loca-llama/results/
as individual JSON files per run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path.home() / ".loca-llama" / "results"


@dataclass
class BenchmarkRecord:
    """A single benchmark result record."""

    type: str  # "speed", "quality", "monitor"
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

    # Quality metrics (quality type)
    quality_scores: dict[str, Any] = field(default_factory=dict)
    # e.g. {"pass_rate": 0.75, "avg_contains": 0.9, "tasks": [...]}

    # Monitor metrics (monitor type)
    monitor_stats: dict[str, Any] = field(default_factory=dict)
    # e.g. {"total_requests": 12, "total_tokens": 4832, "session_duration_s": 300}

    # Cloud comparison (quality --compare)
    cloud_provider: str = ""  # "pi", "claude", ""
    cloud_scores: dict[str, Any] = field(default_factory=dict)

    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.timestamp))
        safe_model = self.model.replace("/", "_").replace(" ", "_")[:60]
        return f"{ts}_{self.type}_{safe_model}.json"


def _ensure_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_result(record: BenchmarkRecord) -> Path:
    """Save a benchmark record to ~/.loca-llama/results/."""
    out_dir = _ensure_dir()
    path = out_dir / record.filename
    path.write_text(json.dumps(asdict(record), indent=2, default=str))
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
