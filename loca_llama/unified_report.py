"""Unified mega-report: combine all benchmark types into one HTML scorecard.

Generates a self-contained HTML file with embedded Canvas charts, heatmaps,
radar charts, and sortable leaderboard. No external dependencies.

Inspired by sql-benchmark.nicklothian.com.

Usage:
    loca-llama report                           # All saved results
    loca-llama report --models Qwen3.5,Llama    # Filter models
    loca-llama report --output scorecard.html    # Custom output path
"""

from __future__ import annotations

import html
import json
import platform
import time
from dataclasses import dataclass, field
from typing import Any

from .benchmark_results import BenchmarkRecord, load_results


# ── Data Loading & Normalization ──────────────────────────────────────────


@dataclass
class ModelScorecard:
    """Aggregated scores for a single model across all benchmark types."""

    model: str
    runtime: str = ""
    hardware: str = ""
    # Per-category data
    speed_data: dict[str, Any] = field(default_factory=dict)
    quality_data: dict[str, Any] = field(default_factory=dict)
    eval_data: dict[str, Any] = field(default_factory=dict)
    sql_data: dict[str, Any] = field(default_factory=dict)
    throughput_data: dict[str, Any] = field(default_factory=dict)
    # Computed
    overall_score: float = 0.0


def load_scorecards(
    models: list[str] | None = None,
    types: list[str] | None = None,
    limit: int = 200,
) -> list[ModelScorecard]:
    """Load benchmark records and aggregate into per-model scorecards."""
    records = load_results(limit=limit)
    if not records:
        return []

    # Group records by model, keeping latest per (model, type)
    latest: dict[tuple[str, str], BenchmarkRecord] = {}
    for r in records:
        if models and not any(m.lower() in r.model.lower() for m in models):
            continue
        if types and r.type not in types:
            continue
        key = (r.model, r.type)
        if key not in latest:
            latest[key] = r

    # Build scorecards per model
    by_model: dict[str, ModelScorecard] = {}
    for (model_name, rec_type), rec in latest.items():
        if model_name not in by_model:
            by_model[model_name] = ModelScorecard(
                model=model_name,
                runtime=rec.runtime,
                hardware=rec.hardware,
            )
        card = by_model[model_name]
        if not card.hardware and rec.hardware:
            card.hardware = rec.hardware
        if not card.runtime and rec.runtime:
            card.runtime = rec.runtime

        if rec_type == "speed":
            card.speed_data = {
                "tokens_per_second": rec.tokens_per_second,
                "ttft_ms": rec.ttft_ms,
                "total_time_ms": rec.total_time_ms,
                "percentiles": rec.speed_percentiles,
                "extra": rec.extra,
            }
        elif rec_type == "quality":
            card.quality_data = rec.quality_scores
        elif rec_type == "eval":
            card.eval_data = rec.quality_scores
        elif rec_type == "sql":
            card.sql_data = rec.quality_scores
        elif rec_type in ("throughput", "monitor"):
            card.throughput_data = rec.throughput_stats or rec.monitor_stats or {
                "tokens_per_second": rec.tokens_per_second,
            }

    # Compute overall scores
    cards = list(by_model.values())
    for card in cards:
        card.overall_score = compute_overall_score(card, cards)

    # Sort by overall score descending
    cards.sort(key=lambda c: c.overall_score, reverse=True)
    return cards


def compute_overall_score(card: ModelScorecard, all_cards: list[ModelScorecard]) -> float:
    """Compute weighted composite score (0-100).

    Weights: Speed 20%, Quality 20%, Eval 25%, SQL 20%, Throughput 15%.
    Missing categories get weight redistributed proportionally.
    """
    weights = {
        "speed": 20, "quality": 20, "eval": 25, "sql": 20, "throughput": 15,
    }
    scores: dict[str, float] = {}

    # Speed: normalize to 0-1 relative to best
    if card.speed_data and card.speed_data.get("tokens_per_second", 0) > 0:
        max_speed = max(
            (c.speed_data.get("tokens_per_second", 0) for c in all_cards if c.speed_data),
            default=1,
        )
        scores["speed"] = min(card.speed_data["tokens_per_second"] / max(max_speed, 1), 1.0)

    # Quality: pass_rate is already 0-1
    if card.quality_data and "pass_rate" in card.quality_data:
        scores["quality"] = card.quality_data["pass_rate"]

    # Eval: average across available benchmarks
    if card.eval_data:
        eval_scores = []
        for key, val in card.eval_data.items():
            if isinstance(val, dict) and "score" in val:
                eval_scores.append(val["score"])
            elif key == "score" and isinstance(val, (int, float)):
                eval_scores.append(val)
        if eval_scores:
            scores["eval"] = sum(eval_scores) / len(eval_scores)

    # SQL: pass_rate
    if card.sql_data and "pass_rate" in card.sql_data:
        scores["sql"] = card.sql_data["pass_rate"]

    # Throughput: normalize to 0-1 relative to best
    if card.throughput_data:
        tps = card.throughput_data.get("throughput_tps", card.throughput_data.get("tokens_per_second", 0))
        if tps > 0:
            max_tps = max(
                (c.throughput_data.get("throughput_tps", c.throughput_data.get("tokens_per_second", 0))
                 for c in all_cards if c.throughput_data),
                default=1,
            )
            scores["throughput"] = min(tps / max(max_tps, 1), 1.0)

    if not scores:
        return 0.0

    # Redistribute weights from missing categories
    available_weight = sum(weights[k] for k in scores)
    if available_weight == 0:
        return 0.0

    total = sum(scores[k] * weights[k] for k in scores)
    return (total / available_weight) * 100


# ── HTML Generation ───────────────────────────────────────────────────────


def generate_unified_report(
    scorecards: list[ModelScorecard],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Generate the complete self-contained HTML mega-report."""
    meta = metadata or {}
    title = html.escape(meta.get("title", "Loca-LLAMA Benchmark Report"))
    hw = html.escape(meta.get("hardware", scorecards[0].hardware if scorecards else platform.machine()))
    timestamp = meta.get("timestamp", time.strftime("%Y-%m-%d %H:%M"))

    sections = [
        _section_leaderboard(scorecards),
        _section_speed_dashboard(scorecards),
        _section_quality_heatmap(scorecards),
        _section_eval_radar(scorecards),
        _section_sql_heatmap(scorecards),
        _section_throughput(scorecards),
        _section_model_cards(scorecards),
    ]

    chart_data = _build_chart_data(scorecards)
    chart_json = json.dumps(chart_data).replace("</", r"<\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{_unified_css()}
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="meta">
    <span>Hardware: {hw}</span>
    <span>Models: {len(scorecards)}</span>
    <span>Date: {html.escape(timestamp)}</span>
  </div>
</div>

{''.join(sections)}

<div class="footer">
  <span>Generated by Loca-LLAMA</span>
  <span>Score colors: <span class="score-high">≥80</span> <span class="score-mid">≥50</span> <span class="score-low">&lt;50</span></span>
</div>

<script>
const REPORT_DATA = {chart_json};
{_chart_js()}
</script>
</body>
</html>"""


# ── Section Builders ──────────────────────────────────────────────────────


def _section_leaderboard(scorecards: list[ModelScorecard]) -> str:
    """Section 1: Sortable model leaderboard."""
    if not scorecards:
        return ""

    rows = ""
    best = {"speed": 0.0, "quality": 0.0, "eval": 0.0, "sql": 0.0, "throughput": 0.0}
    # Find best in each category
    for c in scorecards:
        if c.speed_data:
            best["speed"] = max(best["speed"], c.speed_data.get("tokens_per_second", 0))
        if c.quality_data:
            best["quality"] = max(best["quality"], c.quality_data.get("pass_rate", 0))
        if c.eval_data:
            evals = [v["score"] for v in c.eval_data.values() if isinstance(v, dict) and "score" in v]
            if evals:
                best["eval"] = max(best["eval"], sum(evals) / len(evals))
        if c.sql_data:
            best["sql"] = max(best["sql"], c.sql_data.get("pass_rate", 0))

    for c in scorecards:
        model_safe = html.escape(c.model[:45])

        # Speed cell
        spd = c.speed_data.get("tokens_per_second", 0) if c.speed_data else 0
        spd_cls = _score_class(spd / best["speed"] if best["speed"] else 0)
        spd_cell = f'<td class="{spd_cls}">{spd:.1f} t/s</td>' if spd else '<td class="na">—</td>'

        # Quality cell
        qr = c.quality_data.get("pass_rate", -1) if c.quality_data else -1
        q_cls = _score_class(qr) if qr >= 0 else "na"
        q_cell = f'<td class="{q_cls}">{qr:.0%}</td>' if qr >= 0 else '<td class="na">—</td>'

        # Eval cell
        eval_scores = []
        if c.eval_data:
            for v in c.eval_data.values():
                if isinstance(v, dict) and "score" in v:
                    eval_scores.append(v["score"])
        eval_avg = sum(eval_scores) / len(eval_scores) if eval_scores else -1
        e_cls = _score_class(eval_avg) if eval_avg >= 0 else "na"
        e_cell = f'<td class="{e_cls}">{eval_avg:.0%}</td>' if eval_avg >= 0 else '<td class="na">—</td>'

        # SQL cell
        sql_pr = c.sql_data.get("pass_rate", -1) if c.sql_data else -1
        sql_cls = _score_class(sql_pr) if sql_pr >= 0 else "na"
        sql_p = c.sql_data.get("total_pass", 0) if c.sql_data else 0
        sql_t = c.sql_data.get("total_questions", 0) if c.sql_data else 0
        sql_cell = f'<td class="{sql_cls}">{sql_p}/{sql_t}</td>' if sql_pr >= 0 else '<td class="na">—</td>'

        # Overall
        o_cls = _score_class(c.overall_score / 100)
        badge = " ★" if c == scorecards[0] else ""

        rows += (
            f'<tr>'
            f'<td class="model-name">{model_safe}{badge}</td>'
            f'<td class="{o_cls}"><strong>{c.overall_score:.0f}</strong></td>'
            f'{spd_cell}{q_cell}{e_cell}{sql_cell}'
            f'</tr>\n'
        )

    return f"""
<h2>Model Leaderboard</h2>
<table id="leaderboard">
  <thead><tr>
    <th>Model</th><th onclick="sortTable('leaderboard',1)" style="cursor:pointer">Overall ▼</th>
    <th>Speed</th><th>Quality</th><th>Eval</th><th>SQL</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_speed_dashboard(scorecards: list[ModelScorecard]) -> str:
    """Section 2: Speed charts."""
    cards_with_speed = [c for c in scorecards if c.speed_data and c.speed_data.get("tokens_per_second", 0) > 0]
    if not cards_with_speed:
        return ""

    return """
<h2>Speed Dashboard</h2>
<div class="charts">
  <div class="chart-box">
    <h3>Generation Speed (tok/s)</h3>
    <canvas id="speedChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Time to First Token (ms)</h3>
    <canvas id="ttftChart"></canvas>
  </div>
</div>
<div class="chart-box" style="margin-top:1rem">
  <h3>Speed vs Responsiveness</h3>
  <canvas id="scatterChart" style="height:280px"></canvas>
</div>
"""


def _section_quality_heatmap(scorecards: list[ModelScorecard]) -> str:
    """Section 3: Quality task heatmap."""
    cards_with_quality = [c for c in scorecards if c.quality_data and c.quality_data.get("tasks")]
    if not cards_with_quality:
        return ""

    # Build heatmap
    all_tasks = []
    for c in cards_with_quality:
        for t in c.quality_data.get("tasks", []):
            if t.get("name") and t["name"] not in all_tasks:
                all_tasks.append(t["name"])

    header_cells = "".join(f"<th>{html.escape(t[:12])}</th>" for t in all_tasks)
    heatmap_rows = ""
    for c in cards_with_quality:
        task_map = {t["name"]: t for t in c.quality_data.get("tasks", [])}
        cells = ""
        for task_name in all_tasks:
            t = task_map.get(task_name)
            if t is None:
                cells += '<td class="cell-na">—</td>'
            else:
                runnable = t.get("runnable", 0)
                contains = t.get("contains", 0)
                if runnable == 1.0:
                    cells += '<td class="cell-pass">P</td>'
                elif contains >= 0.8:
                    cells += f'<td class="cell-pass">{contains:.0%}</td>'
                elif contains >= 0.5:
                    cells += f'<td class="cell-error">{contains:.0%}</td>'
                else:
                    cells += f'<td class="cell-fail">F</td>'
        model_safe = html.escape(c.model[:30])
        heatmap_rows += f'<tr><td class="model-name">{model_safe}</td>{cells}</tr>\n'

    return f"""
<h2>Quality Benchmark</h2>
<div class="heatmap-wrapper">
<table>
  <thead><tr><th>Model</th>{header_cells}</tr></thead>
  <tbody>{heatmap_rows}</tbody>
</table>
</div>
"""


def _section_eval_radar(scorecards: list[ModelScorecard]) -> str:
    """Section 4: Eval benchmark radar + bars."""
    cards_with_eval = [c for c in scorecards if c.eval_data]
    if not cards_with_eval:
        return ""

    # Find all benchmark names
    bench_names: list[str] = []
    for c in cards_with_eval:
        for k, v in c.eval_data.items():
            if isinstance(v, dict) and "score" in v and k not in bench_names:
                bench_names.append(k)

    if not bench_names:
        return ""

    # Build bar rows
    header_cells = "".join(f"<th>{html.escape(b[:10])}</th>" for b in bench_names)
    rows = ""
    for c in cards_with_eval:
        cells = ""
        for b in bench_names:
            val = c.eval_data.get(b, {})
            if isinstance(val, dict) and "score" in val:
                score = val["score"]
                cls = _score_class(score)
                correct = val.get("correct", "")
                total = val.get("total", "")
                detail = f" ({correct}/{total})" if correct and total else ""
                cells += f'<td class="{cls}">{score:.0%}{detail}</td>'
            else:
                cells += '<td class="na">—</td>'
        model_safe = html.escape(c.model[:30])
        rows += f'<tr><td class="model-name">{model_safe}</td>{cells}</tr>\n'

    return f"""
<h2>Evaluation Benchmarks</h2>
<div class="charts">
  <div class="chart-box">
    <h3>Capability Radar</h3>
    <canvas id="radarChart" style="height:300px"></canvas>
  </div>
  <div class="chart-box">
    <h3>Per-Benchmark Scores</h3>
    <canvas id="evalBarChart" style="height:300px"></canvas>
  </div>
</div>
<div class="heatmap-wrapper">
<table>
  <thead><tr><th>Model</th>{header_cells}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
"""


def _section_sql_heatmap(scorecards: list[ModelScorecard]) -> str:
    """Section 5: SQL benchmark heatmap."""
    cards_with_sql = [c for c in scorecards if c.sql_data and c.sql_data.get("questions")]
    if not cards_with_sql:
        return ""

    # Get all question IDs
    all_questions: list[dict] = []
    seen_ids: set[int] = set()
    for c in cards_with_sql:
        for q in c.sql_data.get("questions", []):
            qid = q.get("id", 0)
            if qid not in seen_ids:
                seen_ids.add(qid)
                all_questions.append(q)
    all_questions.sort(key=lambda q: q.get("id", 0))

    diff_order = {"trivial": 0, "easy": 1, "medium": 2, "hard": 3}
    all_questions.sort(key=lambda q: (diff_order.get(q.get("difficulty", ""), 99), q.get("id", 0)))

    header_cells = ""
    for q in all_questions:
        diff = q.get("difficulty", "")
        header_cells += f'<th class="tier-{diff}" title="Q{q.get("id")}: {diff}">Q{q.get("id")}</th>'

    heatmap_rows = ""
    for c in cards_with_sql:
        q_map = {q.get("id"): q for q in c.sql_data.get("questions", [])}
        cells = ""
        for q in all_questions:
            qid = q.get("id")
            result = q_map.get(qid)
            if result is None:
                cells += '<td class="cell-na">—</td>'
            else:
                status = result.get("status", "error")
                label = {"pass": "P", "fail": "F", "error": "E"}.get(status, "?")
                retries = result.get("retries", 0)
                retry_mark = f"<sup>{retries}</sup>" if retries > 0 else ""
                cells += f'<td class="cell-{status}">{label}{retry_mark}</td>'
        model_safe = html.escape(c.model[:30])
        heatmap_rows += f'<tr><td class="model-name">{model_safe}</td>{cells}</tr>\n'

    return f"""
<h2>SQL Benchmark</h2>
<div class="heatmap-wrapper">
<table>
  <thead><tr><th>Model</th>{header_cells}</tr></thead>
  <tbody>{heatmap_rows}</tbody>
</table>
</div>
"""


def _section_throughput(scorecards: list[ModelScorecard]) -> str:
    """Section 6: Throughput comparison."""
    cards_with_tp = [c for c in scorecards if c.throughput_data]
    if not cards_with_tp:
        return ""

    rows = ""
    for c in cards_with_tp:
        td = c.throughput_data
        model_safe = html.escape(c.model[:35])
        tps = td.get("throughput_tps", td.get("tokens_per_second", td.get("avg_tps", 0)))
        p50 = td.get("p50_latency_ms", 0)
        p90 = td.get("p90_latency_ms", 0)
        p99 = td.get("p99_latency_ms", 0)
        conc = td.get("concurrency", "—")
        rows += (
            f'<tr><td class="model-name">{model_safe}</td>'
            f'<td>{tps:.1f}</td><td>{conc}</td>'
            f'<td>{p50:.0f}</td><td>{p90:.0f}</td><td>{p99:.0f}</td></tr>\n'
        )

    return f"""
<h2>Throughput</h2>
<table>
  <thead><tr>
    <th>Model</th><th>Throughput (tok/s)</th><th>Concurrency</th>
    <th>p50 (ms)</th><th>p90 (ms)</th><th>p99 (ms)</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_model_cards(scorecards: list[ModelScorecard]) -> str:
    """Section 7: Expandable per-model detail cards."""
    if not scorecards:
        return ""

    cards_html = ""
    for c in scorecards:
        model_safe = html.escape(c.model)
        details = []

        if c.speed_data:
            tps = c.speed_data.get("tokens_per_second", 0)
            ttft = c.speed_data.get("ttft_ms", 0)
            pcts = c.speed_data.get("percentiles", {})
            speed_info = f"Generation: {tps:.1f} tok/s | TTFT: {ttft:.0f}ms"
            if pcts:
                speed_info += f" | p50={pcts.get('p50_tok_per_sec', 0):.1f} p95={pcts.get('p95_tok_per_sec', 0):.1f}"
            details.append(f"<p><strong>Speed:</strong> {speed_info}</p>")

        if c.quality_data:
            pr = c.quality_data.get("pass_rate", 0)
            pc = c.quality_data.get("pass_count", 0)
            tr = c.quality_data.get("total_runnable", 0)
            details.append(f"<p><strong>Quality:</strong> {pc}/{tr} tasks passed ({pr:.0%})</p>")

        if c.eval_data:
            eval_parts = []
            for k, v in c.eval_data.items():
                if isinstance(v, dict) and "score" in v:
                    eval_parts.append(f"{k}: {v['score']:.0%}")
            if eval_parts:
                details.append(f"<p><strong>Eval:</strong> {', '.join(eval_parts)}</p>")

        if c.sql_data:
            sp = c.sql_data.get("total_pass", 0)
            st = c.sql_data.get("total_questions", 0)
            details.append(f"<p><strong>SQL:</strong> {sp}/{st} passed</p>")

        detail_html = "\n".join(details) if details else "<p>No data available</p>"
        o_cls = _score_class(c.overall_score / 100)

        cards_html += (
            f'<details class="model-card">'
            f'<summary>'
            f'<span class="model-name">{model_safe}</span> '
            f'<span class="{o_cls}">Score: {c.overall_score:.0f}</span> '
            f'<span class="runtime-label">{html.escape(c.runtime)}</span>'
            f'</summary>'
            f'<div class="card-body">{detail_html}</div>'
            f'</details>\n'
        )

    return f"""
<h2>Model Details</h2>
{cards_html}
"""


# ── Helpers ───────────────────────────────────────────────────────────────


def _score_class(value: float) -> str:
    """Return CSS class based on score (0-1 scale)."""
    if value >= 0.8:
        return "score-high"
    if value >= 0.5:
        return "score-mid"
    return "score-low"


def _build_chart_data(scorecards: list[ModelScorecard]) -> dict[str, Any]:
    """Build JSON data for all charts."""
    labels = [c.model[:25] for c in scorecards]

    # Speed data
    speed_tps = [c.speed_data.get("tokens_per_second", 0) if c.speed_data else 0 for c in scorecards]
    speed_ttft = [c.speed_data.get("ttft_ms", 0) if c.speed_data else 0 for c in scorecards]

    # Scatter plot points (tok/s, ttft)
    scatter = [
        {"x": c.speed_data.get("tokens_per_second", 0),
         "y": c.speed_data.get("ttft_ms", 0),
         "label": c.model[:20]}
        for c in scorecards if c.speed_data and c.speed_data.get("tokens_per_second", 0) > 0
    ]

    # Eval radar
    eval_benchmarks: list[str] = []
    for c in scorecards:
        if c.eval_data:
            for k, v in c.eval_data.items():
                if isinstance(v, dict) and "score" in v and k not in eval_benchmarks:
                    eval_benchmarks.append(k)

    radar_datasets = []
    colors = ["#7aa2f7", "#9ece6a", "#f7768e", "#ff9e64", "#bb9af7", "#7dcfff", "#e0af68", "#73daca"]
    for i, c in enumerate(scorecards):
        if c.eval_data:
            values = []
            for b in eval_benchmarks:
                val = c.eval_data.get(b, {})
                score = val.get("score", 0) if isinstance(val, dict) else 0
                values.append(round(score * 100, 1))
            if any(v > 0 for v in values):
                radar_datasets.append({
                    "label": c.model[:20],
                    "values": values,
                    "color": colors[i % len(colors)],
                })

    return {
        "labels": labels,
        "speed_tps": speed_tps,
        "speed_ttft": speed_ttft,
        "scatter": scatter,
        "radar_axes": eval_benchmarks,
        "radar_datasets": radar_datasets,
    }


# ── CSS ───────────────────────────────────────────────────────────────────


def _unified_css() -> str:
    return """<style>
  :root {
    --bg: #0f1117; --surface: #1a1d2e; --surface2: #222640;
    --text: #e2e8f0; --text-dim: #8892b0; --accent: #7aa2f7;
    --green: #9ece6a; --green-bg: rgba(158,206,106,0.15);
    --red: #f7768e; --red-bg: rgba(247,118,142,0.15);
    --orange: #ff9e64; --orange-bg: rgba(255,158,100,0.15);
    --border: #2d3154; --border-light: #3b4261;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    max-width: 1400px; margin: 0 auto; padding: 2rem;
  }
  .header { margin-bottom: 2rem; }
  h1 { font-size: 1.75rem; font-weight: 700; color: var(--accent); margin-bottom: 0.5rem; }
  .meta { display: flex; gap: 1.5rem; color: var(--text-dim); font-size: 0.85rem; flex-wrap: wrap; }
  h2 {
    font-size: 1.15rem; font-weight: 600; color: var(--accent);
    margin: 2rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  h3 { font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.75rem; }
  table {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
    background: var(--surface); border-radius: 8px; overflow: hidden;
  }
  th, td { padding: 0.5rem 0.65rem; text-align: left; border-bottom: 1px solid var(--border); }
  th {
    background: var(--surface2); color: var(--accent); font-weight: 600;
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em;
  }
  td { font-variant-numeric: tabular-nums; }
  td:not(:first-child) { text-align: center; }
  .model-name { font-weight: 500; text-align: left !important; white-space: nowrap; }
  .score-high { color: var(--green); font-weight: 600; }
  .score-mid { color: var(--orange); }
  .score-low { color: var(--red); }
  .na { color: var(--text-dim); }
  .cell-pass { background: var(--green-bg); color: var(--green); font-weight: 700; font-size: 0.8rem; }
  .cell-fail { background: var(--red-bg); color: var(--red); font-weight: 700; font-size: 0.8rem; }
  .cell-error { background: var(--orange-bg); color: var(--orange); font-weight: 700; font-size: 0.8rem; }
  .cell-na { color: var(--text-dim); }
  .cell-pass sup, .cell-fail sup, .cell-error sup { font-size: 0.6rem; opacity: 0.7; }
  .tier-trivial { color: #7dcfff; }
  .tier-easy { color: var(--green); }
  .tier-medium { color: var(--orange); }
  .tier-hard { color: var(--red); }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }
  .chart-box {
    background: var(--surface); border-radius: 8px; padding: 1rem;
    border: 1px solid var(--border);
  }
  canvas { width: 100% !important; height: 200px; }
  .heatmap-wrapper { overflow-x: auto; }
  .model-card {
    background: var(--surface); border-radius: 8px; margin-bottom: 0.5rem;
    border: 1px solid var(--border);
  }
  .model-card summary {
    padding: 0.65rem 1rem; cursor: pointer; font-size: 0.85rem;
    display: flex; align-items: center; gap: 0.75rem;
  }
  .model-card summary:hover { background: var(--surface2); }
  .model-card[open] summary { border-bottom: 1px solid var(--border); }
  .card-body { padding: 1rem; font-size: 0.85rem; }
  .card-body p { margin-bottom: 0.5rem; }
  .runtime-label { color: var(--text-dim); margin-left: auto; font-size: 0.8rem; }
  .footer {
    margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: var(--text-dim); font-size: 0.75rem; display: flex; justify-content: space-between;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f8f9fc; --surface: #fff; --surface2: #f0f2f8;
      --text: #1a1d2e; --text-dim: #6b7394; --accent: #2563eb;
      --green: #16a34a; --green-bg: rgba(22,163,74,0.1);
      --red: #dc2626; --red-bg: rgba(220,38,38,0.1);
      --orange: #d97706; --orange-bg: rgba(217,119,6,0.1);
      --border: #e2e8f0;
    }
  }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } body { padding: 1rem; } }
</style>"""


# ── JavaScript Charts ─────────────────────────────────────────────────────


def _chart_js() -> str:
    return """
function getStyle(prop) {
  return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
}

function drawBarChart(canvasId, values, color, unit) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || values.length === 0) return;
  const labels = REPORT_DATA.labels;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const H = parseInt(canvas.style.height) || 200;
  canvas.width = rect.width * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const pad = {top: 20, right: 20, bottom: 50, left: 55};
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const nonZero = values.filter(v => v > 0);
  const maxVal = nonZero.length ? Math.max(...nonZero) * 1.15 : 1;
  const gap = plotW / values.length;
  const barW = Math.min(gap * 0.6, 50);

  values.forEach((v, i) => {
    if (v <= 0) return;
    const x = pad.left + i * gap + (gap - barW) / 2;
    const h = (v / maxVal) * plotH;
    const y = pad.top + plotH - h;
    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, color); grad.addColorStop(1, color + '66');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.roundRect(x, y, barW, h, [4, 4, 0, 0]); ctx.fill();
    ctx.fillStyle = getStyle('--text') || '#e2e8f0';
    ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(unit === '%' ? 0 : 1) + (unit || ''), x + barW/2, y - 4);
    ctx.save(); ctx.translate(x + barW/2, H - pad.bottom + 8);
    ctx.rotate(-Math.PI / 6);
    ctx.fillStyle = getStyle('--text-dim') || '#8892b0';
    ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(labels[i] || '', 0, 0); ctx.restore();
  });

  ctx.strokeStyle = getStyle('--border') || '#2d3154';
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke();
}

function drawScatterPlot(canvasId, points) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || points.length === 0) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const H = parseInt(canvas.style.height) || 280;
  canvas.width = rect.width * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const pad = {top: 20, right: 30, bottom: 40, left: 55};
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const maxX = Math.max(...points.map(p => p.x)) * 1.1 || 1;
  const maxY = Math.max(...points.map(p => p.y)) * 1.1 || 1;

  const colors = ['#7aa2f7','#9ece6a','#f7768e','#ff9e64','#bb9af7','#7dcfff','#e0af68'];
  points.forEach((p, i) => {
    const x = pad.left + (p.x / maxX) * plotW;
    const y = pad.top + plotH - (p.y / maxY) * plotH;
    ctx.fillStyle = colors[i % colors.length];
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = getStyle('--text') || '#e2e8f0';
    ctx.font = '9px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(p.label, x + 8, y + 3);
  });

  ctx.strokeStyle = getStyle('--border') || '#2d3154';
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke();
  ctx.fillStyle = getStyle('--text-dim') || '#8892b0';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('tok/s →', pad.left + plotW/2, H - 5);
  ctx.save(); ctx.translate(12, pad.top + plotH/2);
  ctx.rotate(-Math.PI/2); ctx.fillText('TTFT (ms) →', 0, 0); ctx.restore();
}

function drawRadarChart(canvasId, axes, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || axes.length === 0 || datasets.length === 0) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const H = parseInt(canvas.style.height) || 300;
  canvas.width = rect.width * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) / 2 - 40;
  const n = axes.length;
  const angleStep = (Math.PI * 2) / n;

  // Grid
  for (let ring = 1; ring <= 4; ring++) {
    const r = (R / 4) * ring;
    ctx.strokeStyle = getStyle('--border') || '#2d3154';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = -Math.PI/2 + i * angleStep;
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Axis lines + labels
  ctx.fillStyle = getStyle('--text-dim') || '#8892b0';
  ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    const a = -Math.PI/2 + i * angleStep;
    const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
    ctx.strokeStyle = getStyle('--border') || '#2d3154';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    const lx = cx + Math.cos(a) * (R + 18), ly = cy + Math.sin(a) * (R + 18);
    ctx.fillText(axes[i], lx, ly + 3);
  }

  // Datasets
  datasets.forEach(ds => {
    ctx.strokeStyle = ds.color; ctx.fillStyle = ds.color + '33';
    ctx.lineWidth = 2; ctx.beginPath();
    ds.values.forEach((v, i) => {
      const a = -Math.PI/2 + i * angleStep;
      const r = (v / 100) * R;
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath(); ctx.fill(); ctx.stroke();
  });

  // Legend
  let ly = 10;
  datasets.forEach(ds => {
    ctx.fillStyle = ds.color;
    ctx.fillRect(10, ly, 10, 10);
    ctx.fillStyle = getStyle('--text') || '#e2e8f0';
    ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(ds.label, 24, ly + 9);
    ly += 16;
  });
}

function sortTable(tableId, colIdx) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {
    const av = parseFloat(a.cells[colIdx]?.textContent) || 0;
    const bv = parseFloat(b.cells[colIdx]?.textContent) || 0;
    return bv - av;
  });
  rows.forEach(r => tbody.appendChild(r));
}

// Initialize charts
(function() {
  const d = REPORT_DATA;
  const accent = getStyle('--accent') || '#7aa2f7';
  const green = getStyle('--green') || '#9ece6a';
  const orange = getStyle('--orange') || '#ff9e64';

  drawBarChart('speedChart', d.speed_tps, accent, '');
  drawBarChart('ttftChart', d.speed_ttft, orange, 'ms');
  drawScatterPlot('scatterChart', d.scatter || []);
  drawRadarChart('radarChart', d.radar_axes || [], d.radar_datasets || []);
})();
"""
