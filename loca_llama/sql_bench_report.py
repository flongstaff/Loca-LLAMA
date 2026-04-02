"""Generate self-contained HTML reports for SQL benchmark results.

Produces a rich heatmap-style report inspired by sql-benchmark.nicklothian.com,
with pass/fail/error heatmap, difficulty-tier scoring, speed metrics, and
expandable per-question detail. No external dependencies.
"""

from __future__ import annotations

import html
import json
import platform
from typing import Any

from .benchmark_results import categorize_model, CATEGORY_ORDER
from .sql_bench import (
    SQLTaskResult, DIFFICULTY_ORDER, SCHEMA_DDL, QUESTIONS,
    generate_seed_json,
)


def generate_sql_report(
    results: list[SQLTaskResult],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained HTML report from SQL benchmark results.

    Args:
        results: List of SQLTaskResult from run_sql_benchmark().
        metadata: Optional dict with runtime, hardware, timestamp, title.

    Returns:
        Complete HTML string ready to save as a file.
    """
    meta = metadata or {}
    title = html.escape(meta.get("title", "SQL Benchmark Report"))
    runtime = html.escape(meta.get("runtime", "Unknown"))
    hw_name = html.escape(meta.get("hardware", platform.machine()))
    timestamp = meta.get("timestamp", "")

    models = sorted(set(r.model for r in results))
    difficulties = sorted(
        set(r.difficulty for r in results),
        key=lambda d: DIFFICULTY_ORDER.get(d, 99),
    )
    questions = sorted(
        {(r.question_id, r.difficulty, r.question) for r in results},
        key=lambda x: (DIFFICULTY_ORDER.get(x[1], 99), x[0]),
    )

    # ── Scoreboard rows (grouped by category) ────────────────────────────
    scoreboard_rows = ""
    runtime_str = meta.get("runtime", "")
    grouped_models: dict[str, list[str]] = {}
    for model in models:
        cat = categorize_model(model, runtime_str)
        grouped_models.setdefault(cat, []).append(model)

    num_cols = 6 + len(difficulties)  # model + score + diffs + tps + ttft + time + retries
    for cat in CATEGORY_ORDER:
        cat_models = grouped_models.get(cat)
        if not cat_models:
            continue
        if len(models) > 1:
            scoreboard_rows += (
                f'<tr class="category-header">'
                f'<td colspan="{num_cols}">{html.escape(cat)}</td>'
                f'</tr>\n'
            )
        for model in cat_models:
            mr = [r for r in results if r.model == model]
            total_pass = sum(1 for r in mr if r.status == "pass")
            total_q = len(mr)
            pct = (total_pass / total_q * 100) if total_q else 0

            diff_cells = ""
            for d in difficulties:
                dr = [r for r in mr if r.difficulty == d]
                d_pass = sum(1 for r in dr if r.status == "pass")
                d_total = len(dr)
                d_pct = (d_pass / d_total * 100) if d_total else 0
                color_cls = "score-high" if d_pct >= 80 else "score-mid" if d_pct >= 50 else "score-low"
                diff_cells += f'<td class="{color_cls}">{d_pass}/{d_total}</td>'

            tps_vals = [r.speed_tps for r in mr if r.speed_tps > 0]
            avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
            ttft_vals = [r.ttft_ms for r in mr if r.ttft_ms > 0]
            avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0
            total_time = sum(r.total_ms for r in mr)
            retries = sum(r.retries for r in mr)

            model_short = html.escape(model[:50])
            pct_cls = "score-high" if pct >= 80 else "score-mid" if pct >= 50 else "score-low"

            scoreboard_rows += (
                f'<tr data-model="{model_short}" data-category="{html.escape(cat)}">'
                f'<td class="model-name">{model_short}</td>'
                f'<td class="{pct_cls}"><strong>{total_pass}/{total_q}</strong> ({pct:.0f}%)</td>'
                f'{diff_cells}'
                f'<td>{avg_tps:.1f}</td>'
                f'<td>{avg_ttft:.0f}</td>'
                f'<td>{total_time / 1000:.1f}s</td>'
                f'<td>{retries}</td>'
                f'</tr>\n'
            )

    # Difficulty header cells
    diff_headers = "".join(
        f'<th class="tier-{d}">{d.capitalize()}</th>' for d in difficulties
    )

    # ── Heatmap rows ────────────────────────────────────────────────────
    heatmap_header_cells = ""
    current_diff = ""
    for qid, diff, _ in questions:
        if diff != current_diff:
            current_diff = diff
        heatmap_header_cells += f'<th class="tier-{diff}" title="Q{qid}: {diff}">Q{qid}</th>'

    heatmap_rows = ""
    for model in models:
        model_short = html.escape(model[:35])
        cat = categorize_model(model, runtime_str)
        cells = ""
        for qid, diff, _ in questions:
            r = next(
                (r for r in results if r.model == model and r.question_id == qid),
                None,
            )
            if r is None:
                cells += '<td class="cell-na"></td>'
            else:
                cls = f"cell-{r.status}"
                data_key = html.escape(f"{r.model}_{r.question_id}")
                cells += f'<td class="{cls}" data-cell="{data_key}"></td>'
        heatmap_rows += (
            f'<tr data-model="{model_short}" data-category="{html.escape(cat)}">'
            f'<td class="model-name">{model_short}</td>{cells}</tr>\n'
        )

    # ── Pass rate footer ─────────────────────────────────────────────
    footer_cells = ""
    for qid, diff, _ in questions:
        q_results = [r for r in results if r.question_id == qid]
        total = len(q_results)
        passed = sum(1 for r in q_results if r.status == "pass")
        pct = (passed / total * 100) if total else 0
        pct_cls = "footer-high" if pct >= 80 else "footer-mid" if pct >= 50 else "footer-low"
        footer_cells += f'<td class="{pct_cls}">{pct:.0f}%</td>'
    heatmap_footer = f'<tr class="pass-rate-footer"><td class="model-name">pass rate</td>{footer_cells}</tr>'

    # ── Per-question detail ─────────────────────────────────────────────
    question_details = ""
    for qid, diff, q_text in questions:
        q_text_safe = html.escape(q_text)
        q_results_html = ""
        for model in models:
            r = next(
                (r for r in results if r.model == model and r.question_id == qid),
                None,
            )
            if r is None:
                continue
            status_cls = f"status-{r.status}"
            sql_safe = html.escape(r.generated_sql) if r.generated_sql else "<em>No SQL generated</em>"
            error_html = f'<p class="error-msg">{html.escape(r.error_message)}</p>' if r.error_message else ""
            retry_html = f' <span class="retry-badge">{r.retries} retries</span>' if r.retries > 0 else ""
            speed_html = f"{r.speed_tps:.0f} tok/s" if r.speed_tps > 0 else "N/A"

            # Result preview (first 5 rows)
            result_table = ""
            if r.actual_result and r.actual_columns:
                cols_html = "".join(f"<th>{html.escape(c)}</th>" for c in r.actual_columns)
                rows_html = ""
                for row in r.actual_result[:5]:
                    row_cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
                    rows_html += f"<tr>{row_cells}</tr>"
                if len(r.actual_result) > 5:
                    rows_html += f'<tr><td colspan="{len(r.actual_columns)}">... {len(r.actual_result) - 5} more rows</td></tr>'
                result_table = f'<table class="result-preview"><thead><tr>{cols_html}</tr></thead><tbody>{rows_html}</tbody></table>'

            model_safe = html.escape(model[:40])
            q_results_html += (
                f'<div class="model-result">'
                f'<div class="model-result-header">'
                f'<span class="model-label">{model_safe}</span>'
                f'<span class="{status_cls}">{r.status.upper()}</span>{retry_html}'
                f'<span class="speed-label">{speed_html}</span>'
                f'</div>'
                f'<pre class="sql-code">{sql_safe}</pre>'
                f'{error_html}{result_table}'
                f'</div>'
            )

        tier_cls = f"tier-{diff}"
        question_details += (
            f'<details class="question-detail">'
            f'<summary><span class="{tier_cls} tier-badge">{diff.capitalize()}</span> '
            f'Q{qid}: {q_text_safe}</summary>'
            f'<div class="question-body">{q_results_html}</div>'
            f'</details>\n'
        )

    # ── Chart data ──────────────────────────────────────────────────────
    chart_labels: list[str] = []
    chart_scores: list[float] = []
    chart_tps: list[float] = []
    for model in models:
        mr = [r for r in results if r.model == model]
        total_pass = sum(1 for r in mr if r.status == "pass")
        total_q = len(mr)
        chart_labels.append(model[:25])
        chart_scores.append(total_pass / total_q * 100 if total_q else 0)
        tps_vals = [r.speed_tps for r in mr if r.speed_tps > 0]
        chart_tps.append(sum(tps_vals) / len(tps_vals) if tps_vals else 0)

    # ── Difficulty breakdown data (aggregate across all models) ─────────
    diff_breakdown: list[dict[str, Any]] = []
    for d in difficulties:
        dr = [r for r in results if r.difficulty == d]
        diff_breakdown.append({
            "label": d.capitalize(),
            "pass": sum(1 for r in dr if r.status == "pass"),
            "fail": sum(1 for r in dr if r.status == "fail"),
            "error": sum(1 for r in dr if r.status == "error"),
        })

    # ── Bubble chart data (Pass Rate vs Total Time) ──────────────────
    bubble_data: list[dict[str, Any]] = []
    for model in models:
        mr = [r for r in results if r.model == model]
        total_pass = sum(1 for r in mr if r.status == "pass")
        total_q = len(mr)
        pct = (total_pass / total_q * 100) if total_q else 0
        total_time_s = sum(r.total_ms for r in mr) / 1000
        cat = categorize_model(model, runtime_str)
        bubble_data.append({
            "model": model[:40],
            "pass_rate": round(pct, 1),
            "total_time_s": round(total_time_s, 1),
            "category": cat,
            "score": f"{total_pass}/{total_q}",
        })

    # ── Per-cell data for tooltips ───────────────────────────────────
    cell_data: dict[str, dict[str, Any]] = {}
    for r in results:
        key = f"{r.model}_{r.question_id}"
        cell_data[key] = {
            "model": r.model[:40],
            "qid": r.question_id,
            "difficulty": r.difficulty,
            "status": r.status,
            "tps": round(r.speed_tps, 1),
            "ttft": round(r.ttft_ms, 0),
            "total_ms": round(r.total_ms, 0),
            "retries": r.retries,
        }

    # ── Unique categories for filter dropdown ────────────────────────
    unique_categories = []
    for cat in CATEGORY_ORDER:
        if cat in grouped_models:
            unique_categories.append(cat)

    chart_data_json = json.dumps({
        "labels": chart_labels,
        "scores": chart_scores,
        "tps": chart_tps,
        "difficulty": diff_breakdown,
        "bubbles": bubble_data,
        "cells": cell_data,
        "categories": unique_categories,
    }).replace("</", r"<\/")

    # ── DuckDB-WASM playground data ──────────────────────────────────────
    seed_data_json = json.dumps(generate_seed_json()).replace("</", r"<\/")
    schema_ddl_safe = html.escape(SCHEMA_DDL.strip())
    playground_questions = json.dumps([
        {"id": q.id, "difficulty": q.difficulty, "question": q.question,
         "reference_sql": q.reference_sql}
        for q in QUESTIONS
    ]).replace("</", r"<\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --surface2: #222640;
    --text: #e2e8f0; --text-dim: #8892b0; --accent: #7aa2f7;
    --green: #9ece6a; --green-bg: rgba(158,206,106,0.15);
    --red: #f7768e; --red-bg: rgba(247,118,142,0.15);
    --orange: #ff9e64; --orange-bg: rgba(255,158,100,0.15);
    --border: #2d3154; --border-light: #3b4261;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    max-width: 1400px; margin: 0 auto; padding: 2rem;
  }}

  /* Header */
  .header {{ margin-bottom: 2rem; }}
  h1 {{
    font-size: 1.75rem; font-weight: 700; color: var(--accent);
    letter-spacing: -0.02em; margin-bottom: 0.5rem;
  }}
  .meta {{ display: flex; gap: 1.5rem; color: var(--text-dim); font-size: 0.85rem; flex-wrap: wrap; }}
  .meta span {{ display: flex; align-items: center; gap: 0.3rem; }}

  h2 {{
    font-size: 1.15rem; font-weight: 600; color: var(--accent);
    margin: 2rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}

  /* Tables */
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
    background: var(--surface); border-radius: 8px; overflow: hidden;
  }}
  th, td {{
    padding: 0.5rem 0.65rem; text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: var(--surface2); color: var(--accent);
    font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  td {{ font-variant-numeric: tabular-nums; }}
  td:not(:first-child) {{ text-align: center; }}
  .model-name {{ font-weight: 500; text-align: left !important; white-space: nowrap; }}
  .category-header td {{
    background: var(--surface2); color: var(--accent); font-weight: 700;
    font-size: 0.82rem; padding: 0.55rem 1rem; letter-spacing: 0.02em;
    border-left: 3px solid var(--accent); text-align: left !important;
  }}

  /* Score colors */
  .score-high {{ color: var(--green); font-weight: 600; }}
  .score-mid {{ color: var(--orange); }}
  .score-low {{ color: var(--red); }}

  /* Heatmap cells — solid colored blocks */
  .heatmap-table td:not(.model-name) {{
    width: 30px; min-width: 30px; max-width: 30px; height: 28px;
    padding: 2px; cursor: pointer; border: 1px solid var(--bg);
  }}
  .cell-pass {{ background: var(--green); }}
  .cell-fail {{ background: var(--red); }}
  .cell-error {{ background: var(--orange); }}
  .cell-na {{ background: var(--surface2); }}
  .heatmap-table tr:hover td:not(.model-name) {{ opacity: 0.85; }}
  .heatmap-table tr:hover td.model-name {{ color: var(--accent); }}

  /* Pass rate footer */
  .pass-rate-footer td {{
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    background: var(--surface2) !important; padding: 0.3rem 0.1rem !important;
  }}
  .pass-rate-footer .model-name {{ font-style: italic; font-weight: 500; }}
  .footer-high {{ color: var(--green) !important; }}
  .footer-mid {{ color: var(--orange) !important; }}
  .footer-low {{ color: var(--red) !important; }}

  /* Tooltip */
  #tooltip {{
    display: none; position: fixed; z-index: 1000;
    background: #1a1d2e; color: #e2e8f0; border: 1px solid #3b4261;
    border-radius: 8px; padding: 0.75rem 1rem; max-width: 280px;
    font-size: 0.82rem; line-height: 1.5;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4); pointer-events: none;
  }}
  #tooltip .tt-model {{ font-weight: 700; font-size: 0.88rem; margin-bottom: 0.3rem; }}
  #tooltip .tt-meta {{ color: #8892b0; font-size: 0.78rem; }}
  #tooltip .tt-pass {{ color: #9ece6a; font-weight: 600; }}
  #tooltip .tt-fail {{ color: #f7768e; font-weight: 600; }}
  #tooltip .tt-error {{ color: #ff9e64; font-weight: 600; }}
  #tooltip .tt-detail {{ color: #7aa2f7; font-size: 0.75rem; margin-top: 0.3rem; }}

  /* Filter bar */
  .filter-bar {{
    display: flex; align-items: center; gap: 0.75rem;
    margin-bottom: 0.75rem; flex-wrap: wrap;
  }}
  .filter-bar select {{
    padding: 0.4rem 0.6rem; background: var(--surface);
    color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.82rem; min-width: 100px;
  }}
  .filter-bar input {{
    padding: 0.4rem 0.75rem; background: var(--surface);
    color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.82rem; min-width: 160px;
  }}
  .filter-bar input::placeholder {{ color: var(--text-dim); }}
  .filter-bar .legend {{ margin-left: auto; }}

  /* Bubble chart */
  .bubble-section {{ position: relative; }}
  .bubble-section canvas {{ width: 100% !important; height: 400px; }}
  .zoom-controls {{
    position: absolute; top: 10px; right: 10px;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .zoom-controls button {{
    width: 32px; height: 32px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); border-radius: 4px;
    cursor: pointer; font-size: 1.1rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }}
  .zoom-controls button:hover {{ background: var(--surface2); }}

  /* Tier badges */
  .tier-trivial {{ color: #7dcfff; }}
  .tier-easy {{ color: var(--green); }}
  .tier-medium {{ color: var(--orange); }}
  .tier-hard {{ color: var(--red); }}
  .tier-badge {{
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 4px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .tier-trivial.tier-badge {{ background: rgba(125,207,255,0.15); }}
  .tier-easy.tier-badge {{ background: var(--green-bg); }}
  .tier-medium.tier-badge {{ background: var(--orange-bg); }}
  .tier-hard.tier-badge {{ background: var(--red-bg); }}

  /* Charts */
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
  .charts .chart-box.full-width {{ grid-column: 1 / -1; }}
  .legend {{ display: flex; gap: 1.2rem; margin-top: 0.5rem; font-size: 0.78rem; color: var(--text-dim); }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; }}
  .chart-box {{
    background: var(--surface); border-radius: 8px; padding: 1rem;
    border: 1px solid var(--border);
  }}
  .chart-box h3 {{ font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.75rem; }}
  canvas {{ width: 100% !important; height: 200px; }}

  /* Question details */
  .question-detail {{
    background: var(--surface); border-radius: 8px;
    margin-bottom: 0.5rem; border: 1px solid var(--border);
  }}
  .question-detail summary {{
    padding: 0.65rem 1rem; cursor: pointer; font-size: 0.85rem;
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .question-detail summary:hover {{ background: var(--surface2); }}
  .question-detail[open] summary {{ border-bottom: 1px solid var(--border); }}
  .question-body {{ padding: 1rem; }}
  .model-result {{
    margin-bottom: 1rem; padding: 0.75rem;
    background: var(--surface2); border-radius: 6px;
  }}
  .model-result-header {{
    display: flex; align-items: center; gap: 0.75rem;
    margin-bottom: 0.5rem; font-size: 0.85rem;
  }}
  .model-label {{ font-weight: 600; }}
  .speed-label {{ color: var(--text-dim); margin-left: auto; }}
  .retry-badge {{
    font-size: 0.7rem; padding: 0.1rem 0.4rem;
    background: var(--orange-bg); color: var(--orange);
    border-radius: 3px;
  }}
  .status-pass {{ color: var(--green); font-weight: 700; }}
  .status-fail {{ color: var(--red); font-weight: 700; }}
  .status-error {{ color: var(--orange); font-weight: 700; }}
  .sql-code {{
    background: var(--bg); padding: 0.75rem; border-radius: 4px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem;
    overflow-x: auto; white-space: pre-wrap; color: var(--accent);
    border: 1px solid var(--border);
  }}
  .error-msg {{
    color: var(--red); font-size: 0.8rem;
    margin: 0.5rem 0; padding: 0.4rem 0.6rem;
    background: var(--red-bg); border-radius: 4px;
  }}
  .result-preview {{
    margin-top: 0.5rem; font-size: 0.78rem;
    background: var(--bg); border: 1px solid var(--border);
  }}
  .result-preview th {{ font-size: 0.72rem; padding: 0.3rem 0.5rem; }}
  .result-preview td {{ font-size: 0.75rem; padding: 0.25rem 0.5rem; text-align: left; }}

  /* Footer */
  .footer {{
    margin-top: 2.5rem; padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim); font-size: 0.75rem;
    display: flex; justify-content: space-between;
  }}

  /* Light theme */
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #f8f9fc; --surface: #fff; --surface2: #f0f2f8;
      --text: #1a1d2e; --text-dim: #6b7394; --accent: #2563eb;
      --green: #16a34a; --green-bg: rgba(22,163,74,0.1);
      --red: #dc2626; --red-bg: rgba(220,38,38,0.1);
      --orange: #d97706; --orange-bg: rgba(217,119,6,0.1);
      --border: #e2e8f0; --border-light: #d1d5db;
    }}
  }}

  @media (max-width: 900px) {{
    .charts {{ grid-template-columns: 1fr; }}
    body {{ padding: 1rem; }}
  }}

  /* Heatmap scroll */
  .heatmap-wrapper {{ overflow-x: auto; }}

  /* Flowchart */
  .flow-section {{ margin: 1.5rem 0; }}
  .flow-section canvas {{ width: 100% !important; height: 280px; background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }}

  /* SQL Playground */
  .sql-playground {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }}
  .sql-playground .schema-box {{
    background: var(--surface); border-radius: 8px; padding: 1rem;
    border: 1px solid var(--border); max-height: 400px; overflow-y: auto;
  }}
  .sql-playground .editor-box {{
    background: var(--surface); border-radius: 8px; padding: 1rem;
    border: 1px solid var(--border);
  }}
  .sql-playground h3 {{ font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.75rem; }}
  .sql-playground select {{
    width: 100%; padding: 0.4rem; margin-bottom: 0.5rem;
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; font-size: 0.82rem;
  }}
  .sql-playground textarea {{
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82rem;
    padding: 0.75rem; resize: vertical; min-height: 80px;
  }}
  .sql-playground .btn-row {{ display: flex; gap: 0.5rem; margin-top: 0.5rem; }}
  .sql-playground button {{
    padding: 0.45rem 1.2rem; border: none; border-radius: 4px;
    font-weight: 600; font-size: 0.82rem; cursor: pointer;
  }}
  .sql-playground .btn-run {{ background: var(--accent); color: #fff; }}
  .sql-playground .btn-run:disabled {{ opacity: 0.5; cursor: wait; }}
  .sql-playground .btn-answer {{ background: var(--surface2); color: var(--text-dim); }}
  #sqlResult {{ margin-top: 0.75rem; overflow-x: auto; }}
  #sqlResult table {{ font-size: 0.78rem; }}
  #sqlError {{ margin-top: 0.5rem; }}
  #sqlStatus {{ color: var(--text-dim); font-size: 0.78rem; margin-top: 0.5rem; }}
  @media (max-width: 900px) {{ .sql-playground {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="meta">
    <span>Runtime: {runtime}</span>
    <span>Hardware: {hw_name}</span>
    <span>Questions: {len(questions)}</span>
    <span>Models: {len(models)}</span>
    {f'<span>Date: {html.escape(timestamp)}</span>' if timestamp else ''}
  </div>
</div>

<h2>Evaluation Flow</h2>
<div class="flow-section">
  <canvas id="flowChart"></canvas>
</div>

<h2>Scoreboard</h2>
<table>
  <thead><tr>
    <th>Model</th><th>Score</th>
    {diff_headers}
    <th>Avg tok/s</th><th>Avg TTFT (ms)</th><th>Total Time</th><th>Retries</th>
  </tr></thead>
  <tbody>{scoreboard_rows}</tbody>
</table>

<div class="charts">
  <div class="chart-box">
    <h3>Pass Rate (%)</h3>
    <canvas id="scoreChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Average Speed (tok/s)</h3>
    <canvas id="tpsChart"></canvas>
  </div>
  <div class="chart-box full-width">
    <h3>Results by Difficulty Tier</h3>
    <canvas id="diffChart"></canvas>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:var(--green)"></span> Pass</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--red)"></span> Fail</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--orange)"></span> Error</span>
    </div>
  </div>
</div>

<h2>Pass Rate vs Total Time</h2>
<div class="chart-box bubble-section">
  <canvas id="bubbleChart"></canvas>
  <div class="zoom-controls">
    <button onclick="bubbleZoom(1)" title="Zoom in">+</button>
    <button onclick="bubbleZoom(-1)" title="Zoom out">&minus;</button>
  </div>
</div>

<h2>Question Heatmap</h2>
<div class="filter-bar">
  <select id="catFilter" onchange="filterModels()">
    <option value="">All</option>
  </select>
  <input type="text" id="textFilter" placeholder="Filter..." oninput="filterModels()">
  <div class="legend">
    <span class="legend-item"><span class="legend-swatch" style="background:var(--green)"></span> Pass</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--red)"></span> Fail</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--orange)"></span> Error</span>
  </div>
</div>
<div class="heatmap-wrapper">
<table class="heatmap-table">
  <thead><tr>
    <th>Model</th>
    {heatmap_header_cells}
  </tr></thead>
  <tbody>{heatmap_rows}</tbody>
  <tfoot>{heatmap_footer}</tfoot>
</table>
</div>
<div id="tooltip"></div>

<h2>Question Details</h2>
{question_details}

<h2>Try It Yourself</h2>
<div class="sql-playground">
  <div class="schema-box">
    <h3>Database Schema</h3>
    <pre class="sql-code">{schema_ddl_safe}</pre>
  </div>
  <div class="editor-box">
    <h3>SQL Query</h3>
    <select id="exampleSelect" onchange="loadExample()">
      <option value="">Select an example question...</option>
    </select>
    <textarea id="sqlInput" rows="4" placeholder="SELECT * FROM customers LIMIT 10;"></textarea>
    <div class="btn-row">
      <button class="btn-run" id="runBtn" onclick="runQuery()">Run Query</button>
      <button class="btn-answer" id="answerBtn" onclick="showAnswer()" style="display:none">Show Reference SQL</button>
    </div>
    <div id="sqlStatus"></div>
    <div id="sqlError" class="error-msg" style="display:none"></div>
    <div id="sqlResult"></div>
  </div>
</div>

<div class="footer">
  <span>Generated by Loca-LLAMA SQL Benchmark</span>
  <span>P = Pass, F = Fail, E = Error. Superscript = retry count.</span>
</div>

<script>
(function() {{
  const data = {chart_data_json};

  // ── Agentic loop flowchart ──────────────────────────────────────────
  (function drawFlowChart() {{
    const canvas = document.getElementById('flowChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = 280;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    const s = getComputedStyle(document.documentElement);
    const c = {{
      accent: s.getPropertyValue('--accent').trim() || '#7aa2f7',
      green: s.getPropertyValue('--green').trim() || '#9ece6a',
      red: s.getPropertyValue('--red').trim() || '#f7768e',
      orange: s.getPropertyValue('--orange').trim() || '#ff9e64',
      text: s.getPropertyValue('--text').trim() || '#e2e8f0',
      dim: s.getPropertyValue('--text-dim').trim() || '#8892b0',
      bg: s.getPropertyValue('--surface2').trim() || '#222640',
      border: s.getPropertyValue('--border').trim() || '#2d3154',
    }};
    const bw = 120, bh = 36, dw = 90, dh = 50;
    const cy = H / 2;

    function box(x, y, w, h, label, color) {{
      ctx.fillStyle = color + '22';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.roundRect(x - w/2, y - h/2, w, h, 6); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.text; ctx.font = '12px -apple-system, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, x, y);
    }}
    function diamond(x, y, w, h, label, color) {{
      ctx.fillStyle = color + '22';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, y - h/2); ctx.lineTo(x + w/2, y); ctx.lineTo(x, y + h/2); ctx.lineTo(x - w/2, y);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.text; ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, x, y);
    }}
    function arrow(x1, y1, x2, y2, color) {{
      ctx.strokeStyle = color || c.dim; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      const angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.fillStyle = color || c.dim;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 8 * Math.cos(angle - 0.4), y2 - 8 * Math.sin(angle - 0.4));
      ctx.lineTo(x2 - 8 * Math.cos(angle + 0.4), y2 - 8 * Math.sin(angle + 0.4));
      ctx.closePath(); ctx.fill();
    }}
    function arrowLabel(x, y, label, color) {{
      ctx.fillStyle = color || c.dim;
      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, x, y);
    }}

    // Positions (spread across width)
    const xs = [0.08, 0.24, 0.42, 0.60, 0.78].map(p => p * W);
    const x1 = xs[0], x2 = xs[1], x3 = xs[2], x4 = xs[3], x5 = xs[4];

    // Main flow boxes
    box(x1, cy, bw, bh, 'Schema + Query', c.accent);
    box(x2, cy, bw, bh, 'LLM generates SQL', c.accent);
    box(x3, cy, bw, bh, 'Execute SQL', c.accent);
    diamond(x4, cy, dw + 10, dh, 'Match?', c.orange);

    // Arrows between main boxes
    arrow(x1 + bw/2, cy, x2 - bw/2, cy, c.dim);
    arrow(x2 + bw/2, cy, x3 - bw/2, cy, c.dim);
    arrow(x3 + bw/2, cy, x4 - (dw+10)/2, cy, c.dim);

    // Pass path (down from diamond)
    const passY = cy + 80;
    box(x4, passY, bw, bh, 'Record PASS', c.green);
    arrow(x4, cy + dh/2, x4, passY - bh/2, c.green);
    arrowLabel(x4 + 16, cy + dh/2 + 12, 'Yes', c.green);

    // Fail/retry path (right from diamond)
    diamond(x5, cy, dw, dh, 'Retries < 2?', c.orange);
    arrow(x4 + (dw+10)/2, cy, x5 - dw/2, cy, c.red);
    arrowLabel((x4 + (dw+10)/2 + x5 - dw/2) / 2, cy - 10, 'No', c.red);

    // Record FAIL (down from retries diamond)
    box(x5, passY, bw, bh, 'Record FAIL', c.red);
    arrow(x5, cy + dh/2, x5, passY - bh/2, c.red);
    arrowLabel(x5 + 16, cy + dh/2 + 12, 'No', c.red);

    // Retry loop (up from retries diamond, curved back to LLM box)
    const loopY = cy - 65;
    ctx.strokeStyle = c.orange; ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x5, cy - dh/2);
    ctx.lineTo(x5, loopY);
    ctx.lineTo(x2, loopY);
    ctx.lineTo(x2, cy - bh/2);
    ctx.stroke();
    ctx.setLineDash([]);
    // Arrowhead at end of loop
    ctx.fillStyle = c.orange;
    ctx.beginPath();
    ctx.moveTo(x2, cy - bh/2);
    ctx.lineTo(x2 - 5, cy - bh/2 - 8);
    ctx.lineTo(x2 + 5, cy - bh/2 - 8);
    ctx.closePath(); ctx.fill();
    arrowLabel(x5 + 16, cy - dh/2 - 10, 'Yes', c.orange);
    arrowLabel((x5 + x2) / 2, loopY - 10, 'Feed error back to LLM', c.orange);
  }})();

  function drawBarChart(canvasId, values, color, unit) {{
    const canvas = document.getElementById(canvasId);
    if (!canvas || values.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = 200;
    const pad = {{ top: 20, right: 20, bottom: 50, left: 50 }};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const maxVal = Math.max(...values) * 1.15 || 1;
    const gap = plotW / values.length;
    const barW = Math.min(gap * 0.6, 50);

    values.forEach((v, i) => {{
      const x = pad.left + i * gap + (gap - barW) / 2;
      const h = (v / maxVal) * plotH;
      const y = pad.top + plotH - h;

      // Gradient bar
      const grad = ctx.createLinearGradient(x, y, x, y + h);
      grad.addColorStop(0, color);
      grad.addColorStop(1, color + '66');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, h, [4, 4, 0, 0]);
      ctx.fill();

      // Value label
      const style = getComputedStyle(document.documentElement);
      ctx.fillStyle = style.getPropertyValue('--text').trim() || '#e2e8f0';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(unit === '%' ? 0 : 1) + (unit || ''), x + barW / 2, y - 5);

      // X label (rotated)
      ctx.save();
      ctx.translate(x + barW / 2, H - pad.bottom + 8);
      ctx.rotate(-Math.PI / 6);
      ctx.fillStyle = style.getPropertyValue('--text-dim').trim() || '#8892b0';
      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(data.labels[i], 0, 0);
      ctx.restore();
    }});

    // Y axis
    const style = getComputedStyle(document.documentElement);
    ctx.strokeStyle = style.getPropertyValue('--border').trim() || '#2d3154';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = style.getPropertyValue('--text-dim').trim() || '#8892b0';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
      const v = (maxVal / 4) * i;
      const y = pad.top + plotH - (v / maxVal) * plotH;
      ctx.fillText(v.toFixed(0), pad.left - 6, y + 3);
    }}
  }}

  function drawStackedBarChart(canvasId, items) {{
    const canvas = document.getElementById(canvasId);
    if (!canvas || items.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = 200;
    const pad = {{ top: 20, right: 20, bottom: 40, left: 50 }};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const style = getComputedStyle(document.documentElement);
    const colors = {{
      pass: style.getPropertyValue('--green').trim() || '#9ece6a',
      fail: style.getPropertyValue('--red').trim() || '#f7768e',
      error: style.getPropertyValue('--orange').trim() || '#ff9e64'
    }};
    const maxVal = Math.max(...items.map(d => d.pass + d.fail + d.error)) * 1.15 || 1;
    const gap = plotW / items.length;
    const barW = Math.min(gap * 0.5, 80);

    items.forEach((item, i) => {{
      const x = pad.left + i * gap + (gap - barW) / 2;
      let y = pad.top + plotH;
      const keys = ['pass', 'fail', 'error'];
      // Find the topmost (last drawn) segment for rounded corners
      const lastKey = keys.filter(k => item[k] > 0).pop();
      keys.forEach(key => {{
        const h = (item[key] / maxVal) * plotH;
        if (h > 0) {{
          y -= h;
          ctx.fillStyle = colors[key];
          ctx.beginPath();
          ctx.roundRect(x, y, barW, h, key === lastKey ? [4,4,0,0] : [0,0,0,0]);
          ctx.fill();
        }}
      }});
      // Total label
      const total = item.pass + item.fail + item.error;
      ctx.fillStyle = style.getPropertyValue('--text').trim() || '#e2e8f0';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(total.toString(), x + barW / 2, y - 5);
      // X label
      ctx.fillStyle = style.getPropertyValue('--text-dim').trim() || '#8892b0';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillText(item.label, x + barW / 2, H - pad.bottom + 14);
    }});

    // Y axis
    ctx.strokeStyle = style.getPropertyValue('--border').trim() || '#2d3154';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();
    ctx.fillStyle = style.getPropertyValue('--text-dim').trim() || '#8892b0';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
      const v = (maxVal / 4) * i;
      const yp = pad.top + plotH - (v / maxVal) * plotH;
      ctx.fillText(v.toFixed(0), pad.left - 6, yp + 3);
    }}
  }}

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#7aa2f7';
  const green = getComputedStyle(document.documentElement).getPropertyValue('--green').trim() || '#9ece6a';
  drawBarChart('scoreChart', data.scores, green, '%');
  drawBarChart('tpsChart', data.tps, accent, '');
  drawStackedBarChart('diffChart', data.difficulty || []);

  // ── Tooltip ──────────────────────────────────────────────────────
  const tooltip = document.getElementById('tooltip');
  const cells = data.cells || {{}};

  document.querySelectorAll('.heatmap-table td[data-cell]').forEach(td => {{
    td.addEventListener('mouseenter', function(e) {{
      const key = this.getAttribute('data-cell');
      const c = cells[key];
      if (!c) return;
      const statusCls = 'tt-' + c.status;
      const timeStr = c.total_ms >= 1000
        ? (c.total_ms / 1000).toFixed(1) + 's'
        : c.total_ms.toFixed(0) + 'ms';
      let html = '<div class="tt-model">' + c.model + '</div>';
      html += '<div class="tt-meta">Q' + c.qid + ' &middot; ' + c.difficulty
        + ' &middot; <span class="' + statusCls + '">' + c.status + '</span></div>';
      html += '<div class="tt-meta">' + timeStr + '</div>';
      if (c.tps > 0) html += '<div class="tt-meta">' + c.tps + ' tok/s</div>';
      if (c.ttft > 0) html += '<div class="tt-meta">TTFT: ' + c.ttft + 'ms</div>';
      if (c.retries > 0) html += '<div class="tt-meta">Retries: ' + c.retries + '</div>';
      html += '<div class="tt-detail">click for details</div>';
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      positionTooltip(e);
    }});
    td.addEventListener('mousemove', positionTooltip);
    td.addEventListener('mouseleave', function() {{
      tooltip.style.display = 'none';
    }});
  }});

  function positionTooltip(e) {{
    const pad = 12;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    const tw = 280, th = 200;
    if (x + tw > window.innerWidth) x = e.clientX - tw - pad;
    if (y + th > window.innerHeight) y = e.clientY - th - pad;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }}

  // ── Filter ───────────────────────────────────────────────────────
  const catSelect = document.getElementById('catFilter');
  if (catSelect && data.categories) {{
    data.categories.forEach(c => {{
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      catSelect.appendChild(opt);
    }});
  }}

  function filterModels() {{
    const cat = (document.getElementById('catFilter') || {{}}).value || '';
    const text = ((document.getElementById('textFilter') || {{}}).value || '').toLowerCase();

    // Filter heatmap rows
    document.querySelectorAll('.heatmap-table tbody tr').forEach(tr => {{
      const model = (tr.getAttribute('data-model') || '').toLowerCase();
      const rowCat = tr.getAttribute('data-category') || '';
      const catMatch = !cat || rowCat === cat;
      const textMatch = !text || model.includes(text);
      tr.style.display = catMatch && textMatch ? '' : 'none';
    }});

    // Filter scoreboard rows
    document.querySelectorAll('table:not(.heatmap-table) tbody tr:not(.category-header)').forEach(tr => {{
      const model = (tr.getAttribute('data-model') || '').toLowerCase();
      const rowCat = tr.getAttribute('data-category') || '';
      if (!model) return;
      const catMatch = !cat || rowCat === cat;
      const textMatch = !text || model.includes(text);
      tr.style.display = catMatch && textMatch ? '' : 'none';
    }});

    // Show/hide category headers based on visible children
    document.querySelectorAll('.category-header').forEach(ch => {{
      let next = ch.nextElementSibling;
      let hasVisible = false;
      while (next && !next.classList.contains('category-header')) {{
        if (next.style.display !== 'none') hasVisible = true;
        next = next.nextElementSibling;
      }}
      ch.style.display = hasVisible ? '' : 'none';
    }});
  }}
  window.filterModels = filterModels;

  // ── Bubble Chart (Pass Rate vs Total Time, log scale) ────────────
  let bubbleZoomLevel = 0;

  function drawBubbleChart() {{
    const canvas = document.getElementById('bubbleChart');
    if (!canvas || !data.bubbles || data.bubbles.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = 400;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    const style = getComputedStyle(document.documentElement);
    const colors = {{
      accent: style.getPropertyValue('--accent').trim() || '#7aa2f7',
      green: style.getPropertyValue('--green').trim() || '#9ece6a',
      text: style.getPropertyValue('--text').trim() || '#e2e8f0',
      dim: style.getPropertyValue('--text-dim').trim() || '#8892b0',
      border: style.getPropertyValue('--border').trim() || '#2d3154',
      surface: style.getPropertyValue('--surface').trim() || '#1a1d2e',
    }};

    const pad = {{ top: 30, right: 40, bottom: 50, left: 70 }};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Data ranges with zoom
    const times = data.bubbles.map(b => b.total_time_s).filter(t => t > 0);
    const zoomFactor = Math.pow(0.7, bubbleZoomLevel);
    let minT = Math.max(1, Math.min(...times) * 0.8);
    let maxT = Math.max(...times) * 1.2;
    const midT = Math.sqrt(minT * maxT);
    minT = midT / Math.pow(midT / minT, zoomFactor);
    maxT = midT * Math.pow(maxT / midT, zoomFactor);
    const logMin = Math.log10(minT), logMax = Math.log10(maxT);

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Grid lines (dashed)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 0.5;

    // Y grid (every 10%)
    for (let pct = 0; pct <= 100; pct += 10) {{
      const y = pad.top + plotH - (pct / 100) * plotH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    }}

    // X grid (log ticks)
    const logTicks = [];
    const startPow = Math.floor(logMin);
    const endPow = Math.ceil(logMax);
    for (let p = startPow; p <= endPow; p++) {{
      for (const m of [1, 2, 5]) {{
        const v = m * Math.pow(10, p);
        if (v >= minT && v <= maxT) logTicks.push(v);
      }}
    }}
    logTicks.forEach(v => {{
      const x = pad.left + ((Math.log10(v) - logMin) / (logMax - logMin)) * plotW;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    }});
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = colors.border; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = colors.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let pct = 0; pct <= 100; pct += 25) {{
      const y = pad.top + plotH - (pct / 100) * plotH;
      ctx.fillText(pct + '%', pad.left - 8, y);
    }}

    // X-axis labels
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    logTicks.forEach(v => {{
      const x = pad.left + ((Math.log10(v) - logMin) / (logMax - logMin)) * plotW;
      const label = v >= 1000 ? (v/1000).toFixed(0) + 'ks' : v.toFixed(0) + 's';
      ctx.fillText(label, x, pad.top + plotH + 8);
    }});

    // Axis titles
    ctx.fillStyle = colors.dim; ctx.font = '12px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Total Time (seconds, log scale)', pad.left + plotW / 2, H - 8);
    ctx.save();
    ctx.translate(16, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Pass Rate (%)', 0, 0);
    ctx.restore();

    // Bubbles
    const maxBubbleR = 24, minBubbleR = 6;
    const scores = data.bubbles.map(b => {{
      const parts = b.score.split('/');
      return parseInt(parts[0]) + parseInt(parts[1]);
    }});
    const maxScore = Math.max(...scores), minScore = Math.min(...scores);

    data.bubbles.forEach((b, i) => {{
      if (b.total_time_s <= 0) return;
      const logT = Math.log10(b.total_time_s);
      const x = pad.left + ((logT - logMin) / (logMax - logMin)) * plotW;
      const y = pad.top + plotH - (b.pass_rate / 100) * plotH;

      // Clamp to plot area
      if (x < pad.left || x > pad.left + plotW || y < pad.top || y > pad.top + plotH) return;

      const scoreVal = scores[i];
      const r = maxScore === minScore ? 12
        : minBubbleR + ((scoreVal - minScore) / (maxScore - minScore)) * (maxBubbleR - minBubbleR);

      // Color by category
      const isLocal = b.category.toLowerCase().includes('local')
        || b.category.toLowerCase().includes('gguf');
      const color = isLocal ? colors.green : colors.accent;

      ctx.globalAlpha = 0.6;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();

      // Label
      ctx.fillStyle = colors.text; ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(b.model, x + r + 4, y - 2);
    }});
  }}

  drawBubbleChart();

  function bubbleZoom(dir) {{
    bubbleZoomLevel += dir;
    bubbleZoomLevel = Math.max(-3, Math.min(5, bubbleZoomLevel));
    drawBubbleChart();
  }}
  window.bubbleZoom = bubbleZoom;

  // Redraw on resize
  window.addEventListener('resize', function() {{
    drawBubbleChart();
  }});
}})();
</script>

<!-- DuckDB-WASM SQL Playground -->
<script>
(function() {{
  const SEED = {seed_data_json};
  const PQ = {playground_questions};
  const SCHEMA = `{SCHEMA_DDL.strip()}`;

  // Populate example dropdown
  const sel = document.getElementById('exampleSelect');
  if (sel) {{
    PQ.forEach(q => {{
      const opt = document.createElement('option');
      opt.value = q.id;
      opt.textContent = '[' + q.difficulty + '] Q' + q.id + ': ' + q.question.slice(0, 80);
      sel.appendChild(opt);
    }});
  }}

  let currentRef = '';
  function loadExample() {{
    const qid = parseInt(sel.value);
    const q = PQ.find(x => x.id === qid);
    if (q) {{
      document.getElementById('sqlInput').value = '-- ' + q.question + '\\n';
      currentRef = q.reference_sql;
      document.getElementById('answerBtn').style.display = 'inline-block';
    }} else {{
      currentRef = '';
      document.getElementById('answerBtn').style.display = 'none';
    }}
    document.getElementById('sqlResult').innerHTML = '';
    document.getElementById('sqlError').style.display = 'none';
  }}
  window.loadExample = loadExample;

  function showAnswer() {{
    if (currentRef) {{
      document.getElementById('sqlInput').value = currentRef;
    }}
  }}
  window.showAnswer = showAnswer;

  let db = null, conn = null, loading = false;

  async function initDuckDB() {{
    if (conn) return;
    if (loading) return;
    loading = true;
    const status = document.getElementById('sqlStatus');
    status.textContent = 'Loading DuckDB-WASM...';

    try {{
      // Dynamically load DuckDB-WASM
      await new Promise((resolve, reject) => {{
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0/dist/duckdb-browser-eh.js';
        script.onload = resolve;
        script.onerror = () => reject(new Error('Failed to load DuckDB-WASM from CDN'));
        document.head.appendChild(script);
      }});

      status.textContent = 'Initializing database...';
      const DUCKDB_BUNDLES = duckdb.getJsDelivrBundles();
      const bundle = await duckdb.selectBundle(DUCKDB_BUNDLES);
      const worker = new Worker(bundle.mainWorker);
      const logger = new duckdb.ConsoleLogger();
      db = new duckdb.AsyncDuckDB(logger, worker);
      await db.instantiate(bundle.mainModule);
      conn = await db.connect();

      // Create schema
      await conn.query(SCHEMA);

      // Seed data using batched INSERT statements
      status.textContent = 'Seeding data...';
      function esc(v) {{ return typeof v === 'string' ? "'" + v.replace(/'/g, "''") + "'" : String(v); }}
      function batchInsert(table, rows, batchSize) {{
        const stmts = [];
        for (let i = 0; i < rows.length; i += batchSize) {{
          const chunk = rows.slice(i, i + batchSize);
          const vals = chunk.map(r => '(' + r.map(esc).join(',') + ')').join(',');
          stmts.push('INSERT INTO ' + table + ' VALUES ' + vals);
        }}
        return stmts;
      }}
      const allStmts = [
        ...batchInsert('customers', SEED.customers, 50),
        ...batchInsert('products', SEED.products, 50),
        ...batchInsert('orders', SEED.orders, 50),
        ...batchInsert('order_items', SEED.order_items, 100),
      ];
      for (const sql of allStmts) {{
        await conn.query(sql);
      }}
      status.textContent = 'Database ready. ' + SEED.customers.length + ' customers, '
        + SEED.products.length + ' products, ' + SEED.orders.length + ' orders loaded.';
    }} catch(e) {{
      status.textContent = '';
      loading = false;
      throw e;
    }}
  }}

  async function runQuery() {{
    const btn = document.getElementById('runBtn');
    const errDiv = document.getElementById('sqlError');
    const resultDiv = document.getElementById('sqlResult');
    const sql = document.getElementById('sqlInput').value.trim();
    errDiv.style.display = 'none';
    resultDiv.innerHTML = '';

    if (!sql) return;

    btn.disabled = true;
    btn.textContent = 'Running...';
    try {{
      await initDuckDB();
      const result = await conn.query(sql);
      const columns = result.schema.fields.map(f => f.name);
      const rows = result.toArray().map(row => {{
        const obj = row.toJSON();
        return columns.map(c => obj[c]);
      }});

      // Render table
      let h = '<table><thead><tr>' + columns.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>';
      const maxRows = Math.min(rows.length, 50);
      for (let i = 0; i < maxRows; i++) {{
        h += '<tr>' + rows[i].map(v => '<td style="text-align:left">' + (v === null ? 'NULL' : String(v)) + '</td>').join('') + '</tr>';
      }}
      if (rows.length > 50) {{
        h += '<tr><td colspan="' + columns.length + '">... ' + (rows.length - 50) + ' more rows</td></tr>';
      }}
      h += '</tbody></table>';
      h += '<div style="color:var(--text-dim);font-size:0.75rem;margin-top:0.3rem">' + rows.length + ' rows returned</div>';
      resultDiv.innerHTML = h;
    }} catch(e) {{
      errDiv.textContent = e.message || String(e);
      errDiv.style.display = 'block';
    }} finally {{
      btn.disabled = false;
      btn.textContent = 'Run Query';
    }}
  }}
  window.runQuery = runQuery;
}})();
</script>
</body>
</html>"""
