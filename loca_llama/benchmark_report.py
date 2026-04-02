"""Generate standalone HTML benchmark reports with embedded charts.

Produces self-contained HTML files with inline canvas-based charts,
no external dependencies. Adapted from draftbench's report generation pattern.
"""

from __future__ import annotations

import html
import json
import platform
from typing import Any


def generate_html_report(
    results: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a standalone HTML report from benchmark results.

    Args:
        results: Benchmark status response dict (with runs, aggregate, etc.)
        metadata: Optional dict with hardware info, runtime name, model, etc.

    Returns:
        Complete HTML string ready to save as a file.
    """
    meta = metadata or {}
    title = meta.get("title", "Loca-LLAMA Benchmark Report")
    runtime = html.escape(meta.get("runtime", "Unknown"))
    model = html.escape(meta.get("model", "Unknown"))
    hw_name = html.escape(meta.get("hardware", platform.machine()))
    timestamp = meta.get("timestamp", "")

    agg = results.get("aggregate", {})
    runs = results.get("runs", [])

    # Build aggregate stats table rows
    stats_rows = ""
    if agg.get("success", agg.get("runs", 0) > 0):
        stat_items = [
            ("Avg tok/s", agg.get("avg_tok_per_sec", 0)),
            ("Median tok/s", agg.get("median_tok_per_sec", 0)),
            ("P95 tok/s", agg.get("p95_tok_per_sec", 0)),
            ("Min tok/s", agg.get("min_tok_per_sec", 0)),
            ("Max tok/s", agg.get("max_tok_per_sec", 0)),
            ("Std Dev", agg.get("stddev_tok_per_sec", 0)),
            ("Avg Prefill tok/s", agg.get("avg_prefill_tok_per_sec", 0)),
            ("Avg TTFT (ms)", agg.get("avg_ttft_ms", 0)),
            ("Avg Total (ms)", agg.get("avg_total_ms", 0)),
            ("Tokens Generated", agg.get("total_tokens_generated", 0)),
            ("Runs", agg.get("runs", 0)),
        ]
        for label, value in stat_items:
            formatted = f"{value:.1f}" if isinstance(value, float) else str(value)
            stats_rows += f"<tr><td>{label}</td><td>{formatted}</td></tr>\n"

    # Build per-run table rows
    run_rows = ""
    chart_labels: list[str] = []
    chart_values: list[float] = []
    for r in runs:
        status = "Pass" if r.get("success") else "Fail"
        row_class = "" if r.get("success") else ' class="fail"'
        run_rows += (
            f'<tr{row_class}>'
            f'<td>{r.get("run_number", "")}</td>'
            f"<td>{status}</td>"
            f'<td>{r.get("tokens_per_second", 0)}</td>'
            f'<td>{r.get("prompt_tokens_per_second", 0)}</td>'
            f'<td>{r.get("time_to_first_token_ms", 0)}</td>'
            f'<td>{r.get("total_time_ms", 0)}</td>'
            f'<td>{r.get("generated_tokens", 0)}</td>'
            f"</tr>\n"
        )
        if r.get("success"):
            chart_labels.append(f"Run {r.get('run_number', '')}")
            chart_values.append(r.get("tokens_per_second", 0))

    # Escape </script> sequences to prevent script injection in embedded JSON
    chart_data_json = json.dumps(
        {"labels": chart_labels, "values": chart_values}
    ).replace("</", r"<\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #1a1b26; --surface: #24283b; --text: #c0caf5;
    --accent: #7aa2f7; --green: #9ece6a; --orange: #ff9e64;
    --red: #f7768e; --border: #3b4261;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.6;
  }}
  h1 {{ color: var(--accent); margin-bottom: .5rem; font-size: 1.5rem; }}
  h2 {{ color: var(--accent); margin: 1.5rem 0 .75rem; font-size: 1.2rem; }}
  .meta {{ color: #737aa2; margin-bottom: 1.5rem; font-size: .9rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  table {{
    width: 100%; border-collapse: collapse; margin-bottom: 1rem;
    background: var(--surface); border-radius: 8px; overflow: hidden;
  }}
  th, td {{ padding: .5rem .75rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #1e2030; color: var(--accent); font-weight: 600; }}
  td:nth-child(n+2) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.fail td {{ color: var(--red); }}
  .chart-box {{
    background: var(--surface); border-radius: 8px; padding: 1rem;
    margin-top: 1rem;
  }}
  canvas {{ width: 100% !important; height: 200px; }}
  .footer {{
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: #737aa2; font-size: .8rem;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #f5f5f5; --surface: #fff; --text: #333;
      --accent: #2563eb; --border: #e0e0e0;
    }}
    th {{ background: #f0f0f0; }}
  }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">
  <span>Runtime: {runtime}</span>
  <span>Model: {model}</span>
  <span>Hardware: {hw_name}</span>
  {f'<span>Date: {html.escape(timestamp)}</span>' if timestamp else ''}
</div>

<h2>Aggregate Statistics</h2>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{stats_rows}</tbody>
</table>

<h2>Per-Run Results</h2>
<table>
  <thead><tr>
    <th>Run</th><th>Status</th><th>tok/s</th><th>Prefill tok/s</th>
    <th>TTFT (ms)</th><th>Total (ms)</th><th>Generated</th>
  </tr></thead>
  <tbody>{run_rows}</tbody>
</table>

<div class="chart-box">
  <h2>Generation Speed per Run</h2>
  <canvas id="chart"></canvas>
</div>

<div class="footer">
  Generated by Loca-LLAMA Benchmark Report
</div>

<script>
(function() {{
  const data = {chart_data_json};
  const canvas = document.getElementById('chart');
  if (!canvas || data.values.length === 0) return;
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
  const maxVal = Math.max(...data.values) * 1.15;
  const barW = Math.min(plotW / data.values.length * 0.6, 40);
  const gap = plotW / data.values.length;

  // Bars
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#7aa2f7';
  data.values.forEach((v, i) => {{
    const x = pad.left + i * gap + (gap - barW) / 2;
    const h = (v / maxVal) * plotH;
    const y = pad.top + plotH - h;
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, h, 3);
    ctx.fill();
    // Value label
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#c0caf5';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(1), x + barW / 2, y - 4);
    // X label
    ctx.fillText(data.labels[i], x + barW / 2, H - pad.bottom + 16);
  }});

  // Y axis
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#3b4261';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.stroke();
  // Y label
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#c0caf5';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const v = (maxVal / 4) * i;
    const y = pad.top + plotH - (v / maxVal) * plotH;
    ctx.fillText(v.toFixed(1), pad.left - 6, y + 3);
  }}
}})();
</script>
</body>
</html>"""
