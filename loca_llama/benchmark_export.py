"""Export benchmark results to CSV and markdown with preset information."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkResult


def result_to_row(result: BenchmarkResult) -> dict[str, Any]:
    """Convert a BenchmarkResult to a CSV row dict.
    
    Creates unique rows by combining model name with preset name.
    Same model with different presets will generate separate rows.
    """
    model_name = result.model_name
    preset = result.preset_name or "default"
    
    # Combine model and preset as unique identifier
    if preset and preset != "default":
        csv_key = f"{model_name}@{preset}"
    else:
        csv_key = model_name
    
    return {
        "MODEL": csv_key,
        "BENCHMARK_TYPE": result.runtime,
        "PRESET_NAME": preset,
        "PRESET_CONTEXT_LEN": result.context_length,
        "GEN_SPEED_TOKS": round(result.tokens_per_second, 1) if result.success else "---",
        "CODING_SPEED_TOKS": "---",  # Would need separate benchmark results
        "REASONING_SPEED_TOKS": "---",
        "CREATIVE_SPEED_TOKS": "---",
        "PEAK_MEMORY_GB": "---",  # Would need to be passed separately
        "PEAK_MEMORY_PCT": "---",
        "TTFT_GEN_MS": round(result.time_to_first_token_ms, 0) if result.success else "---",
        "TTFT_CODING_MS": "---",
        "TTFT_REASONING_MS": "---",
        "TTFT_CREATIVE_MS": "---",
        "PREFILL_SPEED": "---",
        "TOTAL_DURATION_S": round(result.total_time_ms / 1000, 1),
        "SUCCESS": "Y" if result.success else "N",
        "ERROR": result.error or "",
        "RUN_NUMBER": result.run_number,
    }


def benchmark_results_to_csv(
    results: list[BenchmarkResult],
    output_path: str,
    memory_data: dict[str, Any] | None = None,
    coding_results: list[BenchmarkResult] | None = None,
    reasoning_results: list[BenchmarkResult] | None = None,
    creative_results: list[BenchmarkResult] | None = None,
) -> None:
    """Export benchmark results to CSV with preset support.
    
    Args:
        results: List of benchmark results (default prompt)
        output_path: Path to CSV file
        memory_data: Optional dict of model->(peak_gb, peak_pct)
        coding_results: Optional results from coding benchmark
        reasoning_results: Optional results from reasoning benchmark
        creative_results: Optional results from creative benchmark
    
    Creates unique rows for each model@preset combination.
    Same model with different presets generates separate rows.
    """
    if not results:
        return

    # Group results by preset to create unique rows
    seen_rows = set()
    rows = []
    memory_data = memory_data or {}
    
    # Map results by preset
    results_by_preset: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        preset = result.preset_name or "default"
        if preset not in results_by_preset:
            results_by_preset[preset] = []
        results_by_preset[preset].append(result)
    
    # Also merge coding, reasoning, creative if provided
    for label, result_list in [
        ("coding", coding_results or []),
        ("reasoning", reasoning_results or []),
        ("creative", creative_results or []),
    ]:
        for result in result_list:
            preset = result.preset_name or "default"
            if preset not in results_by_preset:
                results_by_preset[preset] = []
            results_by_preset[preset].append((result, label))
    
    for preset, preset_results in results_by_preset.items():
        # Find the primary benchmark result (not a label tuple)
        primary_results = [r for r in preset_results if isinstance(r, BenchmarkResult)]
        if not primary_results:
            continue
            
        primary = primary_results[0]  # Use first successful if available
        successful = [r for r in primary_results if isinstance(r, BenchmarkResult) and r.success]
        
        if successful:
            primary = successful[0]
        
        row = result_to_row(primary)
        row_key = (row["MODEL"], row["PRESET_NAME"], row["BENCHMARK_TYPE"])
        
        if row_key in seen_rows:
            # Already have this preset benchmark, keep first occurrence
            continue
        
        seen_rows.add(row_key)
        
        # Add memory data if available
        if memory_data:
            model_key = primary.model_name.split("@")[0] if "@" in primary.model_name else primary.model_name
            if model_key in memory_data:
                row["PEAK_MEMORY_GB"] = memory_data[model_key].get("peak_gb", "---")
                row["PEAK_MEMORY_PCT"] = memory_data[model_key].get("peak_pct", "---")
        
        # Add multi-prompt benchmark results if available
        for preset_result in preset_results:
            if isinstance(preset_result, tuple):
                result, label = preset_result
                if result.success:
                    if label == "coding":
                        row["CODING_SPEED_TOKS"] = round(result.tokens_per_second, 1)
                        row["TTFT_CODING_MS"] = round(result.time_to_first_token_ms, 0)
                    elif label == "reasoning":
                        row["REASONING_SPEED_TOKS"] = round(result.tokens_per_second, 1)
                        row["TTFT_REASONING_MS"] = round(result.time_to_first_token_ms, 0)
                    elif label == "creative":
                        row["CREATIVE_SPEED_TOKS"] = round(result.tokens_per_second, 1)
                        row["TTFT_CREATIVE_MS"] = round(result.time_to_first_token_ms, 0)
        
        rows.append(row)

    # Define CSV columns in desired order
    columns = [
        "MODEL", "BENCHMARK_TYPE", "PRESET_NAME", "PRESET_CONTEXT_LEN",
        "GEN_SPEED_TOKS", "CODING_SPEED_TOKS", "REASONING_SPEED_TOKS", 
        "CREATIVE_SPEED_TOKS", "PEAK_MEMORY_GB", "PEAK_MEMORY_PCT",
        "TTFT_GEN_MS", "TTFT_CODING_MS", "TTFT_REASONING_MS", "TTFT_CREATIVE_MS",
        "PREFILL_SPEED", "TOTAL_DURATION_S", "SUCCESS", "ERROR", "RUN_NUMBER"
    ]

    # Write CSV
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Exported {len(rows)} benchmark result(s) to {output_path}")


def benchmark_results_to_markdown(
    results: list[BenchmarkResult],
    output_path: str | None = None,
    title: str = "Benchmark Results",
    include_preset_info: bool = True,
) -> str:
    """Export benchmark results to markdown format.
    
    Args:
        results: List of benchmark results
        output_path: Optional path to write to file
        title: Title for the markdown document
        include_preset_info: Whether to include preset information
    
    Returns:
        Markdown string
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    
    # Group results by preset
    presets: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        preset = result.preset_name or "default"
        if preset not in presets:
            presets[preset] = []
        presets[preset].append(result)
    
    for preset_name, preset_results in presets.items():
        lines.append(f"## Preset: `{preset_name}`")
        lines.append("")
        
        # Show preset configuration if available
        if include_preset_info and preset_results[0].preset_config:
            config = preset_results[0].preset_config
            lines.append("### Configuration")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for key, value in config.items():
                if isinstance(value, dict):
                    value = str(value)
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        lines.append("### Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        # Calculate summary stats
        successful = [r for r in preset_results if r.success]
        total_runs = len(preset_results)
        
        lines.append(f"| Total Runs | {total_runs} |")
        lines.append(f"| Successful | {len(successful)} |")
        lines.append(f"| Failed | {total_runs - len(successful)} |")
        
        if successful:
            speeds = [r.tokens_per_second for r in successful]
            ttfts = [r.time_to_first_token_ms for r in successful]
            
            lines.append(f"| Avg Speed | {sum(speeds)/len(speeds):.1f} tok/s |")
            lines.append(f"| Min Speed | {min(speeds):.1f} tok/s |")
            lines.append(f"| Max Speed | {max(speeds):.1f} tok/s |")
            lines.append(f"| Avg TTFT | {sum(ttfts)/len(ttfts):.0f} ms |")
        
        lines.append("")
        lines.append("### Per-Run Details")
        lines.append("")
        lines.append("| Run | Success | Speed (tok/s) | TTFT (ms) | Tokens | Error |")
        lines.append("|-----|---------|---------------|-----------|--------|-------|")
        
        for result in preset_results:
            status = "✓" if result.success else "✗"
            speed = f"{result.tokens_per_second:.1f}" if result.success else "---"
            ttft = f"{result.time_to_first_token_ms:.0f}" if result.success else "---"
            tokens = str(result.generated_tokens) if result.success else "---"
            error = result.error or "-"
            
            lines.append(f"| {result.run_number} | {status} | {speed} | {ttft} | {tokens} | {error} |")
        
        lines.append("")
    
    markdown = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(markdown, encoding='utf-8')
        print(f"✓ Exported markdown to {output_path}")
    
    return markdown


def export_benchmarks_with_presets(
    results: list[BenchmarkResult],
    csv_path: str = "benchmark_results.csv",
    md_path: str | None = None,
) -> None:
    """Convenience function to export benchmarks to both CSV and markdown.
    
    Creates multiple CSV rows for same model with different presets.
    Each preset is treated as a separate benchmark entry.
    """
    # Export to CSV
    benchmark_results_to_csv(results, csv_path)
    
    # Export to markdown if requested
    if md_path:
        benchmark_results_to_markdown(results, md_path, title="Benchmark Results with Presets")
