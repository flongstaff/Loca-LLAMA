"""CLI interface for Loca-LLAMA."""

import argparse
import sys

from .hardware import APPLE_SILICON_SPECS, MacSpec, detect_mac
from .models import MODELS, LLMModel
from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS
from .analyzer import (
    analyze_model, analyze_all, max_context_for_model,
    estimate_model_size_gb, estimate_kv_cache_gb, estimate_kv_cache_raw,
    estimate_overhead_gb, recommend_models,
)


# ── ANSI helpers ─────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"


def color_rating(rating: str) -> str:
    if rating == "Excellent":
        return f"{GREEN}{BOLD}{rating}{RESET}"
    elif rating == "Good":
        return f"{GREEN}{rating}{RESET}"
    elif rating == "Tight":
        return f"{YELLOW}{rating}{RESET}"
    elif rating in ("Very Tight", "Barely fits"):
        return f"{YELLOW}{BOLD}{rating}{RESET}"
    else:
        return f"{RED}{BOLD}{rating}{RESET}"


def bar(used: float, total: float, width: int = 30) -> str:
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    empty = width - filled
    if pct <= 0.60:
        color = GREEN
    elif pct <= 0.80:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {pct:.0%}"


def format_size(gb: float) -> str:
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def format_context(ctx: int) -> str:
    if ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)


def format_tok_s(tok_s: float | None) -> str:
    if tok_s is None:
        return f"{DIM}n/a{RESET}"
    if tok_s >= 30:
        return f"{GREEN}{tok_s:.1f} tok/s{RESET}"
    elif tok_s >= 10:
        return f"{YELLOW}{tok_s:.1f} tok/s{RESET}"
    else:
        return f"{RED}{tok_s:.1f} tok/s{RESET}"


# ── Table formatting ─────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")


def print_hw_summary(mac: MacSpec) -> None:
    print(f"  {BOLD}Chip:{RESET}        {mac.chip}")
    print(f"  {BOLD}CPU:{RESET}         {mac.cpu_cores} cores")
    print(f"  {BOLD}GPU:{RESET}         {mac.gpu_cores} cores")
    print(f"  {BOLD}Neural Eng:{RESET}  {mac.neural_engine_cores} cores")
    print(f"  {BOLD}Memory:{RESET}      {mac.memory_gb} GB unified ({format_size(mac.usable_memory_gb)} usable)")
    print(f"  {BOLD}Bandwidth:{RESET}   {mac.memory_bandwidth_gbs:.0f} GB/s")
    print(f"  {BOLD}GPU TFLOPS:{RESET}  {mac.gpu_tflops}")
    print()


def print_compatibility_table(results: list, mac: MacSpec) -> None:
    # Column widths
    name_w = max(len(r.model.name) for r in results) + 1 if results else 20
    quant_w = 8
    size_w = 8
    kv_w = 8
    total_w = 8
    bar_w = 36  # visual bar
    rate_w = 12
    tok_w = 14

    header = (
        f"  {BOLD}{'Model':<{name_w}} {'Quant':<{quant_w}} {'Weights':<{size_w}} "
        f"{'KV $':<{kv_w}} {'Total':<{total_w}} {'Memory Usage':<{bar_w}} "
        f"{'Rating':<{rate_w}} {'Speed':>{tok_w}}{RESET}"
    )
    print(header)
    print(f"  {'─' * (name_w + quant_w + size_w + kv_w + total_w + bar_w + rate_w + tok_w + 6)}")

    for r in results:
        mem_bar = bar(r.total_memory_gb, r.available_memory_gb)
        rating = color_rating(r.rating)
        tok = format_tok_s(r.estimated_tok_per_sec)
        size = format_size(r.model_size_gb)
        kv = format_size(r.kv_cache_gb)
        total = format_size(r.total_memory_gb)

        print(
            f"  {r.model.name:<{name_w}} {r.quant.name:<{quant_w}} {size:<{size_w}} "
            f"{kv:<{kv_w}} {total:<{total_w}} {mem_bar} "
            f"{rating:<{rate_w}} {tok:>{tok_w}}"
        )


def print_detail(result, mac: MacSpec) -> None:
    r = result
    print(f"  {BOLD}Model:{RESET}       {r.model.name} ({r.model.params_billion}B params)")
    print(f"  {BOLD}Family:{RESET}      {r.model.family}")
    print(f"  {BOLD}License:{RESET}     {r.model.license}")
    print(f"  {BOLD}Quant:{RESET}       {r.quant.name} ({r.quant.bits_per_weight:.1f} bpw) — {r.quant.description}")
    print(f"  {BOLD}Context:{RESET}     {format_context(r.context_length)} tokens")
    print()
    print(f"  {BOLD}Memory Breakdown:{RESET}")
    print(f"    Model weights:  {format_size(r.model_size_gb):>8}")
    print(f"    KV cache:       {format_size(r.kv_cache_gb):>8}")
    print(f"    Overhead:       {format_size(r.overhead_gb):>8}")
    print(f"    {'─' * 24}")
    print(f"    {BOLD}Total:          {format_size(r.total_memory_gb):>8}{RESET}")
    print(f"    Available:      {format_size(r.available_memory_gb):>8}")
    print()
    print(f"  {BOLD}Memory:{RESET} {bar(r.total_memory_gb, r.available_memory_gb, 40)}")
    print(f"  {BOLD}Rating:{RESET} {color_rating(r.rating)}")
    if r.estimated_tok_per_sec:
        print(f"  {BOLD}Speed:{RESET}  {format_tok_s(r.estimated_tok_per_sec)} (estimated generation)")
    print()

    # Show max context
    max_ctx = max_context_for_model(mac, r.model, r.quant)
    print(f"  {BOLD}Max context for this quant:{RESET} {format_context(max_ctx)} tokens")
    print(f"  {DIM}(Model max: {format_context(r.model.max_context_length)} tokens){RESET}")


def print_quant_table() -> None:
    print(f"  {BOLD}{'Format':<10} {'Bits/Weight':>11} {'Quality':<12} Description{RESET}")
    print(f"  {'─' * 70}")
    for q in QUANT_FORMATS.values():
        print(f"  {q.name:<10} {q.bits_per_weight:>8.2f} bpw {q.quality_rating:<12} {q.description}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loca-llama",
        description="Loca-LLAMA: Check which LLMs your Mac can run locally.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── list-hw ──
    sub.add_parser("list-hw", help="List all supported Apple Silicon configs")

    # ── list-models ──
    sub.add_parser("list-models", help="List all models in the database")

    # ── list-quants ──
    sub.add_parser("list-quants", help="Explain all quantization formats")

    # ── check ──
    check = sub.add_parser("check", help="Check which models fit your Mac")
    check.add_argument("--hw", default=None, help="Hardware config (e.g. 'M4 Pro 48GB'). Auto-detected if omitted.")
    check.add_argument("--model", help="Filter to a specific model name (substring match)")
    check.add_argument("--family", help="Filter by model family (e.g. Llama, Qwen)")
    check.add_argument("--quant", nargs="+", help="Quantization formats to check (default: recommended)")
    check.add_argument("--context", type=int, help="Context length to evaluate (default: model default)")
    check.add_argument("--all", action="store_true", help="Show all combos (not just those that fit)")

    # ── detail ──
    detail = sub.add_parser("detail", help="Detailed analysis of a specific model")
    detail.add_argument("--hw", default=None, help="Hardware config (e.g. 'M4 Pro 48GB'). Auto-detected if omitted.")
    detail.add_argument("--model", required=True, help="Model name (exact or substring)")
    detail.add_argument("--quant", default="Q4_K_M", help="Quantization format (default: Q4_K_M)")
    detail.add_argument("--context", type=int, help="Context length")

    # ── max-context ──
    mc = sub.add_parser("max-context", help="Find max context length for a model")
    mc.add_argument("--hw", default=None, help="Hardware config. Auto-detected if omitted.")
    mc.add_argument("--model", required=True, help="Model name")
    mc.add_argument("--quant", default="Q4_K_M", help="Quantization format (default: Q4_K_M)")

    # ── recommend ──
    rec = sub.add_parser("recommend", help="Get recommendations for your hardware")
    rec.add_argument("--hw", default=None, help="Hardware config. Auto-detected if omitted.")
    rec.add_argument("--use-case", choices=["general", "coding", "reasoning", "small", "large-context"],
                     default="general", help="Intended use case")

    # ── calc ──
    calc = sub.add_parser("calc", help="Calculate VRAM requirements for a model")
    calc.add_argument("--model", help="Model name (substring match)")
    calc.add_argument("--quant", default="Q4_K_M", help="Quantization format (default: Q4_K_M)")
    calc.add_argument("--params", type=float, help="Parameter count in billions (custom model)")
    calc.add_argument("--bpw", type=float, help="Bits per weight (custom model)")
    calc.add_argument("--context", type=int, default=4096, help="Context length (default: 4096)")
    calc.add_argument("--layers", type=int, help="Number of layers (custom model)")
    calc.add_argument("--kv-heads", type=int, help="Number of KV attention heads (custom model)")
    calc.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    calc.add_argument("--hw", default=None, help="Hardware for fit assessment. Auto-detected if omitted.")

    # ── memory ──
    sub.add_parser("memory", help="Show current memory usage")

    # ── scan ──
    scan = sub.add_parser("scan", help="Scan for locally downloaded models")
    scan.add_argument("--dir", help="Custom directory to scan")

    # ── gpu-optimize ──
    gpu = sub.add_parser("gpu-optimize", help="Optimize GPU layer allocation and batch size")
    gpu.add_argument("--model", help="Model name (substring match)")
    gpu.add_argument("--quant", help="Quantization format (default: Q4_K_M)")
    gpu.add_argument("--context", type=int, default=4096, help="Context length (default: 4096)")
    gpu.add_argument("--compare", action="store_true", help="Compare all quantizations")

    # ── batch-optimize ──
    batch = sub.add_parser("batch-optimize", help="Optimize batch size for maximum throughput")
    batch.add_argument("--model", help="Model name (substring match)")
    batch.add_argument("--quant", help="Quantization format (default: Q4_K_M)")
    batch.add_argument("--context", type=int, default=4096, help="Context length (default: 4096)")
    batch.add_argument("--preference", choices=["high", "balanced", "low"], default="balanced",
                       help="Batch preference: high (max throughput), balanced, low (memory efficient)")
    batch.add_argument("--batch-sizes", nargs="+", help="Specific batch sizes to test")

    # ── benchmark ──
    bench = sub.add_parser("benchmark", help="Run benchmark tests")
    bench.add_argument("--model", required=True, help="Model name")
    bench.add_argument("--model-path", help="Path to GGUF file (overrides model name)")
    bench.add_argument("--quant", help="Quantization format")
    bench.add_argument("--batch-sizes", nargs="+", type=int, default=[256, 512, 1024, 2048],
                       help="Batch sizes to test")
    bench.add_argument("--context-lengths", nargs="+", type=int, default=[512, 1024, 2048],
                       help="Context lengths to test")
    bench.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 = auto)")
    bench.add_argument("--runs", type=int, default=3, help="Number of runs per config")

    # ── quality ──
    quality = sub.add_parser("quality", help="Run quality benchmark (coding tasks)")
    quality.add_argument("--model", help="Test a specific model by ID")
    quality.add_argument("--compare", choices=["pi", "claude"], help="Compare local vs cloud provider")
    quality.add_argument("--runtime", help="Runtime to use (auto-detected if omitted)")

    # ── speed ──
    speed = sub.add_parser("speed", help="Run speed benchmark")
    speed.add_argument("--model", help="Model ID (auto-detected if omitted)")
    speed.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    speed.add_argument("--prompt", choices=["default", "coding", "reasoning", "creative", "json"],
                       default="default", help="Prompt type (default: default)")
    speed.add_argument("--sweep", action="store_true", help="Benchmark all loaded models")
    speed.add_argument("--runtime", help="Runtime to use (auto-detected if omitted)")

    # ── monitor ──
    monitor = sub.add_parser("monitor", help="Proxy monitor for OpenCode sessions")
    monitor.add_argument("--listen", type=int, default=1240, help="Listen port (default: 1240)")
    monitor.add_argument("--target", type=int, default=1234, help="Target LLM server port (default: 1234)")

    # ── results ──
    results = sub.add_parser("results", help="View saved benchmark results")
    results.add_argument("--model", help="Filter by model name")
    results.add_argument("--type", choices=["speed", "quality", "monitor", "eval"], help="Filter by type")
    results.add_argument("--compare", action="store_true", help="Side-by-side latest per model")
    results.add_argument("--limit", type=int, default=30, help="Max results to show (default: 30)")
    results.add_argument("--export", choices=["csv", "html", "md"], help="Export format")
    results.add_argument("--output", help="Output file path for export")

    # ── throughput ──
    tp = sub.add_parser("throughput", help="Concurrent throughput test")
    tp.add_argument("--model", help="Model ID (auto-detected if omitted)")
    tp.add_argument("--concurrency", type=int, default=4, help="Concurrent requests (default: 4)")
    tp.add_argument("--requests", type=int, default=8, help="Total requests (default: 8)")
    tp.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request (default: 100)")
    tp.add_argument("--runtime", help="Runtime to use (auto-detected if omitted)")

    # ── fetch-config ──
    fc = sub.add_parser("fetch-config", help="Fetch model config from HuggingFace")
    fc.add_argument("repo", help="HuggingFace repo ID (e.g. Qwen/Qwen2.5-72B)")
    fc.add_argument("--calc", action="store_true", help="Auto-run VRAM calc with fetched config")
    fc.add_argument("--quant", default="Q4_K_M", help="Quantization for calc (default: Q4_K_M)")
    fc.add_argument("--context", type=int, default=4096, help="Context for calc (default: 4096)")

    # ── eval ──
    ev = sub.add_parser("eval", help="Run standard LLM evaluation benchmarks")
    ev.add_argument("--bench", nargs="+",
                    choices=["gsm8k", "arc", "hellaswag", "ifeval", "humaneval", "mmlu", "all"],
                    default=["all"], help="Benchmarks to run (default: all)")
    ev.add_argument("--model", help="Model ID (auto-detected if omitted)")
    ev.add_argument("--samples", type=int, help="Max samples per benchmark (for quick runs)")
    ev.add_argument("--runtime", help="Runtime to use (auto-detected if omitted)")

    return parser


def resolve_hw(hw_name: str) -> MacSpec | None:
    # Exact match
    if hw_name in APPLE_SILICON_SPECS:
        return APPLE_SILICON_SPECS[hw_name]
    # Case-insensitive search
    for key, spec in APPLE_SILICON_SPECS.items():
        if hw_name.lower() == key.lower():
            return spec
    # Substring match
    matches = [
        (key, spec) for key, spec in APPLE_SILICON_SPECS.items()
        if hw_name.lower() in key.lower()
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        print(f"{RED}Ambiguous hardware name '{hw_name}'. Did you mean:{RESET}")
        for key, _ in matches:
            print(f"  - {key}")
        return None
    print(f"{RED}Unknown hardware: '{hw_name}'. Use 'list-hw' to see options.{RESET}")
    return None


def resolve_hw_or_detect(hw_name: str | None) -> MacSpec | None:
    """Resolve hardware by name, or auto-detect on the current Mac."""
    if hw_name:
        return resolve_hw(hw_name)
    result = detect_mac()
    if result is None:
        print(f"{RED}Could not detect hardware. Use --hw to specify manually.{RESET}")
        print(f"{DIM}Run 'loca-llama list-hw' for options.{RESET}")
        return None
    key, mac = result
    print(f"{GREEN}Detected: {key}{RESET}", file=sys.stderr)
    return mac


def resolve_model(name: str) -> LLMModel | None:
    # Exact match
    for m in MODELS:
        if m.name.lower() == name.lower():
            return m
    # Substring
    matches = [m for m in MODELS if name.lower() in m.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"{RED}Ambiguous model name '{name}'. Matches:{RESET}")
        for m in matches:
            print(f"  - {m.name}")
        return None
    print(f"{RED}Unknown model: '{name}'. Use 'list-models' to see options.{RESET}")
    return None


def filter_models(name: str | None = None, family: str | None = None) -> list[LLMModel]:
    result = MODELS
    if name:
        result = [m for m in result if name.lower() in m.name.lower()]
    if family:
        result = [m for m in result if family.lower() == m.family.lower()]
    return result


def cmd_list_hw() -> None:
    print_header("Supported Apple Silicon Configurations")
    print(f"  {BOLD}{'Name':<22} {'CPU':>4} {'GPU':>4} {'NE':>4} {'RAM':>6} {'BW (GB/s)':>10} {'TFLOPS':>8}{RESET}")
    print(f"  {'─' * 60}")
    for name, spec in APPLE_SILICON_SPECS.items():
        print(
            f"  {name:<22} {spec.cpu_cores:>4} {spec.gpu_cores:>4} "
            f"{spec.neural_engine_cores:>4} {spec.memory_gb:>4} GB {spec.memory_bandwidth_gbs:>8.0f}  "
            f"{spec.gpu_tflops:>7.1f}"
        )


def cmd_list_models() -> None:
    print_header("LLM Model Database")
    print(f"  {BOLD}{'Name':<40} {'Params':>8} {'Context':>8} {'Max Ctx':>8} {'License':<16}{RESET}")
    print(f"  {'─' * 82}")
    for m in MODELS:
        print(
            f"  {m.name:<40} {m.params_billion:>6.1f}B "
            f"{format_context(m.default_context_length):>8} "
            f"{format_context(m.max_context_length):>8} {m.license:<16}"
        )


def cmd_list_quants() -> None:
    print_header("GGUF Quantization Formats")
    print_quant_table()


def cmd_check(args) -> None:
    mac = resolve_hw_or_detect(args.hw)
    if not mac:
        sys.exit(1)

    models = filter_models(args.model, args.family)
    if not models:
        print(f"{RED}No models match your filter.{RESET}")
        sys.exit(1)

    quant_names = args.quant or RECOMMENDED_FORMATS
    only_fits = not args.all

    results = analyze_all(mac, models, quant_names, args.context, only_fits)

    # Sort: fits first, then by total memory
    results.sort(key=lambda r: (not r.fits_in_memory, r.total_memory_gb))

    hw_label = args.hw or f"{mac.chip} {mac.memory_gb}GB"
    print_header(f"LLM Compatibility — {hw_label}")
    print_hw_summary(mac)

    if not results:
        if only_fits:
            print(f"  {RED}No models fit with the selected filters. Try --all to see everything.{RESET}")
        else:
            print(f"  {RED}No models match your filter.{RESET}")
        return

    print_compatibility_table(results, mac)
    fits_count = sum(1 for r in results if r.fits_in_memory)
    print(f"\n  {GREEN}{fits_count}{RESET} configurations fit / {len(results)} shown")


def cmd_detail(args) -> None:
    mac = resolve_hw_or_detect(args.hw)
    if not mac:
        sys.exit(1)
    model = resolve_model(args.model)
    if not model:
        sys.exit(1)
    if args.quant not in QUANT_FORMATS:
        print(f"{RED}Unknown quant format: '{args.quant}'. Use 'list-quants' to see options.{RESET}")
        sys.exit(1)

    quant = QUANT_FORMATS[args.quant]
    result = analyze_model(mac, model, quant, args.context)

    print_header(f"Detailed Analysis — {model.name} @ {quant.name}")
    print_hw_summary(mac)
    print_detail(result, mac)


def cmd_max_context(args) -> None:
    mac = resolve_hw_or_detect(args.hw)
    if not mac:
        sys.exit(1)
    model = resolve_model(args.model)
    if not model:
        sys.exit(1)
    if args.quant not in QUANT_FORMATS:
        print(f"{RED}Unknown quant format: '{args.quant}'{RESET}")
        sys.exit(1)

    quant = QUANT_FORMATS[args.quant]
    max_ctx = max_context_for_model(mac, model, quant)

    print_header(f"Max Context — {model.name} @ {quant.name}")
    print_hw_summary(mac)

    if max_ctx == 0:
        print(f"  {RED}Model doesn't fit in memory at {quant.name} quantization.{RESET}")
    else:
        model_arch_max = model.max_context_length
        effective = min(max_ctx, model_arch_max)
        print(f"  {BOLD}Max context that fits:{RESET}  {format_context(effective)} tokens")
        if max_ctx > model_arch_max:
            print(f"  {DIM}(Memory allows {format_context(max_ctx)}, but model architecture caps at {format_context(model_arch_max)}){RESET}")
        print()

        # Show a context scaling table
        print(f"  {BOLD}{'Context':>10} {'KV Cache':>10} {'Total Mem':>10} {'Fits?':>8}{RESET}")
        print(f"  {'─' * 42}")
        from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb, estimate_overhead_gb
        model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
        overhead = estimate_overhead_gb(model_size)
        for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            if ctx > model_arch_max:
                break
            kv = estimate_kv_cache_gb(model, ctx)
            total = model_size + kv + overhead
            fits = total <= mac.usable_memory_gb
            mark = f"{GREEN}Yes{RESET}" if fits else f"{RED}No{RESET}"
            print(f"  {format_context(ctx):>10} {format_size(kv):>10} {format_size(total):>10} {mark:>8}")


def cmd_recommend(args) -> None:
    mac = resolve_hw_or_detect(args.hw)
    if not mac:
        sys.exit(1)

    use_case = args.use_case
    hw_label = args.hw or f"{mac.chip} {mac.memory_gb}GB"
    print_header(f"Recommendations — {hw_label} ({use_case})")
    print_hw_summary(mac)

    top = recommend_models(mac, use_case=use_case, top_n=8)

    if not top:
        print(f"  {RED}No good recommendations found for this use case.{RESET}")
        return
    print(f"  {BOLD}Top picks (best quality quant that fits comfortably):{RESET}\n")

    # Column widths
    name_w = max(len(r.model.name) for r in top) + 1
    quant_w = 8
    rate_w = 14
    bar_w = 36
    mem_w = 20
    tok_w = 14
    ctx_w = 8

    header = (
        f"  {BOLD}{'#':>2}  {'Model':<{name_w}} {'Quant':<{quant_w}} "
        f"{'Rating':<{rate_w}} {'Memory Usage':<{bar_w}} "
        f"{'Mem':>{mem_w}} {'Speed':>{tok_w}} {'Max Ctx':>{ctx_w}}{RESET}"
    )
    print(header)
    sep_len = 2 + 2 + name_w + quant_w + rate_w + bar_w + mem_w + tok_w + ctx_w + 6
    print(f"  {'─' * sep_len}")

    for i, r in enumerate(top, 1):
        mem_bar = bar(r.total_memory_gb, r.available_memory_gb)
        rating = color_rating(r.rating)
        tok = format_tok_s(r.estimated_tok_per_sec)
        mem_frac = f"{format_size(r.total_memory_gb)}/{format_size(r.available_memory_gb)}"
        max_ctx = format_context(max_context_for_model(mac, r.model, r.quant))

        print(
            f"  {MAGENTA}{i:>2}{RESET}  {BOLD}{r.model.name:<{name_w}}{RESET} "
            f"{CYAN}{r.quant.name:<{quant_w}}{RESET} "
            f"{rating:<{rate_w}} {mem_bar} "
            f"{mem_frac:>{mem_w}} {tok:>{tok_w}} {max_ctx:>{ctx_w}}"
        )

    print(f"\n  {GREEN}{len(top)}{RESET} recommendations shown")


def cmd_calc(args) -> None:
    has_model = args.model is not None
    has_custom = args.params is not None and args.bpw is not None

    if not has_model and not has_custom:
        print(f"{RED}Provide --model or both --params and --bpw.{RESET}")
        sys.exit(1)

    if has_model:
        model = resolve_model(args.model)
        if not model:
            sys.exit(1)
        if args.quant not in QUANT_FORMATS:
            print(f"{RED}Unknown quant format: '{args.quant}'. Use 'list-quants' to see options.{RESET}")
            sys.exit(1)
        quant = QUANT_FORMATS[args.quant]
        if has_custom:
            print(f"{YELLOW}Using --model; ignoring --params/--bpw.{RESET}", file=sys.stderr)

        label = f"{model.name} @ {quant.name}"
        params_b = model.params_billion
        bpw = quant.bits_per_weight
        model_size = estimate_model_size_gb(params_b, bpw)
        kv_cache = estimate_kv_cache_gb(model, args.context)
    else:
        label = f"Custom {args.params:.1f}B @ {args.bpw:.1f} bpw"
        params_b = args.params
        bpw = args.bpw
        model_size = estimate_model_size_gb(params_b, bpw)
        layers = args.layers or 32
        kv_heads = args.kv_heads or 8
        kv_cache = estimate_kv_cache_raw(layers, kv_heads, args.head_dim, args.context)

    overhead = estimate_overhead_gb(model_size)
    total = model_size + kv_cache + overhead

    print_header(f"VRAM Calculation — {label}")
    print(f"  {BOLD}Parameters:{RESET}    {params_b:.1f}B")
    print(f"  {BOLD}Bits/weight:{RESET}   {bpw:.1f} bpw")
    print(f"  {BOLD}Context:{RESET}       {format_context(args.context)} tokens")
    print()
    print(f"  {BOLD}Memory Breakdown:{RESET}")
    print(f"    Model weights:  {format_size(model_size):>8}")
    print(f"    KV cache:       {format_size(kv_cache):>8}")
    print(f"    Overhead:       {format_size(overhead):>8}")
    print(f"    {'─' * 24}")
    print(f"    {BOLD}Total:          {format_size(total):>8}{RESET}")

    # Fit assessment if hardware is available
    mac = resolve_hw_or_detect(args.hw)
    if mac:
        avail = mac.usable_memory_gb
        print()
        print(f"  {BOLD}Hardware:{RESET}      {mac.chip} {mac.memory_gb}GB ({format_size(avail)} usable)")
        print(f"  {BOLD}Usage:{RESET}         {bar(total, avail)}")
        if total <= avail:
            headroom = avail - total
            print(f"  {BOLD}Headroom:{RESET}      {format_size(headroom)} free")
        else:
            over = total - avail
            print(f"  {RED}{BOLD}Over budget:{RESET}   {format_size(over)} over capacity")


def cmd_memory(_args) -> None:
    if sys.platform != "darwin":
        print(f"{RED}Memory monitoring requires macOS.{RESET}")
        sys.exit(1)

    from .memory_monitor import get_memory_sample

    sample = get_memory_sample()

    print_header("Memory Status")
    print(f"  {BOLD}Used:{RESET}      {sample.used_gb:>6.1f} GB")
    print(f"  {BOLD}Free:{RESET}      {sample.free_gb:>6.1f} GB")
    print(f"  {BOLD}Total:{RESET}     {sample.total_gb:>6.1f} GB")
    print(f"  {BOLD}Usage:{RESET}     {bar(sample.used_gb, sample.total_gb)}")

    # Color pressure level
    p = sample.pressure.lower()
    if p == "critical":
        pcolor = RED
    elif p == "warn":
        pcolor = YELLOW
    else:
        pcolor = GREEN
    print(f"  {BOLD}Pressure:{RESET}  {pcolor}{sample.pressure}{RESET}")


def cmd_scan(args) -> None:
    from .scanner import scan_all, scan_custom_dir

    if args.dir:
        models = scan_custom_dir(args.dir)
    else:
        models = scan_all()

    if not models:
        print(f"{YELLOW}No models found.{RESET}")
        return

    print_header("Local Models")
    print(f"  Found {GREEN}{len(models)}{RESET} model(s):\n")

    source_colors = {
        "lm-studio": MAGENTA, "llama.cpp": CYAN,
        "huggingface": YELLOW, "mlx-community": GREEN,
    }

    for i, m in enumerate(models, 1):
        sc = source_colors.get(m.source, WHITE)
        quant_str = f" [{m.quant}]" if m.quant else ""
        print(
            f"  {i:>3}) {BOLD}{m.name}{RESET}{quant_str}\n"
            f"       {format_size(m.size_gb):>8}  {sc}{m.source:<14}{RESET}  {m.format:<10}  {DIM}{m.path}{RESET}"
        )


def cmd_gpu_optimize(args) -> None:
    from .hardware import detect_mac
    from .gpu_optimizer import optimize_for_hardware, compare_quantizations

    # Detect hardware
    detected = detect_mac()
    if detected is None:
        print(f"{RED}Could not detect Mac hardware.{RESET}")
        sys.exit(1)
    mac_key, mac = detected

    # Find model
    model = None
    if args.model:
        for m in MODELS:
            if args.model.lower() in m.name.lower():
                model = m
                break
        if not model:
            print(f"{RED}Model '{args.model}' not found in database.{RESET}")
            print("Available models:")
            for m in MODELS:
                print(f"  - {m.name}")
            sys.exit(1)
    else:
        print(f"{YELLOW}No model specified. Use --model to select one.{RESET}")
        sys.exit(1)

    # Find quantization
    quant = None
    if args.quant:
        if args.quant in QUANT_FORMATS:
            quant = QUANT_FORMATS[args.quant]
        else:
            print(f"{RED}Quantization '{args.quant}' not found.{RESET}")
            print(f"Available: {', '.join(QUANT_FORMATS.keys())}")
            sys.exit(1)
    else:
        # Default to Q4_K_M
        quant = QUANT_FORMATS["Q4_K_M"]

    print_header(f"GPU Optimization — {model.name} @ {quant.name}")
    print_hw_summary(mac)
    print(f"  {BOLD}Model:{RESET}      {model.name}")
    print(f"  {BOLD}Parameters:{RESET}  {model.params_billion:.1f}B")
    print(f"  {BOLD}Layers:{RESET}      {model.num_layers}")
    print(f"  {BOLD}Context:{RESET}     {format_context(args.context)}")
    print()

    if args.compare:
        # Compare all quantizations
        print(f"  {BOLD}Comparing Quantizations:{RESET}\n")
        results = compare_quantizations(mac, model, args.context)

        print(f"  {'Quant':<10} {'Size':>8} {'GPU Layers':>12} {'Offload':>10} {'Tok/s':>10} {'Recommendation'}")
        print(f"  {'-' * 70}")

        for r in results:
            rec = r.recommendation
            rec_color = GREEN if r.offload_percentage == 0 else YELLOW
            print(
                f"  {r.model_size_gb:>7.1f}GB  {r.gpu_layers:>12}/{model.num_layers}  "
                f"{RED if r.offload_percentage > 50 else GREEN}{r.offload_percentage:>9.0f}%{RESET}  "
                f"{format_tok_s(r.estimated_tokens_per_second):>10}  {rec_color}{rec}{RESET}"
            )

        print()
        best = results[0]
        print(f"  {BOLD}Best Option:{RESET} {best.model_size_gb:.1f}GB @ {best.offload_percentage:.0f}% offload")
        print(f"  {BOLD}Recommended Settings:{RESET}")
        print(f"    GPU Layers: {best.gpu_layers}")
        print(f"    Batch Size: {best.batch_size}")
        print(f"    Context: {best.context_length}")
    else:
        # Single optimization
        result = optimize_for_hardware(mac, model, quant, args.context)

        print(f"  {BOLD}Optimal Settings:{RESET}\n")
        print(f"    {BOLD}GPU Layers:{RESET}   {result.gpu_layers}/{model.num_layers}")
        print(f"    {BOLD}Batch Size:{RESET}   {result.batch_size}")
        print(f"    {BOLD}Context:{RESET}      {result.context_length}")
        print()

        print(f"  {BOLD}Memory Usage:{RESET}\n")
        print(f"    Model weights:  {format_size(result.model_size_gb)}")
        print(f"    KV cache:       {format_size(result.kv_cache_gb)}")
        print(f"    Total:          {format_size(result.total_memory_gb)}")
        print(f"    GPU memory:     {format_size(result.gpu_memory_gb)}")
        print(f"    System memory:  {format_size(result.system_memory_gb)}")
        print()

        print(f"  {BOLD}Performance:{RESET}\n")
        print(f"    Offload:        {result.offload_percentage:.0f}% on CPU")
        print(f"    Estimated:      {format_tok_s(result.estimated_tokens_per_second)}")
        print(f"    Latency:        {result.estimated_latency_ms:.0f}ms/token")
        print()

        print(f"  {BOLD}Recommendation:{RESET} {BOLD}{result.recommendation}{RESET}\n")
        for note in result.notes:
            print(f"    • {note}")


def cmd_batch_optimize(args) -> None:
    from .hardware import detect_mac
    from .batch_optimizer import optimize_for_batch_preference, compare_batch_sizes

    # Detect hardware
    detected = detect_mac()
    if detected is None:
        print(f"{RED}Could not detect Mac hardware.{RESET}")
        sys.exit(1)
    mac_key, mac = detected

    # Find model
    model = None
    if args.model:
        for m in MODELS:
            if args.model.lower() in m.name.lower():
                model = m
                break
        if not model:
            print(f"{RED}Model '{args.model}' not found in database.{RESET}")
            sys.exit(1)
    else:
        print(f"{YELLOW}No model specified. Use --model to select one.{RESET}")
        sys.exit(1)

    # Find quantization
    quant = QUANT_FORMATS.get(args.quant, QUANT_FORMATS["Q4_K_M"]) if args.quant else QUANT_FORMATS["Q4_K_M"]

    print_header(f"Batch Optimization — {model.name} @ {quant.name}")
    print_hw_summary(mac)
    print(f"  {BOLD}Model:{RESET}      {model.name}")
    print(f"  {BOLD}Parameters:{RESET}  {model.params_billion:.1f}B")
    print(f"  {BOLD}Layers:{RESET}      {model.num_layers}")
    print(f"  {BOLD}Context:{RESET}     {format_context(args.context)}")
    print(f"  {BOLD}Preference:{RESET}  {args.preference.upper()}")
    print()

    if args.batch_sizes:
        # Test specific batch sizes
        results = compare_batch_sizes(
            mac, model, quant, args.context,
            batch_sizes=[int(b) for b in args.batch_sizes]
        )

        print(f"  {'Batch':>8} {'Throughput':>12} {'Memory':>10} {'Efficiency':>12} {'Recommendation'}")
        print(f"  {'-' * 60}")

        for r in results:
            eff_color = GREEN if r.memory_efficiency > 0.7 else YELLOW
            print(
                f"  {r.batch_size:>8}  {r.estimated_throughput_tok_s:>11.1f} tok/s  "
                f"{format_size(r.total_memory_gb):>8}  {eff_color}{r.memory_efficiency:>11.0%}{RESET}  "
                f"{r.recommendation}"
            )

        print()
        best = results[0]
        print(f"  {BOLD}Best Batch Size:{RESET} {best.batch_size} @ {best.estimated_throughput_tok_s:.1f} tok/s")
    else:
        # Auto-optimize
        result = optimize_for_batch_preference(
            mac, model, quant, args.preference, args.context
        )

        print(f"  {BOLD}Optimal Settings:{RESET}\n")
        print(f"    {BOLD}Batch Size:{RESET}    {result.batch_size}")
        print(f"    {BOLD}Context:{RESET}       {result.context_length}")
        print(f"    {BOLD}GPU Layers:{RESET}    {result.gpu_layers}")
        print(f"    {BOLD}UBatch:{RESET}        {result.ubatch_size}")
        print()

        print(f"  {BOLD}Memory Usage:{RESET}\n")
        print(f"    Total:        {format_size(result.total_memory_gb)}")
        print(f"    GPU:          {format_size(result.gpu_memory_gb)}")
        print(f"    System:       {format_size(result.system_memory_gb)}")
        print(f"    Headroom:     {format_size(result.headroom_gb)}")
        print()

        print(f"  {BOLD}Performance:{RESET}\n")
        print(f"    Throughput:   {result.estimated_throughput_tok_s:.1f} tok/s")
        print(f"    Latency:      {result.estimated_latency_ms:.0f}ms/token")
        print(f"    Efficiency:   {result.memory_efficiency:.0%}")
        print()

        print(f"  {BOLD}Recommendation:{RESET} {BOLD}{result.recommendation}{RESET}\n")
        for note in result.notes:
            print(f"    • {note}")


def cmd_benchmark(args) -> None:
    from .hardware import detect_mac
    from .benchmark_suite import run_full_benchmark_suite, BenchmarkConfig

    # Detect hardware
    detected = detect_mac()
    if detected is None:
        print(f"{RED}Could not detect Mac hardware.{RESET}")
        sys.exit(1)
    mac_key, mac = detected

    # Find model path
    model_path = None
    if args.model_path:
        model_path = args.model_path
    elif args.model:
        # Try to find model in common locations
        from pathlib import Path
        model_name = args.model.lower().replace(" ", "_").replace("-", "_")
        lmstudio_models = Path.home() / ".lmstudio" / "models"
        common_paths = [
            str(lmstudio_models / "*" / f"{model_name}*.gguf"),
        ]

        import glob
        for pattern in common_paths:
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                break

        if not model_path:
            print(f"{RED}Model not found. Use --model-path to specify GGUF file.{RESET}")
            sys.exit(1)
    else:
        print(f"{RED}Please specify --model or --model-path.{RESET}")
        sys.exit(1)

    print_header(f"Benchmark — {model_path.split('/')[-1]}")
    print_hw_summary(mac)
    print(f"  {BOLD}Model:{RESET}    {model_path.split('/')[-1]}")
    print(f"  {BOLD}GPU Layers:{RESET} {args.gpu_layers}")
    print(f"  {BOLD}Batch Sizes:{RESET} {args.batch_sizes}")
    print(f"  {BOLD}Contexts:{RESET}  {args.context_lengths}")
    print(f"  {BOLD}Runs:{RESET}       {args.runs}")
    print()

    print(f"{BOLD}Running benchmark suite...{RESET}\n")

    try:
        results = run_full_benchmark_suite(
            model_path,
            gpu_layers=args.gpu_layers,
            batch_sizes=args.batch_sizes,
            context_lengths=args.context_lengths,
            num_runs=args.runs,
        )

        # Print batch size sweep results
        print(f"{BOLD}Batch Size Sweep Results:{RESET}\n")
        batch_sweep = results[0]
        print(f"  {'Batch':>8} {'TTFT':>10} {'Throughput':>12} {'Success'}")
        print(f"  {'-' * 50}")

        for r in batch_sweep.results:
            status = "✅" if r.success else "❌"
            ttft = f"{r.time_to_first_token_ms:.0f}ms" if r.success else "N/A"
            tps = f"{r.tokens_per_second:.1f}" if r.success else "N/A"
            print(f"  {r.batch_size:>8}  {ttft:>10}  {tps:>12}  {status}")

        print()

        # Print context length sweep results
        print(f"{BOLD}Context Length Sweep Results:{RESET}\n")
        context_sweep = results[1]
        print(f"  {'Context':>10} {'TTFT':>10} {'Throughput':>12} {'Success'}")
        print(f"  {'-' * 50}")

        for r in context_sweep.results:
            status = "✅" if r.success else "❌"
            ttft = f"{r.time_to_first_token_ms:.0f}ms" if r.success else "N/A"
            tps = f"{r.tokens_per_second:.1f}" if r.success else "N/A"
            print(f"  {r.context_length:>10}  {ttft:>10}  {tps:>12}  {status}")

        print()

        # Summary
        if batch_sweep.best_result:
            print(f"{BOLD}Best Configuration:{RESET}")
            print(f"  Batch Size: {batch_sweep.best_result.batch_size}")
            print(f"  Throughput: {batch_sweep.best_result.tokens_per_second:.1f} tok/s")
            print(f"  TTFT: {batch_sweep.best_result.time_to_first_token_ms:.0f}ms")

    except Exception as e:
        print(f"{RED}Benchmark failed: {e}{RESET}")


def _find_runtime(preferred: str | None = None) -> "RuntimeInfo | None":
    """Auto-detect a running LLM runtime, or use the one specified."""
    from .benchmark import detect_all_runtimes, RuntimeInfo

    runtimes = detect_all_runtimes()
    if not runtimes:
        print(f"{RED}No LLM runtime detected. Start LM Studio, oMLX, or llama.cpp server.{RESET}")
        return None

    if preferred:
        for rt in runtimes:
            if preferred.lower() in rt.name.lower():
                return rt
        print(f"{YELLOW}Runtime '{preferred}' not found. Available: {', '.join(r.name for r in runtimes)}{RESET}")

    # Return first detected
    rt = runtimes[0]
    print(f"{GREEN}Using runtime: {rt.name} ({rt.url}){RESET}", file=sys.stderr)
    return rt


def cmd_quality(args) -> None:
    from .quality_bench import (
        run_quality_benchmark, run_quality_comparison,
        get_models_from_api, print_quality_summary,
        results_to_record,
    )
    from .benchmark_results import save_result

    rt = _find_runtime(getattr(args, "runtime", None))
    if not rt:
        sys.exit(1)

    if args.model:
        models = [args.model]
    else:
        models = get_models_from_api(rt.url, rt.api_key)
        if not models:
            models = rt.models
        if not models:
            print(f"{RED}No models found on {rt.name}.{RESET}")
            sys.exit(1)
        print(f"Models found: {len(models)}")
        for m in models:
            print(f"  - {m}")

    if args.compare:
        if len(models) != 1:
            if len(models) > 1:
                print(f"{YELLOW}--compare requires a single model. Use --model to pick one.{RESET}")
                sys.exit(1)
            sys.exit(1)

        local_results, cloud_results = run_quality_comparison(
            rt.url, models[0], args.compare, local_api_key=rt.api_key,
        )

        print("\n--- LOCAL RESULTS ---")
        print_quality_summary(local_results)
        print("\n--- CLOUD RESULTS ---")
        print_quality_summary(cloud_results)

        record = results_to_record(
            local_results, rt.name,
            cloud_provider=args.compare, cloud_results=cloud_results,
        )
        path = save_result(record)
        print(f"\nResults saved to {path}")
    else:
        results = run_quality_benchmark(rt.url, models, api_key=rt.api_key, runtime_name=rt.name)
        print_quality_summary(results)

        # Save each model's results
        for model in set(r.model for r in results):
            model_results = [r for r in results if r.model == model]
            record = results_to_record(model_results, rt.name)
            path = save_result(record)
            print(f"Results saved to {path}")


def cmd_speed(args) -> None:
    from .benchmark import (
        run_benchmark_suite, run_benchmark_sweep,
        aggregate_results, BENCH_PROMPTS,
    )
    from .benchmark_results import BenchmarkRecord, save_result

    rt = _find_runtime(getattr(args, "runtime", None))
    if not rt:
        sys.exit(1)

    if args.model:
        model_ids = [args.model]
    elif args.sweep:
        model_ids = rt.models
        if not model_ids:
            print(f"{RED}No models found on {rt.name}.{RESET}")
            sys.exit(1)
    else:
        model_ids = rt.models[:1]
        if not model_ids:
            print(f"{RED}No models found on {rt.name}.{RESET}")
            sys.exit(1)

    print_header("Speed Benchmark")
    print(f"  {BOLD}Runtime:{RESET}  {rt.name} ({rt.url})")
    print(f"  {BOLD}Models:{RESET}   {', '.join(model_ids)}")
    print(f"  {BOLD}Runs:{RESET}     {args.runs}")
    print(f"  {BOLD}Prompt:{RESET}   {args.prompt}")
    print()

    def progress(run: int, total: int) -> None:
        print(f"    Run {run}/{total}...", end=" ", flush=True)

    if args.sweep and len(model_ids) > 1:
        sweep_results = run_benchmark_sweep(
            rt, model_ids, args.prompt, args.runs,
            run_callback=progress,
        )
        for entry in sweep_results:
            model_id = entry["model_id"]
            agg = entry["aggregate"]
            if agg.get("success"):
                print(f"\n  {BOLD}{model_id}{RESET}")
                print(f"    {agg['avg_tok_per_sec']:.1f} tok/s (median: {agg['median_tok_per_sec']:.1f})")
                print(f"    TTFT: {agg['avg_ttft_ms']:.0f}ms")

                record = BenchmarkRecord(
                    type="speed", model=model_id, runtime=rt.name,
                    tokens_per_second=agg["avg_tok_per_sec"],
                    ttft_ms=agg["avg_ttft_ms"],
                    total_time_ms=agg["avg_total_ms"],
                    generated_tokens=agg["total_tokens_generated"],
                    extra={"median_tps": agg["median_tok_per_sec"], "runs": agg["runs"]},
                )
                save_result(record)
            else:
                print(f"\n  {RED}{model_id}: benchmark failed{RESET}")
    else:
        for model_id in model_ids:
            results = run_benchmark_suite(
                rt, model_id, args.prompt, args.runs,
                progress_callback=progress,
            )
            agg = aggregate_results(results)

            if agg.get("success"):
                print(f"\n  {BOLD}Results for {model_id}:{RESET}")
                print(f"    Avg: {agg['avg_tok_per_sec']:.1f} tok/s")
                print(f"    Median: {agg['median_tok_per_sec']:.1f} tok/s")
                print(f"    P95: {agg['p95_tok_per_sec']:.1f} tok/s")
                print(f"    Min/Max: {agg['min_tok_per_sec']:.1f} / {agg['max_tok_per_sec']:.1f}")
                stddev = agg.get('stddev_tok_per_sec', 0)
                if stddev:
                    print(f"    Std Dev: {stddev:.1f}")
                print(f"    Prefill: {agg['avg_prefill_tok_per_sec']:.1f} tok/s")
                print(f"    TTFT: {agg['avg_ttft_ms']:.0f}ms")
                print(f"    Runs: {agg['runs']}")

                record = BenchmarkRecord(
                    type="speed", model=model_id, runtime=rt.name,
                    tokens_per_second=agg["avg_tok_per_sec"],
                    ttft_ms=agg["avg_ttft_ms"],
                    total_time_ms=agg["avg_total_ms"],
                    generated_tokens=agg["total_tokens_generated"],
                    extra={
                        "median_tps": agg["median_tok_per_sec"],
                        "p95_tps": agg["p95_tok_per_sec"],
                        "stddev_tps": agg.get("stddev_tok_per_sec", 0),
                        "prefill_tps": agg["avg_prefill_tok_per_sec"],
                        "runs": agg["runs"],
                    },
                )
                path = save_result(record)
                print(f"    Saved to {path}")
            else:
                print(f"\n  {RED}Benchmark failed for {model_id}{RESET}")
                for r in results:
                    if r.error:
                        print(f"    {r.error}")
                        break


def cmd_monitor(args) -> None:
    from .proxy_monitor import run_proxy

    run_proxy(listen_port=args.listen, target_port=args.target)


def cmd_results(args) -> None:
    from .benchmark_results import load_results, print_results_table, print_comparison_table

    records = load_results(
        type_filter=getattr(args, "type", None),
        model_filter=args.model,
        limit=args.limit,
    )

    export_fmt = getattr(args, "export", None)
    if export_fmt:
        _export_results(records, export_fmt, getattr(args, "output", None))
        return

    if args.compare:
        print_comparison_table(records)
    else:
        print_results_table(records)


def _export_results(records: list, fmt: str, output_path: str | None) -> None:
    """Export results to CSV, HTML, or Markdown."""
    from .benchmark_results import BenchmarkRecord, RESULTS_DIR

    if not records:
        print(f"{RED}No results to export.{RESET}")
        return

    if fmt == "csv":
        import csv as csv_mod
        path = output_path or str(RESULTS_DIR / "results_export.csv")
        with open(path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=[
                "type", "model", "runtime", "timestamp", "tokens_per_second",
                "ttft_ms", "total_time_ms", "hardware",
            ])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "type": r.type, "model": r.model, "runtime": r.runtime,
                    "timestamp": r.timestamp, "tokens_per_second": r.tokens_per_second,
                    "ttft_ms": r.ttft_ms, "total_time_ms": r.total_time_ms,
                    "hardware": r.hardware,
                })
        print(f"Exported {len(records)} results to {path}")

    elif fmt == "html":
        from .benchmark_report import generate_html_report
        import time as time_mod
        # Build a results dict compatible with generate_html_report
        runs = []
        for r in records:
            runs.append({
                "run_number": len(runs) + 1,
                "success": r.tokens_per_second > 0 or r.type == "quality",
                "tokens_per_second": r.tokens_per_second,
                "prompt_tokens_per_second": 0,
                "time_to_first_token_ms": r.ttft_ms,
                "total_time_ms": r.total_time_ms,
                "generated_tokens": r.generated_tokens,
            })
        html_str = generate_html_report(
            {"runs": runs, "aggregate": {}},
            metadata={
                "title": "Loca-LLAMA Results Export",
                "model": records[0].model if records else "—",
                "runtime": records[0].runtime if records else "—",
                "timestamp": time_mod.strftime("%Y-%m-%d %H:%M"),
            },
        )
        path = output_path or str(RESULTS_DIR / "results_export.html")
        from pathlib import Path
        Path(path).write_text(html_str)
        print(f"Exported HTML report to {path}")

    elif fmt == "md":
        lines = ["# Loca-LLAMA Results", ""]
        lines.append(f"| Type | Model | TPS | TTFT | Runtime |")
        lines.append("|------|-------|-----|------|---------|")
        for r in records:
            tps = f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "—"
            ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "—"
            lines.append(f"| {r.type} | {r.model} | {tps} | {ttft} | {r.runtime} |")
        path = output_path or str(RESULTS_DIR / "results_export.md")
        from pathlib import Path
        Path(path).write_text("\n".join(lines))
        print(f"Exported Markdown to {path}")


def cmd_throughput(args) -> None:
    from .throughput import run_throughput_test
    from .benchmark_results import BenchmarkRecord, save_result

    rt = _find_runtime(getattr(args, "runtime", None))
    if not rt:
        sys.exit(1)

    model_id = args.model or (rt.models[0] if rt.models else None)
    if not model_id:
        print(f"{RED}No model found. Use --model to specify.{RESET}")
        sys.exit(1)

    print_header("Concurrent Throughput Test")
    print(f"  {BOLD}Runtime:{RESET}      {rt.name} ({rt.url})")
    print(f"  {BOLD}Model:{RESET}        {model_id}")
    print(f"  {BOLD}Concurrency:{RESET}  {args.concurrency}")
    print(f"  {BOLD}Requests:{RESET}     {args.requests}")
    print(f"  {BOLD}Max tokens:{RESET}   {args.max_tokens}")
    print()

    def progress(done: int, total: int) -> None:
        print(f"  Request {done}/{total} complete", flush=True)

    result = run_throughput_test(
        base_url=rt.url,
        model_id=model_id,
        concurrency=args.concurrency,
        total_requests=args.requests,
        max_tokens=args.max_tokens,
        api_key=rt.api_key,
        progress_callback=progress,
    )

    print(f"\n  {BOLD}Results:{RESET}")
    print(f"    Throughput:   {GREEN}{result.throughput_tps:.1f} tok/s{RESET} (aggregate)")
    print(f"    Successful:   {result.successful_requests}/{result.total_requests}")
    print(f"    Avg latency:  {result.avg_latency_ms:.0f}ms")
    print(f"    Min/Max:      {result.min_latency_ms:.0f}ms / {result.max_latency_ms:.0f}ms")
    print(f"    Total tokens: {result.total_tokens}")
    print(f"    Wall clock:   {result.elapsed_seconds:.1f}s")
    if result.error_rate > 0:
        print(f"    {RED}Error rate:   {result.error_rate:.0%}{RESET}")

    record = BenchmarkRecord(
        type="speed", model=model_id, runtime=rt.name,
        tokens_per_second=result.throughput_tps,
        generated_tokens=result.total_tokens,
        total_time_ms=result.elapsed_seconds * 1000,
        extra={
            "test_type": "throughput",
            "concurrency": args.concurrency,
            "total_requests": args.requests,
            "avg_latency_ms": result.avg_latency_ms,
            "error_rate": result.error_rate,
        },
    )
    path = save_result(record)
    print(f"\n  Saved to {path}")


def cmd_fetch_config(args) -> None:
    from .hf_templates import fetch_hf_model_config

    print(f"  Fetching config from HuggingFace: {args.repo}...")

    try:
        config = fetch_hf_model_config(args.repo)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        sys.exit(1)

    print_header(f"HuggingFace Model Config — {args.repo}")
    print(f"  {BOLD}Architecture:{RESET}  {config.architecture or '—'}")
    print(f"  {BOLD}Model type:{RESET}    {config.model_type or '—'}")
    print(f"  {BOLD}Layers:{RESET}        {config.num_layers}")
    print(f"  {BOLD}Attn heads:{RESET}    {config.num_attention_heads}")
    print(f"  {BOLD}KV heads:{RESET}      {config.num_kv_heads}")
    print(f"  {BOLD}Head dim:{RESET}      {config.head_dim}")
    print(f"  {BOLD}Hidden size:{RESET}   {config.hidden_size}")
    print(f"  {BOLD}Vocab size:{RESET}    {config.vocab_size}")
    print(f"  {BOLD}Max context:{RESET}   {config.max_position_embeddings}")
    if config.temperature is not None:
        print(f"  {BOLD}Temperature:{RESET}  {config.temperature}")
    if config.top_p is not None:
        print(f"  {BOLD}Top P:{RESET}        {config.top_p}")
    if config.license:
        print(f"  {BOLD}License:{RESET}      {config.license}")

    if args.calc and config.num_layers:
        # Estimate parameter count from architecture
        # hidden_size^2 * num_layers * 12 / 1e9 (rough MLP+attn estimate)
        est_params_b = (config.hidden_size ** 2 * config.num_layers * 12) / 1e9
        quant = args.quant
        if quant not in QUANT_FORMATS:
            print(f"{RED}Unknown quant: {quant}{RESET}")
            return

        bpw = QUANT_FORMATS[quant].bits_per_weight
        model_size = estimate_model_size_gb(est_params_b, bpw)
        kv_cache = estimate_kv_cache_raw(
            config.num_layers, config.num_kv_heads, config.head_dim, args.context,
        )
        overhead = estimate_overhead_gb(model_size)
        total = model_size + kv_cache + overhead

        print(f"\n  {BOLD}VRAM Estimate @ {quant} ({args.context} ctx):{RESET}")
        print(f"    Est. params:    ~{est_params_b:.1f}B")
        print(f"    Model weights:  {format_size(model_size)}")
        print(f"    KV cache:       {format_size(kv_cache)}")
        print(f"    Overhead:       {format_size(overhead)}")
        print(f"    {BOLD}Total:          {format_size(total)}{RESET}")

        mac = resolve_hw_or_detect(None)
        if mac:
            avail = mac.usable_memory_gb
            print(f"    {BOLD}Available:{RESET}      {format_size(avail)}")
            print(f"    {BOLD}Usage:{RESET}          {bar(total, avail)}")


def cmd_eval(args) -> None:
    from .eval_benchmarks import run_eval_suite

    rt = _find_runtime(getattr(args, "runtime", None))
    if not rt:
        sys.exit(1)

    model_id = args.model or (rt.models[0] if rt.models else None)
    if not model_id:
        print(f"{RED}No model found. Use --model to specify.{RESET}")
        sys.exit(1)

    benches = args.bench
    if "all" in benches:
        benches = ["gsm8k", "arc", "hellaswag", "ifeval", "humaneval", "mmlu"]

    print_header(f"LLM Evaluation — {model_id}")
    print(f"  {BOLD}Runtime:{RESET}     {rt.name} ({rt.url})")
    print(f"  {BOLD}Benchmarks:{RESET}  {', '.join(benches)}")
    if args.samples:
        print(f"  {BOLD}Samples:{RESET}     {args.samples} per benchmark")
    print()

    results = run_eval_suite(
        base_url=rt.url,
        model_id=model_id,
        benchmarks=benches,
        max_samples=args.samples,
        api_key=rt.api_key,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {model_id}")
    print(f"{'='*60}\n")
    print(f"  {'Benchmark':<15} {'Score':>8} {'Correct':>10} {'Total':>8}")
    print(f"  {'-'*45}")
    for name, score_data in results.items():
        score_pct = f"{score_data['score']:.1%}"
        correct = score_data["correct"]
        total = score_data["total"]
        print(f"  {name:<15} {score_pct:>8} {correct:>10} {total:>8}")

    # Save results
    from .benchmark_results import BenchmarkRecord, save_result
    record = BenchmarkRecord(
        type="eval",
        model=model_id,
        runtime=rt.name,
        quality_scores={name: data for name, data in results.items()},
        extra={"benchmarks": benches, "samples": args.samples},
    )
    path = save_result(record)
    print(f"\n  Results saved to {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list-hw": lambda: cmd_list_hw(),
        "list-models": lambda: cmd_list_models(),
        "list-quants": lambda: cmd_list_quants(),
        "check": lambda: cmd_check(args),
        "detail": lambda: cmd_detail(args),
        "max-context": lambda: cmd_max_context(args),
        "recommend": lambda: cmd_recommend(args),
        "calc": lambda: cmd_calc(args),
        "memory": lambda: cmd_memory(args),
        "scan": lambda: cmd_scan(args),
        "gpu-optimize": lambda: cmd_gpu_optimize(args),
        "batch-optimize": lambda: cmd_batch_optimize(args),
        "benchmark": lambda: cmd_benchmark(args),
        "quality": lambda: cmd_quality(args),
        "speed": lambda: cmd_speed(args),
        "monitor": lambda: cmd_monitor(args),
        "results": lambda: cmd_results(args),
        "throughput": lambda: cmd_throughput(args),
        "fetch-config": lambda: cmd_fetch_config(args),
        "eval": lambda: cmd_eval(args),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
