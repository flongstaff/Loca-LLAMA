"""CLI interface for Loca-LLAMA."""

import argparse
import sys

from .hardware import APPLE_SILICON_SPECS, MacSpec
from .models import MODELS, LLMModel
from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS
from .analyzer import analyze_model, analyze_all, max_context_for_model


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
    check.add_argument("--hw", required=True, help="Hardware config (e.g. 'M4 Pro 48GB')")
    check.add_argument("--model", help="Filter to a specific model name (substring match)")
    check.add_argument("--family", help="Filter by model family (e.g. Llama, Qwen)")
    check.add_argument("--quant", nargs="+", help="Quantization formats to check (default: recommended)")
    check.add_argument("--context", type=int, help="Context length to evaluate (default: model default)")
    check.add_argument("--all", action="store_true", help="Show all combos (not just those that fit)")

    # ── detail ──
    detail = sub.add_parser("detail", help="Detailed analysis of a specific model")
    detail.add_argument("--hw", required=True, help="Hardware config (e.g. 'M4 Pro 48GB')")
    detail.add_argument("--model", required=True, help="Model name (exact or substring)")
    detail.add_argument("--quant", default="Q4_K_M", help="Quantization format (default: Q4_K_M)")
    detail.add_argument("--context", type=int, help="Context length")

    # ── max-context ──
    mc = sub.add_parser("max-context", help="Find max context length for a model")
    mc.add_argument("--hw", required=True, help="Hardware config")
    mc.add_argument("--model", required=True, help="Model name")
    mc.add_argument("--quant", default="Q4_K_M", help="Quantization format (default: Q4_K_M)")

    # ── recommend ──
    rec = sub.add_parser("recommend", help="Get recommendations for your hardware")
    rec.add_argument("--hw", required=True, help="Hardware config")
    rec.add_argument("--use-case", choices=["general", "coding", "reasoning", "small", "large-context"],
                     default="general", help="Intended use case")

    # ── scan ──
    scan = sub.add_parser("scan", help="Scan for locally downloaded models")
    scan.add_argument("--dir", help="Custom directory to scan")

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
    mac = resolve_hw(args.hw)
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

    print_header(f"LLM Compatibility — {args.hw}")
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
    mac = resolve_hw(args.hw)
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
    mac = resolve_hw(args.hw)
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
    mac = resolve_hw(args.hw)
    if not mac:
        sys.exit(1)

    use_case = args.use_case

    # Filter models by use case
    if use_case == "coding":
        families = ["Qwen", "CodeLlama", "StarCoder", "DeepSeek"]
        models = [m for m in MODELS if m.family in families]
    elif use_case == "reasoning":
        models = [m for m in MODELS if "R1" in m.name or "DeepSeek" in m.family or m.family == "Qwen"]
    elif use_case == "small":
        models = [m for m in MODELS if m.params_billion <= 8]
    elif use_case == "large-context":
        models = [m for m in MODELS if m.max_context_length >= 65536]
    else:
        models = list(MODELS)

    print_header(f"Recommendations — {args.hw} ({use_case})")
    print_hw_summary(mac)

    # Find best configs: prioritize quality (higher quant) that still fits well
    recommendations = []
    seen_models = set()
    for quant_name in ["Q6_K", "Q5_K_M", "Q4_K_M", "Q8_0", "Q3_K_L"]:
        if quant_name not in QUANT_FORMATS:
            continue
        quant = QUANT_FORMATS[quant_name]
        for model in sorted(models, key=lambda m: m.params_billion, reverse=True):
            if model.name in seen_models:
                continue
            result = analyze_model(mac, model, quant)
            if result.fits_in_memory and result.memory_utilization_pct <= 90:
                recommendations.append(result)
                seen_models.add(model.name)

    if not recommendations:
        print(f"  {RED}No good recommendations found for this use case.{RESET}")
        return

    # Sort by params (larger = more capable)
    recommendations.sort(key=lambda r: r.model.params_billion, reverse=True)

    # Show top picks
    top = recommendations[:8]
    print(f"  {BOLD}Top picks (best quality quant that fits comfortably):{RESET}\n")

    for i, r in enumerate(top, 1):
        prefix = f"  {MAGENTA}{i}.{RESET} "
        print(f"{prefix}{BOLD}{r.model.name}{RESET} @ {CYAN}{r.quant.name}{RESET}")
        print(f"     Memory: {format_size(r.total_memory_gb)} / {format_size(r.available_memory_gb)} "
              f"({r.memory_utilization_pct:.0f}%)  |  "
              f"Speed: {format_tok_s(r.estimated_tok_per_sec)}  |  "
              f"Max ctx: {format_context(max_context_for_model(mac, r.model, r.quant))}")
        print()


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
        "lm-studio": MAGENTA, "llama.cpp": CYAN, "ollama": GREEN,
        "huggingface": YELLOW, "mlx-community": GREEN,
    }

    for i, m in enumerate(models, 1):
        sc = source_colors.get(m.source, WHITE)
        quant_str = f" [{m.quant}]" if m.quant else ""
        print(
            f"  {i:>3}) {BOLD}{m.name}{RESET}{quant_str}\n"
            f"       {format_size(m.size_gb):>8}  {sc}{m.source:<14}{RESET}  {m.format:<10}  {DIM}{m.path}{RESET}"
        )


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
        "scan": lambda: cmd_scan(args),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
