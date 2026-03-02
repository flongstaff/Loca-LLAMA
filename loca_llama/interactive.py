"""Interactive terminal UI for Loca-LLAMA."""

import sys

from .hardware import APPLE_SILICON_SPECS, MacSpec
from .models import MODELS
from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS
from .analyzer import analyze_model, max_context_for_model
from .scanner import scan_all, scan_custom_dir, LocalModel
from .hub import search_huggingface, search_gguf_models, search_mlx_models, format_downloads
from .benchmark import (
    detect_llama_cpp_server,
    detect_lm_studio,
    get_lm_studio_models,
    benchmark_openai_api,
    benchmark_llama_cpp_native,
    BenchmarkResult,
)

# ── ANSI ─────────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BLUE = "\033[34m"
BG_CYAN = "\033[46m"
BG_MAGENTA = "\033[45m"


def clear_screen():
    print("\033[2J\033[H", end="")


def print_banner():
    print(f"""
{BOLD}{MAGENTA}  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║   {CYAN}██╗      ██████╗  ██████╗ █████╗                            {MAGENTA}║
  ║   {CYAN}██║     ██╔═══██╗██╔════╝██╔══██╗                           {MAGENTA}║
  ║   {CYAN}██║     ██║   ██║██║     ███████║                           {MAGENTA}║
  ║   {CYAN}██║     ██║   ██║██║     ██╔══██║                           {MAGENTA}║
  ║   {CYAN}███████╗╚██████╔╝╚██████╗██║  ██║                           {MAGENTA}║
  ║   {CYAN}╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝                          {MAGENTA}║
  ║               {WHITE}Local LLM Apple Mac Analyzer{MAGENTA}                    ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝{RESET}
""")


def prompt_choice(prompt: str, options: list[str], allow_back: bool = True) -> int | None:
    """Show a numbered menu and return the selected index, or None for back."""
    print()
    for i, opt in enumerate(options, 1):
        print(f"  {CYAN}{i:>3}{RESET}) {opt}")
    if allow_back:
        print(f"  {DIM}  0) ← Back{RESET}")
    print()
    while True:
        try:
            raw = input(f"  {BOLD}{prompt}{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        raw = raw.strip()
        if not raw:
            continue
        if raw == "0" and allow_back:
            return None
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  {RED}Invalid choice. Enter 1-{len(options)}{' or 0 to go back' if allow_back else ''}.{RESET}")


def prompt_input(prompt: str, default: str = "") -> str:
    """Get text input with optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"  {BOLD}{prompt}{suffix}:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def format_size(gb: float) -> str:
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def format_context(ctx: int) -> str:
    if ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)


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


def bar(used: float, total: float, width: int = 25) -> str:
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


# ── Screens ──────────────────────────────────────────────────────────────────

def select_hardware() -> MacSpec | None:
    """Interactive hardware selection."""
    print(f"\n  {BOLD}{CYAN}Select Your Mac Configuration{RESET}")
    print(f"  {DIM}Choose your Apple Silicon chip and memory{RESET}")

    # Group by chip family
    families: dict[str, list[str]] = {}
    for name in APPLE_SILICON_SPECS:
        chip = name.rsplit(" ", 1)[0]  # e.g. "M4 Pro"
        families.setdefault(chip, []).append(name)

    chip_names = list(families.keys())
    idx = prompt_choice("Select chip:", chip_names, allow_back=False)
    if idx is None:
        return None

    chip = chip_names[idx]
    configs = families[chip]

    if len(configs) == 1:
        selected = configs[0]
    else:
        mem_options = [f"{c} ({APPLE_SILICON_SPECS[c].memory_bandwidth_gbs:.0f} GB/s bandwidth)" for c in configs]
        idx2 = prompt_choice("Select memory:", mem_options)
        if idx2 is None:
            return select_hardware()
        selected = configs[idx2]

    mac = APPLE_SILICON_SPECS[selected]
    print(f"\n  {GREEN}✓{RESET} Selected: {BOLD}{selected}{RESET}")
    print(f"    {mac.cpu_cores} CPU cores, {mac.gpu_cores} GPU cores, {mac.memory_gb} GB, {mac.memory_bandwidth_gbs:.0f} GB/s")
    return mac


def screen_check_models(mac: MacSpec):
    """Check which models from the database can run."""
    print(f"\n  {BOLD}{CYAN}Model Compatibility Check{RESET}")

    # Select quant format
    quant_options = [
        f"{QUANT_FORMATS[q].name} — {QUANT_FORMATS[q].bits_per_weight:.1f} bpw ({QUANT_FORMATS[q].quality_rating})"
        for q in RECOMMENDED_FORMATS
    ]
    quant_options.append("All recommended formats at once")

    idx = prompt_choice("Select quantization:", quant_options)
    if idx is None:
        return

    if idx == len(RECOMMENDED_FORMATS):
        quants = [QUANT_FORMATS[q] for q in RECOMMENDED_FORMATS]
    else:
        quants = [QUANT_FORMATS[RECOMMENDED_FORMATS[idx]]]

    # Filter by family?
    families = sorted(set(m.family for m in MODELS))
    family_options = ["All families"] + families
    fidx = prompt_choice("Filter by model family:", family_options)
    if fidx is None:
        return

    models = MODELS
    if fidx > 0:
        selected_family = families[fidx - 1]
        models = [m for m in models if m.family == selected_family]

    # Analyze
    print(f"\n  {BOLD}Analyzing {len(models)} models × {len(quants)} quant(s) against {mac.chip} {mac.memory_gb}GB...{RESET}\n")

    results = []
    for model in models:
        for quant in quants:
            est = analyze_model(mac, model, quant)
            results.append(est)

    # Split into fits / doesn't fit
    fits = [r for r in results if r.fits_in_memory]
    no_fit = [r for r in results if not r.fits_in_memory]

    fits.sort(key=lambda r: r.model.params_billion, reverse=True)

    if fits:
        print(f"  {GREEN}{BOLD}Models that FIT ({len(fits)}):{RESET}\n")
        name_w = max(len(r.model.name) for r in fits) + 1
        for r in fits:
            max_ctx = max_context_for_model(mac, r.model, r.quant)
            print(
                f"  {r.model.name:<{name_w}} {CYAN}{r.quant.name:<8}{RESET} "
                f"{format_size(r.total_memory_gb):>8} "
                f"{bar(r.total_memory_gb, r.available_memory_gb)} "
                f"{color_rating(r.rating):<14} "
                f"max ctx: {format_context(max_ctx):>6}"
            )
    else:
        print(f"  {RED}No models fit with the selected configuration.{RESET}")

    if no_fit:
        print(f"\n  {RED}{BOLD}Too large ({len(no_fit)}):{RESET}\n")
        name_w = max(len(r.model.name) for r in no_fit) + 1
        for r in sorted(no_fit, key=lambda r: r.total_memory_gb):
            over = r.total_memory_gb - r.available_memory_gb
            print(
                f"  {DIM}{r.model.name:<{name_w}} {r.quant.name:<8} "
                f"{format_size(r.total_memory_gb):>8} — needs {format_size(over)} more{RESET}"
            )

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def screen_scan_local(mac: MacSpec):
    """Scan for locally downloaded models."""
    print(f"\n  {BOLD}{CYAN}Scanning for Local Models...{RESET}\n")

    models = scan_all()

    if not models:
        print(f"  {YELLOW}No models found in standard locations.{RESET}")
        custom = prompt_input("Enter a custom directory to scan (or empty to skip)")
        if custom:
            models = scan_custom_dir(custom)

    if not models:
        print(f"  {RED}No model files found.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    print(f"  {GREEN}Found {len(models)} model(s):{RESET}\n")

    for i, m in enumerate(models, 1):
        source_color = {
            "lm-studio": MAGENTA,
            "llama.cpp": CYAN,
            "huggingface": YELLOW,
            "mlx-community": GREEN,
        }.get(m.source, WHITE)

        quant_str = f" [{m.quant}]" if m.quant else ""
        print(
            f"  {CYAN}{i:>3}{RESET}) {BOLD}{m.name}{RESET}{quant_str}\n"
            f"       {format_size(m.size_gb):>8}  "
            f"{source_color}{m.source:<14}{RESET}  "
            f"{m.format:<12}  "
            f"{DIM}{m.path}{RESET}"
        )

    # Offer to analyze a model
    print()
    idx = prompt_choice("Select a model to analyze:", [m.name for m in models])
    if idx is None:
        return

    selected = models[idx]
    print(f"\n  {BOLD}Analyzing: {selected.name}{RESET}")
    print(f"  Size on disk: {format_size(selected.size_gb)}")
    print(f"  Fits in memory: {'Yes' if selected.size_gb * 1.15 <= mac.usable_memory_gb else 'No'}")
    available = mac.usable_memory_gb
    used = selected.size_gb * 1.15  # rough with overhead
    print(f"  Memory: {bar(used, available, 30)}")

    if selected.size_gb * 1.15 <= mac.usable_memory_gb:
        headroom = available - used
        print(f"  Headroom: {format_size(headroom)} (for KV cache / context)")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def screen_search_hub(mac: MacSpec):
    """Search HuggingFace for models."""
    print(f"\n  {BOLD}{CYAN}Search HuggingFace Hub{RESET}")

    format_options = ["GGUF models (for llama.cpp / LM Studio)", "MLX models (for Apple MLX)", "All models"]
    fidx = prompt_choice("Search for:", format_options)
    if fidx is None:
        return

    query = prompt_input("Search query (e.g. 'llama 8b', 'deepseek coder')")
    if not query:
        return

    print(f"\n  {DIM}Searching HuggingFace...{RESET}\n")

    if fidx == 0:
        results = search_gguf_models(query)
    elif fidx == 1:
        results = search_mlx_models(query)
    else:
        results = search_huggingface(query)

    if not results:
        print(f"  {RED}No results found.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    print(f"  {GREEN}Found {len(results)} model(s):{RESET}\n")

    for i, m in enumerate(results, 1):
        tags = []
        if m.is_gguf:
            tags.append(f"{CYAN}GGUF{RESET}")
        if m.is_mlx:
            tags.append(f"{GREEN}MLX{RESET}")
        tag_str = " ".join(tags)

        print(
            f"  {CYAN}{i:>3}{RESET}) {BOLD}{m.repo_id}{RESET} {tag_str}\n"
            f"       ↓ {format_downloads(m.downloads)}  ♥ {m.likes}"
            f"  {DIM}{m.pipeline_tag or ''}{RESET}"
        )

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def screen_benchmark(mac: MacSpec):
    """Run benchmarks on local models."""
    print(f"\n  {BOLD}{CYAN}Benchmark: LM Studio vs llama.cpp{RESET}\n")

    # Detect running servers
    llama_url = detect_llama_cpp_server()
    lm_url = detect_lm_studio()

    if llama_url:
        print(f"  {GREEN}✓{RESET} llama.cpp server detected at {llama_url}")
    else:
        print(f"  {DIM}✗ llama.cpp server not detected{RESET}")

    if lm_url:
        print(f"  {GREEN}✓{RESET} LM Studio detected at {lm_url}")
        lm_models = get_lm_studio_models(lm_url)
        if lm_models:
            print(f"    Loaded models: {', '.join(lm_models)}")
    else:
        print(f"  {DIM}✗ LM Studio not detected{RESET}")

    if not llama_url and not lm_url:
        print(f"\n  {YELLOW}No runtime servers detected.{RESET}")
        print(f"  {DIM}Start LM Studio or llama.cpp server first, then try again.{RESET}")
        print(f"  {DIM}  llama.cpp: llama-server -m model.gguf --port 8080{RESET}")
        print(f"  {DIM}  LM Studio: Start local server from the app (default port 1234){RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    # Choose what to benchmark
    options = []
    if lm_url:
        lm_models = get_lm_studio_models(lm_url)
        for m in lm_models:
            options.append(f"LM Studio: {m}")
    if llama_url:
        options.append(f"llama.cpp server ({llama_url})")

    # Also offer to benchmark local GGUF files with CLI
    local = scan_all()
    gguf_models = [m for m in local if m.format == "gguf"]
    for m in gguf_models[:10]:
        options.append(f"llama.cpp CLI: {m.name} ({format_size(m.size_gb)})")

    if not options:
        print(f"\n  {RED}No models available for benchmarking.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    idx = prompt_choice("Select model to benchmark:", options)
    if idx is None:
        return

    print(f"\n  {BOLD}Running benchmark...{RESET}")
    print(f"  {DIM}(This may take 30-60 seconds){RESET}\n")

    results: list[BenchmarkResult] = []

    # Determine what was selected and benchmark
    offset = 0
    if lm_url:
        lm_models = get_lm_studio_models(lm_url)
        if idx < len(lm_models):
            r = benchmark_openai_api(lm_url, lm_models[idx], "lm-studio")
            results.append(r)
            # Also try llama.cpp if available
            if llama_url:
                r2 = benchmark_openai_api(llama_url, lm_models[idx], "llama.cpp-server")
                results.append(r2)
        offset = len(lm_models)

    if llama_url and idx == offset:
        r = benchmark_openai_api(llama_url, "default", "llama.cpp-server")
        results.append(r)
        offset += 1

    gguf_start = offset
    if idx >= gguf_start and idx < gguf_start + len(gguf_models):
        gidx = idx - gguf_start
        r = benchmark_llama_cpp_native(str(gguf_models[gidx].path))
        results.append(r)

    # Display results
    print_benchmark_results(results)
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def print_benchmark_results(results: list[BenchmarkResult]):
    """Display benchmark results in a nice format."""
    if not results:
        print(f"  {RED}No benchmark results.{RESET}")
        return

    print(f"\n  {BOLD}{'Runtime':<20} {'Gen Speed':>12} {'Prefill':>12} {'Tokens':>8} {'Time':>10} {'Status':>8}{RESET}")
    print(f"  {'─' * 72}")

    for r in results:
        if r.success:
            tok_s = f"{r.tokens_per_second:.1f} tok/s"
            if r.tokens_per_second >= 30:
                tok_s = f"{GREEN}{tok_s}{RESET}"
            elif r.tokens_per_second >= 10:
                tok_s = f"{YELLOW}{tok_s}{RESET}"
            else:
                tok_s = f"{RED}{tok_s}{RESET}"

            prefill = f"{r.prompt_tokens_per_second:.0f} tok/s"
            tokens = f"{r.generated_tokens}"
            time_str = f"{r.total_time_ms / 1000:.1f}s"
            status = f"{GREEN}OK{RESET}"
        else:
            tok_s = f"{DIM}—{RESET}"
            prefill = f"{DIM}—{RESET}"
            tokens = f"{DIM}—{RESET}"
            time_str = f"{DIM}—{RESET}"
            status = f"{RED}FAIL{RESET}"

        print(f"  {r.runtime:<20} {tok_s:>12} {prefill:>12} {tokens:>8} {time_str:>10} {status:>8}")

    if any(r.error for r in results):
        print()
        for r in results:
            if r.error:
                print(f"  {RED}Error ({r.runtime}): {r.error}{RESET}")

    # Winner
    successful = [r for r in results if r.success]
    if len(successful) >= 2:
        best = max(successful, key=lambda r: r.tokens_per_second)
        print(f"\n  {GREEN}{BOLD}Winner: {best.runtime}{RESET} ({best.tokens_per_second:.1f} tok/s)")


def screen_detail(mac: MacSpec):
    """Detailed analysis of a specific model."""
    print(f"\n  {BOLD}{CYAN}Detailed Model Analysis{RESET}")

    # Select model
    model_names = [f"{m.name} ({m.params_billion}B)" for m in MODELS]
    idx = prompt_choice("Select model:", model_names)
    if idx is None:
        return

    model = MODELS[idx]

    # Select quant
    quant_options = [
        f"{q} — {QUANT_FORMATS[q].bits_per_weight:.1f} bpw ({QUANT_FORMATS[q].quality_rating})"
        for q in QUANT_FORMATS
    ]
    qidx = prompt_choice("Select quantization:", quant_options)
    if qidx is None:
        return

    quant_name = list(QUANT_FORMATS.keys())[qidx]
    quant = QUANT_FORMATS[quant_name]

    result = analyze_model(mac, model, quant)
    max_ctx = max_context_for_model(mac, model, quant)

    # Display
    print(f"\n  {BOLD}{'═' * 60}{RESET}")
    print(f"  {BOLD}{model.name} @ {quant.name}{RESET}")
    print(f"  {BOLD}{'═' * 60}{RESET}\n")

    print(f"  {BOLD}Model Info:{RESET}")
    print(f"    Parameters:     {model.params_billion}B")
    print(f"    Family:         {model.family}")
    print(f"    License:        {model.license}")
    print(f"    Default ctx:    {format_context(model.default_context_length)}")
    print(f"    Max ctx:        {format_context(model.max_context_length)}")

    print(f"\n  {BOLD}Quantization:{RESET}")
    print(f"    Format:         {quant.name} ({quant.bits_per_weight:.2f} bits/weight)")
    print(f"    Quality:        {quant.quality_rating}")
    print(f"    {DIM}{quant.description}{RESET}")

    print(f"\n  {BOLD}Memory Breakdown:{RESET}")
    print(f"    Model weights:  {format_size(result.model_size_gb):>10}")
    print(f"    KV cache:       {format_size(result.kv_cache_gb):>10}  (at {format_context(result.context_length)} ctx)")
    print(f"    Overhead:       {format_size(result.overhead_gb):>10}")
    print(f"    {'─' * 28}")
    print(f"    {BOLD}Total:          {format_size(result.total_memory_gb):>10}{RESET}")
    print(f"    Available:      {format_size(result.available_memory_gb):>10}")

    print(f"\n  {BOLD}Memory:{RESET} {bar(result.total_memory_gb, result.available_memory_gb, 35)}")
    print(f"  {BOLD}Rating:{RESET} {color_rating(result.rating)}")

    if result.estimated_tok_per_sec:
        tok_s = result.estimated_tok_per_sec
        color = GREEN if tok_s >= 30 else (YELLOW if tok_s >= 10 else RED)
        print(f"  {BOLD}Speed:{RESET}  {color}{tok_s:.1f} tok/s{RESET} (estimated)")

    print(f"\n  {BOLD}Context Length Scaling:{RESET}")
    print(f"    Max that fits:  {format_context(min(max_ctx, model.max_context_length))} tokens")
    if max_ctx > model.max_context_length:
        print(f"    {DIM}(Memory allows more, but model architecture caps at {format_context(model.max_context_length)}){RESET}")

    from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb, estimate_overhead_gb
    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    overhead = estimate_overhead_gb(model_size)

    print(f"\n    {'Context':>10} {'KV Cache':>10} {'Total':>10} {'Fits':>6}")
    print(f"    {'─' * 40}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if ctx > model.max_context_length:
            break
        kv = estimate_kv_cache_gb(model, ctx)
        total = model_size + kv + overhead
        fits = total <= mac.usable_memory_gb
        mark = f"{GREEN}✓{RESET}" if fits else f"{RED}✗{RESET}"
        print(f"    {format_context(ctx):>10} {format_size(kv):>10} {format_size(total):>10} {mark:>6}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def main_interactive():
    """Main interactive loop."""
    clear_screen()
    print_banner()

    # Select hardware first
    mac = select_hardware()
    if not mac:
        print(f"\n  {RED}No hardware selected. Exiting.{RESET}")
        sys.exit(0)

    while True:
        print(f"\n  {BOLD}{CYAN}{'═' * 50}{RESET}")
        print(f"  {BOLD}{CYAN}  Main Menu — {mac.chip} {mac.memory_gb}GB{RESET}")
        print(f"  {BOLD}{CYAN}{'═' * 50}{RESET}")

        menu = [
            f"{BOLD}Check Model Compatibility{RESET}     — Which models can I run?",
            f"{BOLD}Scan Local Models{RESET}             — Find downloaded models",
            f"{BOLD}Search HuggingFace{RESET}            — Browse GGUF & MLX models",
            f"{BOLD}Detailed Model Analysis{RESET}       — Deep dive on one model",
            f"{BOLD}Benchmark{RESET}                     — Compare LM Studio vs llama.cpp",
            f"{BOLD}Change Hardware{RESET}               — Switch Mac configuration",
            f"{RED}Exit{RESET}",
        ]

        idx = prompt_choice("What would you like to do?", menu, allow_back=False)

        if idx is None or idx == 6:
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        elif idx == 0:
            screen_check_models(mac)
        elif idx == 1:
            screen_scan_local(mac)
        elif idx == 2:
            screen_search_hub(mac)
        elif idx == 3:
            screen_detail(mac)
        elif idx == 4:
            screen_benchmark(mac)
        elif idx == 5:
            new_mac = select_hardware()
            if new_mac:
                mac = new_mac
