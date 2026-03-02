"""Interactive terminal UI for Loca-LLAMA."""

import sys
import time

from .hardware import APPLE_SILICON_SPECS, MacSpec
from .models import MODELS
from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS
from .analyzer import (
    analyze_model, max_context_for_model,
    estimate_model_size_gb, estimate_kv_cache_gb, estimate_overhead_gb,
)
from .scanner import scan_all, scan_custom_dir, LocalModel
from .hub import (
    search_huggingface, search_gguf_models, search_mlx_models,
    get_model_files, format_downloads, HubModel,
)
from .benchmark import (
    detect_all_runtimes, detect_ollama,
    RuntimeInfo, BenchmarkResult,
    run_benchmark_suite, aggregate_results,
    benchmark_llama_cpp_native,
    BENCH_PROMPTS,
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
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"
BG_CYAN = "\033[46m"
BG_MAGENTA = "\033[45m"
UNDERLINE = "\033[4m"


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


# ── Input helpers ────────────────────────────────────────────────────────────

def prompt_choice(prompt: str, options: list[str], allow_back: bool = True) -> int | None:
    """Show a numbered menu and return the selected index, or None for back."""
    print()
    for i, opt in enumerate(options, 1):
        print(f"  {CYAN}{i:>3}{RESET}) {opt}")
    if allow_back:
        print(f"  {DIM}  0) <- Back{RESET}")
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
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"  {BOLD}{prompt}{suffix}:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {BOLD}{prompt} {suffix}:{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    return raw in ("y", "yes")


# ── Format helpers ───────────────────────────────────────────────────────────

def format_size(gb: float) -> str:
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def format_context(ctx: int) -> str:
    if ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)


def color_rating(rating: str) -> str:
    colors = {
        "Excellent": f"{GREEN}{BOLD}",
        "Good": GREEN,
        "Tight": YELLOW,
        "Very Tight": f"{YELLOW}{BOLD}",
        "Barely fits": f"{YELLOW}{BOLD}",
    }
    c = colors.get(rating, f"{RED}{BOLD}")
    return f"{c}{rating}{RESET}"


def bar(used: float, total: float, width: int = 25) -> str:
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    empty = width - filled
    color = GREEN if pct <= 0.60 else (YELLOW if pct <= 0.80 else RED)
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {pct:.0%}"


def color_speed(tok_s: float) -> str:
    if tok_s >= 30:
        return f"{GREEN}{tok_s:.1f} tok/s{RESET}"
    elif tok_s >= 10:
        return f"{YELLOW}{tok_s:.1f} tok/s{RESET}"
    else:
        return f"{RED}{tok_s:.1f} tok/s{RESET}"


def source_badge(source: str) -> str:
    badges = {
        "lm-studio": f"{BG_MAGENTA}{WHITE}{BOLD} LM Studio {RESET}",
        "llama.cpp": f"{BG_CYAN}{WHITE}{BOLD} llama.cpp {RESET}",
        "ollama": f"{BG_GREEN}{WHITE}{BOLD} Ollama {RESET}",
        "huggingface": f"{BG_YELLOW}{WHITE}{BOLD} HF {RESET}",
        "mlx-community": f"{BG_GREEN}{WHITE}{BOLD} MLX {RESET}",
        "custom": f"{DIM}[custom]{RESET}",
    }
    return badges.get(source, f"{DIM}[{source}]{RESET}")


def runtime_badge(name: str) -> str:
    badges = {
        "lm-studio": f"{BG_MAGENTA}{WHITE}{BOLD} LM Studio {RESET}",
        "llama.cpp-server": f"{BG_CYAN}{WHITE}{BOLD} llama.cpp {RESET}",
        "llama.cpp-cli": f"{BG_CYAN}{WHITE}{BOLD} llama.cpp CLI {RESET}",
        "ollama": f"{BG_GREEN}{WHITE}{BOLD} Ollama {RESET}",
    }
    return badges.get(name, name)


def print_separator(title: str = "", char: str = "─", width: int = 65):
    if title:
        left = 3
        right = width - left - len(title) - 2
        print(f"  {char * left} {BOLD}{title}{RESET} {char * max(right, 1)}")
    else:
        print(f"  {char * width}")


# ── Hardware Selection ───────────────────────────────────────────────────────

def select_hardware() -> MacSpec | None:
    print(f"\n  {BOLD}{CYAN}Select Your Mac Configuration{RESET}")
    print(f"  {DIM}Choose your Apple Silicon chip and memory{RESET}")

    families: dict[str, list[str]] = {}
    for name in APPLE_SILICON_SPECS:
        chip = name.rsplit(" ", 1)[0]
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
        mem_options = [
            f"{c} ({APPLE_SILICON_SPECS[c].memory_bandwidth_gbs:.0f} GB/s bandwidth)"
            for c in configs
        ]
        idx2 = prompt_choice("Select memory:", mem_options)
        if idx2 is None:
            return select_hardware()
        selected = configs[idx2]

    mac = APPLE_SILICON_SPECS[selected]
    print(f"\n  {GREEN}✓{RESET} Selected: {BOLD}{selected}{RESET}")
    print(f"    {mac.cpu_cores} CPU cores, {mac.gpu_cores} GPU cores, "
          f"{mac.memory_gb} GB unified, {mac.memory_bandwidth_gbs:.0f} GB/s")
    return mac


# ── Screen: Check Models ─────────────────────────────────────────────────────

def screen_check_models(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Model Compatibility Check{RESET}")

    quant_options = [
        f"{QUANT_FORMATS[q].name} -- {QUANT_FORMATS[q].bits_per_weight:.1f} bpw ({QUANT_FORMATS[q].quality_rating})"
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

    families = sorted(set(m.family for m in MODELS))
    family_options = ["All families"] + families
    fidx = prompt_choice("Filter by model family:", family_options)
    if fidx is None:
        return
    models = MODELS if fidx == 0 else [m for m in MODELS if m.family == families[fidx - 1]]

    print(f"\n  {DIM}Analyzing {len(models)} models x {len(quants)} quant(s)...{RESET}\n")

    results = []
    for model in models:
        for quant in quants:
            results.append(analyze_model(mac, model, quant))

    fits = sorted([r for r in results if r.fits_in_memory],
                  key=lambda r: r.model.params_billion, reverse=True)
    no_fit = sorted([r for r in results if not r.fits_in_memory],
                    key=lambda r: r.total_memory_gb)

    if fits:
        print_separator(f"Models that FIT ({len(fits)})")
        print()
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

    if no_fit:
        print(f"\n")
        print_separator(f"Too large ({len(no_fit)})")
        print()
        name_w = max(len(r.model.name) for r in no_fit) + 1
        for r in no_fit[:15]:
            over = r.total_memory_gb - r.available_memory_gb
            print(
                f"  {DIM}{r.model.name:<{name_w}} {r.quant.name:<8} "
                f"{format_size(r.total_memory_gb):>8} -- needs {format_size(over)} more{RESET}"
            )
        if len(no_fit) > 15:
            print(f"  {DIM}... and {len(no_fit) - 15} more{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: My Models (unified browser) ──────────────────────────────────────

def screen_my_models(mac: MacSpec):
    """Unified browser for all locally available models."""
    print(f"\n  {BOLD}{CYAN}My Downloaded Models{RESET}")
    print(f"  {DIM}Scanning LM Studio, llama.cpp, Ollama, HuggingFace cache, MLX...{RESET}\n")

    models = scan_all()

    if not models:
        print(f"  {YELLOW}No models found in standard locations.{RESET}")
        print(f"  {DIM}Looked in:{RESET}")
        print(f"  {DIM}  - ~/.cache/lm-studio/models{RESET}")
        print(f"  {DIM}  - ~/llama.cpp/models{RESET}")
        print(f"  {DIM}  - ~/.cache/huggingface/hub{RESET}")
        print(f"  {DIM}  - Ollama API (port 11434){RESET}")
        print()
        custom = prompt_input("Enter a custom directory to scan (or empty to skip)")
        if custom:
            models = scan_custom_dir(custom)
        if not models:
            print(f"  {RED}No model files found.{RESET}")
            input(f"\n  {DIM}Press Enter to continue...{RESET}")
            return

    # Group by source
    by_source: dict[str, list[LocalModel]] = {}
    for m in models:
        by_source.setdefault(m.source, []).append(m)

    # Summary line
    sources_summary = ", ".join(
        f"{source_badge(s)} x{len(ms)}" for s, ms in by_source.items()
    )
    print(f"  Found {BOLD}{len(models)}{RESET} models: {sources_summary}\n")

    # Display grouped
    display_models: list[LocalModel] = []
    for source in ["ollama", "lm-studio", "llama.cpp", "mlx-community", "huggingface", "custom"]:
        group = by_source.get(source, [])
        if not group:
            continue
        print(f"  {source_badge(source)}")
        for m in group:
            idx = len(display_models) + 1
            quant_str = f" {YELLOW}[{m.quant}]{RESET}" if m.quant else ""
            family_str = f" {DIM}({m.family}){RESET}" if m.family else ""
            extra_info = ""
            if m.metadata.get("parameter_size"):
                extra_info = f" {MAGENTA}{m.metadata['parameter_size']}{RESET}"
            print(
                f"    {CYAN}{idx:>3}{RESET}) {BOLD}{m.name}{RESET}{quant_str}{family_str}{extra_info}"
                f"  {DIM}{format_size(m.size_gb)}{RESET}"
            )
            display_models.append(m)
        print()

    # Actions
    while True:
        actions = [
            f"{BOLD}Analyze a model{RESET}        -- Memory breakdown + compatibility",
            f"{BOLD}Benchmark a model{RESET}      -- Test it on available runtimes",
            f"{BOLD}Scan custom directory{RESET}  -- Look somewhere else",
        ]
        aidx = prompt_choice("What next?", actions)
        if aidx is None:
            return

        if aidx == 0:
            # Analyze
            midx = prompt_choice(
                "Select model #:",
                [f"{m.name} ({format_size(m.size_gb)})" for m in display_models],
            )
            if midx is not None:
                _analyze_local_model(mac, display_models[midx])

        elif aidx == 1:
            # Benchmark
            midx = prompt_choice(
                "Select model to benchmark:",
                [f"{m.name} ({format_size(m.size_gb)})" for m in display_models],
            )
            if midx is not None:
                _benchmark_local_model(mac, display_models[midx])

        elif aidx == 2:
            custom = prompt_input("Directory path")
            if custom:
                extra = scan_custom_dir(custom)
                if extra:
                    print(f"  {GREEN}Found {len(extra)} more models!{RESET}")
                    models.extend(extra)
                    display_models.extend(extra)
                else:
                    print(f"  {YELLOW}No models found in that directory.{RESET}")


def _analyze_local_model(mac: MacSpec, m: LocalModel):
    """Show analysis for a local model based on its file size."""
    print(f"\n  {BOLD}{'═' * 55}{RESET}")
    print(f"  {BOLD}{m.name}{RESET}")
    print(f"  {BOLD}{'═' * 55}{RESET}\n")

    print(f"  {BOLD}Source:{RESET}      {source_badge(m.source)}")
    print(f"  {BOLD}Format:{RESET}      {m.format}")
    print(f"  {BOLD}Size:{RESET}        {format_size(m.size_gb)}")
    if m.quant:
        print(f"  {BOLD}Quant:{RESET}       {m.quant}")
    if m.family:
        print(f"  {BOLD}Family:{RESET}      {m.family}")
    if m.repo_id:
        print(f"  {BOLD}Repo:{RESET}        {m.repo_id}")
    print(f"  {BOLD}Path:{RESET}        {DIM}{m.path}{RESET}")

    # Memory analysis
    overhead_factor = 1.15  # ~15% overhead for KV cache + buffers at default ctx
    total_estimated = m.size_gb * overhead_factor
    available = mac.usable_memory_gb
    fits = total_estimated <= available

    print(f"\n  {BOLD}Memory Analysis:{RESET}")
    print(f"    Model file:     {format_size(m.size_gb):>10}")
    print(f"    Est. overhead:  {format_size(m.size_gb * 0.15):>10}  (KV cache + buffers @ 4K ctx)")
    print(f"    {'─' * 28}")
    print(f"    Est. total:     {format_size(total_estimated):>10}")
    print(f"    Available:      {format_size(available):>10}")
    print(f"\n  {BOLD}Memory:{RESET} {bar(total_estimated, available, 35)}")

    if fits:
        headroom = available - total_estimated
        print(f"  {GREEN}{BOLD}Fits!{RESET} {format_size(headroom)} headroom for larger context")

        # Estimate speed from file size
        tok_s = mac.memory_bandwidth_gbs / (m.size_gb * 1.1)
        print(f"  {BOLD}Est. speed:{RESET} {color_speed(tok_s)}")

        # Estimate max context (rough: headroom / ~0.25 MB per 1K ctx as ballpark)
        if headroom > 0:
            # Very rough: ~256 bytes per token per layer pair for typical 32-layer model
            approx_ctx_tokens = int(headroom * 1024 * 1024 * 1024 / (256 * 32 * 2))
            print(f"  {BOLD}Est. max ctx:{RESET} ~{format_context(min(approx_ctx_tokens, 131072))}")
    else:
        over = total_estimated - available
        print(f"  {RED}{BOLD}Does not fit.{RESET} Needs {format_size(over)} more RAM.")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _benchmark_local_model(mac: MacSpec, m: LocalModel):
    """Benchmark a local model across available runtimes."""
    print(f"\n  {BOLD}{CYAN}Benchmark: {m.name}{RESET}\n")

    # Detect runtimes
    runtimes = detect_all_runtimes()

    if not runtimes and m.format != "gguf":
        print(f"  {RED}No runtimes detected and model is not GGUF (can't use llama.cpp CLI).{RESET}")
        print(f"  {DIM}Start LM Studio, llama.cpp server, or Ollama first.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    if runtimes:
        print(f"  {BOLD}Detected runtimes:{RESET}")
        for rt in runtimes:
            ver = f" v{rt.version}" if rt.version else ""
            print(f"    {runtime_badge(rt.name)}{ver}  {DIM}{rt.url}{RESET}")
            for rm in rt.models[:5]:
                print(f"      {DIM}- {rm}{RESET}")
        print()

    # Prompt type
    prompt_options = [
        f"{BOLD}General{RESET}    -- Explain backpropagation (tests reasoning + length)",
        f"{BOLD}Coding{RESET}     -- Write a BST implementation (tests code generation)",
        f"{BOLD}Reasoning{RESET}  -- Logic puzzles (tests step-by-step thinking)",
        f"{BOLD}Creative{RESET}   -- Short story writing (tests fluency + creativity)",
    ]
    pidx = prompt_choice("Select benchmark prompt:", prompt_options)
    if pidx is None:
        return
    prompt_types = ["default", "coding", "reasoning", "creative"]
    prompt_type = prompt_types[pidx]

    # Number of runs
    num_runs_str = prompt_input("Number of runs (first is warmup)", "3")
    try:
        num_runs = max(1, int(num_runs_str))
    except ValueError:
        num_runs = 3

    # Run benchmarks on each detected runtime
    all_results: dict[str, list[BenchmarkResult]] = {}

    for rt in runtimes:
        # Find the best model match on this runtime
        model_id = _match_model_on_runtime(m, rt)
        if not model_id:
            print(f"  {YELLOW}Skipping {rt.name}: model '{m.name}' not found on this runtime.{RESET}")
            continue

        print(f"\n  {runtime_badge(rt.name)} Benchmarking {BOLD}{model_id}{RESET}...")

        def progress(run, total):
            label = "warmup" if run == 1 and total > 1 else f"run {run}"
            print(f"    {DIM}[{label}/{total}]{RESET}", end=" ", flush=True)

        results = run_benchmark_suite(
            rt, model_id, prompt_type, num_runs,
            progress_callback=progress,
        )
        print()
        all_results[rt.name] = results

    # Also try llama.cpp CLI for GGUF files
    if m.format == "gguf" and m.source != "ollama":
        print(f"\n  {runtime_badge('llama.cpp-cli')} Benchmarking {BOLD}{m.name}{RESET}...")

        cli_results = []
        for i in range(1, num_runs + 1):
            label = "warmup" if i == 1 and num_runs > 1 else f"run {i}"
            print(f"    {DIM}[{label}/{num_runs}]{RESET}", end=" ", flush=True)
            r = benchmark_llama_cpp_native(str(m.path), BENCH_PROMPTS[prompt_type], run_number=i)
            cli_results.append(r)
        print()
        if cli_results:
            all_results["llama.cpp-cli"] = cli_results

    if not all_results:
        print(f"\n  {RED}No benchmarks could be run.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    # Display results
    _display_benchmark_comparison(all_results, num_runs)
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _match_model_on_runtime(local: LocalModel, runtime: RuntimeInfo) -> str | None:
    """Try to match a local model to one available on a runtime."""
    name_lower = local.name.lower()
    # Direct match
    for rm in runtime.models:
        if rm.lower() == name_lower:
            return rm
    # Substring match
    for rm in runtime.models:
        if name_lower in rm.lower() or rm.lower() in name_lower:
            return rm
    # For Ollama, try matching family + size
    if local.family and local.metadata.get("parameter_size"):
        for rm in runtime.models:
            if local.family.lower() in rm.lower():
                return rm
    # If only one model loaded, use it
    if len(runtime.models) == 1:
        return runtime.models[0]
    return None


def _display_benchmark_comparison(all_results: dict[str, list[BenchmarkResult]], num_runs: int):
    """Display a rich comparison of benchmark results across runtimes."""
    print(f"\n  {BOLD}{'═' * 65}{RESET}")
    print(f"  {BOLD}  BENCHMARK RESULTS{RESET}")
    print(f"  {BOLD}{'═' * 65}{RESET}")

    # Per-run detail
    if num_runs > 1:
        print(f"\n  {BOLD}Per-Run Detail:{RESET}\n")
        header = f"  {'Runtime':<18} {'Run':>4} {'Gen Speed':>12} {'Prefill':>12} {'TTFT':>10} {'Tokens':>7}"
        print(f"  {BOLD}{header[2:]}{RESET}")
        print_separator()

        for runtime_name, results in all_results.items():
            for r in results:
                if r.success:
                    run_label = f"{'W' if r.run_number == 1 and num_runs > 1 else r.run_number}"
                    print(
                        f"  {runtime_name:<18} {run_label:>4} "
                        f"{color_speed(r.tokens_per_second):>12} "
                        f"{r.prompt_tokens_per_second:>8.0f} tok/s "
                        f"{r.time_to_first_token_ms:>7.0f} ms "
                        f"{r.generated_tokens:>7}"
                    )
                else:
                    run_label = f"{'W' if r.run_number == 1 and num_runs > 1 else r.run_number}"
                    print(f"  {runtime_name:<18} {run_label:>4} {RED}FAILED: {r.error}{RESET}")

    # Aggregated comparison
    print(f"\n  {BOLD}Summary (excluding warmup):{RESET}\n")
    aggregates = {}
    for runtime_name, results in all_results.items():
        agg = aggregate_results(results, skip_first=(num_runs > 1))
        aggregates[runtime_name] = agg

    header = f"  {'Runtime':<18} {'Avg Gen':>12} {'Min':>10} {'Max':>10} {'Avg Prefill':>13} {'Avg TTFT':>10}"
    print(f"  {BOLD}{header[2:]}{RESET}")
    print_separator()

    for runtime_name, agg in aggregates.items():
        if not agg["success"]:
            print(f"  {runtime_name:<18} {RED}All runs failed{RESET}")
            continue
        print(
            f"  {runtime_name:<18} "
            f"{color_speed(agg['avg_tok_per_sec']):>12} "
            f"{agg['min_tok_per_sec']:>7.1f} t/s "
            f"{agg['max_tok_per_sec']:>7.1f} t/s "
            f"{agg['avg_prefill_tok_per_sec']:>9.0f} tok/s "
            f"{agg['avg_ttft_ms']:>7.0f} ms"
        )

    # Winner
    successful = {k: v for k, v in aggregates.items() if v.get("success")}
    if len(successful) >= 2:
        best_gen = max(successful.items(), key=lambda x: x[1]["avg_tok_per_sec"])
        best_prefill = max(successful.items(), key=lambda x: x[1]["avg_prefill_tok_per_sec"])
        best_ttft = min(successful.items(), key=lambda x: x[1]["avg_ttft_ms"])

        print(f"\n  {BOLD}Winner:{RESET}")
        print(f"    Fastest generation:  {runtime_badge(best_gen[0])} {GREEN}{best_gen[1]['avg_tok_per_sec']:.1f} tok/s{RESET}")
        print(f"    Fastest prefill:     {runtime_badge(best_prefill[0])} {GREEN}{best_prefill[1]['avg_prefill_tok_per_sec']:.0f} tok/s{RESET}")
        print(f"    Lowest TTFT:         {runtime_badge(best_ttft[0])} {GREEN}{best_ttft[1]['avg_ttft_ms']:.0f} ms{RESET}")

        # Overall recommendation
        winner_name = best_gen[0]
        second = sorted(successful.items(), key=lambda x: x[1]["avg_tok_per_sec"], reverse=True)
        if len(second) >= 2:
            speedup = best_gen[1]["avg_tok_per_sec"] / second[1][1]["avg_tok_per_sec"]
            if speedup > 1.05:
                print(f"\n    {GREEN}{BOLD}{winner_name}{RESET} is {GREEN}{BOLD}{speedup:.1f}x faster{RESET} for generation")
            else:
                print(f"\n    {DIM}Both runtimes perform similarly{RESET}")
    elif len(successful) == 1:
        name, agg = next(iter(successful.items()))
        print(f"\n  {BOLD}Result:{RESET} {runtime_badge(name)} {GREEN}{agg['avg_tok_per_sec']:.1f} tok/s{RESET}")


# ── Screen: Search & Download ────────────────────────────────────────────────

def screen_search_and_download(mac: MacSpec):
    """Search HuggingFace / Ollama for models with download instructions."""
    print(f"\n  {BOLD}{CYAN}Find & Download Models{RESET}")

    source_options = [
        f"{BOLD}GGUF models{RESET}     -- For llama.cpp / LM Studio (HuggingFace)",
        f"{BOLD}MLX models{RESET}      -- For Apple MLX framework (HuggingFace)",
        f"{BOLD}Ollama models{RESET}   -- One-command install via Ollama",
        f"{BOLD}All HuggingFace{RESET} -- Search everything",
    ]
    sidx = prompt_choice("Where to search?", source_options)
    if sidx is None:
        return

    query = prompt_input("Search query (e.g. 'llama 8b', 'deepseek coder', 'qwen 32b')")
    if not query:
        return

    if sidx == 2:
        _search_ollama(mac, query)
        return

    print(f"\n  {DIM}Searching HuggingFace...{RESET}\n")

    if sidx == 0:
        results = search_gguf_models(query)
    elif sidx == 1:
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

    # Select a model for details / download
    midx = prompt_choice("Select model for details + download commands:", [m.repo_id for m in results])
    if midx is None:
        return

    _show_model_download(mac, results[midx])


def _search_ollama(mac: MacSpec, query: str):
    """Search for Ollama models (uses web search since Ollama has no search API)."""
    print(f"\n  {BOLD}Ollama Model Suggestions for: {query}{RESET}\n")

    # Ollama doesn't have a search API, so we provide curated suggestions
    # based on common model names
    suggestions = _get_ollama_suggestions(query)

    if not suggestions:
        print(f"  {YELLOW}No matching Ollama models found for '{query}'.{RESET}")
        print(f"  {DIM}Try browsing https://ollama.com/library{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    for i, (name, desc, sizes) in enumerate(suggestions, 1):
        print(f"  {CYAN}{i:>3}{RESET}) {BOLD}{name}{RESET}")
        print(f"       {desc}")
        print(f"       Sizes: {', '.join(sizes)}")

    midx = prompt_choice("Select model:", [s[0] for s in suggestions])
    if midx is None:
        return

    name, desc, sizes = suggestions[midx]

    # Pick a size
    if len(sizes) > 1:
        sidx = prompt_choice("Select size:", sizes)
        if sidx is None:
            return
        tag = sizes[sidx]
    else:
        tag = sizes[0]

    model_tag = f"{name}:{tag}" if tag != "latest" else name

    print(f"\n  {BOLD}Download & Run:{RESET}")
    print(f"\n  {GREEN}  ollama pull {model_tag}{RESET}")
    print(f"  {GREEN}  ollama run {model_tag}{RESET}")
    print()
    print(f"  {DIM}Or use via API (after pulling):{RESET}")
    print(f"  {DIM}  curl http://localhost:11434/api/generate -d '{{\"model\":\"{model_tag}\",\"prompt\":\"Hello\"}}'")
    print(f"{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _get_ollama_suggestions(query: str) -> list[tuple[str, str, list[str]]]:
    """Return Ollama model suggestions based on query. (name, description, available_sizes)."""
    catalog = [
        ("llama3.2", "Meta Llama 3.2 -- latest small models", ["1b", "3b"]),
        ("llama3.1", "Meta Llama 3.1 -- powerful open model", ["8b", "70b", "405b"]),
        ("llama3.3", "Meta Llama 3.3 -- latest 70B model", ["70b"]),
        ("qwen2.5", "Alibaba Qwen 2.5 -- strong multilingual model", ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"]),
        ("qwen2.5-coder", "Qwen 2.5 Coder -- specialized for code", ["0.5b", "1.5b", "3b", "7b", "14b", "32b"]),
        ("deepseek-r1", "DeepSeek R1 -- reasoning model distills", ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"]),
        ("deepseek-v3", "DeepSeek V3 -- MoE model", ["671b"]),
        ("mistral", "Mistral 7B -- fast and capable", ["7b"]),
        ("mixtral", "Mixtral MoE -- strong MoE model", ["8x7b", "8x22b"]),
        ("mistral-small", "Mistral Small 24B", ["24b"]),
        ("phi4", "Microsoft Phi-4 14B -- small but powerful", ["14b"]),
        ("phi3", "Microsoft Phi-3 -- efficient small models", ["mini", "medium"]),
        ("gemma2", "Google Gemma 2", ["2b", "9b", "27b"]),
        ("codellama", "Meta CodeLlama -- code-specialized", ["7b", "13b", "34b"]),
        ("starcoder2", "StarCoder 2 -- code model", ["3b", "7b", "15b"]),
        ("command-r", "Cohere Command-R -- RAG optimized", ["35b"]),
        ("command-r-plus", "Cohere Command-R+ -- large RAG model", ["104b"]),
        ("yi", "Yi 1.5 -- strong bilingual model", ["6b", "9b", "34b"]),
        ("falcon", "TII Falcon", ["7b", "40b"]),
        ("nous-hermes2", "Nous Hermes 2 -- fine-tuned Mistral", ["7b"]),
    ]
    q = query.lower()
    matches = []
    for name, desc, sizes in catalog:
        if (q in name.lower() or q in desc.lower()
                or any(q in s for s in sizes)
                or any(word in name.lower() for word in q.split())):
            matches.append((name, desc, sizes))
    return matches


def _show_model_download(mac: MacSpec, model: HubModel):
    """Show download instructions for a HuggingFace model."""
    print(f"\n  {BOLD}{'═' * 60}{RESET}")
    print(f"  {BOLD}{model.repo_id}{RESET}")
    print(f"  {BOLD}{'═' * 60}{RESET}\n")

    print(f"  {BOLD}Author:{RESET}     {model.author}")
    print(f"  {BOLD}Downloads:{RESET}  {format_downloads(model.downloads)}")
    print(f"  {BOLD}Likes:{RESET}      {model.likes}")
    if model.pipeline_tag:
        print(f"  {BOLD}Task:{RESET}       {model.pipeline_tag}")

    tags = []
    if model.is_gguf:
        tags.append(f"{CYAN}GGUF{RESET}")
    if model.is_mlx:
        tags.append(f"{GREEN}MLX{RESET}")
    if tags:
        print(f"  {BOLD}Format:{RESET}     {' '.join(tags)}")

    # Fetch file list
    print(f"\n  {DIM}Fetching file list...{RESET}")
    files = get_model_files(model.repo_id)
    gguf_files = [f for f in files if f["filename"].endswith(".gguf")]
    safetensor_files = [f for f in files if f["filename"].endswith(".safetensors")]

    if gguf_files:
        print(f"\n  {BOLD}Available GGUF files:{RESET}")
        for f in sorted(gguf_files, key=lambda x: x.get("size") or 0):
            size_str = ""
            if f.get("size"):
                size_gb = f["size"] / (1024**3)
                fits = size_gb * 1.15 <= mac.usable_memory_gb
                mark = f"{GREEN}✓{RESET}" if fits else f"{RED}✗{RESET}"
                size_str = f"  {format_size(size_gb)} {mark}"
            print(f"    {f['filename']}{size_str}")

    # Download instructions
    print(f"\n  {BOLD}Download Commands:{RESET}\n")

    if model.is_gguf:
        print(f"  {UNDERLINE}For LM Studio:{RESET}")
        print(f"  {GREEN}  Search for '{model.repo_id}' in LM Studio's model browser{RESET}")
        print()
        print(f"  {UNDERLINE}For llama.cpp (huggingface-cli):{RESET}")
        if gguf_files:
            # Suggest Q4_K_M file if available
            q4_file = next((f for f in gguf_files if "Q4_K_M" in f["filename"]), gguf_files[0])
            print(f"  {GREEN}  huggingface-cli download {model.repo_id} {q4_file['filename']}{RESET}")
        else:
            print(f"  {GREEN}  huggingface-cli download {model.repo_id}{RESET}")
        print()
        print(f"  {UNDERLINE}For llama.cpp (wget):{RESET}")
        if gguf_files:
            q4_file = next((f for f in gguf_files if "Q4_K_M" in f["filename"]), gguf_files[0])
            print(f"  {GREEN}  wget https://huggingface.co/{model.repo_id}/resolve/main/{q4_file['filename']}{RESET}")

    if model.is_mlx:
        print(f"\n  {UNDERLINE}For MLX:{RESET}")
        print(f"  {GREEN}  pip install mlx-lm{RESET}")
        print(f"  {GREEN}  python -m mlx_lm.generate --model {model.repo_id} --prompt \"Hello\"{RESET}")
        print()
        print(f"  {UNDERLINE}Or download for offline use:{RESET}")
        print(f"  {GREEN}  huggingface-cli download {model.repo_id}{RESET}")

    if not model.is_gguf and not model.is_mlx:
        print(f"  {GREEN}  huggingface-cli download {model.repo_id}{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: Benchmark (standalone) ───────────────────────────────────────────

def screen_benchmark(mac: MacSpec):
    """Standalone benchmark screen -- pick runtimes and models to compare."""
    print(f"\n  {BOLD}{CYAN}Benchmark Arena{RESET}")
    print(f"  {DIM}Compare LM Studio vs llama.cpp vs Ollama head-to-head{RESET}\n")

    runtimes = detect_all_runtimes()

    if not runtimes:
        print(f"  {YELLOW}No runtimes detected.{RESET}\n")
        print(f"  To benchmark, start at least one of these:")
        print(f"    {BOLD}Ollama:{RESET}       brew install ollama && ollama serve")
        print(f"    {BOLD}LM Studio:{RESET}    Start local server from the app (port 1234)")
        print(f"    {BOLD}llama.cpp:{RESET}    llama-server -m model.gguf --port 8080")
        print()
        print(f"  {DIM}Tip: Start multiple runtimes with the same model loaded")
        print(f"  to get a direct comparison.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    print(f"  {BOLD}Detected runtimes:{RESET}\n")
    for rt in runtimes:
        ver = f" v{rt.version}" if rt.version else ""
        models_str = ", ".join(rt.models[:3])
        if len(rt.models) > 3:
            models_str += f" (+{len(rt.models) - 3} more)"
        print(f"    {runtime_badge(rt.name)}{ver}")
        print(f"      {DIM}Models: {models_str}{RESET}")
    print()

    # Build unified model list across runtimes
    all_model_ids: list[tuple[RuntimeInfo, str]] = []
    model_display: list[str] = []
    for rt in runtimes:
        for mid in rt.models:
            all_model_ids.append((rt, mid))
            model_display.append(f"{runtime_badge(rt.name)} {mid}")

    # Also check for local GGUF files
    local = scan_all()
    gguf_local = [m for m in local if m.format == "gguf" and m.source != "ollama"]
    for m in gguf_local[:10]:
        all_model_ids.append((None, str(m.path)))  # type: ignore
        model_display.append(f"{runtime_badge('llama.cpp-cli')} {m.name} ({format_size(m.size_gb)})")

    midx = prompt_choice("Select model to benchmark:", model_display)
    if midx is None:
        return

    selected_rt, selected_model = all_model_ids[midx]

    # Prompt type
    prompt_options = [
        f"{BOLD}General{RESET}    -- Explain backpropagation",
        f"{BOLD}Coding{RESET}     -- Write a BST implementation",
        f"{BOLD}Reasoning{RESET}  -- Logic puzzles",
        f"{BOLD}Creative{RESET}   -- Short story writing",
    ]
    pidx = prompt_choice("Benchmark prompt:", prompt_options)
    if pidx is None:
        return
    prompt_type = ["default", "coding", "reasoning", "creative"][pidx]

    num_runs_str = prompt_input("Number of runs (first is warmup)", "3")
    try:
        num_runs = max(1, int(num_runs_str))
    except ValueError:
        num_runs = 3

    all_results: dict[str, list[BenchmarkResult]] = {}

    if selected_rt is None:
        # Local GGUF via CLI
        print(f"\n  {runtime_badge('llama.cpp-cli')} Benchmarking...")
        cli_results = []
        for i in range(1, num_runs + 1):
            label = "warmup" if i == 1 and num_runs > 1 else f"run {i}"
            print(f"    {DIM}[{label}/{num_runs}]{RESET}", end=" ", flush=True)
            r = benchmark_llama_cpp_native(selected_model, BENCH_PROMPTS[prompt_type], run_number=i)
            cli_results.append(r)
        print()
        all_results["llama.cpp-cli"] = cli_results
    else:
        # Run on the selected runtime
        print(f"\n  {runtime_badge(selected_rt.name)} Benchmarking {BOLD}{selected_model}{RESET}...")

        def progress(run, total):
            label = "warmup" if run == 1 and total > 1 else f"run {run}"
            print(f"    {DIM}[{label}/{total}]{RESET}", end=" ", flush=True)

        results = run_benchmark_suite(selected_rt, selected_model, prompt_type, num_runs, progress_callback=progress)
        print()
        all_results[selected_rt.name] = results

        # Offer to also test on other runtimes with same/similar model
        other_runtimes = [rt for rt in runtimes if rt.name != selected_rt.name]
        for ort in other_runtimes:
            # Try to find matching model
            matched = None
            for rm in ort.models:
                if (rm.lower() == selected_model.lower()
                        or selected_model.lower() in rm.lower()
                        or rm.lower() in selected_model.lower()):
                    matched = rm
                    break
            if not matched and len(ort.models) == 1:
                matched = ort.models[0]
            if matched:
                if prompt_yes_no(f"  Also benchmark on {ort.name} ({matched})?"):
                    print(f"\n  {runtime_badge(ort.name)} Benchmarking {BOLD}{matched}{RESET}...")
                    results = run_benchmark_suite(ort, matched, prompt_type, num_runs, progress_callback=progress)
                    print()
                    all_results[ort.name] = results

    _display_benchmark_comparison(all_results, num_runs)
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: Detailed Analysis ────────────────────────────────────────────────

def screen_detail(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Detailed Model Analysis{RESET}")

    model_names = [f"{m.name} ({m.params_billion}B)" for m in MODELS]
    idx = prompt_choice("Select model:", model_names)
    if idx is None:
        return
    model = MODELS[idx]

    quant_options = [
        f"{q} -- {QUANT_FORMATS[q].bits_per_weight:.1f} bpw ({QUANT_FORMATS[q].quality_rating})"
        for q in QUANT_FORMATS
    ]
    qidx = prompt_choice("Select quantization:", quant_options)
    if qidx is None:
        return
    quant_name = list(QUANT_FORMATS.keys())[qidx]
    quant = QUANT_FORMATS[quant_name]

    result = analyze_model(mac, model, quant)
    max_ctx = max_context_for_model(mac, model, quant)

    print(f"\n  {BOLD}{'═' * 60}{RESET}")
    print(f"  {BOLD}{model.name} @ {quant.name}{RESET}")
    print(f"  {BOLD}{'═' * 60}{RESET}\n")

    print(f"  {BOLD}Model:{RESET}       {model.params_billion}B params | {model.family} | {model.license}")
    print(f"  {BOLD}Context:{RESET}     {format_context(model.default_context_length)} default, {format_context(model.max_context_length)} max")
    print(f"  {BOLD}Quant:{RESET}       {quant.name} ({quant.bits_per_weight:.2f} bpw, {quant.quality_rating})")
    print(f"  {DIM}             {quant.description}{RESET}")

    print(f"\n  {BOLD}Memory Breakdown:{RESET}")
    print(f"    Model weights:  {format_size(result.model_size_gb):>10}")
    print(f"    KV cache:       {format_size(result.kv_cache_gb):>10}  (at {format_context(result.context_length)} ctx)")
    print(f"    Overhead:       {format_size(result.overhead_gb):>10}")
    print(f"    {'─' * 28}")
    print(f"    {BOLD}Total:          {format_size(result.total_memory_gb):>10}{RESET}")
    print(f"    Available:      {format_size(result.available_memory_gb):>10}")

    print(f"\n  {BOLD}Memory:{RESET}  {bar(result.total_memory_gb, result.available_memory_gb, 35)}")
    print(f"  {BOLD}Rating:{RESET}  {color_rating(result.rating)}")

    if result.estimated_tok_per_sec:
        print(f"  {BOLD}Speed:{RESET}   {color_speed(result.estimated_tok_per_sec)} (estimated)")

    # Context scaling
    effective_max = min(max_ctx, model.max_context_length)
    print(f"\n  {BOLD}Max context:{RESET} {format_context(effective_max)} tokens")
    if max_ctx > model.max_context_length:
        print(f"  {DIM}(Memory allows {format_context(max_ctx)}, architecture caps at {format_context(model.max_context_length)}){RESET}")

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    overhead = estimate_overhead_gb(model_size)

    print(f"\n  {BOLD}Context Scaling:{RESET}")
    print(f"    {'Context':>10} {'KV Cache':>10} {'Total':>10} {'Fits':>6}")
    print(f"    {'─' * 40}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if ctx > model.max_context_length:
            break
        kv = estimate_kv_cache_gb(model, ctx)
        total = model_size + kv + overhead
        fits = total <= mac.usable_memory_gb
        mark = f"{GREEN}✓{RESET}" if fits else f"{RED}✗{RESET}"
        print(f"    {format_context(ctx):>10} {format_size(kv):>10} {format_size(total):>10} {mark:>6}")

    # Download hints
    print(f"\n  {BOLD}How to get this model:{RESET}")
    name_slug = model.name.lower().replace(" ", "-").replace(".", "")
    print(f"    Ollama:     {GREEN}ollama pull {name_slug}{RESET}")
    print(f"    LM Studio:  {DIM}Search '{model.name}' in the model browser{RESET}")
    print(f"    HuggingFace:{DIM} Search '{model.name} GGUF' on huggingface.co{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Main Loop ────────────────────────────────────────────────────────────────

def main_interactive():
    """Main interactive loop."""
    clear_screen()
    print_banner()

    mac = select_hardware()
    if not mac:
        print(f"\n  {RED}No hardware selected. Exiting.{RESET}")
        sys.exit(0)

    while True:
        # Quick status line
        runtimes = detect_all_runtimes()
        rt_str = " | ".join(runtime_badge(r.name) for r in runtimes) if runtimes else f"{DIM}no runtimes detected{RESET}"

        print(f"\n  {BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"  {BOLD}{CYAN}  {mac.chip} {mac.memory_gb}GB{RESET}  {DIM}|{RESET}  {rt_str}")
        print(f"  {BOLD}{CYAN}{'═' * 60}{RESET}")

        menu = [
            f"{BOLD}My Models{RESET}               -- Browse & analyze downloaded models",
            f"{BOLD}Find & Download{RESET}         -- Search HuggingFace, Ollama, MLX",
            f"{BOLD}Compatibility Check{RESET}     -- Which models can I run?",
            f"{BOLD}Detailed Analysis{RESET}       -- Deep dive on any model",
            f"{BOLD}Benchmark Arena{RESET}         -- LM Studio vs llama.cpp vs Ollama",
            f"{BOLD}Change Hardware{RESET}         -- Switch Mac configuration",
            f"{RED}Exit{RESET}",
        ]

        idx = prompt_choice("What would you like to do?", menu, allow_back=False)

        if idx is None or idx == 6:
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        elif idx == 0:
            screen_my_models(mac)
        elif idx == 1:
            screen_search_and_download(mac)
        elif idx == 2:
            screen_check_models(mac)
        elif idx == 3:
            screen_detail(mac)
        elif idx == 4:
            screen_benchmark(mac)
        elif idx == 5:
            new_mac = select_hardware()
            if new_mac:
                mac = new_mac
