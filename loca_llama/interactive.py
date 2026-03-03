"""Interactive terminal UI for Loca-LLAMA — LM Studio + llama.cpp focused."""

import sys

from .hardware import APPLE_SILICON_SPECS, MacSpec, detect_mac
from .models import MODELS
from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS
from .analyzer import (
    CompatibilityTier, analyze_model, max_context_for_model,
    estimate_model_size_gb, estimate_kv_cache_gb, estimate_overhead_gb,
)
from .scanner import scan_all, scan_custom_dir, LocalModel
from .hub import (
    search_huggingface, search_gguf_models, search_mlx_models,
    get_model_files, format_downloads, HubModel,
)
from .benchmark import (
    detect_all_runtimes, RuntimeInfo, BenchmarkResult,
    run_benchmark_suite, aggregate_results,
    benchmark_llama_cpp_native, BENCH_PROMPTS,
)
from .templates import get_template, get_lm_studio_preset, get_llama_cpp_command, get_llama_cpp_server_command
from .memory_monitor import (
    MemoryMonitor, MemoryReport, get_memory_sample,
    memory_bar, format_memory_report, format_mini_memory_bar, pressure_badge,
)
from .hf_templates import fetch_hf_model_config, resolve_repo_id, format_hf_config

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
{BOLD}{MAGENTA}  +===============================================================+
  |                                                               |
  |   {CYAN}██╗      ██████╗  ██████╗ █████╗                            {MAGENTA}|
  |   {CYAN}██║     ██╔═══██╗██╔════╝██╔══██╗                           {MAGENTA}|
  |   {CYAN}██║     ██║   ██║██║     ███████║                           {MAGENTA}|
  |   {CYAN}██║     ██║   ██║██║     ██╔══██║                           {MAGENTA}|
  |   {CYAN}███████╗╚██████╔╝╚██████╗██║  ██║                           {MAGENTA}|
  |   {CYAN}╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝                          {MAGENTA}|
  |               {WHITE}Local LLM Apple Mac Analyzer{MAGENTA}                    |
  |                                                               |
  +===============================================================+{RESET}
""")

# ── Input helpers ────────────────────────────────────────────────────────────

def prompt_choice(prompt: str, options: list[str], allow_back: bool = True) -> int | None:
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


def tier_badge(tier: CompatibilityTier) -> str:
    badges = {
        CompatibilityTier.FULL_GPU: f"{BG_GREEN}{WHITE}{BOLD} FULL GPU {RESET}",
        CompatibilityTier.COMFORTABLE: f"{GREEN}{BOLD}[Comfortable]{RESET}",
        CompatibilityTier.TIGHT_FIT: f"{YELLOW}{BOLD}[Tight Fit]{RESET}",
        CompatibilityTier.PARTIAL_OFFLOAD: f"{YELLOW}[Partial Offload]{RESET}",
        CompatibilityTier.WONT_FIT: f"{RED}[Won't Fit]{RESET}",
    }
    return badges[tier]


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
    }
    return badges.get(name, name)


def print_separator(title: str = "", char: str = "-", width: int = 65):
    if title:
        left = 3
        right = width - left - len(title) - 2
        print(f"  {char * left} {BOLD}{title}{RESET} {char * max(right, 1)}")
    else:
        print(f"  {char * width}")


# ── Hardware Selection ───────────────────────────────────────────────────────

def select_hardware() -> MacSpec | None:
    # Try auto-detect first
    detected = detect_mac()
    if detected:
        key, mac = detected
        print(f"\n  {GREEN}Detected:{RESET} {BOLD}{key}{RESET}")
        print(f"    {mac.cpu_cores} CPU cores, {mac.gpu_cores} GPU cores, "
              f"{mac.memory_gb} GB unified, {mac.memory_bandwidth_gbs:.0f} GB/s")
        if prompt_yes_no("  Use detected config?", default=True):
            return mac
        print()

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
    print(f"\n  {GREEN}OK{RESET} Selected: {BOLD}{selected}{RESET}")
    print(f"    {mac.cpu_cores} CPU cores, {mac.gpu_cores} GPU cores, "
          f"{mac.memory_gb} GB unified, {mac.memory_bandwidth_gbs:.0f} GB/s")
    return mac


# ── Screen: Compatibility Check with VRAM Tiers ─────────────────────────────

def screen_check_models(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Model Compatibility Check{RESET}")
    print(f"  {DIM}Shows which models fit fully, comfortably, or with partial offloading{RESET}")

    quant_options = [
        f"{QUANT_FORMATS[q].name} -- {QUANT_FORMATS[q].bits_per_weight:.1f} bpw ({QUANT_FORMATS[q].quality_rating})"
        for q in RECOMMENDED_FORMATS
    ]
    quant_options.append("All recommended formats at once")
    idx = prompt_choice("Select quantization:", quant_options)
    if idx is None:
        return
    if idx == len(RECOMMENDED_FORMATS):
        quant_names = RECOMMENDED_FORMATS
    else:
        quant_names = [RECOMMENDED_FORMATS[idx]]

    families = sorted(set(m.family for m in MODELS))
    family_options = ["All families"] + families
    fidx = prompt_choice("Filter by model family:", family_options)
    if fidx is None:
        return
    models = MODELS if fidx == 0 else [m for m in MODELS if m.family == families[fidx - 1]]

    print(f"\n  {DIM}Analyzing {len(models)} models x {len(quant_names)} quant(s)...{RESET}")

    results = []
    for model in models:
        for qn in quant_names:
            results.append(analyze_model(mac, model, QUANT_FORMATS[qn]))

    by_tier: dict[CompatibilityTier, list] = {}
    for r in results:
        by_tier.setdefault(r.tier, []).append(r)

    for tier in [CompatibilityTier.FULL_GPU, CompatibilityTier.COMFORTABLE,
                 CompatibilityTier.TIGHT_FIT, CompatibilityTier.PARTIAL_OFFLOAD,
                 CompatibilityTier.WONT_FIT]:
        group = by_tier.get(tier, [])
        if not group:
            continue

        group.sort(key=lambda r: r.model.params_billion, reverse=True)
        print(f"\n  {tier_badge(tier)} ({len(group)})\n")

        name_w = max(len(r.model.name) for r in group) + 1
        for r in group:
            speed_str = color_speed(r.estimated_tok_per_sec) if r.estimated_tok_per_sec else f"{DIM}n/a{RESET}"
            offload_str = ""
            if r.tier == CompatibilityTier.PARTIAL_OFFLOAD and r.offload_pct:
                offload_str = f" {DIM}({r.gpu_layers}/{r.total_layers} layers GPU){RESET}"

            if r.tier == CompatibilityTier.WONT_FIT:
                over = r.total_memory_gb - r.available_memory_gb
                print(
                    f"    {DIM}{r.model.name:<{name_w}} {r.quant.name:<8} "
                    f"{format_size(r.total_memory_gb):>8} -- needs {format_size(over)} more{RESET}"
                )
            else:
                max_ctx = max_context_for_model(mac, r.model, r.quant)
                print(
                    f"    {r.model.name:<{name_w}} {CYAN}{r.quant.name:<8}{RESET} "
                    f"{format_size(r.total_memory_gb):>8} "
                    f"{bar(r.total_memory_gb, r.available_memory_gb)} "
                    f"{speed_str:>16}  "
                    f"ctx: {format_context(max_ctx):>6}"
                    f"{offload_str}"
                )

    usable = sum(1 for r in results if r.is_usable)
    full = sum(1 for r in results if r.is_fully_gpu)
    partial = sum(1 for r in results if r.tier == CompatibilityTier.PARTIAL_OFFLOAD)
    print(f"\n  {GREEN}{full}{RESET} fully on GPU | {YELLOW}{partial}{RESET} partial offload | {usable}/{len(results)} usable")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: My Models ────────────────────────────────────────────────────────

def screen_my_models(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}My Downloaded Models{RESET}")
    print(f"  {DIM}Scanning LM Studio, llama.cpp, HuggingFace cache...{RESET}\n")

    models = scan_all()

    if not models:
        print(f"  {YELLOW}No models found in standard locations.{RESET}")
        print(f"  {DIM}Looked in:{RESET}")
        print(f"  {DIM}  - ~/.cache/lm-studio/models{RESET}")
        print(f"  {DIM}  - ~/.lmstudio/models{RESET}")
        print(f"  {DIM}  - ~/llama.cpp/models{RESET}")
        print(f"  {DIM}  - ~/.cache/huggingface/hub{RESET}")
        print()
        custom = prompt_input("Enter a custom directory to scan (or empty to skip)")
        if custom:
            models = scan_custom_dir(custom)
        if not models:
            print(f"  {RED}No model files found.{RESET}")
            input(f"\n  {DIM}Press Enter to continue...{RESET}")
            return

    by_source: dict[str, list[LocalModel]] = {}
    for m in models:
        by_source.setdefault(m.source, []).append(m)

    sources_summary = ", ".join(f"{source_badge(s)} x{len(ms)}" for s, ms in by_source.items())
    print(f"  Found {BOLD}{len(models)}{RESET} models: {sources_summary}\n")

    display_models: list[LocalModel] = []
    for source in ["lm-studio", "llama.cpp", "mlx-community", "huggingface", "custom"]:
        group = by_source.get(source, [])
        if not group:
            continue
        print(f"  {source_badge(source)}")
        for m in group:
            idx = len(display_models) + 1
            quant_str = f" {YELLOW}[{m.quant}]{RESET}" if m.quant else ""
            family_str = f" {DIM}({m.family}){RESET}" if m.family else ""
            print(
                f"    {CYAN}{idx:>3}{RESET}) {BOLD}{m.name}{RESET}{quant_str}{family_str}"
                f"  {DIM}{format_size(m.size_gb)}{RESET}"
            )
            display_models.append(m)
        print()

    while True:
        actions = [
            f"{BOLD}Analyze a model{RESET}          -- Memory breakdown, VRAM tier, speed estimate",
            f"{BOLD}Get settings template{RESET}    -- Recommended config for LM Studio / llama.cpp",
            f"{BOLD}Fetch HF settings{RESET}       -- Download official settings from HuggingFace",
            f"{BOLD}Benchmark a model{RESET}        -- Test on LM Studio vs llama.cpp (with memory monitor)",
            f"{BOLD}Scan custom directory{RESET}    -- Look somewhere else",
        ]
        aidx = prompt_choice("What next?", actions)
        if aidx is None:
            return

        model_list_display = [f"{m.name} ({format_size(m.size_gb)})" for m in display_models]
        if aidx == 0:
            midx = prompt_choice("Select model #:", model_list_display)
            if midx is not None:
                _analyze_local_model(mac, display_models[midx])
        elif aidx == 1:
            midx = prompt_choice("Select model #:", model_list_display)
            if midx is not None:
                _show_template(mac, display_models[midx])
        elif aidx == 2:
            midx = prompt_choice("Select model #:", model_list_display)
            if midx is not None:
                _fetch_hf_settings(display_models[midx])
        elif aidx == 3:
            midx = prompt_choice("Select model to benchmark:", model_list_display)
            if midx is not None:
                _benchmark_local_model(mac, display_models[midx])
        elif aidx == 4:
            custom = prompt_input("Directory path")
            if custom:
                extra = scan_custom_dir(custom)
                if extra:
                    print(f"  {GREEN}Found {len(extra)} more models!{RESET}")
                    display_models.extend(extra)
                else:
                    print(f"  {YELLOW}No models found in that directory.{RESET}")


def _analyze_local_model(mac: MacSpec, m: LocalModel):
    print(f"\n  {BOLD}{'=' * 55}{RESET}")
    print(f"  {BOLD}{m.name}{RESET}")
    print(f"  {BOLD}{'=' * 55}{RESET}\n")

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

    total_estimated = m.size_gb * 1.15
    available = mac.usable_memory_gb
    pct = (total_estimated / available) * 100

    if pct <= 75:
        tier = CompatibilityTier.FULL_GPU
    elif pct <= 90:
        tier = CompatibilityTier.COMFORTABLE
    elif pct <= 100:
        tier = CompatibilityTier.TIGHT_FIT
    elif pct <= 150:
        tier = CompatibilityTier.PARTIAL_OFFLOAD
    else:
        tier = CompatibilityTier.WONT_FIT

    print(f"\n  {BOLD}VRAM Tier:{RESET}   {tier_badge(tier)}")
    print(f"\n  {BOLD}Memory Analysis:{RESET}")
    print(f"    Model file:     {format_size(m.size_gb):>10}")
    print(f"    Est. overhead:  {format_size(m.size_gb * 0.15):>10}  (KV cache + buffers @ 4K ctx)")
    print(f"    {'-' * 28}")
    print(f"    Est. total:     {format_size(total_estimated):>10}")
    print(f"    Available:      {format_size(available):>10}")
    print(f"\n  {BOLD}Memory:{RESET} {bar(total_estimated, available, 35)}")

    if tier != CompatibilityTier.WONT_FIT:
        tok_s = mac.memory_bandwidth_gbs / (m.size_gb * 1.1)
        if tier == CompatibilityTier.PARTIAL_OFFLOAD:
            gpu_frac = min(available * 0.85 / m.size_gb, 1.0)
            tok_s = tok_s * gpu_frac * 0.6
        print(f"  {BOLD}Est. speed:{RESET} {color_speed(tok_s)}")

        headroom = available - total_estimated
        if headroom > 0:
            approx_ctx_tokens = int(headroom * 1024 * 1024 * 1024 / (256 * 32 * 2))
            print(f"  {BOLD}Est. max ctx:{RESET} ~{format_context(min(approx_ctx_tokens, 131072))}")
    else:
        over = total_estimated - available
        print(f"  {RED}{BOLD}Does not fit.{RESET} Needs {format_size(over)} more RAM.")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _show_template(mac: MacSpec, m: LocalModel):
    tmpl = get_template(m.name)

    print(f"\n  {BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}Settings Template: {m.name}{RESET}")
    print(f"  {BOLD}{'=' * 60}{RESET}")

    # Try to fetch official settings from HuggingFace
    hf_config = None
    repo_id = m.repo_id
    if not repo_id:
        print(f"\n  {DIM}Looking up model on HuggingFace...{RESET}")
        repo_id = resolve_repo_id(m.name)

    if repo_id:
        print(f"  {DIM}Fetching official settings from {repo_id}...{RESET}")
        def hf_progress(step, total, desc):
            print(f"    {DIM}[{step}/{total}] {desc}{RESET}")
        hf_config = fetch_hf_model_config(repo_id, progress_callback=hf_progress)
        if hf_config and (hf_config.temperature is not None or hf_config.num_layers):
            print(f"\n{format_hf_config(hf_config)}")

    if not tmpl and not hf_config:
        print(f"\n  {YELLOW}No template found for '{m.name}'.{RESET}")
        print(f"  {DIM}Try with a model from a known family (Llama, Qwen, Mistral, etc.){RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    if tmpl:
        mem_gb = mac.memory_gb
        rec_quant = tmpl.quant_48gb if mem_gb >= 48 else (tmpl.quant_24gb if mem_gb <= 24 else tmpl.quant_48gb)

        print(f"\n  {BOLD}Recommended Quantization:{RESET}")
        print(f"    24GB Mac:  {tmpl.quant_24gb}")
        print(f"    48GB Mac:  {tmpl.quant_48gb}")
        print(f"    64GB+ Mac: {tmpl.quant_64gb}")
        print(f"    {GREEN}Your Mac ({mac.memory_gb}GB): {rec_quant}{RESET}")

        print(f"\n  {BOLD}Context Length:{RESET}")
        print(f"    Recommended: {format_context(tmpl.recommended_ctx)}")
        print(f"    Max practical: {format_context(tmpl.max_practical_ctx)}")

        # Merge HF settings with template (HF takes priority where available)
        temp = hf_config.temperature if (hf_config and hf_config.temperature is not None) else tmpl.temperature
        top_p = hf_config.top_p if (hf_config and hf_config.top_p is not None) else tmpl.top_p
        top_k = hf_config.top_k if (hf_config and hf_config.top_k is not None) else tmpl.top_k
        rep_pen = hf_config.repetition_penalty if (hf_config and hf_config.repetition_penalty is not None) else tmpl.repeat_penalty

        hf_tag = f" {GREEN}(from HuggingFace){RESET}"
        print(f"\n  {BOLD}Sampling Parameters:{RESET}")
        print(f"    Temperature:    {temp}{hf_tag if (hf_config and hf_config.temperature is not None) else ''}")
        print(f"    Top P:          {top_p}{hf_tag if (hf_config and hf_config.top_p is not None) else ''}")
        print(f"    Top K:          {top_k}{hf_tag if (hf_config and hf_config.top_k is not None) else ''}")
        print(f"    Min P:          {tmpl.min_p}")
        print(f"    Repeat Penalty: {rep_pen}{hf_tag if (hf_config and hf_config.repetition_penalty is not None) else ''}")

        sys_prompt = (hf_config.system_prompt if (hf_config and hf_config.system_prompt) else tmpl.system_prompt)
        if sys_prompt:
            print(f"\n  {BOLD}System Prompt:{RESET}")
            print(f"    {DIM}{sys_prompt}{RESET}")

        print(f"\n  {BOLD}Chat Template:{RESET} {tmpl.chat_template}")

        if tmpl.notes:
            print(f"\n  {BOLD}Notes:{RESET}")
            print(f"    {tmpl.notes}")

        if tmpl.bench_tok_s_q4:
            print(f"\n  {BOLD}Community Benchmark Reference (M4 Pro 48GB):{RESET}")
            if tmpl.bench_tok_s_q4:
                print(f"    Q4_K_M generation: ~{tmpl.bench_tok_s_q4:.0f} tok/s")
            if tmpl.bench_tok_s_q8:
                print(f"    Q8_0 generation:   ~{tmpl.bench_tok_s_q8:.0f} tok/s")
            if tmpl.bench_prefill_q4:
                print(f"    Q4_K_M prefill:    ~{tmpl.bench_prefill_q4:.0f} tok/s")

        print(f"\n  {BOLD}{MAGENTA}LM Studio Settings:{RESET}")
        preset = get_lm_studio_preset(tmpl, m.name)
        # Override with HF values if available
        if hf_config and hf_config.temperature is not None:
            preset["inference_params"]["temperature"] = hf_config.temperature
        if hf_config and hf_config.top_p is not None:
            preset["inference_params"]["top_p"] = hf_config.top_p
        if hf_config and hf_config.top_k is not None:
            preset["inference_params"]["top_k"] = hf_config.top_k
        if hf_config and hf_config.repetition_penalty is not None:
            preset["inference_params"]["repeat_penalty"] = hf_config.repetition_penalty
        print(f"    Context Length: {preset['context_length']}")
        print(f"    Temperature:    {preset['inference_params']['temperature']}")
        print(f"    Top P:          {preset['inference_params']['top_p']}")
        print(f"    Top K:          {preset['inference_params']['top_k']}")
        print(f"    Min P:          {preset['inference_params']['min_p']}")
        print(f"    Repeat Penalty: {preset['inference_params']['repeat_penalty']}")

        # Build sampling overrides from HF-merged values
        sampling_overrides = {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": rep_pen,
            "min_p": tmpl.min_p,
        }

        print(f"\n  {BOLD}{CYAN}llama.cpp Command:{RESET}")
        cmd = get_llama_cpp_command(tmpl, str(m.path), sampling_overrides=sampling_overrides)
        for line in cmd.split("\n"):
            print(f"    {GREEN}{line}{RESET}")

        print(f"\n  {BOLD}{CYAN}llama-server Command:{RESET}")
        srv_cmd = get_llama_cpp_server_command(tmpl, str(m.path))
        for line in srv_cmd.split("\n"):
            print(f"    {GREEN}{line}{RESET}")

    elif hf_config:
        # No static template, but we have HF data
        print(f"\n  {DIM}No built-in template, using HuggingFace defaults.{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _fetch_hf_settings(m: LocalModel):
    """Fetch and display official model settings from HuggingFace."""
    print(f"\n  {BOLD}{CYAN}Fetching HuggingFace Settings: {m.name}{RESET}\n")

    repo_id = m.repo_id
    if not repo_id:
        print(f"  {DIM}Resolving model on HuggingFace...{RESET}")
        repo_id = resolve_repo_id(m.name)

    if not repo_id:
        print(f"  {YELLOW}Could not find model on HuggingFace.{RESET}")
        manual = prompt_input("Enter HuggingFace repo ID (e.g. Qwen/Qwen2.5-32B-Instruct)")
        if manual:
            repo_id = manual
        else:
            input(f"\n  {DIM}Press Enter to continue...{RESET}")
            return

    def progress(step, total, desc):
        print(f"  {DIM}[{step}/{total}] {desc}{RESET}")

    config = fetch_hf_model_config(repo_id, progress_callback=progress)
    print(f"\n{format_hf_config(config)}")

    # Show raw generation_config.json if available
    if config._raw_generation_config:
        print(f"\n  {BOLD}Raw generation_config.json:{RESET}")
        import json
        for k, v in config._raw_generation_config.items():
            if k.startswith("_") or k == "transformers_version":
                continue
            print(f"    {BOLD}{k}:{RESET} {v}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _benchmark_local_model(mac: MacSpec, m: LocalModel):
    print(f"\n  {BOLD}{CYAN}Benchmark: {m.name}{RESET}\n")

    # Show current memory before benchmark
    sample = get_memory_sample()
    print(f"  {BOLD}Memory before benchmark:{RESET}")
    print(f"  {memory_bar(sample.used_gb, sample.total_gb, 35)}")
    if sample.pressure and sample.pressure != "unknown":
        print(f"  Pressure: {pressure_badge(sample.pressure)}")
    print()

    runtimes = detect_all_runtimes()

    if not runtimes and m.format != "gguf":
        print(f"  {RED}No runtimes detected and model is not GGUF.{RESET}")
        print(f"  {DIM}Start LM Studio or llama.cpp server first.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    if runtimes:
        print(f"  {BOLD}Detected runtimes:{RESET}")
        for rt in runtimes:
            print(f"    {runtime_badge(rt.name)}  {DIM}{rt.url}{RESET}")
            for rm in rt.models[:5]:
                print(f"      {DIM}- {rm}{RESET}")
        print()

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

    # Start memory monitoring
    monitor = MemoryMonitor(interval=0.5)
    monitor.start()

    all_results: dict[str, list[BenchmarkResult]] = {}

    for rt in runtimes:
        model_id = _match_model_on_runtime(m, rt)
        if not model_id:
            print(f"  {YELLOW}Skipping {rt.name}: model not found on this runtime.{RESET}")
            continue

        print(f"\n  {runtime_badge(rt.name)} Benchmarking {BOLD}{model_id}{RESET}...")
        def progress(run, total):
            label = "warmup" if run == 1 and total > 1 else f"run {run}"
            current = monitor.get_current()
            mem_str = f" {format_mini_memory_bar(current)}" if current else ""
            print(f"    {DIM}[{label}/{total}]{RESET}{mem_str}", end=" ", flush=True)

        results = run_benchmark_suite(rt, model_id, prompt_type, num_runs, progress_callback=progress)
        print()
        all_results[rt.name] = results

    if m.format == "gguf":
        print(f"\n  {runtime_badge('llama.cpp-cli')} Benchmarking {BOLD}{m.name}{RESET}...")
        cli_results = []
        for i in range(1, num_runs + 1):
            label = "warmup" if i == 1 and num_runs > 1 else f"run {i}"
            current = monitor.get_current()
            mem_str = f" {format_mini_memory_bar(current)}" if current else ""
            print(f"    {DIM}[{label}/{num_runs}]{RESET}{mem_str}", end=" ", flush=True)
            r = benchmark_llama_cpp_native(str(m.path), BENCH_PROMPTS[prompt_type], run_number=i)
            cli_results.append(r)
        print()
        all_results["llama.cpp-cli"] = cli_results

    # Stop monitoring and get report
    mem_report = monitor.stop()

    if not all_results:
        print(f"\n  {RED}No benchmarks could be run.{RESET}")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    _display_benchmark_comparison(all_results, num_runs, mem_report)
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def _match_model_on_runtime(local: LocalModel, runtime: RuntimeInfo) -> str | None:
    name_lower = local.name.lower()
    for rm in runtime.models:
        if rm.lower() == name_lower:
            return rm
    for rm in runtime.models:
        if name_lower in rm.lower() or rm.lower() in name_lower:
            return rm
    if len(runtime.models) == 1:
        return runtime.models[0]
    return None


def _display_benchmark_comparison(all_results: dict[str, list[BenchmarkResult]], num_runs: int, mem_report: MemoryReport | None = None):
    print(f"\n  {BOLD}{'=' * 65}{RESET}")
    print(f"  {BOLD}  BENCHMARK RESULTS — LM Studio vs llama.cpp{RESET}")
    print(f"  {BOLD}{'=' * 65}{RESET}")

    # Memory report
    if mem_report and mem_report.samples:
        print(f"\n{format_memory_report(mem_report)}")

    if num_runs > 1:
        print(f"\n  {BOLD}Per-Run Detail:{RESET}\n")
        print(f"  {BOLD}{'Runtime':<18} {'Run':>4} {'Gen Speed':>12} {'Prefill':>12} {'TTFT':>10} {'Tokens':>7}{RESET}")
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

    print(f"\n  {BOLD}Summary (excluding warmup):{RESET}\n")
    aggregates = {}
    for runtime_name, results in all_results.items():
        aggregates[runtime_name] = aggregate_results(results, skip_first=(num_runs > 1))

    print(f"  {BOLD}{'Runtime':<18} {'Avg Gen':>12} {'Min':>10} {'Max':>10} {'Avg Prefill':>13} {'Avg TTFT':>10}{RESET}")
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

    successful = {k: v for k, v in aggregates.items() if v.get("success")}
    if len(successful) >= 2:
        best_gen = max(successful.items(), key=lambda x: x[1]["avg_tok_per_sec"])
        second = sorted(successful.items(), key=lambda x: x[1]["avg_tok_per_sec"], reverse=True)
        speedup = best_gen[1]["avg_tok_per_sec"] / second[1][1]["avg_tok_per_sec"]

        print(f"\n  {BOLD}Winner:{RESET} {runtime_badge(best_gen[0])} {GREEN}{best_gen[1]['avg_tok_per_sec']:.1f} tok/s{RESET}")
        if speedup > 1.05:
            print(f"    {GREEN}{BOLD}{speedup:.1f}x faster{RESET} than {second[1][0]}")
        else:
            print(f"    {DIM}Both runtimes perform similarly{RESET}")
    elif len(successful) == 1:
        name, agg = next(iter(successful.items()))
        print(f"\n  {BOLD}Result:{RESET} {runtime_badge(name)} {GREEN}{agg['avg_tok_per_sec']:.1f} tok/s{RESET}")


# ── Screen: Find & Download ─────────────────────────────────────────────────

def screen_search_and_download(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Find & Download Models{RESET}")

    source_options = [
        f"{BOLD}GGUF models{RESET}     -- For LM Studio / llama.cpp (HuggingFace)",
        f"{BOLD}MLX models{RESET}      -- For Apple MLX framework (HuggingFace)",
        f"{BOLD}All HuggingFace{RESET} -- Search everything",
    ]
    sidx = prompt_choice("Where to search?", source_options)
    if sidx is None:
        return

    query = prompt_input("Search query (e.g. 'llama 8b', 'deepseek coder', 'qwen 32b')")
    if not query:
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
            f"       Downloads: {format_downloads(m.downloads)}  Likes: {m.likes}"
            f"  {DIM}{m.pipeline_tag or ''}{RESET}"
        )

    midx = prompt_choice("Select model for details + download:", [m.repo_id for m in results])
    if midx is None:
        return

    _show_model_download(mac, results[midx])


def _show_model_download(mac: MacSpec, model: HubModel):
    print(f"\n  {BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}{model.repo_id}{RESET}")
    print(f"  {BOLD}{'=' * 60}{RESET}\n")

    print(f"  {BOLD}Author:{RESET}     {model.author}")
    print(f"  {BOLD}Downloads:{RESET}  {format_downloads(model.downloads)}")
    print(f"  {BOLD}Likes:{RESET}      {model.likes}")

    print(f"\n  {DIM}Fetching file list...{RESET}")
    files = get_model_files(model.repo_id)
    gguf_files = [f for f in files if f["filename"].endswith(".gguf")]

    if gguf_files:
        print(f"\n  {BOLD}Available GGUF files:{RESET}")
        for f in sorted(gguf_files, key=lambda x: x.get("size") or 0):
            size_str = ""
            if f.get("size"):
                size_gb = f["size"] / (1024**3)
                total_est = size_gb * 1.15
                avail = mac.usable_memory_gb
                pct = (total_est / avail) * 100

                if pct <= 75:
                    tier_mark = f"{GREEN}FULL GPU{RESET}"
                elif pct <= 90:
                    tier_mark = f"{GREEN}Comfortable{RESET}"
                elif pct <= 100:
                    tier_mark = f"{YELLOW}Tight{RESET}"
                elif pct <= 150:
                    tier_mark = f"{YELLOW}Partial{RESET}"
                else:
                    tier_mark = f"{RED}Won't fit{RESET}"
                size_str = f"  {format_size(size_gb)} {tier_mark}"
            print(f"    {f['filename']}{size_str}")

    print(f"\n  {BOLD}Download Commands:{RESET}\n")

    if model.is_gguf:
        print(f"  {UNDERLINE}For LM Studio (recommended):{RESET}")
        print(f"  {GREEN}  Search for '{model.repo_id}' in LM Studio's Discover tab{RESET}")
        print()
        if gguf_files:
            q4_file = next((f for f in gguf_files if "Q4_K_M" in f["filename"]), gguf_files[0])
            print(f"  {UNDERLINE}For llama.cpp:{RESET}")
            print(f"  {GREEN}  huggingface-cli download {model.repo_id} {q4_file['filename']}{RESET}")

    if model.is_mlx:
        print(f"\n  {UNDERLINE}For MLX:{RESET}")
        print(f"  {GREEN}  pip install mlx-lm{RESET}")
        print(f"  {GREEN}  python -m mlx_lm.generate --model {model.repo_id} --prompt \"Hello\"{RESET}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: Detailed Analysis ────────────────────────────────────────────────

def screen_detail(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Detailed Model Analysis{RESET}")

    families = sorted(set(m.family for m in MODELS))
    family_options = ["All families"] + families
    fidx = prompt_choice("Filter by model family:", family_options)
    if fidx is None:
        return
    filtered = MODELS if fidx == 0 else [m for m in MODELS if m.family == families[fidx - 1]]

    model_names = [f"{m.name} ({m.params_billion}B)" for m in filtered]
    idx = prompt_choice("Select model:", model_names)
    if idx is None:
        return
    model = filtered[idx]

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
    tmpl = get_template(model.name)

    print(f"\n  {BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}{model.name} @ {quant.name}{RESET}")
    print(f"  {BOLD}{'=' * 60}{RESET}\n")

    print(f"  {BOLD}Model:{RESET}       {model.params_billion}B params | {model.family} | {model.license}")
    print(f"  {BOLD}Context:{RESET}     {format_context(model.default_context_length)} default, {format_context(model.max_context_length)} max")
    print(f"  {BOLD}Quant:{RESET}       {quant.name} ({quant.bits_per_weight:.2f} bpw, {quant.quality_rating})")

    print(f"\n  {BOLD}VRAM Tier:{RESET}   {tier_badge(result.tier)}")
    if result.tier == CompatibilityTier.PARTIAL_OFFLOAD and result.offload_pct:
        print(f"  {BOLD}GPU Layers:{RESET}  {result.gpu_layers}/{result.total_layers} ({result.offload_pct:.0f}% on GPU)")

    print(f"\n  {BOLD}Memory Breakdown:{RESET}")
    print(f"    Model weights:  {format_size(result.model_size_gb):>10}")
    print(f"    KV cache:       {format_size(result.kv_cache_gb):>10}  (at {format_context(result.context_length)} ctx)")
    print(f"    Overhead:       {format_size(result.overhead_gb):>10}")
    print(f"    {'-' * 28}")
    print(f"    {BOLD}Total:          {format_size(result.total_memory_gb):>10}{RESET}")
    print(f"    Available:      {format_size(result.available_memory_gb):>10}")
    print(f"\n  {BOLD}Memory:{RESET}  {bar(result.total_memory_gb, result.available_memory_gb, 35)}")

    if result.estimated_tok_per_sec:
        print(f"  {BOLD}Speed:{RESET}   {color_speed(result.estimated_tok_per_sec)} (estimated)")

    if tmpl:
        print(f"\n  {BOLD}Recommended Settings:{RESET}")
        print(f"    Context: {format_context(tmpl.recommended_ctx)} | Temp: {tmpl.temperature} | Top P: {tmpl.top_p}")
        if tmpl.notes:
            print(f"    {DIM}{tmpl.notes}{RESET}")
        if tmpl.bench_tok_s_q4:
            print(f"\n  {BOLD}Community Benchmarks (M4 Pro 48GB):{RESET}")
            if tmpl.bench_tok_s_q4:
                print(f"    Q4_K_M: ~{tmpl.bench_tok_s_q4:.0f} tok/s gen | ~{tmpl.bench_prefill_q4:.0f} tok/s prefill")
            if tmpl.bench_tok_s_q8:
                print(f"    Q8_0:   ~{tmpl.bench_tok_s_q8:.0f} tok/s gen")

    effective_max = min(max_ctx, model.max_context_length)
    print(f"\n  {BOLD}Max context:{RESET} {format_context(effective_max)} tokens")

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    overhead = estimate_overhead_gb(model_size)

    print(f"\n  {BOLD}Context Scaling:{RESET}")
    print(f"    {'Context':>10} {'KV Cache':>10} {'Total':>10} {'Fits':>6}")
    print(f"    {'-' * 40}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if ctx > model.max_context_length:
            break
        kv = estimate_kv_cache_gb(model, ctx)
        total = model_size + kv + overhead
        fits = total <= mac.usable_memory_gb
        mark = f"{GREEN}Y{RESET}" if fits else f"{RED}N{RESET}"
        print(f"    {format_context(ctx):>10} {format_size(kv):>10} {format_size(total):>10} {mark:>6}")

    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Screen: Benchmark Arena ──────────────────────────────────────────────────

def screen_benchmark(mac: MacSpec):
    print(f"\n  {BOLD}{CYAN}Benchmark Arena{RESET}")
    print(f"  {DIM}Compare LM Studio vs llama.cpp head-to-head{RESET}\n")

    runtimes = detect_all_runtimes()

    if not runtimes:
        print(f"  {YELLOW}No runtimes detected.{RESET}\n")
        print(f"  To benchmark, start at least one:")
        print(f"    {BOLD}LM Studio:{RESET}    Start local server from the app (port 1234)")
        print(f"    {BOLD}llama.cpp:{RESET}    llama-server -m model.gguf --port 8080")
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
        return

    print(f"  {BOLD}Detected runtimes:{RESET}\n")
    for rt in runtimes:
        models_str = ", ".join(rt.models[:3])
        print(f"    {runtime_badge(rt.name)}")
        print(f"      {DIM}Models: {models_str}{RESET}")
    print()

    all_model_ids: list[tuple[RuntimeInfo | None, str]] = []
    model_display: list[str] = []
    for rt in runtimes:
        for mid in rt.models:
            all_model_ids.append((rt, mid))
            model_display.append(f"{runtime_badge(rt.name)} {mid}")

    local = scan_all()
    gguf_local = [m for m in local if m.format == "gguf"]
    for m in gguf_local[:10]:
        all_model_ids.append((None, str(m.path)))
        model_display.append(f"{runtime_badge('llama.cpp-cli')} {m.name} ({format_size(m.size_gb)})")

    midx = prompt_choice("Select model to benchmark:", model_display)
    if midx is None:
        return

    selected_rt, selected_model = all_model_ids[midx]

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

    # Start memory monitoring
    monitor = MemoryMonitor(interval=0.5)
    monitor.start()

    all_results: dict[str, list[BenchmarkResult]] = {}

    if selected_rt is None:
        print(f"\n  {runtime_badge('llama.cpp-cli')} Benchmarking...")
        cli_results = []
        for i in range(1, num_runs + 1):
            label = "warmup" if i == 1 and num_runs > 1 else f"run {i}"
            current = monitor.get_current()
            mem_str = f" {format_mini_memory_bar(current)}" if current else ""
            print(f"    {DIM}[{label}/{num_runs}]{RESET}{mem_str}", end=" ", flush=True)
            r = benchmark_llama_cpp_native(selected_model, BENCH_PROMPTS[prompt_type], run_number=i)
            cli_results.append(r)
        print()
        all_results["llama.cpp-cli"] = cli_results
    else:
        print(f"\n  {runtime_badge(selected_rt.name)} Benchmarking {BOLD}{selected_model}{RESET}...")
        def progress(run, total):
            label = "warmup" if run == 1 and total > 1 else f"run {run}"
            current = monitor.get_current()
            mem_str = f" {format_mini_memory_bar(current)}" if current else ""
            print(f"    {DIM}[{label}/{total}]{RESET}{mem_str}", end=" ", flush=True)

        results = run_benchmark_suite(selected_rt, selected_model, prompt_type, num_runs, progress_callback=progress)
        print()
        all_results[selected_rt.name] = results

        other_runtimes = [rt for rt in runtimes if rt.name != selected_rt.name]
        for ort in other_runtimes:
            matched = None
            for rm in ort.models:
                if (rm.lower() == selected_model.lower()
                        or selected_model.lower() in rm.lower()
                        or rm.lower() in selected_model.lower()):
                    matched = rm
                    break
            if not matched and len(ort.models) == 1:
                matched = ort.models[0]
            if matched and prompt_yes_no(f"  Also benchmark on {ort.name} ({matched})?"):
                print(f"\n  {runtime_badge(ort.name)} Benchmarking {BOLD}{matched}{RESET}...")
                results = run_benchmark_suite(ort, matched, prompt_type, num_runs, progress_callback=progress)
                print()
                all_results[ort.name] = results

    mem_report = monitor.stop()

    _display_benchmark_comparison(all_results, num_runs, mem_report)
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


# ── Main Loop ────────────────────────────────────────────────────────────────

def main_interactive():
    clear_screen()
    print_banner()

    mac = select_hardware()
    if not mac:
        print(f"\n  {RED}No hardware selected. Exiting.{RESET}")
        sys.exit(0)

    while True:
        clear_screen()
        runtimes = detect_all_runtimes()
        rt_str = " | ".join(runtime_badge(r.name) for r in runtimes) if runtimes else f"{DIM}no runtimes detected{RESET}"

        # Live memory status
        mem = get_memory_sample()
        mem_pct = mem.usage_pct
        mem_color = GREEN if mem_pct <= 60 else (YELLOW if mem_pct <= 80 else RED)
        mem_str = f"{mem_color}{mem.used_gb:.1f}/{mem.total_gb:.0f}GB ({mem_pct:.0f}%){RESET}"

        print(f"\n  {BOLD}{CYAN}{'=' * 60}{RESET}")
        print(f"  {BOLD}{CYAN}  {mac.chip} {mac.memory_gb}GB{RESET}  {DIM}|{RESET}  MEM: {mem_str}  {DIM}|{RESET}  {rt_str}")
        print(f"  {BOLD}{CYAN}{'=' * 60}{RESET}")

        menu = [
            f"{BOLD}My Models{RESET}               -- Browse, analyze, get settings templates",
            f"{BOLD}Find & Download{RESET}         -- Search HuggingFace for GGUF / MLX models",
            f"{BOLD}Compatibility Check{RESET}     -- VRAM tiers: Full GPU / Partial / Won't fit",
            f"{BOLD}Detailed Analysis{RESET}       -- Deep dive on any model with benchmarks",
            f"{BOLD}Benchmark Arena{RESET}         -- LM Studio vs llama.cpp head-to-head",
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
