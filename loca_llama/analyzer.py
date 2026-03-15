"""Core analysis engine: estimates memory, context limits, and performance."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from .hardware import MacSpec
from .models import LLMModel, MODELS
from .quantization import QuantFormat, QUANT_FORMATS, RECOMMENDED_FORMATS


class CompatibilityTier(Enum):
    """How well a model fits in unified memory."""

    FULL_GPU = "full_gpu"           # <=75% memory: everything on GPU, fast, room for context
    COMFORTABLE = "comfortable"     # <=90% memory: fits on GPU but limited context headroom
    TIGHT_FIT = "tight_fit"         # <=100% memory: barely fits, minimal context, may swap
    PARTIAL_OFFLOAD = "partial"     # >100% but <=150%: needs CPU offload for some layers
    WONT_FIT = "wont_fit"           # >150%: too large even with offloading


@dataclass
class ModelEstimate:
    """Full analysis result for one model + quantization + context combo."""

    model: LLMModel
    quant: QuantFormat
    context_length: int

    # Memory breakdown (GB)
    model_size_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_memory_gb: float

    # Feasibility
    fits_in_memory: bool
    available_memory_gb: float
    headroom_gb: float

    # Performance estimates
    estimated_tok_per_sec: float | None

    # VRAM tier
    tier: CompatibilityTier = CompatibilityTier.WONT_FIT

    # Offload info (for partial offload)
    gpu_layers: int | None = None
    total_layers: int | None = None
    offload_pct: float | None = None  # % of layers on GPU

    @property
    def memory_utilization_pct(self) -> float:
        return (self.total_memory_gb / self.available_memory_gb) * 100

    @property
    def rating(self) -> str:
        """User-friendly rating of how well this config runs."""
        if self.tier == CompatibilityTier.WONT_FIT:
            return "Won't fit"
        return {
            CompatibilityTier.FULL_GPU: "Full GPU",
            CompatibilityTier.COMFORTABLE: "Comfortable",
            CompatibilityTier.TIGHT_FIT: "Tight fit",
            CompatibilityTier.PARTIAL_OFFLOAD: "Partial offload",
        }[self.tier]

    @property
    def tier_emoji_label(self) -> str:
        """Short tier label with visual indicator."""
        return {
            CompatibilityTier.FULL_GPU: "[***] Full GPU",
            CompatibilityTier.COMFORTABLE: "[** ] Comfortable",
            CompatibilityTier.TIGHT_FIT: "[*  ] Tight",
            CompatibilityTier.PARTIAL_OFFLOAD: "[~  ] Partial Offload",
            CompatibilityTier.WONT_FIT: "[x  ] Won't Fit",
        }[self.tier]

    @property
    def is_usable(self) -> bool:
        """Can this model run at all (even slowly with offloading)?"""
        return self.tier != CompatibilityTier.WONT_FIT

    @property
    def is_fully_gpu(self) -> bool:
        """Does this fit entirely in GPU memory?"""
        return self.tier in (CompatibilityTier.FULL_GPU, CompatibilityTier.COMFORTABLE, CompatibilityTier.TIGHT_FIT)


def estimate_model_size_gb(params_billion: float, bits_per_weight: float) -> float:
    """Estimate model file size in GB."""
    return (params_billion * 1e9 * bits_per_weight / 8) / (1024**3)


def estimate_kv_cache_raw(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    kv_bits: int = 16,
    inference_mode: str = "streaming",
    model_size_gb: float = 0.0,
) -> float:
    """Estimate KV cache memory in GB from raw architecture parameters.

    Formula: 2 * num_layers * num_kv_heads * head_dim * context_length * (kv_bits/8)

    The factor of 2 accounts for both K and V caches.

    In batch mode, adds activation overhead for processing full sequences
    at once (adapted from llm-inference-calculator's incremental vs bulk
    distinction).
    """
    bytes_per_element = kv_bits / 8
    kv_cache_bytes = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * context_length
        * bytes_per_element
    )
    kv_gb = kv_cache_bytes / (1024**3)

    if inference_mode == "batch" and model_size_gb > 0:
        # Batch mode: full activation storage needed for parallel sequence processing.
        # Activation overhead scales with model size and context length.
        activation_overhead = model_size_gb * 0.5 * (context_length / 2048)
        # Cap at reasonable maximum (2x model size)
        activation_overhead = min(activation_overhead, model_size_gb * 2.0)
        kv_gb += activation_overhead

    return kv_gb


def estimate_kv_cache_gb(
    model: LLMModel,
    context_length: int,
    kv_bits: int = 16,
) -> float:
    """Estimate KV cache memory in GB for a specific model."""
    return estimate_kv_cache_raw(
        model.num_layers, model.num_kv_heads, model.head_dim, context_length, kv_bits
    )


def estimate_overhead_gb(model_size_gb: float) -> float:
    """Estimate runtime overhead (compute buffers, scratch space, etc.)."""
    # ~10% of model size + 0.5 GB base overhead
    return model_size_gb * 0.10 + 0.5


def estimate_tokens_per_second(
    mac: MacSpec,
    model_size_gb: float,
    params_billion: float,
) -> float | None:
    """Rough estimate of generation speed (tokens/sec).

    LLM inference on Apple Silicon is memory-bandwidth-bound.
    tok/s ~ memory_bandwidth / (model_size * 1.1)
    where 1.1 accounts for KV cache reads during generation.
    """
    if model_size_gb <= 0:
        return None
    # During token generation, we read roughly the full model weights per token
    effective_read_gb = model_size_gb * 1.1
    return mac.memory_bandwidth_gbs / effective_read_gb


def estimate_partial_offload_speed(
    mac: MacSpec,
    model_size_gb: float,
    offload_pct: float,
) -> float | None:
    """Estimate speed when only some layers are on GPU.

    When offloading, the GPU-resident layers run at full bandwidth,
    but CPU-resident layers are ~10-20x slower due to CPU memory bandwidth.
    Overall speed is a weighted harmonic mean.
    """
    if model_size_gb <= 0 or offload_pct <= 0:
        return None

    gpu_fraction = offload_pct / 100.0
    cpu_fraction = 1.0 - gpu_fraction

    # GPU tok/s (full bandwidth)
    gpu_speed = mac.memory_bandwidth_gbs / (model_size_gb * gpu_fraction * 1.1)
    # CPU tok/s (~8-12 GB/s effective for LLM on CPU)
    cpu_bandwidth = 12.0  # conservative estimate for Apple Silicon CPU path
    cpu_speed = cpu_bandwidth / (model_size_gb * cpu_fraction * 1.1) if cpu_fraction > 0 else float('inf')

    # Harmonic mean weighted by fraction
    if gpu_speed > 0 and cpu_speed > 0:
        return 1.0 / (gpu_fraction / gpu_speed + cpu_fraction / cpu_speed)
    return None


def compute_tier(
    mac: MacSpec,
    model_size_gb: float,
    total_memory_gb: float,
    model: LLMModel,
) -> tuple[CompatibilityTier, int | None, float | None]:
    """Determine compatibility tier, GPU layers, and offload percentage.

    Returns: (tier, gpu_layers, offload_pct)
    """
    available = mac.usable_memory_gb
    pct = (total_memory_gb / available) * 100 if available > 0 else 999

    if pct <= 75:
        return CompatibilityTier.FULL_GPU, model.num_layers, 100.0
    elif pct <= 90:
        return CompatibilityTier.COMFORTABLE, model.num_layers, 100.0
    elif pct <= 100:
        return CompatibilityTier.TIGHT_FIT, model.num_layers, 100.0
    elif pct <= 150:
        # Partial offload: calculate how many layers fit on GPU
        # model_size is the bulk; we need model_size * fraction + overhead <= available
        overhead = estimate_overhead_gb(model_size_gb)
        mem_for_weights = available - overhead - 1.0  # 1 GB for minimal KV cache
        if mem_for_weights <= 0:
            return CompatibilityTier.WONT_FIT, None, None
        fraction = min(mem_for_weights / model_size_gb, 1.0)
        gpu_layers = int(model.num_layers * fraction)
        offload_pct = (gpu_layers / model.num_layers) * 100
        return CompatibilityTier.PARTIAL_OFFLOAD, gpu_layers, offload_pct
    else:
        return CompatibilityTier.WONT_FIT, None, None


def analyze_model(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    context_length: int | None = None,
) -> ModelEstimate:
    """Analyze a single model/quant/context combo against hardware."""
    ctx = context_length or model.default_context_length

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    kv_cache = estimate_kv_cache_gb(model, ctx)
    overhead = estimate_overhead_gb(model_size)
    total = model_size + kv_cache + overhead

    available = mac.usable_memory_gb
    fits = total <= available
    headroom = available - total

    tier, gpu_layers, offload_pct = compute_tier(mac, model_size, total, model)

    # Estimate speed based on tier
    if tier in (CompatibilityTier.FULL_GPU, CompatibilityTier.COMFORTABLE, CompatibilityTier.TIGHT_FIT):
        tok_s = estimate_tokens_per_second(mac, model_size, model.params_billion)
    elif tier == CompatibilityTier.PARTIAL_OFFLOAD and offload_pct:
        tok_s = estimate_partial_offload_speed(mac, model_size, offload_pct)
    else:
        tok_s = None

    return ModelEstimate(
        model=model,
        quant=quant,
        context_length=ctx,
        model_size_gb=model_size,
        kv_cache_gb=kv_cache,
        overhead_gb=overhead,
        total_memory_gb=total,
        fits_in_memory=fits,
        available_memory_gb=available,
        headroom_gb=headroom,
        estimated_tok_per_sec=tok_s,
        tier=tier,
        gpu_layers=gpu_layers,
        total_layers=model.num_layers,
        offload_pct=offload_pct,
    )


def analyze_all(
    mac: MacSpec,
    models: list[LLMModel],
    quant_names: list[str] | None = None,
    context_length: int | None = None,
    only_fits: bool = False,
    include_partial: bool = False,
) -> list[ModelEstimate]:
    """Analyze all models with given quant formats against hardware.

    Args:
        only_fits: Only return models that fully fit in memory.
        include_partial: Also include partial offload models when only_fits=True.
    """
    formats = quant_names or RECOMMENDED_FORMATS
    quants = [QUANT_FORMATS[q] for q in formats if q in QUANT_FORMATS]

    results = []
    for model in models:
        for quant in quants:
            est = analyze_model(mac, model, quant, context_length)
            if only_fits:
                if include_partial and est.is_usable:
                    results.append(est)
                elif est.fits_in_memory:
                    results.append(est)
            else:
                results.append(est)

    return results


def max_context_for_model(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
) -> int:
    """Find the maximum context length that fits in memory."""
    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    overhead = estimate_overhead_gb(model_size)
    remaining = mac.usable_memory_gb - model_size - overhead

    if remaining <= 0:
        return 0

    # Binary search for max context
    lo, hi = 0, model.max_context_length
    while lo < hi:
        mid = (lo + hi + 1) // 2
        kv = estimate_kv_cache_gb(model, mid)
        if kv <= remaining:
            lo = mid
        else:
            hi = mid - 1

    return lo


# ── Use-case filtering ──────────────────────────────────────────────────────

USE_CASE_FILTERS: dict[str, Callable] = {
    "coding": lambda m: m.family in ("Qwen", "CodeLlama", "StarCoder", "DeepSeek"),
    "reasoning": lambda m: "R1" in m.name or m.family in ("DeepSeek", "Qwen"),
    "small": lambda m: m.params_billion <= 8,
    "large-context": lambda m: m.max_context_length >= 65536,
}

VALID_USE_CASES = ["general", "coding", "reasoning", "small", "large-context"]

_RECOMMEND_QUANT_ORDER = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q8_0", "Q3_K_L"]


def recommend_models(
    mac: MacSpec,
    use_case: str = "general",
    top_n: int = 8,
) -> list[ModelEstimate]:
    """Find the best model+quant combos that fit comfortably on the given hardware.

    Algorithm: iterate quant formats in quality order, test each model,
    keep the first (best-quant) result per model that fits at <=90% utilization,
    sort by params descending, return top N.
    """
    filt = USE_CASE_FILTERS.get(use_case)
    models = [m for m in MODELS if filt(m)] if filt else list(MODELS)

    recommendations: list[ModelEstimate] = []
    seen_models: set[str] = set()

    for quant_name in _RECOMMEND_QUANT_ORDER:
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

    recommendations.sort(key=lambda r: r.model.params_billion, reverse=True)
    return recommendations[:top_n]
