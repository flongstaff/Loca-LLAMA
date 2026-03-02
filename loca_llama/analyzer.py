"""Core analysis engine: estimates memory, context limits, and performance."""

from dataclasses import dataclass
from .hardware import MacSpec
from .models import LLMModel
from .quantization import QuantFormat, QUANT_FORMATS, RECOMMENDED_FORMATS


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

    @property
    def memory_utilization_pct(self) -> float:
        return (self.total_memory_gb / self.available_memory_gb) * 100

    @property
    def rating(self) -> str:
        """User-friendly rating of how well this config runs."""
        if not self.fits_in_memory:
            return "Won't fit"
        pct = self.memory_utilization_pct
        if pct <= 60:
            return "Excellent"
        elif pct <= 75:
            return "Good"
        elif pct <= 85:
            return "Tight"
        elif pct <= 95:
            return "Very Tight"
        else:
            return "Barely fits"


def estimate_model_size_gb(params_billion: float, bits_per_weight: float) -> float:
    """Estimate model file size in GB."""
    return (params_billion * 1e9 * bits_per_weight / 8) / (1024**3)


def estimate_kv_cache_gb(
    model: LLMModel,
    context_length: int,
    kv_bits: int = 16,
) -> float:
    """Estimate KV cache memory in GB.

    Formula: 2 * num_layers * num_kv_heads * head_dim * context_length * (kv_bits/8)

    The factor of 2 accounts for both K and V caches.
    """
    bytes_per_element = kv_bits / 8
    kv_cache_bytes = (
        2
        * model.num_layers
        * model.num_kv_heads
        * model.head_dim
        * context_length
        * bytes_per_element
    )
    return kv_cache_bytes / (1024**3)


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

    tok_s = estimate_tokens_per_second(mac, model_size, model.params_billion) if fits else None

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
    )


def analyze_all(
    mac: MacSpec,
    models: list[LLMModel],
    quant_names: list[str] | None = None,
    context_length: int | None = None,
    only_fits: bool = False,
) -> list[ModelEstimate]:
    """Analyze all models with given quant formats against hardware."""
    formats = quant_names or RECOMMENDED_FORMATS
    quants = [QUANT_FORMATS[q] for q in formats if q in QUANT_FORMATS]

    results = []
    for model in models:
        for quant in quants:
            est = analyze_model(mac, model, quant, context_length)
            if only_fits and not est.fits_in_memory:
                continue
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
