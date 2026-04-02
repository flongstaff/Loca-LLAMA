"""Batch size and context optimization for maximum throughput.

Based on research from:
- https://github.com/alexziskind1/llama-throughput-lab (throughput testing)
- https://github.com/alexziskind1/llm-inference-calculator (VRAM estimation)
- https://github.com/alexziskind1/draftbench (benchmarking framework)

This module helps optimize batch size and context length for maximum
tokens/second while staying within memory constraints.
"""

from dataclasses import dataclass
from .hardware import MacSpec
from .models import LLMModel
from .quantization import QuantFormat


@dataclass
class BatchOptimizationResult:
    """Optimal batch and context configuration."""

    # Settings
    batch_size: int
    context_length: int
    gpu_layers: int
    ubatch_size: int  # Unified batch size (llama.cpp internal)

    # Memory
    total_memory_gb: float
    gpu_memory_gb: float
    system_memory_gb: float
    headroom_gb: float

    # Performance estimates
    estimated_throughput_tok_s: float
    estimated_latency_ms: float
    memory_efficiency: float  # 0-1, higher is better

    # Recommendation
    recommendation: str
    notes: list[str]


def estimate_batch_memory_overhead(
    batch_size: int,
    context_length: int,
    model: LLMModel,
    quant: QuantFormat,
) -> float:
    """Estimate additional memory consumed by batch operations.

    Based on llama.cpp's internal batching implementation:
    - Each batch item requires KV cache storage
    - Intermediate tensors scale with batch × context
    - Overhead is approximately linear with batch size

    Returns: Additional memory in GB
    """
    from .analyzer import estimate_kv_cache_raw

    # Base KV cache per token (f16 = 2 bytes per value)
    # For QKV projections: 3 * num_heads * head_dim * context * 2 bytes
    kv_bytes_per_token = 3 * model.num_kv_heads * model.head_dim * 2

    # Batch overhead: each batch item needs full KV cache
    batch_kv_gb = (batch_size * context_length * kv_bytes_per_token) / (1024**3)

    # Intermediate tensor overhead (activations, attention, MLP)
    # Roughly 4-8x the KV cache for intermediate computations
    intermediate_gb = batch_kv_gb * 6.0

    return batch_kv_gb + intermediate_gb


def estimate_throughput_with_batch(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    batch_size: int,
    context_length: int,
    gpu_layers: int,
) -> float:
    """Estimate tokens/second for a given batch configuration.

    Based on llama-throughput-lab methodology:
    - GPU-bound inference: limited by memory bandwidth
    - Batch size increases throughput up to a point
    - Diminishing returns after optimal batch size

    Returns: Estimated tokens/second
    """
    from .analyzer import estimate_model_size_gb

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    gpu_layer_fraction = gpu_layers / model.num_layers if model.num_layers > 0 else 1.0

    # Base throughput (single token, full GPU)
    # tok/s = memory_bandwidth / (model_size * 1.1)
    base_tok_s = mac.memory_bandwidth_gbs / (model_size * 1.1)

    # Batch size multiplier (based on empirical llama.cpp benchmarks)
    # Batch 1: 1.0x, Batch 64: 2.5x, Batch 128: 3.0x, Batch 256: 3.5x, Batch 512: 3.8x
    batch_multipliers = {
        1: 1.0,
        16: 1.8,
        32: 2.2,
        64: 2.5,
        128: 3.0,
        256: 3.5,
        512: 3.8,
        1024: 4.0,
        2048: 4.1,
    }

    # Find closest multiplier
    closest_batch = min(batch_multipliers.keys(), key=lambda x: abs(x - batch_size))
    batch_mult = batch_multipliers[closest_batch]

    # Context length penalty (longer context = more KV cache reads)
    # Every 1024 tokens adds ~5% overhead
    context_penalty = 1.0 - (context_length / 1024) * 0.05
    context_penalty = max(0.8, context_penalty)  # Cap at 20% penalty

    # GPU utilization factor
    gpu_factor = gpu_layer_fraction

    # Combined throughput estimate
    estimated_tok_s = base_tok_s * batch_mult * context_penalty * gpu_factor

    return estimated_tok_s


def calculate_optimal_batch_size_v2(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    target_throughput: float | None = None,
    max_context: int = 8192,
    preferred_batch_sizes: list[int] | None = None,
) -> BatchOptimizationResult | None:
    """Calculate optimal batch size with throughput estimation.

    Args:
        mac: Hardware specs
        model: Model architecture
        quant: Quantization format
        target_throughput: Desired tokens/second (None for max throughput)
        max_context: Maximum context length to consider
        preferred_batch_sizes: List of batch sizes to test (default: [256, 512, 1024, 2048])

    Returns: BatchOptimizationResult with best configuration
    """
    if preferred_batch_sizes is None:
        preferred_batch_sizes = [256, 512, 1024, 2048, 4096]

    from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    overhead = model_size * 0.10 + 0.5

    available = mac.usable_memory_gb

    # Test different batch sizes and contexts
    best_result = None
    best_score = -1

    for batch_size in preferred_batch_sizes:
        for context_length in [256, 512, 1024, 2048, 4096, 8192]:
            if context_length > max_context:
                continue

            # Calculate memory usage
            kv_cache = estimate_kv_cache_gb(model, context_length)
            batch_overhead = estimate_batch_memory_overhead(batch_size, context_length, model, quant)

            total_memory = model_size + kv_cache + overhead + batch_overhead

            if total_memory > available * 0.95:  # Reserve 5% for safety
                continue

            # Estimate throughput
            gpu_layers = model.num_layers  # Full GPU for best performance
            throughput = estimate_throughput_with_batch(
                mac, model, quant, batch_size, context_length, gpu_layers
            )

            # Score: prioritize throughput, then memory efficiency
            memory_efficiency = 1.0 - (total_memory / available)
            score = throughput * 1.0 + memory_efficiency * 10.0

            if target_throughput and throughput < target_throughput:
                continue

            if score > best_score:
                best_score = score
                best_result = BatchOptimizationResult(
                    batch_size=batch_size,
                    context_length=context_length,
                    gpu_layers=gpu_layers,
                    ubatch_size=min(batch_size * 2, 2048),  # ubatch ~ 2x batch
                    total_memory_gb=total_memory,
                    gpu_memory_gb=total_memory * 0.9,  # ~90% on GPU
                    system_memory_gb=total_memory * 0.1,
                    headroom_gb=available - total_memory,
                    estimated_throughput_tok_s=throughput,
                    estimated_latency_ms=1000 / throughput,
                    memory_efficiency=memory_efficiency,
                    recommendation="Optimal" if throughput >= 10 else "Good",
                    notes=[],
                )

    if best_result:
        best_result.notes.append(f"Estimated throughput: {best_result.estimated_throughput_tok_s:.1f} tok/s")
        best_result.notes.append(f"Memory efficiency: {best_result.memory_efficiency:.0%}")
        if best_result.headroom_gb > 5:
            best_result.notes.append(f"Headroom: {best_result.headroom_gb:.1f}GB available")
        else:
            best_result.notes.append("⚠️ Low headroom - consider reducing batch size")

    return best_result


def compare_batch_sizes(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    context_length: int = 4096,
    batch_sizes: list[int] | None = None,
) -> list[BatchOptimizationResult]:
    """Compare multiple batch sizes for a given model and context.

    Useful for finding the tradeoff between throughput and memory usage.
    """
    if batch_sizes is None:
        batch_sizes = [128, 256, 512, 1024, 2048, 4096]

    results = []
    for batch_size in batch_sizes:
        result = calculate_optimal_batch_size_v2(
            mac, model, quant, max_context=context_length,
            preferred_batch_sizes=[batch_size]
        )
        if result:
            results.append(result)

    # Sort by throughput (descending)
    results.sort(key=lambda r: r.estimated_throughput_tok_s, reverse=True)

    return results


def optimize_for_batch_preference(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    batch_preference: str = "high",  # "high", "balanced", "low"
    context_length: int = 4096,
) -> BatchOptimizationResult | None:
    """Optimize batch size based on user preference.

    Args:
        mac: Hardware specs
        model: Model architecture
        quant: Quantization format
        batch_preference: "high" (max throughput), "balanced", "low" (memory efficient)
        context_length: Desired context length

    Returns: BatchOptimizationResult
    """
    if batch_preference == "high":
        preferred = [2048, 4096, 1024, 512]
    elif batch_preference == "balanced":
        preferred = [1024, 512, 2048, 256]
    else:  # "low"
        preferred = [256, 512, 1024, 128]

    return calculate_optimal_batch_size_v2(
        mac, model, quant, max_context=context_length,
        preferred_batch_sizes=preferred
    )
