"""GPU layer and batch size optimization for Apple Silicon.

This module helps optimize GPU layer allocation and batch size settings
to maximize GPU utilization and minimize CPU offload.

Based on research from:
- https://github.com/alexziskind1/llm-inference-calculator (VRAM estimation)
- https://github.com/alexziskind1/llama-throughput-lab (throughput testing)
"""

from dataclasses import dataclass
from .hardware import MacSpec
from .models import LLMModel
from .quantization import QuantFormat


@dataclass
class GPUOptimizationResult:
    """Recommended GPU configuration for optimal performance."""

    # Optimal GPU settings
    gpu_layers: int
    batch_size: int
    context_length: int

    # Memory usage
    model_size_gb: float
    kv_cache_gb: float
    total_memory_gb: float
    gpu_memory_gb: float  # Estimated GPU VRAM usage
    system_memory_gb: float  # Estimated system RAM usage

    # Performance metrics
    offload_percentage: float  # % of layers on CPU (0 = full GPU)
    estimated_tokens_per_second: float
    estimated_latency_ms: float

    # Recommendation
    recommendation: str
    notes: list[str]


def estimate_gpu_memory_usage(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    context_length: int,
    batch_size: int,
    gpu_layers: int,
) -> tuple[float, float]:
    """Estimate GPU vs system memory usage.

    On Apple Silicon, memory is unified but GPU has priority access.
    We estimate how much goes to GPU vs CPU based on layer allocation.

    Returns: (gpu_memory_gb, system_memory_gb)
    """
    from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb

    # Total model size
    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)

    # KV cache (goes to GPU if layers are on GPU)
    kv_cache = estimate_kv_cache_gb(model, context_length)

    # Overhead (compute buffers, etc.)
    overhead = model_size * 0.10 + 0.5

    # GPU memory: GPU layers + KV cache + GPU overhead
    # Assume GPU gets ~70% of model weights if partial offload
    gpu_layer_fraction = gpu_layers / model.num_layers if model.num_layers > 0 else 1.0
    gpu_memory = (model_size * gpu_layer_fraction) + kv_cache + (overhead * 0.7)

    # System memory: CPU layers + remaining overhead
    system_memory = model_size * (1 - gpu_layer_fraction) + (overhead * 0.3)

    return gpu_memory, system_memory


def calculate_optimal_gpu_layers(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    context_length: int = 4096,
    target_gpu_utilization: float = 0.95,
) -> tuple[int, float]:
    """Calculate optimal number of GPU layers.

    Args:
        mac: Hardware specs
        model: Model architecture
        quant: Quantization format
        context_length: Context length in tokens
        target_gpu_utilization: Target GPU memory utilization (0-1)

    Returns:
        (gpu_layers, offload_percentage)
    """
    from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    kv_cache = estimate_kv_cache_gb(model, context_length)
    overhead = model_size * 0.10 + 0.5
    total_memory = model_size + kv_cache + overhead

    available = mac.usable_memory_gb

    # If model fits entirely, put all layers on GPU
    if total_memory <= available:
        return model.num_layers, 0.0

    # Partial offload: calculate how many layers fit on GPU
    # Reserve 20% for headroom and batch operations
    gpu_budget = available * target_gpu_utilization

    # GPU memory for layers = GPU budget - KV cache - overhead
    mem_for_layers = gpu_budget - kv_cache - (overhead * 0.7)

    if mem_for_layers <= 0:
        # Can't fit any layers on GPU
        return 0, 100.0

    # Calculate fraction of model that fits
    layer_fraction = min(mem_for_layers / model_size, 1.0)
    gpu_layers = int(model.num_layers * layer_fraction)

    # Ensure at least some layers on GPU for efficiency
    gpu_layers = max(gpu_layers, 5)

    offload_pct = ((model.num_layers - gpu_layers) / model.num_layers) * 100

    return gpu_layers, offload_pct


def calculate_optimal_batch_size(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    gpu_layers: int,
    context_length: int,
    max_batch_size: int = 2048,
) -> int:
    """Calculate optimal batch size based on GPU memory availability.

    Larger batch sizes = higher throughput but more GPU memory usage.

    Returns: Optimal batch size (256, 512, 1024, or 2048)
    """
    # Estimate GPU memory usage with different batch sizes
    batch_sizes = [256, 512, 1024, 2048]

    from .analyzer import estimate_model_size_gb, estimate_kv_cache_gb

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    kv_cache = estimate_kv_cache_gb(model, context_length)
    overhead = model_size * 0.10 + 0.5

    gpu_layer_fraction = gpu_layers / model.num_layers if model.num_layers > 0 else 1.0
    gpu_base_memory = (model_size * gpu_layer_fraction) + (overhead * 0.7)

    # Batch size adds ~0.5-1.0GB per 512 tokens of batch
    batch_memory_per_512 = 0.75  # GB

    available_gpu = mac.usable_memory_gb * 0.85  # Reserve 15% for safety

    for batch_size in batch_sizes:
        batch_memory = (batch_size / 512) * batch_memory_per_512
        total_gpu = gpu_base_memory + kv_cache + batch_memory

        if total_gpu <= available_gpu:
            return batch_size

    return 256  # Fallback to smallest batch size


def optimize_for_hardware(
    mac: MacSpec,
    model: LLMModel,
    quant: QuantFormat,
    context_length: int = 4096,
) -> GPUOptimizationResult:
    """Calculate optimal GPU configuration for the given hardware and model.

    Args:
        mac: Hardware specs
        model: Model architecture
        quant: Quantization format
        context_length: Desired context length

    Returns: GPUOptimizationResult with recommendations
    """
    from .analyzer import estimate_model_size_gb, estimate_tokens_per_second, estimate_kv_cache_gb

    # Calculate optimal settings
    gpu_layers, offload_pct = calculate_optimal_gpu_layers(
        mac, model, quant, context_length
    )
    batch_size = calculate_optimal_batch_size(
        mac, model, quant, gpu_layers, context_length
    )

    # Estimate memory usage
    gpu_mem, system_mem = estimate_gpu_memory_usage(
        mac, model, quant, context_length, batch_size, gpu_layers
    )

    model_size = estimate_model_size_gb(model.params_billion, quant.bits_per_weight)
    kv_cache = estimate_kv_cache_gb(model, context_length)

    # Estimate performance
    if offload_pct == 0:
        # Full GPU: use memory bandwidth
        est_tok_s = estimate_tokens_per_second(
            mac, model_size, model.params_billion
        ) or 0
    else:
        # Partial offload: slower due to CPU memory bandwidth
        from .analyzer import estimate_partial_offload_speed
        est_tok_s = estimate_partial_offload_speed(
            mac, model_size, 100 - offload_pct
        ) or 0

    # Generate recommendations
    notes = []
    recommendation = "Optimal"

    if offload_pct == 0:
        notes.append("✅ Full GPU offload - Maximum performance")
        notes.append(f"GPU utilization: ~{(gpu_mem/mac.usable_memory_gb)*100:.0f}%")
    elif offload_pct < 20:
        notes.append("⚠️ Minimal CPU offload - Good performance")
        recommendation = "Good"
    elif offload_pct < 50:
        notes.append("⚠️ Moderate CPU offload - Acceptable performance")
        recommendation = "Acceptable"
        notes.append("Consider reducing context length or batch size")
    else:
        notes.append("❌ High CPU offload - Consider smaller quantization")
        recommendation = "Suboptimal"
        notes.append(f"Try Q4_K_M ({model_size * 0.8:.1f}GB) or Q5_K_M ({model_size * 0.9:.1f}GB)")

    if batch_size < 1024:
        notes.append(f"📝 Reduced batch size ({batch_size}) for memory efficiency")

    if context_length > 4096:
        notes.append(f"⚠️ Large context ({context_length}) increases KV cache")

    return GPUOptimizationResult(
        gpu_layers=gpu_layers,
        batch_size=batch_size,
        context_length=context_length,
        model_size_gb=model_size,
        kv_cache_gb=kv_cache,
        total_memory_gb=model_size + kv_cache + estimate_model_size_gb(model.params_billion, quant.bits_per_weight) * 0.10 + 0.5,
        gpu_memory_gb=gpu_mem,
        system_memory_gb=system_mem,
        offload_percentage=offload_pct,
        estimated_tokens_per_second=est_tok_s or 0,
        estimated_latency_ms=1000 / (est_tok_s or 1),  # Approximate
        recommendation=recommendation,
        notes=notes,
    )


def compare_quantizations(
    mac: MacSpec,
    model: LLMModel,
    context_length: int = 4096,
) -> list[GPUOptimizationResult]:
    """Compare GPU optimization across different quantizations.

    Useful for finding the best quantization that fits entirely on GPU.
    """
    from .quantization import QUANT_FORMATS, RECOMMENDED_FORMATS

    results = []
    for quant_name in RECOMMENDED_FORMATS:
        if quant_name in QUANT_FORMATS:
            quant = QUANT_FORMATS[quant_name]
            result = optimize_for_hardware(mac, model, quant, context_length)
            results.append(result)

    # Sort by offload percentage (prefer full GPU)
    results.sort(key=lambda r: r.offload_percentage)

    return results
