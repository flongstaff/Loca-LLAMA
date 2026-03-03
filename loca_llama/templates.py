"""Model templates: recommended settings, system prompts, and configs.

Each template provides optimal settings for running a model on Apple Silicon
via LM Studio or llama.cpp, including:
- Recommended quantization for different memory budgets
- Context length suggestions
- GPU layer configuration
- Temperature and sampling parameters
- System prompts tuned for the model's chat template
- llama.cpp CLI flags
- LM Studio preset configuration
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelTemplate:
    """Recommended configuration for a specific model."""

    model_pattern: str  # Regex/substring to match model name
    family: str

    # Recommended quant for different memory budgets
    quant_24gb: str  # Best quant for 24GB unified memory
    quant_48gb: str  # Best quant for 48GB unified memory
    quant_64gb: str  # Best quant for 64GB+

    # Context recommendations
    recommended_ctx: int  # Sweet spot for quality + speed
    max_practical_ctx: int  # Max before quality or speed degrades badly

    # Sampling defaults
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    min_p: float

    # System prompt
    system_prompt: str
    chat_template: str  # "chatml", "llama3", "mistral", "phi3", "gemma", "command-r", "deepseek"

    # llama.cpp specific
    llama_cpp_flags: list[str] = field(default_factory=list)

    # Performance notes
    notes: str = ""

    # Benchmark reference scores (known community results on M4 Pro 48GB)
    bench_tok_s_q4: float | None = None  # Q4_K_M generation tok/s
    bench_tok_s_q8: float | None = None  # Q8_0 generation tok/s
    bench_prefill_q4: float | None = None  # Q4_K_M prefill tok/s


# ── Templates ────────────────────────────────────────────────────────────────

TEMPLATES: list[ModelTemplate] = [
    # ── Llama 3.1 / 3.2 / 3.3 ──

    ModelTemplate(
        model_pattern="Llama 3.2 1B",
        family="Llama",
        quant_24gb="Q8_0", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.6, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt="You are a helpful, concise assistant.",
        chat_template="llama3",
        notes="Very fast. Good for simple tasks, classification, extraction. Limited reasoning.",
        bench_tok_s_q4=120.0, bench_tok_s_q8=80.0, bench_prefill_q4=800.0,
    ),
    ModelTemplate(
        model_pattern="Llama 3.2 3B",
        family="Llama",
        quant_24gb="Q8_0", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.6, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt="You are a helpful, concise assistant.",
        chat_template="llama3",
        notes="Fast and capable. Good quality for a small model. Great for mobile/edge.",
        bench_tok_s_q4=55.0, bench_tok_s_q8=35.0, bench_prefill_q4=500.0,
    ),
    ModelTemplate(
        model_pattern="Llama 3.1 8B",
        family="Llama",
        quant_24gb="Q6_K", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt=(
            "You are a helpful, harmless, and honest assistant. "
            "You answer questions accurately and concisely."
        ),
        chat_template="llama3",
        llama_cpp_flags=[],
        notes="Excellent all-rounder at 8B. Strong instruction following. Use flash attention for long context.",
        bench_tok_s_q4=50.0, bench_tok_s_q8=30.0, bench_prefill_q4=450.0,
    ),
    ModelTemplate(
        model_pattern="Llama 3.1 70B",
        family="Llama",
        quant_24gb="Q3_K_M", quant_48gb="Q4_K_M", quant_64gb="Q6_K",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt=(
            "You are a helpful, harmless, and honest assistant. "
            "You provide detailed, well-reasoned answers."
        ),
        chat_template="llama3",
        llama_cpp_flags=[],
        notes="Top-tier open model. On 48GB: Q4_K_M barely fits, limit context to 4-8K. Very slow generation.",
        bench_tok_s_q4=6.0, bench_tok_s_q8=None, bench_prefill_q4=40.0,
    ),
    ModelTemplate(
        model_pattern="Llama 3.3 70B",
        family="Llama",
        quant_24gb="Q3_K_M", quant_48gb="Q4_K_M", quant_64gb="Q6_K",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt=(
            "You are a helpful, harmless, and honest assistant. "
            "You provide detailed, well-reasoned answers."
        ),
        chat_template="llama3",
        llama_cpp_flags=[],
        notes="Latest Llama 70B. Improved over 3.1. Same memory constraints on 48GB.",
        bench_tok_s_q4=6.0, bench_tok_s_q8=None, bench_prefill_q4=40.0,
    ),

    # ── Qwen 2.5 ──

    ModelTemplate(
        model_pattern="Qwen 2.5 7B",
        family="Qwen",
        quant_24gb="Q6_K", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05, min_p=0.05,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        chat_template="chatml",
        notes="Strong multilingual. Excellent for Chinese + English tasks.",
        bench_tok_s_q4=52.0, bench_tok_s_q8=28.0, bench_prefill_q4=420.0,
    ),
    ModelTemplate(
        model_pattern="Qwen 2.5 14B",
        family="Qwen",
        quant_24gb="Q5_K_M", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05, min_p=0.05,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        chat_template="chatml",
        notes="Sweet spot of quality vs speed. Very capable for its size.",
        bench_tok_s_q4=27.0, bench_tok_s_q8=16.0, bench_prefill_q4=250.0,
    ),
    ModelTemplate(
        model_pattern="Qwen 2.5 32B",
        family="Qwen",
        quant_24gb="Q4_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=8192, max_practical_ctx=65536,
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05, min_p=0.05,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        chat_template="chatml",
        notes="Best bang-for-buck on 48GB. Q6_K fits comfortably with room for 64K context.",
        bench_tok_s_q4=12.0, bench_tok_s_q8=7.5, bench_prefill_q4=150.0,
    ),
    ModelTemplate(
        model_pattern="Qwen 2.5 72B",
        family="Qwen",
        quant_24gb="Q2_K", quant_48gb="Q3_K_L", quant_64gb="Q5_K_M",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05, min_p=0.05,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        chat_template="chatml",
        llama_cpp_flags=[],
        notes="Near GPT-4 quality. On 48GB: Q3_K_L fits but slow (~8 tok/s). Worth it for quality.",
        bench_tok_s_q4=None, bench_tok_s_q8=None, bench_prefill_q4=None,
    ),
    ModelTemplate(
        model_pattern="Qwen 2.5 Coder",
        family="Qwen",
        quant_24gb="Q4_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=16384, max_practical_ctx=131072,
        temperature=0.3, top_p=0.9, top_k=20, repeat_penalty=1.0, min_p=0.05,
        system_prompt=(
            "You are Qwen Coder, an expert programming assistant. "
            "Write clean, efficient, well-documented code. "
            "Explain your reasoning step by step when solving complex problems."
        ),
        chat_template="chatml",
        notes="Top-tier coding model. Lower temperature (0.2-0.4) for code. Higher context useful for large files.",
        bench_tok_s_q4=12.0, bench_tok_s_q8=7.5, bench_prefill_q4=150.0,
    ),

    # ── DeepSeek ──

    ModelTemplate(
        model_pattern="DeepSeek-R1-Distill.*7B",
        family="DeepSeek",
        quant_24gb="Q6_K", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.6, top_p=0.95, top_k=50, repeat_penalty=1.0, min_p=0.05,
        system_prompt="",
        chat_template="deepseek",
        notes="Reasoning model (shows chain-of-thought in <think> tags). Great for math/logic.",
        bench_tok_s_q4=52.0, bench_tok_s_q8=28.0, bench_prefill_q4=420.0,
    ),
    ModelTemplate(
        model_pattern="DeepSeek-R1-Distill.*14B",
        family="DeepSeek",
        quant_24gb="Q5_K_M", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=8192, max_practical_ctx=131072,
        temperature=0.6, top_p=0.95, top_k=50, repeat_penalty=1.0, min_p=0.05,
        system_prompt="",
        chat_template="deepseek",
        notes="Best reasoning model for 48GB. Excellent math/science/logic.",
        bench_tok_s_q4=27.0, bench_tok_s_q8=16.0, bench_prefill_q4=250.0,
    ),
    ModelTemplate(
        model_pattern="DeepSeek-R1-Distill.*32B",
        family="DeepSeek",
        quant_24gb="Q3_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=8192, max_practical_ctx=65536,
        temperature=0.6, top_p=0.95, top_k=50, repeat_penalty=1.0, min_p=0.05,
        system_prompt="",
        chat_template="deepseek",
        notes="Very strong reasoning. Q6_K on 48GB gives excellent quality.",
        bench_tok_s_q4=12.0, bench_tok_s_q8=7.5, bench_prefill_q4=150.0,
    ),

    # ── Mistral / Mixtral ──

    ModelTemplate(
        model_pattern="Mistral 7B",
        family="Mistral",
        quant_24gb="Q6_K", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=4096, max_practical_ctx=32768,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.0,
        system_prompt="",
        chat_template="mistral",
        notes="Fast and efficient. Good general purpose. Mistral chat format (no system prompt).",
        bench_tok_s_q4=54.0, bench_tok_s_q8=30.0, bench_prefill_q4=440.0,
    ),
    ModelTemplate(
        model_pattern="Mistral Small 24B",
        family="Mistral",
        quant_24gb="Q4_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=4096, max_practical_ctx=32768,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.0,
        system_prompt="",
        chat_template="mistral",
        notes="Strong 24B model. Fits well on 48GB at Q6_K.",
        bench_tok_s_q4=15.0, bench_tok_s_q8=9.0, bench_prefill_q4=180.0,
    ),
    ModelTemplate(
        model_pattern="Mixtral 8x7B",
        family="Mixtral",
        quant_24gb="Q3_K_M", quant_48gb="Q4_K_M", quant_64gb="Q6_K",
        recommended_ctx=4096, max_practical_ctx=32768,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.0,
        system_prompt="",
        chat_template="mistral",
        notes="MoE model: 47B total but only 13B active per token. Faster than dense 47B.",
        bench_tok_s_q4=10.0, bench_tok_s_q8=6.0, bench_prefill_q4=100.0,
    ),

    # ── Phi ──

    ModelTemplate(
        model_pattern="Phi-3 Mini",
        family="Phi",
        quant_24gb="Q8_0", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=4096, max_practical_ctx=128000,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt="You are a helpful AI assistant.",
        chat_template="phi3",
        notes="Very small, very fast. Good for quick tasks. 128K context supported.",
        bench_tok_s_q4=60.0, bench_tok_s_q8=38.0, bench_prefill_q4=500.0,
    ),
    ModelTemplate(
        model_pattern="Phi-4",
        family="Phi",
        quant_24gb="Q5_K_M", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05,
        system_prompt="You are a helpful AI assistant.",
        chat_template="phi3",
        notes="Punches above its weight. Excellent reasoning for 14B. Great on Apple Silicon.",
        bench_tok_s_q4=25.0, bench_tok_s_q8=15.0, bench_prefill_q4=230.0,
    ),

    # ── Gemma ──

    ModelTemplate(
        model_pattern="Gemma 2 9B",
        family="Gemma",
        quant_24gb="Q6_K", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=4096, max_practical_ctx=8192,
        temperature=0.7, top_p=0.95, top_k=64, repeat_penalty=1.0, min_p=0.0,
        system_prompt="",
        chat_template="gemma",
        notes="Google's strong 9B model. Max 8K context is a limitation.",
        bench_tok_s_q4=48.0, bench_tok_s_q8=26.0, bench_prefill_q4=380.0,
    ),
    ModelTemplate(
        model_pattern="Gemma 2 27B",
        family="Gemma",
        quant_24gb="Q4_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=4096, max_practical_ctx=8192,
        temperature=0.7, top_p=0.95, top_k=64, repeat_penalty=1.0, min_p=0.0,
        system_prompt="",
        chat_template="gemma",
        notes="Very capable 27B model. Only 8K context unfortunately.",
        bench_tok_s_q4=13.0, bench_tok_s_q8=8.0, bench_prefill_q4=160.0,
    ),

    # ── CodeLlama ──

    ModelTemplate(
        model_pattern="CodeLlama 34B",
        family="CodeLlama",
        quant_24gb="Q3_K_M", quant_48gb="Q5_K_M", quant_64gb="Q8_0",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.2, top_p=0.9, top_k=40, repeat_penalty=1.0, min_p=0.05,
        system_prompt="You are an expert programmer. Write clean, efficient code.",
        chat_template="llama3",
        notes="Code-specialized. Use very low temperature (0.1-0.3) for code generation.",
        bench_tok_s_q4=10.0, bench_tok_s_q8=6.0, bench_prefill_q4=120.0,
    ),

    # ── Command-R ──

    ModelTemplate(
        model_pattern="Command-R 35B",
        family="Command",
        quant_24gb="Q4_K_M", quant_48gb="Q6_K", quant_64gb="Q8_0",
        recommended_ctx=4096, max_practical_ctx=131072,
        temperature=0.3, top_p=0.75, top_k=0, repeat_penalty=1.0, min_p=0.0,
        system_prompt=(
            "You are Command-R, a large language model built by Cohere. "
            "You are helpful, harmless, and honest."
        ),
        chat_template="command-r",
        notes="Excellent for RAG and tool use. 128K context. Unique sampling (low temp, low top_p).",
        bench_tok_s_q4=10.0, bench_tok_s_q8=6.0, bench_prefill_q4=120.0,
    ),

    # ── StarCoder ──

    ModelTemplate(
        model_pattern="StarCoder2 15B",
        family="StarCoder",
        quant_24gb="Q5_K_M", quant_48gb="Q8_0", quant_64gb="FP16",
        recommended_ctx=4096, max_practical_ctx=16384,
        temperature=0.2, top_p=0.95, top_k=50, repeat_penalty=1.0, min_p=0.05,
        system_prompt="",
        chat_template="chatml",
        notes="Code completion model. Best with fill-in-the-middle prompts. Use low temperature.",
        bench_tok_s_q4=18.0, bench_tok_s_q8=11.0, bench_prefill_q4=200.0,
    ),
]


def get_template(model_name: str) -> ModelTemplate | None:
    """Find the best matching template for a model name."""
    import re
    for t in TEMPLATES:
        if re.search(t.model_pattern, model_name, re.IGNORECASE):
            return t
    # Fallback: match by family
    for t in TEMPLATES:
        if t.family.lower() in model_name.lower():
            return t
    return None


def get_lm_studio_preset(template: ModelTemplate, model_name: str) -> dict:
    """Generate an LM Studio preset configuration."""
    return {
        "name": f"Loca-LLAMA: {model_name}",
        "inference_params": {
            "temperature": template.temperature,
            "top_p": template.top_p,
            "top_k": template.top_k,
            "min_p": template.min_p,
            "repeat_penalty": template.repeat_penalty,
            "n_predict": -1,
            "seed": -1,
            "stream": True,
        },
        "context_length": template.recommended_ctx,
        "system_prompt": template.system_prompt,
    }


def get_llama_cpp_command(
    template: ModelTemplate,
    model_path: str,
    context_length: int | None = None,
    n_gpu_layers: int = -1,
    sampling_overrides: dict[str, float | int] | None = None,
) -> str:
    """Generate the llama.cpp CLI command for this model."""
    ctx = context_length or template.recommended_ctx
    overrides = sampling_overrides or {}
    temp = overrides.get("temperature", template.temperature)
    top_p = overrides.get("top_p", template.top_p)
    top_k = overrides.get("top_k", template.top_k)
    rep_pen = overrides.get("repeat_penalty", template.repeat_penalty)
    min_p = overrides.get("min_p", template.min_p)

    parts = [
        "llama-cli",
        f"-m {model_path}",
        f"-c {ctx}",
        f"-ngl {n_gpu_layers}",
        f"--temp {temp}",
        f"--top-p {top_p}",
        f"--top-k {top_k}",
        f"--repeat-penalty {rep_pen}",
        f"--min-p {min_p}",
    ]
    if template.system_prompt:
        parts.append(f'--system-prompt "{template.system_prompt}"')
    parts.extend(template.llama_cpp_flags)
    parts.extend(["-fa", "--jinja", "--color"])
    return " \\\n  ".join(parts)


def get_llama_cpp_server_command(
    template: ModelTemplate,
    model_path: str,
    port: int = 8080,
    context_length: int | None = None,
    n_gpu_layers: int = -1,
) -> str:
    """Generate the llama-server command for this model."""
    ctx = context_length or template.recommended_ctx
    parts = [
        "llama-server",
        f"-m {model_path}",
        f"--port {port}",
        f"-c {ctx}",
        f"-ngl {n_gpu_layers}",
        "-fa",
    ]
    parts.extend(template.llama_cpp_flags)
    return " \\\n  ".join(parts)


def detect_model_format_warning(model_path: str) -> str | None:
    """Return a warning string if the model path suggests a non-GGUF format."""
    path_lower = model_path.lower()
    if "mlx" in path_lower:
        return "This model appears to be MLX format — llama-cli requires GGUF"
    if path_lower.endswith(".safetensors"):
        return "This model appears to be safetensors format — llama-cli requires GGUF"
    if path_lower.endswith(".bin") and "pytorch" in path_lower:
        return "This model appears to be PyTorch format — llama-cli requires GGUF"
    return None
