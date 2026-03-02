"""LLM model database with specifications."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModel:
    name: str
    family: str
    params_billion: float
    default_context_length: int
    max_context_length: int
    num_layers: int
    num_kv_heads: int  # for GQA models, this is the KV head count
    head_dim: int
    license: str


# Comprehensive model database
MODELS: list[LLMModel] = [
    # Llama 3.2
    LLMModel("Llama 3.2 1B", "Llama", 1.24, 4096, 131072, 16, 8, 64, "Llama 3.2"),
    LLMModel("Llama 3.2 3B", "Llama", 3.21, 4096, 131072, 28, 8, 128, "Llama 3.2"),
    # Llama 3.1
    LLMModel("Llama 3.1 8B", "Llama", 8.03, 8192, 131072, 32, 8, 128, "Llama 3.1"),
    LLMModel("Llama 3.1 70B", "Llama", 70.6, 8192, 131072, 80, 8, 128, "Llama 3.1"),
    LLMModel("Llama 3.1 405B", "Llama", 405.0, 8192, 131072, 126, 8, 128, "Llama 3.1"),
    # Llama 3.3
    LLMModel("Llama 3.3 70B", "Llama", 70.6, 8192, 131072, 80, 8, 128, "Llama 3.3"),
    # Mistral / Mixtral
    LLMModel("Mistral 7B v0.3", "Mistral", 7.25, 4096, 32768, 32, 8, 128, "Apache 2.0"),
    LLMModel("Mistral Nemo 12B", "Mistral", 12.2, 4096, 131072, 40, 8, 128, "Apache 2.0"),
    LLMModel("Mixtral 8x7B", "Mixtral", 46.7, 4096, 32768, 32, 8, 128, "Apache 2.0"),
    LLMModel("Mixtral 8x22B", "Mixtral", 141.0, 4096, 65536, 56, 8, 128, "Apache 2.0"),
    LLMModel("Mistral Small 24B", "Mistral", 24.0, 4096, 32768, 40, 8, 128, "Apache 2.0"),
    # Phi
    LLMModel("Phi-3 Mini 3.8B", "Phi", 3.82, 4096, 128000, 32, 32, 96, "MIT"),
    LLMModel("Phi-3 Small 7B", "Phi", 7.39, 4096, 128000, 32, 8, 96, "MIT"),
    LLMModel("Phi-3 Medium 14B", "Phi", 14.0, 4096, 128000, 40, 10, 128, "MIT"),
    LLMModel("Phi-4 14B", "Phi", 14.7, 4096, 16384, 40, 10, 128, "MIT"),
    # Gemma 2
    LLMModel("Gemma 2 2B", "Gemma", 2.61, 4096, 8192, 26, 4, 256, "Gemma"),
    LLMModel("Gemma 2 9B", "Gemma", 9.24, 4096, 8192, 42, 8, 256, "Gemma"),
    LLMModel("Gemma 2 27B", "Gemma", 27.23, 4096, 8192, 46, 16, 128, "Gemma"),
    # Qwen 2.5
    LLMModel("Qwen 2.5 0.5B", "Qwen", 0.49, 4096, 32768, 24, 2, 64, "Apache 2.0"),
    LLMModel("Qwen 2.5 1.5B", "Qwen", 1.54, 4096, 32768, 28, 2, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 3B", "Qwen", 3.09, 4096, 32768, 36, 2, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 7B", "Qwen", 7.62, 4096, 131072, 28, 4, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 14B", "Qwen", 14.77, 4096, 131072, 48, 4, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 32B", "Qwen", 32.76, 4096, 131072, 64, 8, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 72B", "Qwen", 72.71, 4096, 131072, 80, 8, 128, "Qwen"),
    # Qwen 2.5 Coder
    LLMModel("Qwen 2.5 Coder 7B", "Qwen", 7.62, 4096, 131072, 28, 4, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 Coder 14B", "Qwen", 14.77, 4096, 131072, 48, 4, 128, "Apache 2.0"),
    LLMModel("Qwen 2.5 Coder 32B", "Qwen", 32.76, 4096, 131072, 64, 8, 128, "Apache 2.0"),
    # DeepSeek
    LLMModel("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek", 1.78, 4096, 131072, 28, 2, 128, "MIT"),
    LLMModel("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek", 7.62, 4096, 131072, 28, 4, 128, "MIT"),
    LLMModel("DeepSeek-R1-Distill-Qwen-14B", "DeepSeek", 14.77, 4096, 131072, 48, 4, 128, "MIT"),
    LLMModel("DeepSeek-R1-Distill-Qwen-32B", "DeepSeek", 32.76, 4096, 131072, 64, 8, 128, "MIT"),
    LLMModel("DeepSeek-R1-Distill-Llama-8B", "DeepSeek", 8.03, 4096, 131072, 32, 8, 128, "MIT"),
    LLMModel("DeepSeek-R1-Distill-Llama-70B", "DeepSeek", 70.6, 4096, 131072, 80, 8, 128, "MIT"),
    LLMModel("DeepSeek V3 671B", "DeepSeek", 671.0, 4096, 131072, 61, 128, 128, "DeepSeek"),
    # Command-R
    LLMModel("Command-R 35B", "Command", 35.0, 4096, 131072, 40, 8, 128, "CC-BY-NC"),
    LLMModel("Command-R+ 104B", "Command", 104.0, 4096, 131072, 64, 8, 128, "CC-BY-NC"),
    # CodeLlama
    LLMModel("CodeLlama 7B", "CodeLlama", 6.74, 4096, 16384, 32, 32, 128, "Llama 2"),
    LLMModel("CodeLlama 13B", "CodeLlama", 13.02, 4096, 16384, 40, 40, 128, "Llama 2"),
    LLMModel("CodeLlama 34B", "CodeLlama", 33.74, 4096, 16384, 48, 8, 128, "Llama 2"),
    # Yi
    LLMModel("Yi 1.5 6B", "Yi", 6.06, 4096, 32768, 32, 4, 128, "Apache 2.0"),
    LLMModel("Yi 1.5 9B", "Yi", 8.83, 4096, 32768, 32, 4, 128, "Apache 2.0"),
    LLMModel("Yi 1.5 34B", "Yi", 34.39, 4096, 32768, 60, 8, 128, "Apache 2.0"),
    # StarCoder 2
    LLMModel("StarCoder2 3B", "StarCoder", 3.03, 4096, 16384, 30, 2, 128, "BigCode-OpenRAIL-M"),
    LLMModel("StarCoder2 7B", "StarCoder", 6.74, 4096, 16384, 32, 4, 128, "BigCode-OpenRAIL-M"),
    LLMModel("StarCoder2 15B", "StarCoder", 15.73, 4096, 16384, 40, 8, 128, "BigCode-OpenRAIL-M"),
    # Nous / Hermes
    LLMModel("Nous Hermes 2 Mistral 7B", "Nous", 7.25, 4096, 32768, 32, 8, 128, "Apache 2.0"),
    # Falcon
    LLMModel("Falcon 7B", "Falcon", 6.92, 2048, 2048, 32, 1, 64, "Apache 2.0"),
    LLMModel("Falcon 40B", "Falcon", 40.0, 2048, 2048, 60, 8, 128, "Apache 2.0"),
]
