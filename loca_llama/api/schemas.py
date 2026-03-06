"""Pydantic request/response models for the Loca-LLAMA API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Hardware ─────────────────────────────────────────────────────────────────

class HardwareResponse(BaseModel):
    name: str
    chip: str
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    memory_gb: int
    memory_bandwidth_gbs: float
    gpu_tflops: float
    usable_memory_gb: float


class HardwareListResponse(BaseModel):
    hardware: list[HardwareResponse]
    count: int


# ── Models ───────────────────────────────────────────────────────────────────

class ModelResponse(BaseModel):
    name: str
    family: str
    params_billion: float
    default_context_length: int
    max_context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    license: str


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    count: int
    families: list[str]


# ── Quantization ─────────────────────────────────────────────────────────────

class QuantResponse(BaseModel):
    name: str
    bits_per_weight: float
    quality_rating: str
    description: str


class QuantListResponse(BaseModel):
    formats: list[QuantResponse]
    recommended: list[str]


# ── Analysis ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    hardware_name: str
    model_name: str
    quant_name: str
    context_length: int | None = None


class AnalyzeResponse(BaseModel):
    model_name: str
    quant_name: str
    context_length: int
    model_size_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_memory_gb: float
    available_memory_gb: float
    headroom_gb: float
    fits_in_memory: bool
    tier: str
    tier_label: str
    memory_utilization_pct: float
    estimated_tok_per_sec: float | None
    gpu_layers: int | None
    total_layers: int | None
    offload_pct: float | None


class AnalyzeAllRequest(BaseModel):
    hardware_name: str
    quant_names: list[str] | None = None
    context_length: int | None = None
    only_fits: bool = False
    include_partial: bool = False
    family: str | None = None


class TierSummary(BaseModel):
    full_gpu: int = 0
    comfortable: int = 0
    tight_fit: int = 0
    partial_offload: int = 0
    wont_fit: int = 0


class AnalyzeAllResponse(BaseModel):
    results: list[AnalyzeResponse]
    count: int
    hardware: str
    summary: TierSummary


class MaxContextRequest(BaseModel):
    hardware_name: str
    model_name: str
    quant_name: str


class MaxContextResponse(BaseModel):
    model_name: str
    quant_name: str
    max_context_length: int
    max_context_k: str


# ── Calculator ──────────────────────────────────────────────────────────────

class CalculatorEstimateRequest(BaseModel):
    params_billion: float = Field(..., ge=0.1, le=1000.0)
    bits_per_weight: float = Field(..., ge=1.0, le=32.0)
    context_length: int = Field(..., ge=128, le=262144)
    num_layers: int = Field(..., ge=1, le=200)
    num_kv_heads: int = Field(..., ge=1, le=128)
    head_dim: int = Field(..., ge=32, le=512)
    kv_bits: Literal[4, 8, 16] = 16


class HardwareCompatibilityItem(BaseModel):
    name: str
    memory_gb: int
    tier: str
    tier_label: str
    headroom_gb: float
    estimated_tok_per_sec: float | None


class CalculatorEstimateResponse(BaseModel):
    model_size_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_memory_gb: float
    on_disk_size_gb: float
    compatible_hardware: list[HardwareCompatibilityItem]


class CalculatorModelItem(BaseModel):
    name: str
    family: str
    params_billion: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    default_context_length: int
    max_context_length: int


class CalculatorModelsResponse(BaseModel):
    models: list[CalculatorModelItem]
    count: int
    families: list[str]


# ── Templates ────────────────────────────────────────────────────────────────

class TemplateResponse(BaseModel):
    model_pattern: str
    family: str
    quant_24gb: str | None = None
    quant_48gb: str | None = None
    quant_64gb: str | None = None
    recommended_ctx: int | None = None
    max_practical_ctx: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    min_p: float | None = None
    system_prompt: str | None = None
    chat_template: str | None = None
    llama_cpp_flags: list[str] = Field(default_factory=list)
    notes: str | None = None
    bench_tok_s_q4: float | None = None
    bench_tok_s_q8: float | None = None
    bench_prefill_q4: float | None = None


class TemplateListResponse(BaseModel):
    templates: list[TemplateResponse]
    count: int


class LMStudioPresetRequest(BaseModel):
    model_name: str


class LMStudioPresetResponse(BaseModel):
    name: str
    inference_params: dict
    context_length: int | None = None
    system_prompt: str | None = None


class LlamaCppCommandRequest(BaseModel):
    model_name: str
    model_path: str
    context_length: int | None = None
    n_gpu_layers: int = -1
    sampling_overrides: dict[str, float | int] | None = None


class LlamaCppCommandResponse(BaseModel):
    command: str


# ── Scanner ──────────────────────────────────────────────────────────────────

class LocalModelResponse(BaseModel):
    name: str
    path: str
    size_gb: float
    format: str
    source: str
    quant: str | None = None
    family: str | None = None
    repo_id: str | None = None


class ScannerResponse(BaseModel):
    models: list[LocalModelResponse]
    count: int
    total_size_gb: float
    sources: dict[str, int]


# ── Hub (HuggingFace) ───────────────────────────────────────────────────────

class HubModelResponse(BaseModel):
    repo_id: str
    name: str
    author: str
    downloads: int
    likes: int
    tags: list[str]
    pipeline_tag: str | None = None
    is_mlx: bool
    is_gguf: bool
    last_modified: str | None = None


class HubSearchResponse(BaseModel):
    results: list[HubModelResponse]
    count: int
    query: str


class HubFileResponse(BaseModel):
    filename: str
    size: int


class HubFilesResponse(BaseModel):
    repo_id: str
    files: list[HubFileResponse]


class HubConfigResponse(BaseModel):
    repo_id: str
    model_type: str = ""
    architecture: str = ""
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    max_position_embeddings: int = 0
    vocab_size: int = 0
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    chat_template: str | None = None
    license: str = ""
    tags: list[str] = Field(default_factory=list)


# ── Runtime ──────────────────────────────────────────────────────────────────

class RuntimeResponse(BaseModel):
    name: str
    url: str
    models: list[str]
    version: str | None = None


class RuntimeStatusResponse(BaseModel):
    runtimes: list[RuntimeResponse]
    count: int


# ── Benchmark ────────────────────────────────────────────────────────────────

class BenchmarkStartRequest(BaseModel):
    runtime_name: str
    model_id: str
    prompt_type: str = "default"
    num_runs: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=200, ge=1, le=4096)
    context_length: int = 4096
    custom_prompt: str | None = None


class BenchmarkStartResponse(BaseModel):
    job_id: str
    status: str


class BenchmarkProgress(BaseModel):
    current_run: int
    total_runs: int


class BenchmarkRunResult(BaseModel):
    run_number: int
    success: bool
    tokens_per_second: float
    prompt_tokens_per_second: float
    time_to_first_token_ms: float
    total_time_ms: float
    prompt_tokens: int
    generated_tokens: int


class BenchmarkAggregate(BaseModel):
    avg_tok_per_sec: float = 0.0
    min_tok_per_sec: float = 0.0
    max_tok_per_sec: float = 0.0
    avg_prefill_tok_per_sec: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_total_ms: float = 0.0
    total_tokens_generated: int = 0
    runs: int = 0


class BenchmarkStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: BenchmarkProgress | None = None
    runs: list[BenchmarkRunResult] | None = None
    aggregate: BenchmarkAggregate | None = None
    error: str | None = None


class BenchmarkPromptsResponse(BaseModel):
    prompts: dict[str, str]


# ── Sweep ───────────────────────────────────────────────────────────────────

class SweepStartRequest(BaseModel):
    runtime_name: str
    model_ids: list[str] = Field(..., min_length=1, max_length=20)
    prompt_type: str = "default"
    num_runs: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=200, ge=1, le=4096)
    context_length: int = 4096
    custom_prompt: str | None = None


class SweepComboResult(BaseModel):
    model_id: str
    runs: list[BenchmarkRunResult] | None = None
    aggregate: BenchmarkAggregate | None = None


class SweepProgress(BaseModel):
    current_combo: int
    total_combos: int
    current_run_in_combo: int
    total_runs_per_combo: int


class SweepStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: SweepProgress | None = None
    combo_results: list[SweepComboResult] | None = None
    error: str | None = None


# ── Memory ───────────────────────────────────────────────────────────────────

class MemoryCurrentResponse(BaseModel):
    used_gb: float
    free_gb: float
    total_gb: float
    usage_pct: float
    pressure: str


class MemorySampleResponse(BaseModel):
    timestamp: float
    used_gb: float
    free_gb: float
    total_gb: float
    usage_pct: float
    pressure: str


class MemoryHistoryResponse(BaseModel):
    samples: list[MemorySampleResponse]
    count: int


class MemoryReportResponse(BaseModel):
    peak_used_gb: float
    baseline_used_gb: float
    delta_gb: float
    total_gb: float
    peak_pct: float
    baseline_pct: float
    duration_sec: float
    sample_count: int
