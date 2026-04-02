"""Pydantic request/response models for the Loca-LLAMA API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


class HardwareDetectResponse(BaseModel):
    detected: bool
    name: str | None = None
    chip: str | None = None
    memory_gb: int | None = None
    reason: str | None = None


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


class ModelDetailResponse(BaseModel):
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
    hardware_name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=200)
    quant_name: str = Field(..., min_length=1, max_length=50)
    context_length: int | None = Field(None, ge=128, le=131072)


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
    hardware_name: str = Field(..., min_length=1, max_length=100)
    quant_names: list[str] | None = None
    context_length: int | None = Field(None, ge=128, le=131072)
    only_fits: bool = False
    include_partial: bool = False
    family: str | None = Field(None, max_length=100)


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
    hardware_name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=200)
    quant_name: str = Field(..., min_length=1, max_length=50)


class MaxContextResponse(BaseModel):
    model_name: str
    quant_name: str
    max_context_length: int
    max_context_k: str


# ── Recommend ───────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    hardware_name: str = Field(..., min_length=1, max_length=100)
    use_case: Literal["general", "coding", "reasoning", "small", "large-context"] = "general"


class RecommendItem(BaseModel):
    rank: int
    model_name: str
    quant_name: str
    tier: str
    tier_label: str
    model_size_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_memory_gb: float
    available_memory_gb: float
    headroom_gb: float
    memory_utilization_pct: float
    estimated_tok_per_sec: float | None
    gpu_layers: int | None
    total_layers: int | None
    offload_pct: float | None
    context_length: int
    max_context_k: str


class RecommendResponse(BaseModel):
    recommendations: list[RecommendItem]
    count: int
    hardware: str
    use_case: str


# ── Calculator ──────────────────────────────────────────────────────────────

class CalculatorEstimateRequest(BaseModel):
    params_billion: float = Field(..., ge=0.1, le=1000.0)
    bits_per_weight: float = Field(..., ge=1.0, le=32.0)
    context_length: int = Field(..., ge=128, le=262144)
    num_layers: int = Field(..., ge=1, le=200)
    num_kv_heads: int = Field(..., ge=1, le=128)
    head_dim: int = Field(..., ge=32, le=512)
    kv_bits: Literal[4, 8, 16] = 16
    inference_mode: Literal["streaming", "batch"] = "streaming"


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
    model_name: str = Field(..., min_length=1, max_length=200)


class LMStudioPresetResponse(BaseModel):
    name: str
    inference_params: dict
    context_length: int | None = None
    system_prompt: str | None = None


_MODEL_PATH_RE = re.compile(r"^[a-zA-Z0-9._/ -]+$")


class LlamaCppCommandRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=200)
    model_path: str = Field(..., min_length=1, max_length=500)
    context_length: int | None = Field(None, ge=128, le=131072)
    n_gpu_layers: int = Field(-1, ge=-1, le=999)
    sampling_overrides: dict[str, float | int] | None = None

    @field_validator("model_path")
    @classmethod
    def reject_shell_metacharacters(cls, v: str) -> str:
        if not _MODEL_PATH_RE.match(v):
            raise ValueError(
                "model_path must only contain alphanumeric characters, "
                "dots, underscores, slashes, hyphens, and spaces"
            )
        return v


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
    context_length: int = Field(default=4096, ge=128, le=262144)
    custom_prompt: str | None = Field(default=None, max_length=10000)


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
    median_tok_per_sec: float = 0.0
    p95_tok_per_sec: float = 0.0
    stddev_tok_per_sec: float = 0.0
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
    context_length: int = Field(default=4096, ge=128, le=262144)
    custom_prompt: str | None = Field(default=None, max_length=10000)


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

# ── Throughput ──────────────────────────────────────────────────────────────

class ThroughputRequestResult(BaseModel):
    request_id: int
    success: bool
    tokens_generated: int = 0
    elapsed_ms: float = 0.0
    tokens_per_second: float = 0.0
    error: str | None = None


class ThroughputStartRequest(BaseModel):
    runtime_name: str
    model_id: str
    concurrency: int = Field(default=4, ge=1, le=32)
    total_requests: int = Field(default=8, ge=1, le=100)
    prompt: str = Field(
        default="Write a brief explanation of how neural networks work.",
        max_length=10000,
    )
    max_tokens: int = Field(default=100, ge=1, le=2048)


class ThroughputResponse(BaseModel):
    job_id: str
    status: str
    concurrency: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    throughput_tps: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_rate: float = 0.0
    per_request: list[ThroughputRequestResult] | None = None
    error: str | None = None


# ── Compare ─────────────────────────────────────────────────────────────────

class CompareStartRequest(BaseModel):
    runtime_a: str
    runtime_b: str
    model_id: str
    prompt_type: str = "default"
    num_runs: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=200, ge=1, le=4096)
    context_length: int = Field(default=4096, ge=128, le=262144)
    custom_prompt: str | None = Field(default=None, max_length=10000)


class CompareResult(BaseModel):
    runtime_name: str
    aggregate: BenchmarkAggregate | None = None


class CompareResponse(BaseModel):
    job_id: str
    status: str
    results: list[CompareResult] | None = None
    speedup_pct: float | None = None
    faster_runtime: str | None = None
    error: str | None = None


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


# ── SQL Benchmark ────────────────────────────────────────────────────────────

class SqlBenchStartRequest(BaseModel):
    runtime_name: str
    model_ids: list[str] = Field(..., min_length=1, max_length=10)
    difficulties: list[str] | None = None
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("difficulties")
    @classmethod
    def validate_difficulties(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            valid = {"trivial", "easy", "medium", "hard"}
            for d in v:
                if d.lower() not in valid:
                    raise ValueError(f"Invalid difficulty: {d}. Must be one of {valid}")
        return v


class SqlBenchQuestionResult(BaseModel):
    question_id: int
    question: str
    difficulty: str
    status: str  # "pass", "fail", "error"
    generated_sql: str = ""
    error_message: str = ""
    speed_tps: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    retries: int = 0


class SqlBenchModelResult(BaseModel):
    model_id: str
    total_pass: int = 0
    total_fail: int = 0
    total_error: int = 0
    score_by_difficulty: dict[str, dict[str, int]] = Field(default_factory=dict)
    avg_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    questions: list[SqlBenchQuestionResult] = Field(default_factory=list)


class SqlBenchProgress(BaseModel):
    current_model: int
    total_models: int
    current_question: int
    total_questions: int


class SqlBenchStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: SqlBenchProgress | None = None
    model_results: list[SqlBenchModelResult] | None = None
    error: str | None = None
