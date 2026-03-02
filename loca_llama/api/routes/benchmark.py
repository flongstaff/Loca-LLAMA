"""Benchmark execution endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from loca_llama.api.dependencies import get_state
from loca_llama.api.schemas import (
    BenchmarkAggregate,
    BenchmarkProgress,
    BenchmarkPromptsResponse,
    BenchmarkRunResult,
    BenchmarkStartRequest,
    BenchmarkStartResponse,
    BenchmarkStatusResponse,
)
from loca_llama.api.state import AppState, BenchmarkJob
from loca_llama.benchmark import (
    BENCH_PROMPTS,
    RuntimeInfo,
    aggregate_results,
    detect_all_runtimes,
    run_benchmark_suite,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/prompts", response_model=BenchmarkPromptsResponse)
async def list_prompts() -> BenchmarkPromptsResponse:
    """Return available benchmark prompt types."""
    return BenchmarkPromptsResponse(prompts=BENCH_PROMPTS)


@router.post("/start", response_model=BenchmarkStartResponse)
async def start_benchmark(
    req: BenchmarkStartRequest,
    state: AppState = Depends(get_state),
) -> BenchmarkStartResponse:
    """Start a benchmark run as a background task."""
    # Detect runtimes and find the requested one
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception as e:
        logger.error("Runtime detection error during benchmark start: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Runtime detection failed: check server logs for details",
        )

    runtime = next((r for r in runtimes if r.name == req.runtime_name), None)
    if runtime is None:
        available = [r.name for r in runtimes]
        raise HTTPException(
            status_code=400,
            detail=f"Runtime '{req.runtime_name}' not found. Available: {available}",
        )

    # Validate model is loaded in the runtime
    if req.model_id not in runtime.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_id}' not loaded in {req.runtime_name}. "
            f"Available: {runtime.models}",
        )

    # Validate prompt type
    if req.prompt_type not in BENCH_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt_type '{req.prompt_type}'. "
            f"Available: {list(BENCH_PROMPTS.keys())}",
        )

    # Create job and launch background task
    async with state._lock:
        job = state.create_benchmark_job(req.runtime_name, req.model_id, req.num_runs)

    task = asyncio.create_task(
        _run_benchmark_background(
            state, job, runtime, req.prompt_type, req.max_tokens, req.context_length
        )
    )
    # Hold a reference so the task isn't garbage-collected
    job.task = task  # type: ignore[attr-defined]

    return BenchmarkStartResponse(job_id=job.job_id, status=job.status)


@router.get("/{job_id}", response_model=BenchmarkStatusResponse)
async def get_benchmark_status(
    job_id: str,
    state: AppState = Depends(get_state),
) -> BenchmarkStatusResponse:
    """Poll benchmark job status, progress, and results."""
    job = state.benchmark_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    progress = BenchmarkProgress(current_run=job.current_run, total_runs=job.num_runs)

    runs = [
        BenchmarkRunResult(
            run_number=r.run_number,
            success=r.success,
            tokens_per_second=round(r.tokens_per_second, 1),
            prompt_tokens_per_second=round(r.prompt_tokens_per_second, 1),
            time_to_first_token_ms=round(r.time_to_first_token_ms, 1),
            total_time_ms=round(r.total_time_ms, 1),
            prompt_tokens=r.prompt_tokens,
            generated_tokens=r.generated_tokens,
        )
        for r in job.results
    ] if job.results else None

    aggregate = None
    if job.aggregate:
        a = job.aggregate
        aggregate = BenchmarkAggregate(
            avg_tok_per_sec=round(a.get("avg_tok_per_sec", 0), 1),
            min_tok_per_sec=round(a.get("min_tok_per_sec", 0), 1),
            max_tok_per_sec=round(a.get("max_tok_per_sec", 0), 1),
            avg_prefill_tok_per_sec=round(a.get("avg_prefill_tok_per_sec", 0), 1),
            avg_ttft_ms=round(a.get("avg_ttft_ms", 0), 1),
            avg_total_ms=round(a.get("avg_total_ms", 0), 1),
            total_tokens_generated=a.get("total_tokens_generated", 0),
            runs=a.get("runs", 0),
        )

    return BenchmarkStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=progress,
        runs=runs,
        aggregate=aggregate,
        error=job.error,
    )


async def _run_benchmark_background(
    state: AppState,
    job: BenchmarkJob,
    runtime: RuntimeInfo,
    prompt_type: str,
    max_tokens: int,
    context_length: int,
) -> None:
    """Run benchmark suite in a background thread with progress updates."""

    loop = asyncio.get_running_loop()

    def progress_callback(current: int, total: int) -> None:
        # Called from worker thread — use call_soon_threadsafe to avoid data race
        loop.call_soon_threadsafe(setattr, job, "current_run", current)

    try:
        results = await asyncio.to_thread(
            run_benchmark_suite,
            runtime,
            job.model_id,
            prompt_type,
            job.num_runs,
            max_tokens,
            context_length,
            progress_callback,
        )
        job.results = results
        job.aggregate = aggregate_results(results)
        job.status = "complete"
    except Exception as e:
        logger.error("Benchmark job %s failed: %s", job.job_id, e)
        job.status = "error"
        job.error = "Benchmark run failed — check server logs for details"
    finally:
        state.cleanup_old_jobs()
