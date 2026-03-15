"""Benchmark execution endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from loca_llama.api.dependencies import get_state
from fastapi.responses import HTMLResponse

from loca_llama.api.schemas import (
    BenchmarkAggregate,
    BenchmarkProgress,
    BenchmarkPromptsResponse,
    BenchmarkRunResult,
    BenchmarkStartRequest,
    BenchmarkStartResponse,
    BenchmarkStatusResponse,
    CompareResult,
    CompareResponse,
    CompareStartRequest,
    SweepComboResult,
    SweepProgress,
    SweepStartRequest,
    SweepStatusResponse,
    ThroughputRequestResult,
    ThroughputResponse,
    ThroughputStartRequest,
)
from loca_llama.api.state import AppState, BenchmarkJob, SweepJob, ThroughputJob, CompareJob
from loca_llama.benchmark import (
    BENCH_PROMPTS,
    RuntimeInfo,
    aggregate_results,
    benchmark_openai_api_streaming,
    detect_all_runtimes,
    run_benchmark_suite,
    run_benchmark_sweep,
)
from loca_llama.benchmark_report import generate_html_report
from loca_llama.throughput import run_throughput_test

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/prompts", response_model=BenchmarkPromptsResponse)
async def list_prompts() -> BenchmarkPromptsResponse:
    """Return available benchmark prompt types."""
    try:
        return BenchmarkPromptsResponse(prompts=BENCH_PROMPTS)
    except Exception:
        logger.exception("Failed to list benchmark prompts")
        raise HTTPException(status_code=500, detail="Failed to list prompts")


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

    # Validate prompt type — "custom" requires custom_prompt text
    if req.prompt_type == "custom":
        if not req.custom_prompt or not req.custom_prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="custom_prompt is required when prompt_type is 'custom'",
            )
    elif req.prompt_type not in BENCH_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt_type '{req.prompt_type}'. "
            f"Available: {[*BENCH_PROMPTS.keys(), 'custom']}",
        )

    # Create job and launch background task
    async with state._lock:
        job = state.create_benchmark_job(req.runtime_name, req.model_id, req.num_runs)

    task = asyncio.create_task(
        _run_benchmark_background(
            state, job, runtime, req.prompt_type, req.max_tokens, req.context_length,
            custom_prompt=req.custom_prompt,
        )
    )
    # Hold a reference so the task isn't garbage-collected
    job.task = task

    return BenchmarkStartResponse(job_id=job.job_id, status=job.status)


@router.get("/stream")
async def stream_benchmark(
    runtime_name: str = Query(...),
    model_id: str = Query(...),
    prompt_type: str = Query(default="default"),
    custom_prompt: str | None = Query(default=None),
    max_tokens: int = Query(default=200, ge=1, le=4096),
) -> StreamingResponse:
    """Stream token-by-token benchmark via SSE.

    Events:
      - token: {"text": str, "elapsed_ms": float, "token_count": int}
      - metrics: {"tokens": int, "elapsed_ms": float, "tok_per_sec": float, "ttft_ms": float}
      - done: {"tokens": int, "elapsed_ms": float, "tok_per_sec": float, "ttft_ms": float}
      - error: {"message": str}
    """
    # Validate prompt
    if prompt_type == "custom":
        if not custom_prompt or not custom_prompt.strip():
            raise HTTPException(status_code=400, detail="custom_prompt required when prompt_type is 'custom'")
        prompt = custom_prompt.strip()
    elif prompt_type not in BENCH_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt_type '{prompt_type}'. Available: {[*BENCH_PROMPTS.keys(), 'custom']}",
        )
    else:
        prompt = BENCH_PROMPTS[prompt_type]

    # Detect runtime
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception as e:
        logger.error("Runtime detection failed for streaming: %s", e)
        raise HTTPException(status_code=500, detail="Runtime detection failed")

    runtime = next((r for r in runtimes if r.name == runtime_name), None)
    if runtime is None:
        raise HTTPException(
            status_code=400,
            detail=f"Runtime '{runtime_name}' not found. Available: {[r.name for r in runtimes]}",
        )

    if model_id not in runtime.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not loaded in {runtime_name}.",
        )

    async def event_generator():
        """Bridge sync streaming generator to async SSE events."""
        queue: asyncio.Queue[tuple[str, float] | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()

        def _run_streaming():
            try:
                for token_text, elapsed_ms in benchmark_openai_api_streaming(
                    runtime.url, model_id, prompt, max_tokens,
                    api_key=runtime.api_key,
                ):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, (token_text, elapsed_ms))
                loop.call_soon_threadsafe(queue.put_nowait, None)  # Signal completion
            except Exception as exc:
                if not stop_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, exc)

        # Run the blocking generator in a thread
        thread_future = loop.run_in_executor(None, _run_streaming)

        token_count = 0
        ttft_ms = 0.0
        last_elapsed = 0.0

        try:
            while True:
                item = await queue.get()

                if item is None:
                    # Stream complete — send final metrics
                    tok_per_sec = (token_count / last_elapsed * 1000) if last_elapsed > 0 else 0
                    done_data = json.dumps({
                        "tokens": token_count,
                        "elapsed_ms": round(last_elapsed, 1),
                        "tok_per_sec": round(tok_per_sec, 1),
                        "ttft_ms": round(ttft_ms, 1),
                    })
                    yield f"event: done\ndata: {done_data}\n\n"
                    break

                if isinstance(item, Exception):
                    error_data = json.dumps({"message": str(item)})
                    yield f"event: error\ndata: {error_data}\n\n"
                    break

                token_text, elapsed_ms = item
                token_count += 1
                last_elapsed = elapsed_ms

                if token_count == 1:
                    ttft_ms = elapsed_ms

                # Token event
                token_data = json.dumps({
                    "text": token_text,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "token_count": token_count,
                })
                yield f"event: token\ndata: {token_data}\n\n"

                # Periodic metrics every 5 tokens
                if token_count % 5 == 0:
                    tok_per_sec = (token_count / elapsed_ms * 1000) if elapsed_ms > 0 else 0
                    metrics_data = json.dumps({
                        "tokens": token_count,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "tok_per_sec": round(tok_per_sec, 1),
                        "ttft_ms": round(ttft_ms, 1),
                    })
                    yield f"event: metrics\ndata: {metrics_data}\n\n"
        except asyncio.CancelledError:
            stop_event.set()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sweep", response_model=BenchmarkStartResponse)
async def start_sweep(
    req: SweepStartRequest,
    state: AppState = Depends(get_state),
) -> BenchmarkStartResponse:
    """Start a sweep benchmark across multiple models."""
    # Detect runtimes and find the requested one
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception as e:
        logger.error("Runtime detection error during sweep start: %s", e)
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

    # Validate all requested models are loaded
    missing = [m for m in req.model_ids if m not in runtime.models]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Models not loaded in {req.runtime_name}: {missing}. "
            f"Available: {runtime.models}",
        )

    # Validate prompt type
    if req.prompt_type == "custom":
        if not req.custom_prompt or not req.custom_prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="custom_prompt is required when prompt_type is 'custom'",
            )
    elif req.prompt_type not in BENCH_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt_type '{req.prompt_type}'. "
            f"Available: {[*BENCH_PROMPTS.keys(), 'custom']}",
        )

    # Create sweep job and launch background task
    async with state._lock:
        job = state.create_sweep_job(req.runtime_name, req.model_ids, req.num_runs)

    task = asyncio.create_task(
        _run_sweep_background(
            state, job, runtime, req.prompt_type, req.max_tokens, req.context_length,
            custom_prompt=req.custom_prompt,
        )
    )
    job.task = task

    return BenchmarkStartResponse(job_id=job.job_id, status=job.status)


@router.get("/sweep/{job_id}", response_model=SweepStatusResponse)
async def get_sweep_status(
    job_id: str,
    state: AppState = Depends(get_state),
) -> SweepStatusResponse:
    """Poll sweep job status, progress, and per-model results."""
    job = state.sweep_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Sweep job '{job_id}' not found")

    progress = SweepProgress(
        current_combo=job.current_combo,
        total_combos=job.total_combos,
        current_run_in_combo=job.current_run_in_combo,
        total_runs_per_combo=job.num_runs,
    )

    combo_results = None
    if job.combo_results:
        combo_results = []
        for cr in job.combo_results:
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
                for r in cr["results"]
            ] if cr.get("results") else None

            agg = None
            if cr.get("aggregate"):
                a = cr["aggregate"]
                agg = BenchmarkAggregate(
                    avg_tok_per_sec=round(a.get("avg_tok_per_sec", 0), 1),
                    min_tok_per_sec=round(a.get("min_tok_per_sec", 0), 1),
                    max_tok_per_sec=round(a.get("max_tok_per_sec", 0), 1),
                    avg_prefill_tok_per_sec=round(a.get("avg_prefill_tok_per_sec", 0), 1),
                    avg_ttft_ms=round(a.get("avg_ttft_ms", 0), 1),
                    avg_total_ms=round(a.get("avg_total_ms", 0), 1),
                    total_tokens_generated=a.get("total_tokens_generated", 0),
                    runs=a.get("runs", 0),
                )

            combo_results.append(SweepComboResult(
                model_id=cr["model_id"],
                runs=runs,
                aggregate=agg,
            ))

    return SweepStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=progress,
        combo_results=combo_results,
        error=job.error,
    )


@router.get("/{job_id}/report")
async def get_benchmark_report(
    job_id: str,
    state: AppState = Depends(get_state),
) -> HTMLResponse:
    """Generate and return a standalone HTML benchmark report."""
    job = state.benchmark_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status != "complete":
        raise HTTPException(status_code=400, detail="Benchmark not yet complete")

    runs_data = []
    if job.results:
        for r in job.results:
            runs_data.append({
                "run_number": r.run_number,
                "success": r.success,
                "tokens_per_second": round(r.tokens_per_second, 1),
                "prompt_tokens_per_second": round(r.prompt_tokens_per_second, 1),
                "time_to_first_token_ms": round(r.time_to_first_token_ms, 1),
                "total_time_ms": round(r.total_time_ms, 1),
                "generated_tokens": r.generated_tokens,
            })

    import datetime
    results_dict = {
        "runs": runs_data,
        "aggregate": job.aggregate or {},
    }
    metadata = {
        "title": f"Benchmark: {job.model_id} on {job.runtime_name}",
        "runtime": job.runtime_name,
        "model": job.model_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    html_content = generate_html_report(results_dict, metadata)
    return HTMLResponse(content=html_content)


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
            median_tok_per_sec=round(a.get("median_tok_per_sec", 0), 1),
            p95_tok_per_sec=round(a.get("p95_tok_per_sec", 0), 1),
            stddev_tok_per_sec=round(a.get("stddev_tok_per_sec", 0), 2),
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
    custom_prompt: str | None = None,
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
            custom_prompt,
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


async def _run_sweep_background(
    state: AppState,
    job: SweepJob,
    runtime: RuntimeInfo,
    prompt_type: str,
    max_tokens: int,
    context_length: int,
    custom_prompt: str | None = None,
) -> None:
    """Run sweep benchmark in a background thread with progress updates."""

    loop = asyncio.get_running_loop()

    def combo_callback(current: int, total: int) -> None:
        loop.call_soon_threadsafe(setattr, job, "current_combo", current)

    def run_callback(current: int, total: int) -> None:
        loop.call_soon_threadsafe(setattr, job, "current_run_in_combo", current)

    try:
        results = await asyncio.to_thread(
            run_benchmark_sweep,
            runtime,
            job.model_ids,
            prompt_type,
            job.num_runs,
            max_tokens,
            context_length,
            combo_callback,
            run_callback,
            custom_prompt,
        )
        job.combo_results = results
        job.status = "complete"
    except Exception as e:
        logger.error("Sweep job %s failed: %s", job.job_id, e)
        job.status = "error"
        job.error = "Sweep run failed — check server logs for details"
    finally:
        state.cleanup_old_jobs()


# ── Throughput Endpoints ────────────────────────────────────────────────────


@router.post("/throughput", response_model=BenchmarkStartResponse)
async def start_throughput(
    req: ThroughputStartRequest,
    state: AppState = Depends(get_state),
) -> BenchmarkStartResponse:
    """Start a concurrent throughput test as a background task."""
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception:
        logger.exception("Runtime detection failed for throughput")
        raise HTTPException(status_code=500, detail="Runtime detection failed")

    runtime = next((r for r in runtimes if r.name == req.runtime_name), None)
    if runtime is None:
        raise HTTPException(
            status_code=400,
            detail=f"Runtime '{req.runtime_name}' not found. Available: {[r.name for r in runtimes]}",
        )

    if req.model_id not in runtime.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_id}' not loaded in {req.runtime_name}.",
        )

    async with state._lock:
        job = state.create_throughput_job(req.runtime_name, req.model_id, req.concurrency, req.total_requests)

    task = asyncio.create_task(
        _run_throughput_background(state, job, runtime, req.prompt, req.max_tokens, req.concurrency, req.total_requests)
    )
    job.task = task

    return BenchmarkStartResponse(job_id=job.job_id, status=job.status)


@router.get("/throughput/{job_id}", response_model=ThroughputResponse)
async def get_throughput_status(
    job_id: str,
    state: AppState = Depends(get_state),
) -> ThroughputResponse:
    """Poll throughput test status and results."""
    job = state.throughput_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Throughput job '{job_id}' not found")

    per_request = None
    if job.result and job.result.get("per_request"):
        per_request = [
            ThroughputRequestResult(
                request_id=r.get("request_id", 0),
                success=r.get("success", False),
                tokens_generated=r.get("tokens_generated", 0),
                elapsed_ms=round(r.get("elapsed_ms", 0), 1),
                tokens_per_second=round(r.get("tokens_per_second", 0), 1),
                error=r.get("error"),
            )
            for r in job.result["per_request"]
        ]

    return ThroughputResponse(
        job_id=job.job_id,
        status=job.status,
        concurrency=job.result.get("concurrency", job.concurrency),
        total_requests=job.result.get("total_requests", job.total_requests),
        successful_requests=job.result.get("successful_requests", 0),
        failed_requests=job.result.get("failed_requests", 0),
        total_tokens=job.result.get("total_tokens", 0),
        elapsed_seconds=round(job.result.get("elapsed_seconds", 0), 2),
        throughput_tps=round(job.result.get("throughput_tps", 0), 1),
        avg_latency_ms=round(job.result.get("avg_latency_ms", 0), 1),
        min_latency_ms=round(job.result.get("min_latency_ms", 0), 1),
        max_latency_ms=round(job.result.get("max_latency_ms", 0), 1),
        error_rate=round(job.result.get("error_rate", 0), 3),
        per_request=per_request,
        error=job.error,
    )


async def _run_throughput_background(
    state: AppState,
    job: "ThroughputJob",
    runtime: RuntimeInfo,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    total_requests: int,
) -> None:
    """Run throughput test in background thread."""
    loop = asyncio.get_running_loop()

    def progress_callback(completed: int, total: int) -> None:
        loop.call_soon_threadsafe(setattr, job, "completed_requests", completed)

    try:
        result = await asyncio.to_thread(
            run_throughput_test,
            runtime.url,
            job.model_id,
            concurrency,
            total_requests,
            prompt,
            max_tokens,
            runtime.api_key,
            progress_callback,
        )
        # Convert dataclass to dict for storage
        from dataclasses import asdict
        job.result = asdict(result)
        job.status = "complete"
    except Exception as e:
        logger.error("Throughput job %s failed: %s", job.job_id, e)
        job.status = "error"
        job.error = "Throughput test failed — check server logs"
    finally:
        state.cleanup_old_jobs()


# ── Compare Endpoints ───────────────────────────────────────────────────────


@router.post("/compare", response_model=BenchmarkStartResponse)
async def start_compare(
    req: CompareStartRequest,
    state: AppState = Depends(get_state),
) -> BenchmarkStartResponse:
    """Compare benchmark results between two runtimes."""
    try:
        runtimes = await asyncio.to_thread(detect_all_runtimes)
    except Exception:
        logger.exception("Runtime detection failed for compare")
        raise HTTPException(status_code=500, detail="Runtime detection failed")

    runtime_a = next((r for r in runtimes if r.name == req.runtime_a), None)
    runtime_b = next((r for r in runtimes if r.name == req.runtime_b), None)

    if runtime_a is None or runtime_b is None:
        available = [r.name for r in runtimes]
        missing = []
        if runtime_a is None:
            missing.append(req.runtime_a)
        if runtime_b is None:
            missing.append(req.runtime_b)
        raise HTTPException(
            status_code=400,
            detail=f"Runtime(s) not found: {missing}. Available: {available}",
        )

    # Model must be loaded on both
    for rt in [runtime_a, runtime_b]:
        if req.model_id not in rt.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req.model_id}' not loaded in {rt.name}.",
            )

    async with state._lock:
        job = state.create_compare_job(req.runtime_a, req.runtime_b, req.model_id, req.num_runs)

    task = asyncio.create_task(
        _run_compare_background(
            state, job, runtime_a, runtime_b,
            req.prompt_type, req.max_tokens, req.context_length, req.custom_prompt,
        )
    )
    job.task = task

    return BenchmarkStartResponse(job_id=job.job_id, status=job.status)


@router.get("/compare/{job_id}", response_model=CompareResponse)
async def get_compare_status(
    job_id: str,
    state: AppState = Depends(get_state),
) -> CompareResponse:
    """Poll compare job status and results."""
    job = state.compare_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Compare job '{job_id}' not found")

    results = None
    if job.results:
        results = []
        for cr in job.results:
            agg = None
            if cr.get("aggregate"):
                a = cr["aggregate"]
                agg = BenchmarkAggregate(
                    avg_tok_per_sec=round(a.get("avg_tok_per_sec", 0), 1),
                    min_tok_per_sec=round(a.get("min_tok_per_sec", 0), 1),
                    max_tok_per_sec=round(a.get("max_tok_per_sec", 0), 1),
                    median_tok_per_sec=round(a.get("median_tok_per_sec", 0), 1),
                    p95_tok_per_sec=round(a.get("p95_tok_per_sec", 0), 1),
                    stddev_tok_per_sec=round(a.get("stddev_tok_per_sec", 0), 2),
                    avg_prefill_tok_per_sec=round(a.get("avg_prefill_tok_per_sec", 0), 1),
                    avg_ttft_ms=round(a.get("avg_ttft_ms", 0), 1),
                    avg_total_ms=round(a.get("avg_total_ms", 0), 1),
                    total_tokens_generated=a.get("total_tokens_generated", 0),
                    runs=a.get("runs", 0),
                )
            results.append(CompareResult(
                runtime_name=cr["runtime_name"],
                aggregate=agg,
            ))

    return CompareResponse(
        job_id=job.job_id,
        status=job.status,
        results=results,
        speedup_pct=round(job.speedup_pct, 1) if job.speedup_pct is not None else None,
        faster_runtime=job.faster_runtime,
        error=job.error,
    )


async def _run_compare_background(
    state: AppState,
    job: "CompareJob",
    runtime_a: RuntimeInfo,
    runtime_b: RuntimeInfo,
    prompt_type: str,
    max_tokens: int,
    context_length: int,
    custom_prompt: str | None = None,
) -> None:
    """Run benchmarks on two runtimes and compare results."""
    try:
        results_a = await asyncio.to_thread(
            run_benchmark_suite,
            runtime_a, job.model_id, prompt_type, job.num_runs,
            max_tokens, context_length, None, custom_prompt,
        )
        agg_a = aggregate_results(results_a)

        results_b = await asyncio.to_thread(
            run_benchmark_suite,
            runtime_b, job.model_id, prompt_type, job.num_runs,
            max_tokens, context_length, None, custom_prompt,
        )
        agg_b = aggregate_results(results_b)

        job.results = [
            {"runtime_name": runtime_a.name, "aggregate": agg_a},
            {"runtime_name": runtime_b.name, "aggregate": agg_b},
        ]

        # Calculate speedup
        speed_a = agg_a.get("avg_tok_per_sec", 0)
        speed_b = agg_b.get("avg_tok_per_sec", 0)
        if speed_a > 0 and speed_b > 0:
            if speed_a >= speed_b:
                job.faster_runtime = runtime_a.name
                job.speedup_pct = ((speed_a - speed_b) / speed_b) * 100
            else:
                job.faster_runtime = runtime_b.name
                job.speedup_pct = ((speed_b - speed_a) / speed_a) * 100

        job.status = "complete"
    except Exception as e:
        logger.error("Compare job %s failed: %s", job.job_id, e)
        job.status = "error"
        job.error = "Runtime comparison failed — check server logs"
    finally:
        state.cleanup_old_jobs()


