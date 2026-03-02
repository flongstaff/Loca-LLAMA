"""In-memory application state for the Loca-LLAMA web API."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from loca_llama.benchmark import BenchmarkResult
from loca_llama.memory_monitor import MemoryMonitor


@dataclass
class BenchmarkJob:
    """Tracks a running or completed benchmark."""

    job_id: str
    status: str  # "running", "complete", "error"
    runtime_name: str
    model_id: str
    num_runs: int
    current_run: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    error: str | None = None


class AppState:
    """Shared application state, created once at startup."""

    def __init__(self) -> None:
        self.memory_monitor = MemoryMonitor(interval=1.0)
        self.benchmark_jobs: dict[str, BenchmarkJob] = {}
        self._lock = asyncio.Lock()

    def create_benchmark_job(
        self, runtime_name: str, model_id: str, num_runs: int
    ) -> BenchmarkJob:
        job_id = str(uuid.uuid4())
        job = BenchmarkJob(
            job_id=job_id,
            status="running",
            runtime_name=runtime_name,
            model_id=model_id,
            num_runs=num_runs,
        )
        self.benchmark_jobs[job_id] = job
        return job

    def cleanup_old_jobs(self, max_jobs: int = 50) -> None:
        """Remove oldest completed jobs if over limit."""
        completed = [
            (k, v)
            for k, v in self.benchmark_jobs.items()
            if v.status in ("complete", "error")
        ]
        if len(completed) > max_jobs:
            for k, _ in completed[: len(completed) - max_jobs]:
                del self.benchmark_jobs[k]
