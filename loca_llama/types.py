"""Shared type aliases and literals for the loca-llama project."""

from __future__ import annotations

from typing import Literal

JobStatus = Literal["running", "complete", "error"]
BenchmarkType = Literal["speed", "quality", "monitor", "eval", "sql", "throughput"]
PromptType = Literal["default", "coding", "reasoning", "creative", "json"]
Difficulty = Literal["trivial", "easy", "medium", "hard"]
SQLResultStatus = Literal["pass", "fail", "error"]
