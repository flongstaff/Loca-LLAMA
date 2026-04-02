"""Tests for benchmark quality improvements."""

import pytest

from loca_llama.eval_benchmarks import (
    _gsm8k_extract_answer,
    _extract_mc_answer,
    _estimate_confidence,
)
from loca_llama.benchmark import _percentile, aggregate_results, BenchmarkResult


# ── GSM8K Answer Extraction ─────────────────────────────────────────────────


class TestGsm8kExtract:
    """Tests for improved _gsm8k_extract_answer."""

    def test_standard_format(self) -> None:
        assert _gsm8k_extract_answer("The answer is #### 42") == "42"

    def test_boxed_format(self) -> None:
        assert _gsm8k_extract_answer("\\boxed{42}") == "42"

    def test_approximately_format(self) -> None:
        result = _gsm8k_extract_answer("The answer is approximately 42")
        assert result == "42"

    def test_dollar_sign(self) -> None:
        result = _gsm8k_extract_answer("The total is $1234")
        assert result == "1234"

    def test_dollar_with_commas(self) -> None:
        result = _gsm8k_extract_answer("The answer is $1,234")
        assert result == "1234"

    def test_therefore_pattern(self) -> None:
        result = _gsm8k_extract_answer("therefore 42 is the final answer.")
        assert result is not None
        # The "therefore" pattern should match via the last-number fallback or the "therefore" regex

    def test_equals_pattern(self) -> None:
        result = _gsm8k_extract_answer("The result = 42")
        assert result == "42"

    def test_amounts_to_pattern(self) -> None:
        result = _gsm8k_extract_answer("This amounts to 42 units")
        assert result == "42"

    def test_strips_thinking_tags(self) -> None:
        text = "<think>Let me calculate... 99 wrong</think>\n#### 42"
        assert _gsm8k_extract_answer(text) == "42"

    def test_last_number_fallback(self) -> None:
        result = _gsm8k_extract_answer("There are 10 apples and 5 oranges, total 15")
        assert result == "15"

    def test_returns_none_for_empty(self) -> None:
        assert _gsm8k_extract_answer("No numbers here") is None


# ── MC Answer Extraction ────────────────────────────────────────────────────


class TestMcExtract:
    """Tests for improved _extract_mc_answer."""

    def test_single_letter(self) -> None:
        assert _extract_mc_answer("A", ["A", "B", "C"]) == "A"

    def test_letter_with_paren(self) -> None:
        assert _extract_mc_answer("B) is correct", ["A", "B", "C"]) == "B"

    def test_json_format(self) -> None:
        assert _extract_mc_answer('{"answer": "C"}', ["A", "B", "C"]) == "C"

    def test_parenthesized(self) -> None:
        assert _extract_mc_answer("The correct answer is (B)", ["A", "B", "C"]) == "B"

    def test_x_is_correct(self) -> None:
        assert _extract_mc_answer("A is correct because...", ["A", "B", "C"]) == "A"

    def test_the_answer_is(self) -> None:
        assert _extract_mc_answer("The answer is C", ["A", "B", "C", "D"]) == "C"

    def test_strips_thinking(self) -> None:
        text = "<think>Hmm, B seems right</think>The answer is A"
        assert _extract_mc_answer(text, ["A", "B", "C"]) == "A"

    def test_first_valid_letter(self) -> None:
        assert _extract_mc_answer("I believe B is the best option", ["A", "B", "C"]) == "B"

    def test_returns_none_for_invalid(self) -> None:
        assert _extract_mc_answer("No valid answer here", ["A", "B"]) is None


# ── Confidence Estimation ───────────────────────────────────────────────────


class TestEstimateConfidence:
    """Tests for _estimate_confidence."""

    def test_high_confidence(self) -> None:
        assert _estimate_confidence("The answer is definitely A.") == 1.0

    def test_low_confidence(self) -> None:
        score = _estimate_confidence("I think it might be A, probably B, maybe C")
        assert score < 0.5

    def test_strips_thinking_tags(self) -> None:
        text = "<think>I think probably maybe</think>The answer is A"
        score = _estimate_confidence(text)
        assert score == 1.0  # Hedging was inside thinking tags


# ── Percentile Computation ──────────────────────────────────────────────────


class TestPercentile:
    """Tests for _percentile helper."""

    def test_p50_even(self) -> None:
        result = _percentile([1, 2, 3, 4], 50)
        assert result == 2.5

    def test_p50_odd(self) -> None:
        result = _percentile([1, 2, 3], 50)
        assert result == 2.0

    def test_p0(self) -> None:
        assert _percentile([1, 2, 3], 0) == 1.0

    def test_p100(self) -> None:
        assert _percentile([1, 2, 3], 100) == 3.0

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([42], 95) == 42.0


# ── Aggregate Results ───────────────────────────────────────────────────────


class TestAggregateResults:
    """Tests for enhanced aggregate_results."""

    def _make_result(self, tps: float, ttft: float, run: int = 1) -> BenchmarkResult:
        return BenchmarkResult(
            model_name="test", runtime="test",
            prompt_tokens=10, generated_tokens=50,
            prompt_eval_time_ms=ttft, eval_time_ms=1000,
            total_time_ms=ttft + 1000,
            tokens_per_second=tps, prompt_tokens_per_second=100,
            context_length=4096, success=True, run_number=run,
        )

    def test_has_percentile_fields(self) -> None:
        results = [self._make_result(25 + i, 100 + i * 10, i + 1) for i in range(5)]
        agg = aggregate_results(results, skip_first=False)
        assert "p50_tok_per_sec" in agg
        assert "p90_tok_per_sec" in agg
        assert "p99_tok_per_sec" in agg
        assert "p50_ttft_ms" in agg
        assert "p90_ttft_ms" in agg

    def test_warmup_detection(self) -> None:
        # First run much slower than rest
        results = [
            self._make_result(10, 200, 1),  # slow warmup
            self._make_result(25, 100, 2),
            self._make_result(26, 95, 3),
            self._make_result(24, 105, 4),
        ]
        agg = aggregate_results(results, skip_first=True)
        assert agg["warmup_excluded"] is True

    def test_no_warmup_when_similar(self) -> None:
        results = [
            self._make_result(25, 100, 1),
            self._make_result(26, 95, 2),
            self._make_result(24, 105, 3),
        ]
        agg = aggregate_results(results, skip_first=True)
        assert agg["warmup_excluded"] is False

    def test_per_token_latency_percentiles(self) -> None:
        results = [self._make_result(25, 100, 1)]
        results[0].extra = {"per_token_latencies": [10.0, 12.0, 11.0, 13.0, 10.5]}
        agg = aggregate_results(results, skip_first=False)
        assert agg["per_token_latency_p50_ms"] > 0
