"""Unit tests for loca_llama.throughput module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from loca_llama.throughput import (
    RequestResult,
    ThroughputResult,
    run_throughput_ramp,
    run_throughput_test,
)


# ---------------------------------------------------------------------------
# RequestResult dataclass
# ---------------------------------------------------------------------------


class TestRequestResult:
    def test_should_create_with_required_fields(self):
        r = RequestResult(request_id=0, success=True)
        assert r.request_id == 0
        assert r.success is True

    def test_should_default_tokens_generated_to_zero(self):
        r = RequestResult(request_id=1, success=False)
        assert r.tokens_generated == 0

    def test_should_default_elapsed_ms_to_zero(self):
        r = RequestResult(request_id=1, success=True)
        assert r.elapsed_ms == 0.0

    def test_should_default_tokens_per_second_to_zero(self):
        r = RequestResult(request_id=1, success=True)
        assert r.tokens_per_second == 0.0

    def test_should_default_ttft_ms_to_zero(self):
        r = RequestResult(request_id=1, success=True)
        assert r.ttft_ms == 0.0

    def test_should_default_error_to_none(self):
        r = RequestResult(request_id=1, success=True)
        assert r.error is None

    def test_should_store_error_message_when_provided(self):
        r = RequestResult(request_id=2, success=False, error="connection refused")
        assert r.error == "connection refused"

    def test_should_store_all_fields(self):
        r = RequestResult(
            request_id=5,
            success=True,
            tokens_generated=200,
            elapsed_ms=1500.0,
            tokens_per_second=133.3,
            ttft_ms=42.0,
        )
        assert r.tokens_generated == 200
        assert r.elapsed_ms == 1500.0
        assert r.tokens_per_second == pytest.approx(133.3)
        assert r.ttft_ms == 42.0


# ---------------------------------------------------------------------------
# ThroughputResult dataclass
# ---------------------------------------------------------------------------


class TestThroughputResult:
    def _make(self, **kwargs) -> ThroughputResult:
        defaults = dict(
            concurrency=2,
            total_requests=4,
            successful_requests=4,
            failed_requests=0,
            total_tokens=400,
            elapsed_seconds=2.0,
            throughput_tps=200.0,
            avg_latency_ms=500.0,
            min_latency_ms=400.0,
            max_latency_ms=600.0,
            avg_ttft_ms=0.0,
            error_rate=0.0,
        )
        defaults.update(kwargs)
        return ThroughputResult(**defaults)

    def test_should_create_with_required_fields(self):
        result = self._make()
        assert result.concurrency == 2
        assert result.total_requests == 4

    def test_should_default_p50_to_zero(self):
        result = self._make()
        assert result.p50_latency_ms == 0.0

    def test_should_default_p90_to_zero(self):
        result = self._make()
        assert result.p90_latency_ms == 0.0

    def test_should_default_p99_to_zero(self):
        result = self._make()
        assert result.p99_latency_ms == 0.0

    def test_should_default_per_request_to_empty_list(self):
        result = self._make()
        assert result.per_request == []

    def test_should_store_per_request_results(self):
        req = RequestResult(request_id=0, success=True)
        result = self._make(per_request=[req])
        assert len(result.per_request) == 1
        assert result.per_request[0] is req

    def test_should_store_percentile_fields(self):
        result = self._make(p50_latency_ms=500.0, p90_latency_ms=550.0, p99_latency_ms=590.0)
        assert result.p50_latency_ms == 500.0
        assert result.p90_latency_ms == 550.0
        assert result.p99_latency_ms == 590.0

    def test_should_compute_error_rate_as_fraction(self):
        result = self._make(failed_requests=1, total_requests=4, error_rate=0.25)
        assert result.error_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _pct via run_throughput_test — percentile correctness
# ---------------------------------------------------------------------------


def _make_request_result(request_id: int, tokens: int, elapsed_ms: float) -> RequestResult:
    """Build a successful RequestResult with known latency."""
    tps = tokens / elapsed_ms * 1000 if elapsed_ms > 0 else 0.0
    return RequestResult(
        request_id=request_id,
        success=True,
        tokens_generated=tokens,
        elapsed_ms=elapsed_ms,
        tokens_per_second=tps,
    )


def _mock_single_request_side_effect(results_by_id: dict[int, RequestResult]):
    """Return a side_effect function that returns pre-built RequestResults by id."""
    def _inner(request_id, *args, **kwargs):
        return results_by_id[request_id]
    return _inner


class TestRunThroughputTestPercentiles:
    """Verify the _pct helper via ThroughputResult percentile fields."""

    def _run_with_latencies(self, latencies_ms: list[float]) -> ThroughputResult:
        """Patch _single_request to return known latencies, then run test."""
        n = len(latencies_ms)
        results_by_id = {
            i: _make_request_result(i, tokens=100, elapsed_ms=ms)
            for i, ms in enumerate(latencies_ms)
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            return run_throughput_test(
                base_url="http://localhost:1234",
                model_id="test-model",
                concurrency=1,
                total_requests=n,
            )

    def test_should_return_p50_equal_median_for_sorted_values(self):
        # Five equal latencies: p50 should be the median (index 2 = 300ms)
        result = self._run_with_latencies([100.0, 200.0, 300.0, 400.0, 500.0])
        assert result.p50_latency_ms == pytest.approx(300.0)

    def test_should_return_p90_near_high_end(self):
        # Ten values 100..1000 ms — p90 should be >= p50
        result = self._run_with_latencies([float(i * 100) for i in range(1, 11)])
        assert result.p90_latency_ms >= result.p50_latency_ms

    def test_should_return_p99_greater_than_or_equal_to_p90(self):
        result = self._run_with_latencies([float(i * 50) for i in range(1, 11)])
        assert result.p99_latency_ms >= result.p90_latency_ms

    def test_should_return_p50_equal_single_value_for_one_request(self):
        result = self._run_with_latencies([250.0])
        assert result.p50_latency_ms == pytest.approx(250.0)

    def test_should_return_p99_equal_max_for_single_request(self):
        result = self._run_with_latencies([250.0])
        assert result.p99_latency_ms == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# run_throughput_test — aggregate metrics
# ---------------------------------------------------------------------------


class TestRunThroughputTest:
    def test_should_count_successful_and_failed_requests(self):
        results_by_id = {
            0: RequestResult(request_id=0, success=True, tokens_generated=100, elapsed_ms=500.0),
            1: RequestResult(request_id=1, success=False, elapsed_ms=10.0, error="timeout"),
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=1,
                total_requests=2,
            )
        assert result.successful_requests == 1
        assert result.failed_requests == 1

    def test_should_sum_tokens_from_successful_requests_only(self):
        results_by_id = {
            0: RequestResult(request_id=0, success=True, tokens_generated=80, elapsed_ms=400.0),
            1: RequestResult(request_id=1, success=True, tokens_generated=120, elapsed_ms=600.0),
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=1,
                total_requests=2,
            )
        assert result.total_tokens == 200

    def test_should_compute_error_rate_as_fraction_of_total(self):
        results_by_id = {
            0: RequestResult(request_id=0, success=False, error="err", elapsed_ms=10.0),
            1: RequestResult(request_id=1, success=False, error="err", elapsed_ms=10.0),
            2: RequestResult(request_id=2, success=True, tokens_generated=50, elapsed_ms=500.0),
            3: RequestResult(request_id=3, success=True, tokens_generated=50, elapsed_ms=500.0),
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=2,
                total_requests=4,
            )
        assert result.error_rate == pytest.approx(0.5)

    def test_should_sort_per_request_results_by_request_id(self):
        results_by_id = {
            0: RequestResult(request_id=0, success=True, tokens_generated=50, elapsed_ms=200.0),
            1: RequestResult(request_id=1, success=True, tokens_generated=50, elapsed_ms=300.0),
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=1,
                total_requests=2,
            )
        ids = [r.request_id for r in result.per_request]
        assert ids == sorted(ids)

    def test_should_report_zero_error_rate_when_all_succeed(self):
        results_by_id = {
            0: RequestResult(request_id=0, success=True, tokens_generated=100, elapsed_ms=500.0),
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=1,
                total_requests=1,
            )
        assert result.error_rate == 0.0

    def test_should_invoke_progress_callback_once_per_request(self):
        results_by_id = {
            i: RequestResult(request_id=i, success=True, tokens_generated=50, elapsed_ms=200.0)
            for i in range(3)
        }
        progress_calls = []
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=1,
                total_requests=3,
                progress_callback=lambda done, total: progress_calls.append((done, total)),
            )
        assert len(progress_calls) == 3

    def test_should_use_provided_concurrency_value(self):
        results_by_id = {
            i: RequestResult(request_id=i, success=True, tokens_generated=50, elapsed_ms=200.0)
            for i in range(2)
        }
        with patch(
            "loca_llama.throughput._single_request",
            side_effect=_mock_single_request_side_effect(results_by_id),
        ):
            result = run_throughput_test(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency=2,
                total_requests=2,
            )
        assert result.concurrency == 2


# ---------------------------------------------------------------------------
# run_throughput_ramp
# ---------------------------------------------------------------------------


def _make_throughput_result(concurrency: int, tps: float) -> ThroughputResult:
    """Build a minimal ThroughputResult for use in ramp tests."""
    return ThroughputResult(
        concurrency=concurrency,
        total_requests=concurrency,
        successful_requests=concurrency,
        failed_requests=0,
        total_tokens=concurrency * 100,
        elapsed_seconds=1.0,
        throughput_tps=tps,
        avg_latency_ms=200.0,
        min_latency_ms=180.0,
        max_latency_ms=220.0,
        avg_ttft_ms=0.0,
        error_rate=0.0,
        p50_latency_ms=200.0,
        p90_latency_ms=210.0,
        p99_latency_ms=219.0,
    )


class TestRunThroughputRamp:
    def test_should_return_one_result_per_concurrency_level(self):
        fake_results = [_make_throughput_result(c, float(c * 50)) for c in [1, 2, 4]]
        with patch(
            "loca_llama.throughput.run_throughput_test",
            side_effect=fake_results,
        ):
            results = run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="test-model",
                concurrency_levels=[1, 2, 4],
            )
        assert len(results) == 3

    def test_should_use_default_concurrency_levels_when_none_provided(self):
        # Default levels are [1, 2, 4, 8] — 4 calls
        fake_results = [_make_throughput_result(c, float(c * 50)) for c in [1, 2, 4, 8]]
        with patch(
            "loca_llama.throughput.run_throughput_test",
            side_effect=fake_results,
        ):
            results = run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="test-model",
            )
        assert len(results) == 4

    def test_should_pass_requests_per_level_as_total_requests(self):
        captured_calls: list[dict] = []

        def _fake_run(*args, **kwargs) -> ThroughputResult:
            captured_calls.append(kwargs)
            return _make_throughput_result(kwargs.get("concurrency", 1), 100.0)

        with patch("loca_llama.throughput.run_throughput_test", side_effect=_fake_run):
            run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="test-model",
                concurrency_levels=[1],
                requests_per_level=6,
            )
        # total_requests must be >= concurrency level and >= requests_per_level
        assert captured_calls[0]["total_requests"] >= 6

    def test_should_ensure_total_requests_at_least_concurrency_level(self):
        """When concurrency > requests_per_level, total_requests is raised to concurrency."""
        captured_calls: list[dict] = []

        def _fake_run(*args, **kwargs) -> ThroughputResult:
            captured_calls.append(kwargs)
            return _make_throughput_result(kwargs.get("concurrency", 1), 50.0)

        with patch("loca_llama.throughput.run_throughput_test", side_effect=_fake_run):
            run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="test-model",
                concurrency_levels=[8],
                requests_per_level=2,
            )
        assert captured_calls[0]["total_requests"] == 8

    def test_should_pass_base_url_and_model_id_to_each_call(self):
        captured_calls: list[dict] = []

        def _fake_run(*args, **kwargs) -> ThroughputResult:
            captured_calls.append(kwargs)
            return _make_throughput_result(kwargs.get("concurrency", 1), 100.0)

        with patch("loca_llama.throughput.run_throughput_test", side_effect=_fake_run):
            run_throughput_ramp(
                base_url="http://localhost:9999",
                model_id="my-model",
                concurrency_levels=[1, 2],
            )
        for call_kwargs in captured_calls:
            assert call_kwargs["base_url"] == "http://localhost:9999"
            assert call_kwargs["model_id"] == "my-model"

    def test_should_pass_api_key_to_each_call(self):
        captured_calls: list[dict] = []

        def _fake_run(*args, **kwargs) -> ThroughputResult:
            captured_calls.append(kwargs)
            return _make_throughput_result(kwargs.get("concurrency", 1), 100.0)

        with patch("loca_llama.throughput.run_throughput_test", side_effect=_fake_run):
            run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency_levels=[1],
                api_key="secret-key",
            )
        assert captured_calls[0]["api_key"] == "secret-key"

    def test_should_fall_back_to_default_levels_when_none_given(self):
        """None concurrency_levels triggers the default [1, 2, 4, 8] fallback."""
        fake_results = [_make_throughput_result(c, float(c * 50)) for c in [1, 2, 4, 8]]
        with patch(
            "loca_llama.throughput.run_throughput_test",
            side_effect=fake_results,
        ) as mock_run:
            results = run_throughput_ramp(
                base_url="http://localhost:1234",
                model_id="m",
                concurrency_levels=None,
            )
        assert mock_run.call_count == 4
        assert len(results) == 4
