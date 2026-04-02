"""Tests for the unified mega-report generator."""

import pytest

from loca_llama.unified_report import (
    ModelScorecard,
    compute_overall_score,
    generate_unified_report,
    load_scorecards,
    _score_class,
    _build_chart_data,
)


# ── Score Class ─────────────────────────────────────────────────────────────


class TestScoreClass:
    """Tests for _score_class helper."""

    def test_high_score(self) -> None:
        assert _score_class(0.85) == "score-high"

    def test_mid_score(self) -> None:
        assert _score_class(0.6) == "score-mid"

    def test_low_score(self) -> None:
        assert _score_class(0.3) == "score-low"

    def test_boundary_high(self) -> None:
        assert _score_class(0.8) == "score-high"

    def test_boundary_mid(self) -> None:
        assert _score_class(0.5) == "score-mid"


# ── Overall Score Computation ───────────────────────────────────────────────


class TestComputeOverallScore:
    """Tests for compute_overall_score."""

    def _make_card(self, **kwargs: object) -> ModelScorecard:
        return ModelScorecard(model="test", **kwargs)

    def test_all_categories(self) -> None:
        cards = [self._make_card(
            speed_data={"tokens_per_second": 30},
            quality_data={"pass_rate": 0.8},
            eval_data={"gsm8k": {"score": 0.7}, "arc": {"score": 0.6}},
            sql_data={"pass_rate": 0.9},
            throughput_data={"throughput_tps": 100},
        )]
        score = compute_overall_score(cards[0], cards)
        assert 0 < score <= 100

    def test_missing_categories_redistribution(self) -> None:
        cards = [self._make_card(
            speed_data={"tokens_per_second": 30},
            quality_data={"pass_rate": 1.0},
        )]
        score = compute_overall_score(cards[0], cards)
        assert score > 0

    def test_empty_card(self) -> None:
        cards = [self._make_card()]
        score = compute_overall_score(cards[0], cards)
        assert score == 0.0

    def test_perfect_score(self) -> None:
        card = self._make_card(
            speed_data={"tokens_per_second": 100},
            quality_data={"pass_rate": 1.0},
            eval_data={"gsm8k": {"score": 1.0}},
            sql_data={"pass_rate": 1.0},
            throughput_data={"throughput_tps": 200},
        )
        cards = [card]
        score = compute_overall_score(card, cards)
        assert score == 100.0


# ── Report Generation ───────────────────────────────────────────────────────


class TestGenerateUnifiedReport:
    """Tests for generate_unified_report."""

    def _sample_cards(self) -> list[ModelScorecard]:
        card = ModelScorecard(
            model="test-model",
            runtime="lm-studio",
            hardware="M4 Pro",
            speed_data={"tokens_per_second": 25, "ttft_ms": 100, "percentiles": {}},
            quality_data={"pass_rate": 0.8, "tasks": [
                {"name": "fizzbuzz", "runnable": 1.0, "contains": 1.0},
            ]},
            eval_data={"gsm8k": {"score": 0.7, "correct": 14, "total": 20}},
            sql_data={"pass_rate": 0.85, "total_pass": 17, "total_questions": 20, "questions": [
                {"id": i, "difficulty": "easy", "status": "pass", "retries": 0}
                for i in range(1, 18)
            ] + [
                {"id": i, "difficulty": "hard", "status": "fail", "retries": 1}
                for i in range(18, 21)
            ]},
            overall_score=75.0,
        )
        return [card]

    def test_contains_all_sections(self) -> None:
        cards = self._sample_cards()
        html = generate_unified_report(cards)
        assert "Model Leaderboard" in html
        assert "Speed Dashboard" in html
        assert "Quality Benchmark" in html
        assert "Evaluation Benchmarks" in html
        assert "SQL Benchmark" in html
        assert "Model Details" in html

    def test_contains_model_name(self) -> None:
        cards = self._sample_cards()
        html = generate_unified_report(cards)
        assert "test-model" in html

    def test_self_contained_html(self) -> None:
        cards = self._sample_cards()
        html = generate_unified_report(cards)
        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "<script>" in html
        assert "</html>" in html

    def test_has_dark_light_theme(self) -> None:
        cards = self._sample_cards()
        html = generate_unified_report(cards)
        assert "prefers-color-scheme: light" in html

    def test_empty_scorecards(self) -> None:
        html = generate_unified_report([])
        assert "<!DOCTYPE html>" in html


# ── Chart Data ──────────────────────────────────────────────────────────────


class TestBuildChartData:
    """Tests for _build_chart_data."""

    def test_speed_data(self) -> None:
        cards = [ModelScorecard(
            model="test", speed_data={"tokens_per_second": 25, "ttft_ms": 100},
        )]
        data = _build_chart_data(cards)
        assert data["speed_tps"] == [25]
        assert data["speed_ttft"] == [100]

    def test_scatter_points(self) -> None:
        cards = [ModelScorecard(
            model="test", speed_data={"tokens_per_second": 25, "ttft_ms": 100},
        )]
        data = _build_chart_data(cards)
        assert len(data["scatter"]) == 1
        assert data["scatter"][0]["x"] == 25

    def test_radar_datasets(self) -> None:
        cards = [ModelScorecard(
            model="test",
            eval_data={"gsm8k": {"score": 0.8}, "arc": {"score": 0.7}},
        )]
        data = _build_chart_data(cards)
        assert "gsm8k" in data["radar_axes"]
        assert len(data["radar_datasets"]) == 1


# ── Load Scorecards (basic) ─────────────────────────────────────────────────


class TestLoadScorecards:
    """Tests for load_scorecards."""

    def test_empty_when_no_results(self, tmp_path: pytest.TempPathFactory) -> None:
        # load_scorecards reads from RESULTS_DIR which may not exist
        # Just verify it returns a list
        result = load_scorecards(models=["nonexistent_model_xyz"])
        assert isinstance(result, list)
