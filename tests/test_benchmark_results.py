"""Unit tests for loca_llama/benchmark_results.py."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

import pytest

from loca_llama.benchmark_results import (
    CATEGORY_ORDER,
    RESULTS_DIR,
    BenchmarkRecord,
    categorize_model,
    detect_hardware_string,
    load_results,
    save_result,
)


# ── categorize_model ───────────────────────────────────────────────────────────


class TestCategorizeModel:
    def test_should_return_cloud_api_for_openrouter_runtime(self):
        assert categorize_model("gpt-4o", "openrouter") == "Cloud API"

    def test_should_return_cloud_api_for_openai_runtime(self):
        assert categorize_model("gpt-4o", "openai") == "Cloud API"

    def test_should_return_cloud_api_for_anthropic_runtime(self):
        assert categorize_model("claude-3-sonnet", "anthropic") == "Cloud API"

    def test_should_return_cloud_api_for_litellm_runtime(self):
        assert categorize_model("gpt-4o", "litellm") == "Cloud API"

    def test_should_return_cloud_api_for_together_runtime(self):
        assert categorize_model("llama-3-70b", "together") == "Cloud API"

    def test_should_return_local_large_for_model_with_35b_params(self):
        assert categorize_model("Qwen3.5-35B", "omlx") == "Local (Large)"

    def test_should_return_local_large_for_model_with_exactly_30b_params(self):
        assert categorize_model("model-30b", "lm-studio") == "Local (Large)"

    def test_should_return_local_medium_for_model_with_14b_params(self):
        assert categorize_model("phi-4-14B", "lm-studio") == "Local (Medium)"

    def test_should_return_local_medium_for_model_with_13b_params(self):
        assert categorize_model("llama-13b", "omlx") == "Local (Medium)"

    def test_should_return_local_small_for_model_with_8b_params(self):
        assert categorize_model("Llama-8B", "omlx") == "Local (Small)"

    def test_should_return_local_small_for_model_with_7b_params(self):
        assert categorize_model("mistral-7b", "llama.cpp") == "Local (Small)"

    def test_should_return_local_small_for_model_with_1b_params(self):
        assert categorize_model("SmolLM-1.7B", "omlx") == "Local (Small)"

    def test_should_return_local_when_no_param_count_in_name(self):
        assert categorize_model("unknown", "omlx") == "Local"

    def test_should_return_local_when_no_param_count_and_local_runtime(self):
        assert categorize_model("my-custom-model", "llama.cpp") == "Local"

    def test_should_match_runtime_case_insensitively_for_cloud_detection(self):
        assert categorize_model("gpt-4o", "OpenRouter") == "Cloud API"

    def test_should_return_local_large_for_70b_model(self):
        assert categorize_model("Llama-3-70B", "lm-studio") == "Local (Large)"


# ── CATEGORY_ORDER ─────────────────────────────────────────────────────────────


class TestCategoryOrder:
    def test_should_have_exactly_five_entries(self):
        assert len(CATEGORY_ORDER) == 5

    def test_should_contain_cloud_api(self):
        assert "Cloud API" in CATEGORY_ORDER

    def test_should_contain_local_large(self):
        assert "Local (Large)" in CATEGORY_ORDER

    def test_should_contain_local_medium(self):
        assert "Local (Medium)" in CATEGORY_ORDER

    def test_should_contain_local_small(self):
        assert "Local (Small)" in CATEGORY_ORDER

    def test_should_contain_local(self):
        assert "Local" in CATEGORY_ORDER

    def test_should_list_cloud_api_before_local_categories(self):
        cloud_idx = CATEGORY_ORDER.index("Cloud API")
        local_idx = CATEGORY_ORDER.index("Local")
        assert cloud_idx < local_idx


# ── BenchmarkRecord ────────────────────────────────────────────────────────────


class TestBenchmarkRecord:
    def _make_record(self, **kwargs) -> BenchmarkRecord:
        defaults = dict(type="speed", model="test-model", runtime="lm-studio")
        defaults.update(kwargs)
        return BenchmarkRecord(**defaults)

    def test_should_instantiate_with_required_fields(self):
        r = self._make_record()
        assert r.type == "speed"
        assert r.model == "test-model"
        assert r.runtime == "lm-studio"

    def test_should_set_timestamp_automatically(self):
        before = time.time()
        r = self._make_record()
        after = time.time()
        assert before <= r.timestamp <= after

    def test_should_convert_to_dict_with_asdict(self):
        r = self._make_record()
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["type"] == "speed"
        assert d["model"] == "test-model"

    def test_should_round_trip_through_dict_creation(self):
        r = self._make_record(
            tokens_per_second=42.5,
            ttft_ms=120.0,
            hardware="M4 Pro 48GB",
        )
        d = asdict(r)
        r2 = BenchmarkRecord(**{
            k: v for k, v in d.items()
            if k in BenchmarkRecord.__dataclass_fields__
        })
        assert r2.model == r.model
        assert r2.tokens_per_second == r.tokens_per_second
        assert r2.hardware == r.hardware

    def test_should_generate_filename_with_type_and_model(self):
        r = self._make_record(model="my-llama-7b", type="quality")
        name = r.filename
        assert "quality" in name
        assert "my-llama-7b" in name
        assert name.endswith(".json")

    def test_should_sanitize_slashes_in_model_name_for_filename(self):
        r = self._make_record(model="org/model-name")
        assert "/" not in r.filename

    def test_should_sanitize_spaces_in_model_name_for_filename(self):
        r = self._make_record(model="My Model Name")
        assert " " not in r.filename

    def test_should_default_quality_scores_to_empty_dict(self):
        r = self._make_record()
        assert r.quality_scores == {}

    def test_should_default_monitor_stats_to_empty_dict(self):
        r = self._make_record()
        assert r.monitor_stats == {}

    def test_should_default_throughput_stats_to_empty_dict(self):
        r = self._make_record()
        assert r.throughput_stats == {}

    def test_should_store_quality_scores_dict(self):
        r = self._make_record(quality_scores={"pass_rate": 0.75})
        assert r.quality_scores["pass_rate"] == 0.75

    def test_should_have_all_required_dataclass_fields(self):
        field_names = {f.name for f in fields(BenchmarkRecord)}
        for name in ("type", "model", "runtime", "timestamp", "hardware",
                     "tokens_per_second", "ttft_ms", "quality_scores",
                     "monitor_stats", "throughput_stats", "cloud_provider",
                     "cloud_scores", "extra"):
            assert name in field_names


# ── save_result / load_results round-trip ─────────────────────────────────────


class TestSaveLoadRoundTrip:
    @pytest.fixture(autouse=True)
    def _patch_results_dir(self, tmp_path, monkeypatch):
        """Redirect RESULTS_DIR to tmp_path for every test in this class."""
        monkeypatch.setattr(
            "loca_llama.benchmark_results.RESULTS_DIR", tmp_path
        )
        self.results_dir = tmp_path

    def _make_record(self, **kwargs) -> BenchmarkRecord:
        defaults = dict(type="speed", model="round-trip-model", runtime="omlx")
        defaults.update(kwargs)
        return BenchmarkRecord(**defaults)

    def test_should_write_json_file_to_results_dir(self):
        r = self._make_record()
        path = save_result(r)
        assert path.exists()
        assert path.suffix == ".json"

    def test_should_save_valid_json(self):
        r = self._make_record()
        path = save_result(r)
        data = json.loads(path.read_text())
        assert data["model"] == "round-trip-model"

    def test_should_load_saved_record_back(self):
        r = self._make_record(tokens_per_second=55.0, hardware="M3 16GB")
        save_result(r)
        loaded = load_results()
        assert len(loaded) == 1
        assert loaded[0].model == "round-trip-model"
        assert loaded[0].tokens_per_second == 55.0
        assert loaded[0].hardware == "M3 16GB"

    def test_should_load_multiple_records_newest_first(self):
        for i in range(3):
            r = self._make_record(model=f"model-{i}")
            # Stagger timestamps by manipulating the record directly
            r.timestamp = time.time() + i
            save_result(r)

        loaded = load_results()
        assert len(loaded) == 3
        # Filenames are timestamp-prefixed; newest first
        assert loaded[0].model == "model-2"

    def test_should_filter_by_type(self):
        save_result(self._make_record(type="speed", model="speed-model"))
        save_result(self._make_record(type="quality", model="quality-model"))

        loaded = load_results(type_filter="speed")
        assert all(r.type == "speed" for r in loaded)
        assert len(loaded) == 1

    def test_should_filter_by_model_substring(self):
        save_result(self._make_record(model="llama-3-8b"))
        save_result(self._make_record(model="mistral-7b"))

        loaded = load_results(model_filter="llama")
        assert len(loaded) == 1
        assert "llama" in loaded[0].model.lower()

    def test_should_respect_limit_parameter(self):
        for i in range(5):
            save_result(self._make_record(model=f"m-{i}"))

        loaded = load_results(limit=3)
        assert len(loaded) == 3

    def test_should_return_empty_list_when_no_results_dir_exists(self, tmp_path, monkeypatch):
        empty = tmp_path / "nonexistent"
        monkeypatch.setattr("loca_llama.benchmark_results.RESULTS_DIR", empty)
        assert load_results() == []

    def test_should_skip_corrupt_json_files_and_continue(self):
        bad_file = self.results_dir / "20990101_000000_speed_bad.json"
        bad_file.write_text("{this is not valid json}")
        r = self._make_record(model="good-model")
        save_result(r)

        loaded = load_results()
        assert len(loaded) == 1
        assert loaded[0].model == "good-model"

    def test_should_roundtrip_quality_scores(self):
        r = self._make_record(
            type="quality",
            quality_scores={"pass_rate": 0.8, "tasks": [{"name": "fizzbuzz", "runnable": 1.0}]},
        )
        save_result(r)
        loaded = load_results()
        assert loaded[0].quality_scores["pass_rate"] == 0.8

    def test_should_roundtrip_extra_metadata(self):
        r = self._make_record(extra={"cost_cents": 3.5})
        save_result(r)
        loaded = load_results()
        assert loaded[0].extra["cost_cents"] == 3.5


# ── detect_hardware_string ─────────────────────────────────────────────────────


class TestDetectHardwareString:
    def test_should_return_non_empty_string(self):
        result = detect_hardware_string()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_should_return_string_when_hardware_detection_raises(self):
        with patch("loca_llama.hardware.detect_mac", side_effect=OSError("no hardware")):
            result = detect_hardware_string()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_should_fall_back_to_platform_machine_on_non_darwin(self):
        import platform as _platform
        with (
            patch("loca_llama.hardware.detect_mac", side_effect=ImportError),
            patch("loca_llama.benchmark_results.platform.system", return_value="Linux"),
        ):
            result = detect_hardware_string()
        assert result == _platform.machine()
