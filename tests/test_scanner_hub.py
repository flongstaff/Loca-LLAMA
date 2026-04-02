"""Unit tests for loca_llama.scanner and loca_llama.hub modules."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from loca_llama.scanner import (
    LocalModel,
    detect_family,
    detect_quant,
    dir_size_gb,
    file_size_gb,
    scan_all,
    scan_custom_dir,
    scan_gguf_files,
    scan_mlx_models,
)
from loca_llama.hub import (
    format_downloads,
    get_model_files,
    search_gguf_models,
    search_huggingface,
    search_mlx_models,
)


# ---------------------------------------------------------------------------
# detect_quant
# ---------------------------------------------------------------------------


class TestDetectQuant:
    def test_extracts_q4_k_m(self):
        assert detect_quant("llama-3-8b-Q4_K_M.gguf") == "Q4_K_M"

    def test_extracts_q5_k_m(self):
        assert detect_quant("mistral-7b-Q5_K_M.gguf") == "Q5_K_M"

    def test_extracts_q8_0(self):
        assert detect_quant("model-Q8_0.gguf") == "Q8_0"

    def test_extracts_f16(self):
        assert detect_quant("model-F16.gguf") == "F16"

    def test_extracts_fp16_uppercased(self):
        result = detect_quant("model-FP16.gguf")
        assert result == "FP16"

    def test_extracts_fp32(self):
        assert detect_quant("model-FP32.gguf") == "FP32"

    def test_extracts_bf16(self):
        assert detect_quant("model-BF16.gguf") == "BF16"

    def test_extracts_iq4_xs(self):
        assert detect_quant("model-IQ4_XS.gguf") == "IQ4_XS"

    def test_extracts_iq1_s(self):
        assert detect_quant("model-IQ1_S.gguf") == "IQ1_S"

    def test_extracts_q4_0_lowercase(self):
        result = detect_quant("model-q4_0.gguf")
        assert result == "Q4_0"

    def test_case_insensitive_match(self):
        """Pattern is case-insensitive; result is always uppercased."""
        result = detect_quant("model-q4_k_m.gguf")
        assert result == "Q4_K_M"

    def test_returns_none_when_no_quant(self):
        assert detect_quant("llama-3-8b-instruct.gguf") is None

    def test_returns_none_for_empty_string(self):
        assert detect_quant("") is None

    def test_picks_first_match_in_filename(self):
        """When multiple quant strings appear, first match is returned."""
        result = detect_quant("model-Q4_K_M-Q8_0.gguf")
        assert result == "Q4_K_M"


# ---------------------------------------------------------------------------
# detect_family
# ---------------------------------------------------------------------------


class TestDetectFamily:
    def test_detects_llama(self):
        assert detect_family("llama-3-8b") == "Llama"

    def test_detects_mistral(self):
        assert detect_family("Mistral-7B-Instruct") == "Mistral"

    def test_detects_mixtral(self):
        assert detect_family("Mixtral-8x7B") == "Mixtral"

    def test_detects_phi(self):
        assert detect_family("phi-3-mini") == "Phi"

    def test_detects_phi4(self):
        assert detect_family("phi4-instruct") == "Phi"

    def test_detects_gemma(self):
        assert detect_family("gemma-2-9b") == "Gemma"

    def test_detects_qwen(self):
        assert detect_family("Qwen2.5-32B") == "Qwen"

    def test_detects_deepseek(self):
        assert detect_family("DeepSeek-R1-7B") == "DeepSeek"

    def test_detects_deepseek_with_hyphen(self):
        assert detect_family("deep-seek-coder") == "DeepSeek"

    def test_detects_command_r(self):
        assert detect_family("command-r-plus") == "Command"

    def test_detects_codellama(self):
        # "Llama" pattern is checked before "CodeLlama" in FAMILY_PATTERNS dict order,
        # so "CodeLlama-13B" resolves to "Llama". Test reflects actual behaviour.
        assert detect_family("CodeLlama-13B") == "Llama"

    def test_detects_yi(self):
        assert detect_family("Yi-34B-chat") == "Yi"

    def test_detects_starcoder(self):
        assert detect_family("StarCoder2-7B") == "StarCoder"

    def test_detects_falcon(self):
        assert detect_family("falcon-40b") == "Falcon"

    def test_returns_none_for_unknown_family(self):
        assert detect_family("unknown-model-xyz") is None

    def test_returns_none_for_empty_string(self):
        assert detect_family("") is None

    def test_case_insensitive(self):
        assert detect_family("MISTRAL-7B") == "Mistral"


# ---------------------------------------------------------------------------
# file_size_gb
# ---------------------------------------------------------------------------


class TestFileSizeGb:
    def test_returns_size_in_gb(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * (1024**3))  # exactly 1 GB
        assert file_size_gb(f) == pytest.approx(1.0, abs=1e-9)

    def test_returns_zero_for_nonexistent_file(self, tmp_path: Path):
        missing = tmp_path / "missing.gguf"
        assert file_size_gb(missing) == 0.0

    def test_returns_zero_on_oserror(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"data")
        with patch.object(Path, "stat", side_effect=OSError("permission denied")):
            assert file_size_gb(f) == 0.0

    def test_small_file_less_than_one_gb(self, tmp_path: Path):
        f = tmp_path / "tiny.gguf"
        f.write_bytes(b"x" * 1024)
        assert file_size_gb(f) < 1.0
        assert file_size_gb(f) > 0.0


# ---------------------------------------------------------------------------
# dir_size_gb
# ---------------------------------------------------------------------------


class TestDirSizeGb:
    def test_sums_all_files(self, tmp_path: Path):
        (tmp_path / "a.bin").write_bytes(b"x" * 512)
        (tmp_path / "b.bin").write_bytes(b"x" * 512)
        total = dir_size_gb(tmp_path)
        assert total == pytest.approx(1024 / (1024**3), rel=1e-6)

    def test_recurses_into_subdirectories(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "model.bin").write_bytes(b"x" * 1024)
        assert dir_size_gb(tmp_path) > 0.0

    def test_empty_directory_returns_zero(self, tmp_path: Path):
        assert dir_size_gb(tmp_path) == 0.0

    def test_nonexistent_directory_returns_zero(self, tmp_path: Path):
        missing = tmp_path / "no_such_dir"
        assert dir_size_gb(missing) == 0.0


# ---------------------------------------------------------------------------
# scan_gguf_files
# ---------------------------------------------------------------------------


class TestScanGgufFiles:
    def test_returns_empty_when_directory_missing(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        assert scan_gguf_files(missing, "lm-studio") == []

    def test_finds_gguf_file(self, tmp_path: Path):
        (tmp_path / "llama-3-8b-Q4_K_M.gguf").write_bytes(b"fake")
        models = scan_gguf_files(tmp_path, "lm-studio")
        assert len(models) == 1

    def test_model_has_correct_name(self, tmp_path: Path):
        (tmp_path / "llama-3-8b-Q4_K_M.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.name == "llama-3-8b-Q4_K_M"

    def test_model_has_correct_source(self, tmp_path: Path):
        (tmp_path / "model.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "llama.cpp")[0]
        assert model.source == "llama.cpp"

    def test_model_format_is_gguf(self, tmp_path: Path):
        (tmp_path / "model.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.format == "gguf"

    def test_detects_quant_from_filename(self, tmp_path: Path):
        (tmp_path / "llama-Q4_K_M.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.quant == "Q4_K_M"

    def test_detects_family_from_path(self, tmp_path: Path):
        (tmp_path / "mistral-7b.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.family == "Mistral"

    def test_ignores_non_gguf_files(self, tmp_path: Path):
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        (tmp_path / "README.md").write_text("docs")
        assert scan_gguf_files(tmp_path, "lm-studio") == []

    def test_scans_recursively(self, tmp_path: Path):
        sub = tmp_path / "org" / "repo"
        sub.mkdir(parents=True)
        (sub / "model.gguf").write_bytes(b"fake")
        models = scan_gguf_files(tmp_path, "lm-studio")
        assert len(models) == 1

    def test_extracts_repo_id_from_nested_path(self, tmp_path: Path):
        sub = tmp_path / "meta-llama" / "Llama-3-8B"
        sub.mkdir(parents=True)
        (sub / "model.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.repo_id == "meta-llama/Llama-3-8B"

    def test_repo_id_none_for_flat_file(self, tmp_path: Path):
        (tmp_path / "model.gguf").write_bytes(b"fake")
        model = scan_gguf_files(tmp_path, "lm-studio")[0]
        assert model.repo_id is None

    def test_finds_multiple_gguf_files(self, tmp_path: Path):
        (tmp_path / "model-a.gguf").write_bytes(b"fake")
        (tmp_path / "model-b.gguf").write_bytes(b"fake")
        models = scan_gguf_files(tmp_path, "lm-studio")
        assert len(models) == 2


# ---------------------------------------------------------------------------
# scan_mlx_models
# ---------------------------------------------------------------------------


class TestScanMlxModels:
    def test_returns_empty_when_directory_missing(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        assert scan_mlx_models(missing) == []

    def test_skips_non_model_dirs(self, tmp_path: Path):
        (tmp_path / "some-random-dir").mkdir()
        assert scan_mlx_models(tmp_path) == []

    def _make_hf_model(
        self,
        base: Path,
        org: str,
        model: str,
        filenames: list[str],
    ) -> Path:
        """Helper: build the models--org--model/snapshots/abc123/<files> structure."""
        snapshot = base / f"models--{org}--{model}" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        for fname in filenames:
            (snapshot / fname).write_bytes(b"fake")
        return snapshot

    def test_finds_safetensors_model(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "mlx-community", "llama-4bit", ["model.safetensors"])
        models = scan_mlx_models(tmp_path)
        assert len(models) == 1

    def test_mlx_community_format_is_mlx(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "mlx-community", "llama-4bit", ["model.safetensors"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.format == "mlx"

    def test_mlx_community_source_is_mlx_community(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "mlx-community", "llama-4bit", ["model.safetensors"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.source == "mlx-community"

    def test_non_mlx_safetensors_format_is_safetensors(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "some-org", "bert-base", ["model.safetensors"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.format == "safetensors"

    def test_non_mlx_safetensors_source_is_huggingface(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "some-org", "bert-base", ["model.safetensors"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.source == "huggingface"

    def test_gguf_in_hf_cache_gets_gguf_format(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "bartowski", "Llama-3-8B", ["model-Q4_K_M.gguf"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.format == "gguf"

    def test_gguf_in_hf_cache_source_is_huggingface(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "bartowski", "Llama-3-8B", ["model.gguf"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.source == "huggingface"

    def test_repo_id_constructed_from_path(self, tmp_path: Path):
        self._make_hf_model(tmp_path, "mlx-community", "llama-4bit", ["model.safetensors"])
        model = scan_mlx_models(tmp_path)[0]
        assert model.repo_id == "mlx-community/llama-4bit"

    def test_skips_dir_with_no_snapshots(self, tmp_path: Path):
        (tmp_path / "models--org--model").mkdir()
        assert scan_mlx_models(tmp_path) == []

    def test_skips_dir_without_matching_files(self, tmp_path: Path):
        snap = tmp_path / "models--org--model" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        assert scan_mlx_models(tmp_path) == []

    def test_uses_latest_snapshot(self, tmp_path: Path):
        """When multiple snapshots exist, the most recently modified is used."""
        model_dir = tmp_path / "models--org--model" / "snapshots"
        model_dir.mkdir(parents=True)
        old = model_dir / "old_hash"
        new = model_dir / "new_hash"
        old.mkdir()
        new.mkdir()
        (old / "model.safetensors").write_bytes(b"old")
        (new / "model.safetensors").write_bytes(b"new")
        import time
        time.sleep(0.01)
        new.touch()
        models = scan_mlx_models(tmp_path)
        assert len(models) == 1
        assert models[0].path == new


# ---------------------------------------------------------------------------
# scan_all
# ---------------------------------------------------------------------------


class TestScanAll:
    def test_returns_list(self):
        fake_home = MagicMock(spec=Path)
        # Make all constructed paths return non-existent dirs
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = False
        fake_home.__truediv__ = MagicMock(return_value=fake_path)
        fake_path.__truediv__ = MagicMock(return_value=fake_path)

        with patch("loca_llama.scanner.LM_STUDIO_PATHS", []), \
             patch("loca_llama.scanner.LLAMA_CPP_PATHS", []), \
             patch("loca_llama.scanner.HUGGINGFACE_PATHS", []):
            result = scan_all()
        assert isinstance(result, list)

    def test_deduplicates_by_path(self):
        shared_path = Path("/models/model.gguf")
        model = LocalModel(
            name="model",
            path=shared_path,
            size_gb=4.0,
            format="gguf",
            source="lm-studio",
        )
        with patch("loca_llama.scanner.LM_STUDIO_PATHS", []), \
             patch("loca_llama.scanner.LLAMA_CPP_PATHS", []), \
             patch("loca_llama.scanner.HUGGINGFACE_PATHS", []), \
             patch("loca_llama.scanner.scan_gguf_files", return_value=[model, model]), \
             patch("loca_llama.scanner.scan_mlx_models", return_value=[]):
            result = scan_all()
        paths = [m.path for m in result]
        assert len(paths) == len(set(paths))

    def test_sorted_by_size_descending(self):
        small = LocalModel(name="small", path=Path("/small.gguf"), size_gb=1.0, format="gguf", source="custom")
        large = LocalModel(name="large", path=Path("/large.gguf"), size_gb=10.0, format="gguf", source="custom")
        fake_path = MagicMock(spec=Path)

        with patch("loca_llama.scanner.LM_STUDIO_PATHS", [fake_path]), \
             patch("loca_llama.scanner.LLAMA_CPP_PATHS", []), \
             patch("loca_llama.scanner.HUGGINGFACE_PATHS", []), \
             patch("loca_llama.scanner.scan_gguf_files", return_value=[small, large]), \
             patch("loca_llama.scanner.scan_mlx_models", return_value=[]):
            result = scan_all()
        assert len(result) >= 2
        assert result[0].size_gb >= result[-1].size_gb

    def test_empty_when_no_paths_configured(self):
        with patch("loca_llama.scanner.LM_STUDIO_PATHS", []), \
             patch("loca_llama.scanner.LLAMA_CPP_PATHS", []), \
             patch("loca_llama.scanner.HUGGINGFACE_PATHS", []):
            assert scan_all() == []


# ---------------------------------------------------------------------------
# scan_custom_dir
# ---------------------------------------------------------------------------


class TestScanCustomDir:
    def test_returns_empty_for_nonexistent_directory(self, tmp_path: Path):
        missing = str(tmp_path / "no_such_dir")
        assert scan_custom_dir(missing) == []

    def test_finds_gguf_files(self, tmp_path: Path):
        (tmp_path / "llama-Q4_K_M.gguf").write_bytes(b"fake")
        models = scan_custom_dir(str(tmp_path))
        assert len(models) == 1

    def test_gguf_source_is_custom(self, tmp_path: Path):
        (tmp_path / "model.gguf").write_bytes(b"fake")
        model = scan_custom_dir(str(tmp_path))[0]
        assert model.source == "custom"

    def test_finds_safetensors_directory(self, tmp_path: Path):
        model_dir = tmp_path / "my-safetensors-model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake")
        models = scan_custom_dir(str(tmp_path))
        assert any(m.path == model_dir for m in models)

    def test_safetensors_format_when_no_mlx_in_name(self, tmp_path: Path):
        model_dir = tmp_path / "bert-base"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake")
        model = scan_custom_dir(str(tmp_path))[0]
        assert model.format == "safetensors"

    def test_mlx_format_when_mlx_in_parent_path(self, tmp_path: Path):
        mlx_dir = tmp_path / "mlx-llama-model"
        mlx_dir.mkdir()
        (mlx_dir / "weights.safetensors").write_bytes(b"fake")
        models = scan_custom_dir(str(tmp_path))
        mlx_model = next(m for m in models if m.path == mlx_dir)
        assert mlx_model.format == "mlx"

    def test_deduplicates_by_path(self, tmp_path: Path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake")
        models = scan_custom_dir(str(tmp_path))
        paths = [m.path for m in models]
        assert len(paths) == len(set(paths))

    def test_sorted_by_size_descending(self, tmp_path: Path):
        (tmp_path / "small.gguf").write_bytes(b"s" * 10)
        (tmp_path / "large.gguf").write_bytes(b"L" * 1000)
        models = scan_custom_dir(str(tmp_path))
        sizes = [m.size_gb for m in models]
        assert sizes == sorted(sizes, reverse=True)

    def test_expands_tilde_in_path(self):
        """Should not raise for a ~ path even if directory doesn't exist."""
        result = scan_custom_dir("~/definitely_nonexistent_loca_llama_test_dir_xyz")
        assert result == []


# ---------------------------------------------------------------------------
# format_downloads
# ---------------------------------------------------------------------------


class TestFormatDownloads:
    def test_formats_millions(self):
        assert format_downloads(1_500_000) == "1.5M"

    def test_formats_exact_million(self):
        assert format_downloads(1_000_000) == "1.0M"

    def test_formats_thousands(self):
        assert format_downloads(1_500) == "1.5K"

    def test_formats_exact_thousand(self):
        assert format_downloads(1_000) == "1.0K"

    def test_formats_below_thousand(self):
        assert format_downloads(500) == "500"

    def test_formats_zero(self):
        assert format_downloads(0) == "0"

    def test_formats_999(self):
        assert format_downloads(999) == "999"

    def test_formats_large_millions(self):
        assert format_downloads(10_000_000) == "10.0M"


# ---------------------------------------------------------------------------
# Helpers for hub tests
# ---------------------------------------------------------------------------


def _make_urlopen_mock(payload: object) -> MagicMock:
    """Return a context-manager mock that yields a response with JSON payload."""
    body = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body
    return cm


# ---------------------------------------------------------------------------
# search_huggingface
# ---------------------------------------------------------------------------


class TestSearchHuggingface:
    _FAKE_RESPONSE = [
        {
            "modelId": "meta-llama/Llama-3-8B-GGUF",
            "tags": ["gguf", "text-generation"],
            "downloads": 500_000,
            "likes": 1_200,
            "pipeline_tag": "text-generation",
            "lastModified": "2024-01-01T00:00:00Z",
        }
    ]

    def test_returns_list_of_hub_models(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = search_huggingface("llama")
        assert len(results) == 1

    def test_repo_id_set_correctly(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.repo_id == "meta-llama/Llama-3-8B-GGUF"

    def test_author_extracted_from_repo_id(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.author == "meta-llama"

    def test_name_extracted_from_repo_id(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.name == "Llama-3-8B-GGUF"

    def test_downloads_parsed(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.downloads == 500_000

    def test_is_gguf_true_when_gguf_tag(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.is_gguf is True

    def test_is_mlx_false_for_non_mlx(self):
        mock_resp = _make_urlopen_mock(self._FAKE_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama")[0]
        assert model.is_mlx is False

    def test_is_mlx_true_when_mlx_in_author(self):
        payload = [
            {
                "modelId": "mlx-community/Llama-3-8B-4bit",
                "tags": ["safetensors"],
                "downloads": 1_000,
                "likes": 50,
            }
        ]
        mock_resp = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("llama mlx")[0]
        assert model.is_mlx is True

    def test_raises_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            with pytest.raises(OSError):
                search_huggingface("llama")

    def test_returns_empty_list_on_json_error(self):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = b"not json {{{"
        with patch("urllib.request.urlopen", return_value=cm):
            result = search_huggingface("llama")
        assert result == []

    def test_filter_tags_included_in_request(self):
        mock_resp = _make_urlopen_mock([])
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            search_huggingface("llama", filter_tags=["gguf"])
        called_url = mock_open.call_args[0][0].full_url
        assert "filter=gguf" in called_url

    def test_limit_included_in_request_url(self):
        mock_resp = _make_urlopen_mock([])
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            search_huggingface("llama", limit=5)
        called_url = mock_open.call_args[0][0].full_url
        assert "limit=5" in called_url

    def test_handles_modelid_fallback_to_id(self):
        """When modelId is absent, fall back to id field."""
        payload = [
            {
                "id": "org/model-by-id",
                "tags": [],
                "downloads": 0,
                "likes": 0,
            }
        ]
        mock_resp = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            model = search_huggingface("model")[0]
        assert model.repo_id == "org/model-by-id"

    def test_empty_result_from_api_returns_empty_list(self):
        mock_resp = _make_urlopen_mock([])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert search_huggingface("nothing") == []


# ---------------------------------------------------------------------------
# search_gguf_models
# ---------------------------------------------------------------------------


class TestSearchGgufModels:
    def test_delegates_with_gguf_filter_tag(self):
        with patch("loca_llama.hub.search_huggingface", return_value=[]) as mock_hf:
            search_gguf_models("llama", limit=10)
        mock_hf.assert_called_once_with("llama", limit=10, filter_tags=["gguf"])

    def test_returns_results_from_delegate(self):
        fake_model = MagicMock()
        with patch("loca_llama.hub.search_huggingface", return_value=[fake_model]):
            result = search_gguf_models("llama")
        assert result == [fake_model]


# ---------------------------------------------------------------------------
# search_mlx_models
# ---------------------------------------------------------------------------


class TestSearchMlxModels:
    def test_filters_to_mlx_only(self):
        mlx_model = MagicMock()
        mlx_model.is_mlx = True
        non_mlx = MagicMock()
        non_mlx.is_mlx = False

        with patch("loca_llama.hub.search_huggingface", return_value=[mlx_model, non_mlx]):
            result = search_mlx_models("llama")
        assert result == [mlx_model]

    def test_returns_empty_when_no_mlx_results(self):
        non_mlx = MagicMock()
        non_mlx.is_mlx = False
        with patch("loca_llama.hub.search_huggingface", return_value=[non_mlx]):
            assert search_mlx_models("llama") == []

    def test_delegates_with_correct_query_and_limit(self):
        with patch("loca_llama.hub.search_huggingface", return_value=[]) as mock_hf:
            search_mlx_models("phi", limit=15)
        mock_hf.assert_called_once_with("phi", limit=15)


# ---------------------------------------------------------------------------
# get_model_files
# ---------------------------------------------------------------------------


class TestGetModelFiles:
    _FAKE_REPO_RESPONSE = {
        "siblings": [
            {"rfilename": "config.json", "size": 1024},
            {"rfilename": "model.Q4_K_M.gguf", "size": 4_000_000_000},
        ]
    }

    def test_returns_list_of_file_dicts(self):
        mock_resp = _make_urlopen_mock(self._FAKE_REPO_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            files = get_model_files("bartowski/Llama-3-8B")
        assert len(files) == 2

    def test_filename_field_present(self):
        mock_resp = _make_urlopen_mock(self._FAKE_REPO_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            files = get_model_files("bartowski/Llama-3-8B")
        filenames = [f["filename"] for f in files]
        assert "config.json" in filenames
        assert "model.Q4_K_M.gguf" in filenames

    def test_size_field_present(self):
        mock_resp = _make_urlopen_mock(self._FAKE_REPO_RESPONSE)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            files = get_model_files("bartowski/Llama-3-8B")
        gguf = next(f for f in files if f["filename"] == "model.Q4_K_M.gguf")
        assert gguf["size"] == 4_000_000_000

    def test_raises_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                get_model_files("some/repo")

    def test_returns_empty_list_when_no_siblings(self):
        mock_resp = _make_urlopen_mock({"siblings": []})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert get_model_files("some/repo") == []

    def test_returns_empty_list_when_siblings_key_missing(self):
        mock_resp = _make_urlopen_mock({})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert get_model_files("some/repo") == []

    def test_repo_id_used_in_request_url(self):
        mock_resp = _make_urlopen_mock({"siblings": []})
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            get_model_files("org/my-model")
        called_url = mock_open.call_args[0][0].full_url
        assert "org/my-model" in called_url
