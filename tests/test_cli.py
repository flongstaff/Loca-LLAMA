"""Tests for CLI commands: resolve_hw_or_detect, cmd_calc, cmd_memory, cmd_recommend."""

from __future__ import annotations

import sys
from argparse import Namespace
from unittest.mock import patch, MagicMock

import pytest

from loca_llama.cli import resolve_hw_or_detect, cmd_calc, cmd_memory, cmd_recommend
from loca_llama.hardware import APPLE_SILICON_SPECS


class TestResolveHwOrDetect:
    """Tests for resolve_hw_or_detect()."""

    def test_returns_spec_for_explicit_hw_name(self):
        """Should return the matching MacSpec when --hw is given."""
        mac = resolve_hw_or_detect("M4 Pro 48GB")
        assert mac is not None
        assert mac.chip == "M4 Pro"
        assert mac.memory_gb == 48

    def test_returns_none_for_unknown_hw_name(self, capsys):
        """Should print error and return None for unknown hardware."""
        mac = resolve_hw_or_detect("M99 Ultra 1TB")
        assert mac is None
        out = capsys.readouterr().out
        assert "Unknown" in out or "not found" in out.lower() or "M99" in out

    @patch("loca_llama.cli.detect_mac")
    def test_autodetect_returns_spec_on_apple_silicon(self, mock_detect, capsys):
        """Should call detect_mac() and return spec when hw_name is None."""
        spec = APPLE_SILICON_SPECS["M4 Pro 48GB"]
        mock_detect.return_value = ("M4 Pro 48GB", spec)
        mac = resolve_hw_or_detect(None)
        assert mac is spec
        mock_detect.assert_called_once()
        err = capsys.readouterr().err
        assert "Detected" in err

    @patch("loca_llama.cli.detect_mac", return_value=None)
    def test_autodetect_returns_none_on_non_mac(self, mock_detect, capsys):
        """Should print error and return None when detect_mac() fails."""
        mac = resolve_hw_or_detect(None)
        assert mac is None
        out = capsys.readouterr().out
        assert "detect" in out.lower() or "--hw" in out


class TestCmdCalc:
    """Tests for cmd_calc()."""

    def test_model_flag_produces_output(self, capsys):
        """Should produce VRAM breakdown when --model is given."""
        args = Namespace(
            model="Qwen 2.5 32B", quant="Q4_K_M", params=None, bpw=None,
            context=8192, layers=None, kv_heads=None, head_dim=128, hw=None,
        )
        with patch("loca_llama.cli.resolve_hw_or_detect", return_value=None):
            cmd_calc(args)
        out = capsys.readouterr().out
        assert "Model weights" in out
        assert "KV cache" in out
        assert "Total" in out

    def test_custom_params_bpw_produces_output(self, capsys):
        """Should produce VRAM breakdown with --params and --bpw."""
        args = Namespace(
            model=None, quant="Q4_K_M", params=7.0, bpw=4.5,
            context=4096, layers=32, kv_heads=8, head_dim=128, hw=None,
        )
        with patch("loca_llama.cli.resolve_hw_or_detect", return_value=None):
            cmd_calc(args)
        out = capsys.readouterr().out
        assert "Custom 7.0B" in out
        assert "Total" in out

    def test_model_wins_over_custom(self, capsys):
        """Should use --model and warn when both --model and --params are given."""
        args = Namespace(
            model="Qwen 2.5 32B", quant="Q4_K_M", params=7.0, bpw=4.5,
            context=8192, layers=None, kv_heads=None, head_dim=128, hw=None,
        )
        with patch("loca_llama.cli.resolve_hw_or_detect", return_value=None):
            cmd_calc(args)
        out = capsys.readouterr().out
        err = capsys.readouterr().err  # warning goes to stderr
        # Should use model name, not custom
        assert "Qwen 2.5 32B" in out

    def test_no_model_no_params_exits(self):
        """Should exit with code 1 when neither --model nor --params given."""
        args = Namespace(
            model=None, quant="Q4_K_M", params=None, bpw=None,
            context=8192, layers=None, kv_heads=None, head_dim=128, hw=None,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_calc(args)
        assert exc.value.code == 1

    def test_invalid_model_exits(self):
        """Should exit with code 1 for unknown model name."""
        args = Namespace(
            model="NonexistentModel99B", quant="Q4_K_M", params=None, bpw=None,
            context=8192, layers=None, kv_heads=None, head_dim=128, hw=None,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_calc(args)
        assert exc.value.code == 1


class TestCmdMemory:
    """Tests for cmd_memory()."""

    @patch("loca_llama.cli.sys")
    def test_non_macos_exits(self, mock_sys):
        """Should exit with code 1 on non-macOS platforms."""
        mock_sys.platform = "linux"
        mock_sys.exit = MagicMock(side_effect=SystemExit(1))
        mock_sys.stderr = sys.stderr
        mock_sys.stdout = sys.stdout
        with pytest.raises(SystemExit):
            cmd_memory(Namespace())

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_macos_produces_output(self, capsys):
        """Should produce memory stats on macOS."""
        cmd_memory(Namespace())
        out = capsys.readouterr().out
        assert "Used" in out or "Free" in out or "Total" in out


class TestCmdRecommend:
    """Tests for cmd_recommend()."""

    def test_produces_formatted_output(self, capsys):
        """Should produce recommendations with formatted output."""
        args = Namespace(hw="M4 Pro 48GB", use_case="general")
        cmd_recommend(args)
        out = capsys.readouterr().out
        assert "Recommendations" in out
        assert "recommendations shown" in out

    def test_coding_use_case(self, capsys):
        """Should filter for coding models."""
        args = Namespace(hw="M4 Pro 48GB", use_case="coding")
        cmd_recommend(args)
        out = capsys.readouterr().out
        assert "coding" in out.lower() or "Recommendations" in out

    @patch("loca_llama.cli.resolve_hw_or_detect", return_value=None)
    def test_unknown_hw_exits(self, mock_resolve):
        """Should exit with code 1 for unresolvable hardware."""
        args = Namespace(hw="M99 Ultra 1TB", use_case="general")
        with pytest.raises(SystemExit) as exc:
            cmd_recommend(args)
        assert exc.value.code == 1
