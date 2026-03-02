"""Unit tests for the core analysis engine."""

from __future__ import annotations

import pytest

from loca_llama.analyzer import (
    CompatibilityTier,
    analyze_model,
    compute_tier,
    estimate_kv_cache_gb,
    estimate_model_size_gb,
    estimate_overhead_gb,
    max_context_for_model,
)
from loca_llama.hardware import APPLE_SILICON_SPECS
from loca_llama.models import MODELS
from loca_llama.quantization import QUANT_FORMATS


@pytest.fixture
def m4_pro_48():
    return APPLE_SILICON_SPECS["M4 Pro 48GB"]


@pytest.fixture
def qwen_32b():
    return next(m for m in MODELS if m.name == "Qwen 2.5 32B")


@pytest.fixture
def q4km():
    return QUANT_FORMATS["Q4_K_M"]


class TestEstimateModelSize:
    def test_known_model_size(self):
        """Q4_K_M (4.85 bpw) on a 32B model should be ~18-19 GB."""
        size = estimate_model_size_gb(32.0, 4.85)
        assert 17.0 < size < 20.0

    def test_higher_bpw_means_larger(self):
        """Higher bits per weight should produce larger model."""
        small = estimate_model_size_gb(7.0, 4.0)
        large = estimate_model_size_gb(7.0, 8.0)
        assert large > small

    def test_zero_params(self):
        assert estimate_model_size_gb(0.0, 4.0) == 0.0


class TestEstimateKvCache:
    def test_kv_cache_scales_with_context(self, qwen_32b):
        """Doubling context should roughly double KV cache."""
        kv_4k = estimate_kv_cache_gb(qwen_32b, 4096)
        kv_8k = estimate_kv_cache_gb(qwen_32b, 8192)
        assert abs(kv_8k / kv_4k - 2.0) < 0.01

    def test_kv_cache_positive(self, qwen_32b):
        kv = estimate_kv_cache_gb(qwen_32b, 4096)
        assert kv > 0


class TestEstimateOverhead:
    def test_overhead_includes_base(self):
        """Even a tiny model has >=0.5 GB base overhead."""
        assert estimate_overhead_gb(0.1) >= 0.5

    def test_overhead_scales(self):
        """Larger model gets more overhead."""
        small = estimate_overhead_gb(5.0)
        large = estimate_overhead_gb(20.0)
        assert large > small


class TestComputeTier:
    def test_full_gpu_under_75pct(self, m4_pro_48):
        """Small model on big hardware should be FULL_GPU."""
        small_model = next(m for m in MODELS if m.params_billion <= 3)
        tier, gpu_layers, offload_pct = compute_tier(m4_pro_48, 2.0, 5.0, small_model)
        assert tier == CompatibilityTier.FULL_GPU
        assert gpu_layers == small_model.num_layers
        assert offload_pct == 100.0

    def test_wont_fit_over_150pct(self, m4_pro_48):
        """Model using >150% memory should be WONT_FIT."""
        big_model = next(m for m in MODELS if m.params_billion >= 70)
        huge_total = m4_pro_48.usable_memory_gb * 2.0
        tier, gpu_layers, offload_pct = compute_tier(m4_pro_48, huge_total, huge_total, big_model)
        assert tier == CompatibilityTier.WONT_FIT
        assert gpu_layers is None


class TestAnalyzeModel:
    def test_returns_valid_estimate(self, m4_pro_48, qwen_32b, q4km):
        est = analyze_model(m4_pro_48, qwen_32b, q4km)
        assert est.model.name == "Qwen 2.5 32B"
        assert est.quant.name == "Q4_K_M"
        assert est.total_memory_gb > 0
        assert est.available_memory_gb == m4_pro_48.usable_memory_gb
        assert est.tier in CompatibilityTier

    def test_default_context_used(self, m4_pro_48, qwen_32b, q4km):
        """Without explicit context, should use model's default."""
        est = analyze_model(m4_pro_48, qwen_32b, q4km)
        assert est.context_length == qwen_32b.default_context_length

    def test_custom_context(self, m4_pro_48, qwen_32b, q4km):
        est = analyze_model(m4_pro_48, qwen_32b, q4km, context_length=2048)
        assert est.context_length == 2048

    def test_memory_breakdown_adds_up(self, m4_pro_48, qwen_32b, q4km):
        est = analyze_model(m4_pro_48, qwen_32b, q4km)
        expected = est.model_size_gb + est.kv_cache_gb + est.overhead_gb
        assert abs(est.total_memory_gb - expected) < 0.01


class TestMaxContext:
    def test_positive_for_small_model(self, m4_pro_48, q4km):
        """Small model on big hardware should have positive max context."""
        small_model = next(m for m in MODELS if m.params_billion <= 3)
        max_ctx = max_context_for_model(m4_pro_48, small_model, q4km)
        assert max_ctx > 0

    def test_respects_model_max(self, m4_pro_48, q4km):
        """Max context should never exceed model's max_context_length."""
        small_model = next(m for m in MODELS if m.params_billion <= 3)
        max_ctx = max_context_for_model(m4_pro_48, small_model, q4km)
        assert max_ctx <= small_model.max_context_length

    def test_zero_for_huge_model(self, q4km):
        """A model too large for the hardware should get 0 context."""
        tiny_hw = APPLE_SILICON_SPECS["M1 8GB"]
        huge_model = next(m for m in MODELS if m.params_billion >= 70)
        max_ctx = max_context_for_model(tiny_hw, huge_model, q4km)
        assert max_ctx == 0
