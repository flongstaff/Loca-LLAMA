"""Data integrity tests for the hardware, models, and quantization databases."""

from __future__ import annotations

import pytest

from loca_llama.hardware import APPLE_SILICON_SPECS
from loca_llama.models import MODELS
from loca_llama.quantization import QUANT_FORMATS, RECOMMENDED_FORMATS


class TestHardwareData:
    def test_database_is_not_empty(self):
        assert len(APPLE_SILICON_SPECS) > 0

    def test_all_spec_names_are_non_empty_strings(self):
        for name in APPLE_SILICON_SPECS:
            assert isinstance(name, str) and name.strip() != ""

    def test_all_specs_have_positive_memory_gb(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.memory_gb > 0, f"{name}: memory_gb must be positive"

    def test_all_specs_have_usable_memory_lte_total(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.usable_memory_gb <= spec.memory_gb, (
                f"{name}: usable_memory_gb ({spec.usable_memory_gb}) "
                f"exceeds memory_gb ({spec.memory_gb})"
            )

    def test_all_specs_have_positive_gpu_cores(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.gpu_cores > 0, f"{name}: gpu_cores must be positive"

    def test_no_duplicate_chip_memory_core_combinations(self):
        seen: set[tuple[str, int, int]] = set()
        for name, spec in APPLE_SILICON_SPECS.items():
            key = (spec.chip, spec.memory_gb, spec.cpu_cores)
            assert key not in seen, (
                f"Duplicate config: chip={spec.chip}, memory_gb={spec.memory_gb}, "
                f"cpu_cores={spec.cpu_cores} (found in entry '{name}')"
            )
            seen.add(key)

    def test_all_specs_have_positive_cpu_cores(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.cpu_cores > 0, f"{name}: cpu_cores must be positive"

    def test_all_specs_have_positive_memory_bandwidth(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.memory_bandwidth_gbs > 0, f"{name}: memory_bandwidth_gbs must be positive"

    def test_all_specs_have_positive_gpu_tflops(self):
        for name, spec in APPLE_SILICON_SPECS.items():
            assert spec.gpu_tflops > 0, f"{name}: gpu_tflops must be positive"


class TestModelsData:
    def test_models_list_is_not_empty(self):
        assert len(MODELS) > 0

    def test_all_model_names_are_unique(self):
        names = [m.name for m in MODELS]
        assert len(names) == len(set(names)), "Duplicate model names found"

    def test_all_params_billion_are_positive(self):
        for model in MODELS:
            assert model.params_billion > 0, f"{model.name}: params_billion must be positive"

    def test_default_context_lte_max_context(self):
        for model in MODELS:
            assert model.default_context_length <= model.max_context_length, (
                f"{model.name}: default_context_length ({model.default_context_length}) "
                f"exceeds max_context_length ({model.max_context_length})"
            )

    def test_all_num_layers_are_positive(self):
        for model in MODELS:
            assert model.num_layers > 0, f"{model.name}: num_layers must be positive"

    def test_all_num_kv_heads_are_positive(self):
        for model in MODELS:
            assert model.num_kv_heads > 0, f"{model.name}: num_kv_heads must be positive"

    def test_all_head_dim_are_positive(self):
        for model in MODELS:
            assert model.head_dim > 0, f"{model.name}: head_dim must be positive"

    def test_all_models_have_non_empty_family(self):
        for model in MODELS:
            assert isinstance(model.family, str) and model.family.strip() != "", (
                f"{model.name}: family must be a non-empty string"
            )

    def test_all_models_have_non_empty_name(self):
        for model in MODELS:
            assert isinstance(model.name, str) and model.name.strip() != ""

    def test_all_models_have_non_empty_license(self):
        for model in MODELS:
            assert isinstance(model.license, str) and model.license.strip() != "", (
                f"{model.name}: license must be a non-empty string"
            )


class TestQuantizationData:
    def test_quant_formats_dict_is_not_empty(self):
        assert len(QUANT_FORMATS) > 0

    def test_all_bits_per_weight_are_positive(self):
        for name, fmt in QUANT_FORMATS.items():
            assert fmt.bits_per_weight > 0, f"{name}: bits_per_weight must be positive"

    def test_all_bits_per_weight_are_at_most_16(self):
        for name, fmt in QUANT_FORMATS.items():
            assert fmt.bits_per_weight <= 16, (
                f"{name}: bits_per_weight ({fmt.bits_per_weight}) exceeds 16"
            )

    def test_all_recommended_formats_exist_in_quant_formats(self):
        for fmt_name in RECOMMENDED_FORMATS:
            assert fmt_name in QUANT_FORMATS, (
                f"Recommended format '{fmt_name}' not found in QUANT_FORMATS"
            )

    def test_all_formats_have_non_empty_name(self):
        for key, fmt in QUANT_FORMATS.items():
            assert isinstance(fmt.name, str) and fmt.name.strip() != "", (
                f"Format keyed '{key}': name must be a non-empty string"
            )

    def test_all_formats_have_non_empty_description(self):
        for key, fmt in QUANT_FORMATS.items():
            assert isinstance(fmt.description, str) and fmt.description.strip() != "", (
                f"Format keyed '{key}': description must be a non-empty string"
            )

    def test_dict_keys_match_format_names(self):
        """Dict keys must equal the .name attribute on each format."""
        for key, fmt in QUANT_FORMATS.items():
            assert key == fmt.name, (
                f"Key '{key}' does not match format name '{fmt.name}'"
            )

    def test_recommended_formats_list_is_not_empty(self):
        assert len(RECOMMENDED_FORMATS) > 0

    def test_no_duplicate_recommended_formats(self):
        assert len(RECOMMENDED_FORMATS) == len(set(RECOMMENDED_FORMATS)), (
            "RECOMMENDED_FORMATS contains duplicates"
        )
