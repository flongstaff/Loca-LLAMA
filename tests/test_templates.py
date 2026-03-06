"""Tests for templates.py command generation."""

from loca_llama.templates import get_llama_cpp_command, get_llama_cpp_server_command, TEMPLATES


def test_llama_cpp_command_uses_jinja_not_interactive():
    """Modern llama.cpp uses --jinja --color, not -i."""
    tmpl = TEMPLATES[0]  # Any template
    cmd = get_llama_cpp_command(tmpl, "/models/test.gguf")
    assert "--jinja" in cmd
    assert "--color" in cmd
    assert " -i" not in cmd
    assert '"-i"' not in cmd


def test_llama_cpp_command_includes_flash_attention():
    """Should include -fa for flash attention on Apple Silicon."""
    tmpl = TEMPLATES[0]
    cmd = get_llama_cpp_command(tmpl, "/models/test.gguf")
    assert "-fa" in cmd


def test_llama_cpp_command_uses_template_defaults():
    """Without overrides, should use template sampling values."""
    tmpl = TEMPLATES[0]
    cmd = get_llama_cpp_command(tmpl, "/models/test.gguf")
    assert f"--temp {tmpl.temperature}" in cmd
    assert f"--top-p {tmpl.top_p}" in cmd
    assert f"--top-k {tmpl.top_k}" in cmd


def test_llama_cpp_command_sampling_overrides():
    """When sampling_overrides provided, use those instead of template values."""
    tmpl = TEMPLATES[0]
    overrides = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "min_p": 0.0,
    }
    cmd = get_llama_cpp_command(tmpl, "/models/test.gguf", sampling_overrides=overrides)
    assert "--temp 1.0" in cmd
    assert "--top-p 0.95" in cmd
    assert "--top-k 20" in cmd
    assert "--repeat-penalty 1.05" in cmd
    assert "--min-p 0.0" in cmd


def test_llama_cpp_command_partial_overrides():
    """Partial overrides should only replace specified values."""
    tmpl = TEMPLATES[0]
    overrides = {"temperature": 0.9}
    cmd = get_llama_cpp_command(tmpl, "/models/test.gguf", sampling_overrides=overrides)
    assert "--temp 0.9" in cmd
    # Other values should come from template
    assert f"--top-p {tmpl.top_p}" in cmd


def test_llama_server_command_includes_flash_attention():
    """llama-server should include -fa for Apple Silicon."""
    tmpl = TEMPLATES[0]
    cmd = get_llama_cpp_server_command(tmpl, "/models/test.gguf")
    assert "-fa" in cmd


def test_llama_server_command_no_interactive_flags():
    """llama-server should not include --jinja or --color (those are CLI-only)."""
    tmpl = TEMPLATES[0]
    cmd = get_llama_cpp_server_command(tmpl, "/models/test.gguf")
    assert "--jinja" not in cmd
    assert " -i" not in cmd


def test_detect_non_gguf_format():
    """Should detect non-GGUF model formats from path."""
    from loca_llama.templates import detect_model_format_warning

    assert detect_model_format_warning("/models/mlx-community/Qwen-4bit") is not None
    assert detect_model_format_warning("/models/model.safetensors") is not None
    assert detect_model_format_warning("/models/model.gguf") is None
    assert detect_model_format_warning("/models/some-model/Q4_K_M.gguf") is None
