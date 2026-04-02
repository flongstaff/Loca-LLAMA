"""Unit tests for runtime.py and benchmark.py modules."""

from __future__ import annotations

import json
import subprocess
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest

from loca_llama.runtime import (
    LoadedModel,
    LlamaCppConnector,
    LMStudioConnector,
    detect_all_connectors,
)
from loca_llama.benchmark import (
    BenchmarkResult,
    RuntimeInfo,
    BENCH_MAX_TOKENS,
    BENCH_PROMPTS,
    _make_fail_result,
    aggregate_results,
    benchmark_llama_cpp_native,
    benchmark_openai_api,
    benchmark_openai_api_streaming,
    detect_all_runtimes,
    detect_llama_cpp_server,
    detect_lm_studio,
    run_benchmark_suite,
    run_benchmark_sweep,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_response(data: dict) -> MagicMock:
    """Build a context-manager mock that returns JSON-encoded data."""
    payload = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_success_result(
    model_name: str = "test-model",
    runtime: str = "lm-studio",
    run_number: int = 1,
    tokens_per_second: float = 50.0,
    prompt_tokens_per_second: float = 100.0,
    prompt_eval_time_ms: float = 200.0,
    total_time_ms: float = 1000.0,
    generated_tokens: int = 100,
) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        runtime=runtime,
        prompt_tokens=10,
        generated_tokens=generated_tokens,
        prompt_eval_time_ms=prompt_eval_time_ms,
        eval_time_ms=800.0,
        total_time_ms=total_time_ms,
        tokens_per_second=tokens_per_second,
        prompt_tokens_per_second=prompt_tokens_per_second,
        context_length=4096,
        success=True,
        run_number=run_number,
    )


# ── LoadedModel ────────────────────────────────────────────────────────────────

class TestLoadedModel:
    def test_should_store_required_fields_when_constructed(self):
        model = LoadedModel(model_id="llama-3-8b", runtime="lm-studio")
        assert model.model_id == "llama-3-8b"
        assert model.runtime == "lm-studio"

    def test_should_default_optional_fields_to_none_when_not_provided(self):
        model = LoadedModel(model_id="x", runtime="y")
        assert model.context_length is None
        assert model.gpu_layers is None

    def test_should_store_optional_fields_when_provided(self):
        model = LoadedModel(model_id="x", runtime="y", context_length=4096, gpu_layers=32)
        assert model.context_length == 4096
        assert model.gpu_layers == 32


# ── LMStudioConnector ──────────────────────────────────────────────────────────

class TestLMStudioConnectorIsRunning:
    def test_should_return_true_when_urlopen_succeeds(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            assert connector.is_running() is True

    def test_should_return_false_when_urlopen_raises(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert connector.is_running() is False

    def test_should_use_configured_base_url_when_checking(self):
        connector = LMStudioConnector(base_url="http://127.0.0.1:9999")
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            connector.is_running()
            url_used = mock_open.call_args[0][0]
            assert "9999" in url_used


class TestLMStudioConnectorListModels:
    def test_should_return_model_ids_when_response_has_data(self):
        connector = LMStudioConnector()
        data = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        with patch("urllib.request.urlopen", return_value=_json_response(data)):
            result = connector.list_models()
        assert result == ["model-a", "model-b"]

    def test_should_return_empty_list_when_data_key_is_missing(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", return_value=_json_response({})):
            result = connector.list_models()
        assert result == []

    def test_should_return_empty_list_when_urlopen_raises(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = connector.list_models()
        assert result == []


class TestLMStudioConnectorGetModelInfo:
    def test_should_return_model_dict_when_request_succeeds(self):
        connector = LMStudioConnector()
        payload = {"id": "llama-3-8b", "object": "model"}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.get_model_info("llama-3-8b")
        assert result == payload

    def test_should_return_none_when_request_raises(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("not found")):
            result = connector.get_model_info("missing-model")
        assert result is None

    def test_should_url_encode_model_id_with_slashes(self):
        connector = LMStudioConnector()
        payload = {"id": "org/model"}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)) as mock_open:
            connector.get_model_info("org/model")
            url_used = mock_open.call_args[0][0]
            assert "org%2Fmodel" in url_used


class TestLMStudioConnectorChat:
    def test_should_return_parsed_response_when_request_succeeds(self):
        connector = LMStudioConnector()
        payload = {"choices": [{"message": {"content": "hello"}}]}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.chat("llama-3-8b", [{"role": "user", "content": "hi"}])
        assert result == payload

    def test_should_send_post_request_with_correct_model_id(self):
        connector = LMStudioConnector()
        payload = {"choices": []}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)) as mock_open:
            connector.chat("my-model", [{"role": "user", "content": "hi"}])
            req = mock_open.call_args[0][0]
            body = json.loads(req.data.decode())
            assert body["model"] == "my-model"

    def test_should_include_messages_in_request_body(self):
        connector = LMStudioConnector()
        messages = [{"role": "user", "content": "tell me a joke"}]
        with patch("urllib.request.urlopen", return_value=_json_response({})) as mock_open:
            connector.chat("m", messages)
            req = mock_open.call_args[0][0]
            body = json.loads(req.data.decode())
            assert body["messages"] == messages

    def test_should_propagate_exception_when_urlopen_raises(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                connector.chat("m", [])


class TestLMStudioConnectorComplete:
    def test_should_return_parsed_response_when_request_succeeds(self):
        connector = LMStudioConnector()
        payload = {"choices": [{"text": "world"}]}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.complete("llama-3-8b", "hello")
        assert result == payload

    def test_should_include_prompt_in_request_body(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", return_value=_json_response({})) as mock_open:
            connector.complete("m", "my prompt")
            req = mock_open.call_args[0][0]
            body = json.loads(req.data.decode())
            assert body["prompt"] == "my prompt"

    def test_should_propagate_exception_when_urlopen_raises(self):
        connector = LMStudioConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                connector.complete("m", "prompt")


# ── LlamaCppConnector ──────────────────────────────────────────────────────────

class TestLlamaCppConnectorIsRunning:
    def test_should_return_true_when_status_is_ok(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", return_value=_json_response({"status": "ok"})):
            assert connector.is_running() is True

    def test_should_return_false_when_status_is_not_ok(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", return_value=_json_response({"status": "loading"})):
            assert connector.is_running() is False

    def test_should_return_false_when_urlopen_raises(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert connector.is_running() is False

    def test_should_return_false_when_status_key_is_missing(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", return_value=_json_response({})):
            assert connector.is_running() is False


class TestLlamaCppConnectorHealth:
    def test_should_return_health_dict_when_request_succeeds(self):
        connector = LlamaCppConnector()
        payload = {"status": "ok", "slots_idle": 1}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.health()
        assert result == payload

    def test_should_return_error_dict_when_urlopen_raises(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = connector.health()
        assert result == {"status": "error"}


class TestLlamaCppConnectorListModels:
    def test_should_return_model_ids_when_response_has_data(self):
        connector = LlamaCppConnector()
        data = {"data": [{"id": "llama-3-8b.gguf"}]}
        with patch("urllib.request.urlopen", return_value=_json_response(data)):
            result = connector.list_models()
        assert result == ["llama-3-8b.gguf"]

    def test_should_return_empty_list_when_urlopen_raises(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = connector.list_models()
        assert result == []


class TestLlamaCppConnectorGetProps:
    def test_should_return_props_dict_when_request_succeeds(self):
        connector = LlamaCppConnector()
        payload = {"n_ctx": 4096, "model": "llama-3"}
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.get_props()
        assert result == payload

    def test_should_return_empty_dict_when_urlopen_raises(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = connector.get_props()
        assert result == {}


class TestLlamaCppConnectorGetSlots:
    def test_should_return_slots_list_when_request_succeeds(self):
        connector = LlamaCppConnector()
        payload = [{"id": 0, "state": 0}]
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            result = connector.get_slots()
        assert result == payload

    def test_should_return_empty_list_when_urlopen_raises(self):
        connector = LlamaCppConnector()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = connector.get_slots()
        assert result == []


class TestLlamaCppConnectorChat:
    def test_should_return_parsed_response_when_request_succeeds(self):
        connector = LlamaCppConnector()
        models_resp = _json_response({"data": [{"id": "llama-3"}]})
        chat_resp = _json_response({"choices": [{"message": {"content": "hi"}}]})
        with patch("urllib.request.urlopen", side_effect=[models_resp, chat_resp]):
            result = connector.chat([{"role": "user", "content": "hey"}])
        assert "choices" in result

    def test_should_use_first_model_when_models_available(self):
        connector = LlamaCppConnector()
        models_resp = _json_response({"data": [{"id": "first-model"}, {"id": "second-model"}]})
        chat_resp = _json_response({})
        with patch("urllib.request.urlopen", side_effect=[models_resp, chat_resp]) as mock_open:
            connector.chat([])
            chat_req = mock_open.call_args[0][0]
            body = json.loads(chat_req.data.decode())
            assert body["model"] == "first-model"

    def test_should_use_default_model_id_when_no_models_available(self):
        connector = LlamaCppConnector()
        models_resp = _json_response({"data": []})
        chat_resp = _json_response({})
        with patch("urllib.request.urlopen", side_effect=[models_resp, chat_resp]) as mock_open:
            connector.chat([])
            chat_req = mock_open.call_args[0][0]
            body = json.loads(chat_req.data.decode())
            assert body["model"] == "default"


class TestLlamaCppConnectorStartServer:
    def test_should_return_none_when_llama_server_not_found(self):
        with patch("shutil.which", return_value=None):
            result = LlamaCppConnector.start_server("/path/to/model.gguf")
        assert result is None

    def test_should_return_popen_when_server_starts_and_health_check_passes(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        health_resp = _json_response({"status": "ok"})

        with patch("shutil.which", return_value="/usr/local/bin/llama-server"), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"), \
             patch("urllib.request.urlopen", return_value=health_resp):
            result = LlamaCppConnector.start_server("/models/llama.gguf", port=8080)

        assert result is mock_proc

    def test_should_return_none_when_process_exits_before_health_check_passes(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # process exited with error

        with patch("shutil.which", return_value="/usr/local/bin/llama-server"), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"), \
             patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = LlamaCppConnector.start_server("/models/llama.gguf")

        assert result is None

    def test_should_include_threads_flag_when_threads_specified(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        health_resp = _json_response({"status": "ok"})

        with patch("shutil.which", return_value="/usr/local/bin/llama-server"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("time.sleep"), \
             patch("urllib.request.urlopen", return_value=health_resp):
            LlamaCppConnector.start_server("/models/llama.gguf", threads=8)

        cmd = mock_popen.call_args[0][0]
        assert "-t" in cmd
        assert "8" in cmd

    def test_should_fallback_to_server_exe_when_llama_server_not_found(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        health_resp = _json_response({"status": "ok"})

        def which_side_effect(name: str) -> str | None:
            return "/usr/local/bin/server" if name == "server" else None

        with patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("time.sleep"), \
             patch("urllib.request.urlopen", return_value=health_resp):
            result = LlamaCppConnector.start_server("/models/llama.gguf")

        assert result is mock_proc
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/server"


# ── detect_all_connectors ──────────────────────────────────────────────────────

class TestDetectAllConnectors:
    def test_should_include_lm_studio_when_it_is_running(self):
        with patch.object(LMStudioConnector, "is_running", return_value=True), \
             patch.object(LlamaCppConnector, "is_running", return_value=False):
            result = detect_all_connectors()
        assert "lm-studio" in result

    def test_should_exclude_lm_studio_when_it_is_not_running(self):
        with patch.object(LMStudioConnector, "is_running", return_value=False), \
             patch.object(LlamaCppConnector, "is_running", return_value=False):
            result = detect_all_connectors()
        assert "lm-studio" not in result

    def test_should_include_llama_cpp_connector_when_it_is_running(self):
        with patch.object(LMStudioConnector, "is_running", return_value=False), \
             patch.object(LlamaCppConnector, "is_running", return_value=True):
            result = detect_all_connectors()
        assert any(k.startswith("llama.cpp:") for k in result)

    def test_should_return_empty_dict_when_no_connectors_are_running(self):
        with patch.object(LMStudioConnector, "is_running", return_value=False), \
             patch.object(LlamaCppConnector, "is_running", return_value=False):
            result = detect_all_connectors()
        assert result == {}

    def test_should_only_include_first_available_llama_cpp_port(self):
        call_count = 0

        def is_running_side_effect(self: LlamaCppConnector) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count == 1  # only first call returns True

        with patch.object(LMStudioConnector, "is_running", return_value=False), \
             patch.object(LlamaCppConnector, "is_running", is_running_side_effect):
            result = detect_all_connectors()

        llama_keys = [k for k in result if k.startswith("llama.cpp:")]
        assert len(llama_keys) == 1


# ── BenchmarkResult ────────────────────────────────────────────────────────────

class TestBenchmarkResult:
    def test_should_return_prompt_eval_time_as_time_to_first_token(self):
        result = _make_success_result(prompt_eval_time_ms=350.5)
        assert result.time_to_first_token_ms == 350.5

    def test_should_construct_with_required_fields(self):
        result = BenchmarkResult(
            model_name="llama-3-8b",
            runtime="lm-studio",
            prompt_tokens=10,
            generated_tokens=50,
            prompt_eval_time_ms=200.0,
            eval_time_ms=800.0,
            total_time_ms=1000.0,
            tokens_per_second=62.5,
            prompt_tokens_per_second=50.0,
            context_length=4096,
            success=True,
        )
        assert result.model_name == "llama-3-8b"
        assert result.success is True

    def test_should_default_run_number_to_one(self):
        result = _make_success_result()
        assert result.run_number == 1

    def test_should_default_extra_to_empty_dict(self):
        result = _make_success_result()
        assert result.extra == {}


# ── _make_fail_result ──────────────────────────────────────────────────────────

class TestMakeFailResult:
    def test_should_return_result_with_success_false(self):
        result = _make_fail_result("my-model", "lm-studio", 4096, "timeout")
        assert result.success is False

    def test_should_store_error_message(self):
        result = _make_fail_result("my-model", "lm-studio", 4096, "connection refused")
        assert result.error == "connection refused"

    def test_should_store_model_name_and_runtime(self):
        result = _make_fail_result("llama-3", "llama.cpp-cli", 2048, "err")
        assert result.model_name == "llama-3"
        assert result.runtime == "llama.cpp-cli"

    def test_should_store_context_length(self):
        result = _make_fail_result("m", "r", 8192, "err")
        assert result.context_length == 8192

    def test_should_store_run_number_when_provided(self):
        result = _make_fail_result("m", "r", 4096, "err", run=3)
        assert result.run_number == 3

    def test_should_set_all_token_counts_to_zero(self):
        result = _make_fail_result("m", "r", 4096, "err")
        assert result.prompt_tokens == 0
        assert result.generated_tokens == 0

    def test_should_set_all_timing_fields_to_zero(self):
        result = _make_fail_result("m", "r", 4096, "err")
        assert result.prompt_eval_time_ms == 0
        assert result.eval_time_ms == 0
        assert result.total_time_ms == 0


# ── benchmark_openai_api ───────────────────────────────────────────────────────

def _sse_api_response(
    content_tokens: list[str] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 50,
    prompt_eval_ms: float = 0.0,
    eval_ms: float = 0.0,
    include_timings: bool = True,
) -> MagicMock:
    """Build an SSE streaming response mock for benchmark_openai_api.

    Produces content delta chunks followed by a final chunk with usage/timings.
    """
    if content_tokens is None:
        content_tokens = ["Hello", " world"]
    events = []
    for tok in content_tokens:
        events.append(f'data: {json.dumps({"choices": [{"delta": {"content": tok}}]})}')
    # Final chunk with usage and optional timings (like llama.cpp server)
    final: dict = {
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "choices": [{"delta": {}}],
    }
    if include_timings:
        final["timings"] = {"prompt_eval_time_ms": prompt_eval_ms, "eval_time_ms": eval_ms}
    events.append(f"data: {json.dumps(final)}")
    events.append("data: [DONE]")

    raw = "\n".join(events) + "\n"
    stream = BytesIO(raw.encode())
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.__iter__ = lambda s: iter(stream.readlines())
    return resp


class TestBenchmarkOpenaiApi:
    def test_should_return_success_result_when_request_succeeds(self):
        resp = _sse_api_response(prompt_tokens=10, completion_tokens=50, eval_ms=800.0)
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        assert result.success is True

    def test_should_extract_usage_tokens_from_response(self):
        resp = _sse_api_response(prompt_tokens=15, completion_tokens=75)
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        assert result.prompt_tokens == 15
        assert result.generated_tokens == 75

    def test_should_extract_timings_from_response_when_present(self):
        resp = _sse_api_response(prompt_eval_ms=250.0, eval_ms=750.0)
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        assert result.prompt_eval_time_ms == 250.0
        assert result.eval_time_ms == 750.0

    def test_should_estimate_timings_from_total_when_no_timings_in_response(self):
        resp = _sse_api_response(
            prompt_tokens=10, completion_tokens=50, include_timings=False,
        )
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        # Without server timings, function estimates from measured wall-clock
        assert result.prompt_eval_time_ms > 0
        assert result.eval_time_ms > 0

    def test_should_compute_tokens_per_second_from_timing(self):
        resp = _sse_api_response(
            content_tokens=[f"t{i}" for i in range(100)],
            completion_tokens=100, eval_ms=1000.0,
        )
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        assert result.tokens_per_second == pytest.approx(99.0)

    def test_should_return_fail_result_when_urlopen_raises(self):
        exc = urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))
        with patch("urllib.request.urlopen", side_effect=exc):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "lm-studio")
        assert result.success is False
        assert "Cannot connect" in result.error

    def test_should_store_run_number_in_result(self):
        resp = _sse_api_response(prompt_tokens=10, completion_tokens=50)
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api(
                "http://127.0.0.1:1234", "llama-3", "lm-studio", run_number=3
            )
        assert result.run_number == 3

    def test_should_store_model_name_and_runtime_in_result(self):
        resp = _sse_api_response(prompt_tokens=10, completion_tokens=50)
        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api(
                "http://127.0.0.1:1234", "my-model", "my-runtime"
            )
        assert result.model_name == "my-model"
        assert result.runtime == "my-runtime"

    def test_should_accept_alternate_timing_key_names(self):
        """llama.cpp server uses prompt_ms/predicted_ms keys."""
        # Build SSE stream with alternate timing keys in final chunk
        events = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "Hello"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": " world"}}]})}',
        ]
        final = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 50},
            "timings": {"prompt_ms": 200.0, "predicted_ms": 800.0},
            "choices": [{"delta": {}}],
        }
        events.append(f"data: {json.dumps(final)}")
        events.append("data: [DONE]")
        raw = "\n".join(events) + "\n"
        stream = BytesIO(raw.encode())
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.__iter__ = lambda s: iter(stream.readlines())

        with patch("urllib.request.urlopen", return_value=resp):
            result = benchmark_openai_api("http://127.0.0.1:1234", "llama-3", "llama.cpp-server")
        assert result.prompt_eval_time_ms == 200.0
        assert result.eval_time_ms == 800.0


# ── benchmark_llama_cpp_native ─────────────────────────────────────────────────

class TestBenchmarkLlamaCppNative:
    _TIMING_OUTPUT = (
        "llama_print_timings: prompt eval time = 200.00 ms / 10 tokens\n"
        "llama_print_timings:       eval time = 800.00 ms / 100 tokens\n"
    )

    def test_should_return_fail_result_when_cli_not_found(self):
        with patch("shutil.which", return_value=None):
            result = benchmark_llama_cpp_native("/models/llama.gguf")
        assert result.success is False
        assert "not found" in result.error

    def test_should_use_model_stem_as_model_name_when_cli_not_found(self):
        with patch("shutil.which", return_value=None):
            result = benchmark_llama_cpp_native("/models/my-model.gguf")
        assert result.model_name == "my-model"

    def test_should_return_success_result_when_subprocess_succeeds(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = self._TIMING_OUTPUT
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is True

    def test_should_extract_prompt_eval_timing_from_stderr(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = self._TIMING_OUTPUT
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.prompt_eval_time_ms == 200.0
        assert result.prompt_tokens == 10

    def test_should_extract_eval_timing_from_stderr(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = self._TIMING_OUTPUT
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.eval_time_ms == 800.0
        assert result.generated_tokens == 100

    def test_should_return_fail_result_on_timeout(self):
        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="llama-cli", timeout=180)):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is False
        assert "Timed out" in result.error

    def test_should_return_fail_result_on_generic_exception(self):
        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", side_effect=RuntimeError("unexpected error")):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is False
        assert "unexpected error" in result.error

    def test_should_return_fail_result_when_nonzero_exit_code(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is False

    def test_should_compute_tokens_per_second_from_parsed_timing(self):
        output = (
            "llama_print_timings: prompt eval time = 200.00 ms / 10 tokens\n"
            "llama_print_timings:       eval time = 1000.00 ms / 50 tokens\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = output
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.tokens_per_second == pytest.approx(50.0)

    def test_should_store_run_number_in_result(self):
        with patch("shutil.which", return_value=None):
            result = benchmark_llama_cpp_native("/models/llama.gguf", run_number=2)
        assert result.run_number == 2

    def test_should_parse_new_format_prompt_and_generation_speeds(self):
        """New llama.cpp builds (b8000+) output [ Prompt: X t/s | Generation: Y t/s ]."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "some output\n[ Prompt: 132.3 t/s | Generation: 28.3 t/s ]\n"

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is True
        assert result.prompt_tokens_per_second == pytest.approx(132.3)
        assert result.tokens_per_second == pytest.approx(28.3)

    def test_should_prefer_old_format_over_new_when_both_present(self):
        """Old format is more detailed (has token counts), prefer it when available."""
        output = (
            "llama_print_timings: prompt eval time = 200.00 ms / 10 tokens\n"
            "llama_print_timings:       eval time = 800.00 ms / 100 tokens\n"
            "[ Prompt: 50.0 t/s | Generation: 125.0 t/s ]\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = output
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.generated_tokens == 100
        assert result.prompt_tokens == 10

    def test_should_return_descriptive_error_when_output_unparseable(self):
        """When returncode is 0 but no timing output is found, error should be descriptive."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "model loaded successfully\nsome text output\n"

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result):
            result = benchmark_llama_cpp_native("/models/llama.gguf")

        assert result.success is False
        assert "Could not parse" in result.error

    def test_should_include_no_conversation_flag_in_command(self):
        """Ensure --no-conversation flag is passed to prevent interactive mode."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = self._TIMING_OUTPUT
        mock_result.stdout = ""

        with patch("shutil.which", return_value="/usr/local/bin/llama-cli"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            benchmark_llama_cpp_native("/models/llama.gguf")

        cmd = mock_run.call_args[0][0]
        assert "--no-conversation" in cmd


# ── run_benchmark_suite ────────────────────────────────────────────────────────

class TestRunBenchmarkSuite:
    def _make_runtime(self, name: str = "lm-studio", url: str = "http://127.0.0.1:1234") -> RuntimeInfo:
        return RuntimeInfo(name=name, url=url, models=["test-model"])

    def test_should_return_one_result_per_run(self):
        runtime = self._make_runtime()
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 50}, "timings": {}}
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()):
            results = run_benchmark_suite(runtime, "test-model", num_runs=3)
        assert len(results) == 3

    def test_should_call_progress_callback_for_each_run(self):
        runtime = self._make_runtime()
        progress_calls = []
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()):
            run_benchmark_suite(
                runtime, "test-model", num_runs=3,
                progress_callback=lambda i, n: progress_calls.append((i, n)),
            )
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    def test_should_use_default_prompt_when_prompt_type_is_default(self):
        runtime = self._make_runtime()
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()) as mock_bench:
            run_benchmark_suite(runtime, "test-model", prompt_type="default", num_runs=1)
        _, kwargs = mock_bench.call_args
        assert BENCH_PROMPTS["default"] in mock_bench.call_args[0]

    def test_should_use_fallback_prompt_when_prompt_type_is_unknown(self):
        runtime = self._make_runtime()
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()) as mock_bench:
            run_benchmark_suite(runtime, "test-model", prompt_type="nonexistent", num_runs=1)
        prompt_arg = mock_bench.call_args[0][3]
        assert prompt_arg == BENCH_PROMPTS["default"]

    def test_should_skip_progress_callback_when_not_provided(self):
        runtime = self._make_runtime()
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()):
            # should not raise
            results = run_benchmark_suite(runtime, "test-model", num_runs=2)
        assert len(results) == 2

    def test_should_pass_run_number_to_each_benchmark_call(self):
        runtime = self._make_runtime()
        with patch("loca_llama.benchmark.benchmark_openai_api", return_value=_make_success_result()) as mock_bench:
            run_benchmark_suite(runtime, "test-model", num_runs=3)
        run_numbers = [c[1].get("run_number") or c[0][-1] for c in mock_bench.call_args_list]
        assert run_numbers == [1, 2, 3]


# ── aggregate_results ──────────────────────────────────────────────────────────

class TestAggregateResults:
    def test_should_return_success_false_when_all_results_failed(self):
        results = [
            _make_fail_result("m", "r", 4096, "err", run=1),
            _make_fail_result("m", "r", 4096, "err", run=2),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["success"] is False

    def test_should_return_zero_runs_when_all_results_failed(self):
        results = [_make_fail_result("m", "r", 4096, "err")]
        agg = aggregate_results(results, skip_first=False)
        assert agg["runs"] == 0

    def test_should_compute_average_tokens_per_second(self):
        results = [
            _make_success_result(tokens_per_second=40.0, run_number=1),
            _make_success_result(tokens_per_second=60.0, run_number=2),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["avg_tok_per_sec"] == pytest.approx(50.0)

    def test_should_skip_first_successful_result_when_skip_first_is_true(self):
        results = [
            _make_success_result(tokens_per_second=10.0, run_number=1),  # warmup
            _make_success_result(tokens_per_second=60.0, run_number=2),
            _make_success_result(tokens_per_second=60.0, run_number=3),
        ]
        agg = aggregate_results(results, skip_first=True)
        assert agg["avg_tok_per_sec"] == pytest.approx(60.0)

    def test_should_not_skip_first_when_only_one_successful_result(self):
        results = [_make_success_result(tokens_per_second=42.0, run_number=1)]
        agg = aggregate_results(results, skip_first=True)
        assert agg["success"] is True
        assert agg["avg_tok_per_sec"] == pytest.approx(42.0)

    def test_should_report_correct_run_count(self):
        results = [
            _make_success_result(run_number=1),
            _make_success_result(run_number=2),
            _make_success_result(run_number=3),
        ]
        agg = aggregate_results(results, skip_first=True)
        assert agg["runs"] == 2  # first was skipped

    def test_should_compute_min_and_max_tokens_per_second(self):
        results = [
            _make_success_result(tokens_per_second=30.0, run_number=1),
            _make_success_result(tokens_per_second=50.0, run_number=2),
            _make_success_result(tokens_per_second=70.0, run_number=3),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["min_tok_per_sec"] == 30.0
        assert agg["max_tok_per_sec"] == 70.0

    def test_should_compute_average_time_to_first_token(self):
        results = [
            _make_success_result(prompt_eval_time_ms=200.0, run_number=1),
            _make_success_result(prompt_eval_time_ms=400.0, run_number=2),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["avg_ttft_ms"] == pytest.approx(300.0)

    def test_should_sum_total_generated_tokens(self):
        results = [
            _make_success_result(generated_tokens=100, run_number=1),
            _make_success_result(generated_tokens=200, run_number=2),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["total_tokens_generated"] == 300

    def test_should_ignore_failed_results_when_computing_aggregates(self):
        results = [
            _make_fail_result("m", "r", 4096, "err", run=1),
            _make_success_result(tokens_per_second=50.0, run_number=2),
        ]
        agg = aggregate_results(results, skip_first=False)
        assert agg["success"] is True
        assert agg["avg_tok_per_sec"] == pytest.approx(50.0)


# ── detect_llama_cpp_server ────────────────────────────────────────────────────

class TestDetectLlamaCppServer:
    def test_should_return_runtime_info_when_server_is_healthy_on_first_port(self):
        health_resp = _json_response({"status": "ok"})
        models_resp = _json_response({"data": [{"id": "llama-3"}]})
        with patch("urllib.request.urlopen", side_effect=[health_resp, models_resp]):
            result = detect_llama_cpp_server()
        assert result is not None
        assert result.name == "llama.cpp-server"
        assert result.models == ["llama-3"]

    def test_should_return_none_when_all_ports_fail(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = detect_llama_cpp_server()
        assert result is None

    def test_should_skip_port_when_status_is_not_ok(self):
        health_resp = _json_response({"status": "loading"})
        with patch("urllib.request.urlopen", side_effect=[health_resp, OSError("refused"), OSError("refused")]):
            result = detect_llama_cpp_server()
        assert result is None

    def test_should_use_placeholder_model_when_models_endpoint_fails(self):
        health_resp = _json_response({"status": "ok"})
        with patch("urllib.request.urlopen", side_effect=[health_resp, OSError("models failed")]):
            result = detect_llama_cpp_server()
        assert result is not None
        assert result.models == ["(loaded model)"]

    def test_should_include_server_url_in_result(self):
        health_resp = _json_response({"status": "ok"})
        models_resp = _json_response({"data": []})
        with patch("urllib.request.urlopen", side_effect=[health_resp, models_resp]):
            result = detect_llama_cpp_server()
        assert result.url == "http://127.0.0.1:8082"


# ── detect_lm_studio ──────────────────────────────────────────────────────────

class TestDetectLmStudio:
    def test_should_return_runtime_info_when_models_are_loaded(self):
        data = {"data": [{"id": "llama-3"}, {"id": "gemma-2"}]}
        with patch("urllib.request.urlopen", return_value=_json_response(data)):
            result = detect_lm_studio()
        assert result is not None
        assert result.name == "lm-studio"
        assert result.models == ["llama-3", "gemma-2"]

    def test_should_return_none_when_models_list_is_empty(self):
        data = {"data": []}
        with patch("urllib.request.urlopen", return_value=_json_response(data)):
            result = detect_lm_studio()
        assert result is None

    def test_should_return_none_when_all_ports_fail(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = detect_lm_studio()
        assert result is None

    def test_should_include_server_url_in_result(self):
        data = {"data": [{"id": "model-1"}]}
        with patch("urllib.request.urlopen", return_value=_json_response(data)):
            result = detect_lm_studio()
        assert result.url == "http://127.0.0.1:1234"


# ── detect_all_runtimes ────────────────────────────────────────────────────────

class TestDetectAllRuntimes:
    def test_should_return_empty_list_when_no_runtimes_detected(self):
        with patch("loca_llama.benchmark.detect_omlx", return_value=None), \
             patch("loca_llama.benchmark.detect_litellm", return_value=None), \
             patch("loca_llama.benchmark.detect_lm_studio", return_value=None), \
             patch("loca_llama.benchmark.detect_llama_cpp_server", return_value=None):
            result = detect_all_runtimes()
        assert result == []

    def test_should_include_lm_studio_when_detected(self):
        lms = RuntimeInfo(name="lm-studio", url="http://127.0.0.1:1234", models=["m"])
        with patch("loca_llama.benchmark.detect_omlx", return_value=None), \
             patch("loca_llama.benchmark.detect_litellm", return_value=None), \
             patch("loca_llama.benchmark.detect_lm_studio", return_value=lms), \
             patch("loca_llama.benchmark.detect_llama_cpp_server", return_value=None):
            result = detect_all_runtimes()
        assert lms in result

    def test_should_include_llama_cpp_when_detected(self):
        lcp = RuntimeInfo(name="llama.cpp-server", url="http://127.0.0.1:8080", models=["m"])
        with patch("loca_llama.benchmark.detect_omlx", return_value=None), \
             patch("loca_llama.benchmark.detect_litellm", return_value=None), \
             patch("loca_llama.benchmark.detect_lm_studio", return_value=None), \
             patch("loca_llama.benchmark.detect_llama_cpp_server", return_value=lcp):
            result = detect_all_runtimes()
        assert lcp in result

    def test_should_return_both_runtimes_when_both_detected(self):
        lms = RuntimeInfo(name="lm-studio", url="http://127.0.0.1:1234", models=["m"])
        lcp = RuntimeInfo(name="llama.cpp-server", url="http://127.0.0.1:8080", models=["m"])
        with patch("loca_llama.benchmark.detect_omlx", return_value=None), \
             patch("loca_llama.benchmark.detect_litellm", return_value=None), \
             patch("loca_llama.benchmark.detect_lm_studio", return_value=lms), \
             patch("loca_llama.benchmark.detect_llama_cpp_server", return_value=lcp):
            result = detect_all_runtimes()
        assert len(result) == 2


# ── benchmark_openai_api_streaming ────────────────────────────────────────────


def _sse_response(events: list[str]) -> MagicMock:
    """Build a mock that simulates line-based SSE streaming.

    Each event string should be a complete SSE line (e.g. 'data: {...}').
    Lines are joined with newlines and converted to a BytesIO for iteration.
    """
    raw = "\n".join(events) + "\n"
    stream = BytesIO(raw.encode())
    resp = MagicMock()
    resp.read = stream.read
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.__iter__ = lambda s: iter(stream.readlines())
    return resp


class TestBenchmarkOpenaiApiStreaming:
    def test_should_yield_tokens_from_sse_stream(self):
        """Successful stream yields (token_text, elapsed_ms) tuples."""
        events = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)):
            tokens = list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "Say hi", max_tokens=50,
            ))
        assert len(tokens) == 2
        assert tokens[0][0] == "Hello"
        assert tokens[1][0] == " world"
        # Both should have positive elapsed ms
        assert all(t[1] > 0 for t in tokens)

    def test_should_stop_on_done_signal(self):
        """Generator terminates cleanly on [DONE]."""
        events = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"B"}}]}',
        ]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)):
            tokens = list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "test",
            ))
        assert len(tokens) == 1
        assert tokens[0][0] == "A"

    def test_should_skip_empty_content_deltas(self):
        """Deltas without content key are skipped."""
        events = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)):
            tokens = list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "test",
            ))
        assert len(tokens) == 1
        assert tokens[0][0] == "ok"

    def test_should_skip_malformed_json_lines(self):
        """Malformed JSON data lines are silently skipped."""
        events = [
            "data: {invalid json}",
            'data: {"choices":[{"delta":{"content":"valid"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)):
            tokens = list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "test",
            ))
        assert len(tokens) == 1
        assert tokens[0][0] == "valid"

    def test_should_skip_sse_comment_lines(self):
        """SSE comment lines starting with ':' are ignored."""
        events = [
            ": this is a comment",
            'data: {"choices":[{"delta":{"content":"tok"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)):
            tokens = list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "test",
            ))
        assert len(tokens) == 1
        assert tokens[0][0] == "tok"

    def test_should_raise_on_connection_error(self):
        """Connection errors propagate to caller."""
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                list(benchmark_openai_api_streaming(
                    "http://localhost:1234", "test-model", "test",
                ))

    def test_should_send_stream_true_in_request(self):
        """Request payload includes stream: true."""
        events = ["data: [DONE]"]
        with patch("urllib.request.urlopen", return_value=_sse_response(events)) as mock_open:
            list(benchmark_openai_api_streaming(
                "http://localhost:1234", "test-model", "test",
            ))
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        assert body["stream"] is True
        assert body["model"] == "test-model"


# ── run_benchmark_sweep ──────────────────────────────────────────────────────

class TestRunBenchmarkSweep:
    """Tests for run_benchmark_sweep()."""

    _RUNTIME = RuntimeInfo(
        name="lm-studio", url="http://localhost:1234",
        models=["model-a", "model-b"], version="0.3",
    )

    def _mock_suite(self, runtime, model_id, *args, **kwargs):
        """Return two fake results tagged with the model_id."""
        return [
            _make_success_result(model_name=model_id, run_number=1),
            _make_success_result(model_name=model_id, run_number=2),
        ]

    @patch("loca_llama.benchmark.aggregate_results", return_value={"runs": 1, "avg_tok_per_sec": 50.0})
    @patch("loca_llama.benchmark.run_benchmark_suite")
    def test_should_return_one_combo_per_model(self, mock_suite, mock_agg):
        """Returns one combo result dict per model in the list."""
        mock_suite.side_effect = self._mock_suite
        results = run_benchmark_sweep(self._RUNTIME, ["model-a", "model-b"])
        assert len(results) == 2
        assert results[0]["model_id"] == "model-a"
        assert results[1]["model_id"] == "model-b"

    @patch("loca_llama.benchmark.aggregate_results", return_value={"runs": 1})
    @patch("loca_llama.benchmark.run_benchmark_suite")
    def test_should_call_combo_callback_for_each_model(self, mock_suite, mock_agg):
        """combo_callback is invoked once per model with 1-based index."""
        mock_suite.side_effect = self._mock_suite
        cb = MagicMock()
        run_benchmark_sweep(self._RUNTIME, ["model-a", "model-b"], combo_callback=cb)
        assert cb.call_count == 2
        cb.assert_any_call(1, 2)
        cb.assert_any_call(2, 2)

    @patch("loca_llama.benchmark.aggregate_results", return_value={"runs": 1})
    @patch("loca_llama.benchmark.run_benchmark_suite")
    def test_should_forward_run_callback_to_suite(self, mock_suite, mock_agg):
        """run_callback is passed through to run_benchmark_suite as progress_callback."""
        mock_suite.side_effect = self._mock_suite
        run_cb = MagicMock()
        run_benchmark_sweep(self._RUNTIME, ["model-a"], run_callback=run_cb)
        # run_callback is the 7th positional arg to run_benchmark_suite
        suite_call = mock_suite.call_args
        assert suite_call[0][6] is run_cb  # progress_callback arg

    @patch("loca_llama.benchmark.aggregate_results", return_value={"runs": 1})
    @patch("loca_llama.benchmark.run_benchmark_suite")
    def test_should_forward_custom_prompt(self, mock_suite, mock_agg):
        """custom_prompt is passed through to run_benchmark_suite."""
        mock_suite.side_effect = self._mock_suite
        run_benchmark_sweep(
            self._RUNTIME, ["model-a"],
            custom_prompt="My custom prompt",
        )
        suite_call = mock_suite.call_args
        assert suite_call[0][7] == "My custom prompt"  # custom_prompt arg

    @patch("loca_llama.benchmark.aggregate_results")
    @patch("loca_llama.benchmark.run_benchmark_suite")
    def test_should_aggregate_each_combo(self, mock_suite, mock_agg):
        """aggregate_results is called once per model combo."""
        mock_suite.side_effect = self._mock_suite
        mock_agg.return_value = {"runs": 1}
        run_benchmark_sweep(self._RUNTIME, ["model-a", "model-b"])
        assert mock_agg.call_count == 2
