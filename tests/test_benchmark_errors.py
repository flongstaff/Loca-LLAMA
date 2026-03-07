"""Tests for format_benchmark_error() in benchmark.py."""

from __future__ import annotations

import json
import socket
import urllib.error

from loca_llama.benchmark import format_benchmark_error


RUNTIME = "lm-studio"
URL = "http://localhost:1234"


class TestFormatBenchmarkError:
    """Tests for format_benchmark_error()."""

    def test_connection_refused(self):
        """Should return connection message for ConnectionRefusedError."""
        exc = ConnectionRefusedError("Connection refused")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "Cannot connect" in msg
        assert RUNTIME in msg
        assert URL in msg

    def test_connection_reset(self):
        """Should return connection message for ConnectionResetError."""
        exc = ConnectionResetError("Connection reset by peer")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "Cannot connect" in msg
        assert RUNTIME in msg

    def test_http_error_400(self):
        """Should return rejected request message for HTTP 400."""
        exc = urllib.error.HTTPError(
            url=URL, code=400, msg="Bad Request",
            hdrs=None, fp=None,
        )
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "rejected" in msg.lower()
        assert "400" in msg

    def test_http_error_500(self):
        """Should return internal error message for HTTP 500."""
        exc = urllib.error.HTTPError(
            url=URL, code=500, msg="Internal Server Error",
            hdrs=None, fp=None,
        )
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "internal error" in msg.lower()
        assert "500" in msg

    def test_url_error_connection_refused(self):
        """Should return connection message for URLError wrapping ConnectionRefusedError."""
        exc = urllib.error.URLError(ConnectionRefusedError("Connection refused"))
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "Cannot connect" in msg
        assert RUNTIME in msg

    def test_url_error_connection_reset(self):
        """Should return crash message for URLError with connection reset."""
        exc = urllib.error.URLError("Connection reset by peer")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "crashed" in msg.lower() or "reset" in msg.lower()

    def test_socket_timeout(self):
        """Should return timeout message for socket.timeout."""
        exc = socket.timeout("timed out")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "timed out" in msg.lower()
        assert RUNTIME in msg

    def test_timeout_error(self):
        """Should return timeout message for TimeoutError."""
        exc = TimeoutError("Request timed out")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "timed out" in msg.lower()

    def test_json_decode_error(self):
        """Should return invalid response message for JSONDecodeError."""
        exc = json.JSONDecodeError("Expecting value", "doc", 0)
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "Invalid response" in msg
        assert RUNTIME in msg

    def test_broken_pipe(self):
        """Should return connection message for BrokenPipeError."""
        exc = BrokenPipeError("Broken pipe")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "Cannot connect" in msg

    def test_generic_exception(self):
        """Should return generic message with exception type for unknown errors."""
        exc = ValueError("something went wrong")
        msg = format_benchmark_error(exc, RUNTIME, URL)
        assert "ValueError" in msg
