"""Transparent HTTP proxy for monitoring OpenCode sessions with local LLMs.

Sits between an LLM client (e.g. opencode on port 1240) and a local LLM server
(e.g. LM Studio on port 1234). Logs every request with timing metrics and prints
live stats. On shutdown (Ctrl+C), prints a summary and saves results.

Usage (via CLI):
    loca-llama monitor                              # Proxy :1240 -> :1234
    loca-llama monitor --listen 1240 --target 1234  # Explicit ports

Then point your client at the proxy:
    OPENAI_BASE_URL=http://127.0.0.1:1240/v1 opencode "fix the bug"
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


_request_logs: list[dict[str, Any]] = []
_lock = threading.Lock()
_target_url: str = "http://127.0.0.1:1234"
_start_time: float = 0.0


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to the target LLM server, logging metrics."""

    def do_GET(self) -> None:
        self._proxy_request("GET")

    def do_POST(self) -> None:
        self._proxy_request("POST")

    def do_OPTIONS(self) -> None:
        self._proxy_request("OPTIONS")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default access logs
        pass

    def _proxy_request(self, method: str) -> None:
        target = _target_url + self.path

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Parse request to detect model and streaming
        is_chat = "/v1/chat/completions" in self.path
        is_streaming = False
        model_name = ""
        if body and is_chat:
            try:
                req_data = json.loads(body)
                model_name = req_data.get("model", "")
                is_streaming = req_data.get("stream", False)
            except json.JSONDecodeError:
                pass

        # Forward headers (skip hop-by-hop)
        skip_headers = {"host", "connection", "transfer-encoding"}
        headers: dict[str, str] = {}
        for key, val in self.headers.items():
            if key.lower() not in skip_headers:
                headers[key] = val

        start = time.monotonic()
        ttft = 0.0
        tokens_generated = 0
        prompt_tokens = 0
        completion_tokens = 0

        try:
            req = urllib.request.Request(target, data=body, headers=headers, method=method)
            resp = urllib.request.urlopen(req, timeout=300)

            # Forward status and headers
            self.send_response(resp.status)
            for key, val in resp.headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.end_headers()

            if is_chat and is_streaming:
                # Stream SSE, counting tokens
                for raw_line in resp:
                    self.wfile.write(raw_line)
                    self.wfile.flush()

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                        choices = obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                if ttft == 0.0:
                                    ttft = (time.monotonic() - start) * 1000
                                tokens_generated += 1
                        # Usage in final chunk
                        usage = obj.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            else:
                # Non-streaming: forward entire response
                response_body = resp.read()
                self.wfile.write(response_body)

                if is_chat:
                    try:
                        resp_data = json.loads(response_body)
                        usage = resp_data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        tokens_generated = completion_tokens
                    except json.JSONDecodeError:
                        pass

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            err_body = e.read()
            self.wfile.write(err_body)
            return
        except (urllib.error.URLError, OSError, TimeoutError, ConnectionError) as e:
            logger.debug("proxy request to %s failed: %s", target, e)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        total_ms = (time.monotonic() - start) * 1000
        if not completion_tokens:
            completion_tokens = tokens_generated
        tps = tokens_generated / (total_ms / 1000) if total_ms > 0 and tokens_generated > 0 else 0

        # Log the request
        if is_chat:
            log_entry = {
                "timestamp": time.time(),
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_ms": ttft,
                "tps": tps,
                "total_ms": total_ms,
                "streaming": is_streaming,
            }
            with _lock:
                _request_logs.append(log_entry)

            # Print live update
            _print_live_status(log_entry)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")


def _print_live_status(latest: dict[str, Any]) -> None:
    """Print a live status line."""
    with _lock:
        total_reqs = len(_request_logs)
        total_tokens = sum(r["completion_tokens"] for r in _request_logs)
        all_tps = [r["tps"] for r in _request_logs if r["tps"] > 0]
        avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0
        all_ttft = [r["ttft_ms"] for r in _request_logs if r["ttft_ms"] > 0]
        avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0

    elapsed = time.monotonic() - _start_time
    model = latest.get("model", "?")[:40]

    # Clear line and print status
    sys.stderr.write(
        f"\r  Requests: {total_reqs} | Tokens: {total_tokens:,} | "
        f"Avg: {avg_tps:.1f} tok/s | TTFT: {avg_ttft:,.0f}ms | "
        f"Model: {model}   "
    )
    sys.stderr.flush()


def print_session_summary() -> dict[str, Any]:
    """Print final session summary and return stats dict."""
    with _lock:
        logs = list(_request_logs)

    if not logs:
        print("\n  No requests logged.")
        return {}

    total_reqs = len(logs)
    total_tokens = sum(r["completion_tokens"] for r in logs)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in logs)
    all_tps = [r["tps"] for r in logs if r["tps"] > 0]
    avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0
    all_ttft = [r["ttft_ms"] for r in logs if r["ttft_ms"] > 0]
    avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0
    total_time = sum(r["total_ms"] for r in logs)
    session_dur = time.monotonic() - _start_time
    models_used = list(set(r["model"] for r in logs if r["model"]))

    print(f"\n\n{'='*60}")
    print("  OpenCode Monitor — Session Summary")
    print(f"{'='*60}")
    print(f"  Requests:        {total_reqs}")
    print(f"  Prompt tokens:   {total_prompt_tokens:,}")
    print(f"  Generated:       {total_tokens:,} tokens")
    print(f"  Avg speed:       {avg_tps:.1f} tok/s")
    print(f"  Avg TTFT:        {avg_ttft:,.0f} ms")
    print(f"  Total LLM time:  {total_time / 1000:.1f}s")
    print(f"  Session:         {session_dur:.0f}s")
    if models_used:
        print(f"  Model(s):        {', '.join(models_used)}")
    print(f"{'='*60}\n")

    return {
        "total_requests": total_reqs,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "avg_tps": avg_tps,
        "avg_ttft_ms": avg_ttft,
        "total_llm_time_ms": total_time,
        "session_duration_s": session_dur,
        "models": models_used,
        "request_logs": logs,
    }


def save_monitor_results(stats: dict[str, Any]) -> None:
    """Save monitor session results."""
    if not stats or not stats.get("total_requests"):
        return

    from .benchmark_results import BenchmarkRecord, save_result

    model = stats["models"][0] if stats.get("models") else "unknown"
    record = BenchmarkRecord(
        type="monitor",
        model=model,
        runtime="proxy-monitor",
        tokens_per_second=stats.get("avg_tps", 0),
        ttft_ms=stats.get("avg_ttft_ms", 0),
        generated_tokens=stats.get("total_tokens", 0),
        prompt_tokens=stats.get("total_prompt_tokens", 0),
        monitor_stats={
            k: v for k, v in stats.items()
            if k != "request_logs"  # Don't save raw logs — too large
        },
    )

    path = save_result(record)
    print(f"  Results saved to {path}")


def run_proxy(
    listen_port: int = 1240,
    target_port: int = 1234,
    target_host: str = "127.0.0.1",
) -> None:
    """Start the monitoring proxy server."""
    global _target_url, _start_time

    _target_url = f"http://{target_host}:{target_port}"
    _start_time = time.monotonic()

    server = HTTPServer(("127.0.0.1", listen_port), ProxyHandler)

    print(f"  OpenCode Monitor")
    print(f"  {'─'*40}")
    print(f"  Proxy:   http://127.0.0.1:{listen_port}")
    print(f"  Target:  {_target_url}")
    print(f"  {'─'*40}")
    print(f"  Set OPENAI_BASE_URL=http://127.0.0.1:{listen_port}/v1")
    print(f"  Press Ctrl+C to stop and see summary.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        stats = print_session_summary()
        save_monitor_results(stats)
