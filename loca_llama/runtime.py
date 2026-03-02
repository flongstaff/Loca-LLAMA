"""Runtime connector: manage models on LM Studio and llama.cpp servers."""

import json
import subprocess
import shutil
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedModel:
    """A model currently loaded in a runtime."""

    model_id: str
    runtime: str
    context_length: int | None = None
    gpu_layers: int | None = None


class LMStudioConnector:
    """Connect to LM Studio's local API.

    LM Studio exposes an OpenAI-compatible API on port 1234.
    It also has management endpoints for loading/unloading models.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:1234"):
        self.base_url = base_url

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=3):
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List models currently loaded in LM Studio."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def get_model_info(self, model_id: str) -> dict | None:
        """Get info about a loaded model."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models/{urllib.parse.quote(model_id, safe='')}", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None

    def chat(
        self,
        model_id: str,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict:
        """Send a chat completion request."""
        payload = json.dumps({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    def complete(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict:
        """Send a text completion request."""
        payload = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())


class LlamaCppConnector:
    """Connect to llama.cpp server.

    llama-server exposes an OpenAI-compatible API plus additional endpoints
    for model management, health checks, and detailed performance metrics.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return data.get("status") == "ok"
        except Exception:
            return False

    def health(self) -> dict:
        """Get detailed server health."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {"status": "error"}

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def get_props(self) -> dict:
        """Get server properties (model info, context size, etc.)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/props", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}

    def get_slots(self) -> list[dict]:
        """Get slot info (active inference tasks)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/slots", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return []

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict:
        """Send a chat completion request."""
        models = self.list_models()
        model_id = models[0] if models else "default"

        payload = json.dumps({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    @staticmethod
    def start_server(
        model_path: str,
        port: int = 8080,
        context_length: int = 4096,
        n_gpu_layers: int = -1,
        threads: int | None = None,
    ) -> subprocess.Popen | None:
        """Start a llama.cpp server as a subprocess.

        Returns the Popen object, or None if the executable isn't found.
        """
        exe = shutil.which("llama-server") or shutil.which("server")
        if not exe:
            return None

        cmd = [
            exe,
            "-m", model_path,
            "--port", str(port),
            "-c", str(context_length),
            "-ngl", str(n_gpu_layers),
        ]
        if threads:
            cmd.extend(["-t", str(threads)])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        for _ in range(30):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        return proc
            except Exception:
                if proc.poll() is not None:
                    return None
                continue

        return proc


def detect_all_connectors() -> dict[str, object]:
    """Detect and return all available runtime connectors."""
    connectors = {}

    lms = LMStudioConnector()
    if lms.is_running():
        connectors["lm-studio"] = lms

    # Try multiple ports for llama.cpp
    for port in [8080, 8081, 8000]:
        lcp = LlamaCppConnector(f"http://127.0.0.1:{port}")
        if lcp.is_running():
            connectors[f"llama.cpp:{port}"] = lcp
            break

    return connectors
