"""Safe code execution sandbox for benchmark evaluation.

Validates Python code via AST analysis before executing in a subprocess,
rejecting dangerous imports, calls, and attribute access patterns.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from typing import Tuple

# Modules that could damage the host system or exfiltrate data
_BLOCKED_MODULES = frozenset({
    "os", "subprocess", "shutil", "socket", "ctypes", "signal",
    "multiprocessing", "threading", "http", "urllib", "requests",
    "pathlib", "tempfile", "glob", "fnmatch", "importlib",
    "webbrowser", "antigravity", "turtle", "tkinter",
    "pickle", "shelve", "marshal", "code", "codeop",
    "compileall", "py_compile",
})

# Built-in functions/names that enable code injection or system access
_BLOCKED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "breakpoint",
    "exit", "quit", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
})

# Attribute access patterns that bypass import restrictions
_BLOCKED_ATTRS = frozenset({
    "system", "popen", "exec", "spawn",
    "__subclasses__", "__bases__", "__globals__",
    "__builtins__", "__code__", "__import__",
})


class UnsafeCodeError(Exception):
    """Raised when code fails AST safety validation."""


def validate_code_ast(code: str) -> None:
    """Parse and validate Python code for safety before execution.

    Raises UnsafeCodeError if the code contains blocked imports,
    function calls, or attribute access patterns.
    Raises SyntaxError if the code cannot be parsed.
    """
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _check_module(node.module)

        # Block dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_CALLS:
                    raise UnsafeCodeError(
                        f"Blocked call: {node.func.id}()"
                    )
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _BLOCKED_ATTRS:
                    raise UnsafeCodeError(
                        f"Blocked attribute call: .{node.func.attr}()"
                    )

        # Block dangerous attribute access (even without call)
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                raise UnsafeCodeError(
                    f"Blocked attribute access: .{node.attr}"
                )


def _check_module(module_name: str) -> None:
    """Check if a module name (possibly dotted) is blocked."""
    top_level = module_name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise UnsafeCodeError(f"Blocked import: {module_name}")


def run_code_safe(code: str, timeout: int = 10) -> bool:
    """Validate and execute Python code in a subprocess sandbox.

    Returns True if the code exits with returncode 0, False otherwise.
    """
    validate_code_ast(code)

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def run_code_safe_with_output(
    code: str, timeout: int = 10
) -> Tuple[bool, str, str]:
    """Validate and execute code, returning (success, stdout, stderr).

    Returns (False, "", error_message) if validation fails.
    """
    try:
        validate_code_ast(code)
    except (UnsafeCodeError, SyntaxError) as e:
        return False, "", str(e)

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (
            result.returncode == 0,
            result.stdout,
            result.stderr,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout ({timeout}s)"
