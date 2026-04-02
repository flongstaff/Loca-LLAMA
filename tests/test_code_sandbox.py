"""Tests for the code execution sandbox — security boundary."""

from __future__ import annotations

import pytest

from loca_llama.code_sandbox import (
    UnsafeCodeError,
    validate_code_ast,
    run_code_safe,
    run_code_safe_with_output,
    _BLOCKED_MODULES,
    _BLOCKED_CALLS,
    _BLOCKED_ATTRS,
)


# ── Blocked Imports ──────────────────────────────────────────────────────────


class TestBlockedImports:
    """Validate that all dangerous modules are rejected."""

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "shutil", "socket", "ctypes", "signal",
        "sys", "builtins", "io", "importlib", "pathlib", "pickle",
    ])
    def test_rejects_blocked_module(self, module: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            validate_code_ast(f"import {module}")

    @pytest.mark.parametrize("module", [
        "os.path", "subprocess.run", "sys.modules", "io.BytesIO",
    ])
    def test_rejects_dotted_blocked_module(self, module: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            validate_code_ast(f"import {module}")

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "sys", "builtins", "io",
    ])
    def test_rejects_from_import(self, module: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            validate_code_ast(f"from {module} import anything")

    def test_allows_safe_imports(self) -> None:
        validate_code_ast("import math")
        validate_code_ast("import json")
        validate_code_ast("import re")
        validate_code_ast("from collections import defaultdict")
        validate_code_ast("import dataclasses")


# ── Blocked Calls ────────────────────────────────────────────────────────────


class TestBlockedCalls:
    """Validate that dangerous built-in calls are rejected."""

    @pytest.mark.parametrize("call", [
        "eval('1+1')", "exec('pass')", "compile('x', '', 'exec')",
        "__import__('os')", "breakpoint()", "open('file.txt')",
        "type('X', (), {})", "globals()", "locals()",
    ])
    def test_rejects_blocked_call(self, call: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked call"):
            validate_code_ast(call)

    def test_allows_safe_calls(self) -> None:
        validate_code_ast("print('hello')")
        validate_code_ast("len([1, 2, 3])")
        validate_code_ast("sorted([3, 1, 2])")
        validate_code_ast("int('42')")
        validate_code_ast("str(123)")
        validate_code_ast("list(range(10))")


# ── Blocked Attributes ───────────────────────────────────────────────────────


class TestBlockedAttrs:
    """Validate that dangerous attribute access is rejected."""

    @pytest.mark.parametrize("attr", [
        "__subclasses__", "__bases__", "__globals__", "__builtins__",
        "__code__", "__import__", "__mro__", "__class__", "__dict__",
    ])
    def test_rejects_blocked_attr_access(self, attr: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked attribute"):
            validate_code_ast(f"x.{attr}")

    @pytest.mark.parametrize("attr", [
        "__subclasses__", "__bases__", "__mro__", "__class__",
    ])
    def test_rejects_blocked_attr_call(self, attr: str) -> None:
        with pytest.raises(UnsafeCodeError, match="Blocked attribute"):
            validate_code_ast(f"x.{attr}()")

    def test_allows_safe_attr_access(self) -> None:
        validate_code_ast("x.append(1)")
        validate_code_ast("'hello'.upper()")
        validate_code_ast("[1,2,3].sort()")


# ── Syntax Errors ────────────────────────────────────────────────────────────


class TestSyntaxValidation:
    """Validate that syntax errors are caught."""

    def test_rejects_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            validate_code_ast("def (broken")

    def test_accepts_valid_code(self) -> None:
        validate_code_ast("x = 1 + 2\nprint(x)")


# ── Execution ────────────────────────────────────────────────────────────────


class TestRunCodeSafe:
    """Test sandboxed code execution."""

    def test_safe_code_returns_true(self) -> None:
        assert run_code_safe("print('hello')") is True

    def test_failing_code_returns_false(self) -> None:
        assert run_code_safe("raise ValueError('oops')") is False

    def test_blocked_code_raises(self) -> None:
        with pytest.raises(UnsafeCodeError):
            run_code_safe("import os; os.system('echo hi')")

    def test_timeout_returns_false(self) -> None:
        assert run_code_safe("while True: pass", timeout=1) is False


class TestRunCodeSafeWithOutput:
    """Test sandboxed execution with output capture."""

    def test_captures_stdout(self) -> None:
        ok, stdout, stderr = run_code_safe_with_output("print('hello')")
        assert ok is True
        assert "hello" in stdout

    def test_returns_stderr_on_failure(self) -> None:
        ok, stdout, stderr = run_code_safe_with_output("raise ValueError('oops')")
        assert ok is False
        assert "ValueError" in stderr

    def test_blocked_code_returns_error(self) -> None:
        ok, stdout, stderr = run_code_safe_with_output("import os")
        assert ok is False
        assert "Blocked import" in stderr

    def test_timeout_returns_error(self) -> None:
        ok, stdout, stderr = run_code_safe_with_output("while True: pass", timeout=1)
        assert ok is False
        assert "Timeout" in stderr

    def test_isolated_mode_blocks_site_packages(self) -> None:
        # -I flag should prevent importing from site-packages
        ok, stdout, _ = run_code_safe_with_output(
            "import math; print(math.pi)"
        )
        # math is stdlib, should still work
        assert ok is True
        assert "3.14" in stdout


# ── Coverage Assertions ──────────────────────────────────────────────────────


class TestBlocklistCompleteness:
    """Verify the blocklists include critical entries."""

    def test_sys_is_blocked(self) -> None:
        assert "sys" in _BLOCKED_MODULES

    def test_builtins_is_blocked(self) -> None:
        assert "builtins" in _BLOCKED_MODULES

    def test_open_is_blocked(self) -> None:
        assert "open" in _BLOCKED_CALLS

    def test_type_is_blocked(self) -> None:
        assert "type" in _BLOCKED_CALLS

    def test_mro_is_blocked(self) -> None:
        assert "__mro__" in _BLOCKED_ATTRS

    def test_class_is_blocked(self) -> None:
        assert "__class__" in _BLOCKED_ATTRS

    def test_dict_is_blocked(self) -> None:
        assert "__dict__" in _BLOCKED_ATTRS
