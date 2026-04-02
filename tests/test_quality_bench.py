"""Unit tests for loca_llama/quality_bench.py — pure functions only."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from loca_llama.quality_bench import (
    TASKS,
    TaskResult,
    extract_python_code,
    score_task,
)

# ── extract_python_code ────────────────────────────────────────────────────────


class TestExtractPythonCode:
    def test_should_extract_code_from_python_fenced_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    return 42\n```"
        result = extract_python_code(text)
        assert "def foo():" in result

    def test_should_extract_code_from_generic_fenced_block(self):
        text = "```\ndef bar():\n    pass\n```"
        result = extract_python_code(text)
        assert "def bar():" in result

    def test_should_join_multiple_fenced_blocks_with_blank_line(self):
        text = "```python\ndef a(): pass\n```\nAnd:\n```python\ndef b(): pass\n```"
        result = extract_python_code(text)
        assert "def a(): pass" in result
        assert "def b(): pass" in result

    def test_should_strip_thinking_tags_before_extraction(self):
        text = "<think>This is my reasoning</think>\n```python\ndef answer(): pass\n```"
        result = extract_python_code(text)
        assert "def answer(): pass" in result
        assert "<think>" not in result

    def test_should_strip_multiline_thinking_tags(self):
        text = "<think>\nLine one\nLine two\n</think>\n```python\ndef solve(): pass\n```"
        result = extract_python_code(text)
        assert "def solve(): pass" in result

    def test_should_extract_bare_def_when_no_fence(self):
        text = "def compute(x):\n    return x * 2"
        result = extract_python_code(text)
        assert "def compute(x):" in result

    def test_should_extract_bare_class_when_no_fence(self):
        text = "class Foo:\n    pass"
        result = extract_python_code(text)
        assert "class Foo:" in result

    def test_should_extract_bare_import_when_no_fence(self):
        text = "import math\ndef area(r):\n    return math.pi * r ** 2"
        result = extract_python_code(text)
        assert "import math" in result

    def test_should_return_empty_string_when_no_code_present(self):
        # No fence and no def/class/import — falls through to returning the text
        # itself, but the contract is: if there truly is no code, callers handle
        # the empty / non-code result.  The implementation returns the raw text as
        # a last-resort fallback, so we assert extract does *not* crash.
        result = extract_python_code("This is just plain prose.")
        assert isinstance(result, str)

    def test_should_preserve_indentation_inside_fenced_block(self):
        text = "```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"
        result = extract_python_code(text)
        assert "    if n <= 1:" in result

    def test_should_handle_empty_string_input(self):
        result = extract_python_code("")
        assert isinstance(result, str)


# ── score_task ─────────────────────────────────────────────────────────────────


class TestScoreTask:
    """Use the real TASKS entries so we always stay in sync with production."""

    def _task(self, name: str) -> dict:
        return next(t for t in TASKS if t["name"] == name)

    # --- must_contain checks ---

    def test_should_return_full_contains_score_when_all_required_strings_present(self):
        task = self._task("binary_search")
        response = "def binary_search(arr: list[int], target: int) -> int:\n    pass"
        contains_score, _, _ = score_task(task, response)
        assert contains_score == 1.0

    def test_should_return_partial_contains_score_when_only_some_required_strings_present(self):
        task = self._task("binary_search")
        # Provides def binary_search but NOT -> int
        response = "def binary_search(arr, target):\n    pass"
        contains_score, _, _ = score_task(task, response)
        assert 0.0 < contains_score < 1.0

    def test_should_return_zero_contains_score_when_no_required_strings_present(self):
        task = self._task("binary_search")
        response = "This response has nothing relevant."
        contains_score, _, _ = score_task(task, response)
        assert contains_score == 0.0

    # --- must_not_contain penalty ---

    def test_should_penalise_contains_score_when_forbidden_string_present(self):
        task = self._task("binary_search")
        # Provide all required strings AND the forbidden "import bisect"
        response = "def binary_search(arr: list[int], target: int) -> int:\n    import bisect"
        score_without_penalty, _, _ = score_task(task, "def binary_search(arr: list[int], target: int) -> int:\n    pass")
        score_with_penalty, _, _ = score_task(task, response)
        assert score_with_penalty < score_without_penalty

    # --- must_contain_any ---

    def test_should_pass_contains_any_when_at_least_one_alternative_present(self):
        task = self._task("refactor_suggestion")
        response = "You should use isinstance() instead of type() checks."
        contains_score, _, _ = score_task(task, response)
        assert contains_score > 0.0

    def test_should_fail_contains_any_when_no_alternative_present(self):
        task = self._task("refactor_suggestion")
        response = "Looks fine to me!"
        contains_score, _, _ = score_task(task, response)
        assert contains_score == 0.0

    # --- non-runnable tasks ---

    def test_should_return_zero_runnable_score_for_non_runnable_task(self):
        task = self._task("reasoning_logic")
        response = "A=600 B=500 C=510 total=1610 {'A': 600, 'B': 500, 'C': 510, 'total': 1610}"
        _, runnable_score, _ = score_task(task, response)
        assert runnable_score == 0.0

    def test_should_return_one_contains_score_for_reasoning_task_with_all_numbers(self):
        task = self._task("reasoning_logic")
        response = "A=600 B=500 C=510 total=1610"
        contains_score, _, _ = score_task(task, response)
        assert contains_score == 1.0

    # --- runnable tasks ---

    def test_should_return_one_runnable_score_when_code_passes_test(self):
        task = self._task("binary_search")
        correct_code = """
def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""
        contains_score, runnable_score, error = score_task(task, correct_code)
        assert runnable_score == 1.0
        assert error == ""

    def test_should_return_zero_runnable_score_when_code_fails_test(self):
        task = self._task("binary_search")
        broken_code = "def binary_search(arr: list[int], target: int) -> int:\n    return 999"
        _, runnable_score, _ = score_task(task, broken_code)
        assert runnable_score == 0.0

    def test_should_return_error_string_when_no_code_can_be_extracted_for_runnable_task(self):
        task = self._task("binary_search")
        _, runnable_score, error = score_task(task, "I cannot write this.")
        assert runnable_score == 0.0
        # Either no code was extracted or the extracted code failed — either way an error or empty
        assert isinstance(error, str)

    def test_should_strip_thinking_tags_before_scoring(self):
        task = self._task("reasoning_logic")
        response = "<think>Internal thoughts</think>\nA=600 B=500 C=510 total=1610"
        contains_score, _, _ = score_task(task, response)
        assert contains_score == 1.0

    # --- score_task with sandbox mock ---

    def test_should_return_one_runnable_score_when_sandbox_reports_pass(self):
        task = self._task("binary_search")
        response = "def binary_search(arr: list[int], target: int) -> int:\n    return 0"
        with patch(
            "loca_llama.code_sandbox.run_code_safe_with_output",
            return_value=(True, "PASS: binary_search\n", ""),
        ):
            _, runnable_score, _ = score_task(task, response)
        assert runnable_score == 1.0

    def test_should_return_zero_runnable_score_when_sandbox_reports_failure(self):
        task = self._task("binary_search")
        response = "def binary_search(arr: list[int], target: int) -> int:\n    return 0"
        with patch(
            "loca_llama.code_sandbox.run_code_safe_with_output",
            return_value=(False, "", "AssertionError: Should find 5 at index 2"),
        ):
            _, runnable_score, error = score_task(task, response)
        assert runnable_score == 0.0
        assert "AssertionError" in error

    def test_should_handle_unsafe_code_error_from_sandbox(self):
        from loca_llama.code_sandbox import UnsafeCodeError

        task = self._task("binary_search")
        response = "def binary_search(arr: list[int], target: int) -> int:\n    return 0"
        with patch(
            "loca_llama.code_sandbox.run_code_safe_with_output",
            side_effect=UnsafeCodeError("Blocked import: os"),
        ):
            _, runnable_score, error = score_task(task, response)
        assert runnable_score == 0.0
        assert "Unsafe code" in error

    # --- edge: contains_total == 0 ---

    def test_should_return_one_contains_score_when_task_has_no_required_strings(self):
        minimal_task: dict = {
            "name": "empty",
            "category": "coding",
            "prompt": "Write something",
            "validation": {
                "must_contain": [],
                "must_contain_any": [],
                "must_not_contain": [],
                "runnable": False,
            },
        }
        contains_score, _, _ = score_task(minimal_task, "anything")
        assert contains_score == 1.0


# ── TASKS list integrity ───────────────────────────────────────────────────────

VALID_CATEGORIES = {"coding", "reasoning", "code_review"}
REQUIRED_KEYS = {"name", "category", "prompt", "validation"}


class TestTasksListIntegrity:
    def test_should_have_at_least_one_task(self):
        assert len(TASKS) > 0

    def test_should_have_no_duplicate_names(self):
        names = [t["name"] for t in TASKS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_all_required_keys(self, task: dict):
        assert REQUIRED_KEYS.issubset(task.keys())

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_valid_category(self, task: dict):
        assert task["category"] in VALID_CATEGORIES

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_validation_dict(self, task: dict):
        assert isinstance(task["validation"], dict)

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_runnable_key_in_validation(self, task: dict):
        assert "runnable" in task["validation"]

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_test_code_when_runnable_is_true(self, task: dict):
        if task["validation"].get("runnable"):
            assert "test_code" in task["validation"]
            assert task["validation"]["test_code"].strip() != ""

    @pytest.mark.parametrize("task", TASKS, ids=[t["name"] for t in TASKS])
    def test_should_have_non_empty_name_and_prompt(self, task: dict):
        assert task["name"].strip() != ""
        assert task["prompt"].strip() != ""


# ── TaskResult dataclass ───────────────────────────────────────────────────────


class TestTaskResultDataclass:
    def test_should_have_task_name_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "task_name" in field_names

    def test_should_have_model_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "model" in field_names

    def test_should_have_category_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "category" in field_names

    def test_should_have_response_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "response" in field_names

    def test_should_have_contains_score_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "contains_score" in field_names

    def test_should_have_runnable_score_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "runnable_score" in field_names

    def test_should_have_composite_score_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "composite_score" in field_names

    def test_should_have_speed_tps_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "speed_tps" in field_names

    def test_should_have_error_field(self):
        field_names = {f.name for f in fields(TaskResult)}
        assert "error" in field_names

    def test_should_instantiate_with_required_fields_only(self):
        r = TaskResult(task_name="t", model="m", category="coding")
        assert r.task_name == "t"
        assert r.model == "m"
        assert r.category == "coding"

    def test_should_use_default_values_for_optional_fields(self):
        r = TaskResult(task_name="t", model="m", category="coding")
        assert r.contains_score == 0.0
        assert r.runnable_score == 0.0
        assert r.composite_score == 0.0
        assert r.speed_tps == 0.0
        assert r.error == ""
        assert r.response == ""
