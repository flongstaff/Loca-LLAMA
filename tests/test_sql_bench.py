"""Tests for the SQL generation benchmark module."""

import sqlite3

import pytest

from loca_llama.sql_bench import (
    QUESTIONS,
    SQLTaskResult,
    compare_results,
    create_benchmark_db,
    extract_sql,
    execute_sql_safe,
    print_sql_summary,
    results_to_record,
    SCHEMA_DDL,
    _generate_seed_sql,
)


# ── Database Creation ───────────────────────────────────────────────────────


class TestCreateBenchmarkDb:
    """Tests for create_benchmark_db()."""

    def test_creates_connection(self) -> None:
        conn = create_benchmark_db()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_has_all_tables(self) -> None:
        conn = create_benchmark_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "customers" in tables
        assert "products" in tables
        assert "orders" in tables
        assert "order_items" in tables
        conn.close()

    def test_has_customer_data(self) -> None:
        conn = create_benchmark_db()
        cursor = conn.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        assert count == 100
        conn.close()

    def test_has_product_data(self) -> None:
        conn = create_benchmark_db()
        cursor = conn.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        assert count == 50
        conn.close()

    def test_has_order_data(self) -> None:
        conn = create_benchmark_db()
        cursor = conn.execute("SELECT COUNT(*) FROM orders")
        count = cursor.fetchone()[0]
        assert count == 500
        conn.close()

    def test_has_order_items(self) -> None:
        conn = create_benchmark_db()
        cursor = conn.execute("SELECT COUNT(*) FROM order_items")
        count = cursor.fetchone()[0]
        assert count > 0  # At least some items
        conn.close()

    def test_deterministic_seed(self) -> None:
        """Seed data should be identical across runs."""
        conn1 = create_benchmark_db()
        conn2 = create_benchmark_db()
        c1 = conn1.execute("SELECT name FROM customers ORDER BY id LIMIT 5").fetchall()
        c2 = conn2.execute("SELECT name FROM customers ORDER BY id LIMIT 5").fetchall()
        assert c1 == c2
        conn1.close()
        conn2.close()


# ── SQL Extraction ──────────────────────────────────────────────────────────


class TestExtractSql:
    """Tests for extract_sql()."""

    def test_extracts_from_sql_fence(self) -> None:
        text = "Here is the query:\n```sql\nSELECT * FROM customers;\n```"
        assert extract_sql(text) == "SELECT * FROM customers;"

    def test_extracts_from_generic_fence(self) -> None:
        text = "```\nSELECT COUNT(*) FROM orders;\n```"
        assert extract_sql(text) == "SELECT COUNT(*) FROM orders;"

    def test_extracts_raw_select(self) -> None:
        text = "The answer is SELECT name FROM products WHERE price > 100;"
        result = extract_sql(text)
        assert "SELECT name FROM products WHERE price > 100" in result

    def test_strips_thinking_tags(self) -> None:
        text = "<think>Let me think...</think>\n```sql\nSELECT 1;\n```"
        assert extract_sql(text) == "SELECT 1;"

    def test_takes_last_sql_block(self) -> None:
        text = "```sql\nSELECT 1;\n```\nActually:\n```sql\nSELECT 2;\n```"
        assert extract_sql(text) == "SELECT 2;"

    def test_handles_empty_input(self) -> None:
        assert extract_sql("") == ""

    def test_handles_no_sql(self) -> None:
        text = "I cannot generate SQL for this question."
        result = extract_sql(text)
        assert isinstance(result, str)


# ── Safe Execution ──────────────────────────────────────────────────────────


class TestExecuteSqlSafe:
    """Tests for execute_sql_safe()."""

    def test_executes_select(self) -> None:
        conn = create_benchmark_db()
        cols, rows = execute_sql_safe(conn, "SELECT COUNT(*) FROM customers;")
        assert len(rows) == 1
        assert rows[0][0] == 100
        conn.close()

    def test_blocks_insert(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_sql_safe(conn, "INSERT INTO customers VALUES (999, 'x', 'x', 'x', 'x', 'x', '2024-01-01');")
        conn.close()

    def test_blocks_drop(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_sql_safe(conn, "DROP TABLE customers;")
        conn.close()

    def test_blocks_delete(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_sql_safe(conn, "DELETE FROM customers;")
        conn.close()

    def test_returns_columns(self) -> None:
        conn = create_benchmark_db()
        cols, rows = execute_sql_safe(conn, "SELECT name, price FROM products LIMIT 1;")
        assert cols == ["name", "price"]
        conn.close()

    def test_raises_on_bad_sql(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(sqlite3.Error):
            execute_sql_safe(conn, "SELECT * FROM nonexistent_table;")
        conn.close()

    def test_blocks_load_extension(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="load_extension"):
            execute_sql_safe(conn, "SELECT load_extension('/tmp/evil.so');")
        conn.close()

    def test_blocks_multiple_statements(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="Multiple SQL"):
            execute_sql_safe(conn, "SELECT 1; DROP TABLE customers;")
        conn.close()

    def test_blocks_long_sql(self) -> None:
        conn = create_benchmark_db()
        long_sql = "SELECT " + ", ".join(["1"] * 2000)
        with pytest.raises(ValueError, match="SQL too long"):
            execute_sql_safe(conn, long_sql)
        conn.close()

    def test_allows_with_cte(self) -> None:
        conn = create_benchmark_db()
        cols, rows = execute_sql_safe(
            conn, "WITH t AS (SELECT 1 AS x) SELECT x FROM t;"
        )
        assert rows[0][0] == 1
        conn.close()

    def test_rejects_non_select(self) -> None:
        conn = create_benchmark_db()
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_sql_safe(conn, "PRAGMA table_info(customers);")
        conn.close()


# ── Result Comparison ───────────────────────────────────────────────────────


class TestCompareResults:
    """Tests for compare_results()."""

    def test_matching_results(self) -> None:
        match, reason = compare_results(
            ["name"], [("Alice",), ("Bob",)],
            ["name"], [("Alice",), ("Bob",)],
        )
        assert match is True
        assert reason == ""

    def test_unordered_match(self) -> None:
        match, _ = compare_results(
            ["name"], [("Alice",), ("Bob",)],
            ["name"], [("Bob",), ("Alice",)],
            order_matters=False,
        )
        assert match is True

    def test_ordered_mismatch(self) -> None:
        match, reason = compare_results(
            ["name"], [("Alice",), ("Bob",)],
            ["name"], [("Bob",), ("Alice",)],
            order_matters=True,
        )
        assert match is False
        assert "mismatch" in reason.lower()

    def test_row_count_mismatch(self) -> None:
        match, reason = compare_results(
            ["name"], [("Alice",)],
            ["name"], [("Alice",), ("Bob",)],
        )
        assert match is False
        assert "row count" in reason.lower()

    def test_column_count_mismatch(self) -> None:
        match, reason = compare_results(
            ["name"], [("Alice",)],
            ["name", "age"], [("Alice", 25)],
        )
        assert match is False
        assert "column count" in reason.lower()

    def test_float_tolerance(self) -> None:
        match, _ = compare_results(
            ["val"], [(100.0,)],
            ["val"], [(100.5,)],
        )
        assert match is True  # 0.5% relative difference, within 1% tolerance

    def test_case_insensitive_strings(self) -> None:
        match, _ = compare_results(
            ["name"], [("ALICE",)],
            ["name"], [("alice",)],
        )
        assert match is True

    def test_none_values(self) -> None:
        match, _ = compare_results(
            ["val"], [(None,)],
            ["val"], [(None,)],
        )
        assert match is True


# ── Questions ───────────────────────────────────────────────────────────────


class TestQuestions:
    """Tests for the question definitions."""

    def test_has_25_questions(self) -> None:
        assert len(QUESTIONS) == 25

    def test_has_all_difficulties(self) -> None:
        difficulties = {q.difficulty for q in QUESTIONS}
        assert difficulties == {"trivial", "easy", "medium", "hard"}

    def test_trivial_count(self) -> None:
        assert sum(1 for q in QUESTIONS if q.difficulty == "trivial") == 4

    def test_easy_count(self) -> None:
        assert sum(1 for q in QUESTIONS if q.difficulty == "easy") == 5

    def test_medium_count(self) -> None:
        assert sum(1 for q in QUESTIONS if q.difficulty == "medium") == 9

    def test_hard_count(self) -> None:
        assert sum(1 for q in QUESTIONS if q.difficulty == "hard") == 7

    def test_all_have_expected_results(self) -> None:
        for q in QUESTIONS:
            assert len(q.expected_columns) > 0, f"Q{q.id} has no expected columns"
            assert len(q.expected_result) > 0 or q.id == 19, f"Q{q.id} has no expected results"

    def test_unique_ids(self) -> None:
        ids = [q.id for q in QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_reference_sql_runs(self) -> None:
        """All reference SQL queries should execute without errors."""
        conn = create_benchmark_db()
        for q in QUESTIONS:
            cursor = conn.execute(q.reference_sql)
            rows = cursor.fetchall()
            assert cursor.description is not None, f"Q{q.id} returned no description"
        conn.close()


# ── Result Persistence ──────────────────────────────────────────────────────


class TestResultsToRecord:
    """Tests for results_to_record()."""

    def test_empty_results(self) -> None:
        record = results_to_record([], "test-runtime")
        assert record.type == "sql"
        assert record.model == "unknown"

    def test_basic_record(self) -> None:
        results = [
            SQLTaskResult(question_id=1, question="Q1", difficulty="trivial", model="test-model", status="pass"),
            SQLTaskResult(question_id=2, question="Q2", difficulty="easy", model="test-model", status="fail"),
            SQLTaskResult(question_id=3, question="Q3", difficulty="medium", model="test-model", status="error"),
        ]
        record = results_to_record(results, "test-runtime")
        assert record.type == "sql"
        assert record.model == "test-model"
        assert record.runtime == "test-runtime"
        assert record.quality_scores["total_pass"] == 1
        assert record.quality_scores["total_fail"] == 1
        assert record.quality_scores["total_error"] == 1
        assert record.quality_scores["total_questions"] == 3
        assert abs(record.quality_scores["pass_rate"] - 1 / 3) < 0.01


# ── Summary Printing ────────────────────────────────────────────────────────


class TestPrintSqlSummary:
    """Tests for print_sql_summary()."""

    def test_no_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_sql_summary([])
        assert "No results" in capsys.readouterr().out

    def test_with_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            SQLTaskResult(question_id=1, question="Q1", difficulty="trivial",
                         model="test", status="pass", speed_tps=25.0, ttft_ms=100.0),
            SQLTaskResult(question_id=2, question="Q2", difficulty="easy",
                         model="test", status="fail"),
        ]
        print_sql_summary(results)
        output = capsys.readouterr().out
        assert "SQL BENCHMARK" in output
        assert "test" in output
