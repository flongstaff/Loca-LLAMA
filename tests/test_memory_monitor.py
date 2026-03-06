"""Unit tests for loca_llama.memory_monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest

from loca_llama.memory_monitor import (
    BG_GREEN,
    BG_RED,
    BG_YELLOW,
    BOLD,
    DIM,
    GREEN,
    RED,
    RESET,
    WHITE,
    YELLOW,
    MemoryMonitor,
    MemoryReport,
    MemorySample,
    _get_memory_usage_darwin,
    _get_memory_usage_linux,
    _get_total_memory_gb,
    format_memory_report,
    format_mini_memory_bar,
    get_memory_sample,
    memory_bar,
    pressure_badge,
)


# ── MemorySample ──────────────────────────────────────────────────────────────


class TestMemorySampleUsagePct:
    def test_usage_pct_normal(self):
        """usage_pct should return (used/total)*100."""
        sample = MemorySample(timestamp=0.0, used_gb=8.0, free_gb=8.0, total_gb=16.0)
        assert sample.usage_pct == pytest.approx(50.0)

    def test_usage_pct_full(self):
        """usage_pct should return 100 when used equals total."""
        sample = MemorySample(timestamp=0.0, used_gb=32.0, free_gb=0.0, total_gb=32.0)
        assert sample.usage_pct == pytest.approx(100.0)

    def test_usage_pct_zero_used(self):
        """usage_pct should return 0 when no memory is used."""
        sample = MemorySample(timestamp=0.0, used_gb=0.0, free_gb=16.0, total_gb=16.0)
        assert sample.usage_pct == pytest.approx(0.0)

    def test_usage_pct_zero_total_returns_zero(self):
        """usage_pct should return 0 when total_gb is 0 (avoid ZeroDivisionError)."""
        sample = MemorySample(timestamp=0.0, used_gb=8.0, free_gb=0.0, total_gb=0.0)
        assert sample.usage_pct == 0


# ── MemoryReport ──────────────────────────────────────────────────────────────


class TestMemoryReportDeltaGb:
    def test_delta_gb_positive(self):
        """delta_gb should return peak minus baseline."""
        report = MemoryReport(peak_used_gb=20.0, baseline_used_gb=10.0, total_gb=32.0)
        assert report.delta_gb == pytest.approx(10.0)

    def test_delta_gb_zero_when_no_change(self):
        """delta_gb should be 0 when peak equals baseline."""
        report = MemoryReport(peak_used_gb=10.0, baseline_used_gb=10.0, total_gb=32.0)
        assert report.delta_gb == pytest.approx(0.0)

    def test_delta_gb_negative_allowed(self):
        """delta_gb can be negative (unusual, but arithmetically valid)."""
        report = MemoryReport(peak_used_gb=8.0, baseline_used_gb=12.0, total_gb=32.0)
        assert report.delta_gb == pytest.approx(-4.0)


class TestMemoryReportPeakPct:
    def test_peak_pct_normal(self):
        """peak_pct should return (peak/total)*100."""
        report = MemoryReport(peak_used_gb=24.0, baseline_used_gb=10.0, total_gb=32.0)
        assert report.peak_pct == pytest.approx(75.0)

    def test_peak_pct_zero_total_returns_zero(self):
        """peak_pct should return 0 when total_gb is 0."""
        report = MemoryReport(peak_used_gb=10.0, baseline_used_gb=5.0, total_gb=0.0)
        assert report.peak_pct == 0


class TestMemoryReportBaselinePct:
    def test_baseline_pct_normal(self):
        """baseline_pct should return (baseline/total)*100."""
        report = MemoryReport(peak_used_gb=20.0, baseline_used_gb=8.0, total_gb=16.0)
        assert report.baseline_pct == pytest.approx(50.0)

    def test_baseline_pct_zero_total_returns_zero(self):
        """baseline_pct should return 0 when total_gb is 0."""
        report = MemoryReport(peak_used_gb=5.0, baseline_used_gb=5.0, total_gb=0.0)
        assert report.baseline_pct == 0


class TestMemoryReportEmptySamples:
    def test_empty_samples_field_default(self):
        """MemoryReport with default factory should have empty samples list."""
        report = MemoryReport(total_gb=32.0)
        assert report.samples == []

    def test_delta_gb_with_empty_samples(self):
        """delta_gb with zero peak and baseline should be 0."""
        report = MemoryReport(total_gb=32.0)
        assert report.delta_gb == pytest.approx(0.0)


# ── _get_total_memory_gb ──────────────────────────────────────────────────────


class TestGetTotalMemoryGb:
    def test_darwin_returns_gb_from_sysctl(self):
        """On darwin, should parse sysctl hw.memsize output and convert to GB."""
        bytes_34gb = 34_359_738_368  # 32 GiB
        mock_result = MagicMock()
        mock_result.stdout = f"{bytes_34gb}\n"

        with patch("sys.platform", "darwin"), \
             patch("subprocess.run", return_value=mock_result):
            total = _get_total_memory_gb()

        assert total == pytest.approx(bytes_34gb / (1024 ** 3))

    def test_darwin_exception_falls_through_to_linux_path(self):
        """On darwin, if sysctl raises, should fall through to /proc/meminfo path."""
        meminfo_content = "MemTotal:       33554432 kB\nMemFree: 0 kB\n"

        with patch("sys.platform", "darwin"), \
             patch("subprocess.run", side_effect=OSError("no sysctl")), \
             patch("builtins.open", mock_open(read_data=meminfo_content)):
            total = _get_total_memory_gb()

        assert total == pytest.approx(33554432 / (1024 ** 2))

    def test_linux_reads_proc_meminfo(self):
        """On linux, should read /proc/meminfo MemTotal line and convert to GB."""
        meminfo_content = "MemTotal:       16777216 kB\nMemFree: 0 kB\n"

        with patch("sys.platform", "linux"), \
             patch("builtins.open", mock_open(read_data=meminfo_content)):
            total = _get_total_memory_gb()

        assert total == pytest.approx(16777216 / (1024 ** 2))

    def test_exception_fallback_returns_zero(self):
        """When all reads fail, should return 0.0."""
        with patch("sys.platform", "linux"), \
             patch("builtins.open", side_effect=OSError("no meminfo")):
            total = _get_total_memory_gb()

        assert total == 0.0


# ── _get_memory_usage_darwin ──────────────────────────────────────────────────


_VM_STAT_OUTPUT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                           1000.
Pages active:                         2000.
Pages inactive:                        500.
Pages speculative:                     100.
Pages throttled:                         0.
Pages wired down:                      800.
Pages purgeable:                         0.
"Translation faults":                    0.
Pages copy-on-write:                     0.
Pages zero filled:                       0.
Pages reactivated:                       0.
Pages purged:                            0.
File-backed pages:                       0.
Anonymous pages:                         0.
Pages stored in compressor:            300.
Pages occupied by compressor:          300.
Decompressions:                          0.
Compressions:                            0.
Pageins:                                 0.
Pageouts:                                0.
Swapins:                                 0.
Swapouts:                                0.
"""

_MEMORY_PRESSURE_NORMAL = "System-wide memory free percentage: 55%\nSystem memory status: normal\n"
_MEMORY_PRESSURE_WARN = "System memory status: warn\n"
_MEMORY_PRESSURE_CRITICAL = "System memory status: critical\n"


class TestGetMemoryUsageDarwin:
    def _make_run(self, vm_stat_out: str, pressure_out: str):
        """Return a side_effect function that dispatches on the command name."""
        def _run(cmd, **kwargs):
            result = MagicMock()
            if cmd[0] == "vm_stat":
                result.stdout = vm_stat_out
            else:
                result.stdout = pressure_out
            return result
        return _run

    def test_parses_page_size_from_header(self):
        """Should extract page_size from the vm_stat header line."""
        run_fn = self._make_run(_VM_STAT_OUTPUT, _MEMORY_PRESSURE_NORMAL)

        with patch("subprocess.run", side_effect=run_fn):
            used_gb, free_gb, pressure = _get_memory_usage_darwin()

        page_size = 16384
        active, wired, compressor = 2000, 800, 300
        free_pages, inactive, speculative = 1000, 500, 100
        expected_used = (active + wired + compressor) * page_size / (1024 ** 3)
        expected_free = (free_pages + inactive + speculative) * page_size / (1024 ** 3)

        assert used_gb == pytest.approx(expected_used)
        assert free_gb == pytest.approx(expected_free)

    def test_pressure_normal(self):
        """Should return pressure='normal' when memory_pressure output says normal."""
        run_fn = self._make_run(_VM_STAT_OUTPUT, _MEMORY_PRESSURE_NORMAL)
        with patch("subprocess.run", side_effect=run_fn):
            _, _, pressure = _get_memory_usage_darwin()
        assert pressure == "normal"

    def test_pressure_warn(self):
        """Should return pressure='warn' when memory_pressure output contains 'warn'."""
        run_fn = self._make_run(_VM_STAT_OUTPUT, _MEMORY_PRESSURE_WARN)
        with patch("subprocess.run", side_effect=run_fn):
            _, _, pressure = _get_memory_usage_darwin()
        assert pressure == "warn"

    def test_pressure_critical(self):
        """Should return pressure='critical' when memory_pressure output contains 'critical'."""
        run_fn = self._make_run(_VM_STAT_OUTPUT, _MEMORY_PRESSURE_CRITICAL)
        with patch("subprocess.run", side_effect=run_fn):
            _, _, pressure = _get_memory_usage_darwin()
        assert pressure == "critical"

    def test_vm_stat_exception_returns_unknown(self):
        """Should return (0, 0, 'unknown') when vm_stat subprocess raises."""
        with patch("subprocess.run", side_effect=OSError("not found")):
            used_gb, free_gb, pressure = _get_memory_usage_darwin()
        assert used_gb == 0.0
        assert free_gb == 0.0
        assert pressure == "unknown"

    def test_memory_pressure_exception_defaults_to_normal(self):
        """Should fall back to 'normal' pressure when memory_pressure command raises."""
        call_count = 0

        def _run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if cmd[0] == "vm_stat":
                result = MagicMock()
                result.stdout = _VM_STAT_OUTPUT
                return result
            raise OSError("memory_pressure not found")

        with patch("subprocess.run", side_effect=_run):
            _, _, pressure = _get_memory_usage_darwin()

        assert pressure == "normal"

    def test_default_page_size_used_when_not_in_header(self):
        """Should use 16384 as default page_size if header has no 'page size of' text."""
        no_header_output = "Pages free:                           1000.\nPages active:                         2000.\n"
        run_fn = self._make_run(no_header_output, _MEMORY_PRESSURE_NORMAL)

        with patch("subprocess.run", side_effect=run_fn):
            used_gb, free_gb, _ = _get_memory_usage_darwin()

        # active=2000, wired=0, compressor=0, page_size=16384
        expected_used = 2000 * 16384 / (1024 ** 3)
        assert used_gb == pytest.approx(expected_used)


# ── _get_memory_usage_linux ───────────────────────────────────────────────────


class TestGetMemoryUsageLinux:
    def _meminfo(self, total_kb: int, available_kb: int) -> str:
        return f"MemTotal:       {total_kb} kB\nMemAvailable:   {available_kb} kB\n"

    def test_normal_pressure_below_75pct(self):
        """Should return pressure='normal' when usage is below 75%."""
        content = self._meminfo(total_kb=1_000_000, available_kb=300_000)  # 70% used
        with patch("builtins.open", mock_open(read_data=content)):
            _, _, pressure = _get_memory_usage_linux()
        assert pressure == "normal"

    def test_warn_pressure_at_75pct(self):
        """Should return pressure='warn' when usage is exactly 75%."""
        content = self._meminfo(total_kb=1_000_000, available_kb=250_000)  # 75% used
        with patch("builtins.open", mock_open(read_data=content)):
            _, _, pressure = _get_memory_usage_linux()
        assert pressure == "warn"

    def test_warn_pressure_between_75_and_90pct(self):
        """Should return pressure='warn' when usage is between 75% and 90%."""
        content = self._meminfo(total_kb=1_000_000, available_kb=150_000)  # 85% used
        with patch("builtins.open", mock_open(read_data=content)):
            _, _, pressure = _get_memory_usage_linux()
        assert pressure == "warn"

    def test_critical_pressure_at_90pct(self):
        """Should return pressure='critical' when usage is exactly 90%."""
        content = self._meminfo(total_kb=1_000_000, available_kb=100_000)  # 90% used
        with patch("builtins.open", mock_open(read_data=content)):
            _, _, pressure = _get_memory_usage_linux()
        assert pressure == "critical"

    def test_critical_pressure_above_90pct(self):
        """Should return pressure='critical' when usage is above 90%."""
        content = self._meminfo(total_kb=1_000_000, available_kb=50_000)  # 95% used
        with patch("builtins.open", mock_open(read_data=content)):
            _, _, pressure = _get_memory_usage_linux()
        assert pressure == "critical"

    def test_used_and_free_gb_values(self):
        """Should correctly compute used_gb and free_gb from MemTotal/MemAvailable."""
        total_kb = 16_777_216  # 16 GiB in KB
        available_kb = 8_388_608  # 8 GiB in KB
        content = self._meminfo(total_kb=total_kb, available_kb=available_kb)

        with patch("builtins.open", mock_open(read_data=content)):
            used_gb, free_gb, _ = _get_memory_usage_linux()

        expected_used = (total_kb - available_kb) / (1024 ** 2)
        expected_free = available_kb / (1024 ** 2)
        assert used_gb == pytest.approx(expected_used)
        assert free_gb == pytest.approx(expected_free)

    def test_exception_returns_unknown(self):
        """Should return (0, 0, 'unknown') when /proc/meminfo cannot be read."""
        with patch("builtins.open", side_effect=OSError("no meminfo")):
            used_gb, free_gb, pressure = _get_memory_usage_linux()
        assert used_gb == 0.0
        assert free_gb == 0.0
        assert pressure == "unknown"


# ── get_memory_sample ─────────────────────────────────────────────────────────


class TestGetMemorySample:
    def test_darwin_calls_darwin_function(self):
        """On darwin, get_memory_sample should delegate to _get_memory_usage_darwin."""
        with patch("sys.platform", "darwin"), \
             patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor._get_memory_usage_darwin", return_value=(10.0, 22.0, "normal")) as mock_darwin, \
             patch("time.monotonic", return_value=5.0):
            sample = get_memory_sample(start_time=1.0)

        mock_darwin.assert_called_once()
        assert sample.used_gb == 10.0
        assert sample.free_gb == 22.0
        assert sample.total_gb == 32.0
        assert sample.pressure == "normal"
        assert sample.timestamp == pytest.approx(4.0)  # 5.0 - 1.0

    def test_linux_calls_linux_function(self):
        """On linux, get_memory_sample should delegate to _get_memory_usage_linux."""
        with patch("sys.platform", "linux"), \
             patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=16.0), \
             patch("loca_llama.memory_monitor._get_memory_usage_linux", return_value=(8.0, 8.0, "warn")) as mock_linux, \
             patch("time.monotonic", return_value=2.0):
            sample = get_memory_sample(start_time=0.0)

        mock_linux.assert_called_once()
        assert sample.pressure == "warn"
        assert sample.timestamp == pytest.approx(2.0)

    def test_default_start_time_zero(self):
        """get_memory_sample with no start_time should use 0.0 as default."""
        with patch("sys.platform", "linux"), \
             patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=16.0), \
             patch("loca_llama.memory_monitor._get_memory_usage_linux", return_value=(4.0, 12.0, "normal")), \
             patch("time.monotonic", return_value=3.0):
            sample = get_memory_sample()

        assert sample.timestamp == pytest.approx(3.0)


# ── MemoryMonitor ─────────────────────────────────────────────────────────────


def _make_sample(used_gb: float = 10.0, total_gb: float = 32.0, ts: float = 0.0) -> MemorySample:
    return MemorySample(
        timestamp=ts,
        used_gb=used_gb,
        free_gb=total_gb - used_gb,
        total_gb=total_gb,
        pressure="normal",
    )


class TestMemoryMonitorStart:
    def test_start_takes_baseline_sample(self):
        """start() should immediately take a baseline sample before spawning thread."""
        baseline = _make_sample(used_gb=8.0)

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", return_value=baseline), \
             patch("time.monotonic", return_value=0.0):
            monitor = MemoryMonitor(interval=60.0)
            monitor.start()
            monitor._running = False  # prevent loop from taking more samples
            if monitor._thread:
                monitor._thread.join(timeout=1.0)

        assert len(monitor._samples) >= 1
        assert monitor._samples[0].used_gb == 8.0

    def test_start_sets_running_flag(self):
        """start() should set _running to True while thread is active."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", return_value=_make_sample()):
            monitor = MemoryMonitor(interval=60.0)
            monitor.start()
            assert monitor._running is True
            monitor._running = False
            if monitor._thread:
                monitor._thread.join(timeout=1.0)

    def test_start_resets_samples_on_restart(self):
        """Calling start() twice should clear previous samples."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", return_value=_make_sample()):
            monitor = MemoryMonitor(interval=60.0)
            monitor._samples = [_make_sample(), _make_sample()]  # pre-populate

            monitor.start()
            monitor._running = False
            if monitor._thread:
                monitor._thread.join(timeout=1.0)

        # After start(), exactly 1 baseline sample should be present (loop didn't fire)
        assert len(monitor._samples) == 1


class TestMemoryMonitorStop:
    def test_stop_takes_final_sample_and_returns_report(self):
        """stop() should append a final sample and return a MemoryReport."""
        baseline = _make_sample(used_gb=8.0, ts=0.0)
        final = _make_sample(used_gb=12.0, ts=1.0)
        call_count = 0

        def _sample_factory(start_time=0.0):
            nonlocal call_count
            call_count += 1
            return baseline if call_count == 1 else final

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", side_effect=_sample_factory), \
             patch("time.monotonic", return_value=0.0):
            monitor = MemoryMonitor(interval=60.0)
            monitor.start()
            monitor._running = False
            if monitor._thread:
                monitor._thread.join(timeout=1.0)
            report = monitor.stop()

        assert isinstance(report, MemoryReport)
        assert report.peak_used_gb == pytest.approx(12.0)
        assert report.baseline_used_gb == pytest.approx(8.0)

    def test_stop_sets_thread_to_none(self):
        """stop() should set _thread to None after joining."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", return_value=_make_sample()):
            monitor = MemoryMonitor(interval=60.0)
            monitor.start()
            monitor.stop()

        assert monitor._thread is None


class TestMemoryMonitorGetCurrent:
    def test_returns_last_sample_when_samples_exist(self):
        """get_current() should return the most recent sample."""
        s1 = _make_sample(used_gb=8.0, ts=0.0)
        s2 = _make_sample(used_gb=12.0, ts=1.0)

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()
            monitor._samples = [s1, s2]

        assert monitor.get_current() is s2

    def test_returns_none_when_no_samples(self):
        """get_current() should return None when _samples is empty."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()

        assert monitor.get_current() is None


class TestMemoryMonitorGetHistory:
    def test_returns_all_samples_within_limit(self):
        """get_history() should return all samples when count is within limit."""
        samples = [_make_sample(ts=float(i)) for i in range(5)]

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()
            monitor._samples = samples

        history = monitor.get_history(limit=60)
        assert len(history) == 5

    def test_respects_limit(self):
        """get_history(limit=3) should return only the last 3 samples."""
        samples = [_make_sample(ts=float(i)) for i in range(10)]

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()
            monitor._samples = samples

        history = monitor.get_history(limit=3)
        assert len(history) == 3
        assert history[-1].timestamp == samples[-1].timestamp

    def test_returns_snapshot_not_reference(self):
        """get_history() should return a copy, not the internal list."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()
            monitor._samples = [_make_sample()]

        history = monitor.get_history()
        history.append(_make_sample())
        assert len(monitor._samples) == 1

    def test_empty_history_returns_empty_list(self):
        """get_history() on a fresh monitor should return []."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()

        assert monitor.get_history() == []


class TestMemoryMonitorGetReport:
    def test_get_report_returns_memory_report(self):
        """get_report() should return a MemoryReport built from current samples."""
        s1 = _make_sample(used_gb=8.0, ts=0.0)
        s2 = _make_sample(used_gb=16.0, ts=2.0)

        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()
            monitor._samples = [s1, s2]

        report = monitor.get_report()
        assert isinstance(report, MemoryReport)
        assert report.peak_used_gb == pytest.approx(16.0)
        assert report.baseline_used_gb == pytest.approx(8.0)

    def test_get_report_empty_samples_returns_default(self):
        """get_report() with no samples should return a MemoryReport with total_gb set."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()

        report = monitor.get_report()
        assert isinstance(report, MemoryReport)
        assert report.total_gb == pytest.approx(32.0)
        assert report.samples == []


class TestMemoryMonitorStartTime:
    def test_start_time_reflects_monotonic_value_at_start(self):
        """start_time property should return _start_time set during start()."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0), \
             patch("loca_llama.memory_monitor.get_memory_sample", return_value=_make_sample()), \
             patch("time.monotonic", return_value=42.5):
            monitor = MemoryMonitor(interval=60.0)
            monitor.start()
            monitor._running = False
            if monitor._thread:
                monitor._thread.join(timeout=1.0)

        assert monitor.start_time == pytest.approx(42.5)

    def test_start_time_default_zero_before_start(self):
        """start_time should be 0.0 before start() is called."""
        with patch("loca_llama.memory_monitor._get_total_memory_gb", return_value=32.0):
            monitor = MemoryMonitor()

        assert monitor.start_time == pytest.approx(0.0)


# ── memory_bar ────────────────────────────────────────────────────────────────


class TestMemoryBar:
    def test_green_at_or_below_60pct(self):
        """memory_bar should use GREEN color when usage <= 60%."""
        bar = memory_bar(used_gb=6.0, total_gb=10.0, width=10)
        assert GREEN in bar
        assert YELLOW not in bar
        assert RED not in bar

    def test_yellow_between_61_and_80pct(self):
        """memory_bar should use YELLOW color when 60% < usage <= 80%."""
        bar = memory_bar(used_gb=7.0, total_gb=10.0, width=10)
        assert YELLOW in bar
        assert RED not in bar

    def test_red_above_80pct(self):
        """memory_bar should use RED color when usage > 80%."""
        bar = memory_bar(used_gb=9.0, total_gb=10.0, width=10)
        assert RED in bar
        assert YELLOW not in bar

    def test_exactly_60pct_is_green(self):
        """memory_bar at exactly 60% should use GREEN (boundary inclusive)."""
        bar = memory_bar(used_gb=6.0, total_gb=10.0, width=10)
        assert GREEN in bar

    def test_exactly_80pct_is_yellow(self):
        """memory_bar at exactly 80% should use YELLOW (boundary inclusive)."""
        bar = memory_bar(used_gb=8.0, total_gb=10.0, width=10)
        assert YELLOW in bar

    def test_zero_total_returns_zero_fill(self):
        """memory_bar with total_gb=0 should not raise and should use zero fill."""
        bar = memory_bar(used_gb=5.0, total_gb=0.0, width=10)
        assert "0/0 GB" in bar

    def test_bar_contains_gb_and_pct_info(self):
        """memory_bar output should include used/total GB and percentage."""
        bar = memory_bar(used_gb=8.0, total_gb=16.0, width=20)
        assert "8.0/16 GB" in bar
        assert "50%" in bar

    def test_bar_width_affects_filled_chars(self):
        """memory_bar should fill proportional number of block chars based on width."""
        bar_wide = memory_bar(used_gb=5.0, total_gb=10.0, width=20)
        bar_narrow = memory_bar(used_gb=5.0, total_gb=10.0, width=10)
        # 50% of 20 = 10 filled, 50% of 10 = 5 filled
        assert bar_wide.count("█") == 10
        assert bar_narrow.count("█") == 5


# ── pressure_badge ────────────────────────────────────────────────────────────


class TestPressureBadge:
    def test_normal_uses_bg_green(self):
        """pressure_badge('normal') should contain BG_GREEN styling."""
        badge = pressure_badge("normal")
        assert BG_GREEN in badge
        assert "NORMAL" in badge

    def test_warn_uses_bg_yellow(self):
        """pressure_badge('warn') should contain BG_YELLOW styling."""
        badge = pressure_badge("warn")
        assert BG_YELLOW in badge
        assert "WARNING" in badge

    def test_critical_uses_bg_red(self):
        """pressure_badge('critical') should contain BG_RED styling."""
        badge = pressure_badge("critical")
        assert BG_RED in badge
        assert "CRITICAL" in badge

    def test_unknown_pressure_uses_dim(self):
        """pressure_badge for an unrecognised value should use DIM and include the value."""
        badge = pressure_badge("unknown_state")
        assert DIM in badge
        assert "unknown_state" in badge

    def test_empty_string_uses_dim(self):
        """pressure_badge for empty string should use DIM fallback."""
        badge = pressure_badge("")
        assert DIM in badge


# ── format_memory_report ──────────────────────────────────────────────────────


class TestFormatMemoryReport:
    def _make_report(self) -> MemoryReport:
        samples = [
            _make_sample(used_gb=10.0, ts=0.0),
            _make_sample(used_gb=20.0, ts=5.0),
        ]
        return MemoryReport(
            samples=samples,
            peak_used_gb=20.0,
            baseline_used_gb=10.0,
            total_gb=32.0,
            duration_sec=5.0,
        )

    def test_output_contains_total_memory(self):
        """format_memory_report output should include Total Memory value."""
        result = format_memory_report(self._make_report())
        assert "32" in result

    def test_output_contains_baseline_gb(self):
        """format_memory_report output should include Baseline GB value."""
        result = format_memory_report(self._make_report())
        assert "10.0 GB" in result

    def test_output_contains_peak_gb(self):
        """format_memory_report output should include Peak GB value."""
        result = format_memory_report(self._make_report())
        assert "20.0 GB" in result

    def test_output_contains_delta(self):
        """format_memory_report output should include Delta value."""
        result = format_memory_report(self._make_report())
        assert "10.0 GB" in result  # delta = 20.0 - 10.0

    def test_output_contains_duration(self):
        """format_memory_report output should include Duration value."""
        result = format_memory_report(self._make_report())
        assert "5.0s" in result

    def test_pressure_badge_included_when_non_normal(self):
        """format_memory_report should include pressure badge when last sample has warn/critical."""
        samples = [_make_sample(used_gb=10.0, ts=0.0)]
        samples[0] = MemorySample(
            timestamp=0.0, used_gb=10.0, free_gb=22.0, total_gb=32.0, pressure="warn"
        )
        report = MemoryReport(
            samples=samples,
            peak_used_gb=10.0,
            baseline_used_gb=10.0,
            total_gb=32.0,
            duration_sec=1.0,
        )
        result = format_memory_report(report)
        assert "WARNING" in result

    def test_pressure_badge_not_included_when_unknown(self):
        """format_memory_report should omit pressure badge when pressure is 'unknown'."""
        sample = MemorySample(
            timestamp=0.0, used_gb=10.0, free_gb=22.0, total_gb=32.0, pressure="unknown"
        )
        report = MemoryReport(
            samples=[sample],
            peak_used_gb=10.0,
            baseline_used_gb=10.0,
            total_gb=32.0,
            duration_sec=1.0,
        )
        result = format_memory_report(report)
        assert "Pressure" not in result

    def test_no_pressure_line_when_no_samples(self):
        """format_memory_report with empty samples should not include Pressure line."""
        report = MemoryReport(total_gb=32.0)
        result = format_memory_report(report)
        assert "Pressure" not in result


# ── format_mini_memory_bar ────────────────────────────────────────────────────


class TestFormatMiniMemoryBar:
    def test_green_at_or_below_60pct(self):
        """format_mini_memory_bar should use GREEN when usage <= 60%."""
        sample = _make_sample(used_gb=6.0)  # 6/32 ≈ 18.75%
        bar = format_mini_memory_bar(sample, width=15)
        assert GREEN in bar

    def test_yellow_between_61_and_80pct(self):
        """format_mini_memory_bar should use YELLOW when 60% < usage <= 80%."""
        sample = MemorySample(timestamp=0.0, used_gb=7.0, free_gb=3.0, total_gb=10.0)
        bar = format_mini_memory_bar(sample, width=10)
        assert YELLOW in bar

    def test_red_above_80pct(self):
        """format_mini_memory_bar should use RED when usage > 80%."""
        sample = MemorySample(timestamp=0.0, used_gb=9.0, free_gb=1.0, total_gb=10.0)
        bar = format_mini_memory_bar(sample, width=10)
        assert RED in bar

    def test_zero_total_returns_zero_fill(self):
        """format_mini_memory_bar with total_gb=0 should not raise and still show used_gb."""
        sample = MemorySample(timestamp=0.0, used_gb=5.0, free_gb=0.0, total_gb=0.0)
        bar = format_mini_memory_bar(sample, width=10)
        # pct=0 so bar is all empty (░), and used_gb=5.0 is still displayed
        assert "5.0G" in bar
        assert "█" not in bar

    def test_output_contains_used_gb(self):
        """format_mini_memory_bar output should include used_gb formatted as xG."""
        sample = MemorySample(timestamp=0.0, used_gb=8.5, free_gb=7.5, total_gb=16.0)
        bar = format_mini_memory_bar(sample, width=10)
        assert "8.5G" in bar

    def test_exactly_60pct_is_green(self):
        """format_mini_memory_bar at exactly 60% should use GREEN."""
        sample = MemorySample(timestamp=0.0, used_gb=6.0, free_gb=4.0, total_gb=10.0)
        bar = format_mini_memory_bar(sample, width=10)
        assert GREEN in bar

    def test_exactly_80pct_is_yellow(self):
        """format_mini_memory_bar at exactly 80% should use YELLOW."""
        sample = MemorySample(timestamp=0.0, used_gb=8.0, free_gb=2.0, total_gb=10.0)
        bar = format_mini_memory_bar(sample, width=10)
        assert YELLOW in bar
