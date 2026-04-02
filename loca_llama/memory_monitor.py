"""Memory monitor: track unified memory usage during benchmarks (nvtop-like).

On Apple Silicon, CPU and GPU share unified memory, so monitoring system
memory gives us a complete picture of VRAM usage during inference.

Uses only stdlib — no psutil or external dependencies required.
"""

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


@dataclass
class MemorySample:
    """A single memory measurement."""

    timestamp: float  # seconds since monitor start
    used_gb: float
    free_gb: float
    total_gb: float
    pressure: str = ""  # "normal", "warn", "critical" (macOS)

    @property
    def usage_pct(self) -> float:
        return (self.used_gb / self.total_gb) * 100 if self.total_gb > 0 else 0


@dataclass
class MemoryReport:
    """Summary of memory usage over a monitoring period."""

    samples: list[MemorySample] = field(default_factory=list)
    peak_used_gb: float = 0.0
    baseline_used_gb: float = 0.0
    total_gb: float = 0.0
    duration_sec: float = 0.0

    @property
    def delta_gb(self) -> float:
        """Memory increase from baseline to peak."""
        return self.peak_used_gb - self.baseline_used_gb

    @property
    def peak_pct(self) -> float:
        return (self.peak_used_gb / self.total_gb) * 100 if self.total_gb > 0 else 0

    @property
    def baseline_pct(self) -> float:
        return (self.baseline_used_gb / self.total_gb) * 100 if self.total_gb > 0 else 0


# ── macOS Memory Reading ────────────────────────────────────────────────────

def _get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass

    # Linux fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass

    return 0.0


def _get_memory_usage_darwin() -> tuple[float, float, str]:
    """Get memory usage on macOS using vm_stat.

    Returns: (used_gb, free_gb, pressure_level)
    """
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        )
        output = result.stdout

        page_size = 16384  # Apple Silicon default
        # Parse page size from first line if available
        for line in output.split("\n"):
            if "page size of" in line:
                try:
                    page_size = int(line.split("page size of")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass
                break

        stats = {}
        for line in output.split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip().rstrip(".")
                try:
                    stats[key] = int(val)
                except ValueError:
                    pass

        # Calculate used memory
        # Active + Wired + Compressor = "used"
        # Inactive + Free + Speculative = "available-ish"
        active = stats.get("pages active", 0)
        wired = stats.get("pages wired down", 0)
        compressor = stats.get("pages occupied by compressor", 0)
        free_pages = stats.get("pages free", 0)
        inactive = stats.get("pages inactive", 0)
        speculative = stats.get("pages speculative", 0)

        used_bytes = (active + wired + compressor) * page_size
        free_bytes = (free_pages + inactive + speculative) * page_size

        used_gb = used_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)

    except Exception:
        return 0.0, 0.0, "unknown"

    # Get memory pressure level
    pressure = "normal"
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5,
        )
        output_lower = result.stdout.lower()
        if "critical" in output_lower:
            pressure = "critical"
        elif "warn" in output_lower:
            pressure = "warn"
    except Exception:
        pass

    return used_gb, free_gb, pressure


def _get_memory_usage_linux() -> tuple[float, float, str]:
    """Get memory usage on Linux from /proc/meminfo.

    Returns: (used_gb, free_gb, pressure_level)
    """
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])

        total_kb = info.get("MemTotal", 0)
        available_kb = info.get("MemAvailable", 0)
        used_kb = total_kb - available_kb
        free_kb = available_kb

        used_gb = used_kb / (1024**2)
        free_gb = free_kb / (1024**2)

        pct = (used_kb / total_kb) * 100 if total_kb > 0 else 0
        if pct >= 90:
            pressure = "critical"
        elif pct >= 75:
            pressure = "warn"
        else:
            pressure = "normal"

        return used_gb, free_gb, pressure

    except Exception:
        return 0.0, 0.0, "unknown"


def get_memory_sample(start_time: float = 0.0) -> MemorySample:
    """Take a single memory measurement."""
    total_gb = _get_total_memory_gb()

    if sys.platform == "darwin":
        used_gb, free_gb, pressure = _get_memory_usage_darwin()
    else:
        used_gb, free_gb, pressure = _get_memory_usage_linux()

    return MemorySample(
        timestamp=time.monotonic() - start_time,
        used_gb=used_gb,
        free_gb=free_gb,
        total_gb=total_gb,
        pressure=pressure,
    )


# ── Memory Monitor (Background Thread) ─────────────────────────────────────

class MemoryMonitor:
    """Background memory monitor that samples at regular intervals.

    Usage:
        monitor = MemoryMonitor(interval=0.5)
        monitor.start()
        # ... run inference ...
        report = monitor.stop()
        print(f"Peak: {report.peak_used_gb:.1f} GB")
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples: list[MemorySample] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._total_gb = _get_total_memory_gb()

    def start(self) -> None:
        """Start monitoring in background thread."""
        with self._lock:
            self._samples = []
        self._running = True
        self._start_time = time.monotonic()

        # Take baseline sample
        baseline = get_memory_sample(self._start_time)
        with self._lock:
            self._samples.append(baseline)

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryReport:
        """Stop monitoring and return report."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

        # Take final sample
        final = get_memory_sample(self._start_time)
        with self._lock:
            self._samples.append(final)

        return self._build_report()

    def get_current(self) -> MemorySample | None:
        """Get the most recent sample."""
        with self._lock:
            if self._samples:
                return self._samples[-1]
            return None

    def get_history(self, limit: int = 60) -> list[MemorySample]:
        """Return recent samples as a snapshot."""
        with self._lock:
            return list(self._samples[-limit:])

    def get_report(self) -> MemoryReport:
        """Return aggregate memory report since monitor start."""
        return self._build_report()

    @property
    def start_time(self) -> float:
        """Monotonic start time for wall-clock timestamp conversion."""
        return self._start_time

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break
            sample = get_memory_sample(self._start_time)
            with self._lock:
                self._samples.append(sample)

    def _build_report(self) -> MemoryReport:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return MemoryReport(total_gb=self._total_gb)

        baseline = samples[0].used_gb
        peak = max(s.used_gb for s in samples)
        duration = samples[-1].timestamp - samples[0].timestamp

        return MemoryReport(
            samples=samples,
            peak_used_gb=peak,
            baseline_used_gb=baseline,
            total_gb=self._total_gb,
            duration_sec=duration,
        )


# ── Display Helpers ─────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
WHITE = "\033[97m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_RED = "\033[41m"


def memory_bar(used_gb: float, total_gb: float, width: int = 30) -> str:
    """Create a visual memory usage bar."""
    pct = min(used_gb / total_gb, 1.0) if total_gb > 0 else 0
    filled = int(pct * width)
    empty = width - filled

    if pct <= 0.60:
        color = GREEN
    elif pct <= 0.80:
        color = YELLOW
    else:
        color = RED

    return f"{color}{'█' * filled}{'░' * empty}{RESET} {used_gb:.1f}/{total_gb:.0f} GB ({pct:.0%})"


def pressure_badge(pressure: str) -> str:
    """Visual badge for memory pressure level."""
    badges = {
        "normal": f"{BG_GREEN}{WHITE}{BOLD} NORMAL {RESET}",
        "warn": f"{BG_YELLOW}{WHITE}{BOLD} WARNING {RESET}",
        "critical": f"{BG_RED}{WHITE}{BOLD} CRITICAL {RESET}",
    }
    return badges.get(pressure, f"{DIM}[{pressure}]{RESET}")


def format_memory_report(report: MemoryReport) -> str:
    """Format a memory report for terminal display."""
    lines = []
    lines.append(f"  {BOLD}Memory Usage Report{RESET}")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  {BOLD}Total Memory:{RESET}    {report.total_gb:.0f} GB")
    lines.append(f"  {BOLD}Baseline:{RESET}        {report.baseline_used_gb:.1f} GB ({report.baseline_pct:.0f}%)")
    lines.append(f"  {BOLD}Peak:{RESET}            {report.peak_used_gb:.1f} GB ({report.peak_pct:.0f}%)")
    lines.append(f"  {BOLD}Delta (model):{RESET}   {report.delta_gb:.1f} GB")
    lines.append(f"  {BOLD}Duration:{RESET}        {report.duration_sec:.1f}s")
    lines.append(f"  {BOLD}Peak:{RESET} {memory_bar(report.peak_used_gb, report.total_gb, 35)}")

    if report.samples:
        last = report.samples[-1]
        if last.pressure and last.pressure != "unknown":
            lines.append(f"  {BOLD}Pressure:{RESET}       {pressure_badge(last.pressure)}")

    return "\n".join(lines)


def print_live_memory(label: str = "") -> None:
    """Print current memory usage (single line, for inline display)."""
    sample = get_memory_sample()
    prefix = f"{label} " if label else ""
    bar_str = memory_bar(sample.used_gb, sample.total_gb, 20)
    press = ""
    if sample.pressure and sample.pressure != "normal" and sample.pressure != "unknown":
        press = f" {pressure_badge(sample.pressure)}"
    print(f"  {prefix}{BOLD}MEM:{RESET} {bar_str}{press}")


def format_mini_memory_bar(sample: MemorySample, width: int = 15) -> str:
    """Compact memory bar for inline benchmark display."""
    pct = min(sample.used_gb / sample.total_gb, 1.0) if sample.total_gb > 0 else 0
    filled = int(pct * width)
    empty = width - filled

    if pct <= 0.60:
        color = GREEN
    elif pct <= 0.80:
        color = YELLOW
    else:
        color = RED

    return f"{color}{'█' * filled}{'░' * empty}{RESET} {sample.used_gb:.1f}G"
