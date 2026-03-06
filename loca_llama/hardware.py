"""Apple Silicon hardware specifications database."""

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class MacSpec:
    chip: str
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    memory_gb: int
    memory_bandwidth_gbs: float  # GB/s
    gpu_tflops: float

    @property
    def usable_memory_gb(self) -> float:
        """Memory available for LLM after OS/app overhead (~3-4 GB)."""
        return self.memory_gb - 4.0


# Pre-defined Apple Silicon configurations
APPLE_SILICON_SPECS: dict[str, MacSpec] = {
    # M1 family
    "M1 8GB": MacSpec("M1", 8, 8, 16, 8, 68.25, 2.6),
    "M1 16GB": MacSpec("M1", 8, 8, 16, 16, 68.25, 2.6),
    "M1 Pro 16GB": MacSpec("M1 Pro", 10, 16, 16, 16, 200.0, 5.2),
    "M1 Pro 32GB": MacSpec("M1 Pro", 10, 16, 16, 32, 200.0, 5.2),
    "M1 Max 32GB": MacSpec("M1 Max", 10, 32, 16, 32, 400.0, 10.4),
    "M1 Max 64GB": MacSpec("M1 Max", 10, 32, 16, 64, 400.0, 10.4),
    "M1 Ultra 64GB": MacSpec("M1 Ultra", 20, 64, 32, 64, 800.0, 20.8),
    "M1 Ultra 128GB": MacSpec("M1 Ultra", 20, 64, 32, 128, 800.0, 20.8),
    # M2 family
    "M2 8GB": MacSpec("M2", 8, 10, 16, 8, 100.0, 3.6),
    "M2 16GB": MacSpec("M2", 8, 10, 16, 16, 100.0, 3.6),
    "M2 24GB": MacSpec("M2", 8, 10, 16, 24, 100.0, 3.6),
    "M2 Pro 16GB": MacSpec("M2 Pro", 12, 19, 16, 16, 200.0, 6.8),
    "M2 Pro 32GB": MacSpec("M2 Pro", 12, 19, 16, 32, 200.0, 6.8),
    "M2 Max 32GB": MacSpec("M2 Max", 12, 38, 16, 32, 400.0, 13.6),
    "M2 Max 64GB": MacSpec("M2 Max", 12, 38, 16, 64, 400.0, 13.6),
    "M2 Max 96GB": MacSpec("M2 Max", 12, 38, 16, 96, 400.0, 13.6),
    "M2 Ultra 64GB": MacSpec("M2 Ultra", 24, 76, 32, 64, 800.0, 27.2),
    "M2 Ultra 128GB": MacSpec("M2 Ultra", 24, 76, 32, 128, 800.0, 27.2),
    "M2 Ultra 192GB": MacSpec("M2 Ultra", 24, 76, 32, 192, 800.0, 27.2),
    # M3 family
    "M3 8GB": MacSpec("M3", 8, 10, 16, 8, 100.0, 4.1),
    "M3 16GB": MacSpec("M3", 8, 10, 16, 16, 100.0, 4.1),
    "M3 24GB": MacSpec("M3", 8, 10, 16, 24, 100.0, 4.1),
    "M3 Pro 18GB": MacSpec("M3 Pro", 12, 18, 16, 18, 150.0, 7.4),
    "M3 Pro 36GB": MacSpec("M3 Pro", 12, 18, 16, 36, 150.0, 7.4),
    "M3 Max 36GB": MacSpec("M3 Max", 16, 40, 16, 36, 400.0, 16.4),
    "M3 Max 48GB": MacSpec("M3 Max", 16, 40, 16, 48, 400.0, 16.4),
    "M3 Max 64GB": MacSpec("M3 Max", 16, 40, 16, 64, 400.0, 16.4),
    "M3 Max 96GB": MacSpec("M3 Max", 14, 30, 16, 96, 300.0, 12.3),
    "M3 Max 128GB": MacSpec("M3 Max", 16, 40, 16, 128, 400.0, 16.4),
    "M3 Ultra 64GB": MacSpec("M3 Ultra", 32, 80, 32, 64, 800.0, 32.8),
    "M3 Ultra 128GB": MacSpec("M3 Ultra", 32, 80, 32, 128, 800.0, 32.8),
    "M3 Ultra 192GB": MacSpec("M3 Ultra", 32, 80, 32, 192, 800.0, 32.8),
    # M4 family
    "M4 16GB": MacSpec("M4", 10, 10, 16, 16, 120.0, 4.6),
    "M4 24GB": MacSpec("M4", 10, 10, 16, 24, 120.0, 4.6),
    "M4 32GB": MacSpec("M4", 10, 10, 16, 32, 120.0, 4.6),
    "M4 Pro 24GB": MacSpec("M4 Pro", 12, 16, 16, 24, 273.0, 7.5),
    "M4 Pro 48GB": MacSpec("M4 Pro", 12, 16, 16, 48, 273.0, 7.5),
    "M4 Pro 24GB (14C)": MacSpec("M4 Pro", 14, 20, 16, 24, 273.0, 9.4),
    "M4 Pro 48GB (14C)": MacSpec("M4 Pro", 14, 20, 16, 48, 273.0, 9.4),
    "M4 Max 36GB": MacSpec("M4 Max", 16, 40, 16, 36, 546.0, 18.7),
    "M4 Max 48GB": MacSpec("M4 Max", 16, 40, 16, 48, 546.0, 18.7),
    "M4 Max 64GB": MacSpec("M4 Max", 16, 40, 16, 64, 546.0, 18.7),
    "M4 Max 128GB": MacSpec("M4 Max", 16, 40, 16, 128, 546.0, 18.7),
    "M4 Ultra 128GB": MacSpec("M4 Ultra", 32, 80, 32, 128, 819.2, 37.4),
    "M4 Ultra 192GB": MacSpec("M4 Ultra", 32, 80, 32, 192, 819.2, 37.4),
    "M4 Ultra 256GB": MacSpec("M4 Ultra", 32, 80, 32, 256, 819.2, 37.4),
}


def _sysctl(key: str) -> str:
    """Read a sysctl value, returning empty string on failure."""
    try:
        return subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def detect_mac() -> tuple[str, MacSpec] | None:
    """Auto-detect current Mac's Apple Silicon chip and memory.

    Returns (key, MacSpec) matching APPLE_SILICON_SPECS, or None.
    """
    brand = _sysctl("machdep.cpu.brand_string")  # e.g. "Apple M4 Pro"
    if not brand.startswith("Apple "):
        return None

    chip_name = brand.removeprefix("Apple ")  # "M4 Pro"

    mem_raw = _sysctl("hw.memsize")
    if not mem_raw:
        return None
    mem_gb = round(int(mem_raw) / (1024 ** 3))

    cpu_cores_raw = _sysctl("hw.physicalcpu")
    cpu_cores = int(cpu_cores_raw) if cpu_cores_raw else 0

    # Try exact match first (standard key format)
    key = f"{chip_name} {mem_gb}GB"
    if key in APPLE_SILICON_SPECS:
        # Verify CPU core count matches — catches variant SKUs like M4 Pro 14C
        spec = APPLE_SILICON_SPECS[key]
        if spec.cpu_cores == cpu_cores:
            return key, spec

    # Try variant key with core-count suffix (e.g. "M4 Pro 48GB (14C)")
    variant_key = f"{chip_name} {mem_gb}GB ({cpu_cores}C)"
    if variant_key in APPLE_SILICON_SPECS:
        return variant_key, APPLE_SILICON_SPECS[variant_key]

    # Fallback: return standard key even if core count doesn't match exactly
    if key in APPLE_SILICON_SPECS:
        return key, APPLE_SILICON_SPECS[key]

    return None
