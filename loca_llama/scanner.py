"""Scan for locally downloaded LLM models from LM Studio and llama.cpp."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LocalModel:
    """A model file found on disk."""

    name: str
    path: Path
    size_gb: float
    format: str  # "gguf", "mlx", "safetensors", "bin"
    source: str  # "lm-studio", "llama.cpp", "huggingface", "mlx-community", "custom"
    quant: str | None = None  # Detected quantization from filename
    family: str | None = None  # Detected model family
    repo_id: str | None = None  # HuggingFace repo id if detectable
    metadata: dict = field(default_factory=dict)


# Known LM Studio model directories (macOS)
LM_STUDIO_PATHS = [
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / ".lmstudio" / "models",
]

# Known llama.cpp model directory
LLAMA_CPP_PATHS = [
    Path.home() / "llama.cpp" / "models",
    Path.home() / ".local" / "share" / "llama.cpp" / "models",
]

# HuggingFace cache
HUGGINGFACE_PATHS = [
    Path.home() / ".cache" / "huggingface" / "hub",
]

# Quantization patterns in filenames
QUANT_PATTERN = re.compile(
    r"(Q[2-8]_[KS0](?:_[SML])?|F16|FP16|FP32|BF16|q4_0|q4_1|q5_0|q5_1|q8_0|IQ[1-4]_[A-Z]+)",
    re.IGNORECASE,
)

# Model family patterns
FAMILY_PATTERNS = {
    "Llama": re.compile(r"llama[-_]?3?\.?[0-9]*", re.IGNORECASE),
    "Mistral": re.compile(r"mistral", re.IGNORECASE),
    "Mixtral": re.compile(r"mixtral", re.IGNORECASE),
    "Phi": re.compile(r"phi[-_]?[234]", re.IGNORECASE),
    "Gemma": re.compile(r"gemma", re.IGNORECASE),
    "Qwen": re.compile(r"qwen", re.IGNORECASE),
    "DeepSeek": re.compile(r"deep[-_]?seek", re.IGNORECASE),
    "Command": re.compile(r"command[-_]?r", re.IGNORECASE),
    "CodeLlama": re.compile(r"code[-_]?llama", re.IGNORECASE),
    "Yi": re.compile(r"\byi[-_]", re.IGNORECASE),
    "StarCoder": re.compile(r"star[-_]?coder", re.IGNORECASE),
    "Falcon": re.compile(r"falcon", re.IGNORECASE),
}


def detect_quant(filename: str) -> str | None:
    """Extract quantization format from filename."""
    match = QUANT_PATTERN.search(filename)
    return match.group(1).upper() if match else None


def detect_family(name: str) -> str | None:
    """Detect model family from name or path."""
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.search(name):
            return family
    return None


def file_size_gb(path: Path) -> float:
    """Get file size in GB."""
    try:
        return path.stat().st_size / (1024**3)
    except OSError:
        return 0.0


def dir_size_gb(path: Path) -> float:
    """Get total size of a directory in GB."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total / (1024**3)


def scan_gguf_files(directory: Path, source: str) -> list[LocalModel]:
    """Scan a directory for GGUF model files."""
    models = []
    if not directory.exists():
        return models

    for gguf_file in directory.rglob("*.gguf"):
        name = gguf_file.stem
        size = file_size_gb(gguf_file)
        quant = detect_quant(name)
        family = detect_family(str(gguf_file))

        # Try to extract repo_id from path structure
        repo_id = None
        rel = None
        try:
            rel = gguf_file.relative_to(directory)
        except ValueError:
            pass
        if rel and len(rel.parts) >= 2:
            repo_id = f"{rel.parts[0]}/{rel.parts[1]}"

        models.append(LocalModel(
            name=name,
            path=gguf_file,
            size_gb=size,
            format="gguf",
            source=source,
            quant=quant,
            family=family,
            repo_id=repo_id,
        ))

    return models


def scan_mlx_models(directory: Path) -> list[LocalModel]:
    """Scan HuggingFace cache for MLX format models."""
    models = []
    if not directory.exists():
        return models

    # HuggingFace cache structure: models--<org>--<model>/snapshots/<hash>/
    for model_dir in directory.iterdir():
        if not model_dir.name.startswith("models--"):
            continue

        parts = model_dir.name.replace("models--", "").split("--", 1)
        if len(parts) != 2:
            continue
        org, model_name = parts

        is_mlx = "mlx" in org.lower() or "mlx" in model_name.lower()

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshot_dirs:
            continue

        latest = snapshot_dirs[0]

        has_safetensors = any(latest.glob("*.safetensors"))
        has_gguf = any(latest.glob("*.gguf"))

        if not (has_safetensors or has_gguf):
            continue

        if has_gguf:
            for gguf in latest.glob("*.gguf"):
                models.append(LocalModel(
                    name=gguf.stem,
                    path=gguf,
                    size_gb=file_size_gb(gguf),
                    format="gguf",
                    source="huggingface",
                    quant=detect_quant(gguf.name),
                    family=detect_family(model_name),
                    repo_id=f"{org}/{model_name}",
                ))
        elif has_safetensors:
            fmt = "mlx" if is_mlx else "safetensors"
            source = "mlx-community" if is_mlx else "huggingface"
            models.append(LocalModel(
                name=model_name,
                path=latest,
                size_gb=dir_size_gb(latest),
                format=fmt,
                source=source,
                quant=detect_quant(model_name),
                family=detect_family(model_name),
                repo_id=f"{org}/{model_name}",
            ))

    return models


def scan_lmstudio_mlx(directory: Path) -> list[LocalModel]:
    """Scan LM Studio directory for MLX/safetensors models.

    LM Studio stores MLX models in flat directories:
    models/<org>/<model-name>/*.safetensors
    """
    models: list[LocalModel] = []
    if not directory.exists():
        return models

    for org_dir in directory.iterdir():
        if not org_dir.is_dir():
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if not any(model_dir.glob("*.safetensors")):
                continue
            is_mlx = "mlx" in org_dir.name.lower() or "mlx" in model_dir.name.lower()
            models.append(LocalModel(
                name=model_dir.name,
                path=model_dir,
                size_gb=dir_size_gb(model_dir),
                format="mlx" if is_mlx else "safetensors",
                source="lm-studio",
                quant=detect_quant(model_dir.name),
                family=detect_family(model_dir.name),
                repo_id=f"{org_dir.name}/{model_dir.name}",
            ))

    return models


def scan_all() -> list[LocalModel]:
    """Scan all known locations for downloaded models."""
    all_models: list[LocalModel] = []

    for path in LM_STUDIO_PATHS:
        all_models.extend(scan_gguf_files(path, "lm-studio"))
        all_models.extend(scan_lmstudio_mlx(path))

    for path in LLAMA_CPP_PATHS:
        all_models.extend(scan_gguf_files(path, "llama.cpp"))

    for path in HUGGINGFACE_PATHS:
        all_models.extend(scan_mlx_models(path))

    # Deduplicate by path
    seen_paths: set[Path] = set()
    unique: list[LocalModel] = []
    for m in all_models:
        if m.path not in seen_paths:
            seen_paths.add(m.path)
            unique.append(m)

    unique.sort(key=lambda m: m.size_gb, reverse=True)
    return unique


def scan_custom_dir(directory: str) -> list[LocalModel]:
    """Scan a custom directory for model files."""
    path = Path(directory).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return []

    models: list[LocalModel] = []
    models.extend(scan_gguf_files(path, "custom"))

    for st_file in path.rglob("*.safetensors"):
        parent = st_file.parent
        if parent in {m.path for m in models}:
            continue
        is_mlx = "mlx" in str(parent).lower()
        models.append(LocalModel(
            name=parent.name,
            path=parent,
            size_gb=dir_size_gb(parent),
            format="mlx" if is_mlx else "safetensors",
            source="mlx-community" if is_mlx else "custom",
            quant=detect_quant(parent.name),
            family=detect_family(parent.name),
        ))

    seen_paths: set[Path] = set()
    unique: list[LocalModel] = []
    for m in models:
        if m.path not in seen_paths:
            seen_paths.add(m.path)
            unique.append(m)

    unique.sort(key=lambda m: m.size_gb, reverse=True)
    return unique
