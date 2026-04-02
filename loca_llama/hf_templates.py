"""Fetch model settings and templates from HuggingFace.

Downloads generation_config.json, config.json, and parses model card metadata
to extract official recommended settings (temperature, top_p, context length, etc.)
from model authors — complementing our static templates with live data.
"""

import json
import logging
import re
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


HF_RAW_URL = "https://huggingface.co/{repo_id}/raw/main/{filename}"
HF_API_URL = "https://huggingface.co/api/models/{repo_id}"

_REPO_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def _validate_repo_id(repo_id: str) -> None:
    """Raise ValueError if repo_id doesn't match the expected owner/repo format."""
    if not _REPO_ID_RE.match(repo_id):
        raise ValueError(f"repo_id must match pattern owner/repo (alphanumeric, dots, dashes, underscores), got {repo_id!r}")


@dataclass
class HFModelConfig:
    """Configuration fetched from HuggingFace for a model."""

    repo_id: str
    model_type: str = ""
    architecture: str = ""

    # From config.json
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    max_position_embeddings: int = 0  # model's max context
    intermediate_size: int = 0
    rope_theta: float = 0.0
    tie_word_embeddings: bool = False

    # From generation_config.json (official recommended settings)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    do_sample: bool | None = None
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None

    # From model card / tokenizer_config.json
    chat_template: str | None = None  # Jinja2 template
    system_prompt: str | None = None

    # Metadata
    license: str = ""
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""
    author: str = ""

    # Raw data for debugging
    _raw_config: dict = field(default_factory=dict, repr=False)
    _raw_generation_config: dict = field(default_factory=dict, repr=False)


def _fetch_json(url: str, timeout: int = 10) -> dict | None:
    """Fetch and parse JSON from a URL.

    Lets urllib.error.URLError propagate so callers (route handlers) can
    distinguish network failures from missing/malformed files.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "loca-llama/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # File not present in repo — not a network error, treat as absent
            logger.debug("_fetch_json 404 for %s", url)
            return None
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("_fetch_json data error for %s: %s", url, e)
        return None


def _fetch_text(url: str, timeout: int = 10) -> str | None:
    """Fetch text content from a URL.

    Lets urllib.error.URLError propagate so callers (route handlers) can
    distinguish network failures from missing/malformed files.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "loca-llama/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("_fetch_text 404 for %s", url)
            return None
        raise
    except (ValueError, OSError) as e:
        logger.warning("_fetch_text data error for %s: %s", url, e)
        return None


def fetch_config_json(repo_id: str) -> dict | None:
    """Fetch config.json from a HuggingFace repo."""
    url = HF_RAW_URL.format(repo_id=repo_id, filename="config.json")
    return _fetch_json(url)


def fetch_generation_config(repo_id: str) -> dict | None:
    """Fetch generation_config.json from a HuggingFace repo."""
    url = HF_RAW_URL.format(repo_id=repo_id, filename="generation_config.json")
    return _fetch_json(url)


def fetch_tokenizer_config(repo_id: str) -> dict | None:
    """Fetch tokenizer_config.json for chat template info."""
    url = HF_RAW_URL.format(repo_id=repo_id, filename="tokenizer_config.json")
    return _fetch_json(url)


def fetch_model_card(repo_id: str) -> str | None:
    """Fetch the model card (README.md) text."""
    url = HF_RAW_URL.format(repo_id=repo_id, filename="README.md")
    return _fetch_text(url)


def fetch_model_api_info(repo_id: str) -> dict | None:
    """Fetch model info from HuggingFace API."""
    url = HF_API_URL.format(repo_id=repo_id)
    return _fetch_json(url)


# ── Parsing ─────────────────────────────────────────────────────────────────

def _parse_config_json(config: dict, result: HFModelConfig) -> None:
    """Extract architecture info from config.json."""
    result._raw_config = config
    result.model_type = config.get("model_type", "")
    result.architecture = ", ".join(config.get("architectures", []))
    result.vocab_size = config.get("vocab_size", 0)
    result.hidden_size = config.get("hidden_size", 0)
    result.intermediate_size = config.get("intermediate_size", 0)
    result.tie_word_embeddings = config.get("tie_word_embeddings", False)

    # Layer count (different keys for different architectures)
    result.num_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 0
    )

    # Attention heads
    result.num_attention_heads = (
        config.get("num_attention_heads")
        or config.get("n_head")
        or 0
    )
    result.num_kv_heads = (
        config.get("num_key_value_heads")
        or config.get("num_kv_heads")
        or result.num_attention_heads  # GQA: default to MHA
    )

    # Head dimension
    if result.hidden_size and result.num_attention_heads:
        result.head_dim = config.get("head_dim", result.hidden_size // result.num_attention_heads)

    # Context length
    result.max_position_embeddings = (
        config.get("max_position_embeddings")
        or config.get("max_seq_len")
        or config.get("n_positions")
        or config.get("seq_length")
        or 0
    )

    # RoPE
    result.rope_theta = config.get("rope_theta", 0.0)
    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        # Some models store extended context info here
        factor = rope_scaling.get("factor", 1.0)
        if factor > 1 and result.max_position_embeddings:
            pass  # max_position_embeddings already reflects this


def _parse_generation_config(gen_config: dict, result: HFModelConfig) -> None:
    """Extract recommended generation settings from generation_config.json."""
    result._raw_generation_config = gen_config

    if "temperature" in gen_config:
        result.temperature = gen_config["temperature"]
    if "top_p" in gen_config:
        result.top_p = gen_config["top_p"]
    if "top_k" in gen_config:
        result.top_k = gen_config["top_k"]
    if "repetition_penalty" in gen_config:
        result.repetition_penalty = gen_config["repetition_penalty"]
    if "max_new_tokens" in gen_config:
        result.max_new_tokens = gen_config["max_new_tokens"]
    if "do_sample" in gen_config:
        result.do_sample = gen_config["do_sample"]
    if "eos_token_id" in gen_config:
        result.eos_token_id = gen_config["eos_token_id"]
    if "bos_token_id" in gen_config:
        result.bos_token_id = gen_config["bos_token_id"]


def _parse_tokenizer_config(tok_config: dict, result: HFModelConfig) -> None:
    """Extract chat template from tokenizer_config.json."""
    if "chat_template" in tok_config:
        result.chat_template = tok_config["chat_template"]


def _parse_model_card_yaml(card_text: str, result: HFModelConfig) -> None:
    """Extract YAML metadata from model card header."""
    # Model cards have YAML frontmatter between --- markers
    match = re.match(r"^---\s*\n(.*?)\n---", card_text, re.DOTALL)
    if not match:
        return

    yaml_block = match.group(1)
    # Simple YAML parsing (avoid pyyaml dependency)
    for line in yaml_block.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()

        if key == "license":
            result.license = val
        elif key == "pipeline_tag":
            result.pipeline_tag = val


def _extract_card_recommendations(card_text: str, result: HFModelConfig) -> None:
    """Try to extract recommended settings from model card prose.

    Many model cards include recommended settings in sections like
    "Recommended Settings", "Inference Parameters", etc.
    """
    # Look for temperature recommendations
    temp_match = re.search(
        r"(?:recommend|suggest|use|set)\w*\s+(?:a\s+)?temperature\s*(?:of|to|=|:)?\s*([\d.]+)",
        card_text, re.IGNORECASE,
    )
    if temp_match and result.temperature is None:
        try:
            result.temperature = float(temp_match.group(1))
        except ValueError:
            pass

    # Look for top_p recommendations
    top_p_match = re.search(
        r"top[_-]?p\s*(?:of|to|=|:)?\s*([\d.]+)",
        card_text, re.IGNORECASE,
    )
    if top_p_match and result.top_p is None:
        try:
            result.top_p = float(top_p_match.group(1))
        except ValueError:
            pass

    # Look for context length mentions
    ctx_match = re.search(
        r"(?:context|sequence)\s*(?:length|window|size)\s*(?:of|:)?\s*([\d,]+)\s*(?:tokens)?",
        card_text, re.IGNORECASE,
    )
    if ctx_match and not result.max_position_embeddings:
        try:
            result.max_position_embeddings = int(ctx_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Look for system prompt in code blocks
    sys_prompt_match = re.search(
        r'(?:system[_ ]?prompt|system[_ ]?message)\s*(?:=|:)\s*["\'](.+?)["\']',
        card_text, re.IGNORECASE,
    )
    if sys_prompt_match and result.system_prompt is None:
        result.system_prompt = sys_prompt_match.group(1)


# ── Main Fetch Function ────────────────────────────────────────────────────

def fetch_hf_model_config(
    repo_id: str,
    fetch_card: bool = True,
    progress_callback=None,
) -> HFModelConfig:
    """Fetch all available configuration from HuggingFace for a model.

    This tries to download:
    1. config.json — architecture info
    2. generation_config.json — official recommended sampling params
    3. tokenizer_config.json — chat template
    4. README.md — model card with metadata and recommendations

    Args:
        repo_id: HuggingFace repo ID (e.g. "Qwen/Qwen2.5-32B-Instruct")
        fetch_card: Whether to also parse the model card (slower)
        progress_callback: Optional callback(step, total, description)
    """
    _validate_repo_id(repo_id)
    result = HFModelConfig(repo_id=repo_id)
    total_steps = 5 if fetch_card else 4

    # Step 1: API info
    if progress_callback:
        progress_callback(1, total_steps, "Fetching model info...")
    api_info = fetch_model_api_info(repo_id)
    if api_info:
        result.tags = api_info.get("tags", [])
        result.pipeline_tag = api_info.get("pipeline_tag", "")
        result.author = repo_id.split("/")[0] if "/" in repo_id else ""

    # Step 2: config.json
    if progress_callback:
        progress_callback(2, total_steps, "Fetching config.json...")
    config = fetch_config_json(repo_id)
    if config:
        _parse_config_json(config, result)

    # Step 3: generation_config.json
    if progress_callback:
        progress_callback(3, total_steps, "Fetching generation_config.json...")
    gen_config = fetch_generation_config(repo_id)
    if gen_config:
        _parse_generation_config(gen_config, result)

    # Step 4: tokenizer_config.json
    if progress_callback:
        progress_callback(4, total_steps, "Fetching tokenizer_config.json...")
    tok_config = fetch_tokenizer_config(repo_id)
    if tok_config:
        _parse_tokenizer_config(tok_config, result)

    # Step 5: Model card
    if fetch_card:
        if progress_callback:
            progress_callback(5, total_steps, "Parsing model card...")
        card = fetch_model_card(repo_id)
        if card:
            _parse_model_card_yaml(card, result)
            _extract_card_recommendations(card, result)

    return result


def resolve_repo_id(model_name: str) -> str | None:
    """Try to resolve a model name to a HuggingFace repo ID.

    Handles common patterns:
    - "Qwen2.5-32B-Instruct" -> searches for "Qwen/Qwen2.5-32B-Instruct"
    - Direct repo IDs pass through: "Qwen/Qwen2.5-32B-Instruct"
    """
    # If it already looks like a repo ID
    if "/" in model_name:
        return model_name

    # Try common author prefixes
    common_authors = {
        "llama": "meta-llama",
        "qwen": "Qwen",
        "deepseek": "deepseek-ai",
        "mistral": "mistralai",
        "mixtral": "mistralai",
        "phi": "microsoft",
        "gemma": "google",
        "command": "CohereForAI",
        "starcoder": "bigcode",
        "codellama": "meta-llama",
        "falcon": "tiiuae",
        "yi": "01-ai",
    }

    name_lower = model_name.lower()
    for pattern, author in common_authors.items():
        if pattern in name_lower:
            # Try the resolved name
            candidate = f"{author}/{model_name}"
            # Verify it exists
            info = fetch_model_api_info(candidate)
            if info and "modelId" in info:
                return candidate

    # Fall back to search
    from .hub import search_huggingface
    results = search_huggingface(model_name, limit=5)
    if results:
        # Return the top result
        return results[0].repo_id

    return None


# ── Display Helpers ─────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def format_hf_config(config: HFModelConfig) -> str:
    """Format HF config for terminal display."""
    lines = []
    lines.append(f"  {BOLD}{CYAN}HuggingFace Model Configuration{RESET}")
    lines.append(f"  {BOLD}Repo:{RESET}          {config.repo_id}")

    if config.architecture:
        lines.append(f"  {BOLD}Architecture:{RESET}  {config.architecture}")
    if config.model_type:
        lines.append(f"  {BOLD}Model Type:{RESET}    {config.model_type}")
    if config.license:
        lines.append(f"  {BOLD}License:{RESET}       {config.license}")

    # Architecture details
    if config.num_layers:
        lines.append(f"\n  {BOLD}Architecture Details:{RESET}")
        lines.append(f"    Layers:          {config.num_layers}")
        lines.append(f"    Attention Heads:  {config.num_attention_heads}")
        if config.num_kv_heads != config.num_attention_heads:
            lines.append(f"    KV Heads (GQA):  {config.num_kv_heads}")
        lines.append(f"    Hidden Size:     {config.hidden_size}")
        if config.head_dim:
            lines.append(f"    Head Dim:        {config.head_dim}")
        if config.max_position_embeddings:
            ctx_k = config.max_position_embeddings // 1024
            lines.append(f"    Max Context:     {ctx_k}K tokens ({config.max_position_embeddings:,})")

    # Generation settings (the key info users want)
    has_gen = any([config.temperature, config.top_p, config.top_k,
                   config.repetition_penalty])
    if has_gen:
        lines.append(f"\n  {BOLD}{GREEN}Official Recommended Settings:{RESET}")
        if config.temperature is not None:
            lines.append(f"    Temperature:      {config.temperature}")
        if config.top_p is not None:
            lines.append(f"    Top P:            {config.top_p}")
        if config.top_k is not None:
            lines.append(f"    Top K:            {config.top_k}")
        if config.repetition_penalty is not None:
            lines.append(f"    Repeat Penalty:   {config.repetition_penalty}")
        if config.max_new_tokens is not None:
            lines.append(f"    Max New Tokens:   {config.max_new_tokens}")
        if config.do_sample is not None:
            lines.append(f"    Do Sample:        {config.do_sample}")
    else:
        lines.append(f"\n  {DIM}No generation_config.json found — using model defaults.{RESET}")

    if config.system_prompt:
        lines.append(f"\n  {BOLD}System Prompt:{RESET}")
        lines.append(f"    {DIM}{config.system_prompt}{RESET}")

    return "\n".join(lines)
