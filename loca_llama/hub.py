"""Search and browse models from HuggingFace and MLX Community."""

import json
import logging
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HubModel:
    """A model available on HuggingFace Hub."""

    repo_id: str
    name: str
    author: str
    downloads: int
    likes: int
    tags: list[str]
    pipeline_tag: str | None
    is_mlx: bool
    is_gguf: bool
    last_modified: str | None = None


HF_API = "https://huggingface.co/api/models"

_SORT_ALLOWLIST = {"downloads", "likes", "trending", "lastModified", "created"}


def search_huggingface(
    query: str,
    limit: int = 20,
    sort: str = "downloads",
    filter_tags: list[str] | None = None,
) -> list[HubModel]:
    """Search HuggingFace Hub for models.

    Args:
        query: Search query (e.g. "llama 8b gguf")
        limit: Max number of results
        sort: Sort by "downloads", "likes", "trending", "lastModified", or "created"
        filter_tags: Optional tag filters (e.g. ["gguf", "text-generation"])
    """
    if sort not in _SORT_ALLOWLIST:
        raise ValueError(f"sort must be one of {sorted(_SORT_ALLOWLIST)}, got {sort!r}")

    params = {
        "search": query,
        "limit": str(limit),
        "sort": sort,
        "direction": "-1",
    }
    if filter_tags:
        params["filter"] = ",".join(filter_tags)

    url = f"{HF_API}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "loca-llama/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        logger.warning("HuggingFace search network error: %s", e)
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("HuggingFace search data error: %s", e)
        return []

    models = []
    for item in data:
        repo_id = item.get("modelId", item.get("id", ""))
        tags = item.get("tags", [])
        author = repo_id.split("/")[0] if "/" in repo_id else ""
        name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        models.append(HubModel(
            repo_id=repo_id,
            name=name,
            author=author,
            downloads=item.get("downloads", 0),
            likes=item.get("likes", 0),
            tags=tags,
            pipeline_tag=item.get("pipeline_tag"),
            is_mlx="mlx" in " ".join(tags).lower() or "mlx" in author.lower(),
            is_gguf="gguf" in " ".join(tags).lower() or "gguf" in name.lower(),
            last_modified=item.get("lastModified"),
        ))

    return models


def search_gguf_models(query: str, limit: int = 20) -> list[HubModel]:
    """Search specifically for GGUF format models."""
    return search_huggingface(query, limit=limit, filter_tags=["gguf"])


def search_mlx_models(query: str, limit: int = 20) -> list[HubModel]:
    """Search specifically for MLX format models."""
    results = search_huggingface(query, limit=limit)
    # Filter to MLX models
    return [m for m in results if m.is_mlx]


def get_model_files(repo_id: str) -> list[dict]:
    """Get list of files in a HuggingFace model repo."""
    url = f"{HF_API}/{repo_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "loca-llama/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        logger.warning("HuggingFace files network error for %s: %s", repo_id, e)
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("HuggingFace files data error for %s: %s", repo_id, e)
        return []

    siblings = data.get("siblings", [])
    return [
        {"filename": s.get("rfilename", ""), "size": s.get("size")}
        for s in siblings
    ]


def format_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
