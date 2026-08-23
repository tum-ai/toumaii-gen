"""Configuration loading and validation for the standalone pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is incomplete or unsafe."""


DEFAULT_CONFIG: dict[str, Any] = {
    "source": {
        "type": "toucan",
        "toucan": {
            "dataset_id": "Agent-Ark/Toucan-1.5M",
            "dataset_revision": None,
            "dataset_name": "Qwen3",
            "size": 2,
            "access_mode": "full",
            "viewer_offset": 0,
            "viewer_page_size": 20,
            "viewer_max_rows": 100,
            "subset_name": "multi-turn",
            "toolcall_mode": "multi",
            "overall_score_threshold": 4.5,
            "desired_tools_used_percentage": 1.0,
            "order_correctness_required": True,
            "use_disk_cache": True,
        },
    },
    "generation": {
        "model_id": None,
        "backend": "auto",
        "use_chat_template": True,
        "use_disk_cache": True,
        "model_ctx": 16384,
        "gpu_memory_utilization": 0.8,
        "temperature": 0.0,
        "max_tokens": 512,
        "annotate": True,
        "backend_kwargs": {},
    },
    "output": {
        "root_dir": "outputs",
        "write_jsonl": True,
        "write_csv_preview": True,
    },
    "hub": {
        "repo_id": None,
        "private": True,
        "split": "train",
        "canonical_config": "canonical",
        "dpo_config": "dpo",
        "sft_config": "sft",
    },
}

_BACKENDS = {"auto", "transformers", "vllm", "vllm-parallel"}
_SOURCES = {"toucan", "toolace", "existing_dpo"}
_SECRET_KEYS = {"token", "hf_token", "api_token", "access_token"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_secret_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _SECRET_KEYS:
                found.append(path)
            found.extend(_find_secret_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_secret_keys(child, f"{prefix}[{index}]"))
    return found


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a merged, validated copy of *config*.

    Authentication values are deliberately rejected in YAML. Use `hf auth
    login` or the `HF_TOKEN` environment variable instead.
    """
    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")

    secret_paths = _find_secret_keys(config)
    if secret_paths:
        raise ConfigError(
            "Do not store tokens in config files; use `hf auth login` or HF_TOKEN. "
            f"Secret-like keys: {', '.join(secret_paths)}"
        )

    merged = _merge(DEFAULT_CONFIG, config)
    source_type = merged["source"].get("type")
    if source_type not in _SOURCES:
        raise ConfigError(f"source.type must be one of {sorted(_SOURCES)}; got {source_type!r}")

    model_id = merged["generation"].get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("generation.model_id must be a non-empty model ID or local path.")

    backend = merged["generation"].get("backend")
    if backend not in _BACKENDS:
        raise ConfigError(
            f"generation.backend must be one of {sorted(_BACKENDS)}; got {backend!r}"
        )

    backend_kwargs = merged["generation"].get("backend_kwargs")
    if not isinstance(backend_kwargs, dict):
        raise ConfigError("generation.backend_kwargs must be a mapping.")
    reserved = {"temperature", "max_tokens", "gpu_memory_utilization", "model_ctx"}
    overlap = reserved.intersection(backend_kwargs)
    if overlap:
        raise ConfigError(
            "Set reserved generation values at generation.<name>, not in backend_kwargs: "
            f"{', '.join(sorted(overlap))}"
        )

    size = merged["source"].get(source_type, {}).get("size")
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ConfigError(f"source.{source_type}.size must be a positive integer or null.")
    if source_type == "toucan":
        access_mode = merged["source"]["toucan"].get("access_mode")
        if access_mode not in {"full", "streaming", "viewer"}:
            raise ConfigError(
                "source.toucan.access_mode must be one of: full, streaming, viewer."
            )
        if access_mode in {"streaming", "viewer"} and size is None:
            raise ConfigError(f"Toucan {access_mode} access requires a finite size.")

    max_tokens = merged["generation"].get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ConfigError("generation.max_tokens must be a positive integer.")

    root_dir = merged["output"].get("root_dir")
    if not isinstance(root_dir, str) or not root_dir.strip():
        raise ConfigError("output.root_dir must be a non-empty path.")

    repo_id = merged["hub"].get("repo_id")
    if repo_id is not None and (not isinstance(repo_id, str) or repo_id.count("/") != 1):
        raise ConfigError("hub.repo_id must have the form '<owner-or-org>/<dataset>'.")
    if merged["hub"].get("private") is not True:
        raise ConfigError("This extraction only permits private Hub uploads: hub.private must be true.")

    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return validate_config(raw)
