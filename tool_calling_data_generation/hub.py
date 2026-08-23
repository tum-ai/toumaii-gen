"""Explicit, private-only Hugging Face Hub upload support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import RepositoryNotFoundError

from tool_calling_data_generation.config import ConfigError


def _assert_repo_owner_visible(identity: dict[str, Any], repo_id: str) -> None:
    owner = repo_id.split("/", 1)[0]
    visible = {identity.get("name")}
    visible.update(
        org.get("name") if isinstance(org, dict) else str(org)
        for org in identity.get("orgs", [])
    )
    if owner not in visible:
        raise PermissionError(
            f"Authenticated as {identity.get('name')!r}, but Hub owner/org {owner!r} "
            "is not visible to this token."
        )


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"generated", "uploaded"}:
        raise ValueError(f"Unsupported manifest status: {manifest.get('status')!r}")
    return manifest_path, manifest


def upload_private_run(config: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """Upload one generated run and verify that the target dataset is private."""
    hub = config["hub"]
    repo_id = hub.get("repo_id")
    if not repo_id:
        raise ConfigError("hub.repo_id is required for upload.")
    if hub.get("private") is not True:
        raise ConfigError("Refusing upload because hub.private is not true.")

    token = get_token()
    if not token:
        raise PermissionError("No Hugging Face token found. Run `hf auth login` or set HF_TOKEN.")

    manifest_file, manifest = _load_manifest(manifest_path)
    api = HfApi(token=token)
    identity = api.whoami()
    _assert_repo_owner_visible(identity, repo_id)

    try:
        info = api.dataset_info(repo_id=repo_id, token=token)
        if info.private is not True:
            raise PermissionError(
                f"Refusing to upload: existing dataset {repo_id!r} is not private."
            )
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=False,
            token=token,
        )

    info = api.dataset_info(repo_id=repo_id, token=token)
    if info.private is not True:
        raise PermissionError(f"Dataset {repo_id!r} failed the private-repository preflight.")

    artifacts = manifest["artifacts"]
    uploads = [
        ("canonical_parquet", hub.get("canonical_config", "canonical")),
        ("dpo_parquet", hub.get("dpo_config", "dpo")),
        ("sft_parquet", hub.get("sft_config", "sft")),
    ]
    uploaded: dict[str, int] = {}
    for artifact_key, config_name in uploads:
        dataset = Dataset.from_parquet(artifacts[artifact_key])
        dataset.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            split=hub.get("split", "train"),
            private=True,
            token=token,
        )
        uploaded[config_name] = len(dataset)

    final_info = api.dataset_info(repo_id=repo_id, token=token)
    if final_info.private is not True:
        raise RuntimeError(f"Post-upload privacy verification failed for {repo_id!r}.")

    manifest["status"] = "uploaded"
    manifest["hub"] = {
        "repo_id": repo_id,
        "private": True,
        "authenticated_as": identity.get("name"),
        "rows_by_config": uploaded,
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest["hub"]

