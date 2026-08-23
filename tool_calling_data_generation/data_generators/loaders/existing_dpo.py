"""
Data loader for pre-generated DPO datasets already in canonical format.

Loads datasets from HuggingFace Hub (or disk) and auto-detects whether the
data is **canonical format** (columns: ``messages``, ``tool_catalog``,
``chosen_response``, ``rejected_response`` ...) or **DPO format** (columns:
``prompt``, ``chosen``, ``rejected``, ``tools`` ...), converting as needed.

Canonical format is the superset — it has everything needed to generate both
SFT and DPO training datasets.
"""

import json
from typing import Optional

from datasets import load_dataset, load_from_disk

from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader
from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset


# Columns that identify the canonical format (superset).
_CANONICAL_COLUMNS = {
    "messages",
    "tool_catalog",
    "chosen_response",
    "rejected_response",
}

# Columns that identify the DPO export format.
_DPO_COLUMNS = {
    "prompt",
    "chosen",
    "rejected",
    "tools",
}


def _is_canonical(cols: set) -> bool:
    """Return True if *cols* looks like the canonical superset format."""
    return bool(cols & _CANONICAL_COLUMNS)


def _is_dpo(cols: set) -> bool:
    """Return True if *cols* looks like the DPO export format."""
    return bool(cols & _DPO_COLUMNS)


def _dpo_to_canonical(ds) -> object:
    """Convert a DPO-format HF dataset to canonical columns in-place."""

    def _map_row(row):
        chosen = _to_json(row.get("chosen"))
        rejected = _to_json(row.get("rejected"))
        tools = _to_json(row.get("tools", []), default=[])
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                metadata = {}

        return {
            "conv_id": row.get("conv_id", ""),
            "turn_id": "0",
            "conv_tracking_id": row.get("conv_tracking_id", row.get("conv_id", "")),
            "custom_system_prompt": None,
            "is_custom_system_prompt_used": False,
            "messages": row.get("prompt", [])
            if isinstance(row.get("prompt"), list)
            else [],
            "tool_catalog": json.dumps(tools) if isinstance(tools, list) else tools,
            "chosen_response": chosen,
            "rejected_response": rejected,
            "metadata": json.dumps(metadata),
        }

    return ds.map(_map_row, remove_columns=ds.column_names)


def _to_json(val, default="[]"):
    """Serialize a value to a JSON string, falling back to *default*."""
    if val is None:
        return default
    if isinstance(val, str):
        try:
            json.loads(val)
            return val
        except (json.JSONDecodeError, ValueError):
            return json.dumps(val)
    if isinstance(val, list):
        return json.dumps(val)
    if isinstance(val, dict):
        return json.dumps(val)
    return default


class ExistingDPOLoader(BaseDataLoader):
    """Load a pre-generated DPO dataset from HuggingFace Hub or disk.

    The dataset may be in one of two formats (auto-detected):

    * **Canonical format** (superset) — columns matching
      ``CanonicalSample`` schema: ``messages``, ``tool_catalog``,
      ``chosen_response``, ``rejected_response``, ...
    * **DPO export format** — columns from ``CanonicalDataset.to_dpo_dataset()``:
      ``prompt``, ``chosen``, ``rejected``, ``tools``, ...

    The result is always a ``CanonicalDataset`` regardless of input format.

    Parameters
    ----------
    dataset_id:
        HuggingFace dataset repo ID or local path on disk.
    split:
        Dataset split to load (default ``"train"``).
    max_samples:
        Maximum number of rows to load (``None`` = all).
    """

    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        max_samples: int | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id=dataset_id,
            split=split,
            config=config,
            max_samples=max_samples,
            **kwargs,
        )

    def load(self) -> CanonicalDataset:
        dataset_id = self._kwargs["dataset_id"]
        split = self._kwargs.get("split", "train")
        max_samples = self._kwargs.get("max_samples")
        config = self._kwargs.get("config")
        ds = load_dataset(path=dataset_id, name=config, split=split)

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        cols = set(ds.column_names)

        if _is_canonical(cols):
            # Already in canonical format — zero-copy wrap.
            return CanonicalDataset.from_hf_dataset(ds)

        if _is_dpo(cols):
            # Convert from DPO export format to canonical.
            ds = _dpo_to_canonical(ds)
            return CanonicalDataset.from_hf_dataset(ds)

        raise ValueError(
            f"Cannot detect format for dataset {dataset_id!r}. "
            f"Columns: {sorted(cols)}. "
            f"Expected canonical columns ({_CANONICAL_COLUMNS}) or "
            f"DPO columns ({_DPO_COLUMNS})."
        )
