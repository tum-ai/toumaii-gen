"""Standalone generation orchestration and local artifact export."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from tool_calling_data_generation import __version__
from tool_calling_data_generation.config import ConfigError
from tool_calling_data_generation.data_generators.backends.base import BaseGenerator
from tool_calling_data_generation.data_generators.dpo_generator import build_dpo_dataset
from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: str
    manifest_path: str
    canonical_parquet: str
    dpo_parquet: str
    sft_parquet: str
    num_canonical_rows: int
    num_dpo_rows: int
    num_sft_rows: int


def resolve_loader(config: dict[str, Any]) -> BaseDataLoader:
    source = config["source"]
    source_type = source["type"]
    params = dict(source.get(source_type, {}))

    if source_type == "toucan":
        from tool_calling_data_generation.data_generators.loaders.toucan import ToucanLoader

        return ToucanLoader(
            dataset_id=params.get("dataset_id", "Agent-Ark/Toucan-1.5M"),
            dataset_revision=params.get("dataset_revision"),
            size=params.get("size"),
            streaming=params.get("access_mode", "full") == "streaming",
            use_viewer_api=params.get("access_mode", "full") == "viewer",
            viewer_offset=params.get("viewer_offset", 0),
            viewer_page_size=params.get("viewer_page_size", 20),
            viewer_max_rows=params.get("viewer_max_rows", 100),
            dataset_name=params.get("dataset_name", "Qwen3"),
            subset_name_required=params.get("subset_name", "multi-turn"),
            toolcall_mode=params.get("toolcall_mode", "single"),
            overall_score_threshold=params.get("overall_score_threshold", 4.5),
            desired_tools_used_percentage_threshold=params.get(
                "desired_tools_used_percentage", 1.0
            ),
            order_correctness_required=params.get("order_correctness_required", True),
            use_disk_cache=params.get("use_disk_cache", True),
        )

    if source_type == "toolace":
        from tool_calling_data_generation.data_generators.loaders.toolace import ToolACELoader

        return ToolACELoader(
            size=params.get("size"),
            use_disk_cache=params.get("use_disk_cache", True),
        )

    if source_type == "existing_dpo":
        from tool_calling_data_generation.data_generators.loaders.existing_dpo import (
            ExistingDPOLoader,
        )

        if not params.get("dataset_id"):
            raise ConfigError("source.existing_dpo.dataset_id is required.")
        return ExistingDPOLoader(
            dataset_id=params["dataset_id"],
            config=params.get("config"),
            split=params.get("split", "train"),
            max_samples=params.get("size"),
        )

    raise ConfigError(f"Unsupported source type: {source_type!r}")


def resolve_backend(config: dict[str, Any]) -> BaseGenerator | None:
    generation = config["generation"]
    backend_name = generation["backend"]
    model_id = generation["model_id"]
    if backend_name == "auto":
        return None
    if backend_name == "transformers":
        from tool_calling_data_generation.data_generators.backends.trl import TRLGenerator

        return TRLGenerator(model_id)
    if backend_name == "vllm":
        from tool_calling_data_generation.data_generators.backends.vllm import vLLMGenerator

        return vLLMGenerator(model_id)
    if backend_name == "vllm-parallel":
        from tool_calling_data_generation.data_generators.backends.vllm_parallel import (
            vLLMParallelGenerator,
        )

        backend_kwargs = generation.get("backend_kwargs", {})
        return vLLMParallelGenerator(model_id, num_gpus=backend_kwargs.get("num_gpus"))
    raise ConfigError(f"Unsupported backend: {backend_name!r}")


def _write_dataset(dataset: Dataset, stem: Path, output: dict[str, Any]) -> str:
    parquet_path = stem.with_suffix(".parquet")
    dataset.to_parquet(str(parquet_path))
    if output.get("write_jsonl", True):
        dataset.to_json(str(stem.with_suffix(".jsonl")), orient="records", lines=True)
    return str(parquet_path.resolve())


def _safe_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Copy JSON-compatible config values into a reproducibility manifest."""
    return json.loads(json.dumps(config, default=str))


def run_generation(config: dict[str, Any]) -> RunArtifacts:
    """Generate a canonical dataset and export canonical, DPO, and SFT views."""
    loader = resolve_loader(config)
    backend = resolve_backend(config)
    generation = config["generation"]
    backend_kwargs = dict(generation.get("backend_kwargs", {}))
    backend_kwargs.pop("num_gpus", None)

    result = build_dpo_dataset(
        loader=loader,
        model_id=generation["model_id"],
        backend=backend,
        use_chat_template=generation.get("use_chat_template", True),
        use_disk_cache=generation.get("use_disk_cache", True),
        model_ctx=generation.get("model_ctx", 16384),
        gpu_memory_utilization=generation.get("gpu_memory_utilization", 0.8),
        temperature=generation.get("temperature", 0.0),
        max_tokens=generation.get("max_tokens", 512),
        annotate=generation.get("annotate", True),
        **backend_kwargs,
    )

    canonical = result.to_hf_dataset()
    dpo = result.to_dpo_dataset()
    sft = result.to_sft_dataset()
    if len(canonical) == 0:
        raise RuntimeError("Generation produced zero canonical rows; no artifacts were uploaded.")

    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(config["output"]["root_dir"]).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    canonical_path = _write_dataset(canonical, run_dir / "canonical", config["output"])
    dpo_path = _write_dataset(dpo, run_dir / "dpo", config["output"])
    sft_path = _write_dataset(sft, run_dir / "sft", config["output"])
    if config["output"].get("write_csv_preview", True):
        result.to_df().head(100).to_csv(run_dir / "canonical-preview.csv", index=False)

    artifacts = RunArtifacts(
        run_dir=str(run_dir.resolve()),
        manifest_path=str((run_dir / "manifest.json").resolve()),
        canonical_parquet=canonical_path,
        dpo_parquet=dpo_path,
        sft_parquet=sft_path,
        num_canonical_rows=len(canonical),
        num_dpo_rows=len(dpo),
        num_sft_rows=len(sft),
    )
    manifest = {
        "schema_version": 1,
        "package_version": __version__,
        "created_at": created.isoformat(),
        "status": "generated",
        "config": _safe_config_snapshot(config),
        "artifacts": asdict(artifacts),
    }
    Path(artifacts.manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Generated %d rows in %s", len(canonical), run_dir)
    return artifacts
