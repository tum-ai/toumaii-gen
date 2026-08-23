"""
Batch annotation pipeline for pre-generated DPO datasets.

Loads multiple canonical-format datasets (from different model-generator runs),
merges them, optionally selects the best pair per conversation across models,
annotates with all registered rewards, filters by threshold,
and saves/pushes the result.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from tool_calling_data_generation.data_generators.annotations import Annotator
from tool_calling_data_generation.data_generators.loaders import ExistingDPOLoader
from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Identifies one canonical-format dataset on HF Hub."""
    dataset_id: str
    model_name: str
    split: str = "train"
    max_samples: Optional[int] = None


@dataclass
class BatchAnnotatorConfig:
    """Configuration for a batch annotation run."""
    datasets: List[DatasetSpec] = field(default_factory=list)
    cross_model_selection: bool = True
    primary_reward: str = "curriculum"
    extra_reward_types: List[str] = field(default_factory=lambda: ["toolrl"])
    threshold: float = 0.1
    do_filter: bool = False
    output_dir: str = "./data/batch_annotated"
    push_to_hub: bool = False
    hf_org: Optional[str] = None
    hf_repo: Optional[str] = None


class BatchAnnotator:
    """Load, merge, filter across models, annotate, and save DPO datasets.

    Typical usage::

        from tool_calling_data_generation.data_generators.batch_annotator import BatchAnnotator,             BatchAnnotatorConfig, DatasetSpec

        config = BatchAnnotatorConfig(
            datasets=[
                DatasetSpec("YOUR_ORG/toucan_dpo_Qwen3-1.7B", "Qwen3-1.7B"),
                DatasetSpec("YOUR_ORG/toucan_dpo_granite-4.1-3b", "granite-4.1-3b"),
            ],
            cross_model_selection=True,
            primary_reward="curriculum",
        )
        ba = BatchAnnotator(config)
        result = ba.run()
    """

    def __init__(self, config: BatchAnnotatorConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> CanonicalDataset:
        """Execute the full batch annotation pipeline."""
        # 1. Load datasets
        cds = self._load_all()

        # 2. Merge and optionally select across models
        if self.config.cross_model_selection and len(cds) > 1:
            merged = self._merge_and_select(cds)
        else:
            merged = self._concat(cds)

        df = merged.to_df()

        # 3. Annotate with all registered rewards
        logger.info("Annotating %d rows with all registered rewards ...", len(df))
        annotator = Annotator()
        df = annotator.annotate(df)

        # 4. Filter by threshold (optional)
        if self.config.do_filter:
            primary_col = f"chosen_{self.config.primary_reward}_reward"
            if primary_col in df.columns:
                before = len(df)
                df = df[df[primary_col] >= self.config.threshold].copy()
                logger.info("Filtered %d -> %d rows (threshold=%.2f)", before, len(df), self.config.threshold)

        # 5. Save
        self._save(df)

        # Convert back to CanonicalDataset for return
        return CanonicalDataset.from_dict_list(df.to_dict("records"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> List[CanonicalDataset]:
        """Load every dataset in the config."""
        cds_list: List[CanonicalDataset] = []
        for spec in self.config.datasets:
            logger.info("Loading %s (model=%s) ...", spec.dataset_id, spec.model_name)
            loader = ExistingDPOLoader(
                dataset_id=spec.dataset_id,
                split=spec.split,
                max_samples=spec.max_samples,
            )
            cd = loader.load()
            # Tag rows with the model name
            hf = cd.to_hf_dataset()
            hf = hf.add_column("model_name", [spec.model_name] * len(hf))
            cds_list.append(CanonicalDataset.from_hf_dataset(hf))
            logger.info("  -> %d rows", len(hf))
        return cds_list

    def _concat(self, cds_list: List[CanonicalDataset]) -> CanonicalDataset:
        """Concatenate multiple datasets (no cross-model selection)."""
        datasets = [cd.to_hf_dataset() for cd in cds_list]
        from datasets import concatenate_datasets
        merged = concatenate_datasets(datasets)
        return CanonicalDataset.from_hf_dataset(merged)

    def _merge_and_select(self, cds_list: List[CanonicalDataset]) -> CanonicalDataset:
        """Merge datasets grouped by conv_id and pick the best pair per group.

        For each ``conv_id`` across all model runs, computes the primary reward
        and keeps the pair with the highest ``chosen - rejected`` margin.
        """
        logger.info("Performing cross-model selection (primary_reward=%s) ...",
                     self.config.primary_reward)

        dfs = [cd.to_df() for cd in cds_list]
        merged_df = pd.concat(dfs, ignore_index=True)

        if "conv_id" not in merged_df.columns:
            logger.warning("conv_id column not found — skipping cross-model selection, using simple concat.")
            return CanonicalDataset.from_dict_list(merged_df.to_dict("records"))

        # Score each row on the primary reward
        from tool_calling_data_generation.rewards import RewardRegistry
        registry = RewardRegistry()
        reward = registry.get(self.config.primary_reward)

        def _score_row(row):
            chosen_calls = self._get_tool_calls(row.get("chosen_response", "[]"))
            gt = {"expected_calls": chosen_calls}
            chosen_text = json.dumps(chosen_calls)
            rejected_text = json.dumps(self._get_tool_calls(row.get("rejected_response", "[]")))
            try:
                c_res = reward.compute(chosen_text, gt)
                r_res = reward.compute(rejected_text, gt)
                return c_res.score - r_res.score if r_res.score is not None else c_res.score
            except Exception:
                return float("-inf")

        merged_df["_reward_margin"] = merged_df.apply(_score_row, axis=1)

        # For each conv_id, keep the row with the highest margin
        best = merged_df.loc[
            merged_df.groupby("conv_id")["_reward_margin"].idxmax()
        ].copy()
        best.drop(columns=["_reward_margin"], inplace=True)

        logger.info("Cross-model selection: %d groups -> %d rows",
                     merged_df["conv_id"].nunique(), len(best))
        return CanonicalDataset.from_dict_list(best.to_dict("records"))

    def _get_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract ``[{name, arguments}, ...]`` from a response dict or JSON string."""
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                return []
        result: List[Dict[str, Any]] = []
        for msg in response or []:
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append({"name": func.get("name", ""), "arguments": args})
        return result

    def _save(self, df: pd.DataFrame) -> str:
        """Save the annotated dataset to disk and optionally push to HF Hub."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save full dataset to disk
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.save_to_disk(self.config.output_dir)
        logger.info("Saved %d rows to %s", len(ds), self.config.output_dir)

        # Save CSV for inspection
        csv_path = os.path.join(self.config.output_dir, "annotated_dataset.csv")
        df.to_csv(csv_path, index=False, quoting=1)
        logger.info("CSV saved to %s", csv_path)

        # Push to HF Hub (optional)
        if self.config.push_to_hub:
            if not self.config.hf_org or not self.config.hf_repo:
                raise ValueError(
                    "hf_org and hf_repo are required when push_to_hub is enabled."
                )
            repo_id = f"{self.config.hf_org}/{self.config.hf_repo}"
            logger.info("Pushing to %s ...", repo_id)
            ds.push_to_hub(repo_id, private=True)
            logger.info("Pushed to %s", repo_id)

        return self.config.output_dir
