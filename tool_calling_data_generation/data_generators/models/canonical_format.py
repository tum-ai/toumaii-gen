import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, Features, Json, Value

from tool_calling_data_generation.data_generators.models.tools import ToolCatalog


# ---------------------------------------------------------------------------
# CanonicalSample
# ---------------------------------------------------------------------------


class CanonicalSample:
    """
     Single sample type used throughout the data pipeline. This type is created by a data loader sample (partially), filled later by the generator, annotators. This sample type is later used in the training pipeline.

     Properties:
     - conv_id: The unique identifier for the conversation.
     - turn_id: The unique identifier for the turn within the conversation.
     - conv_tracking_id: Auto-generated, concatenation of conv_id and turn_id.
     - custom_system_prompt: The custom system prompt for this sample.
     - is_custom_system_prompt_used: Whether a custom system prompt is used for this sample. Else chat_template will be applied.
     - messages: The list of messages in the conversation. This is a list of dictionaries, where each dictionary represents a message with keys "role" and "content". TODO: Make it pydantic model.
     - tool_catalog: The catalog of tools available for this sample. This is a ToolCatalog object.
     - chosen_response: The list of responses that were chosen by the model for this sample. This is also in conversations format if applicable.
     - rejected_response: The list of responses that were rejected by the model for this sample. This is also in conversations format if applicable.
     - metadata: Additional defined properties.
         - source_dataset: The name of the source dataset (e.g. ``"Toucan"``, ``"ToolRM"``).
         - subset_name: The subset or split within the source dataset (e.g. ``"multi-turn"``, ``"single-turn-original"``). Dataset-specific.
         - model_id: The identifier for the model used in this sample.
         - parsed_chosen_tool_calls: The list of parsed chosen tool calls for this sample. This is a list of dictionaries, where each dictionary represents a tool call with keys "name" and "arguments".
         - parsed_rejected_tool_calls: The list of parsed rejected tool calls for this sample. This is a list of dictionaries, where each dictionary represents a tool call with keys "name" and "arguments".
         - parser_used_chosen: The parser used for parsing the chosen tool calls for this sample.
         - parser_used_rejected: The parser used for parsing the rejected tool calls for this sample.
         - is_best_effort_parser_used: Whether the parser used for the rejected response was a best-effort fallback (regex / JSON salvage) rather than a model-family-specific vLLM parser.
         - chosen_rejected_equivalent: Whether the parsed chosen and rejected tool calls are structurally identical (same names and arguments).
         - annotations: The list of annotations for this sample. This is a list of dictionaries, where each dictionary represents an annotation with keys "annotator_id", "annotation_type", and "annotation_value".
         - annotation_scores: Dictionary of reward scores from the annotation phase. Keys are reward names (e.g. ``"chosen_binary_reward"``, ``"binary_reward_diff"``).


     Functions:
     - to_dict: Convert the CanonicalSample object to a dictionary.
     - to_generator_format: Convert the CanonicalSample object to a dictionary in the format expected by the generator.
     - to_sft_format: Convert the CanonicalSample object to a dictionary in the format expected by the SFT training pipeline. The output needs to be in prompt-completions format.
     - to_dpo_format: Convert the CanonicalSample object to a dictionary in the format expected by the DPO training pipeline. The output needs to be in prompt-completions format, where the completions are the chosen and rejected tool calls.

     Good to Know:
     - Format for SFT - example:
     prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}], // If custom system prompt is used, it will be added to the prompt list as the first element.
                                  "completion": [{"role": "assistant", "content": "It is blue."}],
                                  "tools": [{"name": "color_tool", "description": "A tool to determine colors."}]}

     - Format for DPO - example:
     preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}], // If custom system prompt is used, it will be added to the prompt list as the first element.
                           "chosen": [{"role": "assistant", "content": "It is blue."}],
                           "rejected": [{"role": "assistant", "content": "It is green."}],
                           "tools": [{"name": "color_tool", "description": "A tool to determine colors."}]}

    - Format for generator - only the prompt is required, the generator will generate the completion. This will be the rejected_response, chosen_response is prefilled by the loader.
    """

    # ── Metadata default values (key → default) ──────────────────────
    _METADATA_DEFAULTS: Dict[str, Any] = {
        "source_dataset": None,
        "subset_name": None,
        "toolcall_mode": None,
        "model_id": None,
        "parsed_chosen_tool_calls": [],
        "parsed_rejected_tool_calls": [],
        "parser_used_chosen": None,
        "parser_used_rejected": None,
        "is_best_effort_parser_used": False,
        "chosen_rejected_equivalent": False,
        "annotations": [],
        "annotation_scores": {},
    }

    def __init__(
        self,
        conv_id: str,
        turn_id: str,
        messages: Optional[List[Dict[str, str]]] = None,
        custom_system_prompt: Optional[str] = None,
        is_custom_system_prompt_used: bool = False,
        tool_catalog: Optional[ToolCatalog] = None,
        chosen_response: Optional[List[Dict[str, str]]] = None,
        rejected_response: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.conv_id: str = conv_id
        self.turn_id: str = turn_id
        self.conv_tracking_id: str = f"{conv_id}_{turn_id}"
        self.custom_system_prompt: Optional[str] = custom_system_prompt
        self.is_custom_system_prompt_used: bool = is_custom_system_prompt_used
        self.messages: List[Dict[str, str]] = messages or []
        self.tool_catalog: ToolCatalog = tool_catalog or ToolCatalog()
        self.chosen_response: List[Dict[str, str]] = chosen_response or []
        self.rejected_response: List[Dict[str, str]] = rejected_response or []

        # Merge user-provided metadata over the defaults so every
        # expected key is always present (Arrow column stability).
        merged = dict(self._METADATA_DEFAULTS)
        if metadata:
            merged.update(metadata)
        self.metadata: Dict[str, Any] = merged

    # ── Typed metadata accessors ─────────────────────────────────────
    # These read/write directly into self.metadata so both ``.model_id``
    # and ``.metadata["model_id"]`` stay in sync.

    @property
    def model_id(self) -> Optional[str]:
        """The identifier for the model used in this sample."""
        return self.metadata.get("model_id")

    @model_id.setter
    def model_id(self, value: Optional[str]) -> None:
        self.metadata["model_id"] = value

    @property
    def parsed_chosen_tool_calls(self) -> List[Dict[str, Any]]:
        """The list of parsed chosen tool calls for this sample. This is a
        list of dictionaries, where each dictionary represents a tool call
        with keys ``"name"`` and ``"arguments"``."""
        return self.metadata.get("parsed_chosen_tool_calls", [])

    @parsed_chosen_tool_calls.setter
    def parsed_chosen_tool_calls(self, value: List[Dict[str, Any]]) -> None:
        self.metadata["parsed_chosen_tool_calls"] = value

    @property
    def parsed_rejected_tool_calls(self) -> List[Dict[str, Any]]:
        """The list of parsed rejected tool calls for this sample. This is a
        list of dictionaries, where each dictionary represents a tool call
        with keys ``"name"`` and ``"arguments"``."""
        return self.metadata.get("parsed_rejected_tool_calls", [])

    @parsed_rejected_tool_calls.setter
    def parsed_rejected_tool_calls(self, value: List[Dict[str, Any]]) -> None:
        self.metadata["parsed_rejected_tool_calls"] = value

    @property
    def parser_used_chosen(self) -> Optional[str]:
        """The parser used for parsing the chosen tool calls for this sample."""
        return self.metadata.get("parser_used_chosen")

    @parser_used_chosen.setter
    def parser_used_chosen(self, value: Optional[str]) -> None:
        self.metadata["parser_used_chosen"] = value

    @property
    def parser_used_rejected(self) -> Optional[str]:
        """The parser used for parsing the rejected tool calls for this sample."""
        return self.metadata.get("parser_used_rejected")

    @parser_used_rejected.setter
    def parser_used_rejected(self, value: Optional[str]) -> None:
        self.metadata["parser_used_rejected"] = value

    @property
    def annotations(self) -> List[Dict[str, Any]]:
        """The list of annotations for this sample. This is a list of
        dictionaries, where each dictionary represents an annotation with
        keys ``"annotator_id"``, ``"annotation_type"``, and
        ``"annotation_value"``."""
        return self.metadata.get("annotations", [])

    @annotations.setter
    def annotations(self, value: List[Dict[str, Any]]) -> None:
        self.metadata["annotations"] = value

    @property
    def is_best_effort_parser_used(self) -> bool:
        """Whether the parser used for the rejected response was a
        best-effort fallback (regex / JSON salvage) rather than a
        model-family-specific vLLM parser."""
        return self.metadata.get("is_best_effort_parser_used", False)

    @is_best_effort_parser_used.setter
    def is_best_effort_parser_used(self, value: bool) -> None:
        self.metadata["is_best_effort_parser_used"] = value

    @property
    def chosen_rejected_equivalent(self) -> bool:
        """Whether the parsed chosen and rejected tool calls are
        structurally identical (same names and arguments)."""
        return self.metadata.get("chosen_rejected_equivalent", False)

    @chosen_rejected_equivalent.setter
    def chosen_rejected_equivalent(self, value: bool) -> None:
        self.metadata["chosen_rejected_equivalent"] = value

    @property
    def source_dataset(self) -> Optional[str]:
        """The name of the source dataset (e.g. ``"Toucan"``, ``"ToolRM"``)."""
        return self.metadata.get("source_dataset")

    @source_dataset.setter
    def source_dataset(self, value: Optional[str]) -> None:
        self.metadata["source_dataset"] = value

    @property
    def subset_name(self) -> Optional[str]:
        """The subset or split within the source dataset
        (e.g. ``"multi-turn"``, ``"single-turn-original"``)."""
        return self.metadata.get("subset_name")

    @subset_name.setter
    def subset_name(self, value: Optional[str]) -> None:
        self.metadata["subset_name"] = value

    @property
    def toolcall_mode(self) -> Optional[str]:
        """The tool-call mode used when splitting conversation turns
        (``"single"`` or ``"multi"``)."""
        return self.metadata.get("toolcall_mode")

    @toolcall_mode.setter
    def toolcall_mode(self, value: Optional[str]) -> None:
        self.metadata["toolcall_mode"] = value

    # ── helpers ──────────────────────────────────────────────────────

    def _build_prompt_list(self) -> List[Dict[str, str]]:
        """Build the prompt message list, prepending the custom system prompt if used."""
        prompt: List[Dict[str, str]] = []
        if self.is_custom_system_prompt_used and self.custom_system_prompt:
            prompt.append({"role": "system", "content": self.custom_system_prompt})
        prompt.extend(self.messages)
        return prompt

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CanonicalSample object to a dictionary."""
        return {
            "conv_id": self.conv_id,
            "turn_id": self.turn_id,
            "conv_tracking_id": self.conv_tracking_id,
            "custom_system_prompt": self.custom_system_prompt,
            "is_custom_system_prompt_used": self.is_custom_system_prompt_used,
            "messages": self.messages,
            "tool_catalog": self.tool_catalog.to_list(),
            "chosen_response": self.chosen_response,
            "rejected_response": self.rejected_response,
            "metadata": self.metadata,
        }

    def to_generator_format(self) -> Dict[str, Any]:
        """Convert to the format expected by the generator (prompt only, no completion)."""
        return {
            "conv_id": self.conv_id,
            "turn_id": self.turn_id,
            "conv_tracking_id": self.conv_tracking_id,
            "prompt": self._build_prompt_list(),
            "tools": self.tool_catalog.to_list(),
            "chosen_response": self.chosen_response,
            "metadata": self.metadata,
        }

    def to_sft_format(self) -> Dict[str, Any]:
        """Convert to prompt-completion format for SFT training.

        Returns a dict with keys: prompt, completion, tools.
        If custom system prompt is used, it is prepended to the prompt list.
        """
        return {
            "prompt": self._build_prompt_list(),
            "completion": self.chosen_response,
            "tools": self.tool_catalog.to_list(),
        }

    def to_dpo_format(self) -> Dict[str, Any]:
        """Convert to preference format for DPO training.

        Returns a dict with keys: prompt, chosen, rejected, tools.
        If custom system prompt is used, it is prepended to the prompt list.
        """
        return {
            "prompt": self._build_prompt_list(),
            "chosen": self.chosen_response,
            "rejected": self.rejected_response,
            "tools": self.tool_catalog.to_list(),
        }

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "CanonicalSample":
        """Build a ``CanonicalSample`` from a dictionary row
        (matching ``to_dict`` output)."""
        tool_catalog = ToolCatalog.from_list(
            _parse_json_column(row.get("tool_catalog", []), default=[])
        )
        return cls(
            conv_id=row["conv_id"],
            turn_id=row["turn_id"],
            messages=_parse_json_column(row.get("messages", []), default=[]),
            custom_system_prompt=row.get("custom_system_prompt"),
            is_custom_system_prompt_used=row.get("is_custom_system_prompt_used", False),
            tool_catalog=tool_catalog,
            chosen_response=_parse_json_column(
                row.get("chosen_response", []), default=[]
            ),
            rejected_response=_parse_json_column(
                row.get("rejected_response", []), default=[]
            ),
            metadata=_parse_json_column(row.get("metadata", {}), default={}),
        )


# ---------------------------------------------------------------------------
# CanonicalDataset — Arrow-backed with lazy per-sample access
# ---------------------------------------------------------------------------

# Column names that map to the CanonicalSample.to_dict() schema.
_CANONICAL_COLUMNS = [
    "conv_id",
    "turn_id",
    "conv_tracking_id",
    "custom_system_prompt",
    "is_custom_system_prompt_used",
    "messages",
    "tool_catalog",
    "chosen_response",
    "rejected_response",
    "metadata",
]


def _parse_json_column(value: Any, default: Any = None) -> Any:
    """Parse a value that may be a JSON string (Arrow storage) or already
    a Python object.

    *default* is returned when the parsed value is falsy (empty string,
    ``None``, empty list).  Useful when the expected type differs
    (``[]`` for list columns, ``{}`` for dict columns).
    """
    if isinstance(value, str):
        if not value.strip():
            return default
        try:
            parsed = json.loads(value)
            return parsed if parsed or default is None else default
        except json.JSONDecodeError:
            return value if value else default
    return value if value else default


class CanonicalDataset:
    """Collection of ``CanonicalSample`` objects, backed internally by a
    HuggingFace ``Dataset`` for efficient vectorised operations.

    Design
    ------
    * **Arrow path** — ``_ds`` is the source of truth for batch ops
      (``to_sft_dataset``, ``to_dpo_dataset``, ``to_dict_list``, ``map``,
      ``filter``).  These are zero-overhead Arrow operations that scale to
      250k+ rows.
    * **Python-object path** — ``_samples`` is built **lazily** only when
      per-sample access is requested (``get_sample``, ``__iter__``,
      ``add_sample``, ``remove_sample``).  This is O(n) on first access
      because every row is deserialised into ``CanonicalSample`` with
      ``ToolCatalog`` parsing, so prefer batch operations on the hot path.

    Construction
    ------------
    .. code-block:: python

        # Empty
        CanonicalDataset()

        # From list of canonical dicts (fast — stays Arrow)
        CanonicalDataset.from_dict_list(list_of_dicts)

        # From an existing HF Dataset (zero-copy)
        CanonicalDataset.from_hf_dataset(hf_ds)

        # From a pre-built samples dict (for per-sample mutation)
        CanonicalDataset(samples=samples_dict)

    Batch usage (hot path — Arrow-backed, no per-row Python objects)
    ----------------------------------------------------------------
    .. code-block:: python

        sft_ds = dataset.to_sft_dataset()    # Arrow-backed map
        dpo_ds = dataset.to_dpo_dataset()    # Arrow-backed map
        smaller = dataset.filter(lambda r: len(r["messages"]) > 0)
        updated  = dataset.map(my_transform, batched=True, num_proc=8)

    Per-sample usage (convenience — triggers lazy deserialisation)
    --------------------------------------------------------------
    .. code-block:: python

        sample = dataset.get_sample("conv1", "turn3")
        dataset.add_sample(sample)
        for s in dataset:
            print(s.messages)
    """

    def __init__(
        self,
        samples: Optional[Dict[Tuple[str, str], CanonicalSample]] = None,
        dataset: Optional[Dataset] = None,
    ):
        if dataset is not None:
            self._ds: Optional[Dataset] = dataset
            self._samples: Optional[Dict[Tuple[str, str], CanonicalSample]] = None
        elif samples is not None:
            self._samples = samples
            self._ds: Optional[Dataset] = None
        else:
            self._ds = Dataset.from_list([])
            self._samples: Optional[Dict[Tuple[str, str], CanonicalSample]] = None

    # ==================================================================
    # Internal — lazy builders
    # ==================================================================

    def _build_samples_from_ds(self) -> None:
        """Build ``_samples`` dict from ``_ds`` (O(n) — expensive)."""
        samples: Dict[Tuple[str, str], CanonicalSample] = {}
        for row in self._ds:
            sample = CanonicalSample.from_dict(row)
            samples[(sample.conv_id, sample.turn_id)] = sample
        self._samples = samples

    def _build_ds_from_samples(self) -> None:
        """Rebuild ``_ds`` from ``_samples``."""
        rows = [s.to_dict() for s in self._samples.values()]
        self._ds = Dataset.from_list(rows)

    # ==================================================================
    # Public properties
    # ==================================================================

    @property
    def samples(self) -> Dict[Tuple[str, str], CanonicalSample]:
        """Lazily-built dict of ``(conv_id, turn_id) → CanonicalSample``.

        First access triggers O(n) deserialisation from the backing
        Arrow Dataset.  For batch work prefer ``map`` / ``filter`` /
        ``to_*_dataset``.
        """
        if self._samples is None:
            self._build_samples_from_ds()
        return self._samples

    @property
    def _dataset(self) -> Dataset:
        """Always-valid backing HF Dataset.

        Rebuilds from ``_samples`` if the dict was mutated
        (``add_sample`` / ``remove_sample`` invalidated ``_ds``).
        """
        if self._ds is None:
            self._build_ds_from_samples()
        return self._ds

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __len__(self) -> int:
        if self._ds is not None:
            return len(self._ds)
        return len(self.samples)

    def __iter__(self) -> Iterator[CanonicalSample]:
        return iter(self.samples.values())

    def __getitem__(self, index: int) -> CanonicalSample:
        return list(self.samples.values())[index]

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self.samples

    def __repr__(self) -> str:
        n = len(self)
        backed_by = "Arrow" if self._ds is not None else "dict"
        return f"CanonicalDataset(n={n}, backed={backed_by})"

    # ==================================================================
    # Mutation (per-sample — invalidates Arrow cache)
    # ==================================================================

    def add_sample(self, sample: CanonicalSample) -> None:
        """Add or overwrite a single sample.

        Invalidates the backing Arrow Dataset so the next batch
        operation will rebuild it from the dict.
        """
        if self._samples is None:
            self._build_samples_from_ds()
        key = (sample.conv_id, sample.turn_id)
        self._samples[key] = sample
        self._ds = None

    def remove_sample(self, conv_id: str, turn_id: str) -> None:
        """Remove a single sample by its ``(conv_id, turn_id)`` key.

        No-ops silently if the key doesn't exist.
        """
        key = (conv_id, turn_id)
        if key not in self.samples:
            return
        del self._samples[key]
        self._ds = None

    def get_sample(self, conv_id: str, turn_id: str) -> Optional[CanonicalSample]:
        """Retrieve one sample by key (triggers lazy deserialisation)."""
        return self.samples.get((conv_id, turn_id))

    # ==================================================================
    # Batch operations (Arrow-native — no per-row Python objects)
    # ==================================================================

    def map(
        self,
        fn,
        batched: bool = True,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        desc: str = "",
        **kwargs,
    ) -> "CanonicalDataset":
        """Apply *fn* to every row via HF ``Dataset.map``.

        *fn* receives a row dict (``CanonicalSample.to_dict()`` schema)
        and must return a row dict in the same schema.  Uses Arrow-native
        vectorised dispatch — suitable for 250k+ rows.

        Returns a **new** ``CanonicalDataset``.
        """
        new_ds = self._dataset.map(
            fn,
            batched=batched,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=desc,
            **kwargs,
        )
        return CanonicalDataset(dataset=new_ds)

    def filter(
        self,
        fn,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        desc: str = "",
        **kwargs,
    ) -> "CanonicalDataset":
        """Filter rows via HF ``Dataset.filter``.  Returns a **new**
        ``CanonicalDataset``.
        """
        new_ds = self._dataset.filter(
            fn,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=desc,
            **kwargs,
        )
        return CanonicalDataset(dataset=new_ds)

    def select(self, indices: List[int]) -> "CanonicalDataset":
        """Return a new dataset containing only rows at *indices*."""
        new_ds = self._dataset.select(indices)
        return CanonicalDataset(dataset=new_ds)

    def head(self, n: int = 5) -> "CanonicalDataset":
        """Return the first *n* rows as a new dataset."""
        n = min(n, len(self))
        return self.select(list(range(n)))

    # ==================================================================
    # Conversion — in & out
    # ==================================================================

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export all rows as a list of dicts.

        When Arrow-backed this is O(n) Arrow→Python deserialisation
        (faster than per-sample ``to_dict()`` iteration).
        """
        if self._ds is not None:
            return self._ds.to_list()
        return [s.to_dict() for s in self.samples.values()]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> "CanonicalDataset":
        """Create from a list of canonical-format dicts.

        Stays on the Arrow path — no ``CanonicalSample`` objects are
        created until per-sample access is requested.
        """
        return cls(dataset=Dataset.from_list(data))

    @classmethod
    def from_hf_dataset(cls, ds: Dataset) -> "CanonicalDataset":
        """Wrap an existing HF Dataset (zero-copy).

        The Dataset must have columns matching the ``CanonicalSample``
        schema: ``conv_id``, ``turn_id``, ``messages``, ``tool_catalog``,
        ``chosen_response``, ``rejected_response``, etc.

        Missing optional columns are filled with sensible defaults
        during per-sample deserialisation (if ever triggered).
        """
        return cls(dataset=ds)

    def to_hf_dataset(self) -> Dataset:
        """Return the underlying HF Dataset (no copy)."""
        return self._dataset

    # ==================================================================
    # DataFrame export (inspection & debugging)
    # ==================================================================

    def to_df(self) -> "pd.DataFrame":
        """Export to a ``pandas.DataFrame`` with all JSON-string columns
        parsed to native Python objects.

        Useful for inspection, filtering, and analysis.  For training
        export use ``to_sft_dataset()`` or ``to_dpo_dataset()`` instead.
        """
        import pandas as pd

        json_cols = {"tool_catalog", "chosen_response", "rejected_response", "metadata"}
        rows = []
        for row in self._dataset:
            parsed = {}
            for col in self._dataset.column_names:
                val = row[col]
                if col in json_cols:
                    val = _parse_json_column(val, default=val)
                parsed[col] = val
            rows.append(parsed)
        return pd.DataFrame(rows)

    def to_sft_dataset(self) -> Dataset:
        """Convert to a HF Dataset in prompt-completion format (SFT).

        Columns: ``prompt``, ``completion``, ``tools``,
        ``conv_id``, ``conv_tracking_id``, ``metadata``.

        Uses Arrow-native ``ds.map`` under the hood — no per-row
        ``CanonicalSample`` objects are created.
        """

        def _build_prompt_list(
            messages: List[Dict],
            custom_sp: Optional[str],
            is_custom: bool,
        ) -> List[Dict]:
            prompt: List[Dict] = []
            if is_custom and custom_sp:
                prompt.append({"role": "system", "content": custom_sp})
            if isinstance(messages, list):
                prompt.extend(messages)
            return prompt

        def _to_sft(batch: Dict[str, List]) -> Dict[str, List]:
            n = len(batch["conv_id"])
            prompts = []
            completions = []
            tools_list = []
            for i in range(n):
                prompts.append(
                    _build_prompt_list(
                        batch["messages"][i] or [],
                        batch.get("custom_system_prompt", [None] * n)[i],
                        batch.get("is_custom_system_prompt_used", [False] * n)[i],
                    )
                )
                completions.append(
                    _parse_json_column(
                        batch.get("chosen_response", [None] * n)[i], default=[]
                    )
                )
                tc_raw = batch.get("tool_catalog", [None] * n)[i]
                tools_list.append(_parse_json_column(tc_raw, default=[]) or [])
            return {
                "prompt": prompts,
                "completion": completions,
                "tools": tools_list,
                "conv_id": batch["conv_id"],
                "conv_tracking_id": batch["conv_tracking_id"],
                "metadata": batch["metadata"],
            }

        return self._dataset.map(
            _to_sft,
            batched=True,
            remove_columns=self._dataset.column_names,
            desc="Converting to SFT format",
            features=Features(
                {
                    "prompt": Json(),
                    "completion": Json(),
                    "tools": Json(),
                    "conv_id": Value("string"),
                    "conv_tracking_id": Value("string"),
                    "metadata": Json(),
                }
            ),
        )

    def to_dpo_dataset(self) -> Dataset:
        """Convert to a HF Dataset in preference format (DPO).

        Columns: ``prompt``, ``chosen``, ``rejected``, ``tools``,
        ``conv_id``, ``conv_tracking_id``, ``metadata``.

        Uses Arrow-native ``ds.map`` under the hood — no per-row
        ``CanonicalSample`` objects are created.
        """

        def _build_prompt_list(
            messages: List[Dict],
            custom_sp: Optional[str],
            is_custom: bool,
        ) -> List[Dict]:
            prompt: List[Dict] = []
            if is_custom and custom_sp:
                prompt.append({"role": "system", "content": custom_sp})
            if isinstance(messages, list):
                prompt.extend(messages)
            return prompt

        def _to_dpo(batch: Dict[str, List]) -> Dict[str, List]:
            n = len(batch["conv_id"])
            prompts = []
            chosens = []
            rejects = []
            tools_list = []
            for i in range(n):
                prompts.append(
                    _build_prompt_list(
                        batch["messages"][i] or [],
                        batch.get("custom_system_prompt", [None] * n)[i],
                        batch.get("is_custom_system_prompt_used", [False] * n)[i],
                    )
                )
                chosens.append(
                    _parse_json_column(
                        batch.get("chosen_response", [None] * n)[i], default=[]
                    )
                )
                rejects.append(
                    _parse_json_column(
                        batch.get("rejected_response", [None] * n)[i], default=[]
                    )
                )
                tc_raw = batch.get("tool_catalog", [None] * n)[i]
                tools_list.append(_parse_json_column(tc_raw, default=[]) or [])
            return {
                "prompt": prompts,
                "chosen": chosens,
                "rejected": rejects,
                "tools": tools_list,
                "conv_id": batch["conv_id"],
                "conv_tracking_id": batch["conv_tracking_id"],
                "metadata": batch["metadata"],
            }

        return self._dataset.map(
            _to_dpo,
            batched=True,
            remove_columns=self._dataset.column_names,
            desc="Converting to DPO format",
            features=Features(
                {
                    "prompt": Json(),
                    "chosen": Json(),
                    "rejected": Json(),
                    "tools": Json(),
                    "conv_id": Value("string"),
                    "conv_tracking_id": Value("string"),
                    "metadata": Json(),
                }
            ),
        )
