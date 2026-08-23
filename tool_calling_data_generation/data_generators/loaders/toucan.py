"""
Toucan data loader.

Quality-filters the Toucan-1.5M dataset and normalizes each
conversation turn into canonical format.
"""

import json
import os
from typing import Optional

from datasets import Dataset, load_dataset

from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader
from tool_calling_data_generation.data_generators.models.canonical_format import (
    CanonicalDataset,
    _CANONICAL_COLUMNS,
)

# ---------------------------------------------------------------------------
# Quality filter (unchanged core logic)
# ---------------------------------------------------------------------------


def build_sanitized_toucan(
    dataset_id: str = "Agent-Ark/Toucan-1.5M",
    dataset_revision: Optional[str] = None,
    size: Optional[int] = None,
    streaming: bool = False,
    use_viewer_api: bool = False,
    viewer_offset: int = 0,
    viewer_page_size: int = 20,
    viewer_max_rows: int = 100,
    dataset_name: str = "Qwen3",
    overall_score_threshold: float = 4.5,
    desired_tools_used_percentage_threshold: float = 1.0,
    order_correctness_required: bool = True,
    subset_name_required: str = "single-turn-original",
    use_disk_cache: bool = True,
) -> Dataset:
    """Load and quality-filter the Toucan-1.5M dataset.

    Returns a HuggingFace ``Dataset`` with the raw Toucan columns.
    """
    def parse_and_filter(response_quality_assessment: str, subset_name: str):
        if subset_name != subset_name_required:
            return False
        rqa: dict = {}
        if isinstance(response_quality_assessment, dict):
            rqa = response_quality_assessment
        else:
            try:
                rqa = json.loads(response_quality_assessment)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Failed to parse response_quality_assessment: "
                    f"{response_quality_assessment} \n Error: {e}"
                )
        return (
            rqa.get("overall_score", 0) >= overall_score_threshold
            and rqa.get("desired_tools_used_percentage", 0)
            == desired_tools_used_percentage_threshold
            and rqa.get("order_correctness", False) == order_correctness_required
        )

    if use_viewer_api:
        if size is None:
            raise ValueError("Toucan Dataset Viewer mode requires a finite size.")
        if dataset_revision:
            from huggingface_hub import HfApi

            current_revision = HfApi().dataset_info(dataset_id).sha
            if current_revision != dataset_revision:
                raise RuntimeError(
                    "Toucan revision changed; update the smoke offset only after review. "
                    f"Expected {dataset_revision}, found {current_revision}."
                )

        import requests

        selected = []
        scanned = 0
        offset = viewer_offset
        while len(selected) < size and scanned < viewer_max_rows:
            length = min(viewer_page_size, viewer_max_rows - scanned)
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": dataset_id,
                    "config": dataset_name,
                    "split": "train",
                    "offset": offset,
                    "length": length,
                },
                timeout=120,
            )
            response.raise_for_status()
            items = response.json().get("rows", [])
            if not items:
                break
            for item in items:
                if item.get("truncated_cells"):
                    raise RuntimeError(
                        f"Dataset Viewer row {item.get('row_idx')} contains truncated cells."
                    )
                row = item["row"]
                if parse_and_filter(
                    row["response_quality_assessment"], row["subset_name"]
                ):
                    selected.append(row)
                    if len(selected) == size:
                        break
            scanned += len(items)
            offset += len(items)

        if len(selected) < size:
            raise RuntimeError(
                f"Dataset Viewer returned only {len(selected)} qualifying rows after "
                f"scanning {scanned} rows from offset {viewer_offset}."
            )
        filtered = Dataset.from_list(selected)
        print(
            f"Selected {len(filtered)} samples through the Dataset Viewer API "
            f"from offset {viewer_offset}."
        )
        return filtered

    ds = load_dataset(
        dataset_id,
        name=dataset_name,
        split="train",
        streaming=streaming,
        revision=dataset_revision,
    )

    if streaming:
        if size is None:
            raise ValueError("Toucan streaming mode requires a finite size.")
        filtered_stream = ds.filter(
            parse_and_filter,
            input_columns=["response_quality_assessment", "subset_name"],
        )
        # Materialize only matching smoke-test rows, then continue through the
        # same Arrow conversion path as non-streaming runs.
        filtered = Dataset.from_list(list(filtered_stream.take(size)))
    else:
        filtered = ds.filter(
            parse_and_filter,
            load_from_cache_file=use_disk_cache,
            input_columns=["response_quality_assessment", "subset_name"],
            desc="Filtering based on response quality assessment and subset name",
        )
        if size is not None:
            filtered = filtered.select(range(min(size, len(filtered))))

    print(
        f"Selected {len(filtered)} samples from the filtered dataset "
        f"based on the specified context size."
    )
    return filtered


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _parse_json_field(value):
    """Parse a value that may be a JSON string or already a Python object."""
    if isinstance(value, (dict, list)):
        return value
    if not value or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# ujson (used by HF's ``Json`` feature) raises "Value is too big!" for
# integers >= 10**20.  Python's ``json`` handles arbitrary precision, so we
# stringify such values when building tool-call arguments — otherwise the
# ``messages`` column cannot round-trip through ``Json``-feature decode.
_JSON_INT_SAFE_LIMIT = 10**19


def _json_safe_int(obj):
    """Recursively convert integers that ``ujson`` cannot represent to strings.

    ``ujson`` fails on integers >= 10**20 (e.g. tool results like ``100!``,
    which is a 158-digit number).  We convert those to strings so the
    ``messages`` column survives HF's ``Json(decode=True)`` round-trip.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return str(obj) if abs(obj) >= _JSON_INT_SAFE_LIMIT else obj
    if isinstance(obj, dict):
        return {k: _json_safe_int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_int(v) for v in obj]
    return obj


def _normalize_function_call(fc: dict) -> dict:
    """Convert Toucan's ``function_call`` dict to OpenAI ``tool_calls`` format.

    Toucan stores arguments as a JSON string; we parse them into a dict
    so the canonical format is always ``{"type":"function", "function":
    {"name":..., "arguments":{...}}}`` regardless of source.
    """
    args = fc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass  # keep as string — preserves malformed data for DPO
    args = _json_safe_int(args)
    return {
        "type": "function",
        "function": {
            "name": fc.get("name", ""),
            "arguments": args,
        },
    }


def _assistant_tool_call_message(function_call) -> dict:
    """Build an OpenAI-style assistant message with ``tool_calls`` from a
    Toucan ``function_call`` (single dict or list of dicts).

    Storing tool calls in the ``tool_calls`` key (instead of serialising
    them into ``content``) is what lets ``apply_chat_template`` render the
    model-family-specific tool-call syntax (e.g. ``<tool_call>`` for
    Qwen/Gemma/Granite).  This matches the ``chosen_response`` format.
    """
    calls = function_call if isinstance(function_call, list) else [function_call]
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_normalize_function_call(fc) for fc in calls],
    }


# ---------------------------------------------------------------------------
# Batched explode: Toucan rows → canonical-format rows
# ---------------------------------------------------------------------------


def _emit_turn(
    result: dict,
    conv_id: str,
    turn_idx: int,
    history: list,
    tool_catalog_list: list,
    chosen: dict,
    subset_name,
    toolcall_mode: str = "single",
) -> None:
    """Append one exploded canonical row to *result* (in place)."""
    result["conv_id"].append(conv_id)
    result["turn_id"].append(str(turn_idx))
    result["conv_tracking_id"].append(f"{conv_id}_{turn_idx}")
    result["custom_system_prompt"].append(None)
    result["is_custom_system_prompt_used"].append(False)
    result["messages"].append(list(history))
    result["tool_catalog"].append(json.dumps(tool_catalog_list))
    result["chosen_response"].append(json.dumps([chosen]))
    result["rejected_response"].append("[]")
    result["metadata"].append(
        json.dumps(
            {
                "source_dataset": "Toucan",
                "subset_name": subset_name,
                "toolcall_mode": toolcall_mode,
            }
        )
    )


def _toucan_to_canonical_batch(batch: dict, toolcall_mode: str = "single") -> dict:
    """Batched conversion.  One Toucan conversation row may explode into
    multiple canonical rows.

    Turn-splitting depends on ``toolcall_mode``:

    * ``"single"`` (default) — one canonical row per assistant
      ``function_call`` message.  Each ``chosen_response`` contains a
      single tool call.
    * ``"multi"`` — consecutive assistant tool-call messages (with any
      interleaved ``function`` results between them) are merged into a
      single canonical row whose ``chosen_response`` contains **all**
      those tool calls at once.  This trains the model to emit multiple
      calls in one turn.

    Input columns (expected from Toucan):
        ``uuid``, ``messages``, ``available_tools``, ``subset_name``

    Output:
        All columns from ``_CANONICAL_COLUMNS`` (see
        ``CanonicalSample.to_dict()`` for the schema).
    """
    result: dict = {col: [] for col in _CANONICAL_COLUMNS}
    n = len(batch["uuid"])

    for i in range(n):
        conv_id = str(batch["uuid"][i])
        tools_raw = _parse_json_field(batch["available_tools"][i])
        messages_raw = _parse_json_field(batch["messages"][i])
        subset_name = batch.get("subset_name", [None] * n)[i]

        if not isinstance(messages_raw, list):
            continue

        # Strip the system message — tools are stored separately in
        # tool_catalog and the tokenizer's chat_template will render
        # them natively.  We do NOT use a custom system prompt.
        msgs = list(messages_raw)
        if msgs and msgs[0].get("role") == "system":
            msgs = msgs[1:]

        # Normalize available_tools to a list (already parsed above)
        if isinstance(tools_raw, list):
            tool_catalog_list = tools_raw
        elif tools_raw is not None:
            tool_catalog_list = [tools_raw]
        else:
            tool_catalog_list = []

        history: list = []
        turn_idx = 0
        n_msgs = len(msgs)
        j = 0

        while j < n_msgs:
            msg = msgs[j]
            role = msg.get("role")
            content = msg.get("content", "")
            function_call = msg.get("function_call")

            if role == "assistant" and function_call:
                if toolcall_mode == "multi":
                    # ── merge all consecutive tool-call turns ───────
                    # Collect every assistant tool call, skipping any
                    # interleaved function-result messages, until the
                    # next non-tool-call turn (user / text assistant).
                    fcs: list = []
                    k = j
                    while k < n_msgs:
                        m = msgs[k]
                        m_fc = m.get("function_call")
                        if m.get("role") == "assistant" and m_fc:
                            calls = m_fc if isinstance(m_fc, list) else [m_fc]
                            fcs.extend(calls)
                            k += 1
                            while k < n_msgs and msgs[k].get("role") == "function":
                                k += 1
                        else:
                            break

                    chosen = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [_normalize_function_call(fc) for fc in fcs],
                    }
                    _emit_turn(
                        result,
                        conv_id,
                        turn_idx,
                        history,
                        tool_catalog_list,
                        chosen,
                        subset_name,
                        toolcall_mode=toolcall_mode,
                    )
                    turn_idx += 1

                    # Append every collected tool call and its result to
                    # history so subsequent turns have the full context.
                    for m in msgs[j:k]:
                        m_fc = m.get("function_call")
                        if m.get("role") == "assistant" and m_fc:
                            history.append(_assistant_tool_call_message(m_fc))
                        elif m.get("role") == "function":
                            history.append(
                                {"role": "tool", "content": m.get("content", "")}
                            )
                    j = k
                    continue

                # ── single mode: one turn per tool-call message ─────
                chosen = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [_normalize_function_call(function_call)],
                }
                _emit_turn(
                    result,
                    conv_id,
                    turn_idx,
                    history,
                    tool_catalog_list,
                    chosen,
                    subset_name,
                    toolcall_mode=toolcall_mode,
                )
                turn_idx += 1

                # Append chosen to history for subsequent turns
                history.append(_assistant_tool_call_message(function_call))

            elif role == "function":
                # Toucan uses "function" → canonical uses "tool"
                history.append({"role": "tool", "content": content})

            elif role == "assistant" and content:
                # Text-only assistant message (no function_call)
                history.append({"role": "assistant", "content": content})

            else:
                # user, or any other role
                history.append({"role": role, "content": content})

            j += 1

    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


class ToucanLoader(BaseDataLoader):
    """Load a quality-filtered subset of the Toucan-1.5M dataset,
    normalized to canonical format.

    Parameters
    ----------
    dataset_id:
        Hugging Face dataset repository ID. Defaults to Toucan-1.5M.
    dataset_revision:
        Optional immutable Hub commit for reproducible source loading.
    size:
        Maximum number of conversation rows before explosion into
        canonical turns.  ``None`` means no limit.
    streaming:
        Stream and materialize only the first ``size`` matching conversations.
        Requires a finite ``size`` and is intended for smoke tests.
    use_viewer_api:
        Fetch bounded rows from the Hub Dataset Viewer. Intended only for a
        small, revision-checked smoke test.
    dataset_name:
        Toucan subset name passed to
        ``datasets.load_dataset("Agent-Ark/Toucan-1.5M", name=...)``.
    overall_score_threshold:
        Minimum ``overall_score`` from the Toucan quality assessment.
    desired_tools_used_percentage_threshold:
        Required ``desired_tools_used_percentage`` (1.0 = 100%).
    order_correctness_required:
        Require ``order_correctness == True``.
    subset_name_required:
        Toucan ``subset_name`` to filter on (e.g.
        ``"single-turn-original"``, ``"multi-turn"``).
    use_disk_cache:
        Whether to use the HuggingFace ``datasets`` on-disk cache.
    toolcall_mode:
        How to split assistant tool-call messages into canonical turns:

        * ``"single"`` — one canonical row per tool-call message.
          Each ``chosen_response`` contains exactly one tool call.
        * ``"multi"`` — consecutive tool-call messages (with any
          interleaved function results) are merged into one row.
          ``chosen_response`` contains all merged calls, training
          the model to emit multiple calls in one generation.

    Any extra keyword arguments are forwarded to
    :class:`BaseDataLoader` and stored in ``self._kwargs`` for
    potential use by subclasses or downstream code.
    """

    def __init__(
        self,
        dataset_id: str = "Agent-Ark/Toucan-1.5M",
        dataset_revision: Optional[str] = None,
        size: Optional[int] = None,
        streaming: bool = False,
        use_viewer_api: bool = False,
        viewer_offset: int = 0,
        viewer_page_size: int = 20,
        viewer_max_rows: int = 100,
        dataset_name: str = "Qwen3",
        overall_score_threshold: float = 4.5,
        desired_tools_used_percentage_threshold: float = 1.0,
        order_correctness_required: bool = True,
        subset_name_required: str = "multi-turn",
        use_disk_cache: bool = True,
        toolcall_mode: str = "single",
        **extra_kwargs,
    ):
        super().__init__(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            size=size,
            streaming=streaming,
            use_viewer_api=use_viewer_api,
            viewer_offset=viewer_offset,
            viewer_page_size=viewer_page_size,
            viewer_max_rows=viewer_max_rows,
            dataset_name=dataset_name,
            overall_score_threshold=overall_score_threshold,
            desired_tools_used_percentage_threshold=desired_tools_used_percentage_threshold,
            order_correctness_required=order_correctness_required,
            subset_name_required=subset_name_required,
            use_disk_cache=use_disk_cache,
            toolcall_mode=toolcall_mode,
            **extra_kwargs,
        )

    def load(self) -> CanonicalDataset:
        dataset_id = self._kwargs.get("dataset_id", "Agent-Ark/Toucan-1.5M")
        dataset_revision = self._kwargs.get("dataset_revision")
        size = self._kwargs.get("size")
        streaming = self._kwargs.get("streaming", False)
        use_viewer_api = self._kwargs.get("use_viewer_api", False)
        viewer_offset = self._kwargs.get("viewer_offset", 0)
        viewer_page_size = self._kwargs.get("viewer_page_size", 20)
        viewer_max_rows = self._kwargs.get("viewer_max_rows", 100)
        dataset_name = self._kwargs.get("dataset_name", "Qwen3")
        overall_score_threshold = self._kwargs.get("overall_score_threshold", 4.5)
        desired_tools_used_percentage_threshold = self._kwargs.get(
            "desired_tools_used_percentage_threshold", 1.0
        )
        order_correctness_required = self._kwargs.get(
            "order_correctness_required", True
        )
        subset_name_required = self._kwargs.get("subset_name_required", "multi-turn")
        use_disk_cache = self._kwargs.get("use_disk_cache", True)
        toolcall_mode = self._kwargs.get("toolcall_mode", "single")

        # Build a closure so the mode is lexically captured — no
        # reliance on datasets.map(…, fn_kwargs=…) which is fragile
        # and easy to drop during refactors.
        def _convert_batch(batch: dict) -> dict:
            return _toucan_to_canonical_batch(batch, toolcall_mode=toolcall_mode)

        # 1. Quality-filter (HF Dataset — Arrow-native, fast)
        filtered = build_sanitized_toucan(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            size=size,
            streaming=streaming,
            use_viewer_api=use_viewer_api,
            viewer_offset=viewer_offset,
            viewer_page_size=viewer_page_size,
            viewer_max_rows=viewer_max_rows,
            dataset_name=dataset_name,
            overall_score_threshold=overall_score_threshold,
            desired_tools_used_percentage_threshold=desired_tools_used_percentage_threshold,
            order_correctness_required=order_correctness_required,
            subset_name_required=subset_name_required,
            use_disk_cache=use_disk_cache,
        )

        # 2. Select only the columns needed for conversion
        keep = [
            c
            for c in ["uuid", "messages", "available_tools", "subset_name"]
            if c in filtered.column_names
        ]
        filtered = filtered.select_columns(keep)

        # 3. Explode to canonical format (batched Arrow map with
        #    multiprocessing — same performance as the old
        #    preprocess_toolcall_prompts path)
        #
        # ``messages`` now contains OpenAI-style ``tool_calls`` with
        # heterogeneous nested ``arguments`` dicts, which Arrow's struct
        # inference cannot represent.  Declare the column as ``Json``
        # up-front so the schema is stable; it round-trips back to a
        # list of dicts on read.
        from datasets import Features, Json, Value

        _features = Features(
            {
                "conv_id": Value("string"),
                "turn_id": Value("string"),
                "conv_tracking_id": Value("string"),
                "custom_system_prompt": Value("string"),
                "is_custom_system_prompt_used": Value("bool"),
                "messages": Json(),
                "tool_catalog": Value("string"),
                "chosen_response": Value("string"),
                "rejected_response": Value("string"),
                "metadata": Value("string"),
            }
        )
        canonical_ds = filtered.map(
            _convert_batch,
            batched=True,
            batch_size=512,
            remove_columns=filtered.column_names,
            load_from_cache_file=use_disk_cache,
            desc="Toucan → canonical format",
            features=_features,
        )

        return CanonicalDataset.from_hf_dataset(canonical_ds)
