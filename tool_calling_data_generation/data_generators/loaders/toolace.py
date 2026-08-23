"""Load sanitized ToolACE conversations into the canonical DPO format."""

import json
from typing import Any, Optional

from datasets import Dataset, Features, Json, Value

from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader
from tool_calling_data_generation.data_generators.models.canonical_format import (
    CanonicalDataset,
    _CANONICAL_COLUMNS,
)
from tool_calling_data_generation.toolace_sanitizer import build_sanitized_toolace


_SCHEMA_TYPES = {
    "dict": "object",
    "list": "array",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

TOOLACE_MODE_SPLITS = (
    "singleturn_singlecall",
    "singleturn_multicall",
    "multiturn_singlecall",
    "multiturn_multicall",
)


def _parse_json(value: Any) -> Any:
    """Parse JSON strings and leave already-decoded values unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_schema(value: Any) -> Any:
    """Recursively normalize Python-style ToolACE JSON Schema types."""
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = {key: _normalize_schema(item) for key, item in value.items()}
    if isinstance(value.get("type"), str):
        value["type"] = _SCHEMA_TYPES.get(value["type"], value["type"])
    return value


def _normalize_tool(tool: Any) -> Optional[dict]:
    """Wrap a bare ToolACE function in OpenAI's tool-definition shape."""
    tool = _parse_json(tool)
    if not isinstance(tool, dict):
        return None
    tool = _normalize_schema(tool)
    function = tool.get("function")
    if not isinstance(function, dict):
        function = {
            key: value
            for key, value in tool.items()
            if key in {"name", "description", "parameters", "strict"}
        }
    if not function.get("name"):
        return None
    return {"type": "function", "function": function}


def _parse_tool_calls(content: Any) -> Optional[list[dict]]:
    """Return sanitized assistant tool calls, or None for plain text."""
    calls = _parse_json(content)
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list) or not calls:
        return None
    if not all(
        isinstance(call, dict)
        and (
            "name" in call
            or (
                isinstance(call.get("function"), dict)
                and "name" in call["function"]
            )
        )
        for call in calls
    ):
        return None
    return calls


def _normalize_tool_call(call: dict) -> dict:
    """Normalize one ToolACE call to an OpenAI assistant tool call."""
    function = call.get("function")
    if not isinstance(function, dict):
        function = call
    arguments = _parse_json(
        function.get("arguments", function.get("parameters", {}))
    )
    return {
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": arguments,
        },
    }


def _tool_call_message(calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_normalize_tool_call(call) for call in calls],
    }


def _canonical_features() -> Features:
    return Features(
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


def split_toolace_dataset_by_mode(dataset: Dataset) -> dict[str, Dataset]:
    """Partition canonical ToolACE rows into natural turn/call mode splits."""
    indices = {split_name: [] for split_name in TOOLACE_MODE_SPLITS}
    for row_idx, metadata in enumerate(dataset["metadata"]):
        metadata = _parse_json(metadata)
        split_name = metadata.get("subset_name") if isinstance(metadata, dict) else None
        if split_name in indices:
            indices[split_name].append(row_idx)
    return {
        split_name: dataset.select(row_indices)
        for split_name, row_indices in indices.items()
    }


def _emit_turn(
    result: dict,
    conv_id: str,
    turn_idx: int,
    history: list,
    tool_catalog: list,
    chosen: dict,
    source_subset_name: str,
    num_user_turns: int,
    num_tool_calls: int,
) -> None:
    """Append one canonical generation example."""
    turn_mode = "multi" if num_user_turns > 1 else "single"
    toolcall_mode = "multi" if num_tool_calls > 1 else "single"
    subset_name = f"{turn_mode}turn_{toolcall_mode}call"
    canonical_turn_id = f"{toolcall_mode}_{turn_idx}"
    result["conv_id"].append(conv_id)
    result["turn_id"].append(canonical_turn_id)
    result["conv_tracking_id"].append(f"{conv_id}_{canonical_turn_id}")
    result["custom_system_prompt"].append(None)
    result["is_custom_system_prompt_used"].append(False)
    result["messages"].append(list(history))
    result["tool_catalog"].append(json.dumps(tool_catalog, ensure_ascii=False))
    result["chosen_response"].append(json.dumps([chosen], ensure_ascii=False))
    result["rejected_response"].append("[]")
    result["metadata"].append(
        json.dumps(
            {
                "source_dataset": "ToolACE",
                "subset_name": subset_name,
                "source_subset_name": source_subset_name,
                "turn_mode": turn_mode,
                "toolcall_mode": toolcall_mode,
                "num_user_turns": num_user_turns,
                "num_tool_calls": num_tool_calls,
            }
        )
    )


def _message_parts(message: dict) -> tuple[Any, Any]:
    role = message.get("role", message.get("from"))
    role = "user" if role == "human" else role
    content = message.get("content", message.get("value", ""))
    return role, content


def _append_non_call_message(history: list, role: Any, content: Any) -> None:
    if role in {"tool", "function"}:
        history.append({"role": "tool", "content": content})
    elif role == "assistant" and content:
        history.append({"role": "assistant", "content": content})
    elif role not in {None, "system"}:
        history.append({"role": role, "content": content})


def _split_tool_results(content: Any, calls: list[dict]) -> Optional[list[dict]]:
    """Align a combined ToolACE result list with its individual calls."""
    results = _parse_json(content)
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        return None

    by_name: dict[str, list[dict]] = {}
    for result in results:
        name = result.get("name")
        if isinstance(name, str):
            by_name.setdefault(name, []).append(result)

    aligned = []
    for call in calls:
        normalized = _normalize_tool_call(call)
        name = normalized["function"]["name"]
        candidates = by_name.get(name, [])
        if not candidates:
            return None
        result = candidates.pop(0)
        aligned.append(
            {
                "role": "tool",
                "content": json.dumps([result], ensure_ascii=False),
            }
        )
    return aligned


def _emit_singlecall_view(
    result: dict,
    conv_id: str,
    messages: list,
    tool_catalog: list,
    source_subset_name: str,
    num_user_turns: int,
) -> None:
    """Emit a sequential view containing one chosen tool call per row."""
    history = []
    turn_idx = 0
    j = 0
    while j < len(messages):
        message = messages[j]
        if not isinstance(message, dict):
            j += 1
            continue
        role, content = _message_parts(message)
        calls = _parse_tool_calls(content) if role == "assistant" else None
        if not calls:
            _append_non_call_message(history, role, content)
            j += 1
            continue

        aligned_results = None
        if len(calls) > 1 and j + 1 < len(messages):
            next_message = messages[j + 1]
            if isinstance(next_message, dict):
                next_role, next_content = _message_parts(next_message)
                if next_role in {"tool", "function"}:
                    aligned_results = _split_tool_results(next_content, calls)

        if len(calls) > 1 and aligned_results is None:
            # A combined result cannot be safely assigned to individual calls.
            # Keep each single-call target independent, then retain the source
            # parallel call in history for subsequent user turns.
            base_history = list(history)
            for call in calls:
                chosen = _tool_call_message([call])
                _emit_turn(
                    result,
                    conv_id,
                    turn_idx,
                    base_history,
                    tool_catalog,
                    chosen,
                    source_subset_name,
                    num_user_turns,
                    1,
                )
                turn_idx += 1
            history.append(_tool_call_message(calls))
            j += 1
            continue

        for call_idx, call in enumerate(calls):
            chosen = _tool_call_message([call])
            _emit_turn(
                result,
                conv_id,
                turn_idx,
                history,
                tool_catalog,
                chosen,
                source_subset_name,
                num_user_turns,
                1,
            )
            turn_idx += 1
            history.append(chosen)
            if aligned_results is not None:
                history.append(aligned_results[call_idx])

        if aligned_results is not None:
            j += 1  # the following combined tool result was consumed above
        j += 1


def _emit_multicall_view(
    result: dict,
    conv_id: str,
    messages: list,
    tool_catalog: list,
    source_subset_name: str,
    num_user_turns: int,
) -> None:
    """Emit parallel targets by merging call chains separated by tool results."""
    history = []
    turn_idx = 0
    j = 0
    while j < len(messages):
        message = messages[j]
        if not isinstance(message, dict):
            j += 1
            continue
        role, content = _message_parts(message)
        first_calls = _parse_tool_calls(content) if role == "assistant" else None
        if not first_calls:
            _append_non_call_message(history, role, content)
            j += 1
            continue

        calls = []
        source_chain = []
        k = j
        while k < len(messages):
            chain_message = messages[k]
            if not isinstance(chain_message, dict):
                break
            chain_role, chain_content = _message_parts(chain_message)
            chain_calls = (
                _parse_tool_calls(chain_content)
                if chain_role == "assistant"
                else None
            )
            if chain_calls:
                calls.extend(chain_calls)
                source_chain.append(_tool_call_message(chain_calls))
                k += 1
                continue
            if chain_role in {"tool", "function"}:
                source_chain.append({"role": "tool", "content": chain_content})
                k += 1
                continue
            break

        if len(calls) > 1:
            chosen = _tool_call_message(calls)
            _emit_turn(
                result,
                conv_id,
                turn_idx,
                history,
                tool_catalog,
                chosen,
                source_subset_name,
                num_user_turns,
                len(calls),
            )
            turn_idx += 1

        history.extend(source_chain)
        j = k


def _toolace_to_canonical_batch(batch: dict, indices: list[int]) -> dict:
    """Explode ToolACE conversations into sequential and parallel views."""
    result = {column: [] for column in _CANONICAL_COLUMNS}
    conversations = batch.get("conversations", [])

    for row_idx, conversation_raw in enumerate(conversations):
        source_idx = indices[row_idx]
        ids = batch.get("uuid")
        conv_id = (
            str(ids[row_idx])
            if ids and ids[row_idx]
            else f"toolace_{source_idx}"
        )

        tools_raw = _parse_json(
            batch.get("tools", [None] * len(conversations))[row_idx]
        )
        if isinstance(tools_raw, dict):
            tools_raw = [tools_raw]
        tool_catalog = []
        if isinstance(tools_raw, list):
            tool_catalog = [
                normalized
                for tool in tools_raw
                if (normalized := _normalize_tool(tool)) is not None
            ]

        messages = _parse_json(conversation_raw)
        if not isinstance(messages, list):
            continue
        subset_values = batch.get("subset_name")
        source_subset_name = (
            str(subset_values[row_idx])
            if subset_values and subset_values[row_idx]
            else "default"
        )
        num_user_turns = sum(
            1
            for message in messages
            if isinstance(message, dict)
            and message.get("role", message.get("from")) in {"user", "human"}
        )

        _emit_singlecall_view(
            result,
            conv_id,
            messages,
            tool_catalog,
            source_subset_name,
            num_user_turns,
        )
        _emit_multicall_view(
            result,
            conv_id,
            messages,
            tool_catalog,
            source_subset_name,
            num_user_turns,
        )

    return result


class ToolACELoader(BaseDataLoader):
    """Load and normalize Team-ACE/ToolACE for DPO generation.

    The size parameter limits raw conversations before sanitization. Malformed
    source rows can be removed by the sanitizer, so the final count may be
    smaller.
    """

    def __init__(
        self,
        size: Optional[int] = None,
        use_disk_cache: bool = True,
        **extra_kwargs,
    ):
        super().__init__(
            size=size,
            use_disk_cache=use_disk_cache,
            **extra_kwargs,
        )

    def load(self) -> CanonicalDataset:
        sanitized: Dataset = build_sanitized_toolace(size=self._kwargs.get("size"))
        use_disk_cache = self._kwargs.get("use_disk_cache", True)
        features = _canonical_features()

        if not len(sanitized):
            empty = Dataset.from_dict(
                {column: [] for column in _CANONICAL_COLUMNS},
                features=features,
            )
            return CanonicalDataset.from_hf_dataset(empty)

        required = {"conversations", "tools"}
        missing = required.difference(sanitized.column_names)
        if missing:
            raise ValueError(
                "Sanitized ToolACE dataset is missing required columns: "
                + ", ".join(sorted(missing))
            )

        keep = [
            column
            for column in ["uuid", "conversations", "tools", "subset_name"]
            if column in sanitized.column_names
        ]
        sanitized = sanitized.select_columns(keep)

        canonical = sanitized.map(
            _toolace_to_canonical_batch,
            batched=True,
            batch_size=512,
            with_indices=True,
            remove_columns=sanitized.column_names,
            load_from_cache_file=use_disk_cache,
            desc="ToolACE -> canonical format",
            features=features,
        )
        return CanonicalDataset.from_hf_dataset(canonical)
