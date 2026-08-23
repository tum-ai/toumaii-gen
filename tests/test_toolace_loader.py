"""Regression tests for ToolACE canonical conversion."""

import json

from datasets import Dataset

from tool_calling_data_generation.data_generators.loaders import toolace as toolace_module
from tool_calling_data_generation.data_generators.loaders.toolace import (
    ToolACELoader,
    _toolace_to_canonical_batch,
    split_toolace_dataset_by_mode,
)
from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset


def _sanitized_row():
    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "dict",
                "properties": {"city": {"type": "str"}},
                "required": ["city"],
            },
        }
    ]
    conversations = [
        {"role": "user", "content": "Weather in Berlin and Paris?"},
        {
            "role": "assistant",
            "content": json.dumps(
                [
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                ]
            ),
        },
        {
            "role": "tool",
            "content": json.dumps(
                [
                    {
                        "name": "get_weather",
                        "results": {"city": "Berlin", "temperature": "18 C"},
                    },
                    {
                        "name": "get_weather",
                        "results": {"city": "Paris", "temperature": "19 C"},
                    },
                ]
            ),
        },
        {"role": "assistant", "content": "Both cities are mild."},
        {"role": "user", "content": "And Rome?"},
        {
            "role": "assistant",
            "content": json.dumps(
                [{"name": "get_weather", "arguments": {"city": "Rome"}}]
            ),
        },
    ]
    return {
        "conversations": conversations,
        "tools": json.dumps(tools),
    }


def test_toolace_emits_sequential_and_parallel_views_with_history():
    row = _sanitized_row()
    result = _toolace_to_canonical_batch(
        {key: [value] for key, value in row.items()},
        indices=[7],
    )

    assert result["conv_id"] == ["toolace_7"] * 4
    assert result["turn_id"] == ["single_0", "single_1", "single_2", "multi_0"]
    assert result["messages"][0] == [
        {"role": "user", "content": "Weather in Berlin and Paris?"}
    ]

    first_sequential = json.loads(result["chosen_response"][0])[0]
    assert len(first_sequential["tool_calls"]) == 1
    assert first_sequential["tool_calls"][0]["function"]["arguments"] == {
        "city": "Berlin"
    }

    second_history = result["messages"][1]
    assert second_history[1]["tool_calls"] == first_sequential["tool_calls"]
    first_result = json.loads(second_history[2]["content"])[0]
    assert first_result["results"]["city"] == "Berlin"

    later_history = result["messages"][2]
    assert later_history[-2] == {
        "role": "assistant",
        "content": "Both cities are mild.",
    }
    assert later_history[-1] == {"role": "user", "content": "And Rome?"}

    parallel = json.loads(result["chosen_response"][3])[0]
    assert len(parallel["tool_calls"]) == 2
    assert [
        call["function"]["arguments"]["city"] for call in parallel["tool_calls"]
    ] == ["Berlin", "Paris"]

    catalog = json.loads(result["tool_catalog"][0])
    assert catalog[0]["type"] == "function"
    schema = catalog[0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"]["city"]["type"] == "string"

    sequential_metadata = json.loads(result["metadata"][0])
    assert sequential_metadata == {
        "source_dataset": "ToolACE",
        "subset_name": "multiturn_singlecall",
        "source_subset_name": "default",
        "turn_mode": "multi",
        "toolcall_mode": "single",
        "num_user_turns": 2,
        "num_tool_calls": 1,
    }
    parallel_metadata = json.loads(result["metadata"][3])
    assert parallel_metadata["subset_name"] == "multiturn_multicall"
    assert parallel_metadata["turn_mode"] == "multi"
    assert parallel_metadata["toolcall_mode"] == "multi"
    assert parallel_metadata["num_tool_calls"] == 2


def test_toolace_single_turn_modes_are_derived_from_natural_targets():
    row = _sanitized_row()
    one_call = {
        **row,
        "conversations": [
            {"role": "user", "content": "Weather in Berlin?"},
            {
                "role": "assistant",
                "content": json.dumps(
                    [{"name": "get_weather", "arguments": {"city": "Berlin"}}]
                ),
            },
        ],
    }
    two_calls = {
        **row,
        "conversations": [
            {"role": "user", "content": "Weather in Berlin and Paris?"},
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {"name": "get_weather", "arguments": {"city": "Berlin"}},
                        {"name": "get_weather", "arguments": {"city": "Paris"}},
                    ]
                ),
            },
        ],
    }

    result = _toolace_to_canonical_batch(
        {
            "conversations": [one_call["conversations"], two_calls["conversations"]],
            "tools": [one_call["tools"], two_calls["tools"]],
        },
        indices=[1, 2],
    )

    metadata = [json.loads(value) for value in result["metadata"]]
    assert [value["subset_name"] for value in metadata].count(
        "singleturn_singlecall"
    ) == 3
    assert [value["subset_name"] for value in metadata].count(
        "singleturn_multicall"
    ) == 1


def test_sequential_source_chain_is_also_emitted_as_parallel_target():
    row = _sanitized_row()
    row["conversations"] = [
        {"role": "user", "content": "Compare Berlin and Paris."},
        {
            "role": "assistant",
            "content": json.dumps(
                [{"name": "get_weather", "arguments": {"city": "Berlin"}}]
            ),
        },
        {"role": "tool", "content": "Berlin: 18 C"},
        {
            "role": "assistant",
            "content": json.dumps(
                [{"name": "get_weather", "arguments": {"city": "Paris"}}]
            ),
        },
        {"role": "tool", "content": "Paris: 19 C"},
        {"role": "assistant", "content": "Paris is slightly warmer."},
    ]

    result = _toolace_to_canonical_batch(
        {key: [value] for key, value in row.items()},
        indices=[9],
    )

    assert result["turn_id"] == ["single_0", "single_1", "multi_0"]
    second_prompt = result["messages"][1]
    assert second_prompt[-1] == {"role": "tool", "content": "Berlin: 18 C"}
    parallel = json.loads(result["chosen_response"][2])[0]
    assert [
        call["function"]["arguments"]["city"] for call in parallel["tool_calls"]
    ] == ["Berlin", "Paris"]


def test_toolace_upload_partition_contains_all_four_mode_splits():
    dataset = Dataset.from_dict(
        {
            "metadata": [
                json.dumps({"subset_name": "singleturn_singlecall"}),
                json.dumps({"subset_name": "singleturn_multicall"}),
                json.dumps({"subset_name": "multiturn_singlecall"}),
                json.dumps({"subset_name": "multiturn_multicall"}),
            ],
            "value": [1, 2, 3, 4],
        }
    )

    splits = split_toolace_dataset_by_mode(dataset)

    assert list(splits) == [
        "singleturn_singlecall",
        "singleturn_multicall",
        "multiturn_singlecall",
        "multiturn_multicall",
    ]
    assert {name: split["value"] for name, split in splits.items()} == {
        "singleturn_singlecall": [1],
        "singleturn_multicall": [2],
        "multiturn_singlecall": [3],
        "multiturn_multicall": [4],
    }


def test_loader_wraps_sanitizer_and_returns_canonical(monkeypatch):
    seen = {}

    def fake_sanitizer(size=None):
        seen["size"] = size
        return Dataset.from_list([_sanitized_row()])

    monkeypatch.setattr(toolace_module, "build_sanitized_toolace", fake_sanitizer)
    loaded = ToolACELoader(size=3, use_disk_cache=False).load()

    assert isinstance(loaded, CanonicalDataset)
    assert len(loaded) == 4
    assert seen["size"] == 3


def test_exports_accept_mixed_list_and_scalar_json_values(monkeypatch):
    scalar_row = _sanitized_row()
    scalar_tools = json.loads(scalar_row["tools"])
    scalar_tools[0]["parameters"]["properties"]["city"]["default"] = "Berlin"
    scalar_row["tools"] = json.dumps(scalar_tools)

    list_row = json.loads(json.dumps(scalar_row))
    list_tools = json.loads(list_row["tools"])
    list_tools[0]["parameters"]["properties"]["city"]["default"] = [
        "Rome",
        "Milan",
    ]
    list_row["tools"] = json.dumps(list_tools)
    list_row["conversations"] = [
        {"role": "user", "content": "Weather in two cities?"},
        {
            "role": "assistant",
            "content": json.dumps(
                [
                    {
                        "name": "get_weather",
                        "arguments": {"city": ["Rome", "Milan"]},
                    }
                ]
            ),
        },
    ]

    monkeypatch.setattr(
        toolace_module,
        "build_sanitized_toolace",
        lambda size=None: Dataset.from_list([scalar_row, list_row]),
    )
    loaded = ToolACELoader(use_disk_cache=False).load()

    dpo = loaded.to_dpo_dataset()
    sft = loaded.to_sft_dataset()

    assert len(dpo) == len(loaded) == 5
    assert len(sft) == len(loaded)
    assert isinstance(
        dpo[-1]["chosen"][0]["tool_calls"][0]["function"]["arguments"]["city"],
        list,
    )
    assert isinstance(
        dpo[0]["chosen"][0]["tool_calls"][0]["function"]["arguments"]["city"],
        str,
    )

