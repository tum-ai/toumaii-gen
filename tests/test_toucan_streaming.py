import json

from tool_calling_data_generation.data_generators.loaders import toucan


class _FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, function, input_columns):
        return _FakeStream(
            [row for row in self.rows if function(*(row[name] for name in input_columns))]
        )

    def take(self, size):
        return iter(self.rows[:size])


def test_streaming_toucan_materializes_only_requested_matching_rows(monkeypatch):
    matching = {
        "uuid": "keep",
        "messages": [],
        "available_tools": "[]",
        "subset_name": "multi-turn",
        "response_quality_assessment": json.dumps(
            {
                "overall_score": 5.0,
                "desired_tools_used_percentage": 1.0,
                "order_correctness": True,
            }
        ),
    }
    rejected = {**matching, "uuid": "drop", "subset_name": "single-turn-original"}
    seen = {}

    def fake_load_dataset(dataset_id, name, split, streaming, revision):
        seen.update(
            dataset_id=dataset_id,
            name=name,
            split=split,
            streaming=streaming,
            revision=revision,
        )
        return _FakeStream([rejected, matching, {**matching, "uuid": "not-taken"}])

    monkeypatch.setattr(toucan, "load_dataset", fake_load_dataset)
    result = toucan.build_sanitized_toucan(
        dataset_id="org/test-toucan",
        dataset_name="Qwen3",
        size=1,
        streaming=True,
        subset_name_required="multi-turn",
    )

    assert seen == {
        "dataset_id": "org/test-toucan",
        "name": "Qwen3",
        "split": "train",
        "streaming": True,
        "revision": None,
    }
    assert len(result) == 1
    assert result[0]["uuid"] == "keep"
