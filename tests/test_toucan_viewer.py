import json

from tool_calling_data_generation.data_generators.loaders import toucan


class _Response:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return {"rows": self._rows}


def _row(row_idx, score):
    return {
        "row_idx": row_idx,
        "truncated_cells": [],
        "row": {
            "uuid": str(row_idx),
            "messages": "[]",
            "available_tools": "[]",
            "subset_name": "multi-turn",
            "response_quality_assessment": json.dumps(
                {
                    "overall_score": score,
                    "desired_tools_used_percentage": 1.0,
                    "order_correctness": True,
                }
            ),
        },
    }


def test_viewer_mode_filters_a_bounded_slice(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return _Response([_row(400000, 3.0), _row(400001, 4.5), _row(400002, 5.0)])

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    result = toucan.build_sanitized_toucan(
        dataset_id="org/toucan",
        dataset_name="Qwen3",
        size=2,
        use_viewer_api=True,
        viewer_offset=400000,
        viewer_page_size=20,
        viewer_max_rows=20,
        subset_name_required="multi-turn",
    )

    assert result["uuid"] == ["400001", "400002"]
    assert calls[0][1]["offset"] == 400000
    assert calls[0][1]["length"] == 20

