import pytest

from tool_calling_data_generation.rewards import RewardRegistry


def test_toolrl_is_registered_and_scores_an_exact_call():
    registry = RewardRegistry()
    assert registry.list_all() == ["curriculum", "toolrl"]

    expected_call = {"name": "get_weather", "arguments": {"city": "Berlin"}}
    result = registry.get("toolrl").compute(
        '[{"name": "get_weather", "arguments": {"city": "Berlin"}}]',
        {"expected_calls": [expected_call]},
    )

    assert result.score == pytest.approx(1.0)
    assert result.components["r_correct"] > 0
