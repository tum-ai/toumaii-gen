"""
Tests for the Toucan loader's tool-call rendering fix.

Regression test for the bug where assistant tool calls were serialised into
``messages`` as a JSON string in ``content`` (e.g.
``{"name": ..., "arguments": "..."}``) instead of the OpenAI-style
``tool_calls`` key.  Because ``apply_chat_template`` only renders the
model-specific tool-call syntax when an assistant message has a
``tool_calls`` field, the old encoding broke prompt rendering.
"""

import json

from tool_calling_data_generation.data_generators.loaders.toucan import (
    _assistant_tool_call_message,
    _toucan_to_canonical_batch,
)


def _msg(role, content="", function_call=None):
    m = {"role": role, "content": content}
    if function_call is not None:
        m["function_call"] = function_call
    return m


def _fc(name, args):
    return {"name": name, "arguments": args}


# ---------------------------------------------------------------------------
# _assistant_tool_call_message
# ---------------------------------------------------------------------------


def test_tool_call_message_parses_string_args():
    m = _assistant_tool_call_message(
        _fc("get_weather", '{"latitude": 45.5, "longitude": -122.6}')
    )
    assert m["role"] == "assistant"
    assert m["content"] == ""
    tc = m["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == {"latitude": 45.5, "longitude": -122.6}


def test_tool_call_message_preserves_malformed_args():
    """Unparseable arguments stay as strings (data preservation for DPO)."""
    m = _assistant_tool_call_message(_fc("bad_tool", "{not valid json"))
    assert m["tool_calls"][0]["function"]["arguments"] == "{not valid json"


def test_tool_call_message_stringifies_huge_ints():
    """ujson (HF Json feature) fails on ints >= 10**20; we stringify them.

    Regression test for the ``100!`` tool result (a 158-digit integer) that
    crashed ``build_ground_truth`` with "Value is too big!".
    """
    big = 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
    m = _assistant_tool_call_message(_fc("log", json.dumps({"x": big, "base": 10})))
    args = m["tool_calls"][0]["function"]["arguments"]
    assert args["x"] == str(big)
    assert args["base"] == 10  # small ints stay ints


def test_tool_call_message_nested_huge_ints():
    """Huge ints nested in lists/dicts are stringified too."""
    m = _assistant_tool_call_message(
        _fc("f", json.dumps({"items": [1, 10 ** 25, {"y": 10 ** 30}]}))
    )
    args = m["tool_calls"][0]["function"]["arguments"]
    assert args["items"][1] == str(10 ** 25)
    assert args["items"][2]["y"] == str(10 ** 30)
    assert args["items"][0] == 1


def test_tool_call_message_accepts_list():
    m = _assistant_tool_call_message(
        [_fc("a", "{}"), _fc("b", "{}")]
    )
    assert len(m["tool_calls"]) == 2
    assert [t["function"]["name"] for t in m["tool_calls"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# _toucan_to_canonical_batch — multi-turn history rendering
# ---------------------------------------------------------------------------


def _batch(messages):
    return {
        "uuid": ["conv-1"],
        "available_tools": ["[]"],
        "messages": [messages],
        "subset_name": ["multi-turn"],
    }


def test_history_tool_calls_use_tool_calls_key():
    """Assistant tool-call messages in history must carry ``tool_calls``,
    not a JSON-serialised string in ``content``."""
    raw = [
        _msg("user", "What's the weather?"),
        _msg("assistant", function_call=_fc("get_weather", '{"lat": 1.0}')),
        _msg("function", '{"temp": 21.6}'),
        _msg("assistant", "It's 21.6C."),
        _msg("user", "And for tomorrow?"),
        _msg("assistant", function_call=_fc("get_forecast", '{"lat": 1.0}')),
        _msg("function", '{"temp": 22.0}'),
    ]
    result = _toucan_to_canonical_batch(_batch(raw), toolcall_mode="single")

    # conv explodes into one turn per assistant tool-call message
    assert len(result["conv_id"]) == 2
    assert result["turn_id"] == ["0", "1"]

    # turn 0: history has no prior tool calls
    assert result["messages"][0] == [{"role": "user", "content": "What's the weather?"}]

    # turn 1: history includes turn 0's tool call in proper format
    history = result["messages"][1]
    assert history[0] == {"role": "user", "content": "What's the weather?"}
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == ""
    assert history[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {"lat": 1.0}
    assert history[2] == {"role": "tool", "content": '{"temp": 21.6}'}
    assert history[3] == {"role": "assistant", "content": "It's 21.6C."}
    assert history[4] == {"role": "user", "content": "And for tomorrow?"}


def test_single_turn_no_prior_tool_call_in_messages():
    """Turn 0 of a conversation has no tool call in its messages."""
    raw = [
        _msg("user", "What's the weather?"),
        _msg("assistant", function_call=_fc("get_weather", '{"lat": 1.0}')),
        _msg("function", '{"temp": 21.6}'),
    ]
    result = _toucan_to_canonical_batch(_batch(raw), toolcall_mode="single")
    assert result["messages"][0] == [{"role": "user", "content": "What's the weather?"}]
