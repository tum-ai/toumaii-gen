"""
Shared utility functions used across reward implementations.

These are stateless helpers for text extraction, JSON parsing, and
string-matching that both ``ToolRLReward`` and ``CurriculumReward`` rely on.
"""

import json
import re
from typing import Any, Dict, List

from tool_calling_data_generation.rewards.config import RewardConfig

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# XML / tag extraction
# ---------------------------------------------------------------------------


def extract_content(text: str, tag: str) -> List[str]:
    """Return the inner text of every ``<tag>...</tag>`` occurrence."""
    pattern = f"<{tag}[^>]*>(.*?)</{tag}>"
    return re.findall(pattern, text, re.DOTALL)


def extract_tool_call_contents(
    completion: str,
) -> List[
    str
]:  # TODO: use the tokenizer to extract the tool call content instead of regex
    """Extract content from ``<tool_call>`` or ``<tool>`` blocks."""
    contents = re.findall(r"<tool_call[^>]*>(.*?)</tool_call>", completion, re.DOTALL)
    if contents:
        return contents
    return re.findall(r"<tool[^>]*>(.*?)</tool>", completion, re.DOTALL)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def strict_match(v1: Any, v2: Any) -> bool:
    """Type- AND value-strict equality (``5 != 5.0``)."""
    if type(v1) is not type(v2):
        return False
    return v1 == v2


def soft_match(v1: Any, v2: Any) -> bool:
    """Lenient equality used by legacy ToolRL reward.

    Numbers are compared with a small epsilon; strings are lower-cased and
    stripped before comparison.
    """
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(v1 - v2) < 1e-3
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.strip().lower() == v2.strip().lower()
    return v1 == v2


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def is_valid_json(text: str) -> bool:
    """Return True if *text* is parseable JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _parse_tool_calls_from_raw(text: str) -> List[Dict[str, Any]]:
    """Fallback: scan raw text for JSON objects that contain a ``"name"`` key."""
    results: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end_idx = decoder.raw_decode(text[i:])
                if isinstance(obj, dict) and "name" in obj:
                    results.append(obj)
                elif isinstance(obj, list):
                    results.extend(
                        d for d in obj if isinstance(d, dict) and "name" in d
                    )
                i += end_idx
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1
    return results


def parse_tool_calls(completion: str) -> List[Dict[str, Any]]:
    """Parse tool calls from ``<tool_call>`` / ``<tool>`` blocks.

    Falls back to raw-JSON scanning when no XML tags are present.
    """
    tool_contents = extract_tool_call_contents(completion)
    if not tool_contents:
        return _parse_tool_calls_from_raw(completion)

    results: List[Dict[str, Any]] = []
    for content in tool_contents:
        content = content.strip()
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if json_repair:
                try:
                    data = json_repair.loads(content)
                except Exception:
                    continue
            else:
                continue

        if isinstance(data, list):
            results.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            results.append(data)

    return results


# ---------------------------------------------------------------------------
# Format reward (shared by ToolRLReward AND CurriculumReward)
# ---------------------------------------------------------------------------


def calculate_format_reward(
    completion: str,
    config: "RewardConfig | None" = None,
) -> float:
    """R_format: scores the structural quality of tool-call output.

    Handles three cases (in order):

    1. **XML tags present** (``<tool_call>`` / ``<tool>``) — checks JSON
       validity inside each tag block.
    2. **No tags, but valid raw JSON** containing tool calls — returns
       ``format_raw_json_bonus`` (default 1.0).
    3. **Neither** — returns 0.0.

    Returns a float in [0.0, 1.0].
    """
    from tool_calling_data_generation.rewards.config import DEFAULT_REWARD_CONFIG

    if config is None:
        config = DEFAULT_REWARD_CONFIG

    # -- Case 1: XML tags present ----------------------------------------
    has_open = "<tool_call>" in completion or "<tool>" in completion
    has_close = "</tool_call>" in completion or "</tool>" in completion

    if has_open or has_close:
        if not has_open or not has_close:
            return 0.0  # mismatched tags

        tool_contents = extract_tool_call_contents(completion)
        if not tool_contents:
            return config.format_tags_bonus

        all_valid = True
        any_repairable = False
        for content in tool_contents:
            content = content.strip()
            if not content:
                all_valid = False
                continue
            if is_valid_json(content):
                continue
            all_valid = False
            if json_repair is not None:
                try:
                    json_repair.loads(content)
                    any_repairable = True
                except Exception:
                    pass

        if all_valid:
            return config.format_json_valid_bonus
        if any_repairable:
            return config.format_json_repair_bonus
        return config.format_tags_bonus

    # -- Case 2: No XML tags — check for raw JSON tool calls -------------
    stripped = completion.strip()
    if stripped:
        try:
            data = json.loads(stripped)
            if _looks_like_tool_call(data):
                return config.format_raw_json_bonus
        except (json.JSONDecodeError, ValueError):
            pass

    # -- Case 3: Nothing recognisable ------------------------------------
    return 0.0


def _looks_like_tool_call(data) -> bool:
    """Return True if *data* looks like a tool-call JSON structure."""
    if isinstance(data, dict):
        return "name" in data and "arguments" in data
    if isinstance(data, list) and len(data) > 0:
        return all(
            isinstance(d, dict) and "name" in d and "arguments" in d for d in data
        )
    return False
