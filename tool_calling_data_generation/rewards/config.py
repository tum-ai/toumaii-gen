"""Shared reward configuration.

A single ``RewardConfig`` dataclass controls all threshold / weight parameters
used across the format, correctness, and normalisation layers.  Import the
module-level ``DEFAULT_REWARD_CONFIG`` singleton for most use-cases.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    # ------------------------------------------------------------------
    # Format rewards
    # ------------------------------------------------------------------

    # Score for having both open and close tool-call tags
    format_tags_bonus: float = 0.2

    # Granular tag bonuses
    format_think_open_bonus: float = 0.1
    format_think_close_bonus: float = 0.1
    format_action_open_bonus: float = 0.1
    format_action_close_bonus: float = 0.1
    format_order_bonus: float = 0.2

    # Minimum tag score before JSON validity is checked
    format_min_reward_threshold: float = 0.4

    # JSON parsing rewards
    format_json_valid_bonus: float = 1.0
    format_json_repair_bonus: float = 0.8

    # Score awarded when the completion is valid JSON *without* XML tags
    # but still contains well-formed tool calls (``name`` + ``arguments``).
    # e.g. ``[{"name": "f", "arguments": {"x": 1}}]``
    format_raw_json_bonus: float = 1.0

    # ------------------------------------------------------------------
    # Correctness rewards
    # ------------------------------------------------------------------

    # Penalty per hallucinated / unmatched predicted call
    correctness_unmatched_penalty: float = 1.0

    # Minimum SequenceMatcher ratio for a "soft" name match
    correctness_name_soft_threshold: float = 0.5

    # Absolute bounds before normalisation (used by ToolRLReward)
    correctness_max_reward: float = 3.0
    correctness_min_reward: float = -3.0

    # Weight for ordering penalty (0.0 = disabled, 1.0 = full)
    correctness_order_penalty_weight: float = 0.5

    # ------------------------------------------------------------------
    # ToolRL normalisation boundaries
    # ------------------------------------------------------------------
    toolrl_min_reward: float = -3.0
    toolrl_max_reward: float = 4.0


DEFAULT_REWARD_CONFIG = RewardConfig()
