"""
Binary Reward (from ToolN1 paper).

Returns 1.0 only when BOTH format and correctness are perfect, else 0.0.
"""

from typing import Dict

from tool_calling_data_generation.rewards.base import Reward, RewardResult
from tool_calling_data_generation.rewards.toolrl import calculate_correctness_reward
from tool_calling_data_generation.rewards.utils import calculate_format_reward


class BinaryReward(Reward):
    """Binary reward: 1.0 if perfect, 0.0 otherwise.

    Perfect means:
        * ``R_format == format_json_valid_bonus`` (i.e. valid JSON in tags)
        * ``R_correctness == correctness_max_reward`` (perfect Hungarian match)
    """

    name = "binary"
    disabled = True  # experimental — not enabled in default annotation runs

    def compute(self, completion: str, ground_truth: Dict) -> RewardResult:
        expected = ground_truth.get("expected_calls", [])
        r_format = calculate_format_reward(completion, self.config)
        r_correct = calculate_correctness_reward(completion, expected, self.config)

        perfect = (
            r_format == self.config.format_json_valid_bonus
            and r_correct == self.config.correctness_max_reward
        )
        score = 1.0 if perfect else 0.0
        return RewardResult(
            score=score, components={"r_format": r_format, "r_correct": r_correct}
        )
