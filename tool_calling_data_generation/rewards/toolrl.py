"""
ToolRL Reward (baseline).

Implements the reward function from the ToolRL paper:
    R = R_format + R_correctness

Both components are normalised to [0, 1] using the bounds defined in
``RewardConfig``.
"""

import json
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from tool_calling_data_generation.rewards.base import Reward, RewardResult, norm_r
from tool_calling_data_generation.rewards.config import RewardConfig, DEFAULT_REWARD_CONFIG
from tool_calling_data_generation.rewards.utils import (
    calculate_format_reward,
    parse_tool_calls,
)


# ---------------------------------------------------------------------------
# ToolRLReward
# ---------------------------------------------------------------------------


def calculate_correctness_reward(
    completion: str,
    expected_calls: List[Dict[str, Any]],
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """R_correct: evaluates accuracy of tool invocations.

    Uses Hungarian matching on tool names (Jaccard) and argument key-value
    pairs.  Returns a float in ``[correctness_min_reward, correctness_max_reward]``.
    """
    pred_calls = parse_tool_calls(completion)
    n_gt = len(expected_calls)
    n_pred = len(pred_calls)

    if n_gt == 0:
        return (
            config.correctness_max_reward
            if n_pred == 0
            else config.correctness_min_reward
        )

    # -- tool-name Jaccard --
    pred_names = {c.get("name") for c in pred_calls if c.get("name")}
    gt_names = {c.get("name") for c in expected_calls if c.get("name")}
    union_names = pred_names | gt_names
    r_name = len(pred_names & gt_names) / len(union_names) if union_names else 1.0

    # -- key-value cost matrix --
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt_call in enumerate(expected_calls):
        gt_params = _safe_params(gt_call)
        gt_keys = set(gt_params.keys())
        for j, pred_call in enumerate(pred_calls):
            pred_params = _safe_params(pred_call, keys=("parameters", "arguments"))
            pred_keys = set(pred_params.keys())

            key_union = gt_keys | pred_keys
            s_param = len(gt_keys & pred_keys) / len(key_union) if key_union else 1.0
            s_value = sum(
                1.0
                for k in gt_keys
                if k in pred_params and pred_params[k] == gt_params[k]
            )
            cost_matrix[i, j] = s_param + s_value

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    total_pair_score = cost_matrix[row_ind, col_ind].sum()

    R_max = r_name + total_pair_score
    total_gt_params = sum(len(_safe_params(c)) for c in expected_calls)
    S_max = 1.0 + n_gt + total_gt_params

    range_r = config.correctness_max_reward - config.correctness_min_reward
    res = range_r * (R_max / S_max) + config.correctness_min_reward
    return max(
        config.correctness_min_reward,
        min(config.correctness_max_reward, res),
    )


def _safe_params(call: Dict, keys=("arguments",)) -> Dict[str, Any]:
    """Extract parameters dict from a tool call, handling str/None gracefully."""
    for k in keys:
        params = call.get(k)
        if params is None:
            continue
        if isinstance(params, dict):
            return params
        if isinstance(params, str):
            try:
                return json.loads(params)
            except (json.JSONDecodeError, ValueError):
                return {}
    return {}


# ---------------------------------------------------------------------------
# ToolRLReward
# ---------------------------------------------------------------------------


class ToolRLReward(Reward):
    """Reward function from the ToolRL paper.

    ``score = norm_r(R_format + R_correctness, toolrl_min, toolrl_max)``
    """

    name = "toolrl"
    disabled = False

    def __init__(self, config: RewardConfig = None, **kwargs):
        super().__init__(config=config, **kwargs)

    def compute(self, completion: str, ground_truth: Dict) -> RewardResult:
        expected = ground_truth.get("expected_calls", [])
        r_format = calculate_format_reward(completion, self.config)
        r_correct = calculate_correctness_reward(completion, expected, self.config)
        raw = r_format + r_correct
        score = norm_r(
            raw, self.config.toolrl_min_reward, self.config.toolrl_max_reward
        )
        return RewardResult(
            score=score, components={"r_format": r_format, "r_correct": r_correct}
        )
