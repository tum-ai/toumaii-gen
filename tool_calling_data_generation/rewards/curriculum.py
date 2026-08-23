"""
Curriculum Reward.

A weighted multi-component reward designed for curriculum learning:
    score = (w_f * R_format + w_ns * R_name_soft + w_nt * R_name_strict + w_a * R_args)
            ─────────────────────────────────────────────────────────────────────────
                              w_f + w_ns + w_nt + w_a

The weights can be updated mid-training to shift focus from format → names → arguments.
"""

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tool_calling_data_generation.rewards.base import Reward, RewardResult
from tool_calling_data_generation.rewards.config import RewardConfig, DEFAULT_REWARD_CONFIG
from tool_calling_data_generation.rewards.utils import calculate_format_reward, parse_tool_calls, strict_match


# ---------------------------------------------------------------------------
# Granular correctness (split name / args)
# ---------------------------------------------------------------------------


def calculate_granular_correctness(
    completion: str,
    expected_calls: List[Dict[str, Any]],
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> Tuple[float, float, float]:
    """Return ``(r_name_soft, r_name_strict, r_args_score)``.

    * *r_name_soft* — fuzzy name matching via SequenceMatcher.
    * *r_name_strict* — exact case-sensitive name match.
    * *r_args_score* — raw argument match score (unnormalised).
    """
    pred_calls = parse_tool_calls(completion)
    n_gt = len(expected_calls)
    n_pred = len(pred_calls)

    if n_gt == 0:
        return (1.0, 1.0, 0.0) if n_pred == 0 else (0.0, 0.0, 0.0)

    # ---- names -----------------------------------------------------------
    pred_names_raw = [str(c.get("name", "")) for c in pred_calls]
    gt_names_raw = [str(c.get("name", "")) for c in expected_calls]

    pred_lower = [n.strip().lower() for n in pred_names_raw]
    gt_lower = [n.strip().lower() for n in gt_names_raw]

    name_cost = np.zeros((n_gt, n_pred))
    for i, gn in enumerate(gt_lower):
        for j, pn in enumerate(pred_lower):
            sim = SequenceMatcher(None, gn, pn).ratio()
            name_cost[i, j] = (
                sim if sim > config.correctness_name_soft_threshold else 0.0
            )

    row_n, col_n = linear_sum_assignment(-name_cost)
    name_soft_sum = name_cost[row_n, col_n].sum()

    name_strict_sum = sum(
        1.0 for i, j in zip(row_n, col_n) if gt_names_raw[i] == pred_names_raw[j]
    )

    denom = max(n_gt, n_pred)
    r_name_soft = name_soft_sum / denom if denom > 0 else 0.0
    r_name_strict = name_strict_sum / denom if denom > 0 else 0.0

    # penalise dead matches
    dead = sum(1 for i, j in zip(row_n, col_n) if name_cost[i, j] == 0)
    if dead > 0 and denom > 0:
        penalty = dead * config.correctness_unmatched_penalty / denom
        r_name_soft = max(0.0, r_name_soft - penalty)
        r_name_strict = max(0.0, r_name_strict - penalty)

    # ---- arguments (Hungarian) -------------------------------------------
    cost = np.zeros((n_gt, n_pred))
    for i, gt_call in enumerate(expected_calls):
        gt_params = _safe_dict(gt_call.get("arguments"))
        gt_keys = set(gt_params.keys())
        for j, pred_call in enumerate(pred_calls):
            pred_params = _safe_dict(
                pred_call.get("parameters", pred_call.get("arguments"))
            )
            pred_keys = set(pred_params.keys())

            ku = gt_keys | pred_keys
            s_param = len(gt_keys & pred_keys) / len(ku) if ku else 1.0

            s_value = sum(
                1.0
                for k in gt_keys
                if k in pred_params and strict_match(pred_params[k], gt_params[k])
            )
            cost[i, j] = s_param + s_value

    row_a, col_a = linear_sum_assignment(-cost)
    match_score = cost[row_a, col_a].sum()

    # penalties
    n_unmatched = n_pred - len(col_a)
    dead_args = sum(1 for i, j in zip(row_a, col_a) if cost[i, j] == 0)
    penalty = (n_unmatched + dead_args) * config.correctness_unmatched_penalty

    # ---- order penalty (uses arg alignment) ------------------------------
    if n_gt >= 2 and n_pred >= 2 and config.correctness_order_penalty_weight > 0:
        matched = sorted(zip(row_a, col_a), key=lambda x: x[0])
        pred_order = [p for _, p in matched]
        inversions = 0
        max_inv = 0
        for a in range(len(pred_order)):
            for b in range(a + 1, len(pred_order)):
                max_inv += 1
                if pred_order[a] > pred_order[b]:
                    inversions += 1
        if max_inv > 0:
            order_pen = config.correctness_order_penalty_weight * (inversions / max_inv)
            r_name_soft *= 1.0 - order_pen
            r_name_strict *= 1.0 - order_pen

    r_args = max(0.0, match_score - penalty)
    return r_name_soft, r_name_strict, r_args


def _safe_dict(val: Any) -> Dict[str, Any]:
    """Coerce *val* to a dict, handling JSON strings and None."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ---------------------------------------------------------------------------
# CurriculumReward
# ---------------------------------------------------------------------------


class CurriculumReward(Reward):
    """Weighted multi-component reward for curriculum DPO.

    Weights can be updated at any time via ``update_weights()``, allowing a
    single instance to serve across curriculum stages.
    """

    name = "curriculum"
    disabled = False

    def __init__(
        self,
        weight_format: float = 1.0,
        weight_name_soft: float = 1.0,
        weight_name_strict: float = 0.0,
        weight_args: float = 1.0,
        config: RewardConfig = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.w_format = weight_format
        self.w_name_soft = weight_name_soft
        self.w_name_strict = weight_name_strict
        self.w_args = weight_args

    # -- weight management ------------------------------------------------

    def update_weights(
        self,
        w_format: float,
        w_name_soft: float,
        w_name_strict: float,
        w_args: float,
    ) -> None:
        self.w_format = w_format
        self.w_name_soft = w_name_soft
        self.w_name_strict = w_name_strict
        self.w_args = w_args

    # -- compute ----------------------------------------------------------

    def compute(self, completion: str, ground_truth: Dict) -> RewardResult:
        expected = ground_truth.get("expected_calls", [])

        r_format = calculate_format_reward(completion, self.config)
        r_name_soft, r_name_strict, r_args_raw = calculate_granular_correctness(
            completion, expected, self.config
        )

        # normalise args
        total_possible = sum(
            (1.0 + len(_safe_dict(c.get("arguments")))) for c in expected
        )
        if total_possible > 0:
            norm_r_args = r_args_raw / total_possible
        elif len(expected) == 0 and len(parse_tool_calls(completion)) == 0:
            norm_r_args = 1.0
        else:
            norm_r_args = 0.0

        weight_total = (
            self.w_format + self.w_name_soft + self.w_name_strict + self.w_args
        )
        if weight_total == 0:
            return RewardResult(score=0.0)

        score = (
            self.w_format * r_format
            + self.w_name_soft * r_name_soft
            + self.w_name_strict * r_name_strict
            + self.w_args * norm_r_args
        ) / weight_total

        return RewardResult(
            score=score,
            components={
                "r_format": r_format,
                "r_name_soft": r_name_soft,
                "r_name_strict": r_name_strict,
                "r_args": norm_r_args,
            },
        )
