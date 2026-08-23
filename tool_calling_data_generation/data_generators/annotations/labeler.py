"""
Preference Labeler.

Given N completions for a prompt, scores all of them with a reward function
and selects the (chosen, rejected) pair with the highest reward margin,
provided both quality and margin thresholds are met.
"""

from typing import Any, Dict, List, Optional

from tool_calling_data_generation.rewards.base import Reward


class PreferenceLabeler:
    """Selects chosen/rejected pairs for DPO / IPO training.

    Typical usage::

        labeler = PreferenceLabeler(reward_fn=my_reward, min_chosen_reward=0.8)
        pair = labeler.label(prompt, completions, ground_truth)
        # pair is None if no valid pair could be formed
    """

    def __init__(
        self,
        reward_fn: Reward,
        min_chosen_reward: float = 0.8,
        margin_threshold: float = 0.3,
    ):
        """
        Args:
            reward_fn: An instantiated ``Reward`` subclass.
            min_chosen_reward: Minimum score for a completion to be "chosen".
            margin_threshold: Minimum score gap between chosen and rejected.
        """
        self.reward_fn = reward_fn
        self.min_chosen_reward = min_chosen_reward
        self.margin_threshold = margin_threshold

    def label(
        self,
        prompt: str,
        completions: List[str],
        ground_truth: Dict[str, Any],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Score all completions and pick the best pair.

        Returns:
            A dict with ``prompt``, ``chosen``, ``rejected``, and ``metadata``
            if a valid pair is found, otherwise ``None``.
        """
        results = self.reward_fn.batch(completions, [ground_truth] * len(completions))
        pairs = list(zip([r.score for r in results], completions))
        pairs.sort(key=lambda x: x[0], reverse=True)

        best_score, best_comp = pairs[0]
        worst_score, worst_comp = pairs[-1]

        if (
            best_score >= self.min_chosen_reward
            and (best_score - worst_score) >= self.margin_threshold
        ):
            return {
                "prompt": prompt,
                "chosen": best_comp,
                "rejected": worst_comp,
                "metadata": {
                    "reward_chosen": round(best_score, 4),
                    "reward_rejected": round(worst_score, 4),
                    "margin": round(best_score - worst_score, 4),
                },
            }
        return None

    def label_many(
        self,
        prompts: List[str],
        completions_list: List[List[str]],
        ground_truths: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Optional[Dict[str, Any]]]:
        """Batch version of ``label()``."""
        return [
            self.label(p, comps, gt, **kwargs)
            for p, comps, gt in zip(prompts, completions_list, ground_truths)
        ]


