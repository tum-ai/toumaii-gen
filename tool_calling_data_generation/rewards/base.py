"""
Base classes for the pluggable reward system.

All reward implementations inherit from `Reward` and return a standardized
`RewardResult`.  New rewards are auto-discovered by `RewardRegistry` — no
manual registration needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tool_calling_data_generation.rewards.config import RewardConfig, DEFAULT_REWARD_CONFIG


# ---------------------------------------------------------------------------
# RewardResult
# ---------------------------------------------------------------------------


@dataclass
class RewardResult:
    """Standardised return type for every reward function.

    Attributes:
        score: The aggregated reward score (always in [0.0, 1.0]).
        components: Optional dict of named sub-scores for transparency
            (e.g. ``{"r_format": 0.8, "r_name_soft": 1.0, "r_args": 0.9}``).
            Useful for curriculum learning and debugging.
    """

    score: float
    components: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward (abstract base)
# ---------------------------------------------------------------------------


class Reward(ABC):
    """Abstract base for all reward functions.

    Subclasses MUST implement ``compute(completion, ground_truth)`` which
    scores a **single** completion.  The ``batch`` helper is provided for
    convenience — override it if you have a vectorised implementation.

    Class attributes:
        name: Human-readable key used by ``RewardRegistry``.
            Falls back to ``cls.__name__`` if not set.
        disabled: When ``True`` the registry skips this reward entirely.
            Set to ``True`` for rewards that are experimental, deprecated,
            or not yet ready for production.

    Usage::

        result: RewardResult = reward.compute("<tool_call>...</tool_call>",
                                              {"expected_calls": [...]})
    """

    # Optional human-readable name used by the registry.
    # If not set the registry falls back to ``cls.__name__``.
    name: Optional[str] = None

    # When True the RewardRegistry skips this reward during discovery.
    # Set in subclasses, e.g.:  disabled = True
    disabled: bool = False

    def __init__(self, config: Optional[RewardConfig] = None, **kwargs):
        self.config = config or DEFAULT_REWARD_CONFIG

    # -- single-completion interface (must override) -----------------------

    @abstractmethod
    def compute(self, completion: str, ground_truth: Dict) -> RewardResult:
        """Score a single completion against its ground truth.

        Args:
            completion: Raw model output string (may contain ``<tool_call>``
                tags, JSON, etc.).
            ground_truth: Dict with at least an ``"expected_calls"`` key
                containing ``List[Dict]`` of ground-truth tool calls.

        Returns:
            ``RewardResult`` with at minimum a ``score`` field.
        """
        ...

    # -- batch helpers -----------------------------------------------------

    def batch(
        self, completions: List[str], ground_truths: List[Dict]
    ) -> List[RewardResult]:
        """Score a batch.  Override for vectorised implementations."""
        return [self.compute(c, gt) for c, gt in zip(completions, ground_truths)]

    # -- legacy ------------------------------------------------------------
    # Keep __call__ for backward compat with code that does reward_fn(...)

    def __call__(
        self, completions: List[str], ground_truth: List[Dict], **kwargs
    ) -> List[float]:
        """Legacy interface — returns raw float scores only."""
        return [r.score for r in self.batch(completions, ground_truth)]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def norm_r(r: float, min_r: float, max_r: float) -> float:
    """Normalise *r* to [0, 1] given min/max bounds."""
    if max_r == min_r:
        return 0.0
    return max(0.0, min(1.0, (r - min_r) / (max_r - min_r)))
