"""
Pluggable Reward System
=======================

Auto-discovers all ``Reward`` subclasses so adding a new reward is as simple
as dropping a file into this package and subclassing ``Reward``.

Usage::

    from tool_calling_data_generation.rewards import RewardRegistry

    registry = RewardRegistry()
    registry.compute_all("chosen", completion_text, ground_truth_dict)

    # Or instantiate a specific reward:
    toolrl = registry.get("toolrl")
    result = toolrl.compute(completion, ground_truth)
"""

from typing import Dict, List, Optional, Type

from tool_calling_data_generation.rewards.base import Reward, RewardResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RewardRegistry:
    """Auto-discovers ``Reward`` subclasses and provides a unified interface.

    Discovery walks ``Reward.__subclasses__()`` recursively so rewards defined
    in *any* imported module are picked up automatically.  No manual
    registration required.
    """

    def __init__(self):
        self._reward_types: Dict[str, Type[Reward]] = {}
        self._instances: Dict[str, Reward] = {}
        self._discover()

    # -- discovery --------------------------------------------------------

    def _discover(self) -> None:
        """Walk ``Reward.__subclasses__()`` recursively."""

        def walk(cls: Type[Reward]):
            for sub in cls.__subclasses__():
                if sub is Reward:
                    continue
                # Honour the per-class disabled flag
                if getattr(sub, "disabled", False):
                    walk(sub)  # still walk children (a subclass may re-enable)
                    continue
                name = getattr(sub, "name", None) or sub.__name__
                self._reward_types[name] = sub
                walk(sub)

        walk(Reward)

    # -- lookup -----------------------------------------------------------

    def get(self, name: str, **kwargs) -> Reward:
        """Return an *instance* of the reward identified by *name*.

        Instances are cached so repeated calls with the same *name* and
        *kwargs* reuse the same object.  Pass different kwargs to force a
        new instance.
        """
        cache_key = name + str(sorted(kwargs.items()))
        if cache_key in self._instances:
            return self._instances[cache_key]

        cls = self._reward_types.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown reward '{name}'. Available: {list(self._reward_types)}"
            )
        instance = cls(**kwargs)
        self._instances[cache_key] = instance
        return instance

    def list_all(self) -> List[str]:
        """Return the names of all discovered rewards."""
        return sorted(self._reward_types)

    # -- convenience: run all rewards at once -----------------------------

    def compute_all(
        self, completion: str, ground_truth: Dict, **init_kwargs
    ) -> Dict[str, RewardResult]:
        """Run **every** registered reward on a single completion.

        Returns ``{reward_name: RewardResult}``.
        """
        results: Dict[str, RewardResult] = {}
        for name in self._reward_types:
            try:
                r = self.get(name, **init_kwargs)
                results[name] = r.compute(completion, ground_truth)
            except Exception as exc:
                results[name] = RewardResult(
                    score=0.0,
                    components={"error": str(exc)},
                )
        return results


# ---------------------------------------------------------------------------
# Convenience imports so user code can do ``from tool_calling_data_generation.rewards import ToolRLReward``
# ---------------------------------------------------------------------------

from tool_calling_data_generation.rewards.binary import BinaryReward  # noqa: E402, F401
from tool_calling_data_generation.rewards.curriculum import CurriculumReward  # noqa: E402, F401
from tool_calling_data_generation.rewards.toolrl import ToolRLReward  # noqa: E402, F401
