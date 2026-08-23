"""
Data-loader abstraction for DPO dataset generation.

Every dataset source (Toucan, ToolACE, …) implements ``load() ->
CanonicalDataset`` so the ``dpo_generator`` pipeline is source-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional

from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset


class BaseDataLoader(ABC):
    """Abstract base for dataset loaders.

    Subclasses must implement ``load()`` which returns a
    ``CanonicalDataset`` containing canonical-format samples
    (``conv_id``, ``turn_id``, ``messages``, ``tool_catalog``,
    ``chosen_response``, etc.).
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @abstractmethod
    def load(self) -> CanonicalDataset:
        """Load the dataset and return it in canonical format.

        The returned ``CanonicalDataset`` must contain the columns
        defined in ``CanonicalSample.to_dict()``.  At minimum the
        loader populates: ``conv_id``, ``turn_id``, ``messages``,
        ``tool_catalog``, and ``chosen_response``.  The remaining
        fields (``rejected_response``, ``metadata``) are filled by
        downstream pipeline stages.
        """
        ...
