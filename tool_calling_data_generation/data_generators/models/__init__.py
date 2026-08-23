"""Canonical dataset models."""

from tool_calling_data_generation.data_generators.models.canonical_format import (
    CanonicalDataset,
    CanonicalSample,
)

CanonicalFormat = CanonicalSample  # compatibility name used by the old converter API

__all__ = ["CanonicalDataset", "CanonicalFormat", "CanonicalSample"]
