# Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
# Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Canonical dataset models."""

from tool_calling_data_generation.data_generators.models.canonical_format import (
    CanonicalDataset,
    CanonicalSample,
)

CanonicalFormat = CanonicalSample  # compatibility name used by the old converter API

__all__ = ["CanonicalDataset", "CanonicalFormat", "CanonicalSample"]
