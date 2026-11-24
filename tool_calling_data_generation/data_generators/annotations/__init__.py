# Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
# Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Annotations sub-package for the DPO data generator.

Provides:
* ``Annotator`` — runs all registered rewards on a DataFrame of completions.
* ``PreferenceLabeler`` — selects chosen/rejected pairs via reward scoring.
"""

from tool_calling_data_generation.data_generators.annotations.annotator import Annotator
from tool_calling_data_generation.data_generators.annotations.labeler import PreferenceLabeler

__all__ = ["Annotator", "PreferenceLabeler"]


