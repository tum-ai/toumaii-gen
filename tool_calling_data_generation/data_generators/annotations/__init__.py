"""
Annotations sub-package for the DPO data generator.

Provides:
* ``Annotator`` — runs all registered rewards on a DataFrame of completions.
* ``PreferenceLabeler`` — selects chosen/rejected pairs via reward scoring.
"""

from tool_calling_data_generation.data_generators.annotations.annotator import Annotator
from tool_calling_data_generation.data_generators.annotations.labeler import PreferenceLabeler

__all__ = ["Annotator", "PreferenceLabeler"]


