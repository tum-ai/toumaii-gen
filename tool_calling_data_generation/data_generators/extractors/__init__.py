# Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
# Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tool-call extractors for parsing model outputs into structured tool calls.

Provides:
* ``BaseExtractor`` — abstract base class.
* ``ExtractionResult`` — standardised dataclass for parsed output.
* ``ToolCallExtractor`` — concrete implementation using vLLM parsers
  with regex + JSON-salvage fallback.
"""

from tool_calling_data_generation.data_generators.extractors.base import (
    BaseExtractor,
    ExtractionResult,
)
from tool_calling_data_generation.data_generators.extractors.tool_call_extractor import (
    ToolCallExtractor,
    clean_tool_call_content,
    TOOL_CALL_TAG_MAP,
)

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ToolCallExtractor",
    "clean_tool_call_content",
    "TOOL_CALL_TAG_MAP",
]
