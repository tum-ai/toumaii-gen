"""
Generation backends for the DPO data pipeline.

- ``BaseGenerator`` — abstract interface all backends implement.
- ``vLLMGenerator`` — single-GPU vLLM-based generation (CUDA only).
- ``vLLMParallelGenerator`` — data-parallel vLLM: one engine per GPU,
  prompt batch sharded across all GPUs (CUDA only).
- ``TRLGenerator`` — pure HuggingFace Transformers (CUDA / MPS / CPU).
"""

from tool_calling_data_generation.data_generators.backends.base import BaseGenerator
from tool_calling_data_generation.data_generators.backends.trl import TRLGenerator


def __getattr__(name):
    """Load CUDA-only backends only when explicitly requested."""
    if name == "vLLMGenerator":
        from tool_calling_data_generation.data_generators.backends.vllm import vLLMGenerator

        return vLLMGenerator
    if name == "vLLMParallelGenerator":
        from tool_calling_data_generation.data_generators.backends.vllm_parallel import (
            vLLMParallelGenerator,
        )

        return vLLMParallelGenerator
    raise AttributeError(name)

__all__ = [
    "BaseGenerator",
    "TRLGenerator",
    "vLLMGenerator",
    "vLLMParallelGenerator",
]

