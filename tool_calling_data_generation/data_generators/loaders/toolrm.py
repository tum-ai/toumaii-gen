"""
ToolACE data loader.

Loads the ToolACE dataset (``Team-ACE/ToolACE``) and converts it into the
canonical format expected by ``preprocess_toolcall_prompts``.

.. note::
    The ``build_sanitized_toolace`` helper lives in
    ``tool_calling_data_generation.toolace_sanitizer``. Once that
    module exists this loader is a thin adapter.
"""

from datasets import Dataset

from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader


class ToolRMLoader(BaseDataLoader):
    """Load ToolACE as a HuggingFace ``Dataset``.

    Parameters
    ----------
    size : int, optional
        Maximum number of conversations to include.
    """

    def load(self) -> Dataset:
        """Load and return a processed ToolACE ``Dataset``.

        Raises
        ------
        NotImplementedError
            Until ``tool_calling_data_generation.toolace_sanitizer`` is wired
            implemented and returns a ``Dataset`` with columns ``messages``,
            ``available_tools``, ``uuid``, and ``subset_name``.
        """
        # TODO: wire build_sanitized_toolace from the standalone package.
        #
        # Once available, replace this stub with:
        #
        #   from tool_calling_data_generation.toolace_sanitizer import build_sanitized_toolace
        #   return build_sanitized_toolace(size=self._kwargs.get("size"))
        #
        # The returned Dataset must have these columns:
        #   messages          – List[Dict]  (OpenAI-style conversation)
        #   available_tools   – List[Dict]  (tool definitions)
        #   uuid              – str         (unique conversation id)
        #   subset_name       – str         (e.g. "multi-turn")
        raise NotImplementedError(
            "build_sanitized_toolace is not yet implemented.  "
            "Wire tool_calling_data_generation.toolace_sanitizer into this loader."
        )
