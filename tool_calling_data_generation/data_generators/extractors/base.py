"""
Base classes for tool-call extraction.

Mirrors the ``BaseGenerator`` pattern — each model family can have its own
extractor, or use the generic ``ToolCallExtractor`` which delegates to vLLM's
``ToolParserManager`` plus a best-effort regex fallback.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Standardised return from any extractor.

    Attributes:
        conversational_preamble: Natural-language text that precedes or
            surrounds tool calls (the model's "thinking" output).
        detected_tool_calls: Parsed tool calls as ``List[Dict]``, each with
            ``name`` and ``arguments`` keys.
        raw_model_output: The original, unmodified model output string.
        parser_used: Name of the parser that produced this result
            (e.g. ``"vLLM_qwen3_coder"``, ``"best_effort_decoder"``).
        is_best_effort: ``True`` when the primary parser failed and the
            regex / JSON-salvage fallback was used.
    """

    conversational_preamble: str = ""
    detected_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_model_output: str = ""
    parser_used: str = "none"
    is_best_effort: bool = False


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------


class BaseExtractor(ABC):
    """Abstract base for tool-call extractors.

    Subclasses must implement ``extract(model_output, model_id)`` which
    parses a single raw model output string into an ``ExtractionResult``.

    The ``extract_batch`` default uses a ``ThreadPool`` for concurrency.
    Override if you have a vectorised implementation.
    """

    @abstractmethod
    def extract(self, model_output: str, model_id: str) -> ExtractionResult:
        """Parse a single raw model output.

        Args:
            model_output: The raw string output from the model.
            model_id: HuggingFace model ID (used to select the correct parser
                family, e.g. ``"Qwen/Qwen2.5-0.5B"``).

        Returns:
            ``ExtractionResult`` with parsed tool calls and metadata.
        """
        ...

    def extract_batch(
        self,
        model_outputs: List[str],
        model_id: str,
        uuids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
    ) -> Dict[str, ExtractionResult]:
        """Parse a batch of outputs, keyed by uuid (sequential, no threading).

        Args:
            model_outputs: List of raw output strings.
            model_id: HuggingFace model ID.
            uuids: Optional list of unique IDs.  If not provided, ``str(i)``
                is used as the key.
            num_workers: Ignored — kept for backward compatibility.

        Returns:
            ``{uuid: ExtractionResult}`` dict.
        """
        from tqdm import tqdm

        if uuids is None:
            uuids = [str(i) for i in range(len(model_outputs))]

        results: Dict[str, ExtractionResult] = {}
        for uid, output in tqdm(
            zip(uuids, model_outputs),
            total=len(model_outputs),
            desc="Extracting tool calls",
        ):
            results[uid] = self.extract(output, model_id)

        return results
