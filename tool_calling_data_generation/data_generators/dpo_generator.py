"""
DPO Data Generator
==================

Orchestrates the full DPO data-generation pipeline in a single pass:

    1. Load dataset (loader → CanonicalDataset)
    2. Prepare for generation (render prompts, filter overlength)
    3. Generate completions (vLLM / TRL) → raw model outputs (rejected)
    4. Extract tool calls from rejected outputs
    5. Fill rejected_response + metadata into CanonicalDataset
    6. Annotate with all rewards → scores in metadata

The resulting ``CanonicalDataset`` is ready for training export
(``.to_sft_dataset()`` / ``.to_dpo_dataset()``) or HuggingFace upload.

Usage::

    from tool_calling_data_generation.data_generators.loaders import ToucanLoader
    from tool_calling_data_generation.data_generators.dpo_generator import build_dpo_dataset

    loader = ToucanLoader(size=500, subset_name_required="multi-turn")
    cd = build_dpo_dataset(loader, model_id="Qwen/Qwen2.5-0.5B")
    sft_ds = cd.to_sft_dataset()   # ready for SFT training
    dpo_ds = cd.to_dpo_dataset()   # ready for DPO training
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

from tool_calling_data_generation.data_generators.annotations import Annotator
from tool_calling_data_generation.data_generators.backends.base import BaseGenerator
from tool_calling_data_generation.data_generators.backends.trl import TRLGenerator
from tool_calling_data_generation.data_generators.extractors import ToolCallExtractor
from tool_calling_data_generation.data_generators.helpers import prepare_canonical_for_generation
from tool_calling_data_generation.data_generators.loaders.base import BaseDataLoader
from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_backend(model_id: str) -> BaseGenerator:
    """Return a vLLM backend if vLLM is importable, else ``TRLGenerator``.

    Set ``VLLM_NUM_GPUS`` (e.g. ``VLLM_NUM_GPUS=8``) to opt into the
    data-parallel ``vLLMParallelGenerator`` instead of the single-GPU
    ``vLLMGenerator``.
    """
    try:
        import vllm  # noqa: F401
    except ImportError:
        logger.info("vLLM not available — using TRLGenerator backend.")
        return TRLGenerator(model_id)

    num_gpus = os.environ.get("VLLM_NUM_GPUS")
    if num_gpus:
        from tool_calling_data_generation.data_generators.backends import vLLMParallelGenerator

        logger.info(
            "vLLM detected — using vLLMParallelGenerator backend (VLLM_NUM_GPUS=%s).",
            num_gpus,
        )
        return vLLMParallelGenerator(model_id, num_gpus=int(num_gpus))

    logger.info("vLLM detected — using vLLMGenerator backend.")
    from tool_calling_data_generation.data_generators.backends.vllm import vLLMGenerator

    return vLLMGenerator(model_id)


def _extract_tc_from_response(
    response: Any,
) -> List[Dict[str, Any]]:
    """Extract ``[{name, arguments}]`` from a response.

    Accepts a raw list of dicts, a JSON-string-encoded list, or an
    already-parsed list (Arrow → Python deserialisation).
    """
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return []
    result: List[Dict[str, Any]] = []
    for msg in response or []:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            result.append(
                {"name": func.get("name", ""), "arguments": func.get("arguments", {})}
            )
    return result


def _calls_equivalent(
    chosen_calls: List[Dict[str, Any]], rejected_calls: List[Any]
) -> bool:
    """Compare two parsed tool-call lists for structural equality.

    Only dict entries are compared; non-dict (unparseable raw strings)
    are treated as not matching.
    """
    cc = [c for c in chosen_calls if isinstance(c, dict)]
    rc = [r for r in rejected_calls if isinstance(r, dict)]
    if len(cc) != len(rc):
        return False
    for a, b in zip(cc, rc):
        if a.get("name") != b.get("name"):
            return False
        if a.get("arguments") != b.get("arguments"):
            return False
    return True


def _strip_nulls(obj: Any) -> Any:
    """Recursively remove keys with ``None`` values from a dict."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def _clean_preamble(preamble: str) -> str:
    """Strip tool-call tags and their content from a preamble string.

    Some vLLM parsers return the full raw model output as
    ``conversational_preamble`` (including ``<tool_call>`` tags).
    This removes everything between and including the common tag
    pairs, leaving only natural-language text.
    """
    import re

    tags = [
        (r"<tool_call>", r"</tool_call>"),
        (r"<\|tool_call>", r"<tool_call\|>"),
        (r"<function_call>", r"</function_call>"),
    ]
    for start, end in tags:
        preamble = re.sub(
            rf"{re.escape(start)}.*?{re.escape(end)}", "", preamble, flags=re.DOTALL
        )
    # Collapse whitespace left behind by tag removal
    preamble = re.sub(r"\s{2,}", " ", preamble)
    return preamble.strip()


def _tc_list_to_response(
    tool_calls: List[Any],
    preamble: str = "",
) -> List[Dict[str, Any]]:
    """Build a normalized response from a list of detected tool calls.

    Handles both parsed dicts (``{"name":..., "arguments":...}``) and
    raw strings (unparseable best-effort output).  Strings are appended
    to the preamble so the malformed data is preserved for DPO training.
    """
    tc_objs: List[Dict[str, Any]] = []
    extra_content: List[str] = []

    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_objs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": _strip_nulls(tc.get("arguments", {})),
                    },
                }
            )
        elif isinstance(tc, str) and tc.strip():
            # Unparseable — preserve as content for DPO
            extra_content.append(tc)

    content = preamble
    if extra_content:
        content = (preamble + "\n" if preamble else "") + "\n".join(extra_content)

    if tc_objs:
        return [{"role": "assistant", "content": content, "tool_calls": tc_objs}]
    else:
        return [{"role": "assistant", "content": content, "tool_calls": []}]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dpo_dataset(
    loader: BaseDataLoader,
    model_id: str = "Qwen/Qwen2.5-0.5B",
    backend: Optional[BaseGenerator] = None,
    use_chat_template: bool = True,
    use_disk_cache: bool = True,
    model_ctx: int = 16384,
    gpu_memory_utilization: float = 0.8,
    temperature: float = 0.0,
    max_tokens: int = 512,
    annotate: bool = True,
    dont_unload_backend: bool = False,
    **backend_kwargs,
) -> CanonicalDataset:
    """Run the full DPO generation pipeline and return a populated
    ``CanonicalDataset``.

    Every sample in the returned dataset has ``rejected_response``
    and all ``metadata`` fields fully populated (parsed tool calls,
    parser info, annotation scores).  Call ``.to_sft_dataset()`` or
    ``.to_dpo_dataset()`` to export for training.

    Args:
        loader: A ``BaseDataLoader`` that returns a ``CanonicalDataset``.
            Use ``ToucanLoader`` or ``ToolACELoader``.
        model_id: HuggingFace model ID for generation.
        backend: An optional ``BaseGenerator`` instance.  If ``None``,
            auto-detected: vLLM if installed, otherwise TRL/Transformers.
        use_chat_template: Use tokenizer's ``apply_chat_template``.
        use_disk_cache: Cache dataset preprocessing on disk.
        model_ctx: Maximum model context length.
        gpu_memory_utilization: vLLM GPU memory fraction (ignored by TRL).
        temperature: Sampling temperature (0 = greedy).
        max_tokens: Maximum new tokens per generation.
        annotate: If ``True``, run all registered rewards and store
            scores in ``metadata["annotation_scores"]``.
        dont_unload_backend: If ``True``, do not unload the backend after generation.
        **backend_kwargs: Extra kwargs forwarded to backend initialization and
            generation. Backends ignore values they do not support.

    Returns:
        A fully populated ``CanonicalDataset`` ready for training export.
    """
    usable_ctx = model_ctx - 1
    prompt_format = "chat_template" if use_chat_template else "custom"

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ------------------------------------------------------------------
    # 2. Backend (generator)
    # ------------------------------------------------------------------
    if backend is not None:
        logger.info("Using provided backend: %s", type(backend).__name__)
    if backend is None:
        logger.info("Auto-detecting backend for model: %s", model_id)
        backend = _auto_detect_backend(model_id)

    # ------------------------------------------------------------------
    # 3. Load (pluggable loader → CanonicalDataset) & prepare
    # ------------------------------------------------------------------
    logger.info("Loading dataset via %s …", type(loader).__name__)
    cd = loader.load()

    cd = prepare_canonical_for_generation(
        cd,
        tokenizer,
        usable_ctx,
        max_new_tokens=max_tokens,
        prompt_format=prompt_format,
        cache_disable=not use_disk_cache,
        model_id=model_id,
    )

    hf_ds = cd.to_hf_dataset()
    prompts: List[str] = hf_ds["prompt"]
    uuids: List[str] = hf_ds["conv_tracking_id"]
    raw_outputs_ref: List[str] = []  # mutable ref for the map closure below

    total_prompts = len(prompts)
    logger.info("Prepared %d prompts for generation.", total_prompts)

    # ------------------------------------------------------------------
    # 4. Generate completions (rejected)
    # ------------------------------------------------------------------
    if not backend.is_backend_loaded:
        backend.initialize_backend(
            gpu_memory_utilization=gpu_memory_utilization,
            model_ctx=model_ctx,
            **backend_kwargs,
        )
    try:
        logger.info("Generating completions …")
        raw_outputs: List[str] = backend.generate(
            prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            **backend_kwargs,
        )

        logger.info("Generation complete.  %d outputs received.", len(raw_outputs))

        # ------------------------------------------------------------------
        # 5. Extract tool calls from model outputs
        # ------------------------------------------------------------------
        logger.info("Extracting tool calls …")
        extractor = ToolCallExtractor(tokenizer=backend.tokenizer)

        resp_map = extractor.extract_batch(raw_outputs, model_id, uuids=uuids)

        # ------------------------------------------------------------------
        # 6. Fill rejected_response + metadata into CanonicalDataset
        # ------------------------------------------------------------------
        logger.info("Filling rejected responses and metadata …")

        def _fill_rejected_and_meta(batch: dict) -> dict:
            n = len(batch["conv_tracking_id"])
            rejected_responses: list = []
            metadata_list = list(batch.get("metadata", ["{}" for _ in range(n)]))

            for i in range(n):
                uid = batch["conv_tracking_id"][i]
                rr = resp_map.get(uid)
                raw_meta = metadata_list[i]
                m = (
                    json.loads(raw_meta)
                    if isinstance(raw_meta, str)
                    else (dict(raw_meta) if raw_meta else {})
                )

                # Model identity
                m["model_id"] = model_id

                # Chosen tool calls — directly from normalized format
                chosen_calls = _extract_tc_from_response(
                    batch["chosen_response"][i]
                    if i < len(batch["chosen_response"])
                    else []
                )
                m["parsed_chosen_tool_calls"] = chosen_calls
                m["parser_used_chosen"] = "from_canonical"

                if rr is not None:
                    resp = _tc_list_to_response(
                        rr.detected_tool_calls,
                        preamble=_clean_preamble(rr.conversational_preamble),
                    )
                    rejected_responses.append(json.dumps(resp))
                    m["parsed_rejected_tool_calls"] = [
                        {
                            "name": tc["name"],
                            "arguments": _strip_nulls(tc.get("arguments", {})),
                        }
                        for tc in rr.detected_tool_calls
                        if isinstance(tc, dict)
                    ]
                    m["parser_used_rejected"] = rr.parser_used
                    m["is_best_effort_parser_used"] = rr.is_best_effort
                    m["chosen_rejected_equivalent"] = _calls_equivalent(
                        chosen_calls, rr.detected_tool_calls
                    )
                else:
                    # Extraction completely failed
                    fallback = raw_outputs[i] if i < len(raw_outputs) else ""
                    rejected_responses.append(
                        json.dumps(
                            [
                                {
                                    "role": "assistant",
                                    "content": fallback,
                                    "tool_calls": [],
                                }
                            ]
                        )
                    )
                    m["parsed_rejected_tool_calls"] = []
                    m["parser_used_rejected"] = "none"
                    m["is_best_effort_parser_used"] = True
                    m["chosen_rejected_equivalent"] = False

                metadata_list[i] = m

            return {
                "rejected_response": rejected_responses,
                "metadata": [json.dumps(m) for m in metadata_list],
            }

        cd = cd.map(
            _fill_rejected_and_meta,
            batched=True,
            load_from_cache_file=False,
            desc="Filling rejected_response + metadata",
        )

        # ------------------------------------------------------------------
        # 7. Annotate with all rewards
        # ------------------------------------------------------------------
        if annotate:
            logger.info("Annotating with all registered rewards …")
            annotator = Annotator()

            def _run_annotations(batch: dict) -> dict:
                n = len(batch["conv_tracking_id"])
                metadata_list = list(batch.get("metadata", ["{}" for _ in range(n)]))

                for i in range(n):
                    raw_meta = metadata_list[i]
                    m = (
                        json.loads(raw_meta)
                        if isinstance(raw_meta, str)
                        else (dict(raw_meta) if raw_meta else {})
                    )

                    # Serialize for reward computation
                    chosen_calls = _extract_tc_from_response(
                        batch["chosen_response"][i]
                        if i < len(batch["chosen_response"])
                        else []
                    )
                    rej_calls = _extract_tc_from_response(
                        batch["rejected_response"][i]
                        if i < len(batch["rejected_response"])
                        else []
                    )
                    chosen_str = json.dumps(chosen_calls)
                    rejected_str = json.dumps(rej_calls)

                    # Parse tool_catalog from the batch
                    tc_raw = batch.get("tool_catalog", [None] * n)[i]
                    tc = _parse_tool_catalog(tc_raw)

                    row_scores = annotator.annotate_row(
                        chosen_str,
                        rejected_str,
                        chosen_calls,
                        tool_catalog=tc,
                    )
                    m["annotation_scores"] = row_scores
                    metadata_list[i] = m

                return {"metadata": [json.dumps(m) for m in metadata_list]}

            cd = cd.map(
                _run_annotations,
                batched=True,
                load_from_cache_file=False,
                desc="Running reward annotations",
            )
            logger.info(
                "Annotation complete.  %d rewards applied.",
                len(annotator.registry.list_all()),
            )

        logger.info("DPO dataset built: %d samples.", len(cd))

    finally:
        if not dont_unload_backend and backend is not None:
            logger.info("Unloading backend to free GPU/CPU resources …")
            backend.unload_backend()

    return cd


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def build_dpo_dataset_from_toucan(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    model_size: int = 100,
    subset_name: str = "multi-turn",
    toolcall_mode: str = "single",
    **kwargs,
) -> CanonicalDataset:
    """Convenience wrapper that creates a ``ToucanLoader`` and calls
    ``build_dpo_dataset``.

    Args:
        toolcall_mode: ``"single"`` for one tool call per turn, or
            ``"multi"`` to merge consecutive tool-call turns into a
            single turn with multiple tool calls (mirrors the old
            generator's ``--toolcall_mode`` flag).

    All extra ``**kwargs`` are forwarded to ``build_dpo_dataset``.
    """
    from tool_calling_data_generation.data_generators.loaders import ToucanLoader

    loader = ToucanLoader(
        size=model_size,
        subset_name_required=subset_name,
        toolcall_mode=toolcall_mode,
    )
    return build_dpo_dataset(loader, model_id=model_id, **kwargs)


def build_dpo_dataset_from_toolace(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    model_size: Optional[int] = 100,
    use_disk_cache: bool = True,
    **kwargs,
) -> CanonicalDataset:
    """Create a ToolACE loader and run the full DPO generation pipeline."""
    from tool_calling_data_generation.data_generators.loaders import ToolACELoader

    loader = ToolACELoader(
        size=model_size,
        use_disk_cache=use_disk_cache,
    )
    return build_dpo_dataset(loader, model_id=model_id, **kwargs)


def to_analysis_dataframe(cd: CanonicalDataset) -> pd.DataFrame:
    """Export a populated ``CanonicalDataset`` to a flat DataFrame for
    inspection / debugging.  This is a lossy convenience — use
    ``cd.to_sft_dataset()`` or ``cd.to_dpo_dataset()`` for training.

    Columns: ``conv_id``, ``conv_tracking_id``, ``prompt``,
    ``chosen``, ``rejected``, ``tool_calls_chosen``,
    ``tool_calls_rejected``, ``parser_used_rejected``,
    ``chosen_rejected_equivalent``, ``is_best_effort``,
    plus annotation score columns when available.
    """
    hf_ds = cd.to_hf_dataset()
    n = len(hf_ds)

    rows = []
    for i in range(n):
        row = hf_ds[i]
        meta = row.get("metadata", {})

        chosen_calls = _extract_tc_from_response(row.get("chosen_response", []))
        rej_calls = _extract_tc_from_response(row.get("rejected_response", []))
        parsed_rej = [
            tc
            for tc in meta.get("parsed_rejected_tool_calls", [])
            if isinstance(tc, dict)
        ]

        entry = {
            "conv_id": row.get("conv_id"),
            "conv_tracking_id": row.get("conv_tracking_id"),
            "prompt": row.get("prompt", ""),
            "chosen": json.dumps(chosen_calls),
            "rejected": json.dumps(rej_calls),
            "tool_catalog": _parse_tool_catalog(row.get("tool_catalog", None)),
            "tool_calls_chosen": [
                tc
                for tc in meta.get("parsed_chosen_tool_calls", [])
                if isinstance(tc, dict)
            ],
            "tool_calls_rejected": parsed_rej,
            "parser_used_rejected": meta.get("parser_used_rejected"),
            "chosen_rejected_equivalent": meta.get("chosen_rejected_equivalent", False),
            "is_best_effort": meta.get("is_best_effort_parser_used", False),
        }

        # Flatten annotation scores into top-level columns
        for k, v in meta.get("annotation_scores", {}).items():
            entry[k] = v

        rows.append(entry)

    return pd.DataFrame(rows)


def _parse_tool_catalog(val: Any) -> Optional[List[Dict[str, Any]]]:
    """Parse tool_catalog from JSON string or list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None

