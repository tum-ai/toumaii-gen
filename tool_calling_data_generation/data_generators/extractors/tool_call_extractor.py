"""
Generic tool-call extractor that delegates to vLLM's ``ToolParserManager``
first, then falls back to regex + JSON salvage when the primary parser fails.

Model-family detection is based on the HuggingFace model ID string.
"""

import contextlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple

from tool_calling_data_generation.data_generators.extractors.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag map — extend this when adding new model families
# ---------------------------------------------------------------------------

TOOL_CALL_TAG_MAP: Dict[str, Tuple[str, str]] = {
    "qwen": (r"<tool_call>", r"</tool_call>"),
    "granite": (r"<tool_call>", r"</tool_call>"),
    "gemma": (r"<\|tool_call>", r"<tool_call\|>"),
}

# vLLM ToolParserManager names keyed by model family
PARSER_MAP: Dict[str, str] = {
    "qwen": "qwen3_coder",
    "gemma": "gemma4",
    "granite": "granite4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_family(model_id: str) -> str:
    """Return the first matching family key from *model_id* (lowercased)."""
    return next((k for k in TOOL_CALL_TAG_MAP if k in model_id.lower()), "qwen")


def clean_tool_call_content(raw_content: str, model_id: str) -> str:
    """Strip ``<tool_call>`` / ``</tool_call>`` tags from *raw_content*."""
    family = _model_family(model_id)
    start, end = TOOL_CALL_TAG_MAP[family]
    return raw_content.strip().replace(start, "").replace(end, "").strip()


# ---------------------------------------------------------------------------
# Gemma4 custom-format helpers
# ---------------------------------------------------------------------------


def _gemma4_value(value_str: str):
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.lower() in ("null", "none", "nil"):
        return None
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    return value_str


def _gemma4_args(args_str: str) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format::

        key:<|"|">value<|"|">
        key:<|"|">value<|"|">,num:42,flag:true
        nested:{inner:<|"|">val<|"|">}
        items:[<|"|">a<|"|">,<|"|">b<|"|">]
    """
    if not args_str or not args_str.strip():
        return {}
    result: dict = {}
    i, n = 0, len(args_str)
    _SD = '<|"|>'
    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n or args_str[i] == "}":
            break
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1
        if i >= n:
            result[key] = ""
            break
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break
        if args_str[i:].startswith(_SD):
            i += len(_SD)
            val_start = i
            end_pos = args_str.find(_SD, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(_SD)
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(_SD):
                    i += len(_SD)
                    nd = args_str.find(_SD, i)
                    i = n if nd == -1 else nd + len(_SD)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            inner = args_str[obj_start : i - 1] if depth == 0 else args_str[obj_start:i]
            result[key] = _gemma4_args(inner)
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(_SD):
                    i += len(_SD)
                    nd = args_str.find(_SD, i)
                    i = n if nd == -1 else nd + len(_SD)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth == 0:
                result[key] = _gemma4_array(args_str[arr_start : i - 1])
            else:
                result[key] = []
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if i > val_start:
                result[key] = _gemma4_value(args_str[val_start:i])
    return result


def _gemma4_array(arr_str: str) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i, n = 0, len(arr_str)
    _SD = '<|"|>'
    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n or arr_str[i] == "]":
            break
        if arr_str[i:].startswith(_SD):
            i += len(_SD)
            end_pos = arr_str.find(_SD, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(_SD)
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(_SD):
                    i += len(_SD)
                    nd = arr_str.find(_SD, i)
                    i = n if nd == -1 else nd + len(_SD)
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            inner = arr_str[obj_start : i - 1] if depth == 0 else arr_str[obj_start:i]
            items.append(_gemma4_args(inner))
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(_SD):
                    i += len(_SD)
                    nd = arr_str.find(_SD, i)
                    i = n if nd == -1 else nd + len(_SD)
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth == 0:
                items.append(_gemma4_array(arr_str[sub_start : i - 1]))
            else:
                items.append([])
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if i > val_start:
                items.append(_gemma4_value(arr_str[val_start:i]))
    return items


def _try_gemma4_extraction(raw: str) -> dict | None:
    """Try to parse a Gemma4 tool-call string into ``{name, arguments}``.

    Accepts both the full ``<|tool_call|>call:...<tool_call|>`` form and
    the inner ``call:func_name{...}`` bare format.
    """
    import re as _re

    m = _re.search(r"call:([\w\-\.]+)\{(.*)\}", raw, _re.DOTALL)
    if not m:
        return None
    func_name = m.group(1).strip()
    args_str = m.group(2).strip()
    if not func_name:
        return None
    return {"name": func_name, "arguments": _gemma4_args(args_str)}


# ---------------------------------------------------------------------------
# Best-effort (regex / JSON salvage) fallback
# ---------------------------------------------------------------------------


def best_effort_extraction(
    text: str, model_id: str
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """Regex-based extraction with JSON-salvage fallback.

    Returns:
        ``(preamble, tools, is_salvaged)`` where *is_salvaged* is ``True``
        when the partial-JSON salvage path was used.
    """
    family = _model_family(model_id)
    start_tag, end_tag = TOOL_CALL_TAG_MAP[family]
    pattern = re.compile(rf"{start_tag}(.*?){end_tag}", re.DOTALL)

    tools: List[Dict[str, Any]] = []
    is_salvaged = False

    # Strip tags → clean preamble
    preamble = re.sub(pattern, "", text).strip()

    # Try structured extraction from inside tags
    for match in pattern.finditer(text):
        raw = match.group(1).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                tools.append({"name": parsed.get("name"), "arguments": args})
                continue
            except json.JSONDecodeError:
                pass
        # Try Gemma4 custom format
        g4 = _try_gemma4_extraction(raw)
        if g4 is not None:
            tools.append(g4)
            is_salvaged = True
            continue
        tools.append(raw)  # unparseable — store raw

    # Salvage: scan for bare JSON objects with "name" + "arguments"
    if not tools:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            m = re.search(r"\{", text[pos:])
            if not m:
                break
            pos += m.start()
            try:
                obj, idx = decoder.raw_decode(text[pos:])
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tools.append({"name": obj["name"], "arguments": args})
                    is_salvaged = True
                    preamble = preamble.replace(text[pos : pos + idx], "").strip()
                pos += idx
            except json.JSONDecodeError:
                pos += 1

    # Salvage (Gemma4): scan for call:func_name{...} in bare text
    if not tools:
        for g4_match in re.finditer(r"call:([\w\-\.]+)\{", text):
            start = g4_match.start()
            brace_depth = 1
            j = g4_match.end()
            while j < len(text) and brace_depth > 0:
                if text[j : j + 5] == '<|"|>':
                    j += 5
                elif text[j] == "{":
                    brace_depth += 1
                    j += 1
                elif text[j] == "}":
                    brace_depth -= 1
                    j += 1
                else:
                    j += 1
            segment = text[start:j]
            g4 = _try_gemma4_extraction(segment)
            if g4 is not None:
                tools.append(g4)
                is_salvaged = True
                preamble = preamble.replace(segment, "").strip()

    return preamble, tools, is_salvaged


# ---------------------------------------------------------------------------
# Log suppression context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _suppress_vllm_logs() -> Generator[None, None, None]:
    """Temporarily suppress vLLM-internal log output and stderr noise.

    vLLM parsers (granite4, qwen3_coder, etc.) use ``assert`` statements
    wrapped in ``try/except`` blocks that call ``traceback.print_exc()``
    directly to stderr — not through Python's ``logging`` module.
    Suppressing both the logger and stderr avoids flooding the terminal
    on large datasets.
    """
    vllm_logger = logging.getLogger("vllm")
    prev_level = vllm_logger.level
    vllm_logger.setLevel(logging.CRITICAL)
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(null_fd)
        vllm_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# ToolCallExtractor
# ---------------------------------------------------------------------------


class ToolCallExtractor(BaseExtractor):
    """Extract tool calls using vLLM parsers with best-effort fallback.

    Usage::

        extractor = ToolCallExtractor(tokenizer)   # or ToolCallExtractor(llm)
        result = extractor.extract(
            "<tool_call>{\"name\":\"get_weather\",...}</tool_call>",
            model_id="Qwen/Qwen2.5-0.5B"
        )
        print(result.detected_tool_calls)
    """

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: A HuggingFace tokenizer, or a vLLM ``LLM`` instance
                (which has a ``get_tokenizer()`` method), or ``None``.
                Only needed when using vLLM *ToolParserManager* integration.
                If ``None`` or vLLM is not installed, the best-effort
                fallback is always used.
        """
        # Accept both a plain tokenizer and a vLLM LLM wrapper
        if tokenizer is not None and hasattr(tokenizer, "get_tokenizer"):
            self._tokenizer = tokenizer.get_tokenizer()
        else:
            self._tokenizer = tokenizer

        self._parser_cache: Dict[str, Any] = {}
        self._vllm_available: Optional[bool] = None  # tri-state

    # ------------------------------------------------------------------
    # vLLM availability check
    # ------------------------------------------------------------------

    @property
    def vllm_available(self) -> bool:
        """Lazily-checked — ``True`` if the ``vllm`` package is importable."""
        if self._vllm_available is None:
            try:
                import vllm.tool_parsers  # noqa: F401

                self._vllm_available = True
            except ImportError:
                self._vllm_available = False
        return self._vllm_available

    # ------------------------------------------------------------------
    # Single extraction
    # ------------------------------------------------------------------

    def extract(self, model_output: str, model_id: str) -> ExtractionResult:
        generated = model_output.strip()

        # -- Try vLLM parser --------------------------------------------
        tool_parser = self._get_parser(model_id)
        if tool_parser is not None:
            try:
                with _suppress_vllm_logs():
                    extracted = tool_parser.extract_tool_calls(
                        model_output=generated, request=[]
                    )
            except Exception:
                extracted = None
        else:
            extracted = None

        preamble = generated
        tools: List[Dict[str, Any]] = []
        has_tools = False
        parser_used = "best_effort_decoder"

        if extracted is not None:
            if extracted.content is not None:
                preamble = extracted.content.strip()
            has_tools = extracted.tools_called
            if has_tools:
                for call in extracted.tool_calls:
                    if call.type == "function" and call.function:
                        args = call.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        tools.append({"name": call.function.name, "arguments": args})
                family = _model_family(model_id)
                parser_used = f"vLLM_{PARSER_MAP.get(family, 'unknown')}"

        # -- Fallback: regex + JSON salvage ------------------------------
        if not tools:
            preamble, tools, is_salvaged = best_effort_extraction(generated, model_id)
            parser_used = (
                "best_effort_salvage" if is_salvaged else "best_effort_decoder"
            )

        # -- Final safety net -------------------------------------------
        if not preamble and not tools:
            preamble = generated

        return ExtractionResult(
            conversational_preamble=preamble,
            detected_tool_calls=tools,
            raw_model_output=generated,
            parser_used=parser_used,
            is_best_effort=("best_effort" in parser_used),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_parser(self, model_id: str):
        """Return a cached vLLM tool parser for *model_id*."""
        family = _model_family(model_id)
        parser_name = PARSER_MAP.get(family)
        if parser_name is None:
            return None

        if parser_name in self._parser_cache:
            return self._parser_cache[parser_name]

        if self._tokenizer is None:
            logger.debug("No tokenizer provided — best-effort fallback only.")
            return None

        if not self.vllm_available:
            logger.debug("vLLM not installed — best-effort fallback only.")
            return None

        try:
            from vllm.tool_parsers import ToolParserManager

            parser = ToolParserManager.get_tool_parser(parser_name)(
                tokenizer=self._tokenizer
            )
            self._parser_cache[parser_name] = parser
            return parser
        except Exception as exc:
            logger.debug("Failed to initialise vLLM parser '%s': %s", parser_name, exc)
            return None
