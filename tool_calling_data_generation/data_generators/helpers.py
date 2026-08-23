import ast
import json
import os
import sys
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from datasets import Dataset

from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset


@contextmanager
def _suppress_dill_bytecode():
    """Temporarily redirect stdout to devnull.

    ``dill==0.4.1`` calls ``dis.dis()`` during function serialisation
    (``dill.detect.nestedglobals``, line 148) to inspect bytecode for
    free-variable extraction.  The output is captured internally by
    ``dill.temp.capture('stdout')`` but can leak when Prefect or
    another framework wraps ``sys.stdout``.  Suppressing stdout during
    ``datasets.map`` calls prevents the bytecode disassembly from
    appearing as noisy INFO log messages.
    """
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def try_parse_json_like(val: Any):
    if isinstance(val, (dict, list)):
        return val
    if val is None:
        return None
    if not isinstance(val, str):
        return None

    s = val.strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def get_conv_id(entry: Dict[str, Any]) -> str:
    existing = entry.get("uuid")
    if existing is None:
        return str(uuid.uuid4())
    return str(existing)


def build_system_prompt(available_tools: Any, prompt_format: str = "custom") -> str:
    if not available_tools:
        return "You are a helpful assistant."

    # When using the chat_template format, the tokenizer's apply_chat_template
    # handles tool rendering natively via the `tools` parameter. We should NOT
    # embed tools in the system message content, otherwise they get duplicated.
    if prompt_format == "chat_template":
        return "You are a helpful assistant."

    tools = try_parse_json_like(available_tools)
    if not isinstance(tools, list):
        tools_str = str(available_tools)
    else:
        tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

    return (
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures:\n"
        f"{tools_str}\n\n"
        "For each function call, return a json object with function name and arguments:\n"
        '{"name": <function-name>, "arguments": <args-json-object>}'
    )


def normalize_messages(messages_raw: Any) -> List[Dict[str, Any]]:
    messages = try_parse_json_like(messages_raw)
    if not isinstance(messages, list):
        return []

    out = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append(msg)

    if out and out[0].get("role") == "system":
        out = out[1:]

    return out


def to_template_messages(
    history: List[Dict[str, str]],
    clean_sys: str,
) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": clean_sys}]
    msgs.extend(history[1:])
    return msgs


def render_prompt(
    history: List[Dict[str, str]],
    clean_sys: str,
    tokenizer,
    prompt_format: str = "custom",
    available_tools: Any = None,
) -> str:
    if prompt_format == "chat_template":
        if (
            not hasattr(tokenizer, "apply_chat_template")
            or tokenizer.chat_template is None
        ):
            raise ValueError(
                "prompt_format='chat_template' but tokenizer has no chat_template"
            )
        messages = to_template_messages(history, clean_sys)
        # Parse tools if they are stored as a string (e.g., JSON string)
        parsed_tools = try_parse_json_like(available_tools)
        if not isinstance(parsed_tools, list):
            parsed_tools = None
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=parsed_tools,
            add_generation_prompt=True,
        )

    if prompt_format == "custom":
        prompt_lines = [f"System: {clean_sys}"]
        prompt_lines.extend(f"{h['role']}: {h['content']}" for h in history[1:])
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines)

    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def build_prompt_examples_from_entry(
    entry: Dict[str, Any],
    tokenizer,
    prompt_format: str = "custom",
) -> Dict[str, List[Any]]:
    available_tools = entry.get("available_tools")
    clean_sys = build_system_prompt(available_tools, prompt_format)
    conv_id = get_conv_id(entry)

    messages = normalize_messages(entry.get("messages", "[]"))
    history = [{"role": "system", "content": clean_sys}]

    prompts = []
    raw_prompts = []
    chosens = []
    metas = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        function_call = msg.get("function_call")

        if role == "assistant" and function_call:
            chosen = json.dumps(function_call, ensure_ascii=False)

            prompt = render_prompt(
                history=history,
                clean_sys=clean_sys,
                tokenizer=tokenizer,
                prompt_format=prompt_format,
                available_tools=available_tools,
            )

            prompts.append(prompt)
            raw_prompts.append(json.dumps(history[1:], ensure_ascii=False))
            chosens.append(chosen)
            metas.append(
                {
                    "conv_id": conv_id,
                    "conv_id_tracking": f"{conv_id}_{i}",
                    "turn_idx": i,
                    "tools": available_tools,
                    "subset_name": entry.get("subset_name"),
                }
            )

            history.append({"role": "assistant", "content": chosen})

        elif role == "assistant" and content:
            history.append({"role": "assistant", "content": content})

        elif role == "function":
            history.append({"role": "tool", "content": content})

        else:
            history.append({"role": role, "content": content})

    return {
        "prompt": prompts,
        "raw_prompt": raw_prompts,
        "chosen": chosens,
        "meta": metas,
    }


def clean_chosen_arguments(chosen_str: str) -> str:
    try:
        chosen = json.loads(chosen_str)
        if isinstance(chosen, dict) and "arguments" in chosen:
            args = chosen["arguments"]
            if isinstance(args, str):
                try:
                    args_json = json.loads(args)
                    chosen["arguments"] = args_json
                except json.JSONDecodeError:
                    pass
            return json.dumps(chosen, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return chosen_str


def explode_prompt_rows(
    batch: Dict[str, List[Any]],
    tokenizer,
    prompt_format: str,
) -> Dict[str, List[Any]]:
    out_prompt = []
    out_raw_prompt = []
    out_chosen = []
    out_meta = []

    n = len(batch["messages"])
    for idx in range(n):
        entry = {k: batch[k][idx] for k in batch}
        built = build_prompt_examples_from_entry(
            entry=entry,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
        )

        out_prompt.extend(built["prompt"])
        out_raw_prompt.extend(built["raw_prompt"])
        out_chosen.extend(built["chosen"])
        out_meta.extend(built["meta"])

    return {
        "prompt": out_prompt,
        "raw_prompt": out_raw_prompt,
        "chosen": out_chosen,
        "meta": out_meta,
    }


def add_prompt_lengths(batch: Dict[str, List[Any]], tokenizer) -> Dict[str, List[int]]:
    enc = tokenizer(
        batch["prompt"],
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )
    return {"prompt_len": enc["length"]}


def preprocess_toolcall_prompts(
    ds: Dataset,
    tokenizer,
    usable_ctx: int,
    max_new_tokens: int = 8192,
    num_proc: Optional[int] = None,
    cache_disable: bool = False,
    prompt_format: str = "custom",
    keep_columns: Optional[List[str]] = None,
):
    if keep_columns is None:
        keep_columns = [
            c
            for c in ["uuid", "messages", "available_tools", "subset_name"]
            if c in ds.column_names
        ]

    ds = ds.select_columns(keep_columns)

    worker_count = num_proc if (num_proc and num_proc > 1) else os.cpu_count()
    cache_ok = not cache_disable

    ds_prompts = ds.map(
        lambda batch: explode_prompt_rows(batch, tokenizer, prompt_format),
        remove_columns=ds.column_names,
        num_proc=worker_count,
        batched=True,
        batch_size=512,
        load_from_cache_file=cache_ok,
        desc="Building prompt rows",
    )

    ds_prompts = ds_prompts.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]),
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Dropping empty rows",
    )

    ds_prompts = ds_prompts.map(
        lambda batch: add_prompt_lengths(batch, tokenizer),
        batched=True,
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Tokenizing prompts for length",
    )

    ds_prompts = ds_prompts.filter(
        lambda x: x["prompt_len"] + max_new_tokens <= usable_ctx,
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Filtering overlength prompts",
    )

    skipped_truncation = len(
        ds_prompts.filter(
            lambda x: x["prompt_len"] + max_new_tokens > usable_ctx,
            num_proc=worker_count,
            load_from_cache_file=False,
            desc="Counting skipped prompts",
        )
    )

    # add prompt_format to metadata
    def add_format_to_meta(batch):
        for meta in batch["meta"]:
            meta["prompt_format"] = prompt_format
        return batch

    ds_prompts = ds_prompts.map(
        add_format_to_meta,
        batched=True,
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Adding prompt format to metadata",
    )

    # Clean chosen arguments
    def clean_chosen(batch):
        batch["chosen"] = [clean_chosen_arguments(chosen) for chosen in batch["chosen"]]
        return batch

    ds_prompts = ds_prompts.map(
        clean_chosen,
        batched=True,
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Cleaning chosen arguments",
    )

    # Promote conv_id and conv_id_tracking from meta to top-level columns
    # Promote conv_id, conv_id_tracking, tools, and subset_name from meta to top-level columns
    def promote_meta_fields(batch):
        batch["conv_id"] = [m["conv_id"] for m in batch["meta"]]
        batch["conv_id_tracking"] = [m["conv_id_tracking"] for m in batch["meta"]]
        batch["tools"] = [m["tools"] for m in batch["meta"]]
        batch["subset_name"] = [m["subset_name"] for m in batch["meta"]]
        batch["type"] = ["tool_call"] * len(batch["meta"])
        return batch

    ds_prompts = ds_prompts.map(
        promote_meta_fields,
        batched=True,
        num_proc=worker_count,
        load_from_cache_file=cache_ok,
        desc="Promoting conv_id, conv_id_tracking, tools, subset_name, and type to columns",
    )

    return ds_prompts, len(ds), skipped_truncation, len(ds_prompts)


# ---------------------------------------------------------------------------
# CanonicalDataset pipeline helpers
# ---------------------------------------------------------------------------
def _no_thinking_kwargs(model_id: str = "") -> dict[str, Any]:
    model_id = model_id.lower()

    # Extend this registry as models are added to your pipeline.
    if "qwen" in model_id or "gemma" in model_id:
        return {"enable_thinking": False}

    # No documented template control exists for this model family.
    return {}


def _render_prompt_from_canonical(
    messages: list,
    tools: list,
    tokenizer,
    prompt_format: str = "chat_template",
    model_id: str = "",
) -> str:
    """Render a single prompt string from canonical ``messages`` and ``tools``."""
    if prompt_format == "chat_template":
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        chat_messages = [system_msg] + list(messages)
        if (
            not hasattr(tokenizer, "apply_chat_template")
            or tokenizer.chat_template is None
        ):
            raise ValueError(
                "prompt_format='chat_template' but tokenizer has no chat_template"
            )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            tools=tools if tools else None,
            add_generation_prompt=True,
            **_no_thinking_kwargs(model_id=model_id),
        )

    # custom format
    prompt_lines = ["System: You are a helpful assistant."]
    prompt_lines.extend(f"{h['role']}: {h['content']}" for h in messages)
    prompt_lines.append("Assistant:")
    return "\n".join(prompt_lines)


def prepare_canonical_for_generation(
    cd: "CanonicalDataset",
    tokenizer,
    usable_ctx: int,
    max_new_tokens: int = 512,
    prompt_format: str = "chat_template",
    num_proc: Optional[int] = None,
    cache_disable: bool = False,
    model_id: str = "",
) -> "CanonicalDataset":
    """Render prompts and filter overlength samples from a ``CanonicalDataset``.

    This is the canonical-dataset equivalent of
    ``preprocess_toolcall_prompts``.  It assumes the dataset already
    contains ``messages`` and ``tool_catalog`` columns (populated by
    the loader).  It adds ``prompt`` and ``prompt_len`` columns and
    drops samples whose tokenized prompt exceeds ``usable_ctx -
    max_new_tokens``.

    Args:
        cd: CanonicalDataset from a loader.
        tokenizer: HuggingFace tokenizer (must have ``apply_chat_template``
            if *prompt_format* is ``"chat_template"``).
        usable_ctx: Maximum model context length in tokens.
        max_new_tokens: Tokens reserved for generation.
        prompt_format: ``"chat_template"`` or ``"custom"``.
        num_proc: Number of worker processes for ``ds.map``.
        cache_disable: Disable HF datasets disk cache.

    Returns:
        A new ``CanonicalDataset`` with ``prompt`` and ``prompt_len``
        columns added, filtered to fits-in-context samples.
    """
    from tool_calling_data_generation.data_generators.models.canonical_format import CanonicalDataset

    worker_count = num_proc if (num_proc and num_proc > 1) else os.cpu_count()
    cache_ok = not cache_disable

    # -- 1. Render prompt strings -----------------------------------------
    def _render_batch(batch):
        n = len(batch["conv_id"])
        prompts = []
        for i in range(n):
            msgs = batch["messages"][i] or []
            tools_raw = batch["tool_catalog"][i]
            if isinstance(tools_raw, str):
                import json as _json

                tools = _json.loads(tools_raw) if tools_raw else []
            else:
                tools = tools_raw or []
            prompts.append(
                _render_prompt_from_canonical(
                    msgs, tools, tokenizer, prompt_format, model_id=model_id
                )
            )
        return {"prompt": prompts}

    with _suppress_dill_bytecode():
        cd = cd.map(
            _render_batch,
            batched=True,
            num_proc=worker_count,
            load_from_cache_file=cache_ok,
            desc="Rendering prompts from canonical format",
        )

    # -- 2. Tokenize for length -------------------------------------------
    def _add_lengths(batch):
        enc = tokenizer(
            batch["prompt"],
            add_special_tokens=False,
            return_length=True,
            padding=False,
            truncation=False,
        )
        return {"prompt_len": enc["length"]}

    with _suppress_dill_bytecode():
        cd = cd.map(
            _add_lengths,
            batched=True,
            num_proc=worker_count,
            load_from_cache_file=cache_ok,
            desc="Tokenizing prompts for length",
        )

    # -- 3. Filter overlength ---------------------------------------------
    with _suppress_dill_bytecode():
        cd = cd.filter(
            lambda r: r["prompt_len"] + max_new_tokens <= usable_ctx,
            num_proc=worker_count,
            load_from_cache_file=cache_ok,
            desc="Filtering overlength prompts",
        )

    return cd


