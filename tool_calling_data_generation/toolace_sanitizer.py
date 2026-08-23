import json
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
import ast

# ----------------- Configuration -----------------

SYSTEM_PROMPT_INTRO = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.
Here is a list of functions in JSON format that you can invoke:
"""

SYSTEM_PROMPT_OUTRO = """
Should you decide to return the function call(s), you MUST return them in the following JSON format:
[{"name": "function_name", "arguments": {"arg_name": "arg_value"}}]

NO other text MUST be included.
"""

# ----------------- Helpers -----------------

def sanitize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "tool_" + name
    return name

#TODO: Move from sequence based to Regex or more systematic approach
def extract_tools_list_from_system(system_text: str):
    anchor = "Here is a list of functions"
    start_pos = system_text.find(anchor)
    s = system_text[start_pos:] if start_pos != -1 else system_text

    i0 = s.find("[")
    if i0 == -1:
        return None

    depth = 0
    start = None
    in_str, quote, esc = False, "", False

    for i in range(i0, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    block = s[start:i + 1]
                    try:
                        obj = json.loads(block)
                        if isinstance(obj, list) and any("name" in x for x in obj):
                            return obj
                    except:
                        pass
                    return None
    return None


def parse_args_string(arg_str: str):
    if not arg_str.strip():
        return {}

    items, depth, buf = [], 0, ""
    in_str, quote, esc = False, "", False

    for ch in arg_str:
        if in_str:
            buf += ch
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue

        if ch in ("'", '"'):
            in_str, quote = True, ch
            buf += ch
            continue

        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1

        if ch == "," and depth == 0:
            items.append(buf.strip())
            buf = ""
        else:
            buf += ch

    if buf.strip():
        items.append(buf.strip())

    out = {}
    for it in items:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()

        try:
            if v == "True":
                v_parsed = True
            elif v == "False":
                v_parsed = False
            elif v == "None":
                v_parsed = None
            else:
                v_parsed = ast.literal_eval(v)
        except:
            v_parsed = v.strip("'\"")

        out[k] = v_parsed

    return out


def parse_assistant_call(s: str, allowed_names: set, name_map: dict):
    s = s.strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    if not s:
        return None

    parts, depth, buf = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1

        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch

    if buf.strip():
        parts.append(buf.strip())

    tool_calls = []

    for part in parts:
        m = re.match(r"^([a-zA-Z0-9_ \.]+)\s*\((.*)\)\s*$", part)
        if not m:
            continue

        raw_name, raw_args = m.groups()
        name = name_map.get(raw_name, sanitize_name(raw_name))

        if name not in allowed_names:
            continue

        args = parse_args_string(raw_args)
        tool_calls.append({"name": name, "arguments": args})

    if not tool_calls:
        return None

    return tool_calls


def process_tool_response_content(content: str, name_map: dict):
    content = content.strip()

    if content.startswith("[") and content.endswith("]"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        raw = item["name"]
                        item["name"] = name_map.get(raw, sanitize_name(raw))
                return json.dumps(data, ensure_ascii=False)
        except:
            pass

    return content

def build_sanitized_toolace(size=None):
    split_str = f"train[:{size}]" if size else "train"
    dataset = load_dataset("Team-ACE/ToolACE", "default", split=split_str)

    all_records = []

    for example in tqdm(dataset):
        original_system_text = example.get("system", "")
        tools_list = extract_tools_list_from_system(original_system_text)
        if tools_list is None:
            continue

        name_map = {}
        for t in tools_list:
            original_name = t.get("name", "")
            if original_name:
                clean_name = sanitize_name(original_name)
                name_map[original_name] = clean_name
                t["name"] = clean_name

        allowed_names = set(name_map.values())

        tools_json_str = json.dumps(tools_list, separators=(",", ":"))
        new_system_prompt = f"{SYSTEM_PROMPT_INTRO}{tools_json_str}\n{SYSTEM_PROMPT_OUTRO}"

        out_messages = []
        valid = True

        for msg in example.get("conversations", []):
            role = msg.get("from")
            val = msg.get("value", "")

            if role == "assistant":
                s_val = val.strip()
                if s_val.startswith("[") and "(" in s_val:
                    parsed = parse_assistant_call(val, allowed_names, name_map)
                    if parsed:
                        val = json.dumps(parsed, ensure_ascii=False)
                    else:
                        valid = False
                        break

            elif role == "tool":
                val = process_tool_response_content(val, name_map)

            out_messages.append({
                "role": "user" if role == "human" else role,
                "content": val
            })

        if valid and out_messages:
            all_records.append({
                "system": new_system_prompt,
                "conversations": out_messages,
                "tools": tools_json_str
            })

    return Dataset.from_list(all_records)



