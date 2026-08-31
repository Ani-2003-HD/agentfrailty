"""
Tool-call parsing.

Shared by the kill-test and (later) the agent loop, so that what counts as a
well-formed call is defined in exactly one place. If the agent loop and the
scorer disagreed about what parses, every number downstream would be wrong in
a way that is very hard to see.

Deliberately LENIENT about surface form and STRICT about content. We are
measuring whether the model can decide and express an action, not whether it
matches one house style of JSON. A model that wraps its call in a markdown
fence has not failed at tool use. A model that invents a tool has.

Returns raw parse facts only -- no notion of "correct". Correctness needs a
task goal, which lives elsewhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Common wrappers small models emit. Stripped before JSON hunting.
_FENCE = re.compile(r"```(?:json|tool_call|python)?\s*|\s*```")
_TAGS = re.compile(r"</?(?:tool_call|function_call|tool|function)>", re.I)

# Keys different model families use for the same two concepts.
_NAME_KEYS = ("name", "tool", "tool_name", "function", "action")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "action_input")


@dataclass
class ParsedCall:
    ok: bool = False
    name: Optional[str] = None
    args: Optional[dict] = None
    error: str = ""
    span: str = ""  # the JSON text actually parsed, for auditing


def _candidates(text: str):
    """
    Yield balanced {...} substrings, outermost first.

    A regex cannot match balanced braces, and small models nest objects inside
    "arguments" constantly, so scan explicitly.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def parse_tool_call(text: str) -> ParsedCall:
    """Extract a single tool call from raw model output."""
    if not text or not text.strip():
        return ParsedCall(error="empty_output")

    cleaned = _TAGS.sub(" ", _FENCE.sub(" ", text))

    saw_json = False
    for span in _candidates(cleaned):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        saw_json = True

        name = next((obj[k] for k in _NAME_KEYS
                     if isinstance(obj.get(k), str) and obj[k].strip()), None)
        if name is None:
            continue

        args = next((obj[k] for k in _ARG_KEYS if isinstance(obj.get(k), dict)), None)
        if args is None:
            # Some models flatten arguments to the top level. Accept that, but
            # only after removing the name key, so {"name": "x"} does not
            # silently become an argument.
            leftover = {k: v for k, v in obj.items()
                        if k not in _NAME_KEYS and k not in _ARG_KEYS}
            args = leftover if leftover else None

        return ParsedCall(ok=True, name=name.strip(), args=args, span=span)

    return ParsedCall(error="json_found_but_no_tool_name" if saw_json else "no_json")


def render_tools(tools: list[dict]) -> str:
    """Human/model-readable tool listing for the system prompt."""
    return json.dumps(tools, indent=2)
