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
from dataclasses import dataclass, field
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
    span: str = ""       # the JSON text actually parsed, for auditing
    repaired: bool = False   # an arithmetic expression was evaluated to a literal
    repairs: list = field(default_factory=list)   # [(raw expr, value)]


# Values that are arithmetic EXPRESSIONS rather than literals, e.g.
#   {"name": "submit", "arguments": {"total": 67 - 2 - 46}}
# This is not valid JSON, but the model has done the work and stated the right
# sum -- 67-2-46 is 19. Rejecting it measures JSON typing discipline and calls
# the result arithmetic, which is the ceiling trap quantcost was built to avoid.
#
# So: evaluate it, and RECORD that we did. `repaired` travels with the parse, so
# analysis can report strict and lenient numbers separately and the choice stays
# reversible without re-running inference.
_EXPR = re.compile(r'("(?:[A-Za-z_][\w]*)"\s*:\s*)(-?[\d\s()+\-*/.]{3,}?)(\s*[,}])')


def _safe_arith(expr: str):
    """Evaluate a pure-arithmetic expression. Returns None if it is anything else."""
    import ast

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        val = eval(compile(tree, "<arith>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return val if isinstance(val, (int, float)) else None


def repair_arithmetic(text: str):
    """Replace expression-valued fields with their evaluated literals."""
    repairs = []

    def sub(m):
        prefix, expr, tail = m.group(1), m.group(2), m.group(3)
        if re.fullmatch(r"-?\s*[\d.]+\s*", expr):
            return m.group(0)          # already a literal; leave it alone
        val = _safe_arith(expr)
        if val is None:
            return m.group(0)
        repairs.append((expr.strip(), val))
        return f"{prefix}{val}{tail}"

    return _EXPR.sub(sub, text), repairs


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

    result = _parse_from(cleaned)
    if result.ok:
        return result

    # Second pass: the model may have written an arithmetic expression where a
    # number belongs. Evaluate and retry.
    repaired_text, repairs = repair_arithmetic(cleaned)
    if repairs:
        second = _parse_from(repaired_text)
        if second.ok:
            second.repaired = True
            second.repairs = repairs
            return second
    return result


def _parse_from(cleaned: str) -> ParsedCall:
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
