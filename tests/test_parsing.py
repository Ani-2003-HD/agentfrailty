"""
Parser tests.

quantcost's lesson: test the scorer BEFORE the big run, not after. The parser
is the single point where a bug corrupts every downstream number invisibly --
a model that tool-calls fine but trips the parser looks exactly like a model
that cannot tool-call.

The cases below are real surface forms small models produce.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.parsing import parse_tool_call  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  ok  {name}")


# -- things that MUST parse -------------------------------------------------

def test_plain():
    p = parse_tool_call('{"name": "get_weather", "arguments": {"city": "Bengaluru"}}')
    check("plain", p.ok and p.name == "get_weather" and p.args["city"] == "Bengaluru")


def test_markdown_fence():
    p = parse_tool_call('```json\n{"name": "get_weather", "arguments": {"city": "X"}}\n```')
    check("fenced", p.ok and p.name == "get_weather")


def test_preamble_prose():
    p = parse_tool_call('Sure! I will look that up.\n{"name": "get_weather", "arguments": {"city": "X"}}')
    check("preamble", p.ok and p.name == "get_weather")


def test_tool_call_tags():
    p = parse_tool_call('<tool_call>{"name": "get_weather", "arguments": {"city": "X"}}</tool_call>')
    check("xml tags", p.ok and p.name == "get_weather")


def test_alt_keys():
    p = parse_tool_call('{"tool": "get_weather", "parameters": {"city": "X"}}')
    check("alt keys", p.ok and p.name == "get_weather" and p.args["city"] == "X")


def test_flattened_args():
    p = parse_tool_call('{"name": "get_weather", "city": "X", "unit": "celsius"}')
    check("flattened args", p.ok and p.args["city"] == "X" and "name" not in p.args)


def test_nested_braces():
    """Balanced-brace scanning: a naive regex truncates this."""
    p = parse_tool_call('{"name": "send_email", "arguments": {"body": {"html": "<p>hi</p>"}}}')
    check("nested braces", p.ok and isinstance(p.args["body"], dict))


# -- arithmetic expressions where a number belongs --------------------------
# Found in the very first smoke run: qwen2.5-1.5b emitted
#   {"name": "submit", "arguments": {"total": 67 - 2 - 46}}
# 67-2-46 is 19, which was the correct answer. Rejecting it would score a
# formatting slip as an arithmetic failure.

def test_arithmetic_expression_is_repaired():
    p = parse_tool_call('{"name": "submit", "arguments": {"total": 67 - 2 - 46}}')
    check("expression evaluated", p.ok and p.args["total"] == 19)
    check("repair is flagged", p.repaired)
    check("repair is recorded for audit", p.repairs == [("67 - 2 - 46", 19)])


def test_parenthesised_expression():
    p = parse_tool_call('{"name": "submit", "arguments": {"total": (67 - 2) - 46}}')
    check("parenthesised expression evaluated", p.ok and p.args["total"] == 19)


def test_plain_literal_is_not_marked_repaired():
    p = parse_tool_call('{"name": "submit", "arguments": {"total": 19}}')
    check("literal parses without repair", p.ok and not p.repaired)


def test_negative_literal_is_not_repaired():
    p = parse_tool_call('{"name": "submit", "arguments": {"total": -5}}')
    check("negative literal untouched", p.ok and p.args["total"] == -5
          and not p.repaired)


def test_repair_never_executes_code():
    """The repair pass evaluates arithmetic only -- never names or calls."""
    hostile = [
        '{"name": "submit", "arguments": {"total": __import__("os").system("x")}}',
        '{"name": "submit", "arguments": {"total": open("/etc/passwd")}}',
        '{"name": "submit", "arguments": {"total": [].__class__}}',
    ]
    check("code in a numeric slot never parses",
          all(not parse_tool_call(h).ok for h in hostile))


# -- things that MUST NOT parse ---------------------------------------------

def test_empty():
    check("empty", not parse_tool_call("").ok)


def test_prose_only():
    p = parse_tool_call("The temperature in Bengaluru is about 24 degrees celsius.")
    check("prose only", not p.ok and p.error == "no_json")


def test_json_without_name():
    p = parse_tool_call('{"city": "Bengaluru", "unit": "celsius"}')
    check("json without name", not p.ok and p.error == "json_found_but_no_tool_name")


def test_broken_json():
    p = parse_tool_call('{"name": "get_weather", "arguments": {"city": ')
    check("truncated json", not p.ok)


def test_name_not_string():
    check("non-string name", not parse_tool_call('{"name": 42, "arguments": {}}').ok)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} parser tests")
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} passed")
