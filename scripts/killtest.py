#!/usr/bin/env python3
"""
Kill-test: can small local models emit a well-formed tool call at all?

This is the Phase-2 gate. If a model cannot clear a ONE-STEP tool call
reliably, it cannot participate in a chain-length study -- any decay curve
measured with it would be measuring the task, not the depth. That is
quantcost's ceiling trap, caught before it costs a week.

Grading is four-level on purpose. "Emitted a call" and "emitted the RIGHT call"
are different capabilities, and collapsing them is exactly how quantcost's JSON
task produced a meaningless 100% across every quantisation rung.

The task is deliberately trivial: one obviously-correct tool, one plausible
decoy, one unambiguous prompt. A floor test, not a difficulty test.

Run:
    ollama serve
    ollama pull qwen2.5:1.5b-instruct
    python scripts/killtest.py --models qwen2.5:1.5b-instruct qwen2.5:0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.parsing import parse_tool_call, render_tools   # noqa: E402
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime   # noqa: E402

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

KNOWN_TOOLS = {t["name"] for t in TOOLS}
EXPECTED_TOOL = "get_weather"

PROMPT = """You have access to these tools:

{tools}

To use a tool, reply with ONLY a JSON object of this form:
{{"name": "<tool name>", "arguments": {{...}}}}

Reply with nothing else.

User: What is the temperature in Bengaluru right now, in celsius?"""


def grade(parsed) -> str:
    """Four-level grade. Raw parse -> verdict, all in one visible place."""
    if not parsed.ok:
        return "no_call"
    if parsed.name not in KNOWN_TOOLS:
        return "hallucinated_tool"
    if parsed.name != EXPECTED_TOOL:
        return "wrong_tool"
    args = parsed.args or {}
    city = str(args.get("city", "")).strip().lower()
    unit = str(args.get("unit", "")).strip().lower()
    if "bengaluru" not in city and "bangalore" not in city:
        return "bad_args"
    if unit != "celsius":
        return "bad_args"
    return "correct"


GRADES = ["correct", "bad_args", "wrong_tool", "hallucinated_tool",
          "no_call", "runtime_error"]


def run_model(tag, runs, temperature, max_tokens, out_fh):
    prompt = PROMPT.format(tools=render_tools(TOOLS))
    counts, samples = Counter(), []
    t_model = time.time()

    rt = OllamaRuntime(tag)
    try:
        rt.load()
        rt.warmup()
        for rep in range(runs):
            # Same seed convention as quantcost: fixed per repeat and shared
            # across models, so comparisons are paired.
            seed = 1000 + rep
            gen = rt.generate(prompt, max_tokens=max_tokens,
                              temperature=temperature, seed=seed)
            if gen.error:
                g, parsed = "runtime_error", None
            else:
                parsed = parse_tool_call(gen.text)
                g = grade(parsed)
            counts[g] += 1

            out_fh.write(json.dumps({
                "model": tag, "repeat": rep, "seed": seed,
                "temperature": temperature,
                "raw_output": gen.text, "error": gen.error,
                "parse_ok": bool(parsed and parsed.ok),
                "tool_name": parsed.name if parsed else None,
                "tool_args": parsed.args if parsed else None,
                "grade": g,
                "total_s": gen.total_s,
                "completion_tokens": gen.completion_tokens,
            }, ensure_ascii=False) + "\n")
            out_fh.flush()

            if g != "correct" and len(samples) < 3:
                samples.append((rep, g, (gen.text or gen.error)[:160].replace("\n", " ")))
            print(f"\r  {tag}  {rep + 1}/{runs}  [{g}]".ljust(70),
                  end="", file=sys.stderr)
    finally:
        rt.unload()   # keep_alive=0; on 8 GB a resident model guarantees swap
    print(file=sys.stderr)

    return counts, samples, round(time.time() - t_model, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out", default="results/killtest_raw.jsonl")
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    results = {}
    with open(a.out, "a", encoding="utf-8") as fh:
        for tag in a.models:
            counts, samples, secs = run_model(
                tag, a.runs, a.temperature, a.max_tokens, fh)
            results[tag] = (counts, samples, secs)

    w = max(len(m) for m in results) + 2
    print(f"\nkill-test | runs={a.runs} | temp={a.temperature}\n")
    print("model".ljust(w) + "".join(g[:9].rjust(11) for g in GRADES) + "     sec")
    print("-" * (w + 11 * len(GRADES) + 8))
    for tag, (counts, _, secs) in results.items():
        print(tag.ljust(w)
              + "".join(str(counts.get(g, 0)).rjust(11) for g in GRADES)
              + str(secs).rjust(8))

    print("\n--- failures ---")
    for tag, (_, samples, _) in results.items():
        for rep, g, txt in samples:
            print(f"  {tag} rep{rep} [{g}]: {txt}")
        if not samples:
            print(f"  {tag}: none")

    print("\n--- gate ---")
    for tag, (counts, _, _) in results.items():
        pct = 100 * counts.get("correct", 0) / a.runs
        called = 100 * (a.runs - counts.get("no_call", 0)
                        - counts.get("runtime_error", 0)) / a.runs
        if pct >= 90:
            v = "PASS      usable for chain-length work"
        elif pct >= 60:
            v = "MARGINAL  usable only if the ceiling trap is handled explicitly"
        else:
            v = "FAIL      cannot participate; N=1 is already broken"
        print(f"  {tag:<28} correct {pct:5.1f}%   emitted-a-call {called:5.1f}%   {v}")

    print(f"\nraw -> {a.out}")


if __name__ == "__main__":
    main()
