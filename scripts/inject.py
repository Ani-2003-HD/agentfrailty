#!/usr/bin/env python3
"""
Step F: the healed-history experiment.

Holds context LENGTH fixed and varies only the CORRECTNESS of its contents, then
measures accuracy on the very next step.

  rate 0.00  -> a fully healed history. Any gap from step-1 accuracy is
                LONG-CONTEXT degradation.
  rate rises -> if accuracy falls with it, that is SELF-CONDITIONING, because
                nothing else about the prompt changed.

Following 2509.09677 S3.2. Going beyond it on the LOCATION axis, which their
Appendix I calls intractable in their setting.

    python3 scripts/inject.py --model qwen2.5:1.5b-instruct --repeats 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import (  # noqa: E402
    STOP_SEQUENCES, SYSTEM_TEMPLATE, render_transcript,
)
from agentfrailty.envs.ledger import LedgerEnv, make_task  # noqa: E402
from agentfrailty.injection import (  # noqa: E402
    InjectionSpec, build_history, grade_probe,
)
from agentfrailty.parsing import parse_tool_call, render_tools  # noqa: E402
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime  # noqa: E402
from agentfrailty.schema import code_version  # noqa: E402

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--n", type=int, default=14, help="task chain length")
    ap.add_argument("--distractors", type=int, default=30)
    ap.add_argument("--prefix-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--task-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--locations", nargs="+", default=["uniform"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/injection.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cells = [(loc, rate, ts, rep)
             for loc in a.locations for rate in RATES
             for ts in a.task_seeds for rep in range(a.repeats)]

    print(f"model        : {a.model}")
    print(f"task         : n={a.n} distractors={a.distractors}")
    print(f"prefix steps : {a.prefix_steps}  (context held fixed)")
    print(f"rates        : {RATES}")
    print(f"locations    : {a.locations}")
    print(f"probes       : {len(cells)}  "
          f"({len(a.task_seeds)} tasks x {a.repeats} repeats per cell)\n")

    rt = OllamaRuntime(a.model, stop=STOP_SEQUENCES)
    results = defaultdict(Counter)
    ctx_chars = defaultdict(list)
    t0 = time.time()

    try:
        rt.load()
        rt.warmup()
        with open(a.out, "a", encoding="utf-8") as fh:
            for k, (loc, rate, ts, rep) in enumerate(cells):
                task = make_task(seed=ts, n=a.n, n_distractors=a.distractors)
                spec = InjectionSpec(error_rate=rate,
                                     prefix_steps=a.prefix_steps,
                                     location=loc, seed=ts * 100 + rep)
                h = build_history(task, spec)

                env = LedgerEnv(task)
                system = SYSTEM_TEMPLATE.format(
                    tools=render_tools(env.tool_specs()),
                    task=env.initial_observation())
                prompt = render_transcript(system, h.turns)

                gen = rt.generate(prompt, max_tokens=128,
                                  temperature=a.temperature,
                                  seed=1000 + rep)
                if gen.error:
                    verdict, parsed = "runtime_error", None
                else:
                    parsed = parse_tool_call(gen.text)
                    verdict = grade_probe(
                        h,
                        parsed.name if parsed.ok else None,
                        parsed.args if parsed.ok else None)

                results[(loc, rate)][verdict] += 1
                ctx_chars[(loc, rate)].append(len(prompt))

                fh.write(json.dumps({
                    "model": a.model, "location": loc, "error_rate": rate,
                    "task_seed": ts, "repeat": rep,
                    "prefix_steps": a.prefix_steps,
                    "injected_indices": h.injected_indices,
                    "final_position": h.final_position,
                    "expected_target": h.expected_target,
                    "prompt_chars": len(prompt),
                    "raw_output": gen.text, "error": gen.error,
                    "tool": parsed.name if parsed and parsed.ok else None,
                    "args": parsed.args if parsed and parsed.ok else None,
                    "verdict": verdict,
                    "code": code_version(),
                }, ensure_ascii=False) + "\n")
                fh.flush()

                el = time.time() - t0
                eta = (len(cells) - k - 1) / ((k + 1) / el) if el else 0
                print(f"\r  {k + 1}/{len(cells)}  {loc} rate={rate:.2f} "
                      f"[{verdict}]  eta {eta / 60:4.1f}m".ljust(80), end="")
    finally:
        rt.unload()
    print("\n")

    # -- the fixed-length control, verified on the ACTUAL prompts sent -------
    print("context length actually sent (the control -- must be flat):")
    for loc in a.locations:
        for rate in RATES:
            v = ctx_chars[(loc, rate)]
            if v:
                print(f"  {loc:<8} rate={rate:.2f}  mean {sum(v) / len(v):7.0f} chars"
                      f"  min {min(v)}  max {max(v)}")
    allv = [x for v in ctx_chars.values() for x in v]
    if allv:
        drift = (max(allv) - min(allv)) / min(allv)
        print(f"  overall spread: {drift:.1%}"
              f"{'   <- OK' if drift < 0.15 else '   <- TOO WIDE, confounded'}")

    print("\nprobe accuracy by injected error rate")
    print("(falling with rate => self-conditioning; flat => long-context only)\n")
    for loc in a.locations:
        print(f"  location = {loc}")
        base = None
        for rate in RATES:
            c = results[(loc, rate)]
            n = sum(c.values())
            if not n:
                continue
            k = c["correct"]
            p = k / n
            lo, hi = wilson(k, n)
            if base is None:
                base = p
            bar = "#" * int(round(p * 30))
            delta = f"  {p - base:+.2f}" if base is not None else ""
            print(f"    rate {rate:.2f}  {p:>5.2f} [{lo:.2f},{hi:.2f}] "
                  f"n={n:<4} {bar}{delta}")
        print(f"    verdicts: "
              f"{ {r: dict(results[(loc, r)]) for r in RATES} }")
        print()

    print(f"raw -> {a.out}")


if __name__ == "__main__":
    main()
