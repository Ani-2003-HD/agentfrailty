#!/usr/bin/env python3
"""
The frailty ablation: WHY are some instances impossible?

The pilot found instance-level frailty so strong it is bimodal -- at n=6, three
instances succeeded ~0% of the time and eight ~95%, ICC 0.784. Every single
failure was a PREMATURE SUBMIT: perfect navigation, then stopping one or two
records short, near-deterministically per instance (14/15, 15/15, 0/15).

So something about a specific instance decides whether the model believes it has
finished. This isolates what.

  base            the instance exactly as the pilot ran it
  values-only     same words, different numbers
  ids-only        same numbers, different words
  both            fresh labels and numbers (control)

The graph topology is identical in every arm, so the only thing varying is the
factor named. If hard instances stay hard under new values, the cause is lexical
or structural. If they become easy, it is the numbers. If neither factor moves
them, the cause is a deeper transcript-level effect and that is a finding too.

    python3 scripts/ablate.py --hard 3 5 10 --easy 11 12 4 --variants 5 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import (  # noqa: E402
    STOP_SEQUENCES, AgentConfig, calls_from_steps, run_episode,
)
from agentfrailty.envs.ledger import make_task, remix_task  # noqa: E402
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime  # noqa: E402
from agentfrailty.schema import ModelSpec, code_version  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402


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
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=12)
    ap.add_argument("--hard", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--easy", type=int, nargs="+", default=[11, 12, 4])
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/ablation.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # (label, base_seed, arm, ids_seed, values_seed)
    jobs = []
    for label, seeds in (("hard", a.hard), ("easy", a.easy)):
        for bs in seeds:
            jobs.append((label, bs, "base", None, None))
            for v in range(a.variants):
                jobs.append((label, bs, "values-only", None, 500 + v))
                jobs.append((label, bs, "ids-only", 500 + v, None))
                jobs.append((label, bs, "both", 900 + v, 900 + v))

    total = len(jobs) * a.repeats
    print(f"model    : {a.model}")
    print(f"n        : {a.n}   distractors {a.distractors}")
    print(f"hard     : {a.hard}")
    print(f"easy     : {a.easy}")
    print(f"variants : {a.variants} per arm x {a.repeats} repeats")
    print(f"episodes : {total}  (~{total * 3.7 / 60:.0f} min)\n")

    rt = OllamaRuntime(a.model, stop=STOP_SEQUENCES)
    spec = ModelSpec(name=a.model, quant="unknown", runtime="ollama",
                     repo_or_path=a.model)
    res = defaultdict(lambda: [0, 0])          # (label, arm) -> [ok, n]
    per_base = defaultdict(lambda: [0, 0])     # (label, base, arm) -> [ok, n]
    premature = defaultdict(lambda: [0, 0])
    t0 = time.time()
    done = 0

    try:
        rt.load()
        spec.runtime_version = rt.version()
        rt.warmup()
        with open(a.out, "a", encoding="utf-8") as fh:
            for (label, bs, arm, i_s, v_s) in jobs:
                base = make_task(seed=bs, n=a.n, n_distractors=a.distractors)
                task = (base if arm == "base"
                        else remix_task(base, ids_seed=i_s, values_seed=v_s))
                for rep in range(a.repeats):
                    row = run_episode(
                        task, rt, spec,
                        AgentConfig(temperature=a.temperature, seed=1000 + rep),
                        repeat=rep)
                    g = grade_episode(task, calls_from_steps(row.steps))
                    ok = 1 if g.walked_canonical else 0
                    res[(label, arm)][0] += ok
                    res[(label, arm)][1] += 1
                    per_base[(label, bs, arm)][0] += ok
                    per_base[(label, bs, arm)][1] += 1
                    prem = g.error_counts.get("premature_submit", 0) > 0
                    premature[(label, arm)][0] += 1 if prem else 0
                    premature[(label, arm)][1] += 1

                    fh.write(json.dumps({
                        "model": a.model, "group": label, "base_seed": bs,
                        "arm": arm, "ids_seed": i_s, "values_seed": v_s,
                        "task_id": task.task_id, "chain_length": a.n,
                        "repeat": rep,
                        "walked_canonical": bool(ok),
                        "outcome_correct": g.outcome_correct,
                        "first_error_index": g.first_error_index,
                        "error_counts": g.error_counts,
                        "termination": row.termination,
                        "n_steps": row.n_steps_taken,
                        "code": code_version(),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    el = time.time() - t0
                    eta = (total - done) / (done / el) if el else 0
                    print(f"\r  {done}/{total}  {label}/{arm} s{bs} "
                          f"eta {eta / 60:4.1f}m".ljust(72), end="")
    finally:
        rt.unload()
    print("\n")

    print("navigation success by arm")
    print("(hard instances becoming easy under an arm => that factor is the cause)\n")
    print(f"  {'group':<7}{'arm':<14}{'rate':>7}  {'95% CI':<16}{'n':>5}"
          f"{'premature submit':>19}")
    print("  " + "-" * 70)
    for label in ("hard", "easy"):
        for arm in ("base", "values-only", "ids-only", "both"):
            k, n = res[(label, arm)]
            if not n:
                continue
            lo, hi = wilson(k, n)
            pk, pn = premature[(label, arm)]
            print(f"  {label:<7}{arm:<14}{k / n:>7.2f}  "
                  f"[{lo:.2f},{hi:.2f}]{'':<4}{n:>5}{pk / pn:>19.2f}")
        print()

    print("per base instance")
    print(f"  {'group':<7}{'seed':>5}  " +
          "".join(f"{arm:>14}" for arm in ("base", "values-only", "ids-only", "both")))
    print("  " + "-" * 66)
    for label in ("hard", "easy"):
        seeds = a.hard if label == "hard" else a.easy
        for bs in seeds:
            cells = []
            for arm in ("base", "values-only", "ids-only", "both"):
                k, n = per_base[(label, bs, arm)]
                cells.append(f"{k / n:.2f}" if n else "-")
            print(f"  {label:<7}{bs:>5}  " + "".join(f"{c:>14}" for c in cells))
    print(f"\nraw -> {a.out}")

    # -- the verdict --
    hb = res[("hard", "base")]
    hv = res[("hard", "values-only")]
    hi_ = res[("hard", "ids-only")]
    if all(x[1] for x in (hb, hv, hi_)):
        b, v, i = hb[0] / hb[1], hv[0] / hv[1], hi_[0] / hi_[1]
        print("\n--- reading ---")
        print(f"  hard instances: base {b:.2f} | new values {v:.2f} | new ids {i:.2f}")
        if v - b > 0.25 and i - b < 0.15:
            print("  => THE NUMBERS drive it. Difficulty lives in the values.")
        elif i - b > 0.25 and v - b < 0.15:
            print("  => THE WORDS drive it. Difficulty is lexical.")
        elif v - b > 0.25 and i - b > 0.25:
            print("  => BOTH factors move it; difficulty is not attributable to one.")
        else:
            print("  => NEITHER factor rescues them. Difficulty survives relabelling")
            print("     and revaluing, so it is carried by the wiring itself or by")
            print("     something deeper in the transcript. A finding in its own right.")


if __name__ == "__main__":
    main()
