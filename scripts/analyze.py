#!/usr/bin/env python3
"""
Read episodes.jsonl, re-grade every episode, print the calibration tables.

RE-GRADES FROM RAW ON EVERY INVOCATION. Nothing derived is stored, so when the
scorer changes this picks it up with no re-inference -- the design decision that
saved quantcost a day of recompute, and matters more here because grading a
multi-step episode has more judgement calls in it.

Stdlib only, so it runs without installing the analysis extras.

    python3 scripts/analyze.py --in results/calibrate.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import calls_from_steps  # noqa: E402
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402


def wilson(k, n, z=1.96):
    """Wilson interval. Exact-ish at small n, and unlike the normal
    approximation it does not produce impossible bounds at 0 or 1 -- which is
    where most of these cells sit."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def task_for(row):
    """Rebuild the exact task from its id -- generation is deterministic, so no
    task data needs to be stored in the rows."""
    tid = row["task_id"]                      # ledger-n{N}-d{D}-k{K}-s{S}
    parts = tid.split("-")
    n = int(parts[1][1:]); d = int(parts[2][1:])
    k = int(parts[3][1:]); s = int(parts[4][1:])
    return make_task(seed=s, n=n, n_distractors=d, keys_per_step=k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/calibrate.jsonl")
    a = p.parse_args()

    rows = []
    with open(a.inp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # tolerate a torn final line
    if not rows:
        sys.exit(f"no episodes in {a.inp}")

    # Provenance check FIRST. An append-only file can accumulate rows from
    # several versions of the code; averaging over them is how a fixed bug
    # quietly contaminates the fixed data.
    versions = defaultdict(int)
    for r in rows:
        c = r.get("code", {})
        versions[(c.get("commit", "")[:8] or "unknown", bool(c.get("dirty")))] += 1
    if len(versions) > 1:
        print("!! WARNING: this file mixes code versions --")
        for (c, d), k in sorted(versions.items(), key=lambda x: -x[1]):
            print(f"     {c}{' (dirty)' if d else '':<9}  {k} episodes")
        print("   Filter by commit, or delete and re-run. Do not average "
              "across a bug fix.\n")
    elif versions:
        (c, d), k = next(iter(versions.items()))
        print(f"code: {c}{' (DIRTY TREE)' if d else ''}   {k} episodes\n")

    cells = defaultdict(list)
    for r in rows:
        t = task_for(r)
        g = grade_episode(t, calls_from_steps(r["steps"]))
        cells[(r["model"]["name"], r["chain_length"])].append((r, g, t))

    models = sorted({m for m, _ in cells})
    ns = sorted({n for _, n in cells})

    print(f"{len(rows)} episodes | {len(models)} models | chain lengths {ns}\n")

    for m in models:
        print(f"=== {m} ===")
        print(f"{'n':>3} {'eps':>4}  {'navigation':<22}{'outcome':<22}"
              f"{'arith|reads':<14}{'repaired':<9}{'steps':<7}{'terminations'}")
        print("-" * 108)
        for n in ns:
            items = cells.get((m, n))
            if not items:
                continue
            k = len(items)
            nav = sum(1 for _, g, _ in items if g.walked_canonical)
            out = sum(1 for _, g, _ in items if g.outcome_correct)
            ar = [g.arithmetic_correct_given_reads for _, g, _ in items
                  if g.arithmetic_correct_given_reads is not None]
            rep = sum(1 for r, _, _ in items
                      for s in r["steps"] if s.get("repaired"))
            steps = sum(r["n_steps_taken"] for r, _, _ in items) / k
            terms = defaultdict(int)
            for r, _, _ in items:
                terms[r["termination"]] += 1

            lo, hi = wilson(nav, k)
            lo2, hi2 = wilson(out, k)
            print(f"{n:>3} {k:>4}  "
                  f"{nav / k:>5.2f} [{lo:.2f},{hi:.2f}]     "
                  f"{out / k:>5.2f} [{lo2:.2f},{hi2:.2f}]     "
                  f"{(sum(1 for x in ar if x) / len(ar) if ar else float('nan')):>5.2f}"
                  f" ({len(ar):>2})  "
                  f"{rep:>6}   {steps:>5.1f}  "
                  f"{dict(terms)}")
        print()

    # -- conditional step accuracy, the curve that matters -------------------
    print("conditional step accuracy by step index")
    print("(navigation only -- a submit step is locally correct whenever the")
    print(" chain had ended, regardless of the number submitted)")
    print("flat => failures look independent; falling => dependence or frailty\n")
    for m in models:
        print(f"  {m}")
        num = defaultdict(int); den = defaultdict(int)
        for (mm, _), items in cells.items():
            if mm != m:
                continue
            for _, g, _ in items:
                for s in g.steps:
                    den[s.index] += 1
                    num[s.index] += 1 if s.locally_correct else 0
        for i in sorted(den):
            if den[i] < 3:
                continue          # too few episodes reach here to plot
            v = num[i] / den[i]
            lo, hi = wilson(num[i], den[i])
            bar = "#" * int(round(v * 28))
            print(f"    step {i:>2}  {v:>5.2f} [{lo:.2f},{hi:.2f}] "
                  f"n={den[i]:<4} {bar}")
        print()

    env_errs = sum(len(r["final_state"].get("env_errors", [])) for r in rows)
    dirty = sum(1 for r in rows if not r.get("timings_clean", True))
    print(f"ENV ERRORS: {env_errs}  (must be 0)")
    print(f"episodes with unclean timing/thermal conditions: {dirty}/{len(rows)}")


if __name__ == "__main__":
    main()
