#!/usr/bin/env python3
"""
The word-set sweep: what makes a set of words hard?

The ablation showed difficulty is lexical. Holding the graph and the numbers
fixed and swapping only the five-letter words moved every instance sharply --
hard ones from 0.00 to 0.32-0.70, easy ones from 1.00 down to 0.46-0.58 --
while swapping the numbers barely moved anything. And with words AND values
held fixed, three completely different graph topologies gave IDENTICAL
performance (0.66 on all six bases). So: not the numbers, not the structure.
The tokens.

This sweeps many word sets over a fixed graph and fixed values, to answer:

  1. How wide is the distribution of difficulty across word draws? If it is
     bimodal, "difficulty" is not a property of a task but of its vocabulary.
  2. Is difficulty PREDICTABLE from features of the word set?

FEATURES, and why each is plausible:

  prompt_tokens   the model's own tokenisation of the whole prompt, taken from
                  prompt_eval_count. Words that split into more subword tokens
                  make a longer, noisier transcript. This is the mechanism most
                  likely to matter and it costs nothing to record.
  letter_overlap  mean shared letters between consecutive path ids -- lexical
                  confusability along the chain.
  shared_initial  path ids sharing a first letter; the cheapest confusion.
  repeat_letters  ids with doubled letters, which tokenise irregularly.

    python3 scripts/wordsweep.py --base-seed 11 --word-sets 40 --repeats 10
"""

from __future__ import annotations

import argparse
import json
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
from agentfrailty.stats import (  # noqa: E402
    boot_icc, dispersion, pearson_r, wilson,
)


def features(task) -> dict:
    """Lexical features of a task's canonical path."""
    path = task.canonical_path
    overlaps = [len(set(path[i]) & set(path[i + 1])) for i in range(len(path) - 1)]
    initials = [w[0] for w in path]
    return {
        "letter_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "shared_initial": len(initials) - len(set(initials)),
        "repeat_letters": sum(1 for w in path if len(set(w)) < len(w)),
        "distinct_letters": sum(len(set(w)) for w in path) / len(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--base-seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=12)
    ap.add_argument("--word-sets", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/wordsweep.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    base = make_task(seed=a.base_seed, n=a.n, n_distractors=a.distractors)
    total = a.word_sets * a.repeats

    print(f"model      : {a.model}")
    print(f"base       : seed {a.base_seed}, n={a.n}  "
          f"(graph and values held FIXED)")
    print(f"word sets  : {a.word_sets} x {a.repeats} repeats = {total} episodes"
          f"  (~{total * 3.7 / 60:.0f} min)\n")

    rt = OllamaRuntime(a.model, stop=STOP_SEQUENCES)
    spec = ModelSpec(name=a.model, quant="unknown", runtime="ollama",
                     repo_or_path=a.model)
    per_set = {}
    t0, done = time.time(), 0

    try:
        rt.load()
        spec.runtime_version = rt.version()
        rt.warmup()
        with open(a.out, "a", encoding="utf-8") as fh:
            for w in range(a.word_sets):
                ids_seed = 2000 + w
                task = remix_task(base, ids_seed=ids_seed)
                f = features(task)
                ok = 0
                toks = []
                for rep in range(a.repeats):
                    row = run_episode(
                        task, rt, spec,
                        AgentConfig(temperature=a.temperature, seed=1000 + rep),
                        repeat=rep)
                    g = grade_episode(task, calls_from_steps(row.steps))
                    ok += 1 if g.walked_canonical else 0
                    if row.steps:
                        toks.append(row.steps[0]["prompt_tokens"])
                    fh.write(json.dumps({
                        "model": a.model, "base_seed": a.base_seed,
                        "ids_seed": ids_seed, "repeat": rep,
                        "path": task.canonical_path,
                        "walked_canonical": bool(g.walked_canonical),
                        "first_error_index": g.first_error_index,
                        "error_counts": g.error_counts,
                        "prompt_tokens": row.steps[0]["prompt_tokens"] if row.steps else 0,
                        "features": f, "code": code_version(),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    el = time.time() - t0
                    print(f"\r  {done}/{total}  set {w} eta "
                          f"{((total - done) / (done / el) / 60) if el else 0:4.1f}m"
                          .ljust(60), end="")
                per_set[ids_seed] = {
                    "ok": ok, "n": a.repeats, "rate": ok / a.repeats,
                    "path": task.canonical_path,
                    "prompt_tokens": sum(toks) / len(toks) if toks else 0,
                    **f,
                }
    finally:
        rt.unload()
    print("\n")

    rates = sorted(v["rate"] for v in per_set.values())
    print("distribution of difficulty across word sets")
    print(f"  {len(rates)} word sets, identical graph and values\n")
    buckets = defaultdict(int)
    for r in rates:
        buckets[min(int(r * 10), 9)] += 1
    for b in range(10):
        lab = f"{b/10:.1f}-{(b+1)/10:.1f}"
        print(f"    {lab:<9} {'#' * buckets[b]}{'' if buckets[b] else ''} "
              f"{buckets[b] or ''}")
    print(f"\n  min {min(rates):.2f}   median {rates[len(rates)//2]:.2f}   "
          f"max {max(rates):.2f}   spread {max(rates)-min(rates):.2f}")

    counts = [v["ok"] for v in per_set.values()]
    ks = [v["n"] for v in per_set.values()]
    d = dispersion(counts, ks)
    if d:
        p, phi, icc = d
        lo, hi = boot_icc(counts, ks, seed=11)
        print(f"\n  pooled p={p:.2f}  phi={phi:.2f}  "
              f"ICC(word set)={icc:.3f} [{lo:.3f},{hi:.3f}]")
        print("  => the VOCABULARY alone carries this much of the variance"
              if lo > 0 else "  => word set does not explain the variance")

    print("\n\ndoes any feature predict difficulty?\n")
    feats = ["prompt_tokens", "letter_overlap", "shared_initial",
             "repeat_letters", "distinct_letters"]
    ys = [v["rate"] for v in per_set.values()]
    print(f"  {'feature':<18}{'r':>8}   interpretation")
    print("  " + "-" * 60)
    for f in feats:
        xs = [v[f] for v in per_set.values()]
        r = pearson_r(xs, ys)
        if r != r:
            tag = "undefined (no variation)"
        elif abs(r) < 0.25:
            tag = "no relationship"
        elif r < 0:
            tag = "MORE of this => HARDER"
        else:
            tag = "MORE of this => EASIER"
        print(f"  {f:<18}{r:>8.3f}   {tag}")

    print("\n\nhardest and easiest word sets\n")
    ordered = sorted(per_set.items(), key=lambda kv: kv[1]["rate"])
    for tag, items in (("HARDEST", ordered[:4]), ("EASIEST", ordered[-4:])):
        print(f"  {tag}")
        for ids_seed, v in items:
            lo, hi = wilson(v["ok"], v["n"])
            print(f"    {v['rate']:.2f} [{lo:.2f},{hi:.2f}]  "
                  f"tok={v['prompt_tokens']:.0f}  "
                  f"{' -> '.join(v['path'])}")
        print()

    print(f"raw -> {a.out}")


if __name__ == "__main__":
    main()
