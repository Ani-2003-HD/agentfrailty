#!/usr/bin/env python3
"""
Curated vocabulary test: is instance difficulty ABSTRACTNESS?

The word sweep found difficulty is lexical and large -- the same task, same
graph, same values, varying only the six path words, spans 0.00 to 1.00 with a
bimodal distribution and ICC 0.717. It is NOT tokenisation: prompt length was
flat at 353 tokens across every word set.

Eyeballing the extremes suggested a pattern:

  hardest   about, story, order, later, never, usual, occur, pause, haste, vague
  easiest   floor, snake, cargo, table, brain, night, pulse, trail, burst

Abstract, temporal and discourse words on one side; concrete physical nouns on
the other. `pause`, `later`, `never`, `haste` are literally about stopping and
time -- and the failure mode is ALWAYS premature submit, never a wrong id.

HYPOTHESIS: word sets drawn from instruction-like or temporal vocabulary make
the model read its own transcript as prose rather than as opaque identifiers,
and prose has endings.

DESIGN, and the discipline in it:

  * The pools below were written from a PRINCIPLED definition -- concrete means
    physically imageable, abstract means not -- and deliberately include words
    that cut AGAINST the hypothesis (`third`, `until`, `count` appeared in easy
    sets from the sweep but are abstract, so they go in the abstract pool). No
    cherry-picking from the observed extremes.
  * Distractor ids are FIXED and identical across arms, so only the path
    vocabulary varies.
  * A random arm drawn from the full WORDS list is the control.
  * Graph and values are held fixed throughout, as in the sweep.

    python3 scripts/vocab.py --draws 10 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import (  # noqa: E402
    STOP_SEQUENCES, AgentConfig, calls_from_steps, run_episode,
)
from agentfrailty.envs.ledger import (  # noqa: E402
    WORDS, make_task, relabel_task,
)
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime  # noqa: E402
from agentfrailty.schema import ModelSpec, code_version  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402
from agentfrailty.stats import permutation_test, wilson  # noqa: E402

# Non-imageable: temporal, logical, discourse, quantity.
ABSTRACT = [
    "about", "after", "again", "cause", "doubt", "event", "extra", "first",
    "focus", "given", "issue", "later", "level", "logic", "maybe", "merit",
    "never", "occur", "order", "other", "proof", "quite", "sense", "since",
    "still", "thing", "third", "total", "until", "usual", "vague", "worth",
]

# Physically imageable objects.
CONCRETE = [
    "apple", "brain", "bread", "cabin", "cargo", "chair", "cliff", "cloth",
    "crown", "floor", "glove", "grass", "horse", "house", "knife", "lemon",
    "mouse", "onion", "piano", "plant", "river", "robot", "shirt", "skirt",
    "snake", "stone", "table", "tiger", "torch", "train", "wagon", "wheel",
]

# Identical in every arm, so only the PATH vocabulary varies.
FIXED_DISTRACTORS = [
    "paper", "metal", "water", "light", "sound", "color",
    "shape", "frame", "angle", "board", "stick", "brick",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--base-seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/vocab.jsonl")
    a = ap.parse_args()

    assert all(len(w) == 5 for w in ABSTRACT + CONCRETE + FIXED_DISTRACTORS)
    assert not (set(ABSTRACT) & set(CONCRETE))
    assert not (set(FIXED_DISTRACTORS) & (set(ABSTRACT) | set(CONCRETE)))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    base = make_task(seed=a.base_seed, n=a.n,
                     n_distractors=len(FIXED_DISTRACTORS))

    arms = {"abstract": ABSTRACT, "concrete": CONCRETE, "random": WORDS}
    jobs = []
    for arm, pool in arms.items():
        rng = random.Random(hash(arm) % 10000)
        for d in range(a.draws):
            path = rng.sample([w for w in pool if w not in FIXED_DISTRACTORS],
                              a.n)
            jobs.append((arm, d, path))

    total = len(jobs) * a.repeats
    print(f"model : {a.model}")
    print(f"base  : seed {a.base_seed}, n={a.n} (graph + values FIXED, "
          f"distractors FIXED)")
    print(f"arms  : {list(arms)}  x {a.draws} draws x {a.repeats} repeats")
    print(f"episodes: {total}  (~{total * 3.7 / 60:.0f} min)\n")

    rt = OllamaRuntime(a.model, stop=STOP_SEQUENCES)
    spec = ModelSpec(name=a.model, quant="unknown", runtime="ollama",
                     repo_or_path=a.model)
    per_draw = {}
    t0, done = time.time(), 0

    try:
        rt.load()
        spec.runtime_version = rt.version()
        rt.warmup()
        with open(a.out, "a", encoding="utf-8") as fh:
            for arm, d, path in jobs:
                task = relabel_task(base, path + FIXED_DISTRACTORS)
                ok, toks, prem = 0, [], 0
                for rep in range(a.repeats):
                    row = run_episode(
                        task, rt, spec,
                        AgentConfig(temperature=a.temperature, seed=1000 + rep),
                        repeat=rep)
                    g = grade_episode(task, calls_from_steps(row.steps))
                    ok += 1 if g.walked_canonical else 0
                    prem += 1 if g.error_counts.get("premature_submit") else 0
                    if row.steps:
                        toks.append(row.steps[0]["prompt_tokens"])
                    fh.write(json.dumps({
                        "model": a.model, "arm": arm, "draw": d, "path": path,
                        "repeat": rep,
                        "walked_canonical": bool(g.walked_canonical),
                        "first_error_index": g.first_error_index,
                        "error_counts": g.error_counts,
                        "prompt_tokens": row.steps[0]["prompt_tokens"] if row.steps else 0,
                        "code": code_version(),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    el = time.time() - t0
                    print(f"\r  {done}/{total}  {arm} draw {d}  eta "
                          f"{((total - done) / (done / el) / 60) if el else 0:4.1f}m"
                          .ljust(60), end="")
                per_draw[(arm, d)] = {
                    "rate": ok / a.repeats, "prem": prem / a.repeats,
                    "path": path,
                    "tok": sum(toks) / len(toks) if toks else 0,
                }
    finally:
        rt.unload()
    print("\n")

    print("by arm\n")
    print(f"  {'arm':<10}{'rate':>7}  {'95% CI':<15}{'premature':>11}"
          f"{'tokens':>9}   per-draw rates")
    print("  " + "-" * 86)
    rates = {}
    for arm in arms:
        ds = [v for (aa, _), v in per_draw.items() if aa == arm]
        rates[arm] = [v["rate"] for v in ds]
        k = int(round(sum(v["rate"] for v in ds) * a.repeats))
        n = len(ds) * a.repeats
        lo, hi = wilson(k, n)
        prem = sum(v["prem"] for v in ds) / len(ds)
        tok = sum(v["tok"] for v in ds) / len(ds)
        print(f"  {arm:<10}{sum(rates[arm])/len(ds):>7.2f}  "
              f"[{lo:.2f},{hi:.2f}]{'':<3}{prem:>11.2f}{tok:>9.0f}   "
              f"{' '.join(f'{x:.1f}' for x in sorted(rates[arm]))}")

    print("\n\npermutation tests on per-draw rates\n")
    for x, y in (("abstract", "concrete"), ("abstract", "random"),
                 ("concrete", "random")):
        d, p = permutation_test(rates[x], rates[y], seed=7)
        star = "  *** significant" if p < 0.05 else "  (not significant)"
        print(f"  {x:<9} vs {y:<9}  diff {d:+.3f}   p = {p:.4f}{star}")

    print("\n\nhypothesis: abstract/temporal words => more premature submits\n")
    ab = sum(1 for x in rates["abstract"] if x < 0.5)
    co = sum(1 for x in rates["concrete"] if x < 0.5)
    print(f"  draws below 0.5:  abstract {ab}/{len(rates['abstract'])}   "
          f"concrete {co}/{len(rates['concrete'])}")
    dv = sum(rates["abstract"]) / len(rates["abstract"]) \
        - sum(rates["concrete"]) / len(rates["concrete"])
    if dv < -0.2:
        print("  => SUPPORTED: abstract vocabulary is materially harder.")
    elif dv > 0.2:
        print("  => REVERSED: concrete vocabulary is harder. The eyeball "
              "pattern was backwards.")
    else:
        print("  => NOT SUPPORTED: abstractness does not explain the effect.")
        print("     The lexical effect is real (ICC 0.72) but this is not its"
              " cause.")

    print("\n  worst draws in each arm:")
    for arm in arms:
        ds = sorted(((v["rate"], v["path"]) for (aa, _), v in per_draw.items()
                     if aa == arm))[:2]
        for r, path in ds:
            print(f"    {arm:<9} {r:.2f}  {' -> '.join(path)}")

    print(f"\nraw -> {a.out}")


if __name__ == "__main__":
    main()
