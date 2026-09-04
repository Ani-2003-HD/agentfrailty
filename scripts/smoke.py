#!/usr/bin/env python3
"""
Step D: smoke run. A few real episodes, printed step by step.

This is an EYEBALL test, not a measurement. The point is to read actual
transcripts and catch the things tests cannot: a prompt that confuses the model,
an observation format it misreads, a systematic quirk that would silently bias
every number in the full run.

Read the transcripts. Do not skip to the summary.

    ollama serve
    python3 scripts/smoke.py --model qwen2.5:1.5b-instruct --n 3 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import AgentConfig, calls_from_steps, run_episode  # noqa: E402
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime  # noqa: E402
from agentfrailty.schema import EpisodeWriter, ModelSpec  # noqa: E402
from agentfrailty.scorers.trajectory import (  # noqa: E402
    grade_episode, step_accuracy_by_index,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b-instruct")
    p.add_argument("--n", type=int, default=3, help="chain length")
    p.add_argument("--distractors", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--task-seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", default="results/smoke.jsonl")
    p.add_argument("--quiet", action="store_true", help="skip raw transcripts")
    a = p.parse_args()

    task = make_task(seed=a.task_seed, n=a.n, n_distractors=a.distractors)
    model = ModelSpec(name=a.model, quant="unknown", runtime="ollama",
                      repo_or_path=a.model)

    print(f"task      : {task.task_id}")
    print(f"canonical : {' -> '.join(task.canonical_path)}")
    print(f"goal total: {task.goal_total}")
    print(f"model     : {a.model}   repeats={a.repeats}  temp={a.temperature}\n")

    rt = OllamaRuntime(a.model)
    grades, rows = [], []
    try:
        rt.load()
        model.runtime_version = rt.version()
        model.weight_bytes = rt.weight_bytes()
        rt.warmup()

        with EpisodeWriter(a.out) as w:
            for rep in range(a.repeats):
                row = run_episode(
                    task, rt, model,
                    AgentConfig(temperature=a.temperature, seed=1000 + rep),
                    repeat=rep,
                )
                w.write(row)
                rows.append(row)
                g = grade_episode(task, calls_from_steps(row.steps))
                grades.append(g)

                print(f"--- repeat {rep}  [{row.termination}]  "
                      f"{row.n_steps_taken} steps  {row.total_s:.1f}s")
                print(f"{'i':>2}  {'canon':<6}{'local':<6}{'error':<18}"
                      f"{'call':<34}expected")
                for s, sr in zip(g.steps, row.steps):
                    call = f"{s.tool}({s.target!r})" if s.tool else "<unparsed>"
                    print(f"{s.index:>2}  {str(s.on_canonical):<6}"
                          f"{str(s.locally_correct):<6}{s.error_type or '-':<18}"
                          f"{call[:33]:<34}{s.expected_action}")
                    if not a.quiet and not s.locally_correct:
                        raw = (sr['raw_output'] or '').replace('\n', ' ')[:150]
                        print(f"      raw: {raw}")
                print(f"    submitted={g.submitted} goal={g.goal_total} "
                      f"correct={g.outcome_correct}\n")
    finally:
        rt.unload()

    # -- summary --
    n = len(grades)
    solved = sum(1 for g in grades if g.outcome_correct)
    walked = sum(1 for g in grades if g.walked_canonical)
    env_errs = sum(len(r.final_state["env_errors"]) for r in rows)
    parse_fail = sum(1 for r in rows for s in r.steps if not s["parse_ok"])
    total_steps = sum(r.n_steps_taken for r in rows)

    print("=" * 60)
    print(f"outcome correct      : {solved}/{n}")
    print(f"walked canonical     : {walked}/{n}")
    print(f"parse failures       : {parse_fail}/{total_steps} steps")
    print(f"ENV ERRORS           : {env_errs}   (must be 0)")
    print(f"terminations         : "
          f"{ {t: sum(1 for r in rows if r.termination == t) for t in {r.termination for r in rows}} }")

    errs = {}
    for g in grades:
        for k, v in g.error_counts.items():
            errs[k] = errs.get(k, 0) + v
    print(f"error types          : {errs or 'none'}")

    curve = step_accuracy_by_index(grades)
    print("\nconditional step accuracy by index "
          "(flat => independent, falling => dependence):")
    for i, v in curve.items():
        bar = "#" * int(round(v * 40))
        print(f"  step {i:>2}  {v:>5.2f}  {bar}")

    print(f"\nraw -> {a.out}")
    if env_errs:
        print("\n!! ENV ERRORS PRESENT -- ledger.py has a bug. Stop and fix it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
