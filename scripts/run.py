#!/usr/bin/env python3
"""
The runner. Serves Step E (calibration) and Step I (the full run) -- same code,
different config, so what gets calibrated is what gets run.

LOOP ORDER IS DELIBERATE: model on the outside, so each set of weights loads
once and is evicted before the next. Iterating tasks outermost would reload
weights constantly and guarantee swap on 8 GB. (quantcost's lesson, unchanged.)

WHAT THIS RUN DESIGN IS FOR -- the thing no published study has:
  many repeats of the SAME task instance, across several task instances, at
  each chain length. Repeats of one instance estimate that instance's own
  hazard; spread across instances estimates heterogeneity. Every prior study
  has one or the other, never both, so their observed decay is ambiguous
  between frailty and within-run dependence.

Resume is keyed on (model, task, n, repeat, temperature, scaffold). A run that
dies at hour six restarts where it stopped.

    python3 scripts/run.py --config configs/calibrate.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import (  # noqa: E402
    STOP_SEQUENCES, AgentConfig, run_episode,
)
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.runtimes.ollama_runtime import OllamaRuntime  # noqa: E402
from agentfrailty.schema import (  # noqa: E402
    EpisodeWriter, HostInfo, ModelSpec, completed_keys,
)
from agentfrailty.sysmon import RunGuard, cooldown, snapshot  # noqa: E402

DEFAULT = {
    "models": ["qwen2.5:1.5b-instruct", "qwen2.5:0.5b-instruct"],
    "chain_lengths": [1, 2, 3, 5, 8, 12, 20],
    "task_seeds": [1, 2, 3],       # distinct task instances per chain length
    "repeats": 10,                 # repeats of EACH instance
    "temperatures": [0.7],
    "distractors": 5,
    "out": "results/episodes.jsonl",
    "cooldown_s": 0.0,
    "scaffold": "react",
}


def load_config(path):
    cfg = dict(DEFAULT)
    if path:
        import yaml
        with open(path) as fh:
            cfg.update(yaml.safe_load(fh) or {})
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--out")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the cell count, run nothing")
    a = p.parse_args()

    cfg = load_config(a.config)
    out = a.out or cfg["out"]
    run_id = uuid.uuid4().hex[:12]
    host = HostInfo()

    # Enumerate the plan first, so the size of the run is known before it starts.
    plan = []
    for model_tag in cfg["models"]:
        for temp in cfg["temperatures"]:
            for n in cfg["chain_lengths"]:
                for ts in cfg["task_seeds"]:
                    for rep in range(cfg["repeats"]):
                        plan.append((model_tag, temp, n, ts, rep))

    done = set() if a.no_resume else completed_keys(out)
    todo = [c for c in plan
            if (c[0], "unknown", "ollama", "ledger",
                f"ledger-n{c[2]}-d{cfg['distractors']}-k1-s{c[3]}",
                c[2], c[4], c[1], cfg["scaffold"]) not in done]

    print(f"run_id     : {run_id}")
    print(f"models     : {cfg['models']}")
    print(f"chain len  : {cfg['chain_lengths']}")
    print(f"task seeds : {cfg['task_seeds']}  x  repeats {cfg['repeats']}")
    print(f"temps      : {cfg['temperatures']}")
    print(f"episodes   : {len(plan)} planned, {len(done)} already done, "
          f"{len(todo)} to run")
    print(f"out        : {out}")
    if a.dry_run:
        return

    t0 = time.time()
    n_written = 0
    with EpisodeWriter(out) as w:
        for model_tag in cfg["models"]:
            cells = [c for c in todo if c[0] == model_tag]
            if not cells:
                continue
            print(f"\n=== {model_tag}  ({len(cells)} episodes) ===")
            rt = OllamaRuntime(model_tag, stop=STOP_SEQUENCES)
            try:
                rt.load()
                spec = ModelSpec(
                    name=model_tag, quant="unknown", runtime="ollama",
                    repo_or_path=model_tag, runtime_version=rt.version(),
                    weight_bytes=rt.weight_bytes(),
                )
                rt.warmup()

                for k, (_, temp, n, ts, rep) in enumerate(cells):
                    task = make_task(seed=ts, n=n,
                                     n_distractors=cfg["distractors"])
                    mem_before = snapshot().to_dict()
                    with RunGuard() as guard:
                        row = run_episode(
                            task, rt, spec,
                            AgentConfig(temperature=temp,
                                        seed=1000 + rep,   # paired across cells
                                        scaffold=cfg["scaffold"]),
                            run_id=run_id, repeat=rep,
                        )
                    row.host = asdict(host)
                    row.mem_before = mem_before
                    row.mem_after = snapshot().to_dict()
                    health = guard.health.to_dict() if hasattr(guard, "health") else {}
                    row.health = health
                    row.timings_clean = bool(health.get("clean", True))
                    w.write(row)
                    n_written += 1

                    if row.final_state.get("env_errors"):
                        print("\n!! ENV ERROR -- ledger.py has a bug. Stopping.")
                        print(row.final_state["env_errors"])
                        sys.exit(1)

                    el = time.time() - t0
                    rate = n_written / el if el else 0
                    eta = (len(todo) - n_written) / rate if rate else 0
                    print(f"\r  {k + 1}/{len(cells)}  n={n:<3} seed={ts} rep={rep:<3}"
                          f" {row.termination:<14} {row.n_steps_taken:>2} steps"
                          f"  eta {eta / 60:5.1f}m".ljust(96), end="")
                    if cfg["cooldown_s"]:
                        cooldown(cfg["cooldown_s"])
            finally:
                rt.unload()   # keep_alive=0; a resident model swaps on 8 GB
            print()

    print(f"\nwrote {n_written} episodes to {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
