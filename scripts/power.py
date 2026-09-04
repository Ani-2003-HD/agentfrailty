#!/usr/bin/env python3
"""
Step G analysis: can we see frailty, and how big must the full run be?

THE QUESTION THIS ANSWERS

Every published study of agent decay is ambiguous between two mechanisms:

  frailty            some task instances are simply harder
  within-run dependence  one error raises the odds of the next

2603.29231 infers the second from super-linear decay -- but a gap between the
mean of p and (mean p)^T is exactly what heterogeneity ALONE produces
(Jensen's inequality: E[p^T] != (E[p])^T). They never consider the alternative;
the words "heterogeneity", "frailty" and "random effect" do not appear.

Separating them needs many repeats of the SAME instance across MANY instances.
This script measures whether that separation is visible in the pilot.

THE STATISTIC

For m task instances, each run k times, with y_i successes:

  phi = (1/(m-1)) * sum_i (y_i - k*p)^2 / (k*p*(1-p))        Pearson dispersion

Under a pure binomial model (every instance identical) E[phi] = 1. phi > 1 means
instances differ -- overdispersion. Converted to an intraclass correlation:

  ICC = (phi - 1) / (k - 1)

ICC is the share of variance attributable to WHICH INSTANCE you drew, and it is
exactly the frailty term. Verified absent from tau-bench, tau^2-bench and BFCL,
none of which publish run-to-run variance at all.

Bootstrap CIs over instances, because phi is itself noisy at small m.

Stdlib only.

    python3 scripts/power.py --in results/pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import calls_from_steps  # noqa: E402
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402


def task_for(row):
    parts = row["task_id"].split("-")
    return make_task(seed=int(parts[4][1:]), n=int(parts[1][1:]),
                     n_distractors=int(parts[2][1:]),
                     keys_per_step=int(parts[3][1:]))


def dispersion(counts, k_each):
    """
    Pearson dispersion phi and the implied ICC.

    counts   successes per instance
    k_each   trials per instance
    Returns (p_hat, phi, icc) or None when undefined (p at 0 or 1).
    """
    m = len(counts)
    if m < 2:
        return None
    total_k = sum(k_each)
    p = sum(counts) / total_k
    if p <= 0 or p >= 1:
        return None
    chi = sum((y - k * p) ** 2 / (k * p * (1 - p))
              for y, k in zip(counts, k_each))
    phi = chi / (m - 1)
    kbar = total_k / m
    icc = (phi - 1) / (kbar - 1) if kbar > 1 else float("nan")
    return p, phi, icc


def boot_icc(counts, k_each, n_boot=2000, seed=0):
    rng = random.Random(seed)
    m = len(counts)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        r = dispersion([counts[i] for i in idx], [k_each[i] for i in idx])
        if r:
            out.append(r[2])
    if not out:
        return (float("nan"), float("nan"))
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))])


def simulate_width(icc, p, m, k, n_boot=400, seed=1):
    """
    Bootstrap the ICC confidence width at a hypothetical (m instances, k repeats).

    Instance rates are drawn from a beta matched to (p, icc); each instance then
    gets k Bernoulli draws. This is the power calculation: it says how much data
    is needed before the frailty estimate is precise enough to be worth quoting.
    """
    rng = random.Random(seed)
    if not (0 < p < 1) or icc <= 0:
        return float("nan")
    # beta with mean p and intraclass correlation icc
    nu = (1 - icc) / icc
    a, b = p * nu, (1 - p) * nu
    if a <= 0 or b <= 0:
        return float("nan")
    widths = []
    for _ in range(n_boot):
        counts = []
        for _ in range(m):
            pi = rng.betavariate(a, b)
            counts.append(sum(1 for _ in range(k) if rng.random() < pi))
        r = dispersion(counts, [k] * m)
        if r:
            widths.append(r[2])
    if len(widths) < 30:
        return float("nan")
    widths.sort()
    lo = widths[int(0.025 * len(widths))]
    hi = widths[int(0.975 * len(widths))]
    return hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/pilot.jsonl")
    ap.add_argument("--target-width", type=float, default=0.20,
                    help="acceptable width of the ICC 95%% interval")
    a = ap.parse_args()

    rows = []
    with open(a.inp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        sys.exit(f"no episodes in {a.inp}")

    versions = {(r.get("code", {}).get("commit", "")[:8],
                 bool(r.get("code", {}).get("dirty"))) for r in rows}
    if len(versions) > 1:
        print(f"!! WARNING: {len(versions)} code versions in this file: "
              f"{sorted(versions)}\n   Do not average across a bug fix.\n")

    # grade -> per (model, n, instance) success counts
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = grade_episode(task_for(r), calls_from_steps(r["steps"]))
        key = (r["model"]["name"], r["chain_length"])
        cells[key][r["task_id"]].append(1 if g.walked_canonical else 0)

    print(f"{len(rows)} episodes\n")
    print("FRAILTY: is variance between instances larger than chance?\n")
    print(f"  {'model':<24}{'n':>3}{'inst':>6}{'k':>4}{'p':>7}"
          f"{'phi':>7}{'ICC':>7}   {'ICC 95% CI':<18}verdict")
    print("  " + "-" * 96)

    findings = {}
    for (model, n), inst in sorted(cells.items()):
        counts = [sum(v) for v in inst.values()]
        ks = [len(v) for v in inst.values()]
        r = dispersion(counts, ks)
        if r is None:
            p = sum(counts) / max(1, sum(ks))
            print(f"  {model:<24}{n:>3}{len(inst):>6}{min(ks):>4}{p:>7.2f}"
                  f"{'-':>7}{'-':>7}   {'-':<18}"
                  f"{'at ceiling' if p >= 1 else 'at floor'} -- undefined")
            continue
        p, phi, icc = r
        lo, hi = boot_icc(counts, ks, seed=n)
        if lo > 0:
            verdict = "FRAILTY PRESENT"
        elif hi < 0.05:
            verdict = "instances look alike"
        else:
            verdict = "underpowered"
        print(f"  {model:<24}{n:>3}{len(inst):>6}{min(ks):>4}{p:>7.2f}"
              f"{phi:>7.2f}{icc:>7.3f}   [{lo:>6.3f},{hi:>6.3f}]   {verdict}")
        findings[(model, n)] = (p, icc)

        rates = sorted(c / k for c, k in zip(counts, ks))
        print(f"  {'':<24}   per-instance rates: "
              f"{' '.join(f'{x:.2f}' for x in rates)}")

    if not findings:
        print("\nEvery cell is at ceiling or floor -- the ladder needs adjusting"
              " before a power calculation means anything.")
        return

    # -- power ------------------------------------------------------------
    print(f"\n\nPOWER: instances x repeats needed for an ICC interval "
          f"narrower than {a.target_width}\n")
    ref = max(findings.items(), key=lambda kv: kv[1][1])
    (model, n), (p, icc) = ref
    print(f"  using the strongest observed signal: {model} n={n}  "
          f"p={p:.2f} ICC={icc:.3f}\n")
    print(f"  {'instances':>10}{'repeats':>9}{'episodes':>10}{'ICC width':>12}"
          f"{'hours':>8}")
    print("  " + "-" * 50)
    for m in (12, 20, 30, 40):
        for k in (15, 20, 30):
            w = simulate_width(icc, p, m, k, seed=m * 100 + k)
            eps = m * k * 4          # x4 chain lengths
            hrs = eps * 2.2 / 3600
            mark = "  <- sufficient" if w == w and w <= a.target_width else ""
            print(f"  {m:>10}{k:>9}{eps:>10}{w:>12.3f}{hrs:>8.1f}{mark}")

    print("\n  (episodes counts one model across four chain lengths; "
          "hours at the measured 2.2 s/episode)")


if __name__ == "__main__":
    main()
