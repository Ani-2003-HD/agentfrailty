#!/usr/bin/env python3
"""
Regenerate every published figure from the raw results.

Nothing derived is stored, so these are computed from raw rows on every run --
the same discipline as the analysis. Each figure is emitted light and dark.

    python3 scripts/plot.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentfrailty import theme  # noqa: E402
from agentfrailty.agent import calls_from_steps  # noqa: E402
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402
from agentfrailty.stats import dispersion, wilson  # noqa: E402

RES, FIG = "results", "figures"
RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def probe_rates(rows, location=None):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        if location and r.get("location") != location:
            continue
        c[r["error_rate"]][1] += 1
        if r["verdict"] == "correct":
            c[r["error_rate"]][0] += 1
    return c


# ---------------------------------------------------------------- figure 1
def fig_self_conditioning(t):
    files = [("qwen2.5 0.5B", "injection_05b.jsonl"),
             ("qwen2.5 1.5B", "injection.jsonl"),
             ("qwen2.5 3B", "injection_3b.jsonl")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ends = []
    theme.apply(
        fig, ax, t,
        title="Agents condition on their own mistakes",
        subtitle="Context length held to 0.4%. Only the correctness of the "
                 "history varies.",
        xlabel="fraction of the 8-step history that is wrong",
        ylabel="next step correct")

    for i, (label, fname) in enumerate(files):
        rows = [r for r in load(fname) if r.get("location", "uniform") == "uniform"]
        if not rows:
            continue
        c = probe_rates(rows)
        xs = [r for r in RATES if c[r][1]]
        ys = [c[r][0] / c[r][1] for r in xs]
        col = t["series"][i]
        los = [wilson(c[r][0], c[r][1])[0] for r in xs]
        his = [wilson(c[r][0], c[r][1])[1] for r in xs]
        ax.fill_between(xs, los, his, color=col, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=t["surface"], markeredgewidth=1.6,
                label=label, zorder=3)
        ends.append((xs[-1], ys[-1], label))

    ax.set_ylim(0, 1.02)
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{int(r*100)}%" for r in RATES])
    ax.set_xlim(-0.03, 1.30)
    theme.direct_labels(ax, ends, t)
    theme.legend(ax, t)
    return theme.save(fig, os.path.join(FIG, "self-conditioning"), t)


# ---------------------------------------------------------------- figure 2
def fig_recency(t):
    rows = load("injection_location.jsonl")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ends = []
    theme.apply(
        fig, ax, t,
        title="Recent mistakes matter; old ones barely register",
        subtitle="Same error rate, same history length — only WHERE the "
                 "errors sit differs.",
        xlabel="fraction of the history that is wrong",
        ylabel="next step correct")

    for i, loc in enumerate(("early", "late")):
        c = probe_rates(rows, location=loc)
        xs = [r for r in RATES if c[r][1]]
        if not xs:
            continue
        ys = [c[r][0] / c[r][1] for r in xs]
        col = t["series"][i]
        ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=t["surface"], markeredgewidth=1.6,
                label=f"errors clustered {loc}", zorder=3)
        ends.append((xs[-1], ys[-1], loc))

    ce, cl = probe_rates(rows, "early"), probe_rates(rows, "late")
    if ce[0.25][1] and cl[0.25][1]:
        ye = ce[0.25][0] / ce[0.25][1]
        yl = cl[0.25][0] / cl[0.25][1]
        ax.annotate("", xy=(0.25, ye), xytext=(0.25, yl),
                    arrowprops=dict(arrowstyle="<->", color=t["muted"], lw=1.2))
        ax.text(0.27, (ye + yl) / 2, f"{abs(ye - yl) * 100:.0f} points",
                color=t["ink2"], fontsize=9, va="center")

    ax.set_ylim(0, 1.02)
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{int(r*100)}%" for r in RATES])
    ax.set_xlim(-0.03, 1.20)
    theme.direct_labels(ax, ends, t)
    theme.legend(ax, t)
    return theme.save(fig, os.path.join(FIG, "recency"), t)


# ---------------------------------------------------------------- figure 3
def fig_lexical(t):
    rows = load("wordsweep.jsonl")
    if not rows:
        return None
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["ids_seed"]][1] += 1
        if r["walked_canonical"]:
            per[r["ids_seed"]][0] += 1
    rates = sorted(v[0] / v[1] for v in per.values())
    d = dispersion([v[0] for v in per.values()], [v[1] for v in per.values()])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    theme.apply(
        fig, ax, t,
        title="The same task, re-worded, is either trivial or impossible",
        subtitle="One instance. Identical graph and identical numbers. Only the "
                 "six words change.",
        xlabel="success rate of a word set", ylabel="number of word sets")

    col = t["series"][0]
    ax.hist(rates, bins=[i / 10 for i in range(11)], color=col,
            edgecolor=t["surface"], linewidth=2, zorder=3)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xticklabels(["0", "", ".2", "", ".4", "", ".6", "", ".8", "", "1"])
    # counts are integers; matplotlib's default 17.5 / 12.5 ticks are nonsense
    # on a "number of word sets" axis
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if d:
        ax.text(0.5, 0.94, f"ICC = {d[2]:.2f}", transform=ax.transAxes,
                color=t["ink"], fontsize=11, ha="center", fontweight="600")
        ax.text(0.5, 0.87,
                "the vocabulary alone carries 72% of the variance",
                transform=ax.transAxes, color=t["ink2"], fontsize=9,
                ha="center")
    return theme.save(fig, os.path.join(FIG, "lexical-frailty"), t)


# ---------------------------------------------------------------- figure 4
def fig_capabilities(t):
    rows = [r for r in load("calibrate.jsonl")
            if r["model"]["name"] == "qwen2.5:1.5b-instruct"]
    if not rows:
        return None
    nav = defaultdict(lambda: [0, 0])
    ari = defaultdict(lambda: [0, 0])
    for r in rows:
        p = r["task_id"].split("-")
        task = make_task(seed=int(p[4][1:]), n=int(p[1][1:]),
                         n_distractors=int(p[2][1:]))
        g = grade_episode(task, calls_from_steps(r["steps"]))
        n = r["chain_length"]
        nav[n][1] += 1
        nav[n][0] += 1 if g.walked_canonical else 0
        if g.arithmetic_correct_given_reads is not None:
            ari[n][1] += 1
            ari[n][0] += 1 if g.arithmetic_correct_given_reads else 0

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ends = []
    theme.apply(
        fig, ax, t,
        # Titled to what the intervals actually support. The navigation cliff
        # is solid (0.93 [0.70,0.99] at n=5 -> 0.13 [0.04,0.38] at n=8, no
        # overlap). The arithmetic curve is noise-dominated at 3 instances per
        # cell, so the figure must not imply a clean second breaking point.
        title="Navigation holds — then falls off a cliff",
        subtitle="qwen2.5-1.5B. Arithmetic erodes earlier, but at this sample "
                 "size that curve is noise-dominated.",
        xlabel="chain length (required tool calls)", ylabel="success rate")

    for i, (label, data) in enumerate((("navigation", nav), ("arithmetic", ari))):
        xs = sorted(k for k in data if data[k][1])
        ys = [data[k][0] / data[k][1] for k in xs]
        col = t["series"][i]
        # Confidence bands are not decoration here. Only 3 task instances per
        # cell, and instance heterogeneity is large (ICC 0.78 at n=6) -- so the
        # arithmetic curve appears to RISE from n=3 to n=5, which is not a
        # chain-length effect. Drawing bare lines would imply a real
        # non-monotonicity we know to be noise.
        los = [wilson(data[k][0], data[k][1])[0] for k in xs]
        his = [wilson(data[k][0], data[k][1])[1] for k in xs]
        ax.fill_between(xs, los, his, color=col, alpha=0.13, linewidth=0,
                        zorder=2)
        ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=t["surface"], markeredgewidth=1.6,
                label=label, zorder=3)
        ends.append((xs[-1], ys[-1], label))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, 14.2)
    ax.text(0.5, -0.20,
            "15 episodes per point, 3 task instances per cell — the bands are "
            "wide because instance\ndifficulty varies far more than chain "
            "length does at this sample size.",
            transform=ax.transAxes, ha="center", va="top",
            color=t["muted"], fontsize=8.5)
    theme.direct_labels(ax, ends, t)
    theme.legend(ax, t)
    return theme.save(fig, os.path.join(FIG, "two-capabilities"), t)


def main():
    os.makedirs(FIG, exist_ok=True)
    made = []
    for t in theme.MODES:
        for fn in (fig_self_conditioning, fig_recency, fig_lexical,
                   fig_capabilities):
            try:
                out = fn(t)
                if out:
                    made.append(out)
            except Exception as e:      # a missing input must not kill the rest
                print(f"  !! {fn.__name__} ({t['mode']}): {type(e).__name__}: {e}")
    for m in sorted(made):
        print("  wrote", m)
    print(f"\n{len(made)} figures")


if __name__ == "__main__":
    main()
