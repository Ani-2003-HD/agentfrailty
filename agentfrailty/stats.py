"""
Shared statistics.

Lifted out of scripts/power.py once a second consumer appeared. The frailty
estimate is the project's central statistic; it belongs in the package with the
rest of the tested code, not in a script.

Stdlib only, so the analysis runs without installing extras.
"""

from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson interval for a proportion.

    Used rather than the normal approximation because most cells here sit near
    0 or 1, where the normal approximation produces impossible bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def dispersion(counts, k_each):
    """
    Pearson dispersion phi and the implied intraclass correlation.

    For m groups, each run k times, with y_i successes:

        phi = (1/(m-1)) * sum_i (y_i - k*p)^2 / (k*p*(1-p))

    Under a pure binomial model -- every group identical -- E[phi] = 1. phi > 1
    means the groups genuinely differ. The ICC,

        ICC = (phi - 1) / (kbar - 1)

    is the share of variance attributable to WHICH GROUP was drawn. That is the
    frailty term: exactly the quantity 2603.29231 never estimates before
    attributing its super-linear decay entirely to within-run dependence.

    Returns (p, phi, icc), or None when undefined (p at 0 or 1, or m < 2).
    """
    m = len(counts)
    if m < 2:
        return None
    total_k = sum(k_each)
    if total_k <= 0:
        return None
    p = sum(counts) / total_k
    if p <= 0 or p >= 1:
        return None
    chi = sum((y - k * p) ** 2 / (k * p * (1 - p))
              for y, k in zip(counts, k_each))
    phi = chi / (m - 1)
    kbar = total_k / m
    icc = (phi - 1) / (kbar - 1) if kbar > 1 else float("nan")
    return p, phi, icc


def boot_icc(counts, k_each, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Bootstrap CI for the ICC, resampling groups. phi is noisy at small m."""
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


def simulate_width(icc: float, p: float, m: int, k: int,
                   n_boot: int = 400, seed: int = 1) -> float:
    """
    Width of the ICC 95% interval at a hypothetical (m groups, k repeats).

    Group rates are drawn from a beta matched to (p, icc); each group then gets
    k Bernoulli draws. This is the power calculation -- how much data before the
    frailty estimate is precise enough to quote.
    """
    rng = random.Random(seed)
    if not (0 < p < 1) or icc <= 0:
        return float("nan")
    nu = (1 - icc) / icc
    a, b = p * nu, (1 - p) * nu
    if a <= 0 or b <= 0:
        return float("nan")
    widths = []
    for _ in range(n_boot):
        counts = []
        for _ in range(m):
            pi = rng.betavariate(a, b)       # once per group, not per trial
            counts.append(sum(1 for _ in range(k) if rng.random() < pi))
        r = dispersion(counts, [k] * m)
        if r:
            widths.append(r[2])
    if len(widths) < 30:
        return float("nan")
    widths.sort()
    return widths[int(0.975 * len(widths))] - widths[int(0.025 * len(widths))]


def pearson_r(xs, ys) -> float:
    """Correlation, for relating word-set features to difficulty."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)
