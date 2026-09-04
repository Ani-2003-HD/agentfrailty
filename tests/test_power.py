"""
Dispersion / ICC tests.

The frailty estimate is the project's central statistic, so it must be verified
against data whose answer we already know:

  * data generated from ONE binomial rate must give ICC ~ 0
  * data generated from instances with DIFFERENT rates must give ICC > 0

Without this the statistic could be quietly wrong and every conclusion with it.
"""

import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

spec = importlib.util.spec_from_file_location(
    "power", os.path.join(HERE, "..", "scripts", "power.py"))
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)

FAILURES = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


def test_homogeneous_gives_zero_icc():
    rng = random.Random(0)
    counts = [sum(1 for _ in range(20) if rng.random() < 0.6) for _ in range(20)]
    p, phi, icc = power.dispersion(counts, [20] * 20)
    lo, hi = power.boot_icc(counts, [20] * 20, seed=1)
    check(f"one true rate -> phi~1 (phi={phi:.2f})", 0.5 < phi < 1.8)
    check(f"one true rate -> ICC~0 (ICC={icc:.3f})", abs(icc) < 0.05)
    check(f"one true rate -> CI includes 0 [{lo:.3f},{hi:.3f}]", lo <= 0 <= hi)


def test_heterogeneous_is_detected():
    rng = random.Random(0)
    # The instance rate is drawn ONCE PER INSTANCE. Drawing it per trial (the
    # first version of this test) averages the mixture away and produces
    # homogeneous data -- the test then "fails" against a correct estimator.
    counts = []
    for _ in range(20):
        pi = rng.choice([0.25, 0.85])
        counts.append(sum(1 for _ in range(20) if rng.random() < pi))
    p, phi, icc = power.dispersion(counts, [20] * 20)
    lo, hi = power.boot_icc(counts, [20] * 20, seed=2)
    check(f"mixed rates -> phi >> 1 (phi={phi:.2f})", phi > 3)
    check(f"mixed rates -> ICC clearly positive (ICC={icc:.3f})", icc > 0.15)
    check(f"mixed rates -> CI excludes 0 [{lo:.3f},{hi:.3f}]", lo > 0)


def test_undefined_at_ceiling_and_floor():
    check("all-success is undefined", power.dispersion([20] * 10, [20] * 10) is None)
    check("all-failure is undefined", power.dispersion([0] * 10, [20] * 10) is None)
    check("single instance is undefined", power.dispersion([5], [20]) is None)


def test_more_data_narrows_the_interval():
    w_small = power.simulate_width(0.3, 0.6, m=12, k=15, seed=3)
    w_large = power.simulate_width(0.3, 0.6, m=40, k=30, seed=3)
    check(f"more instances+repeats narrows the ICC CI "
          f"({w_small:.3f} -> {w_large:.3f})", w_large < w_small)


def test_icc_recovered_from_known_beta():
    """Generate with a known ICC; the estimator should land near it."""
    for true_icc in (0.10, 0.30):
        rng = random.Random(7)
        nu = (1 - true_icc) / true_icc
        a, b = 0.6 * nu, 0.4 * nu
        counts = []
        for _ in range(60):
            pi = rng.betavariate(a, b)     # once per instance, not per trial
            counts.append(sum(1 for _ in range(30) if rng.random() < pi))
        _, _, icc = power.dispersion(counts, [30] * 60)
        check(f"ICC recovered for true={true_icc:.2f} (got {icc:.3f})",
              abs(icc - true_icc) < 0.10)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} dispersion test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
