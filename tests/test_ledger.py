"""
Ledger environment tests.

These are the Step A gate. Two properties matter more than the rest and the
whole design collapses without them:

  * N is REQUIRED, not observed -- the task cannot be solved in fewer than n
    get_record calls, and the id of position i+1 is not learnable before
    reading position i.
  * The environment NEVER raises, for any input. An env crash mid-episode is
    indistinguishable from a model failure in the data, which would silently
    corrupt every number downstream.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.envs.ledger import (  # noqa: E402
    WORDS, LedgerEnv, LedgerTask, make_task, solve,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}")
        FAILURES.append(name)


# -- vocabulary -------------------------------------------------------------

def test_vocabulary():
    bad = [w for w in WORDS if len(w) != 5]
    check(f"all ids are five letters (offenders: {bad[:5]})", not bad)
    check("ids are unique", len(WORDS) == len(set(WORDS)))
    check("vocabulary is large enough for long chains", len(WORDS) >= 300)


# -- N is required ----------------------------------------------------------

def test_canonical_path_length():
    for n in (1, 2, 3, 5, 8, 12, 20):
        t = make_task(seed=1, n=n, n_distractors=5)
        check(f"n={n}: canonical path has exactly n ids",
              len(t.canonical_path) == n)


def test_solve_uses_exactly_n_calls():
    for n in (1, 3, 8, 20):
        t = make_task(seed=7, n=n, n_distractors=10)
        path, total = solve(t)
        check(f"n={n}: solve walks n records", len(path) == n)
        check(f"n={n}: solve recovers goal_total", total == t.goal_total)
        check(f"n={n}: required_calls == n+1", t.required_calls == n + 1)


def test_no_shortcut():
    """
    The core claim. Position i+1's id must not be learnable before reading
    position i -- not from the prompt, and not from any record other than i.
    """
    t = make_task(seed=11, n=6, n_distractors=8)
    obs = LedgerEnv(t).initial_observation()

    later = [rid for rid in t.canonical_path[1:] if rid in obs]
    check(f"only the start id appears in the prompt (leaked: {later})", not later)

    ok = True
    for i in range(len(t.canonical_path) - 1):
        nxt = t.canonical_path[i + 1]
        # Which records reveal `nxt`? Only position i may.
        revealers = [rid for rid, rec in t.records.items() if rec.next == nxt]
        if revealers != [t.canonical_path[i]]:
            ok = False
    check("each path id is revealed by exactly one predecessor", ok)


def test_distractors_never_rejoin_the_path():
    """A distractor pointing into the chain would let a lost agent finish by
    accident, destroying the per-step oracle."""
    ok = True
    for seed in range(50):
        t = make_task(seed=seed, n=6, n_distractors=12)
        path = set(t.canonical_path)
        for rid, rec in t.records.items():
            if rid not in path and rec.next in path:
                ok = False
    check("no distractor points into the canonical path", ok)


def test_path_terminates():
    t = make_task(seed=3, n=7, n_distractors=4)
    last = t.records[t.canonical_path[-1]]
    check("last record terminates the chain", last.next is None)


# -- determinism ------------------------------------------------------------

def test_determinism():
    a = make_task(seed=42, n=8, n_distractors=10)
    b = make_task(seed=42, n=8, n_distractors=10)
    check("same seed reproduces the task exactly", a.to_dict() == b.to_dict())

    c = make_task(seed=43, n=8, n_distractors=10)
    check("different seed gives a different task", a.to_dict() != c.to_dict())


def test_generation_ignores_global_random_state():
    random.seed(0)
    a = make_task(seed=5, n=5, n_distractors=5)
    random.seed(999)
    [random.random() for _ in range(100)]
    b = make_task(seed=5, n=5, n_distractors=5)
    check("global random state does not affect generation", a.to_dict() == b.to_dict())


# -- the environment never fails -------------------------------------------

def test_unknown_id_is_an_agent_error_not_an_env_error():
    t = make_task(seed=2, n=3)
    env = LedgerEnv(t)
    r = env.call("get_record", {"record_id": "zzzzz"})
    check("unknown id returns ok=True", r.ok)
    check("unknown id reports an agent-facing error", "error" in r.payload)
    check("unknown id records no env_error", not env.state()["env_errors"])


def test_env_never_raises_on_garbage():
    t = make_task(seed=4, n=3, n_distractors=3)
    env = LedgerEnv(t)
    garbage = [
        ("get_record", None), ("get_record", []), ("get_record", {}),
        ("get_record", {"record_id": None}), ("get_record", {"record_id": 42}),
        ("get_record", {"record_id": ["about"]}), ("get_record", {"wrong": "key"}),
        ("submit", {"total": "not a number"}), ("submit", {"total": None}),
        ("submit", {"total": True}), ("submit", {}), ("submit", {"total": [1]}),
        ("nonexistent_tool", {"x": 1}), (None, None), (42, {"a": 1}),
        ("", ""), ("get_record", "record_id=about"),
    ]
    raised = None
    for name, args in garbage:
        try:
            env.call(name, args)
        except Exception as e:
            raised = f"{name!r}/{args!r} -> {e!r}"
            break
    check(f"no input raises (first raise: {raised})", raised is None)
    check("garbage produced no env_errors", not env.state()["env_errors"])


def test_submit_accepts_stringified_int():
    """Small models emit "42" rather than 42 constantly. Accepting it measures
    execution rather than JSON typing discipline."""
    t = make_task(seed=6, n=2)
    env = LedgerEnv(t)
    env.call("submit", {"total": " -17 "})
    check("stringified total is coerced", env.submitted == -17)


def test_bool_is_not_a_total():
    t = make_task(seed=6, n=2)
    env = LedgerEnv(t)
    env.call("submit", {"total": True})
    check("True is rejected as a total", env.submitted is None)


def test_calls_are_logged_in_order():
    t = make_task(seed=8, n=3)
    env = LedgerEnv(t)
    env.call("get_record", {"record_id": t.start_id})
    env.call("get_record", {"record_id": "zzzzz"})
    env.call("submit", {"total": 0})
    st = env.state()
    check("all calls logged", st["n_calls"] == 3)
    check("get_record args recorded in order",
          st["get_record_calls"] == [t.start_id, "zzzzz"])
    check("finished flag set by submit", st["finished"] is True)


# -- fuzz -------------------------------------------------------------------

def test_fuzz_many_configurations():
    ok = True
    detail = ""
    for seed in range(200):
        n = 1 + seed % 20
        d = seed % 15
        try:
            t = make_task(seed=seed, n=n, n_distractors=d)
            path, total = solve(t)
            if len(path) != n or total != t.goal_total:
                ok, detail = False, f"seed={seed} n={n} d={d}"
                break
        except Exception as e:
            ok, detail = False, f"seed={seed} n={n} d={d} raised {e!r}"
            break
    check(f"200 generated tasks all solve cleanly ({detail})", ok)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} ledger test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
