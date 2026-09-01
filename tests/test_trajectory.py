"""
Trajectory oracle tests.

The Step B gate. The property that matters most is the on_canonical /
locally_correct distinction: an agent that takes one wrong turn and then follows
its wrong chain perfectly must be OFF canonical but LOCALLY CORRECT thereafter.
Get that wrong and the conditional step-accuracy curve -- the whole basis for
telling self-conditioning apart from independent failure -- is meaningless.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.envs.ledger import make_task, solve  # noqa: E402
from agentfrailty.scorers.trajectory import (  # noqa: E402
    ERR_LATE_CONTINUE, ERR_MALFORMED, ERR_NONEXISTENT, ERR_OFF_PATH,
    ERR_PREMATURE_SUBMIT, ERR_REVISIT, ERR_WRONG_TOOL,
    grade_episode, step_accuracy_by_index,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}")
        FAILURES.append(name)


def get(rid):
    return {"tool": "get_record", "args": {"record_id": rid}, "parse_ok": True}


def sub(total):
    return {"tool": "submit", "args": {"total": total}, "parse_ok": True}


def perfect_calls(t):
    return [get(r) for r in t.canonical_path] + [sub(t.goal_total)]


# -- the perfect run --------------------------------------------------------

def test_perfect_episode():
    t = make_task(seed=1, n=5, n_distractors=4)
    g = grade_episode(t, perfect_calls(t))
    check("perfect: every step locally correct", g.n_locally_correct == g.n_steps)
    check("perfect: every step on canonical", g.n_on_canonical == g.n_steps)
    check("perfect: no first error", g.first_error_index is None)
    check("perfect: outcome correct", g.outcome_correct)
    check("perfect: walked the canonical path", g.walked_canonical)
    check("perfect: reached chain end", g.reached_chain_end)
    check("perfect: observed total == goal", g.observed_total == t.goal_total)
    check("perfect: n+1 steps", g.n_steps == t.n + 1)
    check("perfect: no errors recorded", g.error_counts == {})


# -- THE CENTRAL CASE -------------------------------------------------------

def test_wrong_turn_then_perfect_following():
    """
    One wrong turn, then flawless traversal of the WRONG chain.

    Must be: off canonical from the wrong turn onward, but locally correct at
    every step after it. This is the case the whole design rests on.
    """
    t = make_task(seed=11, n=6, n_distractors=12)

    # Pick the distractor with the LONGEST onward chain. Taking an arbitrary one
    # exercised a single follow-up step, which is far too weak a test of the
    # property this whole design rests on.
    def chain_len(start):
        seen, cur = set(), start
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = t.records[cur].next
        return len(seen)

    off = [r for r in t.records if r not in t.canonical_path]
    distractor = max(off, key=chain_len)
    assert chain_len(distractor) >= 3, "need a distractor chain worth testing"

    calls = [get(t.canonical_path[0]), get(distractor)]
    # Follow the distractor's own chain faithfully.
    cur = t.records[distractor].next
    while cur is not None:
        calls.append(get(cur))
        cur = t.records[cur].next
    calls.append(sub(0))

    g = grade_episode(t, calls)
    steps = g.steps

    check("wrong turn: step 0 locally correct", steps[0].locally_correct)
    check("wrong turn: step 1 flagged off_path",
          steps[1].error_type == ERR_OFF_PATH and not steps[1].locally_correct)
    after = steps[2:-1]
    check(f"wrong turn: all {len(after)} later reads are LOCALLY CORRECT",
          all(s.locally_correct for s in after))
    check("wrong turn: those later reads are NOT on canonical",
          all(not s.on_canonical for s in after))
    check("wrong turn: first_error_index == 1", g.first_error_index == 1)
    check("wrong turn: did not walk canonical", not g.walked_canonical)
    check("wrong turn: position_on_path False after the wrong turn",
          all(not s.position_on_path for s in after))


def test_recovery_is_detected():
    """Wander off, then come back to the canonical chain."""
    t = make_task(seed=13, n=5, n_distractors=6)
    distractor = next(r for r in t.records if r not in t.canonical_path)
    calls = [
        get(t.canonical_path[0]),
        get(distractor),                 # off path
        get(t.canonical_path[1]),        # back on path
        get(t.canonical_path[2]),
    ]
    g = grade_episode(t, calls)
    check("recovery: counted once", g.n_recoveries == 1)
    check("recovery: flagged on the returning step", g.steps[2].recovered)
    check("recovery: step after recovery is locally correct",
          g.steps[3].locally_correct)


# -- error taxonomy ---------------------------------------------------------

def test_nonexistent_id():
    t = make_task(seed=2, n=4)
    g = grade_episode(t, [get(t.canonical_path[0]), get("zzzzz")])
    check("nonexistent id classified", g.steps[1].error_type == ERR_NONEXISTENT)
    check("nonexistent id does not move the agent",
          g.steps[1].position_before == t.canonical_path[0])


def test_revisit():
    t = make_task(seed=3, n=4)
    g = grade_episode(t, [get(t.canonical_path[0]), get(t.canonical_path[0])])
    check("revisit classified", g.steps[1].error_type == ERR_REVISIT)


def test_revisit_not_double_counted():
    t = make_task(seed=3, n=4)
    first = t.records[t.canonical_path[0]].value
    g = grade_episode(t, [get(t.canonical_path[0]), get(t.canonical_path[0])])
    check("revisit counted once in observed_total", g.observed_total == first)


def test_premature_submit():
    t = make_task(seed=4, n=5)
    g = grade_episode(t, [get(t.canonical_path[0]), sub(t.goal_total)])
    check("premature submit classified",
          g.steps[1].error_type == ERR_PREMATURE_SUBMIT)
    check("premature submit with a right-looking total is still wrong",
          not g.steps[1].locally_correct)


def test_late_continue():
    t = make_task(seed=5, n=3)
    calls = [get(r) for r in t.canonical_path] + [get(t.canonical_path[0])]
    g = grade_episode(t, calls)
    check("reading after the chain ended classified",
          g.steps[-1].error_type == ERR_LATE_CONTINUE)


def test_wrong_tool_and_malformed():
    t = make_task(seed=6, n=3)
    g = grade_episode(t, [
        {"tool": "send_email", "args": {}, "parse_ok": True},
        {"tool": None, "args": None, "parse_ok": False},
        {"tool": "get_record", "args": {"wrong_key": "x"}, "parse_ok": True},
    ])
    check("unknown tool classified", g.steps[0].error_type == ERR_WRONG_TOOL)
    check("unparsed step classified", g.steps[1].error_type == ERR_MALFORMED)
    check("missing argument classified", g.steps[2].error_type == ERR_MALFORMED)


# -- arithmetic vs navigation ----------------------------------------------

def test_navigation_right_arithmetic_wrong():
    t = make_task(seed=7, n=4)
    calls = [get(r) for r in t.canonical_path] + [sub(t.goal_total + 1)]
    g = grade_episode(t, calls)
    check("nav/arith: walked canonical", g.walked_canonical)
    check("nav/arith: outcome wrong", not g.outcome_correct)
    check("nav/arith: arithmetic wrong given reads",
          g.arithmetic_correct_given_reads is False)


def test_navigation_wrong_arithmetic_right():
    """Walked the wrong chain but summed what it saw correctly -- arithmetic
    capability intact, navigation capability failed."""
    t = make_task(seed=17, n=4, n_distractors=6)
    d = next(r for r in t.records if r not in t.canonical_path)
    calls = [get(t.canonical_path[0]), get(d)]
    observed = t.records[t.canonical_path[0]].value + t.records[d].value
    cur = t.records[d].next
    while cur is not None:
        calls.append(get(cur))
        observed += t.records[cur].value
        cur = t.records[cur].next
    calls.append(sub(observed))
    g = grade_episode(t, calls)
    check("nav wrong / arith right: outcome wrong", not g.outcome_correct)
    check("nav wrong / arith right: arithmetic correct given reads",
          g.arithmetic_correct_given_reads is True)


def test_stringified_total_matches_env_leniency():
    t = make_task(seed=8, n=3)
    calls = [get(r) for r in t.canonical_path] + [sub(str(t.goal_total))]
    g = grade_episode(t, calls)
    check("scorer coerces '42' exactly as the env does", g.outcome_correct)


# -- aggregate --------------------------------------------------------------

def test_step_accuracy_curve_is_flat_for_perfect_agents():
    ts = [make_task(seed=s, n=6, n_distractors=5) for s in range(20)]
    grades = [grade_episode(t, perfect_calls(t)) for t in ts]
    curve = step_accuracy_by_index(grades)
    check("perfect agents give a flat curve at 1.0",
          all(abs(v - 1.0) < 1e-9 for v in curve.values()))
    check("curve covers n+1 indices", len(curve) == 7)


def test_grades_against_real_env_log():
    """Grade the env's OWN call log, so the two agree on format."""
    from agentfrailty.envs.ledger import LedgerEnv
    t = make_task(seed=9, n=4, n_distractors=3)
    env = LedgerEnv(t)
    path, total = solve(t)
    calls = [{"tool": c["tool"], "args": c["args"], "parse_ok": True}
             for c in LedgerEnv(t).calls]
    env2 = LedgerEnv(t)
    for rid in path:
        env2.call("get_record", {"record_id": rid})
    env2.call("submit", {"total": total})
    calls = [{"tool": c["tool"], "args": c["args"], "parse_ok": True}
             for c in env2.calls]
    g = grade_episode(t, calls)
    check("env log grades as a perfect episode",
          g.outcome_correct and g.n_locally_correct == g.n_steps)
    _ = calls


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} oracle test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
