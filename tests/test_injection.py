"""
Injection tests -- the Step F gate.

Two properties carry the entire experiment:

  CONTEXT LENGTH HELD ~FIXED ACROSS ERROR RATES. The design varies content while
  holding length constant. If a 100%-error history were systematically longer or
  shorter than a 0% one, the measured effect would be long-context degradation
  in disguise -- the exact confound this control exists to remove.

  HISTORIES MUST BE COHERENT. Every observation must be the TRUE data for the
  record actually called. A history whose observations contradict its calls
  tests contradiction-resolution, not execution -- the objection 2509.09677's
  Appendix I raises against naive CoT injection.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import render_transcript  # noqa: E402
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.injection import (  # noqa: E402
    LOCATION_EARLY, LOCATION_LATE, InjectionSpec, build_history, grade_probe,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}")
        FAILURES.append(name)


TASK = make_task(seed=42, n=14, n_distractors=25)
RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def hist(rate, seed, steps=8, loc="uniform"):
    return build_history(TASK, InjectionSpec(error_rate=rate, prefix_steps=steps,
                                             location=loc, seed=seed))


def test_context_length_is_stable_across_error_rates():
    lens = {r: sum(hist(r, s).n_chars for s in range(20)) / 20 for r in RATES}
    lo, hi = min(lens.values()), max(lens.values())
    drift = (hi - lo) / lo
    check(f"context length varies <5% across rates (drift={drift:.3%})",
          drift < 0.05)
    steps = {r: {hist(r, s).steps_built for s in range(20)} for r in RATES}
    check(f"every arm builds exactly 8 steps ({steps[1.0]})",
          all(v == {8} for v in steps.values()))


def test_history_is_coherent():
    ok, detail = True, ""
    for r in RATES:
        for s in range(15):
            for t in hist(r, s).turns:
                call = json.loads(t.assistant)
                obs = json.loads(t.observation)
                rid = call["arguments"]["record_id"]
                rec = TASK.records[rid]
                if not (obs["id"] == rid and obs["value"] == rec.value
                        and obs["next"] == rec.next):
                    ok, detail = False, f"rate={r} seed={s} rid={rid}"
    check(f"observations always match the record called ({detail})", ok)


def test_expected_next_follows_from_final_position():
    ok = True
    for r in RATES:
        for s in range(15):
            h = hist(r, s)
            if h.final_position is None:
                continue
            true_next = TASK.records[h.final_position].next
            if h.chain_ended:
                ok &= (true_next is None and h.expected_tool == "submit")
            else:
                ok &= (h.expected_target == true_next)
    check("expected next action follows the final position", ok)


def test_error_rate_is_honoured():
    for r in RATES:
        counts = [len(hist(r, s).injected_indices) for s in range(20)]
        mean = sum(counts) / len(counts)
        want = min(round(r * 8), 7)   # step 0 is never injected
        check(f"rate {r:.2f}: {mean:.1f} injected (want ~{want})",
              abs(mean - want) <= 0.6)


def test_zero_rate_is_the_canonical_path():
    h = hist(0.0, 3, steps=6)
    called = [json.loads(t.assistant)["arguments"]["record_id"] for t in h.turns]
    check("0% history walks the canonical path", called == TASK.canonical_path[:6])
    check("0% history injects nothing", h.injected_indices == [])


def test_step_zero_is_never_injected():
    check("step 0 is never injected",
          all(0 not in hist(r, s).injected_indices
              for r in RATES for s in range(20)))


def test_location_policies_differ():
    e = hist(0.5, 1, loc=LOCATION_EARLY)
    l = hist(0.5, 1, loc=LOCATION_LATE)
    check(f"early clusters at the start {e.injected_indices}",
          max(e.injected_indices) < 5)
    check(f"late clusters at the end {l.injected_indices}",
          min(l.injected_indices) >= 4)
    check("early and late differ", set(e.injected_indices) != set(l.injected_indices))
    check("same length either way", e.steps_built == l.steps_built)


def test_determinism():
    a, b, c = hist(0.5, 9), hist(0.5, 9), hist(0.5, 10)
    check("same spec reproduces the same history",
          [t.assistant for t in a.turns] == [t.assistant for t in b.turns])
    check("different seed differs",
          [t.assistant for t in a.turns] != [t.assistant for t in c.turns])


def test_injected_transcript_renders_like_a_real_one():
    h = hist(0.5, 2, steps=5)
    text = render_transcript("SYS", h.turns)
    check("renders through the same function", text.startswith("SYS"))
    check("one assistant turn per step plus the cue",
          text.count("\nAssistant:") == 6)
    check("one tool result per step", text.count("\nTool result:") == 5)
    check("ends on the assistant cue", text.rstrip().endswith("Assistant:"))


def test_probe_is_always_a_navigation_decision():
    """
    Every arm's probe must ask the SAME KIND of question.

    A history that dead-ends makes the probe "submit"; one that does not makes
    it "which record next". If high-error arms dead-ended more often, the arms
    would be answering different questions and the comparison would be void.
    """
    bad = [(r, s) for r in RATES for s in range(30)
           if hist(r, s).expected_tool != "get_record"]
    check(f"probe is get_record in every arm (offenders: {bad[:3]})", not bad)
    check("probe always has a concrete target",
          all(hist(r, s).expected_target is not None
              for r in RATES for s in range(30)))


def test_grade_probe():
    h = hist(0.75, 4, steps=6)
    check("correct probe", grade_probe(h, "get_record",
          {"record_id": h.expected_target}) == "correct")
    check("whitespace tolerated", grade_probe(h, "get_record",
          {"record_id": f" {h.expected_target} "}) == "correct")
    check("wrong target", grade_probe(h, "get_record",
          {"record_id": "zzzzz"}) == "wrong_target")
    check("wrong tool", grade_probe(h, "submit", {"total": 0}) == "wrong_tool")
    check("malformed", grade_probe(h, None, None) == "malformed")


def test_probe_is_conditional_not_canonical():
    """A high-error history leaves the agent off-path. The probe is graded
    against THAT position -- grading against the canonical path would score the
    injected history rather than the model."""
    h = hist(1.0, 5, steps=6)
    off = h.final_position not in TASK.canonical_path
    check("100% history leaves the agent off the canonical path", off)
    if off:
        check("expected target is not the canonical next",
              h.expected_target not in TASK.canonical_path)
        check("following the wrong chain scores correct",
              grade_probe(h, "get_record",
                          {"record_id": h.expected_target}) == "correct")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} injection test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
