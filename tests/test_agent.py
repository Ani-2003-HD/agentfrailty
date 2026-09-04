"""
Agent loop tests, driven by a scripted fake runtime.

No model is involved: the runtime replays a fixed list of strings. That makes
every case deterministic and lets us test the loop's behaviour on outputs a real
model produces rarely (malformed JSON, hallucinated tools, runtime failures)
without waiting for one to happen.

The most important test here is `test_transcript_carries_prior_errors`. Self-
conditioning can only be measured if the model actually SEES its own mistakes on
the next turn. If the loop ever silently dropped or summarised history, the
study would measure nothing and look fine doing it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.agent import (  # noqa: E402
    AgentConfig, Turn, calls_from_steps, max_steps_for, render_transcript,
    run_episode,
)
from agentfrailty.envs.ledger import make_task  # noqa: E402
from agentfrailty.schema import GenResult, ModelSpec  # noqa: E402
from agentfrailty.scorers.trajectory import grade_episode  # noqa: E402

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}")
        FAILURES.append(name)


class FakeRuntime:
    """Replays scripted outputs and records every prompt it was given."""

    def __init__(self, outputs, error_at=None):
        self.outputs = list(outputs)
        self.error_at = error_at
        self.prompts = []
        self.calls = 0

    def generate(self, prompt, max_tokens=128, temperature=0.0, seed=0):
        self.prompts.append(prompt)
        i = self.calls
        self.calls += 1
        if self.error_at is not None and i == self.error_at:
            return GenResult(text="", error="ConnectionError('boom')")
        text = self.outputs[i] if i < len(self.outputs) else "{}"
        return GenResult(text=text, prompt_tokens=100 + i, completion_tokens=12,
                         ttft_s=0.01, total_s=0.1)


def call_json(rid):
    return '{"name": "get_record", "arguments": {"record_id": "%s"}}' % rid


def submit_json(total):
    return '{"name": "submit", "arguments": {"total": %d}}' % total


MODEL = ModelSpec(name="fake", quant="none", runtime="fake")


def perfect_script(t):
    return [call_json(r) for r in t.canonical_path] + [submit_json(t.goal_total)]


# -- happy path -------------------------------------------------------------

def test_perfect_episode():
    t = make_task(seed=1, n=4, n_distractors=3)
    rt = FakeRuntime(perfect_script(t))
    row = run_episode(t, rt, MODEL, AgentConfig(), repeat=0)

    check("perfect: terminated by the agent", row.termination == "agent_stopped")
    check("perfect: n+1 steps taken", row.n_steps_taken == t.n + 1)
    check("perfect: submitted the goal total",
          row.final_state["submitted"] == t.goal_total)
    check("perfect: no env errors", not row.final_state["env_errors"])
    check("perfect: every step parsed", all(s["parse_ok"] for s in row.steps))
    check("perfect: chain_length recorded", row.chain_length == t.n)

    g = grade_episode(t, calls_from_steps(row.steps))
    check("perfect: oracle agrees it is perfect",
          g.outcome_correct and g.n_locally_correct == g.n_steps)


# -- THE MECHANISM PRECONDITION --------------------------------------------

def test_transcript_carries_prior_errors():
    """
    The model must see its own mistakes on later turns, or self-conditioning
    cannot occur and the study measures nothing.
    """
    t = make_task(seed=5, n=4, n_distractors=4)
    bad = "not json at all"
    script = [call_json(t.canonical_path[0]), bad,
              call_json("zzzzz"), call_json(t.canonical_path[1])]
    rt = FakeRuntime(script)
    run_episode(t, rt, MODEL, AgentConfig())

    check("prompts grow monotonically",
          all(len(rt.prompts[i]) < len(rt.prompts[i + 1])
              for i in range(len(rt.prompts) - 1)))
    check("the malformed reply appears in every later prompt",
          all(bad in p for p in rt.prompts[2:]))
    check("the hallucinated id appears in the following prompt",
          "zzzzz" in rt.prompts[3])
    check("the error the env returned appears in the following prompt",
          "no record with id" in rt.prompts[3])


def test_hallucinated_continuation_never_enters_the_transcript():
    """
    REGRESSION TEST for the bug the first smoke run exposed.

    Small models keep writing past their own turn, fabricating "Tool result:"
    lines and continuing the conversation. Appending that raw text put invented
    observations into context, so values the model later "read" were its own
    hallucinations -- silently corrupting every arithmetic number in the study.

    Only the parsed JSON span may enter the transcript. The full text stays in
    raw_output for auditing.
    """
    t = make_task(seed=21, n=3, n_distractors=3)
    fabricated = (
        call_json(t.canonical_path[0])
        + ' Tool result: {"id": "GHOST", "value": 9999, "next": "PHANTOM"}'
        + ' Assistant: {"name": "get_record", "arguments": {"record_id": "PHANTOM"}}'
    )
    script = [fabricated] + [call_json(r) for r in t.canonical_path[1:]] \
        + [submit_json(t.goal_total)]
    rt = FakeRuntime(script)
    row = run_episode(t, rt, MODEL, AgentConfig())

    later = rt.prompts[1:]
    check("fabricated tool result never reaches a later prompt",
          all("GHOST" not in p and "9999" not in p for p in later))
    check("fabricated follow-up call never reaches a later prompt",
          all("PHANTOM" not in p for p in later))
    check("the real call was still executed",
          row.steps[0]["tool_args"]["record_id"] == t.canonical_path[0])
    check("raw_output preserves the full text for auditing",
          "GHOST" in row.steps[0]["raw_output"])
    check("the real observation IS in the later prompt",
          str(t.records[t.canonical_path[0]].value) in later[0])


def test_stop_sequences_cover_the_transcript_delimiters():
    from agentfrailty.agent import STOP_SEQUENCES, render_transcript
    turns = [Turn(assistant="a", observation="o")]
    text = render_transcript("SYS", turns)
    check("every delimiter the transcript emits has a stop sequence",
          all(any(d.strip() in text for d in [seq]) for seq in ["Tool result:"])
          and "\nAssistant:" in text)
    check("STOP_SEQUENCES is non-empty", len(STOP_SEQUENCES) >= 2)


def test_prefix_turns_are_used():
    """The hook Step F's error injection depends on."""
    t = make_task(seed=6, n=3)
    marker = '{"name": "get_record", "arguments": {"record_id": "INJECTED"}}'
    prefix = [Turn(assistant=marker, observation='{"error": "planted"}')]
    rt = FakeRuntime(perfect_script(t))
    run_episode(t, rt, MODEL, AgentConfig(), prefix_turns=prefix)
    check("injected history reaches the first prompt", marker in rt.prompts[0])
    check("injected observation reaches the first prompt",
          "planted" in rt.prompts[0])


def test_prefix_turns_are_not_mutated():
    t = make_task(seed=6, n=3)
    prefix = [Turn(assistant="x", observation="y")]
    rt = FakeRuntime(perfect_script(t))
    run_episode(t, rt, MODEL, AgentConfig(), prefix_turns=prefix)
    check("caller's prefix list is not mutated", len(prefix) == 1)


# -- termination reasons ----------------------------------------------------

def test_step_cap():
    t = make_task(seed=2, n=5, n_distractors=3)
    rt = FakeRuntime([call_json(t.canonical_path[0])] * 50)   # revisits forever
    row = run_episode(t, rt, MODEL, AgentConfig(max_steps=6))
    check("step cap: terminated as step_cap", row.termination == "step_cap")
    check("step cap: took exactly the budget", row.n_steps_taken == 6)


def test_runtime_error_is_its_own_termination():
    t = make_task(seed=3, n=4)
    rt = FakeRuntime(perfect_script(t), error_at=2)
    row = run_episode(t, rt, MODEL, AgentConfig())
    check("runtime error: own termination reason",
          row.termination == "runtime_error")
    check("runtime error: recorded on the step",
          "ConnectionError" in row.steps[-1]["error"])
    check("runtime error: not confused with step_cap",
          row.n_steps_taken < row.max_steps)


def test_malformed_output_still_consumes_a_step():
    t = make_task(seed=4, n=3)
    rt = FakeRuntime(["I think I should look at the first record."]
                     + perfect_script(t))
    row = run_episode(t, rt, MODEL, AgentConfig())
    check("malformed: parse_ok False on step 0", row.steps[0]["parse_ok"] is False)
    check("malformed: parse error recorded",
          row.steps[0]["parse_error"] in ("no_json", "json_found_but_no_tool_name"))
    check("malformed: the step was still consumed", row.n_steps_taken >= 2)
    check("malformed: env recorded no env_error",
          not row.final_state["env_errors"])


# -- retries ----------------------------------------------------------------

def test_retry_policy_is_recorded():
    t = make_task(seed=7, n=3)
    rt = FakeRuntime(perfect_script(t))
    row = run_episode(t, rt, MODEL,
                      AgentConfig(retry_policy="reprompt_on_parse_failure:2"))
    check("retry policy pinned into the row",
          row.retry_policy == "reprompt_on_parse_failure:2")


def test_retries_actually_reprompt():
    t = make_task(seed=8, n=2)
    # two junk replies, then a good one -- with 2 retries allowed the step
    # should still succeed
    rt = FakeRuntime(["junk", "junk", call_json(t.canonical_path[0]),
                      call_json(t.canonical_path[1]),
                      submit_json(t.goal_total)])
    row = run_episode(t, rt, MODEL,
                      AgentConfig(retry_policy="reprompt_on_parse_failure:2"))
    check("retries recovered the step", row.steps[0]["parse_ok"] is True)
    check("retry count recorded on the step", "retries=" in row.steps[0]["error"])
    check("no retries by default",
          run_episode(t, FakeRuntime(["junk"] * 9), MODEL,
                      AgentConfig()).steps[0]["parse_ok"] is False)


# -- plumbing ---------------------------------------------------------------

def test_max_steps_default_is_generous():
    t = make_task(seed=9, n=10)
    check("default budget exceeds the minimum needed",
          max_steps_for(t) > t.required_calls)


def test_render_transcript_is_pure():
    turns = [Turn(assistant="a1", observation="o1"),
             Turn(assistant="a2", observation="o2")]
    a = render_transcript("SYS", turns)
    b = render_transcript("SYS", turns)
    check("render_transcript is deterministic", a == b)
    check("transcript contains every turn in order",
          a.index("a1") < a.index("o1") < a.index("a2") < a.index("o2"))
    check("transcript ends with the assistant cue", a.rstrip().endswith("Assistant:"))


def test_row_is_json_serialisable():
    import json
    t = make_task(seed=10, n=3)
    rt = FakeRuntime(perfect_script(t))
    row = run_episode(t, rt, MODEL, AgentConfig())
    s = row.to_json()
    check("row serialises to JSON", isinstance(s, str) and len(s) > 100)
    back = json.loads(s)
    check("round-trips with steps intact", len(back["steps"]) == row.n_steps_taken)
    check("no derived verdict in the row", "success" not in back)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} agent-loop test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
