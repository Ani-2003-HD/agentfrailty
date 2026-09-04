"""
Step F: controlled error injection -- the healed-history control.

WHY THIS IS THE LOAD-BEARING EXPERIMENT

The calibration found conditional step accuracy declining WITHIN a fixed chain
length (qwen2.5-1.5b at n=12: 1.00, 1.00, 1.00, 1.00, 0.87, 0.69, 0.33). Three
explanations survive that observation:

    1. self-conditioning  -- the model conditions on its own past errors
    2. long-context       -- the transcript is simply longer by step 6
    3. selection          -- episodes still running at step 6 are the troubled ones

Only a counterfactual separates 1 from 2. Following 2509.09677 S3.2: hold context
LENGTH fixed and vary only the CORRECTNESS of its contents.

    "If we fully heal the history, with a 0% error rate, degradation in the
     model's turn accuracy between turn 1 and a later turn can be attributed to
     long-context issues. If a model's accuracy for a fixed later turn
     consistently worsens with increasing error rate in prior turns, this would
     support our self-conditioning hypothesis."

WHERE THIS GOES BEYOND THEM

Their Appendix I abandons the controlled version of this experiment:

    "there are multiple distinct points of failure within a single trace: an
     error in the retrieval step (looking up an incorrect value) or an error in
     the composition step (an arithmetic mistake). A controlled experiment would
     need to systematically manage the type, frequency, and location of these
     injected errors, making the setup intractable."

In a ledger with a ground-truth oracle at every step it is tractable: a
navigation error is a wrong record_id and an arithmetic error is a wrong total,
and the two are objectively distinguishable. This module controls FREQUENCY and
LOCATION; TYPE needs the running-total task variant and is deliberately deferred
rather than faked.

INJECTED HISTORIES ARE COHERENT, AND THAT IS A DEPARTURE WORTH STATING

In their task the plan is given up front, so a wrong answer does not change what
comes next. In a pointer chase the plan is DISCOVERED, so a wrong turn genuinely
relocates the agent. An injected wrong step therefore shows the TRUE data for
whatever record was called -- which is what a really derailed transcript looks
like -- and the locally-correct next action is computed from where the history
actually leaves the agent. An incoherent history (wrong call, right observation)
would test how a model resolves contradiction, not how it executes: the same
confound 2509.09677's Appendix I raises against naive CoT injection.

The exact wrong-value procedure in the original is in supplementary code we do
not have, so the scheme here is ours and is documented as such.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from .agent import Turn, format_observation
from .envs.ledger import LedgerTask

LOCATION_UNIFORM = "uniform"
LOCATION_EARLY = "early"
LOCATION_LATE = "late"


@dataclass
class InjectionSpec:
    """One cell of the injection experiment."""

    error_rate: float = 0.0        # fraction of prefix steps that are wrong
    prefix_steps: int = 6          # history length, held FIXED across rates
    location: str = LOCATION_UNIFORM
    seed: int = 0

    def label(self) -> str:
        return (f"rate={self.error_rate:.2f}|steps={self.prefix_steps}"
                f"|loc={self.location}")


@dataclass
class InjectedHistory:
    turns: list = field(default_factory=list)
    injected_indices: list = field(default_factory=list)
    final_position: Optional[str] = None    # record the history leaves us at
    chain_ended: bool = False               # that record's next is None
    expected_tool: str = ""                 # locally correct next action
    expected_target: Optional[str] = None
    steps_built: int = 0
    n_chars: int = 0                        # proxy for context length

    def expected_desc(self) -> str:
        return ("submit(total)" if self.expected_tool == "submit"
                else f"get_record({self.expected_target!r})")


def _depth(records, start, limit) -> int:
    """
    How many further steps a chain from `start` supports, capped at `limit`.

    Cycle-guarded: distractors point at each other at random and can form loops.
    A loop is fine for our purposes (it supports arbitrarily many steps) but the
    walk must still terminate.
    """
    seen = set()
    cur, d = start, 0
    while cur is not None and cur not in seen and d < limit:
        seen.add(cur)
        cur = records[cur].next
        d += 1
    return d


def _wrong_positions(spec: InjectionSpec, n: int, rng: random.Random) -> set:
    """
    Which prefix steps are wrong.

    Step 0 is never injected: with no history yet there is nothing to condition
    on, and a wrong first step would change the whole trajectory rather than
    seed it.
    """
    k = int(round(spec.error_rate * n))
    if k <= 0:
        return set()
    candidates = list(range(1, n))
    if not candidates:
        return set()
    k = min(k, len(candidates))

    if spec.location == LOCATION_EARLY:
        return set(candidates[:k])
    if spec.location == LOCATION_LATE:
        return set(candidates[-k:])
    return set(rng.sample(candidates, k))


def build_history(task: LedgerTask, spec: InjectionSpec) -> InjectedHistory:
    """
    Build a synthetic transcript prefix with a controlled error rate.

    Format-identical to a real one, because the agent loop renders both through
    the same pure `render_transcript`. If injection built its context differently
    any difference measured could be an artefact of that.
    """
    rng = random.Random(spec.seed * 7919 + int(spec.error_rate * 1000))
    records = task.records
    path_set = set(task.canonical_path)

    wrong_at = _wrong_positions(spec, spec.prefix_steps, rng)

    turns: list[Turn] = []
    injected: list[int] = []
    position: Optional[str] = None
    visited: set = set()
    chain_ended = False

    for i in range(spec.prefix_steps):
        correct = task.start_id if position is None else records[position].next
        if correct is None:
            chain_ended = True
            break

        if i in wrong_at:
            # A plausible wrong turn -- but it MUST support the rest of the
            # history.
            #
            # The first version of this only checked `next is not None`, one
            # step of lookahead. Wrong turns landed on distractors whose chains
            # dead-ended, the history stopped early, and high-error arms built
            # systematically SHORTER contexts than low-error ones -- 49% length
            # drift, which would have made long-context degradation look exactly
            # like self-conditioning. The fixed-length control is the whole
            # experiment, so depth is now a hard requirement.
            # +1 so the history does not END on a terminal record: the probe
            # must be a NAVIGATION decision in every arm. If a high-error
            # history dead-ended, its probe would be "submit" while a healed
            # history's probe was "which record next" -- different questions,
            # and the arms would no longer be comparable.
            need = spec.prefix_steps - i + 1      # steps still to build, plus one
            pool = [r for r in records
                    if r != correct and r not in visited
                    and r not in path_set
                    and _depth(records, r, need) >= need]
            if not pool:
                # Fall back to an on-path record at the WRONG position: still a
                # realistic confusion, and guaranteed to have depth.
                pool = [r for r in records
                        if r != correct and r not in visited
                        and _depth(records, r, need) >= need]
            if not pool:
                target = correct          # cannot inject without truncating
            else:
                target = rng.choice(sorted(pool))
                injected.append(i)
        else:
            target = correct

        rec = records[target]
        call = json.dumps({"name": "get_record",
                           "arguments": {"record_id": target}},
                          ensure_ascii=False)
        obs = format_observation({"id": rec.id, "value": rec.value,
                                  "next": rec.next})
        turns.append(Turn(assistant=call, observation=obs))

        visited.add(target)
        position = target
        chain_ended = rec.next is None

    nxt = None if position is None else records[position].next
    h = InjectedHistory(
        turns=turns,
        injected_indices=injected,
        final_position=position,
        chain_ended=chain_ended or nxt is None,
        steps_built=len(turns),
        n_chars=sum(len(t.assistant) + len(t.observation) for t in turns),
    )
    h.expected_tool = "submit" if h.chain_ended else "get_record"
    h.expected_target = None if h.chain_ended else nxt
    return h


def grade_probe(history: InjectedHistory, tool: Optional[str],
                args: Optional[dict]) -> str:
    """
    Grade the model's FIRST action after the injected history.

    Returns "correct" | "wrong_target" | "wrong_tool" | "malformed".

    Conditional in the same sense as 2509.09677's turn accuracy: correct given
    where the history left the agent, regardless of how it got there. That is
    what makes the arms comparable -- a 100%-error history leaves the agent
    somewhere different from a 0% one, and grading against the canonical path
    would score the history rather than the model.
    """
    if tool is None:
        return "malformed"
    if tool != history.expected_tool:
        return "wrong_tool"
    if history.expected_tool == "submit":
        return "correct"
    rid = (args or {}).get("record_id")
    rid = rid.strip() if isinstance(rid, str) else rid
    return "correct" if rid == history.expected_target else "wrong_target"
