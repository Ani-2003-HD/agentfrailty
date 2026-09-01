"""
Trajectory oracle: grade a recorded episode against the ledger's canonical path.

THE CENTRAL DESIGN DECISION -- every step is graded TWO ways, and the pair is
what makes this study possible:

  on_canonical   Did the agent do what a PERFECT agent would do at this step
                 index? This is absorbing: one wrong turn and every later step
                 is off-canonical even if the agent is behaving sensibly given
                 where it now is. Gives the survival view.

  locally_correct  Given where the agent ACTUALLY is -- the last record it
                 successfully read -- did it take the right next action? An
                 agent that misread a pointer once and then followed the wrong
                 chain flawlessly is locally correct at every subsequent step.

This mirrors 2509.09677's conditional step accuracy: "the fraction of samples
where the state update from step i-1 to step i is correct, REGARDLESS of the
correctness of the model's state at step i-1." Without the conditional view you
cannot tell the two hypotheses apart:

  * If failures were independent, locally_correct should be flat in step index.
  * If self-conditioning is real, locally_correct DECLINES with step index --
    the agent gets worse at following a pointer purely because its context now
    contains its own earlier mistakes.

The absorbing view alone cannot show this, because after the first error it is
zero by construction regardless of what the model does.

A SECOND DECOMPOSITION, for free: navigation and arithmetic are separate
capabilities and are graded separately. quantcost found three capabilities with
three different breaking points; the same may hold here, and collapsing them
would hide it.

JUDGEMENT CALLS, recorded because they will be revisited (which is exactly why
raw rows are stored and never derived verdicts):

  1. Re-reading a record already read is counted ONCE toward the observed sum.
     A careful agent would not double-count on a revisit. Alternative defensible
     choice: count every read. Change here, re-grade, no re-inference.
  2. `submit` before the chain has ended is `premature_submit`, even if the
     total happens to be right.
  3. Arithmetic is graded twice: against the true goal, and against the values
     the agent actually observed. The second isolates arithmetic failure from
     navigation failure -- an agent that walked the wrong chain but summed it
     correctly has an arithmetic capability that is intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from ..envs.ledger import LedgerTask

# Error taxonomy. Every non-locally-correct step gets exactly one of these.
ERR_REVISIT = "revisit"                    # id already successfully read
ERR_OFF_PATH = "off_path_existing"         # a real record, but not the right one
ERR_NONEXISTENT = "nonexistent_id"         # invented an id
ERR_PREMATURE_SUBMIT = "premature_submit"  # submitted before the chain ended
ERR_LATE_CONTINUE = "late_continue"        # kept reading after the chain ended
ERR_WRONG_TOOL = "wrong_tool"              # a tool that does not exist
ERR_MALFORMED = "malformed"                # unparseable / missing arguments


@dataclass
class StepGrade:
    index: int
    tool: Optional[str] = None
    target: Any = None                 # record_id, or submitted total

    on_canonical: bool = False
    locally_correct: bool = False
    error_type: str = ""

    position_before: Optional[str] = None    # last record successfully read
    position_on_path: bool = False           # is that record on the canonical path
    expected_action: str = ""                # what a locally-correct step was
    recovered: bool = False                  # off-path -> on-path at this step


@dataclass
class EpisodeGrade:
    task_id: str = ""
    n: int = 0

    steps: list = field(default_factory=list)
    n_steps: int = 0
    n_locally_correct: int = 0
    n_on_canonical: int = 0
    first_error_index: Optional[int] = None   # first step not locally correct
    n_recoveries: int = 0
    error_counts: dict = field(default_factory=dict)

    # navigation
    reached_chain_end: bool = False           # read the final record of the chain
    walked_canonical: bool = False            # visited the whole path, in order

    # arithmetic
    submitted: Optional[int] = None
    goal_total: int = 0
    observed_total: int = 0                   # sum over distinct reads, in order
    outcome_correct: bool = False             # submitted == goal_total
    arithmetic_correct_given_reads: Optional[bool] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) if not isinstance(s, dict) else s for s in self.steps]
        return d


def grade_episode(task: LedgerTask, calls: list) -> EpisodeGrade:
    """
    Grade one episode.

    `calls` is a list of dicts: {"tool": str|None, "args": dict|None,
    "parse_ok": bool}. That is derivable from either the env's own call log or
    the persisted StepRecords, so grading never needs re-inference.
    """
    path = task.canonical_path
    path_set = set(path)
    records = task.records

    g = EpisodeGrade(task_id=task.task_id, n=task.n, goal_total=task.goal_total)

    position: Optional[str] = None     # last record successfully read
    chain_ended = False                # that record's next was None
    read_order: list[str] = []         # distinct ids, first-read order
    was_off_path = False

    for i, c in enumerate(calls):
        tool = c.get("tool")
        args = c.get("args") if isinstance(c.get("args"), dict) else {}
        parsed = c.get("parse_ok", True)

        # What SHOULD happen here, given where the agent actually is?
        if chain_ended:
            expected_tool, expected_target = "submit", None
            expected_desc = "submit(total)"
        elif position is None:
            expected_tool, expected_target = "get_record", task.start_id
            expected_desc = f"get_record({task.start_id!r})"
        else:
            expected_target = records[position].next
            expected_tool = "get_record"
            expected_desc = f"get_record({expected_target!r})"

        s = StepGrade(
            index=i,
            tool=tool,
            position_before=position,
            position_on_path=(position in path_set) if position else True,
            expected_action=expected_desc,
        )

        # The perfect agent's action at THIS step index (absorbing view).
        if i < task.n:
            s.on_canonical = (tool == "get_record"
                              and args.get("record_id") == path[i])
        elif i == task.n:
            s.on_canonical = (tool == "submit"
                              and _as_int(args.get("total")) == task.goal_total)

        # ---- classify ----
        if not parsed or tool is None:
            s.error_type = ERR_MALFORMED

        elif tool == "get_record":
            rid = args.get("record_id")
            rid = rid.strip() if isinstance(rid, str) else rid
            s.target = rid

            if not isinstance(rid, str) or "record_id" not in args:
                s.error_type = ERR_MALFORMED
            elif chain_ended:
                s.error_type = ERR_LATE_CONTINUE
            elif rid not in records:
                s.error_type = ERR_NONEXISTENT
            elif rid in read_order:
                s.error_type = ERR_REVISIT
            elif rid != expected_target:
                s.error_type = ERR_OFF_PATH
            else:
                s.locally_correct = True

            # The move happens whenever the record exists -- a wrong turn still
            # moves the agent. This is what lets a later step be locally correct
            # again, and what makes recovery observable.
            if isinstance(rid, str) and rid in records and not chain_ended:
                if rid not in read_order:
                    read_order.append(rid)
                    g.observed_total += records[rid].value
                was_off_before = position is not None and position not in path_set
                position = rid
                chain_ended = records[rid].next is None
                if was_off_before and rid in path_set:
                    s.recovered = True
                    g.n_recoveries += 1

        elif tool == "submit":
            total = _as_int(args.get("total"))
            s.target = total
            if total is None:
                s.error_type = ERR_MALFORMED
            elif not chain_ended:
                s.error_type = ERR_PREMATURE_SUBMIT
            else:
                s.locally_correct = True
            if total is not None:
                g.submitted = total

        else:
            s.error_type = ERR_WRONG_TOOL

        if not s.locally_correct and g.first_error_index is None:
            g.first_error_index = i
        if s.error_type:
            g.error_counts[s.error_type] = g.error_counts.get(s.error_type, 0) + 1
        if position is not None and position not in path_set:
            was_off_path = True

        g.steps.append(s)

        if tool == "submit" and g.submitted is not None:
            break   # episode is over; anything after is not part of the run

    # ---- episode-level ----
    g.n_steps = len(g.steps)
    g.n_locally_correct = sum(1 for s in g.steps if s.locally_correct)
    g.n_on_canonical = sum(1 for s in g.steps if s.on_canonical)
    g.reached_chain_end = chain_ended
    g.walked_canonical = read_order[:task.n] == path
    g.outcome_correct = g.submitted is not None and g.submitted == task.goal_total
    if g.submitted is not None:
        g.arithmetic_correct_given_reads = (g.submitted == g.observed_total)
    _ = was_off_path
    return g


def _as_int(v: Any) -> Optional[int]:
    """Match LedgerEnv's leniency exactly, so grading and execution agree."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


def step_accuracy_by_index(grades: list) -> dict:
    """
    Conditional per-step accuracy as a function of step index, pooled over
    episodes -- the quantity whose SLOPE distinguishes the two hypotheses.

    Flat  => per-step failures look independent.
    Falls => within-run dependence (self-conditioning), OR frailty. Telling
             those apart needs the same task repeated many times, which is what
             the run design provides and the analysis stage exploits.
    """
    num: dict[int, int] = {}
    den: dict[int, int] = {}
    for g in grades:
        for s in g.steps:
            den[s.index] = den.get(s.index, 0) + 1
            if s.locally_correct:
                num[s.index] = num.get(s.index, 0) + 1
    return {i: (num.get(i, 0) / den[i]) for i in sorted(den)}
