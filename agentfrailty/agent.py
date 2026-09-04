"""
The agent loop.

A plain ReAct-style loop: show the model the tools and the transcript so far,
read one tool call, execute it, append the result, repeat. No framework. The
scaffold is a variable in this study, so it has to be something we control
completely rather than something whose behaviour changes with a dependency bump.

FOUR DECISIONS THAT SHAPE THE MEASUREMENT:

  1. THE FULL TRANSCRIPT IS REPLAYED EVERY TURN.
     This is not an implementation shortcut -- it is the mechanism under test.
     Self-conditioning (2509.09677) is the model conditioning on its own past
     errors. A loop that summarised the history, or fed back only the last
     result, would suppress the effect we are trying to measure.

  2. TRANSCRIPT CONSTRUCTION IS SEPARATE FROM EXECUTION.
     `render_transcript` is a pure function of a list of turns. Step F's
     error-injection arm needs to build a SYNTHETIC history with a controlled
     error rate and hand it to the same loop -- exactly 2509.09677's design,
     where context length is held fixed and only the correctness of its contents
     varies. Splitting these now avoids a rewrite later.

  3. THE RETRY POLICY IS RECORDED, NEVER HIDDEN.
     "Succeeded first try" and "succeeded after two reprompts" are different
     results. Retries are counted per step and the policy is pinned into the
     episode row. Default is none: retrying on a parse failure would mask
     exactly the failure mode we care about.

  4. THE STEP CAP PRODUCES RIGHT-CENSORED DATA, NOT MISSING DATA.
     An episode that hits the cap has not "failed" -- it was stopped while still
     running. Survival analysis handles right-censoring properly, but only if
     the cap is recorded as a distinct termination reason rather than folded in
     with failure. Hence `termination`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Optional

from .envs.ledger import LedgerEnv, LedgerTask
from .parsing import parse_tool_call, render_tools
from .schema import EpisodeRow, ModelSpec, StepRecord

# The transcript delimiters. The loop owns them, so it also owns the stop
# sequences that keep a model inside its own turn.
STOP_SEQUENCES = ["Tool result:", "\nAssistant:", "\nUser:", "\nTask:"]

SYSTEM_TEMPLATE = """You are an agent that completes a task by calling tools.

Available tools:

{tools}

To call a tool, reply with ONLY a JSON object of this form:
{{"name": "<tool name>", "arguments": {{...}}}}

Reply with nothing else -- no explanation, no markdown fences.
Call exactly one tool per reply.

Task:
{task}
"""


@dataclass
class AgentConfig:
    max_steps: int = 0            # 0 -> derived from the task (see max_steps_for)
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 128         # one JSON call; more just invites prose
    retry_policy: str = "none"    # "none" | "reprompt_on_parse_failure:<k>"
    scaffold: str = "react"

    def retries_allowed(self) -> int:
        if self.retry_policy.startswith("reprompt_on_parse_failure:"):
            try:
                return int(self.retry_policy.split(":", 1)[1])
            except ValueError:
                return 0
        return 0


def max_steps_for(task: LedgerTask, slack: int = 3, factor: float = 2.0) -> int:
    """
    Step budget for a task.

    Generous on purpose. A tight cap censors wandering agents early, and a
    censored episode carries less information than a completed one. The cost of
    slack is wall-clock; the cost of a tight cap is data.
    """
    return int(task.n * factor) + slack


# -- transcript -------------------------------------------------------------

@dataclass
class Turn:
    """One exchange. `assistant` is the raw model text; `observation` is what
    the environment said back."""

    assistant: str = ""
    observation: str = ""


def render_transcript(system: str, turns: list) -> str:
    """
    Pure function: (system prompt, turns) -> the exact string the model sees.

    Kept pure so Step F can synthesise a history with a chosen error rate and
    feed it through the identical code path. If injection built its context a
    different way, any difference it measured could be an artefact of that.
    """
    parts = [system]
    for t in turns:
        if t.assistant:
            parts.append(f"\nAssistant: {t.assistant}")
        if t.observation:
            parts.append(f"\nTool result: {t.observation}")
    parts.append("\nAssistant:")
    return "".join(parts)


def format_observation(payload: Any) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False)


# -- the loop ---------------------------------------------------------------

def run_episode(
    task: LedgerTask,
    runtime,
    model: ModelSpec,
    config: AgentConfig,
    run_id: str = "",
    repeat: int = 0,
    prefix_turns: Optional[list] = None,
) -> EpisodeRow:
    """
    Run one episode and return a fully populated EpisodeRow.

    `prefix_turns` seeds the transcript with a pre-built history -- the hook
    Step F uses for error injection. When None, the episode starts clean.

    Records raw facts only. Nothing here decides whether the episode succeeded;
    that is the trajectory oracle's job at analysis time.
    """
    env = LedgerEnv(task)
    system = SYSTEM_TEMPLATE.format(
        tools=render_tools(env.tool_specs()),
        task=env.initial_observation(),
    )

    turns: list[Turn] = list(prefix_turns) if prefix_turns else []
    steps: list[StepRecord] = []
    max_steps = config.max_steps or max_steps_for(task)
    termination = "step_cap"
    t_episode = time.perf_counter()

    for i in range(max_steps):
        prompt = render_transcript(system, turns)

        gen = None
        parsed = None
        retries = 0
        for attempt in range(config.retries_allowed() + 1):
            gen = runtime.generate(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                # Seed varies with step so a step is not forced to repeat the
                # previous one verbatim, but stays deterministic given (seed, i).
                seed=config.seed + i * 1000 + attempt,
            )
            if gen.error:
                break
            parsed = parse_tool_call(gen.text)
            if parsed.ok:
                break
            retries = attempt + 1 if attempt < config.retries_allowed() else attempt

        rec = StepRecord(index=i, raw_output=gen.text if gen else "")

        if gen is not None and gen.error:
            rec.error = gen.error
            steps.append(rec)
            termination = "runtime_error"
            break

        rec.prompt_tokens = gen.prompt_tokens
        rec.completion_tokens = gen.completion_tokens
        rec.context_tokens = gen.prompt_tokens
        rec.ttft_s = gen.ttft_s
        rec.total_s = gen.total_s

        if parsed is not None and parsed.ok:
            rec.parse_ok = True
            rec.tool_name = parsed.name
            rec.tool_args = parsed.args
            rec.repaired = parsed.repaired
            rec.repairs = list(parsed.repairs)
        else:
            rec.parse_ok = False
            rec.parse_error = parsed.error if parsed else "no_generation"

        # Execute. The env serves malformed calls too -- it never raises, and an
        # unparseable reply still consumes a step, which is the honest accounting.
        result = env.call(rec.tool_name, rec.tool_args)
        rec.env_ok = result.ok
        rec.env_result = result.payload
        rec.env_error = result.env_error
        rec.state_after = {
            "n_calls": len(env.calls),
            "finished": env.finished,
            "submitted": env.submitted,
        }
        if retries:
            rec.error = f"retries={retries}"

        steps.append(rec)
        # Append the PARSED SPAN, not the raw text.
        #
        # A small model often keeps writing past its own turn, fabricating
        # "Tool result:" lines and continuing the conversation. Feeding that
        # back would put invented observations into context, and every value the
        # model later "read" would be its own hallucination. Stop sequences
        # prevent most of it; this makes it structurally impossible.
        #
        # rec.raw_output still holds the full text, so nothing is lost for
        # auditing -- only the context is cleaned.
        assistant_text = parsed.span if (parsed and parsed.ok and parsed.span) \
            else (gen.text or "")
        turns.append(Turn(
            assistant=assistant_text,
            observation=format_observation(result.payload),
        ))

        if env.finished:
            termination = "agent_stopped"
            break

    row = EpisodeRow(
        run_id=run_id or uuid.uuid4().hex[:12],
        episode_id=uuid.uuid4().hex[:12],
        model=asdict(model),
        task_family="ledger",
        task_id=task.task_id,
        chain_length=task.n,
        repeat=repeat,
        temperature=config.temperature,
        seed=config.seed,
        scaffold=config.scaffold,
        max_steps=max_steps,
        retry_policy=config.retry_policy,
        steps=[asdict(s) for s in steps],
        goal_state={"total": task.goal_total, "path": task.canonical_path},
        final_state=env.state(),
        termination=termination,
        n_steps_taken=len(steps),
        total_s=time.perf_counter() - t_episode,
    )
    return row


def calls_from_steps(steps: list) -> list:
    """
    Adapt persisted StepRecords into the trajectory oracle's input format.

    Keeps grading independent of the loop: the oracle can re-grade any stored
    episode without re-running inference, which is the whole point of storing
    raw rows.
    """
    out = []
    for s in steps:
        d = s if isinstance(s, dict) else asdict(s)
        out.append({
            "tool": d.get("tool_name"),
            "args": d.get("tool_args"),
            "parse_ok": d.get("parse_ok", False),
        })
    return out
