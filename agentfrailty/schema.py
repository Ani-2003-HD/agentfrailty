"""
Episode schema.

One JSONL row per EPISODE -- a complete agent run on one task instance. This is
the unit that differs from quantcost: there, a row was one generation; here a
row contains the whole multi-step trajectory, because the questions this project
asks are about structure *within* a run.

Two design rules, both load-bearing:

  1. RECORD RAW FACTS, NEVER DERIVED VERDICTS.
     There is deliberately no `success` field. Success is a function of
     final_state vs goal_state, computed at analysis time. quantcost re-graded
     6,000 rows in about a second when the scorer improved; the same discipline
     matters more here, because the grading of a multi-step episode has more
     judgement calls in it and will certainly change at least once.

  2. THE PER-STEP VECTOR IS THE POINT.
     Estimating a hazard rate needs to know what happened at every step, not
     just how the episode ended. `steps` carries one StepRecord per model call,
     with raw output preserved. Anything we later wish we had measured about
     step i has to be recoverable from here without re-running inference.

Storage note: episodes are much fatter than quantcost rows (a 20-step episode
holds 20 raw completions). At ~30 repeats x many tasks this file gets large.
It is written uncompressed during the run and gzipped for the repo; only the
gzip and the derived summary are committed.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

SCHEMA_VERSION = 1


def _sw_vers() -> str:
    try:
        return subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return ""


def _chip() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:
        return ""


def code_version() -> dict:
    """
    Which version of this code produced a row.

    Added after the first smoke runs: results.jsonl is append-only, so three
    runs from three versions of the parser landed in one file and the summary
    silently averaged over data we already knew was corrupt. Provenance in the
    row makes that impossible to do by accident -- filter on commit, and a
    dirty tree is visible rather than assumed away.
    """
    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True, text=True, check=False,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ).stdout.strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty, "schema_version": SCHEMA_VERSION}


@dataclass
class HostInfo:
    """Captured once per session, pinned into every row for reproducibility."""

    chip: str = field(default_factory=_chip)
    macos_version: str = field(default_factory=_sw_vers)
    machine: str = field(default_factory=lambda: platform.machine())
    total_ram_bytes: int = 0
    python_version: str = field(default_factory=platform.python_version)

    def __post_init__(self):
        if not self.total_ram_bytes:
            try:
                self.total_ram_bytes = int(
                    subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, check=False,
                    ).stdout.strip() or 0
                )
            except Exception:
                self.total_ram_bytes = 0


@dataclass
class ModelSpec:
    """Identifies exactly which artifact was measured."""

    name: str = ""            # "qwen2.5-1.5b-instruct"
    quant: str = ""           # normalised: "q4", "q8", "bf16"
    quant_native: str = ""    # runtime's own label, e.g. "Q4_K_M"
    runtime: str = ""         # "ollama" | "llamacpp" | "mlx"
    repo_or_path: str = ""
    weight_bytes: int = 0
    runtime_version: str = ""

    def key(self) -> str:
        return f"{self.name}|{self.quant}|{self.runtime}"


@dataclass
class GenResult:
    """
    Raw output of ONE model call. No scoring, no judgement.

    This is the runtime layer's return type, carried over unchanged from
    quantcost so the adapters work here untouched. In this project it is an
    intermediate: the runner turns each GenResult into a StepRecord and it is
    the StepRecord that gets persisted.
    """

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: Optional[float] = None
    total_s: float = 0.0
    error: str = ""

    @property
    def decode_tps(self) -> Optional[float]:
        """Tokens/sec excluding prefill. None when unmeasurable."""
        if self.ttft_s is None or self.completion_tokens <= 1:
            return None
        decode_s = self.total_s - self.ttft_s
        return self.completion_tokens / decode_s if decode_s > 0 else None


@dataclass
class StepRecord:
    """
    One model call inside an episode. Raw only -- no judgement.

    `tool_name` / `tool_args` are the result of PARSING, which is mechanical and
    cheap to redo, so they are stored as a convenience. Whether the step was
    *correct* is not stored: that depends on the task's goal and on scorer
    logic, both of which will change.
    """

    index: int = 0
    raw_output: str = ""

    # parse outcome (mechanical, re-derivable from raw_output)
    parse_ok: bool = False
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    parse_error: str = ""
    # An arithmetic expression was evaluated to a literal to make the call
    # parse, e.g. {"total": 67 - 2 - 46} -> {"total": 19}. Recorded rather than
    # hidden: strict and lenient accuracy are different numbers, and which one
    # to report is an analysis decision, not a runner decision.
    repaired: bool = False
    repairs: list = field(default_factory=list)

    # environment outcome -- what the world actually did in response
    env_ok: bool = False
    env_result: Any = None
    env_error: str = ""

    # state snapshot AFTER this step, so hazard can be conditioned on state
    state_after: dict = field(default_factory=dict)

    # cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0     # running total fed to the model at this step
    ttft_s: Optional[float] = None
    total_s: float = 0.0

    error: str = ""             # transport/runtime failure at this step


@dataclass
class EpisodeRow:
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    episode_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # what was measured
    model: dict = field(default_factory=dict)
    task_family: str = ""       # e.g. "ledger"
    task_id: str = ""           # a specific generated instance
    chain_length: int = 0       # n: minimum tool calls this instance requires
    repeat: int = 0             # which of the k repeats of THIS task instance

    # conditions -- pinned so cells are comparable
    temperature: float = 0.0
    seed: int = 0
    scaffold: str = "react"
    max_steps: int = 0
    retry_policy: str = "none"  # recorded explicitly; "succeeded with 3 retries"
                                # and "succeeded first try" are different results

    # the trajectory
    steps: list = field(default_factory=list)

    # raw outcome facts. NOT a verdict.
    goal_state: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    termination: str = ""       # "agent_stopped" | "step_cap" | "context_overflow"
                                # | "runtime_error" | "env_error"
    n_steps_taken: int = 0
    total_s: float = 0.0

    # trustworthiness of timing/thermal conditions (from sysmon)
    health: dict = field(default_factory=dict)
    timings_clean: bool = True

    # environment and provenance
    host: dict = field(default_factory=dict)
    code: dict = field(default_factory=code_version)
    mem_before: dict = field(default_factory=dict)
    mem_after: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class EpisodeWriter:
    """
    Append-only JSONL writer, fsynced per episode.

    Same reasoning as quantcost, more so: a frailty study needs many repeats of
    the same task and will run for many hours. Losing a buffered tail at hour
    six is worse than the I/O cost of flushing.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, row: EpisodeRow) -> None:
        with self._lock:
            self._fh.write(row.to_json() + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *a) -> bool:
        self.close()
        return False


def completed_keys(path: str) -> set:
    """
    Already-finished episodes, for resume.

    The key includes repeat AND task_id: the whole design rests on running the
    same instance many times, so a resume that collapsed repeats would silently
    destroy the thing being measured.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
            m = d.get("model", {})
            done.add((
                m.get("name"), m.get("quant"), m.get("runtime"),
                d.get("task_family"), d.get("task_id"),
                d.get("chain_length"), d.get("repeat"),
                d.get("temperature"), d.get("scaffold"),
            ))
    return done
