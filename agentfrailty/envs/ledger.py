"""
The pointer-chase ledger environment.

A store of records, each holding a value and a pointer to the next record. The
agent starts at a known record, reads its value, follows the pointer, and repeats
until the chain ends, then submits the running total.

WHY THIS SHAPE -- each property is load-bearing, not decoration:

  1. N IS REQUIRED, NOT OBSERVED.
     A chain of N records needs exactly N get_record calls plus one submit.
     There is no shortcut: the id of position i+1 is only learnable by reading
     position i. tau^2-bench and TRAJECT-Bench both plot success against the
     number of calls an agent HAPPENED to make, which confounds "harder task"
     with "longer chain". Here the chain length is set by construction.

  2. IT IS INHERENTLY SEQUENTIAL.
     arXiv 2509.09677 uses a running sum and concedes in its own limitations
     that summation is associative and therefore theoretically parallelizable --
     a model could in principle do it in one shot. A pointer chase cannot be
     parallelized: the dependency is structural, not arithmetic.

  3. EVERY STEP HAS AN OBJECTIVE ORACLE.
     At position i there is exactly one correct call. That gives per-step
     correctness without human judgement, which is precisely what 2509.09677's
     Appendix A lacked when it tried to find self-conditioning in GAIA/ALFWorld
     /WebShop ("correctness of steps is subjective to determine on these tasks").
     It also makes RECOVERY visible: an agent that wanders off the canonical
     path and returns to it is measurable, and recovery is what a hazard model
     needs in order to be more than a survival curve.

  4. THE ENVIRONMENT NEVER FAILS.
     Pure in-memory dicts, no I/O, no clock, no network. `call()` never raises.
     An unknown record id is an AGENT error, reported as a normal tool result --
     never an environment error. The two must never be confused, or the study
     measures this file instead of the model.

Difficulty is tunable on three independent axes:
    n                chain length        (the primary independent variable)
    n_distractors    confusable records  (raises wrong-path probability)
    keys_per_step    arithmetic load     (the analogue of 2509.09677's K)

Five-letter words are used for ids and two-digit values, following 2509.09677's
reasoning: minimise errors that come from tokenisation rather than execution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Common five-letter words. Deliberately ordinary and non-thematic: an id that
# hints at its own position would leak the answer.
WORDS = [
    "about", "above", "actor", "adult", "after", "again", "agent", "alarm",
    "album", "alert", "alike", "alive", "alone", "along", "alter", "among",
    "anger", "angle", "apart", "apple", "apply", "arena", "argue", "arise",
    "armor", "array", "arrow", "aside", "asset", "avoid", "awake", "award",
    "aware", "badly", "baker", "basic", "batch", "beach", "beard", "begin",
    "being", "below", "bench", "birth", "black", "blade", "blame", "blank",
    "blast", "blend", "blind", "block", "blood", "board", "boost", "booth",
    "bound", "brain", "brand", "brass", "brave", "bread", "break", "breed",
    "brief", "bring", "broad", "broke", "brown", "brush", "build", "built",
    "bunch", "burst", "cabin", "cable", "cache", "candy", "canon", "cargo",
    "carry", "carve", "catch", "cause", "chain", "chair", "chalk", "charm",
    "chart", "chase", "cheap", "check", "chest", "chief", "child", "chose",
    "civil", "claim", "class", "clean", "clear", "clerk", "click", "cliff",
    "climb", "clock", "close", "cloth", "cloud", "coach", "coast", "count",
    "court", "cover", "crack", "craft", "crash", "crate", "crazy", "cream",
    "crime", "cross", "crowd", "crown", "crude", "curve", "cycle", "daily",
    "dance", "dealt", "death", "debut", "delay", "delta", "dense", "depth",
    "doubt", "draft", "drain", "drama", "drank", "dream", "dress", "drift",
    "drink", "drive", "drove", "eager", "early", "earth", "eight", "elect",
    "elite", "empty", "enemy", "enjoy", "enter", "entry", "equal", "error",
    "essay", "event", "every", "exact", "exist", "extra", "faith", "false",
    "fault", "favor", "feast", "fence", "fever", "field", "fiber", "fifth",
    "fight", "final", "first", "flame", "flash", "fleet", "float", "floor",
    "focus", "force", "forge", "forth", "found", "frame", "fraud", "fresh",
    "front", "frost", "fruit", "fully", "funny", "giant", "given", "glass",
    "globe", "glory", "grace", "grade", "grain", "grand", "grant", "grape",
    "graph", "grasp", "grass", "grave", "great", "green", "greet", "grief",
    "gross", "group", "guard", "guess", "guest", "guide", "habit", "handy",
    "happy", "harsh", "haste", "heart", "heavy", "hedge", "hence", "hobby",
    "honey", "honor", "horse", "hotel", "house", "human", "humor", "hurry",
    "ideal", "image", "imply", "index", "inner", "input", "irony", "issue",
    "ivory", "joint", "judge", "juice", "known", "label", "labor", "large",
    "laser", "later", "laugh", "layer", "learn", "lease", "least", "leave",
    "legal", "lemon", "level", "lever", "light", "limit", "linen", "liver",
    "local", "lodge", "logic", "loose", "lower", "loyal", "lucky", "lunar",
    "lunch", "magic", "major", "maker", "march", "match", "maybe", "mayor",
    "meant", "medal", "media", "mercy", "merit", "metal", "meter", "midst",
    "might", "minor", "minus", "mixed", "model", "money", "month", "moral",
    "motor", "mount", "mouse", "mouth", "movie", "music", "naked", "nasty",
    "naval", "nerve", "never", "newly", "night", "noble", "noise", "north",
    "novel", "nurse", "occur", "ocean", "offer", "often", "olive", "onion",
    "onset", "opera", "orbit", "order", "organ", "other", "ought", "outer",
    "owner", "paint", "panel", "panic", "paper", "party", "patch", "pause",
    "peace", "pearl", "penny", "phase", "phone", "photo", "piano", "piece",
    "pilot", "pitch", "place", "plain", "plane", "plant", "plate", "plaza",
    "point", "polar", "porch", "pound", "power", "press", "price", "pride",
    "prime", "print", "prior", "prize", "probe", "proof", "proud", "prove",
    "pulse", "punch", "pupil", "purse", "quest", "queue", "quick", "quiet",
    "quite", "quota", "radar", "radio", "raise", "rally", "range", "rapid",
    "ratio", "reach", "ready", "realm", "rebel", "refer", "reign", "relax",
    "relay", "renew", "reply", "rider", "ridge", "rifle", "right", "rigid",
    "risky", "rival", "river", "roast", "robot", "rocky", "roman", "rough",
    "round", "route", "royal", "rugby", "ruler", "rural", "sadly", "saint",
    "salad", "sales", "sauce", "scale", "scene", "scope", "score", "scout",
    "sense", "serve", "seven", "shade", "shaft", "shake", "shall", "shape",
    "share", "sharp", "sheep", "sheet", "shelf", "shell", "shift", "shine",
    "shirt", "shock", "shoot", "shore", "short", "shown", "sight", "silly",
    "since", "siren", "sixth", "skill", "slate", "sleep", "slice", "slide",
    "slope", "small", "smart", "smell", "smile", "smoke", "snake", "solar",
    "solid", "solve", "sorry", "sound", "south", "space", "spare", "speak",
    "speed", "spend", "spent", "spice", "spine", "spite", "split", "spoke",
    "sport", "squad", "stack", "staff", "stage", "stain", "stake", "stamp",
    "stand", "stare", "start", "state", "steam", "steel", "steep", "steer",
    "stick", "stiff", "still", "stock", "stone", "stood", "store", "storm",
    "story", "stove", "strap", "straw", "strip", "stuck", "study", "stuff",
    "style", "sugar", "suite", "sunny", "super", "sweet", "swift", "swing",
    "sword", "table", "taken", "tally", "taste", "teach", "tempo", "tenth",
    "thank", "theft", "their", "theme", "there", "thick", "thigh", "thing",
    "think", "third", "those", "three", "threw", "throw", "thumb", "tiger",
    "tight", "timer", "tired", "title", "today", "token", "topic", "torch",
    "total", "touch", "tough", "tower", "toxic", "trace", "track", "trade",
    "trail", "train", "trait", "trash", "treat", "trend", "trial", "tribe",
    "trick", "troop", "truck", "truly", "trunk", "trust", "truth", "twice",
    "twist", "ultra", "uncle", "under", "union", "unite", "unity", "until",
    "upper", "upset", "urban", "urged", "usage", "usual", "vague", "valid",
    "value", "valve", "vapor", "vault", "venue", "verse", "video", "villa",
    "viral", "virus", "visit", "vital", "vivid", "vocal", "voice", "voter",
    "wagon", "waste", "watch", "water", "weigh", "weird", "wheat", "wheel",
    "where", "which", "while", "white", "whole", "whose", "widen", "widow",
    "width", "witch", "woman", "world", "worry", "worse", "worst", "worth",
    "would", "wound", "wrist", "write", "wrong", "wrote", "yield", "young",
    "yours", "youth", "zebra",
]

VALUE_MIN, VALUE_MAX = -99, 99
END = None  # terminal pointer


@dataclass
class Record:
    id: str
    value: int
    next: Optional[str]


@dataclass
class LedgerTask:
    """
    One generated instance. Fully determined by (seed, n, n_distractors,
    keys_per_step) -- regeneration must be byte-identical, so nothing here may
    depend on dict ordering, time, or process state.
    """

    task_id: str = ""
    seed: int = 0
    n: int = 0                       # REQUIRED get_record calls
    n_distractors: int = 0
    keys_per_step: int = 1
    start_id: str = ""
    records: dict = field(default_factory=dict)   # id -> Record (path + distractors)
    canonical_path: list = field(default_factory=list)  # ids in visit order
    goal_total: int = 0

    @property
    def required_calls(self) -> int:
        """N get_record calls plus one submit."""
        return self.n + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["records"] = {k: asdict(v) for k, v in self.records.items()}
        return d


def make_task(
    seed: int,
    n: int,
    n_distractors: int = 0,
    keys_per_step: int = 1,
) -> LedgerTask:
    """
    Generate one task instance.

    Determinism: a single local Random seeded here. No global random state is
    touched, so generation is unaffected by anything else running.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    need = n + n_distractors
    if need > len(WORDS):
        raise ValueError(f"need {need} unique ids, vocabulary has {len(WORDS)}")

    rng = random.Random(seed)
    ids = rng.sample(WORDS, need)
    path_ids, distractor_ids = ids[:n], ids[n:]

    records: dict[str, Record] = {}

    # The chain. Each record points at the next; the last terminates.
    total = 0
    for i, rid in enumerate(path_ids):
        value = rng.randint(VALUE_MIN, VALUE_MAX)
        nxt = path_ids[i + 1] if i + 1 < n else END
        records[rid] = Record(id=rid, value=value, next=nxt)
        total += value

    # Distractors exist to make wrong turns possible and confusable. They must
    # never point INTO the canonical path: a distractor that rejoins the chain
    # would let a lost agent accidentally finish, which would corrupt the
    # per-step oracle.
    for j, rid in enumerate(distractor_ids):
        value = rng.randint(VALUE_MIN, VALUE_MAX)
        others = [d for d in distractor_ids if d != rid]
        nxt = rng.choice(others) if others and rng.random() < 0.7 else END
        records[rid] = Record(id=rid, value=value, next=nxt)

    return LedgerTask(
        task_id=f"ledger-n{n}-d{n_distractors}-k{keys_per_step}-s{seed}",
        seed=seed,
        n=n,
        n_distractors=n_distractors,
        keys_per_step=keys_per_step,
        start_id=path_ids[0],
        records=records,
        canonical_path=list(path_ids),
        goal_total=total,
    )


@dataclass
class ToolResult:
    """
    Outcome of one tool call.

    `ok` describes whether the ENVIRONMENT served the call, not whether the
    agent was right to make it. Asking for a record that does not exist is a
    served call with ok=True and an `error` payload -- an agent mistake, not an
    environment failure. `env_error` is reserved for bugs in this file and must
    stay empty for the whole study.
    """

    ok: bool = True
    payload: Any = None
    env_error: str = ""


class LedgerEnv:
    """
    Deterministic, in-memory, never-raising.

    Records every call so the trajectory can be graded against the canonical
    path afterwards. Holds no opinion about correctness.
    """

    TOOLS = [
        {
            "name": "get_record",
            "description": (
                "Look up one record in the ledger. Returns its value and the id "
                "of the next record in the chain, or null if the chain ends here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "Record id to read"}
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "submit",
            "description": "Submit the final total once the chain has ended.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer", "description": "Sum of all values"}
                },
                "required": ["total"],
            },
        },
    ]

    def __init__(self, task: LedgerTask):
        self.task = task
        self.calls: list[dict] = []      # every call, in order
        self.submitted: Optional[int] = None
        self.finished = False

    # -- observation ------------------------------------------------------

    def initial_observation(self) -> str:
        return (
            "You are reading a chain of records in a ledger.\n\n"
            f"Start at record '{self.task.start_id}'. Read its value and add it "
            "to a running total, then follow its 'next' pointer to the next "
            "record. Continue until a record's 'next' is null.\n\n"
            "Then call submit with the total of every value you read.\n\n"
            "Read exactly one record per step."
        )

    def tool_specs(self) -> list:
        return self.TOOLS

    # -- the only mutating entry point ------------------------------------

    def call(self, name: Any, args: Any) -> ToolResult:
        """
        Execute one tool call. NEVER raises, for any input whatsoever.

        Anything malformed is recorded and returned as a normal result, because
        a crash here would end an episode in a way that looks like a model
        failure and is not.
        """
        try:
            return self._call(name, args)
        except Exception as e:  # pragma: no cover -- must never fire
            self.calls.append({"tool": name, "args": args, "result": None,
                               "env_error": repr(e)})
            return ToolResult(ok=False, env_error=repr(e))

    def _call(self, name: Any, args: Any) -> ToolResult:
        args = args if isinstance(args, dict) else {}

        if name == "get_record":
            rid = args.get("record_id")
            rid = rid.strip() if isinstance(rid, str) else rid
            rec = self.task.records.get(rid) if isinstance(rid, str) else None
            if rec is None:
                payload = {"error": f"no record with id {rid!r}"}
            else:
                payload = {"id": rec.id, "value": rec.value, "next": rec.next}
            self.calls.append({"tool": "get_record", "args": {"record_id": rid},
                               "result": payload, "env_error": ""})
            return ToolResult(ok=True, payload=payload)

        if name == "submit":
            total = args.get("total")
            if isinstance(total, bool):       # bool is an int subclass; reject
                total = None
            if isinstance(total, str):
                try:
                    total = int(total.strip())
                except (ValueError, AttributeError):
                    total = None
            if not isinstance(total, int):
                payload = {"error": "submit requires an integer 'total'"}
                self.calls.append({"tool": "submit", "args": {"total": args.get("total")},
                                   "result": payload, "env_error": ""})
                return ToolResult(ok=True, payload=payload)

            self.submitted = total
            self.finished = True
            payload = {"accepted": True}
            self.calls.append({"tool": "submit", "args": {"total": total},
                               "result": payload, "env_error": ""})
            return ToolResult(ok=True, payload=payload)

        payload = {"error": f"unknown tool {name!r}"}
        self.calls.append({"tool": name, "args": args, "result": payload,
                           "env_error": ""})
        return ToolResult(ok=True, payload=payload)

    # -- state ------------------------------------------------------------

    def state(self) -> dict:
        """Raw facts only. Whether the episode SUCCEEDED is computed elsewhere."""
        return {
            "n_calls": len(self.calls),
            "get_record_calls": [c["args"].get("record_id")
                                 for c in self.calls if c["tool"] == "get_record"],
            "submitted": self.submitted,
            "finished": self.finished,
            "env_errors": [c["env_error"] for c in self.calls if c["env_error"]],
        }


def solve(task: LedgerTask) -> tuple[list, int]:
    """
    The canonical solution: what a perfect agent does.

    Used to PROVE the task needs exactly n get_record calls, and as the
    reference the per-step oracle grades against.
    """
    env = LedgerEnv(task)
    path, total = [], 0
    rid = task.start_id
    while rid is not None:
        res = env.call("get_record", {"record_id": rid})
        rec = res.payload
        assert "error" not in rec, f"canonical path hit a missing record: {rid}"
        path.append(rid)
        total += rec["value"]
        rid = rec["next"]
    env.call("submit", {"total": total})
    return path, total
