"""
Runtime adapter interface.

Fairness rules every adapter must honour, because violating them silently
produces a benchmark that measures adapter differences instead of runtime
differences:

  1. TTFT is measured from just before the generate call to the first emitted
     token. Model loading happens in load() and is never counted.
  2. Every runtime gets the same prompt string. Chat templating is applied by
     the adapter using the model's own template, since that is what a real user
     of that runtime would get.
  3. Greedy decoding (temperature 0) everywhere. Deterministic settings do not
     make Metal bit-deterministic -- hence repeats -- but they remove sampling
     as a source of variance.
  4. A warmup generation runs after load and is discarded. First-call overhead
     (kernel compilation, lazy allocation) is not steady-state throughput.
"""

from __future__ import annotations

import time
from typing import Optional

from ..schema import GenResult


class Runtime:
    name: str = "base"

    def __init__(self, model_path: str, **kw):
        self.model_path = model_path
        self.kw = kw
        self._loaded = False

    # -- lifecycle --
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        """Release weights. Critical on 8 GB: a leaked model guarantees swap."""
        self._loaded = False

    def version(self) -> str:
        return ""

    def weight_bytes(self) -> int:
        return 0

    # -- generation --
    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0
    ) -> GenResult:
        raise NotImplementedError

    def warmup(self, max_tokens: int = 8) -> None:
        """Discarded generation to absorb first-call overhead."""
        try:
            self.generate("Hello.", max_tokens=max_tokens)
        except Exception:
            pass

    # -- context manager so weights are always freed --
    def __enter__(self) -> "Runtime":
        self.load()
        return self

    def __exit__(self, *a) -> bool:
        self.unload()
        return False


class _Timer:
    """Tracks TTFT and total wall time across a token stream."""

    def __init__(self):
        self.start = time.perf_counter()
        self.ttft: Optional[float] = None
        self.n = 0

    def tick(self) -> None:
        if self.ttft is None:
            self.ttft = time.perf_counter() - self.start
        self.n += 1

    def done(self) -> float:
        return time.perf_counter() - self.start
