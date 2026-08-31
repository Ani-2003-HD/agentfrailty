"""
Ollama adapter (HTTP API on localhost:11434).

Included not because it is fast but because it is what most people actually
run. If Ollama's defaults cost users measurable quality or speed versus the
same weights driven directly, that is a finding worth publishing -- it affects
far more people than the MLX-vs-llama.cpp question does.

Ollama is a llama.cpp wrapper, so treat its numbers as "llama.cpp plus default
settings and server overhead", not an independent third engine.
"""

from __future__ import annotations

import json
from typing import Optional

from .base import Runtime, _Timer
from ..schema import GenResult

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaRuntime(Runtime):
    name = "ollama"

    def __init__(self, model_path: str, host: str = DEFAULT_HOST, **kw):
        # model_path here is an Ollama model tag, e.g. "gemma3:4b-it-q4_K_M"
        super().__init__(model_path, **kw)
        self.host = host.rstrip("/")

    def load(self) -> None:
        import requests

        # Empty prompt triggers a load without generating; keeps model-load
        # time out of the first measured generation.
        requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model_path, "prompt": "", "stream": False},
            timeout=600,
        ).raise_for_status()
        self._loaded = True

    def unload(self) -> None:
        import requests

        try:
            # keep_alive=0 evicts immediately -- essential on 8 GB, otherwise
            # Ollama holds the previous model resident for 5 minutes by default
            # and the next load swaps.
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model_path, "prompt": "", "keep_alive": 0},
                timeout=120,
            )
        except Exception:
            pass
        self._loaded = False

    def version(self) -> str:
        import requests

        try:
            r = requests.get(f"{self.host}/api/version", timeout=10)
            return r.json().get("version", "")
        except Exception:
            return ""

    def weight_bytes(self) -> int:
        import requests

        try:
            r = requests.post(
                f"{self.host}/api/show", json={"model": self.model_path}, timeout=30
            )
            return int(r.json().get("size", 0) or 0)
        except Exception:
            return 0

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0
    ) -> GenResult:
        import requests

        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        t = _Timer()
        chunks: list[str] = []
        prompt_tokens = 0
        try:
            with requests.post(
                f"{self.host}/api/generate", json=payload, stream=True, timeout=1800
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    piece = d.get("response", "")
                    if piece:
                        t.tick()
                        chunks.append(piece)
                    if d.get("done"):
                        prompt_tokens = int(d.get("prompt_eval_count", 0) or 0)
        except Exception as e:
            return GenResult(
                text="".join(chunks), total_s=t.done(), ttft_s=t.ttft, error=repr(e)
            )

        return GenResult(
            text="".join(chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=t.n,
            ttft_s=t.ttft,
            total_s=t.done(),
        )
