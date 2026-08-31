"""MLX adapter (mlx-lm). Apple's own framework; the speed leader in prior work."""

from __future__ import annotations

import gc
import os
from typing import Optional

from .base import Runtime, _Timer
from ..schema import GenResult


class MlxRuntime(Runtime):
    name = "mlx"

    def __init__(self, model_path: str, **kw):
        super().__init__(model_path, **kw)
        self._model = None
        self._tok = None

    def load(self) -> None:
        from mlx_lm import load as mlx_load  # imported lazily: macOS-only dep

        self._model, self._tok = mlx_load(self.model_path)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tok = None
        self._loaded = False
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()  # returns Metal buffers; without this the next
            # model load can push an 8 GB machine into swap
        except Exception:
            pass

    def version(self) -> str:
        try:
            import mlx_lm

            return getattr(mlx_lm, "__version__", "")
        except Exception:
            return ""

    def weight_bytes(self) -> int:
        total = 0
        p = self.model_path
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith((".safetensors", ".npz", ".bin")):
                        total += os.path.getsize(os.path.join(root, f))
        return total

    def _apply_template(self, prompt: str) -> str:
        tok = self._tok
        tmpl = getattr(tok, "chat_template", None)
        if tmpl:
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0
    ) -> GenResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        mx.random.seed(seed)
        text = self._apply_template(prompt)
        sampler = make_sampler(temp=temperature)

        t = _Timer()
        chunks: list[str] = []
        prompt_tokens = 0
        try:
            for resp in stream_generate(
                self._model,
                self._tok,
                text,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                t.tick()
                chunks.append(resp.text)
                prompt_tokens = getattr(resp, "prompt_tokens", prompt_tokens) or 0
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
