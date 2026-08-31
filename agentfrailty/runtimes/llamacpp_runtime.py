"""
llama.cpp adapter via llama-cpp-python (GGUF weights, Metal backend).

Note for the writeup: llama.cpp's Q4_K_M and MLX's 4-bit are different
quantization algorithms, not two implementations of one thing. K-quants use
mixed per-block precision; MLX uses uniform group-wise affine quantization.
Comparing them at matched *nominal* bit-width is one of the questions this
benchmark exists to answer -- it is not an apples-to-apples assumption.
"""

from __future__ import annotations

import gc
import os
from typing import Optional

from .base import Runtime, _Timer
from ..schema import GenResult


class LlamaCppRuntime(Runtime):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = offload everything to Metal
        **kw,
    ):
        super().__init__(model_path, **kw)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            logits_all=False,
        )
        self._loaded = True

    def unload(self) -> None:
        try:
            if self._llm is not None:
                self._llm.close()
        except Exception:
            pass
        self._llm = None
        self._loaded = False
        gc.collect()

    def version(self) -> str:
        try:
            import llama_cpp

            return getattr(llama_cpp, "__version__", "")
        except Exception:
            return ""

    def weight_bytes(self) -> int:
        try:
            return os.path.getsize(self.model_path)
        except OSError:
            return 0

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0
    ) -> GenResult:
        t = _Timer()
        chunks: list[str] = []
        prompt_tokens = 0
        try:
            # create_chat_completion applies the GGUF's embedded chat template,
            # which is what an actual llama.cpp user gets.
            stream = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                stream=True,
            )
            for part in stream:
                delta = part["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    t.tick()
                    chunks.append(piece)
            try:
                prompt_tokens = int(self._llm.n_tokens)
            except Exception:
                prompt_tokens = 0
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
