"""
TRL / HuggingFace Transformers generation backend.

Uses ``model.generate()`` directly — no vLLM dependency.  This is the
default backend for environments where vLLM is unavailable (Apple MLX,
AMD, CPU-only, etc.).

Usage::

    backend = TRLGenerator("Qwen/Qwen2.5-1.5B-Instruct")
    backend.initialize_backend(device_map="auto")
    outputs: List[str] = backend.generate(prompts, temperature=0.7, max_tokens=512)
"""

from __future__ import annotations

import logging
from typing import List

import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_calling_data_generation.data_generators.backends.base import BaseGenerator

logger = logging.getLogger(__name__)


class TRLGenerator(BaseGenerator):
    """Generate completions with plain HuggingFace Transformers.

    Parameters
    ----------
    model_id:
        HuggingFace model name or path (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
    """

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def initialize_backend(self, **kwargs) -> None:
        """Load the model and tokeniser.

        Keyword Args:
            device_map: passed to ``from_pretrained`` (default ``"auto"`` —
                works for CUDA / MPS / CPU).
            torch_dtype: default ``torch.bfloat16`` when CUDA is available,
                otherwise ``torch.float32``.
            model_kwargs: extra kwargs forwarded to
                ``AutoModelForCausalLM.from_pretrained``.
            tokenizer_kwargs: extra kwargs forwarded to
                ``AutoTokenizer.from_pretrained``.
        """
        device_map = kwargs.get("device_map", "auto")
        model_kwargs: dict = kwargs.get("model_kwargs", {})
        tokenizer_kwargs: dict = kwargs.get("tokenizer_kwargs", {})

        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info(
            "Loading model %s (device_map=%s, dtype=%s) …",
            self.model_id,
            device_map,
            torch_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        # Decoder-only models MUST use left-padding for correct generation
        self.tokenizer.padding_side = "left"

        # Ensure pad_token is set (required for batched generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_backend_loaded = True
        logger.info("TRL backend ready.  Model: %s", self.model_id)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate one completion per prompt.

        Keyword Args:
            temperature: sampling temperature (default 0.7).
            max_tokens: maximum new tokens to generate (default 512).
            top_p: nucleus sampling (default 0.95).
            batch_size: number of prompts to process at once (default 8).
            gen_kwargs: extra kwargs forwarded to ``model.generate()``.
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        top_p = kwargs.get("top_p", 0.95)
        batch_size: int = kwargs.get("batch_size", 8)
        extra_gen: dict = kwargs.get("gen_kwargs", {})

        # Mutate the model's own generation_config so no flags are
        # rejected as "not valid" (transformers 5.x compat).
        gc = self.model.generation_config
        gc.max_new_tokens = max_tokens
        gc.temperature = temperature
        gc.top_p = top_p
        gc.do_sample = temperature > 0
        gc.pad_token_id = self.tokenizer.pad_token_id
        gc.eos_token_id = self.tokenizer.eos_token_id
        for k, v in extra_gen.items():
            setattr(gc, k, v)

        all_outputs: List[str] = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(prompts), batch_size),
            total=num_batches,
            desc="TRL generation",
            unit="batch",
        ):
            batch = prompts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(**inputs)

            # Decode only the generated part (strip the prompt)
            prompt_lengths = inputs.input_ids.shape[1]
            completions = self.tokenizer.batch_decode(
                generated[:, prompt_lengths:], skip_special_tokens=True
            )

            all_outputs.extend([c.strip() for c in completions])

        return all_outputs

    def unload_backend(self) -> None:
        """Unload the model and free GPU/CPU resources."""
        if hasattr(self, "model"):
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


