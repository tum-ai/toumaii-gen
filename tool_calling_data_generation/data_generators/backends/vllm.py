import logging
import os
from typing import List

from vllm import LLM, SamplingParams

from tool_calling_data_generation.data_generators.backends.base import BaseGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
from dotenv import load_dotenv

load_dotenv()

enable_gpu_brrr = os.getenv("GPU_GO_BRRR", "False").lower() == "true"
if os.getenv("GPU_GO_BRRR", "False").lower() == "true":
    print("🔥🔥🔥 vLLM Preferences are overriden, GPU_GO_BRRR is true. 🔥🔥🔥")

import torch
import gc

logging.getLogger("vllm").setLevel(logging.ERROR)


class vLLMGenerator(BaseGenerator):
    def initialize_backend(self, **kwargs) -> None:
        turbo_mode = kwargs.get("turbo_mode", "False")
        if turbo_mode.lower() == "true" or enable_gpu_brrr:
            gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.95)
            model_ctx = kwargs.get("model_ctx", 16384)
            self._llm = LLM(
                self.model_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=model_ctx,
                enforce_eager=True,
                max_num_batched_tokens=131072,
                enable_chunked_prefill=False,
                max_num_seqs=2048,
                tensor_parallel_size=1,
            )
            self.is_backend_loaded = True
            self.tokenizer = self._llm.get_tokenizer()
        else:
            gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)
            model_ctx = kwargs.get("model_ctx", 16384)
            self._llm = LLM(
                self.model_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=model_ctx,
            )
            self.is_backend_loaded = True
            self.tokenizer = self._llm.get_tokenizer()

    @property
    def LLM(self):
        """Backward-compat alias for code that references ``backend.LLM``."""
        try:
            return self._llm
        except AttributeError:
            raise AttributeError(
                "vLLMGenerator.LLM is only available after initialize_backend()"
            ) from None

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self._llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def unload_backend(self) -> None:
        """Unload the model and free GPU/CPU resources.

        Uses vLLM's sleep mode (level 2) to discard all GPU memory
        (weights + KV cache) across all worker processes, then drops
        Python references and forces GC.
        """
        if hasattr(self, "_llm"):
            try:
                self._llm.sleep(level=2)
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "vLLM sleep(level=2) failed; falling back to del", exc_info=True
                )
            del self._llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


