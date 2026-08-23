from abc import abstractmethod
from typing import List


class BaseGenerator:
    """
    Base class for all data generators. Multiple Generators classes are
    possible: vLLM, plain HF/TRL, SGLang, and others.

    The main goal is to abstract the generation loop from the rest of the
    codebase so that the DPO pipeline works identically regardless of backend.
    """

    def __init__(self, model_id: str):
        self.model_id: str = model_id
        self.tokenizer = None  # set by initialize_backend
        self.is_backend_loaded: bool = False

    @abstractmethod
    def initialize_backend(self, **kwargs) -> None:
        """Load model & tokenizer.  Populates ``self.tokenizer``."""
        ...

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate one completion per prompt.

        Returns:
            A ``List[str]`` the same length as *prompts*.
        """
        ...
        pass

    @abstractmethod
    def unload_backend(self) -> None:
        """Unload the model and free GPU/CPU resources.

        This should handle any backend-specific cleanup such as:
        - Clearing CUDA cache
        - Deleting model references
        - Shutting down inference servers (e.g. vLLM, SGLang)
        """
        ...
        pass


