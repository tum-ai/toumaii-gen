"""Generation pipeline, loaders, backends, extractors, and canonical models."""

__all__ = ["build_dpo_dataset"]


def __getattr__(name):
    """Avoid importing Transformers/Torch when consumers only need loaders."""
    if name == "build_dpo_dataset":
        from tool_calling_data_generation.data_generators.dpo_generator import (
            build_dpo_dataset,
        )

        return build_dpo_dataset
    raise AttributeError(name)

