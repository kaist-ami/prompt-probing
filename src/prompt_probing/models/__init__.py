"""MLLM model wrappers for prompt probing."""

from .base import BaseMLLM
from .llava import LLaVA
from .instructblip import InstructBLIP
from .gpt4v import GPT4V

__all__ = ["BaseMLLM", "LLaVA", "InstructBLIP", "GPT4V"]


def load_model(model_name: str, **kwargs) -> BaseMLLM:
    """Factory function to load a model by name.

    Args:
        model_name: One of 'llava-1.5-7b', 'llava-1.5-13b',
                    'instructblip-vicuna-7b', 'instructblip-vicuna-13b',
                    'gpt-4v', 'gemini-pro-vision', 'claude-3-opus', etc.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An instantiated BaseMLLM subclass.

    Raises:
        ValueError: If ``model_name`` is not recognised.
    """
    registry: dict[str, type[BaseMLLM]] = {
        "llava-1.5-7b": LLaVA,
        "llava-1.5-13b": LLaVA,
        "instructblip-vicuna-7b": InstructBLIP,
        "instructblip-vicuna-13b": InstructBLIP,
        "gpt-4v": GPT4V,
    }
    model_name_lower = model_name.lower()
    if model_name_lower not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {sorted(registry)}"
        )
    return registry[model_name_lower](model_name=model_name, **kwargs)
