"""Abstract base class for all MLLM wrappers."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Union

from PIL import Image


ImageInput = Union[str, Path, Image.Image]


class BaseMLLM(abc.ABC):
    """Abstract interface for Multimodal Large Language Models.

    All model-specific subclasses must implement :meth:`generate` so that the
    rest of the probing framework can interact with any MLLM through a single,
    consistent API.

    Args:
        model_name: Human-readable identifier for the model variant
            (e.g. ``"llava-1.5-7b"``).
        device: PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        max_new_tokens: Maximum number of tokens to generate per call.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        image: ImageInput | None = None,
        **kwargs,
    ) -> str:
        """Generate a text response given a prompt and optional image.

        Args:
            prompt: The text prompt to send to the model.
            image: An optional image (file path or PIL Image).
            **kwargs: Additional generation parameters (temperature, top_p, …).

        Returns:
            The model's text response as a string.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _load_image(self, image: ImageInput) -> Image.Image:
        """Normalise *image* to a :class:`PIL.Image.Image` instance."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(image).convert("RGB")

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(model_name={self.model_name!r}, device={self.device!r})"
