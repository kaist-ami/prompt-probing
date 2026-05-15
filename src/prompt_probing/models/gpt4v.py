"""GPT-4V (and compatible OpenAI vision models) wrapper for prompt probing."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

from .base import BaseMLLM, ImageInput


class GPT4V(BaseMLLM):
    """Wrapper for OpenAI's GPT-4V (and GPT-4o) vision API.

    Authentication is handled via the ``OPENAI_API_KEY`` environment variable
    or by passing ``api_key`` directly.

    Args:
        model_name: OpenAI model identifier, e.g. ``"gpt-4-vision-preview"``
            or ``"gpt-4o"``.
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when not provided.
        max_new_tokens: Maximum tokens to generate.
        device: Unused for API-based models; kept for interface consistency.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        api_key: str | None = None,
        max_new_tokens: int = 256,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use GPT-4V. "
                "Install it with:  pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Encode a PIL image to a base64 data-URL string."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image: ImageInput | None = None,
        temperature: float = 0.0,
        detail: str = "high",
        **kwargs,
    ) -> str:
        """Call the OpenAI vision API.

        Args:
            prompt: Text prompt.
            image: Optional image (file path or PIL Image).
            temperature: Sampling temperature.
            detail: OpenAI image detail level (``"low"`` or ``"high"``).
            **kwargs: Forwarded to the chat completions API.

        Returns:
            The model's text response.
        """
        if self._client is None:
            self.load()

        content: list[dict] = [{"type": "text", "text": prompt}]

        if image is not None:
            pil_image = self._load_image(image)
            b64 = self._image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": detail,
                    },
                }
            )

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content.strip()
