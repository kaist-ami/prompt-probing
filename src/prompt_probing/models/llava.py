"""LLaVA model wrapper for prompt probing."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from .base import BaseMLLM, ImageInput

# HuggingFace model IDs for each variant
_MODEL_IDS: dict[str, str] = {
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
}


class LLaVA(BaseMLLM):
    """Wrapper for LLaVA-1.5 models (7B and 13B variants).

    Requires the ``transformers`` and ``torch`` packages.  On first call to
    :meth:`load` the model weights are downloaded from HuggingFace Hub.

    Args:
        model_name: One of ``"llava-1.5-7b"`` or ``"llava-1.5-13b"``.
        device: PyTorch device string.
        max_new_tokens: Upper bound on generated token count.
        load_in_4bit: Whether to load the model in 4-bit quantisation
            (requires ``bitsandbytes``).
    """

    def __init__(
        self,
        model_name: str = "llava-1.5-7b",
        device: str = "cuda",
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load LLaVA weights and processor from HuggingFace Hub."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required to use LLaVA. "
                "Install them with:  pip install transformers torch"
            ) from exc

        hf_id = _MODEL_IDS.get(self.model_name.lower(), _MODEL_IDS["llava-1.5-7b"])
        load_kwargs: dict = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, device_map=self.device, torch_dtype=torch.float16, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(hf_id)

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image: ImageInput | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Run inference with LLaVA.

        Args:
            prompt: Text prompt (may include ``<image>`` token).
            image: Optional image input.
            temperature: Sampling temperature (0 → greedy decoding).
            **kwargs: Forwarded to ``model.generate``.

        Returns:
            Decoded text response.
        """
        if self._model is None or self._processor is None:
            self.load()

        import torch

        pil_image = self._load_image(image) if image is not None else None
        inputs = self._processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = temperature > 0.0
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                **kwargs,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()
