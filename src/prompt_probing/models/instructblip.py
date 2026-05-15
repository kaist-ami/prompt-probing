"""InstructBLIP model wrapper for prompt probing."""

from __future__ import annotations

from .base import BaseMLLM, ImageInput

_MODEL_IDS: dict[str, str] = {
    "instructblip-vicuna-7b": "Salesforce/instructblip-vicuna-7b",
    "instructblip-vicuna-13b": "Salesforce/instructblip-vicuna-13b",
}


class InstructBLIP(BaseMLLM):
    """Wrapper for InstructBLIP (Vicuna-7B and Vicuna-13B variants).

    Args:
        model_name: One of ``"instructblip-vicuna-7b"`` or
            ``"instructblip-vicuna-13b"``.
        device: PyTorch device string.
        max_new_tokens: Upper bound on generated token count.
    """

    def __init__(
        self,
        model_name: str = "instructblip-vicuna-7b",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load InstructBLIP weights and processor from HuggingFace Hub."""
        try:
            from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required to use InstructBLIP. "
                "Install them with:  pip install transformers torch"
            ) from exc

        hf_id = _MODEL_IDS.get(
            self.model_name.lower(), _MODEL_IDS["instructblip-vicuna-7b"]
        )
        self._model = InstructBlipForConditionalGeneration.from_pretrained(
            hf_id, device_map=self.device, torch_dtype=torch.float16
        )
        self._processor = InstructBlipProcessor.from_pretrained(hf_id)

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image: ImageInput | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Run inference with InstructBLIP.

        Args:
            prompt: Text instruction for the model.
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
        inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt")
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
        return self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
