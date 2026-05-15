"""Tests for model wrappers (mocked — no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prompt_probing.models import load_model
from prompt_probing.models.base import BaseMLLM
from prompt_probing.models.llava import LLaVA
from prompt_probing.models.instructblip import InstructBLIP
from prompt_probing.models.gpt4v import GPT4V


class TestLoadModel:
    def test_load_llava_7b(self):
        model = load_model("llava-1.5-7b")
        assert isinstance(model, LLaVA)

    def test_load_llava_13b(self):
        model = load_model("llava-1.5-13b")
        assert isinstance(model, LLaVA)

    def test_load_instructblip_7b(self):
        model = load_model("instructblip-vicuna-7b")
        assert isinstance(model, InstructBLIP)

    def test_load_gpt4v(self):
        model = load_model("gpt-4v")
        assert isinstance(model, GPT4V)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent-model-xyz")

    def test_case_insensitive(self):
        model = load_model("LLaVA-1.5-7B")
        assert isinstance(model, LLaVA)


class TestBaseMLLM:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseMLLM(model_name="test")  # type: ignore[abstract]

    def test_load_image_from_pil(self):
        from PIL import Image

        class ConcreteModel(BaseMLLM):
            def load(self): pass
            def generate(self, prompt, image=None, **kw): return ""

        model = ConcreteModel(model_name="test")
        img = Image.new("RGB", (10, 10))
        loaded = model._load_image(img)
        assert loaded is img  # same object returned


class TestLLaVA:
    def test_repr(self):
        model = LLaVA(model_name="llava-1.5-7b")
        assert "LLaVA" in repr(model)
        assert "llava-1.5-7b" in repr(model)

    def test_generate_raises_without_transformers(self):
        model = LLaVA(model_name="llava-1.5-7b", device="cpu")
        # Should raise ImportError if transformers is not available
        with patch("builtins.__import__", side_effect=ImportError("no transformers")):
            with pytest.raises(ImportError):
                model.load()


class TestGPT4V:
    def test_image_to_base64_roundtrip(self):
        import base64
        from PIL import Image
        from io import BytesIO

        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        b64 = GPT4V._image_to_base64(img)
        decoded = base64.b64decode(b64)
        restored = Image.open(BytesIO(decoded))
        assert restored.size == (4, 4)

    def test_generate_calls_openai_client(self):
        model = GPT4V(model_name="gpt-4-vision-preview", api_key="fake-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Test response  "
        mock_client.chat.completions.create.return_value = mock_response
        model._client = mock_client

        result = model.generate("Hello, what do you see?")
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()
