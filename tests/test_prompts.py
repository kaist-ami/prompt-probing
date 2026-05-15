"""Tests for prompt builder and templates."""

from __future__ import annotations

import pytest

from prompt_probing.prompts.builder import PromptBuilder, ProbePrompt
from prompt_probing.prompts.templates import PromptType, PROMPT_TEMPLATES


class TestPromptType:
    def test_all_types_present(self):
        assert set(PromptType) == {
            PromptType.DIRECT,
            PromptType.ORDINAL,
            PromptType.CONTRASTIVE,
            PromptType.CHAIN,
        }

    def test_string_values(self):
        assert PromptType.DIRECT.value == "direct"
        assert PromptType.ORDINAL.value == "ordinal"
        assert PromptType.CONTRASTIVE.value == "contrastive"
        assert PromptType.CHAIN.value == "chain"


class TestPromptTemplates:
    def test_all_types_have_templates(self):
        for pt in PromptType:
            assert pt in PROMPT_TEMPLATES, f"Missing templates for {pt}"

    def test_templates_are_strings(self):
        for pt, subtypes in PROMPT_TEMPLATES.items():
            for subtype, template in subtypes.items():
                assert isinstance(template, str), (
                    f"Template {pt}/{subtype} is not a string"
                )

    def test_ordinal_contains_attribute_placeholder(self):
        for subtype, template in PROMPT_TEMPLATES[PromptType.ORDINAL].items():
            assert "{attribute}" in template, (
                f"ordinal/{subtype} template missing {{attribute}} placeholder"
            )


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder(attribute="size")

    def test_build_returns_probe_prompt(self):
        prompt = self.builder.build(PromptType.ORDINAL, subtype="single")
        assert isinstance(prompt, ProbePrompt)

    def test_build_fills_attribute(self):
        prompt = self.builder.build(PromptType.DIRECT, subtype="single")
        assert "size" in prompt.text

    def test_build_pairwise_with_item_kwargs(self):
        prompt = self.builder.build(
            PromptType.CONTRASTIVE,
            subtype="pairwise",
            item_a="cat",
            item_b="elephant",
        )
        assert "cat" in prompt.text
        assert "elephant" in prompt.text

    def test_build_from_string_type(self):
        prompt = self.builder.build("chain", subtype="single")
        assert prompt.prompt_type == PromptType.CHAIN

    def test_build_invalid_type_raises(self):
        with pytest.raises(ValueError):
            self.builder.build("nonexistent_type", subtype="single")

    def test_build_invalid_subtype_raises(self):
        with pytest.raises(KeyError):
            self.builder.build(PromptType.ORDINAL, subtype="nonexistent")

    def test_build_all_returns_list(self):
        prompts = self.builder.build_all()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_build_all_filtered_subtype(self):
        prompts = self.builder.build_all(subtypes=["single"])
        for p in prompts:
            assert p.subtype == "single"

    def test_probe_prompt_attributes(self):
        prompt = self.builder.build(PromptType.ORDINAL, subtype="single")
        assert prompt.attribute == "size"
        assert prompt.prompt_type == PromptType.ORDINAL
        assert prompt.subtype == "single"
        assert isinstance(prompt.metadata, dict)
