"""High-level prompt builder for ordinal probing experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .templates import PROMPT_TEMPLATES, PromptType


@dataclass
class ProbePrompt:
    """A fully rendered prompt ready to be sent to an MLLM.

    Attributes:
        text: The rendered prompt string.
        prompt_type: The :class:`PromptType` family this prompt belongs to.
        attribute: The ordinal attribute being probed.
        subtype: Template subtype within the family (e.g. ``"pairwise"``).
        metadata: Arbitrary key-value pairs for bookkeeping.
    """

    text: str
    prompt_type: PromptType
    attribute: str
    subtype: str
    metadata: dict = field(default_factory=dict)


class PromptBuilder:
    """Build ordinal probe prompts for a given attribute and prompt type.

    Example::

        builder = PromptBuilder(attribute="size")
        prompt = builder.build(PromptType.ORDINAL, subtype="single")
        print(prompt.text)

    Args:
        attribute: The ordinal attribute to probe (e.g. ``"size"``,
            ``"weight"``, ``"brightness"``).
    """

    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    # ------------------------------------------------------------------

    def build(
        self,
        prompt_type: PromptType | str,
        subtype: str = "pairwise",
        **format_kwargs,
    ) -> ProbePrompt:
        """Render a single prompt template.

        Args:
            prompt_type: The :class:`PromptType` (or its string value) to use.
            subtype: Template subtype within the family.
            **format_kwargs: Additional keyword arguments forwarded to
                ``str.format`` (e.g. ``item_a="cat"``, ``item_b="dog"``).

        Returns:
            A :class:`ProbePrompt` instance with the rendered text.

        Raises:
            KeyError: If *prompt_type* or *subtype* is not found in
                :data:`PROMPT_TEMPLATES`.
        """
        if isinstance(prompt_type, str):
            prompt_type = PromptType(prompt_type)

        template = PROMPT_TEMPLATES[prompt_type][subtype]
        rendered = template.format(attribute=self.attribute, **format_kwargs)
        return ProbePrompt(
            text=rendered,
            prompt_type=prompt_type,
            attribute=self.attribute,
            subtype=subtype,
        )

    # ------------------------------------------------------------------

    def build_all(
        self,
        subtypes: Sequence[str] | None = None,
        **format_kwargs,
    ) -> list[ProbePrompt]:
        """Build prompts for every :class:`PromptType` (and given subtypes).

        Args:
            subtypes: Optional list of subtype names to build for each type.
                Defaults to all subtypes defined for each type.
            **format_kwargs: Forwarded to :meth:`build`.

        Returns:
            A flat list of :class:`ProbePrompt` instances.
        """
        prompts: list[ProbePrompt] = []
        for prompt_type, subtype_map in PROMPT_TEMPLATES.items():
            for st in subtype_map:
                if subtypes is None or st in subtypes:
                    try:
                        prompts.append(self.build(prompt_type, st, **format_kwargs))
                    except (KeyError, IndexError):
                        continue
        return prompts
