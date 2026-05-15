"""Prompt template definitions for ordinal probing."""

from __future__ import annotations

from enum import Enum


class PromptType(str, Enum):
    """Enumeration of supported ordinal prompt families."""

    DIRECT = "direct"
    """Direct binary comparison: 'Which is larger, A or B?'"""

    ORDINAL = "ordinal"
    """Scale-grounded query: 'On a scale of 1–10, how X is this?'"""

    CONTRASTIVE = "contrastive"
    """Paired contrastive ranking over a list of items."""

    CHAIN = "chain"
    """Chain-of-thought guided ordinal reasoning."""


# ---------------------------------------------------------------------------
# Template strings
# Each template may contain the following placeholders:
#   {attribute}  — the ordinal attribute being probed (e.g. "size", "weight")
#   {item_a}     — first item / description
#   {item_b}     — second item / description
#   {items}      — comma-separated list of items (for list prompts)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[PromptType, dict[str, str]] = {
    PromptType.DIRECT: {
        "pairwise": (
            "Look at the two images provided.\n"
            "Which object has greater {attribute}: the one on the left or the one on the right?\n"
            "Answer with exactly one of: 'left', 'right', or 'equal'."
        ),
        "single": (
            "Look at the image.\n"
            "Compared to a typical everyday object, what is the {attribute} of this object?\n"
            "Answer with one word: 'very low', 'low', 'medium', 'high', or 'very high'."
        ),
    },
    PromptType.ORDINAL: {
        "pairwise": (
            "You will see two images.\n"
            "On a scale from 1 (lowest) to 10 (highest), rate the {attribute} of each object.\n"
            "First image score: <score_1>\n"
            "Second image score: <score_2>\n"
            "Respond in the exact format:\n"
            "Score 1: X\nScore 2: Y"
        ),
        "single": (
            "Look at the image.\n"
            "On a scale from 1 (lowest {attribute}) to 10 (highest {attribute}), "
            "rate the {attribute} of the main object.\n"
            "Respond with a single integer between 1 and 10."
        ),
    },
    PromptType.CONTRASTIVE: {
        "list": (
            "You are given images of the following objects: {items}.\n"
            "Rank them from lowest to highest {attribute}.\n"
            "Respond with a comma-separated list of the object names in order."
        ),
        "pairwise": (
            "Compare {item_a} and {item_b}.\n"
            "Which has higher {attribute}?\n"
            "Answer with exactly one of: '{item_a}', '{item_b}', or 'equal'."
        ),
    },
    PromptType.CHAIN: {
        "pairwise": (
            "Think step by step about the relative {attribute} of the two objects shown.\n"
            "Step 1: Describe the {attribute} of the first object.\n"
            "Step 2: Describe the {attribute} of the second object.\n"
            "Step 3: Compare them and state which has higher {attribute}.\n"
            "Final answer (one of 'left', 'right', 'equal'):"
        ),
        "single": (
            "Think step by step to estimate the {attribute} of the object in the image.\n"
            "Step 1: Identify the object.\n"
            "Step 2: Recall typical values of {attribute} for this object.\n"
            "Step 3: Rate it on a 1–10 scale.\n"
            "Final score (integer 1–10):"
        ),
    },
}
