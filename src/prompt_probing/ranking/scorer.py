"""High-level RankabilityScorer that ties models, prompts, and metrics together."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..models.base import BaseMLLM, ImageInput
from ..prompts.builder import PromptBuilder, ProbePrompt
from ..prompts.templates import PromptType
from .metrics import kendall_tau, spearman_rho, rankability_score


@dataclass
class ProbeResult:
    """Result of a single probe (one prompt × one instance).

    Attributes:
        prompt: The :class:`~prompt_probing.prompts.builder.ProbePrompt` used.
        response: Raw text response from the MLLM.
        predicted_score: Numeric score parsed from the response (``None`` if
            parsing failed).
        ground_truth_score: Reference ordinal label for this instance.
        metadata: Arbitrary extra info (item name, dataset, …).
    """

    prompt: ProbePrompt
    response: str
    predicted_score: float | None
    ground_truth_score: float
    metadata: dict = field(default_factory=dict)


class RankabilityScorer:
    """Run prompt probing experiments and aggregate rankability scores.

    Args:
        model: An initialised :class:`~prompt_probing.models.base.BaseMLLM`
            instance.
        attribute: The ordinal attribute to evaluate (e.g. ``"size"``).
        prompt_types: Which :class:`~prompt_probing.prompts.templates.PromptType`
            families to include.  Defaults to all four types.
    """

    def __init__(
        self,
        model: BaseMLLM,
        attribute: str,
        prompt_types: Sequence[PromptType] | None = None,
    ) -> None:
        self.model = model
        self.attribute = attribute
        self.prompt_types = list(prompt_types or PromptType)
        self._builder = PromptBuilder(attribute=attribute)

    # ------------------------------------------------------------------

    def probe_single(
        self,
        image: ImageInput,
        ground_truth: float,
        prompt_type: PromptType = PromptType.ORDINAL,
        subtype: str = "single",
        **prompt_kwargs,
    ) -> ProbeResult:
        """Probe a single image and return a :class:`ProbeResult`.

        Args:
            image: Image to probe.
            ground_truth: Ground-truth ordinal value for this image.
            prompt_type: Which prompt family to use.
            subtype: Template subtype.
            **prompt_kwargs: Forwarded to the prompt template formatter.

        Returns:
            A :class:`ProbeResult` with the raw response and parsed score.
        """
        prompt = self._builder.build(prompt_type, subtype, **prompt_kwargs)
        response = self.model.generate(prompt=prompt.text, image=image)
        predicted = self._parse_score(response)
        return ProbeResult(
            prompt=prompt,
            response=response,
            predicted_score=predicted,
            ground_truth_score=ground_truth,
        )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        images: Sequence[ImageInput],
        ground_truths: Sequence[float],
        prompt_type: PromptType = PromptType.ORDINAL,
        subtype: str = "single",
        metric: str = "kendall_tau",
        **prompt_kwargs,
    ) -> dict[str, Any]:
        """Evaluate rankability over a collection of images.

        Args:
            images: Sequence of image inputs.
            ground_truths: Corresponding ordinal labels.
            prompt_type: Prompt family to use.
            subtype: Template subtype.
            metric: Correlation metric (``"kendall_tau"`` or ``"spearman_rho"``).
            **prompt_kwargs: Forwarded to the prompt builder.

        Returns:
            A dict with keys ``"rankability_score"``, ``"results"``,
            ``"predicted_scores"``, and ``"ground_truths"``.
        """
        results: list[ProbeResult] = []
        for image, gt in zip(images, ground_truths):
            result = self.probe_single(
                image, gt, prompt_type=prompt_type, subtype=subtype, **prompt_kwargs
            )
            results.append(result)

        predicted = [r.predicted_score for r in results]
        valid_mask = [p is not None for p in predicted]
        valid_pred = [p for p in predicted if p is not None]
        valid_gt = [
            gt for gt, valid in zip(ground_truths, valid_mask) if valid
        ]

        rs = rankability_score(
            [[p] for p in valid_pred],
            [[g] for g in valid_gt],
            metric=metric,
        ) if len(valid_pred) > 1 else float("nan")

        return {
            "rankability_score": rs,
            "metric": metric,
            "n_valid": len(valid_pred),
            "n_total": len(results),
            "predicted_scores": predicted,
            "ground_truths": list(ground_truths),
            "results": results,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> float | None:
        """Extract the first numeric value from a model response.

        Args:
            response: Raw text response from the model.

        Returns:
            The first float found, or ``None`` if no number is present.
        """
        import re

        match = re.search(r"[-+]?\d*\.?\d+", response)
        if match:
            return float(match.group())
        return None

    # ------------------------------------------------------------------

    def save_results(self, results: dict[str, Any], output_path: str | Path) -> None:
        """Serialise evaluation results to a JSON file.

        Args:
            results: Output dict from :meth:`evaluate`.
            output_path: Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            k: v
            for k, v in results.items()
            if k != "results"  # ProbeResult objects are not JSON-serialisable
        }
        serialisable["probe_results"] = [
            {
                "response": r.response,
                "predicted_score": r.predicted_score,
                "ground_truth_score": r.ground_truth_score,
                "prompt_type": r.prompt.prompt_type.value,
                "subtype": r.prompt.subtype,
                "metadata": r.metadata,
            }
            for r in results.get("results", [])
        ]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)
