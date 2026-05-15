#!/usr/bin/env python3
"""Main script for running zero-shot rankability probing experiments.

Example usage::

    python scripts/run_probe.py \\
        --model llava-1.5-7b \\
        --dataset things \\
        --data-root data/things \\
        --attribute size \\
        --prompt-type ordinal \\
        --output results/llava_things_size.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run zero-shot ordinal prompt probing on an MLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. llava-1.5-7b, gpt-4v).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["things", "winoground"],
        help="Benchmark dataset to probe.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--attribute",
        required=True,
        help="Ordinal attribute to probe (e.g. size, weight).",
    )
    parser.add_argument(
        "--prompt-type",
        default="ordinal",
        choices=["direct", "ordinal", "contrastive", "chain"],
        help="Prompt family to use.",
    )
    parser.add_argument(
        "--prompt-subtype",
        default="single",
        help="Prompt subtype within the chosen family.",
    )
    parser.add_argument(
        "--metric",
        default="kendall_tau",
        choices=["kendall_tau", "spearman_rho"],
        help="Rank correlation metric for evaluation.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device (cuda / cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per query.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model)
    try:
        from prompt_probing.models import load_model

        model = load_model(args.model, device=args.device, max_new_tokens=args.max_new_tokens)
        model.load()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    logger.info(
        "Loading dataset '%s' (attribute=%s, split=%s) from %s",
        args.dataset,
        args.attribute,
        args.split,
        args.data_root,
    )
    try:
        from prompt_probing.data import load_dataset

        dataset = load_dataset(
            name=args.dataset,
            root=args.data_root,
            attribute=args.attribute,
            split=args.split,
        )
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    logger.info("Dataset loaded: %d samples", len(dataset))

    # ------------------------------------------------------------------
    # Run probing
    # ------------------------------------------------------------------
    logger.info(
        "Running probing with prompt_type=%s subtype=%s",
        args.prompt_type,
        args.prompt_subtype,
    )

    from prompt_probing.ranking import RankabilityScorer
    from prompt_probing.prompts.templates import PromptType

    scorer = RankabilityScorer(model=model, attribute=args.attribute)
    images = [s.image for s in dataset]
    ground_truths = dataset.labels

    results = scorer.evaluate(
        images=images,
        ground_truths=ground_truths,
        prompt_type=PromptType(args.prompt_type),
        subtype=args.prompt_subtype,
        metric=args.metric,
    )

    logger.info(
        "Rankability Score (%s): %.4f  (valid=%d/%d)",
        args.metric,
        results["rankability_score"],
        results["n_valid"],
        results["n_total"],
    )

    scorer.save_results(results, args.output)
    logger.info("Results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
