#!/usr/bin/env python3
"""Aggregate and display evaluation results from probing experiments.

Example usage::

    python scripts/evaluate.py \\
        --results results/llava_things_size.json \\
        --metric kendall_tau spearman_rho
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate rankability scores from probing experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        type=Path,
        help="One or more JSON result files produced by run_probe.py.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["kendall_tau"],
        choices=["kendall_tau", "spearman_rho"],
        help="Rank correlation metric(s) to report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a summary JSON.",
    )
    return parser


def load_result(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_file(result: dict, metrics: list[str]) -> dict[str, float]:
    from prompt_probing.ranking.metrics import kendall_tau, spearman_rho

    predicted = [r["predicted_score"] for r in result.get("probe_results", [])]
    ground_truths = [r["ground_truth_score"] for r in result.get("probe_results", [])]

    # Filter out None predictions
    pairs = [(p, g) for p, g in zip(predicted, ground_truths) if p is not None]
    if not pairs:
        return {m: float("nan") for m in metrics}

    pred_valid, gt_valid = zip(*pairs)
    scores: dict[str, float] = {}
    for metric in metrics:
        fn = {"kendall_tau": kendall_tau, "spearman_rho": spearman_rho}[metric]
        scores[metric] = fn(list(pred_valid), list(gt_valid))
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary: list[dict] = []

    for path in args.results:
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue

        result = load_result(path)
        scores = evaluate_file(result, args.metric)
        entry = {"file": str(path), **scores}
        summary.append(entry)

        score_str = "  ".join(f"{m}={v:.4f}" for m, v in scores.items())
        logger.info("%s  →  %s", path.name, score_str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Summary saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
