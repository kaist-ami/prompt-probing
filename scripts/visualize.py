#!/usr/bin/env python3
"""Visualise rankability results across models, datasets, and attributes.

Example usage::

    python scripts/visualize.py \\
        --results results/ \\
        --output figures/
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
        description="Visualise rankability scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Directory of JSON result files, or a single summary JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write figure files.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format.",
    )
    return parser


def load_results(results_path: Path) -> list[dict]:
    """Load all JSON result files from a directory (or a single file)."""
    if results_path.is_file():
        with open(results_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]

    records: list[dict] = []
    for p in sorted(results_path.glob("*.json")):
        with open(p, encoding="utf-8") as fh:
            records.append({"file": p.stem, **json.load(fh)})
    return records


def plot_rankability_bar(records: list[dict], output_dir: Path, fmt: str) -> None:
    """Create a bar chart of rankability scores per result file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — skipping plots.")
        return

    names = [r.get("file", str(i)) for i, r in enumerate(records)]
    scores = [r.get("rankability_score", float("nan")) for r in records]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    bars = ax.bar(names, scores, color="#4C72B0", edgecolor="white")
    ax.set_ylabel("Rankability Score")
    ax.set_title("Zero-Shot Rankability Scores")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = output_dir / f"rankability_bar.{fmt}"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    records = load_results(args.results)
    if not records:
        logger.error("No result files found in %s", args.results)
        return 1

    plot_rankability_bar(records, args.output, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
