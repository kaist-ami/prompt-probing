"""Rank correlation metrics used to measure zero-shot rankability."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def kendall_tau(
    predicted: Sequence[float],
    ground_truth: Sequence[float],
) -> float:
    """Compute Kendall's tau-b rank correlation coefficient.

    Args:
        predicted: Model-predicted scores or ranks (higher = more of attribute).
        ground_truth: Ground-truth ordinal labels (higher = more of attribute).

    Returns:
        Kendall's tau-b in the range [-1, 1].

    Raises:
        ValueError: If the sequences have different lengths or fewer than 2
            elements.
    """
    pred = np.asarray(predicted, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)

    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted and ground_truth must have the same shape, "
            f"got {pred.shape} vs {gt.shape}"
        )
    if pred.size < 2:
        raise ValueError("At least 2 elements are required to compute rank correlation.")

    n = pred.size
    concordant = discordant = ties_pred = ties_gt = 0

    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = pred[i] - pred[j]
            gt_diff = gt[i] - gt[j]
            prod = pred_diff * gt_diff
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if pred_diff == 0:
                    ties_pred += 1
                if gt_diff == 0:
                    ties_gt += 1

    n0 = n * (n - 1) / 2
    denominator = np.sqrt((n0 - ties_pred) * (n0 - ties_gt))
    if denominator == 0:
        return 0.0
    return (concordant - discordant) / denominator


def spearman_rho(
    predicted: Sequence[float],
    ground_truth: Sequence[float],
) -> float:
    """Compute Spearman's rank correlation coefficient (ρ).

    Args:
        predicted: Model-predicted scores or ranks.
        ground_truth: Ground-truth ordinal labels.

    Returns:
        Spearman's ρ in the range [-1, 1].

    Raises:
        ValueError: If the sequences have different lengths or fewer than 2
            elements.
    """
    pred = np.asarray(predicted, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)

    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted and ground_truth must have the same shape, "
            f"got {pred.shape} vs {gt.shape}"
        )
    if pred.size < 2:
        raise ValueError("At least 2 elements are required to compute rank correlation.")

    def _rank(arr: np.ndarray) -> np.ndarray:
        """Convert values to ranks (average ties)."""
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        # Average ties
        for val in np.unique(arr):
            mask = arr == val
            ranks[mask] = ranks[mask].mean()
        return ranks

    pred_ranks = _rank(pred)
    gt_ranks = _rank(gt)
    d = pred_ranks - gt_ranks
    n = pred.size
    return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1)))


def rankability_score(
    predicted_list: Sequence[Sequence[float]],
    ground_truth_list: Sequence[Sequence[float]],
    metric: str = "kendall_tau",
) -> float:
    """Compute the mean rank correlation (Rankability Score) across instances.

    Args:
        predicted_list: A list of predicted score sequences, one per instance.
        ground_truth_list: A list of ground-truth label sequences, one per
            instance (must be the same length as *predicted_list*).
        metric: Which correlation metric to use — ``"kendall_tau"`` (default)
            or ``"spearman_rho"``.

    Returns:
        Mean rank correlation across all instances.

    Raises:
        ValueError: If *metric* is not recognised or the input lists differ in
            length.
    """
    if len(predicted_list) != len(ground_truth_list):
        raise ValueError(
            "predicted_list and ground_truth_list must have the same number of "
            f"instances, got {len(predicted_list)} vs {len(ground_truth_list)}"
        )

    _metric_fn = {"kendall_tau": kendall_tau, "spearman_rho": spearman_rho}.get(metric)
    if _metric_fn is None:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose one of: 'kendall_tau', 'spearman_rho'."
        )

    scores = [
        _metric_fn(pred, gt)
        for pred, gt in zip(predicted_list, ground_truth_list)
    ]
    return float(np.mean(scores))
