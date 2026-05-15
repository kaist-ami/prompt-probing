"""Tests for rankability metrics."""

from __future__ import annotations

import math

import pytest

from prompt_probing.ranking.metrics import kendall_tau, spearman_rho, rankability_score


class TestKendallTau:
    def test_perfect_agreement(self):
        assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        assert kendall_tau([4, 3, 2, 1], [1, 2, 3, 4]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Two-element case with opposite ordering
        assert kendall_tau([2, 1], [1, 2]) == pytest.approx(-1.0)

    def test_tied_values_return_zero_when_all_tied(self):
        # When all predictions are the same, tau = 0
        assert kendall_tau([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)

    def test_partial_concordance(self):
        tau = kendall_tau([1, 2, 4, 3], [1, 2, 3, 4])
        assert -1.0 <= tau <= 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            kendall_tau([1, 2, 3], [1, 2])

    def test_too_few_elements_raise(self):
        with pytest.raises(ValueError, match="At least 2"):
            kendall_tau([1], [1])


class TestSpearmanRho:
    def test_perfect_agreement(self):
        assert spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        assert spearman_rho([4, 3, 2, 1], [1, 2, 3, 4]) == pytest.approx(-1.0)

    def test_partial_correlation(self):
        rho = spearman_rho([1, 3, 2, 4], [1, 2, 3, 4])
        assert -1.0 <= rho <= 1.0
        assert rho > 0  # mostly concordant

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            spearman_rho([1, 2, 3], [1, 2])

    def test_too_few_elements_raise(self):
        with pytest.raises(ValueError, match="At least 2"):
            spearman_rho([1], [1])


class TestRankabilityScore:
    def test_perfect_rankings(self):
        rs = rankability_score(
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]],
        )
        assert rs == pytest.approx(1.0)

    def test_mixed_rankings(self):
        rs = rankability_score(
            [[1, 2, 3], [3, 2, 1]],
            [[1, 2, 3], [1, 2, 3]],
        )
        assert rs == pytest.approx(0.0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            rankability_score([[1, 2]], [[1, 2]], metric="bad_metric")

    def test_mismatched_list_lengths_raise(self):
        with pytest.raises(ValueError, match="same number of instances"):
            rankability_score([[1, 2]], [[1, 2], [3, 4]])

    def test_spearman_metric(self):
        rs = rankability_score(
            [[1, 2, 3]],
            [[1, 2, 3]],
            metric="spearman_rho",
        )
        assert rs == pytest.approx(1.0)
