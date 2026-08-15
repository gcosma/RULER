"""Tests for the statistical analysis (paper Section 4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from paper.stats import (
    benjamini_hochberg,
    mixed_effects_test,
    rank_biserial,
    significance_stars,
    wilcoxon_signed_rank,
)


def clustered_frame(effect=-0.002, between_sd=0.003, within_sd=0.001, seed=0):
    """10 datasets x 10 seeds with a shared effect and dataset-level offsets."""
    rng = np.random.RandomState(seed)
    rows = []
    for dataset in range(10):
        offset = rng.normal(0, between_sd)
        for _ in range(10):
            rows.append(
                {
                    "dataset": f"ds{dataset}",
                    "value": effect + offset + rng.normal(0, within_sd),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wilcoxon and effect size
# ---------------------------------------------------------------------------


def test_rank_biserial_matches_the_published_formula():
    assert rank_biserial(10, 0.0) == pytest.approx(1.0)
    assert rank_biserial(10, 27.5) == pytest.approx(1.0 - 4 * 27.5 / 110)


def test_wilcoxon_agrees_with_scipy():
    values = np.array([0.3, -0.1, 0.5, 0.2, 0.4, -0.2, 0.6, 0.1, 0.35, 0.25])
    result = wilcoxon_signed_rank(values, null=0.0)
    expected = scipy_stats.wilcoxon(values, alternative="two-sided")
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.n == 10


def test_wilcoxon_tests_against_a_non_zero_null():
    """M4 is tested against 0.50, not zero."""
    values = np.full(10, 0.50) + np.array(
        [0.01, 0.02, -0.005, 0.03, 0.015, 0.02, 0.01, 0.04, 0.005, 0.02]
    )
    assert wilcoxon_signed_rank(values, null=0.50).p_value < 0.05
    assert wilcoxon_signed_rank(values, null=0.52).p_value > 0.05


def test_wilcoxon_drops_exact_ties_with_the_null():
    values = np.concatenate([np.zeros(4), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    assert wilcoxon_signed_rank(values, null=0.0).n == 6


def test_wilcoxon_returns_nan_below_the_minimum_sample_size():
    result = wilcoxon_signed_rank([0.1, 0.2, 0.3], null=0.0, min_n=5)
    assert np.isnan(result.p_value)
    assert np.isnan(result.rank_biserial)


def test_five_seeds_cannot_reach_significance():
    """Appendix A.14: at N = 5 the smallest attainable two-sided p is 0.0625.

    The five-seed diagnostics therefore report estimates rather than p-values,
    and the code must not imply otherwise.
    """
    perfectly_consistent = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = wilcoxon_signed_rank(perfectly_consistent, null=0.0, min_n=3)
    assert result.p_value == pytest.approx(2 / 2**5)
    assert result.p_value > 0.05


# ---------------------------------------------------------------------------
# Mixed-effects model
# ---------------------------------------------------------------------------


def test_mixed_model_recovers_the_effect():
    result = mixed_effects_test(clustered_frame(effect=-0.002), "value", "dataset")
    assert result.estimate == pytest.approx(-0.002, abs=0.002)
    assert result.n == 100
    assert result.n_groups == 10


def test_mixed_model_accounts_for_clustering():
    """The random intercept must be more conservative than assuming independence.

    With ten seeds sharing each dataset the observations are not independent;
    a test that ignores that treats between-dataset variation as extra
    evidence and overstates significance.
    """
    frame = clustered_frame(effect=-0.002, between_sd=0.003, within_sd=0.001)
    mixed = mixed_effects_test(frame, "value", "dataset")
    naive = scipy_stats.ttest_1samp(frame["value"], 0.0).pvalue
    assert mixed.p_value > naive


def test_icc_reports_the_between_dataset_variance_share():
    """A high ICC is why M4 is reported per dataset rather than pooled."""
    dominated_by_dataset = mixed_effects_test(
        clustered_frame(between_sd=0.01, within_sd=0.0005), "value", "dataset"
    )
    assert dominated_by_dataset.icc > 0.8

    dominated_by_noise = mixed_effects_test(
        clustered_frame(between_sd=0.0002, within_sd=0.01), "value", "dataset"
    )
    assert dominated_by_noise.icc < 0.3


def test_mixed_model_tests_against_a_non_zero_null():
    frame = clustered_frame(effect=0.52, between_sd=0.005, within_sd=0.002)
    against_zero = mixed_effects_test(frame, "value", "dataset", null=0.0)
    against_half = mixed_effects_test(frame, "value", "dataset", null=0.50)
    assert against_zero.estimate == pytest.approx(0.52, abs=0.01)
    assert against_half.estimate == pytest.approx(0.02, abs=0.01)


def test_mixed_model_reports_rather_than_raises_on_a_degenerate_fit():
    """Appendix A.6 hits a singular REML fit and falls back to Wilcoxon."""
    frame = pd.DataFrame({"dataset": ["a"] * 5, "value": [0.1] * 5})
    result = mixed_effects_test(frame, "value", "dataset")
    assert not result.converged
    assert np.isnan(result.p_value)


def test_mixed_model_ignores_missing_values():
    frame = clustered_frame()
    frame.loc[:9, "value"] = np.nan
    assert mixed_effects_test(frame, "value", "dataset").n == 90


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_matches_the_hand_computation():
    adjusted = benjamini_hochberg([0.001, 0.01, 0.04, 0.2])
    assert adjusted == pytest.approx([0.004, 0.02, 0.0533, 0.2], abs=1e-4)


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.RandomState(0)
    p_values = rng.uniform(0, 1, 40)
    adjusted = benjamini_hochberg(p_values)
    assert (adjusted <= 1.0).all()
    assert (adjusted >= p_values - 1e-12).all(), "adjustment must not lower a p-value"
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_benjamini_hochberg_passes_through_missing_values():
    adjusted = benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(adjusted[1])
    assert np.isfinite(adjusted[[0, 2]]).all()


def test_significance_stars():
    assert significance_stars(0.0005) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.5) == "ns"
    assert significance_stars(np.nan) == "n/a"
