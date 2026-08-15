"""Statistical analysis for the RULER evaluation (paper Section 4.3).

The primary analysis is a linear mixed-effects model with a random intercept
for dataset, fitted to all N = 100 observations (10 datasets x 10 seeds) per
condition.  The random intercept is what makes the test valid here: the 100
observations are not independent, since ten of them share each dataset, and
ignoring that clustering would treat between-dataset variation as if it were
extra evidence.

With only 10 clusters the mixed model can overstate significance, so a
Wilcoxon signed-rank test on the 10 dataset-level means is reported alongside
it as a conservative check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "WilcoxonResult",
    "MixedModelResult",
    "rank_biserial",
    "wilcoxon_signed_rank",
    "mixed_effects_test",
    "benjamini_hochberg",
    "significance_stars",
]


@dataclass
class WilcoxonResult:
    """A Wilcoxon signed-rank test: statistic, p-value and effect size."""

    statistic: float
    p_value: float
    rank_biserial: float
    n: int


@dataclass
class MixedModelResult:
    """Fixed intercept of a random-intercept model, tested against a null."""

    estimate: float
    std_error: float
    z: float
    p_value: float
    icc: float
    n: int
    n_groups: int
    converged: bool


def rank_biserial(n: int, statistic: float) -> float:
    """Rank-biserial correlation ``r_rb = 1 - 4W / (n(n+1))``.

    ``n`` counts non-tied differences, matching what the Wilcoxon test uses.
    By convention 0.10, 0.30 and 0.50 mark small, medium and large effects.
    """
    if n <= 0:
        return float("nan")
    return 1.0 - (4.0 * statistic) / (n * (n + 1))


def wilcoxon_signed_rank(
    values, null: float = 0.0, min_n: int = 5
) -> WilcoxonResult:
    """Two-sided one-sample Wilcoxon signed-rank test against ``null``.

    Exact ties with the null carry no rank information and are dropped, which
    is the standard treatment and matches what ``rank_biserial`` assumes.

    Parameters
    ----------
    min_n
        Below this many non-tied differences the test is not reported. The
        smallest attainable two-sided p-value is ``2 / 2**n``, so at n = 5 no
        result can reach p < 0.05 however consistent the data; the paper notes
        this explicitly for the five-seed diagnostics (Appendix A.14).
    """
    differences = np.asarray(values, dtype=float) - null
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]
    n = len(differences)
    if n < min_n:
        return WilcoxonResult(np.nan, np.nan, np.nan, n)
    statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
    return WilcoxonResult(
        statistic=float(statistic),
        p_value=float(p_value),
        rank_biserial=rank_biserial(n, float(statistic)),
        n=n,
    )


def mixed_effects_test(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str = "dataset",
    null: float = 0.0,
) -> MixedModelResult:
    """Random-intercept model of ``value - null``, testing the fixed intercept.

    Fitted by restricted maximum likelihood; the intercept is tested with a
    Wald z-statistic.  The intra-class correlation
    ``ICC = var_between / (var_between + var_within)`` reports how much of the
    variance is attributable to dataset identity -- a high ICC means the metric
    is dataset-specific and better read per dataset than pooled, which is why
    the paper reports M4 as a per-dataset diagnostic (ICC = 0.89) while M2
    supports population-level inference.

    A singular fit -- zero estimated between-dataset variance -- is reported
    with ``converged=False`` rather than raised, since it is informative in
    itself: Appendix A.6 hits exactly this case under mini-batch training and
    falls back to the Wilcoxon test.
    """
    import statsmodels.formula.api as smf

    data = frame[[value_column, group_column]].dropna().copy()
    data = data.rename(columns={value_column: "_value", group_column: "_group"})
    data["_value"] = data["_value"] - null

    n = len(data)
    n_groups = data["_group"].nunique()
    if n < 3 or n_groups < 2:
        return MixedModelResult(np.nan, np.nan, np.nan, np.nan, np.nan, n, n_groups, False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm("_value ~ 1", data, groups=data["_group"]).fit(reml=True)
        except Exception:
            return MixedModelResult(
                np.nan, np.nan, np.nan, np.nan, np.nan, n, n_groups, False
            )

    estimate = float(fit.params["Intercept"])
    std_error = float(fit.bse["Intercept"])
    between = float(np.asarray(fit.cov_re)[0, 0])
    within = float(fit.scale)

    if not np.isfinite(std_error) or std_error <= 0:
        return MixedModelResult(
            estimate, np.nan, np.nan, np.nan, np.nan, n, n_groups, False
        )

    z = estimate / std_error
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    total = between + within
    icc = float(between / total) if total > 0 else np.nan

    return MixedModelResult(
        estimate=estimate,
        std_error=std_error,
        z=float(z),
        p_value=p_value,
        icc=icc,
        n=n,
        n_groups=n_groups,
        # A zero between-dataset variance means REML could not identify the
        # random effect; the Wald p-value is then not trustworthy.
        converged=bool(getattr(fit, "converged", True)) and between > 0,
    )


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, controlling the false discovery rate.

    Used for the 24 post-hoc pairwise method comparisons of Section 5.3. NaN
    inputs are excluded from the ranking and returned as NaN.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted

    values = p_values[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    scaled = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p-value down, so an adjusted value
    # never exceeds that of a less significant test.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]

    out = np.empty(m)
    out[order] = np.minimum(scaled, 1.0)
    adjusted[finite] = out
    return adjusted


def significance_stars(p_value: float) -> str:
    """``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, else ``ns``."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"
