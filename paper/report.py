"""Aggregation of raw results into the paper's tables.

Each function takes the tidy frame produced by :func:`ruler.pipeline.run_experiment`
and returns a DataFrame corresponding to one published table.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .config import M2_NULL, M4_NULL, METHODS, MIA_NULL, MIA_PASS_WINDOW
from .stats import (
    benjamini_hochberg,
    mixed_effects_test,
    rank_biserial,
    significance_stars,
    wilcoxon_signed_rank,
)

__all__ = [
    "primary_table",
    "output_level_table",
    "per_dataset_mia_table",
    "pairwise_method_table",
    "dataset_level_table",
    "m2_baseline_sensitivity",
]


def _methods_in(frame: pd.DataFrame) -> list[str]:
    """Methods present, in the paper's ordering, with any extras appended."""
    present = set(frame["method"].unique()) - {"Oracle"}
    ordered = [m for m in METHODS if m in present]
    return ordered + sorted(present - set(ordered))


def primary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 1: M2 and M4 per (forget fraction, method).

    Reports the linear mixed-effects test on all N = 100 observations as the
    primary inference, and the Wilcoxon signed-rank test on the 10
    dataset-level means as the conservative check.  The ICC accompanies each
    row because it determines how the metric should be read: a high ICC means
    dataset identity dominates and the metric belongs in a per-dataset
    diagnostic rather than a pooled test.
    """
    rows = []
    for forget_fraction in sorted(frame["forget_fraction"].unique()):
        for method in _methods_in(frame):
            subset = frame[
                (frame["forget_fraction"] == forget_fraction)
                & (frame["method"] == method)
            ]
            if subset.empty:
                continue
            row = {
                "forget_fraction": forget_fraction,
                "method": method,
                "n": len(subset),
            }
            for metric, null in (("m2", M2_NULL), ("m4", M4_NULL)):
                lmm = mixed_effects_test(subset, metric, "dataset", null=null)
                dataset_means = subset.groupby("dataset")[metric].mean()
                wilcoxon = wilcoxon_signed_rank(dataset_means.to_numpy(), null=null)
                row.update(
                    {
                        metric: subset[metric].mean(),
                        f"{metric}_p_lmm": lmm.p_value,
                        f"{metric}_icc": lmm.icc,
                        f"{metric}_lmm_converged": lmm.converged,
                        f"{metric}_p_wilcoxon": wilcoxon.p_value,
                        f"{metric}_r_rb": wilcoxon.rank_biserial,
                        f"{metric}_sig": significance_stars(lmm.p_value),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def output_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 3: output-level evaluation summary, mean +/- SD per condition.

    ``mia_passes`` applies the paper's criterion: the mean MIA accuracy across
    training seeds must lie within +/-0.05 of chance.
    """
    columns = ["mia_accuracy", "forget_accuracy", "retain_accuracy", "test_accuracy"]
    rows = []
    for forget_fraction in sorted(frame["forget_fraction"].unique()):
        for method in _methods_in(frame) + ["Oracle"]:
            subset = frame[
                (frame["forget_fraction"] == forget_fraction)
                & (frame["method"] == method)
            ]
            if subset.empty:
                continue
            row = {
                "forget_fraction": forget_fraction,
                "method": method,
                "n": len(subset),
            }
            for column in columns:
                row[f"{column}_mean"] = subset[column].mean()
                row[f"{column}_sd"] = subset[column].std(ddof=1)
            row["mia_passes"] = bool(
                abs(row["mia_accuracy_mean"] - MIA_NULL) < MIA_PASS_WINDOW
            )
            rows.append(row)
    return pd.DataFrame(rows)


def per_dataset_mia_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 4: per-dataset MIA accuracy, with the pass window flagged.

    The aggregate in :func:`output_level_table` masks real per-dataset
    variation. Where a method breaches the window, the oracle usually breaches
    it in the same direction -- a dataset-level characteristic such as class
    imbalance rather than incomplete forgetting -- which is why the oracle is
    included as a column here.
    """
    table = (
        frame.pivot_table(
            index=["forget_fraction", "dataset_name"],
            columns="method",
            values="mia_accuracy",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    method_columns = [c for c in table.columns if c in set(frame["method"].unique())]
    breaches = (table[method_columns] - MIA_NULL).abs() >= MIA_PASS_WINDOW
    table["n_breaching"] = breaches.sum(axis=1)
    return table


def dataset_level_table(frame: pd.DataFrame, metric: str = "m2") -> pd.DataFrame:
    """Table 10: one-sample Wilcoxon on the 10 dataset-level means.

    Averaging within a dataset before testing removes the within-dataset
    correlation that makes the raw N = 100 observations non-independent.
    """
    null = M2_NULL if metric == "m2" else M4_NULL
    rows = []
    for forget_fraction in sorted(frame["forget_fraction"].unique()):
        for method in _methods_in(frame):
            subset = frame[
                (frame["forget_fraction"] == forget_fraction)
                & (frame["method"] == method)
            ]
            if subset.empty:
                continue
            dataset_means = subset.groupby("dataset")[metric].mean()
            result = wilcoxon_signed_rank(dataset_means.to_numpy(), null=null)
            rows.append(
                {
                    "forget_fraction": forget_fraction,
                    "method": method,
                    "metric": metric,
                    "mean": dataset_means.mean(),
                    "n_datasets": len(dataset_means),
                    "p": result.p_value,
                    "r_rb": result.rank_biserial,
                    "sig": significance_stars(result.p_value),
                }
            )
    return pd.DataFrame(rows)


def pairwise_method_table(
    frame: pd.DataFrame, metrics: tuple[str, ...] = ("m2", "m4", "mia_accuracy")
) -> pd.DataFrame:
    """Section 5.3: post-hoc pairwise method comparisons with BH correction.

    Comparisons are paired on (dataset, seed): the same original model and the
    same forget set underlie both methods, so a paired test is the right one
    and is far more sensitive than an unpaired comparison would be.
    Adjustment is applied across the whole family of tests at once.
    """
    rows = []
    for forget_fraction in sorted(frame["forget_fraction"].unique()):
        subset = frame[frame["forget_fraction"] == forget_fraction]
        methods = _methods_in(subset)
        for metric in metrics:
            for left, right in itertools.combinations(methods, 2):
                paired = (
                    subset[subset["method"].isin([left, right])]
                    .pivot_table(
                        index=["dataset", "seed"], columns="method", values=metric
                    )
                    .dropna()
                )
                if len(paired) < 5 or left not in paired or right not in paired:
                    continue
                differences = (paired[left] - paired[right]).to_numpy()
                nonzero = differences[differences != 0]
                if len(nonzero) < 5:
                    continue
                statistic, p_value = stats.wilcoxon(nonzero, alternative="two-sided")
                rows.append(
                    {
                        "forget_fraction": forget_fraction,
                        "metric": metric,
                        "method_a": left,
                        "method_b": right,
                        "mean_difference": differences.mean(),
                        "p": float(p_value),
                        "r_rb": rank_biserial(len(nonzero), float(statistic)),
                        "n_pairs": len(paired),
                    }
                )

    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_adjusted"] = benjamini_hochberg(table["p"].to_numpy())
        table["sig_adjusted"] = table["p_adjusted"].map(significance_stars)
    return table


def m2_baseline_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 5: M2 under the median baseline versus an arithmetic mean baseline.

    The comparison matters because the retain similarity distribution is
    right-skewed: a mean baseline sits above the median by an amount comparable
    to the gap itself, which masks the signal and reverses its sign in several
    conditions. ``m2_mean_baseline`` is recorded by every run.
    """
    if "m2_mean_baseline" not in frame.columns:
        raise KeyError(
            "frame lacks 'm2_mean_baseline'; it is written by ruler >= 1.1, so "
            "this results file predates that and must be regenerated"
        )
    rows = []
    for forget_fraction in sorted(frame["forget_fraction"].unique()):
        for method in _methods_in(frame):
            subset = frame[
                (frame["forget_fraction"] == forget_fraction)
                & (frame["method"] == method)
            ]
            if subset.empty:
                continue
            row = {"forget_fraction": forget_fraction, "method": method}
            for label, column in (("median", "m2"), ("mean", "m2_mean_baseline")):
                dataset_means = subset.groupby("dataset")[column].mean()
                result = wilcoxon_signed_rank(dataset_means.to_numpy(), null=M2_NULL)
                row[f"m2_{label}"] = subset[column].mean()
                row[f"p_{label}"] = result.p_value
                row[f"r_rb_{label}"] = result.rank_biserial
            row["sign_reversed"] = bool(
                np.sign(row["m2_median"]) != np.sign(row["m2_mean"])
            )
            rows.append(row)
    return pd.DataFrame(rows)
