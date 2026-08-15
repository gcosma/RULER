"""End-to-end tests of the experiment pipeline, output metrics and reporting.

Breast Cancer ships with scikit-learn, so these run without network access and
exercise the full path from raw data through unlearning to the paper's tables,
using the real cached checkpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.config import MIA_NULL
from paper.data import load_dataset
from paper.models import TabularMLP
from paper.outputs import accuracy, mia_accuracy, per_sample_loss
from paper.pipeline import run_condition
from paper.report import output_level_table, pairwise_method_table, primary_table
from paper.train import train_model


@pytest.fixture(scope="module")
def breast_cancer():
    return load_dataset("breast_cancer")


@pytest.fixture(scope="module")
def condition(breast_cancer, checkpoints_dir, has_checkpoints):
    if not has_checkpoints:
        pytest.skip("no cached checkpoints")
    return run_condition(
        breast_cancer, 0.05, seed=0, checkpoint_dir=checkpoints_dir
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_dataset_shape_matches_the_specification(breast_cancer):
    assert breast_cancer.n_train == 455
    assert breast_cancer.input_dim == 30
    assert len(breast_cancer.x_test) == 114


def test_features_are_standardised_on_the_training_partition(breast_cancer):
    assert np.allclose(breast_cancer.x_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(breast_cancer.x_train.std(axis=0), 1.0, atol=1e-4)


def test_split_is_stratified(breast_cancer):
    train_rate = breast_cancer.y_train.mean()
    test_rate = breast_cancer.y_test.mean()
    assert abs(train_rate - test_rate) < 0.02


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("not_a_dataset")


def test_cached_checkpoints_are_accurate_on_this_preprocessing(
    breast_cancer, checkpoints_dir, has_checkpoints
):
    """A mismatched split or scaler would show up as degraded accuracy.

    The cached weights were trained by the authors; if this repository's
    preprocessing differed from theirs, these models would not classify the
    data well.
    """
    if not has_checkpoints:
        pytest.skip("no cached checkpoints")
    from paper.train import load_model

    model = load_model(checkpoints_dir / "breast_cancer_seed0_orig.pt", 30)
    assert accuracy(model, breast_cancer.x_train, breast_cancer.y_train) > 0.95
    assert accuracy(model, breast_cancer.x_test, breast_cancer.y_test) > 0.90


# ---------------------------------------------------------------------------
# Output-level metrics
# ---------------------------------------------------------------------------


def test_per_sample_loss_is_lower_for_correct_predictions():
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.randn(60, 4) - 2, rng.randn(60, 4) + 2]).astype(np.float32)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(np.int64)
    model = train_model(x, y, seed=0, epochs=200)

    losses = per_sample_loss(model, x, y)
    flipped = per_sample_loss(model, x, 1 - y)
    assert losses.mean() < flipped.mean()


def test_mia_is_near_chance_for_a_model_that_memorises_nothing():
    """Train and held-out records drawn from one distribution are indistinguishable."""
    rng = np.random.RandomState(0)
    x = rng.randn(600, 5).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    model = train_model(x[:400], y[:400], seed=0, epochs=50)

    score = mia_accuracy(
        model, x[:40], y[:40], x[40:400], y[40:400], x[400:], y[400:]
    )
    assert abs(score - MIA_NULL) < 0.15


def test_mia_returns_nan_without_enough_held_out_data():
    model = TabularMLP(4).eval()
    x, y = np.random.randn(10, 4), np.zeros(10, dtype=np.int64)
    assert np.isnan(mia_accuracy(model, x, y, x, y, x[:2], y[:2]))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_condition_produces_one_row_per_method(condition):
    methods = {row["method"] for row in condition.rows}
    assert methods == {"Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB", "Oracle"}


def test_condition_rows_carry_every_metric(condition):
    required = {
        "m1", "m2", "m3", "m4",
        "mia_accuracy", "forget_accuracy", "retain_accuracy", "test_accuracy",
        "dataset", "method", "forget_fraction", "seed", "pre_unlearning_m4",
    }
    assert required <= set(condition.rows[0])


def test_metrics_are_finite_and_in_range(condition):
    frame = pd.DataFrame(condition.rows)
    assert frame[["m1", "m2", "m3", "m4"]].notna().all().all()
    assert frame["m4"].between(0.0, 1.0).all()
    assert frame["m1"].between(-1.0, 1.0).all()
    for column in ("retain_accuracy", "test_accuracy", "forget_accuracy"):
        assert frame[column].between(0.0, 1.0).all(), column


def test_oracle_row_has_a_zero_gap_by_construction(condition):
    """The oracle compared against itself must give M1 = 1 and M2 = 0."""
    oracle = next(r for r in condition.rows if r["method"] == "Oracle")
    assert oracle["m1"] == pytest.approx(1.0, abs=1e-6)
    assert oracle["m2"] == pytest.approx(0.0, abs=1e-6)


def test_forget_set_size_follows_the_published_rule(condition):
    """455 training records at ff = 5% gives floor(22.75) = 22 (Table 9)."""
    assert condition.rows[0]["n_forget"] == 22
    assert condition.rows[0]["n_retain"] == 433


def test_unlearned_models_retain_predictive_performance(condition):
    """Section 5.1: retain and test accuracy are preserved at oracle levels."""
    for row in condition.rows:
        assert row["retain_accuracy"] > 0.90, row["method"]
        assert row["test_accuracy"] > 0.85, row["method"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_results():
    """A results frame spanning several datasets, so the LMM is identifiable."""
    rng = np.random.RandomState(0)
    rows = []
    for dataset in range(4):
        offset = rng.normal(0, 0.001)
        for forget_fraction in (0.01, 0.05):
            for seed in range(10):
                for method in ("Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB"):
                    rows.append(
                        {
                            "dataset": f"ds{dataset}",
                            "dataset_name": f"Dataset {dataset}",
                            "forget_fraction": forget_fraction,
                            "seed": seed,
                            "method": method,
                            "m2": -0.002 + offset + rng.normal(0, 0.0005),
                            "m4": 0.53 + offset + rng.normal(0, 0.01),
                            "mia_accuracy": 0.50 + rng.normal(0, 0.02),
                            "forget_accuracy": 0.85,
                            "retain_accuracy": 0.84,
                            "test_accuracy": 0.83,
                        }
                    )
    return pd.DataFrame(rows)


def test_primary_table_shape_and_significance(synthetic_results):
    table = primary_table(synthetic_results)
    assert len(table) == 8  # 2 forget fractions x 4 methods
    assert (table["n"] == 40).all()
    # A consistent negative M2 should register as significant.
    assert (table["m2_p_lmm"] < 0.05).all()
    assert table["m2"].lt(0).all()


def test_output_level_table_applies_the_pass_window(synthetic_results):
    table = output_level_table(synthetic_results)
    assert table["mia_passes"].all()

    breached = synthetic_results.copy()
    breached["mia_accuracy"] = 0.70
    assert not output_level_table(breached)["mia_passes"].any()


def test_pairwise_table_is_corrected_for_multiplicity(synthetic_results):
    table = pairwise_method_table(synthetic_results)
    assert not table.empty
    assert (table["p_adjusted"] >= table["p"] - 1e-12).all()
