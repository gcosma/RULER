"""Output-level evaluation: the criteria RULER is contrasted against.

Existing unlearning protocols verify erasure with three output-level criteria
(paper Section 1): membership inference accuracy near chance, preserved
retain-set accuracy, and forget-set accuracy matching a retrain oracle.  The
paper's central claim is that a model can satisfy all three while still
encoding forget-set information in its intermediate representations, so these
metrics are computed alongside the representation-level ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import SUBSAMPLE_SEED
from .models import TabularMLP, predict_proba

__all__ = ["OutputMetrics", "accuracy", "per_sample_loss", "mia_accuracy", "output_metrics"]

_EPS = 1e-12


def accuracy(model: TabularMLP, x, y, device="cpu") -> float:
    """Plain classification accuracy."""
    if len(x) == 0:
        return float("nan")
    predictions = predict_proba(model, x, device=device).argmax(axis=1)
    return float(np.mean(predictions == np.asarray(y)))


def per_sample_loss(model: TabularMLP, x, y, device="cpu") -> np.ndarray:
    """Per-record cross-entropy loss, the signal the MIA thresholds on."""
    probabilities = predict_proba(model, x, device=device)
    true_class = probabilities[np.arange(len(y)), np.asarray(y)]
    return -np.log(np.maximum(true_class, _EPS))


def _balanced_accuracy(is_member: np.ndarray, predicted_member: np.ndarray) -> float:
    """Mean of per-class recall, so class imbalance cannot inflate the score."""
    recalls = []
    for label in (0, 1):
        mask = is_member == label
        if mask.any():
            recalls.append(np.mean(predicted_member[mask] == label))
    return float(np.mean(recalls)) if recalls else float("nan")


def _best_threshold(losses: np.ndarray, is_member: np.ndarray) -> float:
    """Loss threshold maximising balanced accuracy, predicting member below it.

    Members are expected to have *lower* loss than non-members, so the attack
    predicts membership when the loss falls below the threshold.
    """
    candidates = np.unique(losses)
    if len(candidates) > 512:
        candidates = np.quantile(losses, np.linspace(0.0, 1.0, 512))
    midpoints = np.concatenate(
        [[-np.inf], (candidates[:-1] + candidates[1:]) / 2.0, [np.inf]]
    )
    scores = [
        _balanced_accuracy(is_member, (losses < threshold).astype(int))
        for threshold in midpoints
    ]
    return float(midpoints[int(np.nanargmax(scores))])


def mia_accuracy(
    model: TabularMLP,
    x_forget,
    y_forget,
    x_retain,
    y_retain,
    x_test,
    y_test,
    *,
    device="cpu",
    seed: int = SUBSAMPLE_SEED,
) -> float:
    """Balanced accuracy of a threshold-based membership inference attack.

    The attack is calibrated and evaluated on disjoint data:

    1. Calibration fits a loss threshold separating *retain* records (members)
       from one half of the held-out test set (non-members).
    2. Evaluation applies that threshold to *forget* records against the other
       half of the test set, and reports balanced accuracy.

    Chance is 0.50, meaning forget records are indistinguishable from records
    the model never saw.  Because the threshold is fitted on separate data, the
    score can fall either side of chance: above 0.50 indicates forget records
    still look like members, below 0.50 indicates they have become *more*
    conspicuous than non-members.  A criterion fitted and evaluated on the same
    records could never fall below chance and would overstate protection.

    The two evaluation groups are subsampled to equal size so that neither
    dominates the balanced accuracy through sheer count.
    """
    rng = np.random.RandomState(seed)
    n_test = len(x_test)
    if n_test < 4 or len(x_forget) == 0:
        return float("nan")

    test_order = rng.permutation(n_test)
    calib_idx, eval_idx = np.array_split(test_order, 2)

    retain_losses = per_sample_loss(model, x_retain, y_retain, device=device)
    test_losses = per_sample_loss(model, x_test, y_test, device=device)
    forget_losses = per_sample_loss(model, x_forget, y_forget, device=device)

    # Calibrate on balanced groups so the threshold is not dragged towards the
    # larger retain set.
    n_calib = min(len(retain_losses), len(calib_idx))
    retain_calib = rng.choice(retain_losses, n_calib, replace=False)
    calib_losses = np.concatenate([retain_calib, test_losses[calib_idx][:n_calib]])
    calib_labels = np.concatenate([np.ones(n_calib, int), np.zeros(n_calib, int)])
    threshold = _best_threshold(calib_losses, calib_labels)

    n_eval = min(len(forget_losses), len(eval_idx))
    eval_losses = np.concatenate(
        [
            rng.choice(forget_losses, n_eval, replace=False),
            test_losses[eval_idx][:n_eval],
        ]
    )
    eval_labels = np.concatenate([np.ones(n_eval, int), np.zeros(n_eval, int)])
    return _balanced_accuracy(eval_labels, (eval_losses < threshold).astype(int))


@dataclass
class OutputMetrics:
    """The three output-level criteria for one condition."""

    mia_accuracy: float
    forget_accuracy: float
    retain_accuracy: float
    test_accuracy: float

    def as_row(self) -> dict:
        return asdict(self)


def output_metrics(
    model: TabularMLP,
    *,
    x_forget,
    y_forget,
    x_retain,
    y_retain,
    x_test,
    y_test,
    device="cpu",
) -> OutputMetrics:
    """MIA accuracy plus forget, retain and test accuracy for one model."""
    return OutputMetrics(
        mia_accuracy=mia_accuracy(
            model,
            x_forget,
            y_forget,
            x_retain,
            y_retain,
            x_test,
            y_test,
            device=device,
        ),
        forget_accuracy=accuracy(model, x_forget, y_forget, device=device),
        retain_accuracy=accuracy(model, x_retain, y_retain, device=device),
        test_accuracy=accuracy(model, x_test, y_test, device=device),
    )
