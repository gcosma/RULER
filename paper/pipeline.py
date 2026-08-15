"""The experiment loop: one place where all the pieces are wired together.

For each (dataset, forget fraction, training seed) the pipeline

1. partitions the training set into retain and forget sets,
2. loads or trains the original model and the retrain oracle,
3. records the *pre-unlearning* M4 diagnostic on the original model,
4. applies each unlearning method,
5. computes representation-level (M1-M4) and output-level (MIA, accuracies)
   metrics for each unlearned model, and for the oracle as a reference.

Results are returned as tidy rows, one per (dataset, ff, seed, method), which
:mod:`ruler.report` then aggregates into the paper's tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence

import pandas as pd

from .config import (
    FORGET_FRACTIONS,
    M2_RETAIN_SUBSAMPLE,
    M4_RETAIN_CAP,
    METHODS,
    SUBSAMPLE_SEED,
    TRAIN_SEEDS,
    UNLEARN_SEED,
)
from .data import TabularDataset, dataset_names, load_dataset
import numpy as np

from ruler import cosine_similarity, m1, m2, m3, m4
from .models import penultimate
from .outputs import output_metrics
from .partition import make_partition
from .train import checkpoint_path, get_or_train
from .unlearn import apply_unlearning

__all__ = ["ConditionResult", "run_condition", "run_experiment"]


@dataclass
class ConditionResult:
    """Result rows for one condition, plus its pre-unlearning M4 diagnostic."""

    rows: list[dict]
    pre_unlearning_m4: float


def _subsample(embeddings, cap, seed=SUBSAMPLE_SEED):
    """Reproducible subsample of retain rows, or the array unchanged."""
    if len(embeddings) <= cap:
        return embeddings
    idx = np.random.RandomState(seed).choice(len(embeddings), cap, replace=False)
    return embeddings[idx]


def _representation_row(*, unlearned, original, oracle, x_forget, x_retain, device):
    """Embed once per model, then compute M1-M4 with the paper's subsampling.

    Section 4.3: the M2 baseline uses min(500, |D_r|) retain records and the M4
    gallery is capped at 2,000. The metrics themselves come from the library.
    """
    unlearned_forget = penultimate(unlearned, x_forget, device=device)
    unlearned_retain = penultimate(unlearned, x_retain, device=device)
    oracle_forget = penultimate(oracle, x_forget, device=device)
    oracle_retain = penultimate(oracle, x_retain, device=device)
    original_forget = penultimate(original, x_forget, device=device)

    # The two draws are independent by design: 500 paired records suffice for
    # the calibration baseline, while the gallery defines the geometry M4 ranks
    # against and benefits from more.
    n_retain = len(unlearned_retain)
    keep = _subsample(np.arange(n_retain), M2_RETAIN_SUBSAMPLE)
    gallery = _subsample(unlearned_retain, M4_RETAIN_CAP)

    retain_similarity = cosine_similarity(
        unlearned_retain[keep], oracle_retain[keep]
    )
    forget_similarity = cosine_similarity(unlearned_forget, oracle_forget)

    return {
        "m1": m1(unlearned_forget, oracle_forget),
        "m2": m2(
            unlearned_forget,
            oracle_forget,
            unlearned_retain[keep],
            oracle_retain[keep],
        ),
        "m3": m3(unlearned_forget, original_forget, oracle_forget),
        "m4": m4(unlearned_forget, gallery),
        "median_retain_similarity": float(np.median(retain_similarity)),
        "mean_retain_similarity": float(np.mean(retain_similarity)),
        # Appendix A.5 compares M2 against a mean rather than median baseline.
        "m2_mean_baseline": float(
            np.mean(forget_similarity) - np.mean(retain_similarity)
        ),
        "n_forget": len(unlearned_forget),
        "n_retain_calibration": len(keep),
        "n_retain_gallery": len(gallery),
    }


def run_condition(
    dataset: TabularDataset,
    forget_fraction: float,
    seed: int,
    *,
    checkpoint_dir: str | Path = "checkpoints",
    methods: Sequence[str] = METHODS,
    unlearn_seed: int = UNLEARN_SEED,
    batch_size: int | None = None,
    unlearn_lr: float | None = None,
    forget_seed: int | None = None,
    include_oracle_row: bool = True,
    device: str = "cpu",
    train_kwargs: dict | None = None,
) -> ConditionResult:
    """Evaluate every method for one (dataset, forget fraction, seed).

    Parameters
    ----------
    unlearn_lr, batch_size
        Overrides for the sensitivity analyses of Appendix A.6 and A.7. Leave
        as ``None`` for the primary configuration.
    forget_seed
        Overrides the forget-set sampling seed for Appendix A.8. Changing it
        also changes the oracle, which is retrained on the new retain set, so
        such runs must use a separate checkpoint directory.
    include_oracle_row
        Also evaluate the oracle itself. Its output-level metrics are the
        reference the paper's pass criterion is read against, and its M4 gives
        the per-dataset baseline of Appendix Fig. 7.
    """
    train_kwargs = dict(train_kwargs or {})
    partition = (
        make_partition(dataset.n_train, forget_fraction)
        if forget_seed is None
        else make_partition(dataset.n_train, forget_fraction, seed=forget_seed)
    )

    x_forget = dataset.x_train[partition.forget_idx]
    y_forget = dataset.y_train[partition.forget_idx]
    x_retain = dataset.x_train[partition.retain_idx]
    y_retain = dataset.y_train[partition.retain_idx]

    # The original model and the oracle share `seed`: the paired-seed design.
    original, _ = get_or_train(
        checkpoint_path(checkpoint_dir, dataset.key, seed, "orig"),
        dataset.x_train,
        dataset.y_train,
        seed=seed,
        input_dim=dataset.input_dim,
        device=device,
        **train_kwargs,
    )
    oracle, _ = get_or_train(
        checkpoint_path(checkpoint_dir, dataset.key, seed, "oracle", forget_fraction),
        x_retain,
        y_retain,
        seed=seed,
        input_dim=dataset.input_dim,
        device=device,
        **train_kwargs,
    )

    # Pre-unlearning diagnostic: was anything memorised to begin with?
    pre_m4 = _representation_row(
        unlearned=original,
        original=original,
        oracle=oracle,
        x_forget=x_forget,
        x_retain=x_retain,
        device=device,
    )["m4"]

    shared = {
        "dataset": dataset.key,
        "dataset_name": dataset.display_name,
        "forget_fraction": forget_fraction,
        "seed": seed,
        "n_forget": partition.n_forget,
        "n_retain": partition.n_retain,
        "pre_unlearning_m4": pre_m4,
    }

    unlearn_kwargs = {"seed": unlearn_seed, "batch_size": batch_size, "device": device}
    if unlearn_lr is not None:
        unlearn_kwargs["lr"] = unlearn_lr

    rows: list[dict] = []
    for method in methods:
        unlearned = apply_unlearning(
            method,
            original,
            x_forget=x_forget,
            y_forget=y_forget,
            x_retain=x_retain,
            y_retain=y_retain,
            **unlearn_kwargs,
        )
        representation = _representation_row(
            unlearned=unlearned,
            original=original,
            oracle=oracle,
            x_forget=x_forget,
            x_retain=x_retain,
            device=device,
        )
        outputs = output_metrics(
            unlearned,
            x_forget=x_forget,
            y_forget=y_forget,
            x_retain=x_retain,
            y_retain=y_retain,
            x_test=dataset.x_test,
            y_test=dataset.y_test,
            device=device,
        )
        rows.append(
            {**shared, "method": method, **representation, **outputs.as_row()}
        )

    if include_oracle_row:
        # M2 and M3 are identically zero for the oracle compared with itself,
        # so only M4 and the output-level metrics are meaningful here.
        oracle_representation = _representation_row(
            unlearned=oracle,
            original=original,
            oracle=oracle,
            x_forget=x_forget,
            x_retain=x_retain,
            device=device,
        )
        oracle_outputs = output_metrics(
            oracle,
            x_forget=x_forget,
            y_forget=y_forget,
            x_retain=x_retain,
            y_retain=y_retain,
            x_test=dataset.x_test,
            y_test=dataset.y_test,
            device=device,
        )
        rows.append(
            {
                **shared,
                "method": "Oracle",
                **oracle_representation,
                **oracle_outputs.as_row(),
            }
        )

    return ConditionResult(rows=rows, pre_unlearning_m4=pre_m4)


def run_experiment(
    datasets: Iterable[str] | None = None,
    forget_fractions: Sequence[float] = FORGET_FRACTIONS,
    seeds: Sequence[int] = TRAIN_SEEDS,
    *,
    checkpoint_dir: str | Path = "checkpoints",
    methods: Sequence[str] = METHODS,
    progress: Callable[[str], None] | None = print,
    **condition_kwargs,
) -> pd.DataFrame:
    """Run the full grid and return one tidy row per (dataset, ff, seed, method).

    The primary experiment is 10 datasets x 3 forget fractions x 10 seeds,
    giving N = 100 observations per (method, forget fraction) condition.
    """
    datasets = list(datasets) if datasets is not None else dataset_names()
    rows: list[dict] = []

    for name in datasets:
        dataset = load_dataset(name)
        if progress:
            progress(
                f"{dataset.display_name}: {dataset.n_train} train, "
                f"{dataset.input_dim} features"
            )
        for forget_fraction in forget_fractions:
            for seed in seeds:
                result = run_condition(
                    dataset,
                    forget_fraction,
                    seed,
                    checkpoint_dir=checkpoint_dir,
                    methods=methods,
                    **condition_kwargs,
                )
                rows.extend(result.rows)
            if progress:
                progress(
                    f"  ff={forget_fraction:.0%}: {len(seeds)} seeds, "
                    f"pre-unlearning M4={result.pre_unlearning_m4:.4f}"
                )

    return pd.DataFrame(rows)
