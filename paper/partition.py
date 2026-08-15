"""Retain/forget partitioning of the training set (paper Section 4.1).

The forget set is drawn once per (dataset, forget fraction) and reused for
every unlearning method and every training seed, so that method comparisons
are not confounded by which records were selected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import FORGET_SEED, MIN_FORGET_SIZE

__all__ = ["Partition", "forget_set_size", "make_partition"]


def forget_set_size(n_train: int, forget_fraction: float) -> int:
    """``max(10, floor(ff * |D|))`` -- the paper's forget-set size rule.

    The floor of 10 prevents degenerate forget sets on the smaller datasets:
    at ff = 1% Heart Disease, Breast Cancer and German Credit would otherwise
    contribute only 2 to 8 records.
    """
    if not 0.0 < forget_fraction <= 1.0:
        raise ValueError(f"forget fraction must be in (0, 1], got {forget_fraction}")
    size = max(MIN_FORGET_SIZE, math.floor(forget_fraction * n_train))
    if size > n_train:
        raise ValueError(
            f"forget set of {size} records exceeds the training set ({n_train})"
        )
    return size


@dataclass(frozen=True)
class Partition:
    """Indices into the training set defining the retain and forget sets."""

    forget_idx: np.ndarray
    retain_idx: np.ndarray
    forget_fraction: float
    seed: int
    #: Number of whole groups erased, when partitioning by group.
    n_forget_groups: int | None = None

    @property
    def n_forget(self) -> int:
        return len(self.forget_idx)

    @property
    def n_retain(self) -> int:
        return len(self.retain_idx)

    def split(self, *arrays):
        """Split each array into ``(forget_part, retain_part)`` pairs."""
        return [(a[self.forget_idx], a[self.retain_idx]) for a in arrays]


def make_partition(
    n_train: int,
    forget_fraction: float,
    seed: int = FORGET_SEED,
    groups=None,
) -> Partition:
    """Sample the forget set without replacement at a fixed random state.

    Parameters
    ----------
    seed
        Fixed at 999 for all primary experiments. Appendix A.8 varies it over
        999-1003 to confirm the findings do not depend on which records were
        selected.
    groups
        Optional length-``n_train`` array of group labels -- a patient id, a
        document id, a face identity. When given, *whole groups* are erased and
        ``forget_fraction`` is read as a fraction of groups rather than of
        records.

        Use this whenever the unit someone asks you to erase is larger than one
        row. Splitting a group across the two sets silently corrupts the
        metrics: the "forgotten" rows still have near neighbours from the same
        group sitting in the retain set, so M4 reads high for a reason that has
        nothing to do with what the model memorised.

    Examples
    --------
    Erase whole patients rather than individual sentences::

        partition = make_partition(len(sentences), 0.05, groups=patient_ids)
    """
    if groups is None:
        size = forget_set_size(n_train, forget_fraction)
        rng = np.random.RandomState(seed)
        forget_idx = np.sort(rng.choice(n_train, size, replace=False))
        retain_idx = np.setdiff1d(np.arange(n_train), forget_idx, assume_unique=True)
        return Partition(
            forget_idx=forget_idx,
            retain_idx=retain_idx,
            forget_fraction=forget_fraction,
            seed=seed,
        )

    groups = np.asarray(groups)
    if len(groups) != n_train:
        raise ValueError(
            f"groups has {len(groups)} labels but the training set has {n_train} "
            "records; there must be one label per record"
        )
    if not 0.0 < forget_fraction <= 1.0:
        raise ValueError(f"forget fraction must be in (0, 1], got {forget_fraction}")

    unique = np.unique(groups)
    n_forget_groups = max(1, math.floor(forget_fraction * len(unique)))
    rng = np.random.RandomState(seed)
    forget_groups = rng.choice(unique, n_forget_groups, replace=False)

    is_forget = np.isin(groups, forget_groups)
    return Partition(
        forget_idx=np.flatnonzero(is_forget),
        retain_idx=np.flatnonzero(~is_forget),
        forget_fraction=forget_fraction,
        seed=seed,
        n_forget_groups=n_forget_groups,
    )
