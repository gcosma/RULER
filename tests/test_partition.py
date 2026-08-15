"""Tests for the retain/forget partition and the dataset specification.

The forget-set sizes published in Appendix Table 9 are a complete fingerprint
of the data pipeline: they are reproduced only by the exact row counts, an
80/20 split, and the ``max(10, floor(ff * n_train))`` rule.  Checking them here
verifies the dataset specification and the partition logic together, without
needing to download anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paper.config import FORGET_FRACTIONS, MIN_FORGET_SIZE, TEST_SIZE
from paper.data import DATASETS
from paper.partition import forget_set_size, make_partition

#: |D_f| per dataset at ff = 1%, 5%, 10%, transcribed from Appendix Table 9.
PAPER_FORGET_SIZES = {
    "heart_disease": (10, 12, 24),
    "breast_cancer": (10, 22, 45),
    "german_credit": (10, 40, 80),
    "wine_quality": (12, 63, 127),
    "phoneme": (43, 216, 432),
    "magic": (152, 760, 1521),
    "bank_marketing": (361, 1808, 3616),
    "electricity": (362, 1812, 3624),
    "adult": (390, 1953, 3907),
    "diabetes130": (814, 4070, 8141),
}


def train_size(n_rows: int) -> int:
    """Training rows after ``train_test_split(test_size=0.2)``."""
    return n_rows - math.ceil(TEST_SIZE * n_rows)


def test_every_dataset_has_published_forget_sizes():
    assert set(DATASETS) == set(PAPER_FORGET_SIZES)


@pytest.mark.parametrize("name", sorted(PAPER_FORGET_SIZES))
def test_forget_sizes_match_paper_table_9(name):
    """The spec's row count must reproduce all three published forget sizes."""
    n_train = train_size(DATASETS[name].n_rows)
    computed = tuple(
        forget_set_size(n_train, fraction) for fraction in FORGET_FRACTIONS
    )
    assert computed == PAPER_FORGET_SIZES[name]


@pytest.mark.parametrize("name", sorted(PAPER_FORGET_SIZES))
def test_feature_counts_match_the_cached_checkpoints(name, checkpoints_dir):
    """Spec feature counts must match the input dimension of the cached weights."""
    import torch

    path = checkpoints_dir / f"{name}_seed0_orig.pt"
    if not path.exists():
        pytest.skip(f"no cached checkpoint for {name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["net.0.weight"].shape[1] == DATASETS[name].n_features


def test_floor_applies_to_small_datasets():
    """Three datasets would otherwise get fewer than 10 forget records at 1%."""
    floored = [
        name
        for name, sizes in PAPER_FORGET_SIZES.items()
        if math.floor(0.01 * train_size(DATASETS[name].n_rows)) < MIN_FORGET_SIZE
    ]
    assert set(floored) == {"heart_disease", "breast_cancer", "german_credit"}
    for name in floored:
        assert PAPER_FORGET_SIZES[name][0] == MIN_FORGET_SIZE


def test_partition_is_a_disjoint_cover():
    partition = make_partition(1000, 0.05)
    assert partition.n_forget == 50
    assert partition.n_retain == 950
    assert not set(partition.forget_idx) & set(partition.retain_idx)
    combined = np.concatenate([partition.forget_idx, partition.retain_idx])
    assert np.array_equal(np.sort(combined), np.arange(1000))


def test_partition_is_reproducible_at_a_fixed_seed():
    """The same forget set must be used by every method and every training seed."""
    first = make_partition(500, 0.10)
    second = make_partition(500, 0.10)
    assert np.array_equal(first.forget_idx, second.forget_idx)


def test_partition_changes_with_the_forget_seed():
    """Appendix A.8 varies this seed to test robustness to record selection."""
    default = make_partition(500, 0.10)
    varied = make_partition(500, 0.10, seed=1000)
    assert not np.array_equal(default.forget_idx, varied.forget_idx)
    assert varied.n_forget == default.n_forget


def test_partition_split_helper_aligns_with_indices():
    values = np.arange(100) * 10
    partition = make_partition(100, 0.10)
    (forget_part, retain_part), = partition.split(values)
    assert np.array_equal(forget_part, values[partition.forget_idx])
    assert np.array_equal(retain_part, values[partition.retain_idx])


def test_forget_fraction_must_be_valid():
    with pytest.raises(ValueError):
        forget_set_size(100, 0.0)
    with pytest.raises(ValueError):
        forget_set_size(100, 1.5)


def test_forget_set_cannot_exceed_the_training_set():
    """The floor of 10 must not silently over-draw from a tiny dataset."""
    with pytest.raises(ValueError, match="exceeds the training set"):
        forget_set_size(5, 0.01)
