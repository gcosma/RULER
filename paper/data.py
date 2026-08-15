"""The ten tabular datasets of the primary evaluation (paper Section 4.1).

Nine datasets come from OpenML and Breast Cancer from ``sklearn.datasets``.
Each is split 80/20 into training and held-out test partitions with stratified
sampling at random state 999, and standardised with a ``StandardScaler`` fitted
on the training partition only.

Every dataset carries its expected row and feature counts, taken from the
paper.  A load that does not match them raises rather than proceeding, because
a silently different OpenML version would change the forget-set sizes and make
the cached checkpoints in ``checkpoints/`` invalid without any visible error.
``tests/test_data_spec.py`` checks these counts against the forget-set sizes
published in Appendix Table 9 without needing network access.

Offline use
-----------
If OpenML is unreachable, place a CSV per dataset in ``$RULER_DATA_DIR``
(default ``./data``) named ``<dataset>.csv``, with the target in the last
column.  Loaders prefer that file when it exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SPLIT_SEED, TEST_SIZE

__all__ = ["DATASETS", "DatasetSpec", "TabularDataset", "load_dataset", "dataset_names"]


def _data_dir() -> Path:
    return Path(os.environ.get("RULER_DATA_DIR", "data"))


@dataclass(frozen=True)
class DatasetSpec:
    """How to obtain and prepare one dataset.

    Attributes
    ----------
    openml_id
        OpenML data id, or ``None`` for Breast Cancer which ships with sklearn.
    n_rows, n_features
        Expected shape after preprocessing. Enforced on load.
    binarise
        Name of the rule turning a multi-class target into a binary one. Three
        of the ten datasets need this (Section 4.1).
    drop_columns
        Columns removed before use, e.g. row identifiers that carry no signal
        but would otherwise be counted as features.
    """

    key: str
    display_name: str
    openml_id: int | None
    n_rows: int
    n_features: int
    binarise: str | None = None
    drop_columns: tuple[str, ...] = ()
    note: str = ""


# Row and feature counts below are pinned by Appendix Table 9: they are the
# only counts that reproduce every published forget-set size under
# max(10, floor(ff * n_train)) with an 80/20 split.
DATASETS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        key="adult",
        display_name="Adult Income",
        openml_id=1590,
        n_rows=48842,
        n_features=14,
        note="Categorical columns are ordinal-encoded, keeping 14 features.",
    ),
    "diabetes130": DatasetSpec(
        key="diabetes130",
        display_name="Diabetes 130-US",
        openml_id=4541,
        n_rows=101766,
        n_features=49,
        binarise="readmitted",
    ),
    "breast_cancer": DatasetSpec(
        key="breast_cancer",
        display_name="Breast Cancer",
        openml_id=None,
        n_rows=569,
        n_features=30,
        note="sklearn.datasets.load_breast_cancer",
    ),
    "heart_disease": DatasetSpec(
        key="heart_disease",
        display_name="Heart Disease",
        openml_id=43672,
        n_rows=303,
        n_features=13,
        binarise="positive_class",
        note="Cleveland processed subset: 303 records, 13 features, target 0-4.",
    ),
    "german_credit": DatasetSpec(
        key="german_credit",
        display_name="German Credit",
        openml_id=31,
        n_rows=1000,
        n_features=20,
    ),
    "bank_marketing": DatasetSpec(
        key="bank_marketing",
        display_name="Bank Marketing",
        openml_id=1461,
        n_rows=45211,
        n_features=16,
    ),
    "wine_quality": DatasetSpec(
        key="wine_quality",
        display_name="Wine Quality",
        openml_id=40691,
        n_rows=1599,
        n_features=11,
        binarise="wine_quality",
        note="Red wine subset; quality score binarised at >= 6.",
    ),
    "phoneme": DatasetSpec(
        key="phoneme",
        display_name="Phoneme",
        openml_id=1489,
        n_rows=5404,
        n_features=5,
    ),
    "magic": DatasetSpec(
        key="magic",
        display_name="Magic Telescope",
        openml_id=1120,
        n_rows=19020,
        n_features=10,
        drop_columns=("ID",),
    ),
    "electricity": DatasetSpec(
        key="electricity",
        display_name="Electricity",
        openml_id=151,
        n_rows=45312,
        n_features=8,
    ),
}


def dataset_names() -> list[str]:
    """Dataset keys in the order used by the paper's per-dataset tables."""
    return list(DATASETS)


@dataclass
class TabularDataset:
    """A loaded, split and standardised dataset."""

    key: str
    display_name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_train(self) -> int:
        return self.x_train.shape[0]


# ---------------------------------------------------------------------------
# Target binarisation (Section 4.1: "three with multi-class targets are
# binarised"). Each rule is stated explicitly because the choice of threshold
# determines class balance and therefore the achievable accuracies.
# ---------------------------------------------------------------------------


def _binarise(y: pd.Series, rule: str) -> np.ndarray:
    if rule == "readmitted":
        # Diabetes 130-US: readmitted in {"<30", ">30", "NO"}.
        # Positive class = readmitted at any horizon.
        return (y.astype(str).str.upper() != "NO").astype(int).to_numpy()
    if rule == "positive_class":
        # Heart Disease: "num" in 0..4, where 0 is absence of disease.
        return (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    if rule == "wine_quality":
        # Wine Quality: quality score in 3..8, split at the conventional
        # "good wine" threshold of 6.
        return (pd.to_numeric(y, errors="coerce") >= 6).astype(int).to_numpy()
    raise ValueError(f"unknown binarisation rule {rule!r}")


def _encode_features(x: pd.DataFrame) -> np.ndarray:
    """Numeric matrix from a mixed-type frame.

    Categorical and object columns are ordinal-encoded rather than one-hot
    encoded, which is what keeps the feature counts at their raw column counts
    (Adult stays at 14 features, for example) and matches the input dimensions
    of the cached checkpoints.
    """
    encoded = pd.DataFrame(index=x.index)
    for column in x.columns:
        series = x[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
            encoded[column] = series.astype("category").cat.codes.astype(np.float64)
        else:
            encoded[column] = pd.to_numeric(series, errors="coerce").astype(np.float64)
    # Ordinal codes use -1 for missing; numeric columns may hold NaN. Both are
    # filled with the column median so that no record is silently dropped,
    # which would change the row count and the forget-set sizes.
    return encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0).to_numpy()


def _fetch_frame(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Raw features and target, from a local CSV if present else the source."""
    local = _data_dir() / f"{spec.key}.csv"
    if local.exists():
        frame = pd.read_csv(local)
        return frame.iloc[:, :-1], frame.iloc[:, -1]

    if spec.openml_id is None:
        bundle = load_breast_cancer(as_frame=True)
        return bundle.data, bundle.target

    try:
        bundle = fetch_openml(data_id=spec.openml_id, as_frame=True)
    except Exception as exc:  # network, version or parser failure
        raise RuntimeError(
            f"could not fetch {spec.display_name} (OpenML id {spec.openml_id}): {exc}\n"
            f"Place a CSV at {local} with the target in the last column to load "
            f"it offline."
        ) from exc
    return bundle.data, bundle.target


def load_dataset(name: str, *, validate: bool = True) -> TabularDataset:
    """Load, binarise, split and standardise one dataset.

    Parameters
    ----------
    name
        A key of :data:`DATASETS`.
    validate
        Enforce the expected row and feature counts. Leave this on unless you
        are deliberately working with a variant, since the cached checkpoints
        assume the published shapes.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {dataset_names()}")
    spec = DATASETS[name]

    x_raw, y_raw = _fetch_frame(spec)
    x_raw = x_raw.drop(columns=[c for c in spec.drop_columns if c in x_raw.columns])

    y = (
        _binarise(y_raw, spec.binarise)
        if spec.binarise
        else pd.Series(y_raw).astype("category").cat.codes.to_numpy()
    )
    x = _encode_features(x_raw)

    if validate:
        if x.shape[0] != spec.n_rows:
            raise ValueError(
                f"{spec.display_name}: expected {spec.n_rows} rows, got {x.shape[0]}. "
                "The source version differs from the one used in the paper; the "
                "cached checkpoints would not correspond to this data."
            )
        if x.shape[1] != spec.n_features:
            raise ValueError(
                f"{spec.display_name}: expected {spec.n_features} features, got "
                f"{x.shape[1]}."
            )
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"{spec.display_name}: target is not binary after preprocessing "
                f"({len(np.unique(y))} classes)."
            )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    scaler = StandardScaler().fit(x_train)
    return TabularDataset(
        key=spec.key,
        display_name=spec.display_name,
        x_train=scaler.transform(x_train).astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        x_test=scaler.transform(x_test).astype(np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
    )
