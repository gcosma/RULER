"""The four RULER metrics.

Each takes penultimate-layer embeddings as ``(n, p)`` arrays and returns a
number. Nothing here knows about models, datasets, frameworks or training --
you run your experiment however you like and hand over the embeddings.

Which layer
-----------
**The activation immediately before the task-specific output head** (Section
3.2). Formally, if your model is ``f = g(h(x))`` with ``g`` the output head,
these metrics measure ``h(x)``:

===============================  ==========================================
Architecture                     Layer
===============================  ==========================================
MLP                              the last hidden activation (second ReLU in
                                 the paper's tabular MLP)
Residual MLP / FT-Transformer    the final hidden layer
Three-layer CNN                  the second fully-connected layer (256-d)
ResNet-18                        the post-global-average-pooling activation
                                 (512-d), not the last convolutional block
BERT-family                      the ``[CLS]`` output of the final
                                 transformer layer (768-d), not the LM head
===============================  ==========================================

Not an earlier layer: those share lower-level features across records and do
not separate individuals. Not the output layer: it collapses the representation
into logits or per-token distributions, discarding the geometric structure the
metrics read.

The embedding dimension is whatever the architecture gives -- 128, 256, 512,
768 -- and need only be consistent within a single comparison. Embeddings are
L2-normalised internally, so their scale does not matter.

Three of the metrics compare an unlearned model against a *retrain oracle*, a
model trained from scratch on the retain set alone:

    m1(unlearned_forget, oracle_forget)                           Eq. 2
    m2(unlearned_forget, oracle_forget,
       unlearned_retain, oracle_retain)                           Eq. 3, null 0
    m3(unlearned_forget, original_forget, oracle_forget)          Eq. 4, null 0

The fourth needs no oracle -- only the unlearned model's own geometry:

    m4(forget, retain)                                            Eqs. 5-7, null 0.50

``all_metrics`` computes all four in one call.

Two things the caller is responsible for
----------------------------------------
**Paired seeds.** ``m1``, ``m2`` and ``m3`` compare two models. Cosine
similarity is not rotation-invariant, so the original model and the oracle must
be trained from the same random initialisation; otherwise the difference you
measure is initialisation geometry, not unlearning (0.99 similarity when
paired, 0.44 when not).

**Row alignment.** ``*_retain`` arrays are compared row by row across models,
so they must hold the same records in the same order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["m1", "m2", "m3", "m4", "all_metrics", "cosine_similarity"]

# Guards division by zero for an all-zero embedding, which a dead ReLU unit can
# produce. Such a row stays at zero rather than becoming NaN.
_EPS = 1e-12


def _normalise(embeddings, name: str) -> np.ndarray:
    """Validate shape and finiteness, then scale each row to unit length.

    Finiteness is checked because the alternative is silent garbage, not an
    error: a NaN row propagates through the similarity matrix into
    ``searchsorted`` and produces a plausible-looking rank, and a magnitude
    whose squared norm overflows float64 silently normalises to a zero row.
    Both come from real situations -- a diverged model, or embeddings taken
    from the wrong tensor -- and both would otherwise be read as findings.
    """
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (n, p) array, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        n_bad = int(np.sum(~np.isfinite(array).all(axis=1)))
        raise ValueError(
            f"{name} contains NaN or inf in {n_bad} row(s). Embeddings from a "
            "diverged model (or the wrong tensor) would otherwise produce a "
            "plausible-looking metric value rather than an error."
        )
    with np.errstate(over="ignore"):     # overflow is diagnosed below, not warned
        norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError(
            f"{name} has row magnitudes whose squared norm overflows float64; "
            "rescale the embeddings (their scale does not affect the metrics)."
        )
    # The mirror case: rows around 1e-300 have squared norms below float64's
    # smallest denormal, so a genuinely nonzero row gets norm 0 and would be
    # scored as garbage (an all-underflowed forget set yields m4 = 1.0).
    underflowed = (norms.ravel() == 0.0) & np.any(array != 0.0, axis=1)
    if underflowed.any():
        raise ValueError(
            f"{name} has {int(underflowed.sum())} nonzero row(s) whose squared "
            "norm underflows float64 to zero; rescale the embeddings (their "
            "scale does not affect the metrics)."
        )
    return array / np.maximum(norms, _EPS)


def cosine_similarity(a, b) -> np.ndarray:
    """Row-wise cosine similarity between two aligned sets of embeddings (Eq. 1).

    Row *i* of ``a`` is compared with row *i* of ``b``. Returns a length-``n``
    array.
    """
    left, right = _normalise(a, "a"), _normalise(b, "b")
    if left.shape != right.shape:
        raise ValueError(
            f"shape mismatch: {left.shape} vs {right.shape}. These must hold the "
            "same records in the same order."
        )
    # Rounding in the dot product can land a few float64 ulps outside the
    # mathematical range; clip so the bound holds exactly.
    return np.clip(np.sum(left * right, axis=1), -1.0, 1.0)


def m1(unlearned_forget, oracle_forget) -> float:
    """Mean similarity of forget records to the oracle (Eq. 2).

    Near 1.0 means the unlearned model places forget records where the oracle
    does. M1 has no fixed null -- its expected value depends on the dataset and
    seed -- so use ``m2``, which calibrates it, for anything you want to test.

    Parameters
    ----------
    unlearned_forget, oracle_forget
        The forget records' **penultimate-layer** embeddings -- the activation
        immediately before the output head -- under the unlearned model and
        under the retrain oracle, in the same order.
    """
    return float(np.mean(cosine_similarity(unlearned_forget, oracle_forget)))


def m2(unlearned_forget, oracle_forget, unlearned_retain, oracle_retain) -> float:
    """Signed calibration gap (Eq. 3). **Null 0.**

    ``M1`` minus the *median* similarity that retain records achieve between the
    same two models.

    - **Negative**: forget records sit further from the oracle than retained
      records do -- residual memorisation.
    - **Positive**: unlearning pushed them closer to the oracle than retained
      records -- over-correction.

    The median is deliberate on the retain side: that distribution is
    right-skewed, and a handful of records with unusually high similarity
    inflate the mean by about the size of the gap itself, masking the signal.
    The forget side uses the mean because the forget set is small and every
    record, including outliers that may signal incomplete erasure, should count.

    Parameters
    ----------
    unlearned_forget, oracle_forget
        The forget records' **penultimate-layer** embeddings under the
        unlearned model and the oracle.
    unlearned_retain, oracle_retain
        The retain records under the same two models. Compared row by row, so
        both must hold the same records in the same order.

    Only meaningful under the paired-seed design; see the module docstring.
    """
    forget_similarity = cosine_similarity(unlearned_forget, oracle_forget)
    retain_similarity = cosine_similarity(unlearned_retain, oracle_retain)
    return float(np.mean(forget_similarity) - np.median(retain_similarity))


def m3(unlearned_forget, original_forget, oracle_forget) -> float:
    """Representation shift towards the oracle (Eq. 4). **Null 0.**

    How far unlearning moved forget records towards where the oracle puts them.
    Positive is the intended direction; negative means they moved further away.

    Parameters
    ----------
    unlearned_forget, original_forget, oracle_forget
        The same forget records' **penultimate-layer** embeddings under the
        unlearned model, the original model before unlearning, and the oracle.
    """
    after = cosine_similarity(unlearned_forget, oracle_forget)
    before = cosine_similarity(original_forget, oracle_forget)
    return float(np.mean(after - before))


def _nearest_neighbour(query: np.ndarray, gallery: np.ndarray, exclude_self: bool):
    """Highest cosine similarity from each query row to the gallery.

    Computed in row chunks, so a large gallery never materialises an n x n
    matrix.
    """
    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = max(1, min(512, query.shape[0]))
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        block = query[start:stop] @ gallery.T
        if exclude_self:
            rows = np.arange(stop - start)
            block[rows, rows + start] = -np.inf
        out[start:stop] = block.max(axis=1)
    return out


def m4(forget, retain) -> float:
    """Oracle-free percentile rank (Eqs. 5-7). **Null 0.50.**

    For each forget record, take its similarity to the nearest retain record,
    then rank that value within the retain records' own leave-one-out
    nearest-neighbour similarities. A record that blends into the retain set
    sits at the median of that distribution, giving 0.50.

    - **Above 0.50**: the forget record is closer to the retain manifold than a
      typical retained record -- residual memorisation.
    - **Below 0.50**: it has been pushed further from the retain distribution
      than correct retraining would leave it -- over-displacement.

    Parameters
    ----------
    forget, retain
        **Penultimate-layer** embeddings -- the activation immediately before
        the output head -- of the forget and retain records, both under the
        *same* model, which must be in evaluation mode.

    No oracle is involved, so this also works on the original model as a
    **pre-unlearning diagnostic**: near 0.50 there means the records were never
    memorised, and unlearning has nothing to remove.

    The leave-one-out exclusion matters: a retain record would otherwise match
    itself at 1.0, so both quantities measure similarity to the closest
    *distinct* retain record.

    Note on sample size: this is a mean over forget records, and under the null
    each contributes a uniform value, so its standard deviation is
    ``1/sqrt(12n)``. On 10 forget records that is 0.09 -- values between about
    0.32 and 0.68 are consistent with no memorisation at all. Read small forget
    sets with that in mind.
    """
    forget_embeddings = _normalise(forget, "forget")
    retain_embeddings = _normalise(retain, "retain")
    if forget_embeddings.shape[1] != retain_embeddings.shape[1]:
        raise ValueError(
            f"embedding dimension mismatch: forget {forget_embeddings.shape[1]} "
            f"vs retain {retain_embeddings.shape[1]}"
        )
    if retain_embeddings.shape[0] < 2:
        raise ValueError(
            "m4 needs at least 2 retain records to form a leave-one-out "
            f"nearest-neighbour distribution, got {retain_embeddings.shape[0]}"
        )

    forget_nn = _nearest_neighbour(forget_embeddings, retain_embeddings, False)
    retain_nn = _nearest_neighbour(retain_embeddings, retain_embeddings, True)

    # Eq. 7 counts retain records whose own neighbour similarity is <= each
    # forget record's. Sorting once turns that into a binary search.
    ranks = np.searchsorted(np.sort(retain_nn), forget_nn, side="right")
    return float(np.mean(ranks / len(retain_nn)))


def all_metrics(
    unlearned_forget,
    unlearned_retain,
    oracle_forget,
    oracle_retain,
    original_forget=None,
) -> dict:
    """All four metrics in one call.

    Parameters
    ----------
    unlearned_forget, unlearned_retain
        Forget and retain records embedded under the unlearned model, at the
        **penultimate layer** -- the activation immediately before the output
        head.
    oracle_forget, oracle_retain
        The same records under the retrain oracle, in the same order.
    original_forget
        Forget records under the original model. Optional; without it ``m3``
        is returned as ``None`` rather than a misleading zero.

    Returns
    -------
    dict with keys ``m1``, ``m2``, ``m3``, ``m4``.

    Examples
    --------
    >>> scores = all_metrics(u_forget, u_retain, o_forget, o_retain, orig_forget)
    >>> scores["m2"], scores["m4"]     # doctest: +SKIP
    (-0.0076, 0.612)
    """
    return {
        "m1": m1(unlearned_forget, oracle_forget),
        "m2": m2(unlearned_forget, oracle_forget, unlearned_retain, oracle_retain),
        "m3": (
            None
            if original_forget is None
            else m3(unlearned_forget, original_forget, oracle_forget)
        ),
        "m4": m4(unlearned_forget, unlearned_retain),
    }
