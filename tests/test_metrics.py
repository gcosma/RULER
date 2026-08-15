"""Tests for the four RULER metrics.

These check the properties the paper's claims rest on: that each metric equals
its defining equation, sits at its null when the null genuinely holds, and moves
in the stated direction when it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from ruler import all_metrics, cosine_similarity, m1, m2, m3, m4


@pytest.fixture
def rng():
    return np.random.RandomState(0)


# ---------------------------------------------------------------------------
# cosine_similarity (Eq. 1)
# ---------------------------------------------------------------------------


def test_cosine_similarity_matches_the_definition(rng):
    a, b = rng.randn(20, 6), rng.randn(20, 6)
    expected = [
        np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]))
        for i in range(len(a))
    ]
    assert np.allclose(cosine_similarity(a, b), expected)


def test_cosine_similarity_ignores_magnitude(rng):
    """Section 3.2: embeddings are L2-normalised, so scale must not matter."""
    a, b = rng.randn(15, 5), rng.randn(15, 5)
    assert np.allclose(cosine_similarity(a, b), cosine_similarity(a * 7.5, b * 0.2))


def test_identical_embeddings_give_one(rng):
    a = rng.randn(10, 5)
    assert np.allclose(cosine_similarity(a, a), 1.0)


def test_zero_rows_do_not_produce_nan():
    """A dead ReLU unit can emit an all-zero embedding."""
    assert np.all(np.isfinite(cosine_similarity(np.zeros((3, 4)), np.ones((3, 4)))))


def test_misaligned_arrays_are_rejected(rng):
    with pytest.raises(ValueError, match="same records in the same order"):
        cosine_similarity(rng.randn(10, 5), rng.randn(11, 5))


def test_one_dimensional_input_is_rejected(rng):
    with pytest.raises(ValueError, match="2-D"):
        m4(rng.randn(10), rng.randn(50, 5))


def test_empty_input_is_rejected(rng):
    with pytest.raises(ValueError, match="empty"):
        m4(np.empty((0, 5)), rng.randn(50, 5))


# ---------------------------------------------------------------------------
# m1 and m2 (Eqs. 2, 3)
# ---------------------------------------------------------------------------


def test_m1_is_one_against_itself(rng):
    embeddings = rng.randn(12, 7)
    assert m1(embeddings, embeddings) == pytest.approx(1.0)


def test_m2_equals_m1_minus_the_retain_median(rng):
    """M2 = M1 - median retain similarity, exactly as Eq. 3 states."""
    u_forget, o_forget = rng.randn(9, 5), rng.randn(9, 5)
    u_retain, o_retain = rng.randn(40, 5), rng.randn(40, 5)

    expected = m1(u_forget, o_forget) - np.median(
        cosine_similarity(u_retain, o_retain)
    )
    assert m2(u_forget, o_forget, u_retain, o_retain) == pytest.approx(expected)


def test_m2_is_zero_when_forget_and_retain_are_exchangeable():
    """The null is attainable, not true only by construction (Appendix A.5)."""
    gaps = []
    for trial in range(60):
        trial_rng = np.random.RandomState(trial)
        records = trial_rng.randn(400, 16)
        # One shared pair of random "models": both sets pass through the same maps.
        map_u, map_o = trial_rng.randn(16, 16), trial_rng.randn(16, 16)
        forget, retain = records[:40], records[40:]
        gaps.append(
            m2(forget @ map_u, forget @ map_o, retain @ map_u, retain @ map_o)
        )
    assert abs(np.mean(gaps)) < 0.02


def test_m2_is_negative_when_forget_records_are_displaced(rng):
    records = rng.randn(300, 12)
    forget, retain = records[:30], records[30:]
    displaced = forget + rng.randn(*forget.shape) * 0.5
    assert m2(displaced, forget, retain, retain) < 0


def test_m2_uses_the_median_not_the_mean_baseline():
    """A right-skewed retain distribution must not drag the baseline upward.

    Appendix A.5: swapping the median for the mean shifts M2 by an amount
    comparable to the gap itself, so the two must differ here.
    """
    forget = np.tile([[1.0, 0.0]], (5, 1))
    oracle_forget = np.tile([[1.0, 0.05]], (5, 1))
    angles = np.concatenate([np.full(45, 0.4), np.full(5, 0.0)])
    retain = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    oracle_retain = np.tile([[1.0, 0.0]], (50, 1))

    with_median = m2(forget, oracle_forget, retain, oracle_retain)
    with_mean = m1(forget, oracle_forget) - np.mean(
        cosine_similarity(retain, oracle_retain)
    )
    assert with_median != pytest.approx(with_mean)


# ---------------------------------------------------------------------------
# m3 (Eq. 4)
# ---------------------------------------------------------------------------


def test_m3_is_zero_when_unlearning_changes_nothing(rng):
    original, oracle = rng.randn(20, 6), rng.randn(20, 6)
    assert m3(original, original, oracle) == pytest.approx(0.0)


def test_m3_is_positive_when_records_move_towards_the_oracle(rng):
    oracle = rng.randn(20, 6)
    original = oracle + rng.randn(20, 6) * 0.8
    closer = oracle + (original - oracle) * 0.1
    assert m3(closer, original, oracle) > 0


# ---------------------------------------------------------------------------
# m4 (Eqs. 5-7)
# ---------------------------------------------------------------------------


def test_m4_matches_the_naive_definition(rng):
    """The sorted-search implementation must equal the literal double loop."""
    forget, retain = rng.randn(11, 5), rng.randn(60, 5)

    f = forget / np.linalg.norm(forget, axis=1, keepdims=True)
    r = retain / np.linalg.norm(retain, axis=1, keepdims=True)
    similarity = r @ r.T
    np.fill_diagonal(similarity, -np.inf)
    s_retain = similarity.max(axis=1)                        # Eq. 6
    s_forget = (f @ r.T).max(axis=1)                         # Eq. 5
    expected = np.mean([np.mean(s_retain <= v) for v in s_forget])   # Eq. 7

    assert m4(forget, retain) == pytest.approx(expected)


def test_m4_sits_at_the_null_for_exchangeable_records():
    ranks = [
        m4(
            np.random.RandomState(t).randn(360, 10)[:40],
            np.random.RandomState(t).randn(360, 10)[40:],
        )
        for t in range(40)
    ]
    assert np.mean(ranks) == pytest.approx(0.50, abs=0.05)


def test_m4_is_high_under_residual_memorisation(rng):
    """Forget records still embedded where retained ones sit rank at the top."""
    retain = rng.randn(80, 8)
    assert m4(retain[:10] + 1e-6, retain) > 0.9


def test_m4_is_low_for_over_displaced_records(rng):
    retain = rng.randn(80, 8) * 0.1 + np.eye(8)[0] * 5
    forget = rng.randn(10, 8) * 0.1 + np.eye(8)[7] * 5
    assert m4(forget, retain) < 0.2


def test_m4_stays_within_zero_and_one(rng):
    for _ in range(20):
        assert 0.0 <= m4(rng.randn(7, 4), rng.randn(50, 4)) <= 1.0


def test_m4_excludes_self_matches_on_the_retain_side(rng):
    """Eq. 6 is leave-one-out: a retain record must not match itself at 1.0.

    Without the exclusion every retain neighbour similarity would be exactly
    1.0, so no independent forget record could ever reach the distribution and
    every rank would collapse to 0. Independent records landing near 0.50 is
    what shows the exclusion is applied.
    """
    records = rng.randn(400, 6)
    forget, retain = records[:40], records[40:]
    assert m4(forget, retain) == pytest.approx(0.50, abs=0.20)


def test_m4_is_one_when_the_forget_records_are_in_the_gallery(rng):
    """A forget record has no self-match to exclude, so an exact retain-set
    match ranks above every leave-one-out neighbour similarity."""
    retain = rng.randn(40, 6)
    assert m4(retain[:5], retain) == pytest.approx(1.0)


def test_m4_rejects_a_degenerate_retain_set(rng):
    with pytest.raises(ValueError, match="at least 2 retain records"):
        m4(rng.randn(5, 3), rng.randn(1, 3))


def test_m4_rejects_mismatched_dimensions(rng):
    with pytest.raises(ValueError, match="dimension mismatch"):
        m4(rng.randn(5, 3), rng.randn(20, 4))


def test_m4_handles_a_large_gallery_without_a_full_matrix(rng):
    """Chunking means a big retain set must not blow up."""
    assert 0.0 <= m4(rng.randn(20, 16), rng.randn(20000, 16)) <= 1.0


# ---------------------------------------------------------------------------
# all_metrics
# ---------------------------------------------------------------------------


def test_all_metrics_agrees_with_the_individual_functions(rng):
    u_forget, u_retain = rng.randn(25, 8), rng.randn(200, 8)
    o_forget, o_retain = rng.randn(25, 8), rng.randn(200, 8)
    original = rng.randn(25, 8)

    scores = all_metrics(u_forget, u_retain, o_forget, o_retain, original)
    assert scores["m1"] == pytest.approx(m1(u_forget, o_forget))
    assert scores["m2"] == pytest.approx(m2(u_forget, o_forget, u_retain, o_retain))
    assert scores["m3"] == pytest.approx(m3(u_forget, original, o_forget))
    assert scores["m4"] == pytest.approx(m4(u_forget, u_retain))


def test_all_metrics_withholds_m3_without_the_original(rng):
    """No 'before' means no shift; None rather than a misleading zero."""
    u_forget, u_retain = rng.randn(25, 8), rng.randn(200, 8)
    scores = all_metrics(u_forget, u_retain, u_forget, u_retain)
    assert scores["m3"] is None
    assert scores["m1"] == pytest.approx(1.0)


def test_oracle_against_itself_gives_a_zero_gap(rng):
    """The internal consistency check: M1 = 1 and M2 = 0."""
    forget, retain = rng.randn(25, 10), rng.randn(200, 10)
    scores = all_metrics(forget, retain, forget, retain, forget)
    assert scores["m1"] == pytest.approx(1.0)
    assert scores["m2"] == pytest.approx(0.0, abs=1e-9)
    assert scores["m3"] == pytest.approx(0.0, abs=1e-9)


def test_library_needs_only_numpy():
    """The point of the split: no framework, no paper code, no heavy stack."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, numpy as np, ruler;"
         "r = np.random.RandomState(0).randn(300, 8);"
         "ruler.m4(r[:30], r[30:]);"
         "heavy = [m for m in ('torch','pandas','sklearn','statsmodels','scipy',"
         "'matplotlib') if m in sys.modules];"
         "print(','.join(heavy))"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"pulled in: {result.stdout.strip()}"


# ---------------------------------------------------------------------------
# Non-finite inputs must be refused, not silently scored
# ---------------------------------------------------------------------------


def test_nan_embeddings_are_refused(rng):
    """A NaN row would otherwise flow through searchsorted into a
    plausible-looking rank -- a fabricated finding, not an error."""
    retain = rng.randn(100, 8)
    bad = rng.randn(10, 8)
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf in 1 row"):
        m4(bad, retain)
    with pytest.raises(ValueError, match="NaN or inf"):
        m2(bad, retain[:10], retain, retain)


def test_inf_embeddings_are_refused(rng):
    bad = rng.randn(10, 8)
    bad[5, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        m4(bad, rng.randn(100, 8))


def test_norm_overflow_is_refused(rng):
    """1e300-scale rows overflow the squared norm and would silently
    normalise to zero vectors, yielding m4 = 0 from nothing."""
    huge = rng.randn(10, 8) * 1e300
    with pytest.raises(ValueError, match="overflows float64"):
        m4(huge, rng.randn(100, 8))


def test_ordinary_dtypes_and_layouts_still_pass(rng):
    retain = rng.randn(100, 8)
    assert 0.0 <= m4(rng.randint(-5, 5, (20, 8)), retain) <= 1.0        # ints
    assert 0.0 <= m4(rng.randn(20, 8).astype(np.float16), retain) <= 1.0
    assert 0.0 <= m4(rng.randn(40, 16)[::2, ::2], retain) <= 1.0        # views


def test_cosine_similarity_never_leaves_its_mathematical_range(rng):
    """Dot-product rounding can exceed 1.0 by a few ulps; the result is clipped.

    Without the clip, cosine_similarity(a, a) returns values like
    1.0000000000000004 for a substantial fraction of rows.
    """
    a = rng.randn(1000, 8)
    self_similarity = cosine_similarity(a, a)
    assert (self_similarity <= 1.0).all()
    assert (self_similarity >= -1.0).all()
    assert np.allclose(self_similarity, 1.0)
    assert np.allclose(cosine_similarity(a, -a), -1.0)


def test_norm_underflow_is_refused(rng):
    """The mirror of overflow: 1e-300-scale rows have squared norms below
    float64's smallest denormal, and would otherwise score as m4 = 1.0 --
    a fabricated 'fully memorised' verdict from pure underflow."""
    tiny = rng.randn(10, 8) * 1e-300
    with pytest.raises(ValueError, match="underflows float64"):
        m4(tiny, rng.randn(100, 8))
    # exact-zero rows (a dead ReLU) remain legitimate input
    with_dead_row = np.vstack([rng.randn(9, 8), np.zeros(8)])
    assert 0.0 <= m4(with_dead_row, rng.randn(100, 8)) <= 1.0
