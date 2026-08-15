"""Tests for the tabular MLP, checkpoint loading and unlearning.

Several of these run against the real cached checkpoints, which is the only way
to confirm that the reconstructed architecture and the penultimate-layer choice
match the models the paper's results were computed from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paper.config import HIDDEN_DIM, OUTPUT_DIM, TRAIN_SEEDS
from ruler import cosine_similarity
from paper.models import TabularMLP, penultimate, predict_proba
from paper.train import checkpoint_path, load_model, train_model
from paper.unlearn import apply_unlearning


@pytest.fixture
def toy_data():
    """A separable two-class problem, so training visibly succeeds."""
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.randn(120, 6) - 1.2, rng.randn(120, 6) + 1.2]).astype(
        np.float32
    )
    y = np.concatenate([np.zeros(120), np.ones(120)]).astype(np.int64)
    return x, y


def test_architecture_matches_figure_2():
    """d -> 128 -> 128 -> 2, with dropout after each ReLU."""
    model = TabularMLP(14)
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (14, HIDDEN_DIM),
        (HIDDEN_DIM, HIDDEN_DIM),
        (HIDDEN_DIM, OUTPUT_DIM),
    ]
    assert sum(isinstance(m, torch.nn.Dropout) for m in model.net) == 2
    assert sum(isinstance(m, torch.nn.ReLU) for m in model.net) == 2


def test_checkpoint_keys_match_the_cached_layout():
    """Cached checkpoints use net.0 / net.3 / net.6; the module must too."""
    assert set(TabularMLP(5).state_dict()) == {
        "net.0.weight", "net.0.bias",
        "net.3.weight", "net.3.bias",
        "net.6.weight", "net.6.bias",
    }


def test_penultimate_is_the_second_relu_output():
    """RULER measures the activation just before the output head (Fig. 2)."""
    model = TabularMLP(4).eval()
    x = torch.randn(9, 4)
    assert torch.allclose(model.penultimate(x), model.net[:5](x))


def test_penultimate_is_non_negative_and_correctly_shaped():
    """It is a ReLU output, so it cannot be negative."""
    embeddings = penultimate(TabularMLP(4).eval(), np.random.randn(9, 4))
    assert embeddings.shape == (9, HIDDEN_DIM)
    assert (embeddings >= 0).all()


def test_penultimate_disables_dropout():
    """Embeddings must be deterministic: the metrics compare them across models."""
    model = TabularMLP(4)
    model.train()  # dropout active
    x = np.random.randn(20, 4)
    assert np.allclose(penultimate(model, x), penultimate(model, x))
    assert model.training, "training mode should be restored afterwards"


def test_predict_proba_returns_a_distribution():
    probabilities = predict_proba(TabularMLP(4).eval(), np.random.randn(12, 4))
    assert probabilities.shape == (12, OUTPUT_DIM)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_training_is_reproducible_at_a_fixed_seed(toy_data):
    x, y = toy_data
    first = train_model(x, y, seed=3, epochs=5)
    second = train_model(x, y, seed=3, epochs=5)
    assert np.allclose(penultimate(first, x), penultimate(second, x))


def test_training_learns_a_separable_problem(toy_data):
    x, y = toy_data
    model = train_model(x, y, seed=0, epochs=200)
    accuracy = (predict_proba(model, x).argmax(axis=1) == y).mean()
    assert accuracy > 0.9


def test_load_model_rejects_a_dimension_mismatch(tmp_path):
    torch.save(TabularMLP(10).state_dict(), tmp_path / "m.pt")
    with pytest.raises(ValueError, match="different version of this dataset"):
        load_model(tmp_path / "m.pt", input_dim=7)


def test_checkpoint_path_naming():
    assert checkpoint_path("ckpt", "adult", 3, "orig").name == "adult_seed3_orig.pt"
    assert (
        checkpoint_path("ckpt", "adult", 3, "oracle", 0.05).name
        == "adult_seed3_oracle_ff05.pt"
    )
    assert (
        checkpoint_path("ckpt", "adult", 3, "oracle", 0.10).name
        == "adult_seed3_oracle_ff10.pt"
    )
    with pytest.raises(ValueError):
        checkpoint_path("ckpt", "adult", 3, "oracle")


# ---------------------------------------------------------------------------
# Against the real cached checkpoints
# ---------------------------------------------------------------------------


def test_cached_checkpoints_load_into_the_reconstructed_architecture(
    checkpoints_dir, has_checkpoints
):
    if not has_checkpoints:
        pytest.skip("no cached checkpoints")
    model = load_model(checkpoints_dir / "breast_cancer_seed0_orig.pt", 30)
    assert model.input_dim == 30
    assert penultimate(model, np.random.randn(5, 30)).shape == (5, HIDDEN_DIM)


def test_paired_seed_design_holds_in_the_cached_checkpoints(
    checkpoints_dir, has_checkpoints
):
    """Appendix A.16: same-seed pairs are far more similar than different-seed pairs.

    The paper reports a grand mean of 0.993 for same-seed original-oracle pairs
    against 0.431 for differently-seeded oracle pairs. Reproducing that gap
    confirms both the paired-seed design and the penultimate-layer choice; a
    wrong layer would not separate the two conditions.
    """
    if not has_checkpoints:
        pytest.skip("no cached checkpoints")

    x = np.random.RandomState(0).randn(400, 30).astype(np.float32)
    originals, oracles = {}, {}
    for seed in TRAIN_SEEDS:
        original = checkpoints_dir / f"breast_cancer_seed{seed}_orig.pt"
        oracle = checkpoints_dir / f"breast_cancer_seed{seed}_oracle_ff05.pt"
        if not (original.exists() and oracle.exists()):
            pytest.skip("incomplete checkpoint set")
        originals[seed] = penultimate(load_model(original, 30), x)
        oracles[seed] = penultimate(load_model(oracle, 30), x)

    same = [
        cosine_similarity(originals[s], oracles[s]).mean() for s in originals
    ]
    different = [
        cosine_similarity(oracles[i], oracles[j]).mean()
        for i in oracles
        for j in oracles
        if i < j
    ]
    assert np.mean(same) > 0.95
    assert np.mean(different) < np.mean(same) - 0.3


# ---------------------------------------------------------------------------
# Unlearning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB", "Bad Teacher"]
)
def test_unlearning_runs_and_leaves_the_original_untouched(method, toy_data):
    """Each method must return a new model without mutating the original."""
    x, y = toy_data
    original = train_model(x, y, seed=0, epochs=20)
    before = penultimate(original, x).copy()

    unlearned = apply_unlearning(
        method,
        original,
        x_forget=x[:20],
        y_forget=y[:20],
        x_retain=x[20:],
        y_retain=y[20:],
        epochs=2,
    )
    assert unlearned is not original
    assert np.allclose(penultimate(original, x), before), "original was modified"
    assert np.isfinite(penultimate(unlearned, x)).all()


@pytest.mark.parametrize("method", ["Gradient Ascent", "NegGrad+", "SCRUB"])
def test_unlearning_is_deterministic_given_the_seed(method, toy_data):
    """Unlearned models are recomputed rather than cached, so this must hold."""
    x, y = toy_data
    original = train_model(x, y, seed=0, epochs=20)
    kwargs = {"x_forget": x[:20], "y_forget": y[:20],
              "x_retain": x[20:], "y_retain": y[20:], "epochs": 2}
    first = apply_unlearning(method, original, **kwargs)
    second = apply_unlearning(method, original, **kwargs)
    assert np.allclose(penultimate(first, x), penultimate(second, x))


def test_gradient_ascent_degrades_forget_set_accuracy(toy_data):
    """Ascending the forget loss must actually raise it."""
    x, y = toy_data
    original = train_model(x, y, seed=0, epochs=150)
    x_forget, y_forget = x[:30], y[:30]
    before = (predict_proba(original, x_forget).argmax(axis=1) == y_forget).mean()

    unlearned = apply_unlearning(
        "Gradient Ascent",
        original,
        x_forget=x_forget,
        y_forget=y_forget,
        x_retain=x[30:],
        y_retain=y[30:],
        lr=1e-2,
        epochs=30,
    )
    after = (predict_proba(unlearned, x_forget).argmax(axis=1) == y_forget).mean()
    assert after < before


def test_unknown_method_is_rejected(toy_data):
    x, y = toy_data
    with pytest.raises(ValueError, match="unknown unlearning method"):
        apply_unlearning(
            "Magic",
            TabularMLP(6),
            x_forget=x[:5],
            y_forget=y[:5],
            x_retain=x[5:],
            y_retain=y[5:],
        )
