"""Training of the original models and retrain oracles (paper Section 4.1).

Three model types are used throughout:

* the **original** model, trained on the full training set;
* the **retrain oracle**, trained from scratch on the retain set alone -- the
  gold-standard reference, a model with no knowledge of the forget set;
* the **unlearned** model, produced by applying an unlearning algorithm to the
  original model (see :mod:`ruler.unlearn`).

For a given training seed the original model and the oracle are initialised
with the *same* seed.  This paired-seed design (Section 3.3) is what makes
Lens 1 meaningful: cosine similarity is not rotation-invariant, so without a
shared initialisation the representational difference between two models would
be dominated by geometric variation across random initialisations rather than
by the unlearning procedure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from .config import BATCH_SIZE, TRAIN_EPOCHS, TRAIN_LR
from .models import TabularMLP, set_seed

__all__ = [
    "train_model",
    "checkpoint_path",
    "save_model",
    "load_model",
    "get_or_train",
]


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    input_dim: int | None = None,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    batch_size: int | None = BATCH_SIZE,
    device: torch.device | str = "cpu",
) -> TabularMLP:
    """Train a :class:`~ruler.models.TabularMLP` with Adam and cross-entropy.

    Parameters
    ----------
    batch_size
        ``None`` gives full-batch gradient descent, the primary setting: it
        removes optimisation noise so that representational differences
        isolate unlearning dynamics. Appendix A.6 repeats the ff = 5%
        experiment with mini-batches of 128 to confirm robustness.
    """
    device = torch.device(device)
    set_seed(seed)

    model = TabularMLP(input_dim or x.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)

    model.train()
    for _ in range(epochs):
        if batch_size is None:
            optimiser.zero_grad()
            criterion(model(x_t), y_t).backward()
            optimiser.step()
            continue
        order = torch.randperm(len(x_t), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimiser.zero_grad()
            criterion(model(x_t[idx]), y_t[idx]).backward()
            optimiser.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint cache
# ---------------------------------------------------------------------------
#
# Naming matches the checkpoints shipped in ``checkpoints/``:
#   <dataset>_seed<i>_orig.pt
#   <dataset>_seed<i>_oracle_ff<01|05|10>.pt
# Unlearned models are not cached: they are cheap to recompute from the
# original model and are fully determined by it plus the fixed unlearning seed.


def format_forget_fraction(forget_fraction: float) -> str:
    """Filename fragment for a forget fraction, without collisions.

    The paper's fractions keep their two-digit form so the shipped checkpoints
    resolve: 0.01 -> ``ff01``. Anything else is written precisely, with the
    decimal point as ``p`` (0.125 -> ``ff12p5``), since rounding alone would map
    both 0.12 and 0.125 to ``ff12`` and silently overwrite one oracle.
    """
    percent = forget_fraction * 100.0
    rounded = round(percent)
    if abs(percent - rounded) < 1e-9 and 1 <= rounded <= 99:
        return f"ff{rounded:02d}"
    return f"ff{percent:g}".replace(".", "p")


def checkpoint_path(
    checkpoint_dir: str | Path,
    dataset: str,
    seed: int,
    kind: str = "orig",
    forget_fraction: float | None = None,
) -> Path:
    """Path for a cached model.

    Parameters
    ----------
    kind
        ``"orig"`` or ``"oracle"``. An oracle also requires ``forget_fraction``
        since a separate oracle is trained per forget fraction.
    """
    if kind == "orig":
        name = f"{dataset}_seed{seed}_orig.pt"
    elif kind == "oracle":
        if forget_fraction is None:
            raise ValueError("an oracle checkpoint requires a forget fraction")
        # Delegated so non-paper fractions cannot collide: rounding to two
        # digits would map both 0.12 and 0.125 to "ff12", silently overwriting
        # one oracle with another.
        name = f"{dataset}_seed{seed}_oracle_{format_forget_fraction(forget_fraction)}.pt"
    else:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    return Path(checkpoint_dir) / name


def save_model(model: TabularMLP, path: str | Path) -> None:
    """Write a model's weights, creating the parent directory if needed.

    Saves a bare ``state_dict``, matching the format of the checkpoints shipped
    with the paper. For a checkpoint that records its own seed, role and forget
    set, use :func:`ruler.checkpoints.save_checkpoint`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path, input_dim: int, device: torch.device | str = "cpu"
) -> TabularMLP:
    """Load a cached model, checking it was trained on matching features."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    cached_dim = state["net.0.weight"].shape[1]
    if cached_dim != input_dim:
        raise ValueError(
            f"{Path(path).name} expects {cached_dim} input features but the "
            f"dataset provides {input_dim}. The checkpoint was trained on a "
            f"different version of this dataset."
        )
    model = TabularMLP(input_dim)
    model.load_state_dict(state)
    model.to(torch.device(device)).eval()
    return model


def get_or_train(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    input_dim: int,
    device: torch.device | str = "cpu",
    **train_kwargs,
) -> tuple[TabularMLP, bool]:
    """Load a cached model if present, otherwise train it and cache it.

    Returns ``(model, was_cached)``.
    """
    path = Path(path)
    if path.exists():
        return load_model(path, input_dim, device=device), True
    model = train_model(
        x, y, seed=seed, input_dim=input_dim, device=device, **train_kwargs
    )
    save_model(model, path)
    return model, False
