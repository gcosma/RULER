"""The tabular MLP used in the primary experiments (paper Fig. 2).

The architecture is deliberately held constant across all ten datasets and all
three model types (original, unlearned, retrain oracle) so that representational
differences reflect the unlearning procedure rather than architectural variation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import DROPOUT, HIDDEN_DIM, OUTPUT_DIM

__all__ = ["TabularMLP", "penultimate", "predict_proba", "set_seed"]

#: Index one past the penultimate activation in ``TabularMLP.net``.
#:
#: ``net`` is [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear], so
#: ``net[:5]`` ends at the second ReLU -- the final feature abstraction before
#: the output head, which is what RULER measures. Earlier layers share
#: lower-level features across records and the output layer collapses the
#: representation into two logits, discarding the geometry the metrics need.
_PENULTIMATE_END = 5


def set_seed(seed: int) -> None:
    """Seed torch and numpy for reproducible initialisation and training."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class TabularMLP(nn.Module):
    """Two-hidden-layer MLP, ``d -> 128 -> 128 -> 2`` with dropout after each ReLU.

    The module is a single ``nn.Sequential`` named ``net`` so that checkpoint
    keys are ``net.0``, ``net.3`` and ``net.6``; the cached checkpoints in
    ``checkpoints/`` use exactly this layout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer activation ``h(x)``, shape ``(n, hidden_dim)``."""
        return self.net[:_PENULTIMATE_END](x)


def _as_tensor(x, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


@torch.no_grad()
def penultimate(
    model: TabularMLP, x, device: torch.device | str = "cpu", batch_size: int = 8192
) -> np.ndarray:
    """Extract penultimate-layer embeddings as a ``(n, hidden_dim)`` array.

    Evaluation mode is forced: dropout must be inactive or the "embedding" of a
    record would differ between calls, and the metrics compare embeddings
    across models record by record.
    """
    was_training = model.training
    model.eval()
    device = torch.device(device)
    model.to(device)
    x = _as_tensor(x, device)
    chunks = [
        model.penultimate(x[i : i + batch_size]).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.hidden_dim))


@torch.no_grad()
def predict_proba(
    model: TabularMLP, x, device: torch.device | str = "cpu", batch_size: int = 8192
) -> np.ndarray:
    """Softmax class probabilities, shape ``(n, output_dim)``."""
    was_training = model.training
    model.eval()
    device = torch.device(device)
    model.to(device)
    x = _as_tensor(x, device)
    chunks = [
        torch.softmax(model(x[i : i + batch_size]), dim=1).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, OUTPUT_DIM))
