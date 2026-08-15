"""Approximate unlearning methods (paper Section 4.2 and Appendix A.11).

All methods start from a copy of the original model and use Adam at a fixed
unlearning learning rate, held constant across datasets and forget fractions.
Each is deterministic given the unlearning seed, which is why unlearned models
are recomputed rather than cached.

Four methods modify the original model's parameters directly or by
distillation from it:

* :func:`gradient_ascent` -- ascent on the forget set only.
* :func:`neggrad_plus`    -- ascent on forget, descent on retain.
* :func:`fine_tuning`     -- descent on retain only, relying on catastrophic
  forgetting.
* :func:`scrub`           -- distillation from the frozen original.

A fifth, :func:`bad_teacher`, uses a fundamentally different mechanism -- a
*randomly initialised* teacher -- and is evaluated separately (Section 5.4) to
test whether the paper's discordance is specific to any one algorithm.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from .config import (
    ALPHA,
    BAD_TEACHER_SEED,
    GA_EPOCHS,
    METHODS,
    TEMPERATURE,
    UNLEARN_EPOCHS,
    UNLEARN_LR,
    UNLEARN_SEED,
)
from .models import TabularMLP

__all__ = [
    "gradient_ascent",
    "neggrad_plus",
    "fine_tuning",
    "scrub",
    "bad_teacher",
    "UNLEARN_METHODS",
    "apply_unlearning",
]


def _prepare(model: TabularMLP, seed: int, device) -> tuple[TabularMLP, torch.device]:
    """Seeded working copy of the original model, in training mode.

    Dropout is active during unlearning, so seeding here is what makes each
    method reproducible from a given original model.
    """
    device = torch.device(device)
    torch.manual_seed(seed)
    student = copy.deepcopy(model).to(device)
    student.train()
    return student, device


def _tensors(x, y, device) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    if y is None:
        return x_t, None
    return x_t, torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)


def _batches(n: int, batch_size: int | None, device):
    """Index batches for one epoch; a single full-batch pass when unbatched."""
    if batch_size is None:
        yield slice(None)
        return
    order = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _kl_to_teacher(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL(teacher || student) on temperature-softened distributions.

    Scaled by ``T**2`` so the gradient magnitude is comparable to an
    unsoftened objective, the standard distillation convention.
    """
    student_log_p = F.log_softmax(student_logits / temperature, dim=1)
    teacher_p = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_p, teacher_p, reduction="batchmean") * temperature**2


# ---------------------------------------------------------------------------
# Gradient-based methods
# ---------------------------------------------------------------------------


def gradient_ascent(
    model: TabularMLP,
    x_forget,
    y_forget,
    *,
    epochs: int = GA_EPOCHS,
    lr: float = UNLEARN_LR,
    seed: int = UNLEARN_SEED,
    batch_size: int | None = None,
    device="cpu",
) -> TabularMLP:
    """Maximise cross-entropy loss on the forget set."""
    student, device = _prepare(model, seed, device)
    xf, yf = _tensors(x_forget, y_forget, device)
    optimiser = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for idx in _batches(len(xf), batch_size, device):
            optimiser.zero_grad()
            # Ascent on the forget loss: descend its negation.
            (-criterion(student(xf[idx]), yf[idx])).backward()
            optimiser.step()

    student.eval()
    return student


def neggrad_plus(
    model: TabularMLP,
    x_forget,
    y_forget,
    x_retain,
    y_retain,
    *,
    epochs: int = UNLEARN_EPOCHS,
    lr: float = UNLEARN_LR,
    alpha: float = ALPHA,
    seed: int = UNLEARN_SEED,
    batch_size: int | None = None,
    device="cpu",
) -> TabularMLP:
    """Descent on the retain set combined with ascent on the forget set.

    The objective is ``alpha * CE(retain) - (1 - alpha) * CE(forget)``: the
    retain term preserves predictive performance while the forget term pushes
    the model away from the records being erased.
    """
    student, device = _prepare(model, seed, device)
    xf, yf = _tensors(x_forget, y_forget, device)
    xr, yr = _tensors(x_retain, y_retain, device)
    optimiser = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for idx in _batches(len(xr), batch_size, device):
            optimiser.zero_grad()
            retain_loss = criterion(student(xr[idx]), yr[idx])
            forget_loss = criterion(student(xf), yf)
            (alpha * retain_loss - (1 - alpha) * forget_loss).backward()
            optimiser.step()

    student.eval()
    return student


def fine_tuning(
    model: TabularMLP,
    x_retain,
    y_retain,
    *,
    epochs: int = UNLEARN_EPOCHS,
    lr: float = UNLEARN_LR,
    seed: int = UNLEARN_SEED,
    batch_size: int | None = None,
    device="cpu",
) -> TabularMLP:
    """Continue gradient descent on the retain set alone.

    Nothing in this objective refers to the forget set: erasure is left to
    catastrophic forgetting.
    """
    student, device = _prepare(model, seed, device)
    xr, yr = _tensors(x_retain, y_retain, device)
    optimiser = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for idx in _batches(len(xr), batch_size, device):
            optimiser.zero_grad()
            criterion(student(xr[idx]), yr[idx]).backward()
            optimiser.step()

    student.eval()
    return student


def scrub(
    model: TabularMLP,
    x_forget,
    x_retain,
    y_retain,
    *,
    epochs: int = UNLEARN_EPOCHS,
    lr: float = UNLEARN_LR,
    alpha: float = ALPHA,
    temperature: float = TEMPERATURE,
    seed: int = UNLEARN_SEED,
    batch_size: int | None = None,
    device="cpu",
) -> TabularMLP:
    """Distil from the frozen original model into a student copy.

    The student is pulled towards the teacher on retain records and pushed away
    from it on forget records:
    ``alpha * KL(retain) - (1 - alpha) * KL(forget)``.

    The teacher is the *original* model, so the objective is expressed purely
    in output-distribution terms and never refers to the representation space
    the RULER metrics measure.
    """
    student, device = _prepare(model, seed, device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    xf, _ = _tensors(x_forget, None, device)
    xr, _ = _tensors(x_retain, y_retain, device)
    optimiser = optim.Adam(student.parameters(), lr=lr)

    with torch.no_grad():
        teacher_forget = teacher(xf)

    for _ in range(epochs):
        for idx in _batches(len(xr), batch_size, device):
            optimiser.zero_grad()
            with torch.no_grad():
                teacher_retain = teacher(xr[idx])
            retain_kl = _kl_to_teacher(student(xr[idx]), teacher_retain, temperature)
            forget_kl = _kl_to_teacher(student(xf), teacher_forget, temperature)
            (alpha * retain_kl - (1 - alpha) * forget_kl).backward()
            optimiser.step()

    student.eval()
    return student


def bad_teacher(
    model: TabularMLP,
    x_forget,
    x_retain,
    y_retain,
    *,
    epochs: int = UNLEARN_EPOCHS,
    lr: float = UNLEARN_LR,
    alpha: float = ALPHA,
    temperature: float = TEMPERATURE,
    seed: int = UNLEARN_SEED,
    teacher_seed: int = BAD_TEACHER_SEED,
    batch_size: int | None = None,
    device="cpu",
) -> TabularMLP:
    """Bad Teacher: distil towards a *randomly initialised* teacher.

    Adapted from Chundawat et al. to a single-teacher objective. On forget
    records the student is trained to match the random teacher's output
    distribution; on retain records it minimises both the KL divergence to that
    teacher and the cross-entropy loss on the true labels, so the retain set is
    preserved by the labels rather than by a competent teacher. The original
    model supplies only the student's initialisation.

    Because it erases by imitating noise rather than by ascending the forget
    loss, this method tests whether the paper's discordance is a property of
    the unlearning task rather than of any single algorithm.

    ``teacher_seed`` selects the random teacher's initialisation; Appendix A.11
    repeats the analysis over seeds 101 and 102 to confirm the findings do not
    depend on it.
    """
    student, device = _prepare(model, seed, device)

    torch.manual_seed(teacher_seed)
    teacher = TabularMLP(model.input_dim).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    xf, _ = _tensors(x_forget, None, device)
    xr, yr = _tensors(x_retain, y_retain, device)
    optimiser = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        teacher_forget = teacher(xf)

    for _ in range(epochs):
        for idx in _batches(len(xr), batch_size, device):
            optimiser.zero_grad()
            with torch.no_grad():
                teacher_retain = teacher(xr[idx])
            retain_logits = student(xr[idx])
            retain_loss = alpha * criterion(retain_logits, yr[idx]) + (
                1 - alpha
            ) * _kl_to_teacher(retain_logits, teacher_retain, temperature)
            forget_loss = _kl_to_teacher(student(xf), teacher_forget, temperature)
            (retain_loss + forget_loss).backward()
            optimiser.step()

    student.eval()
    return student


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

#: Every implemented method: the four of the primary evaluation plus Bad
#: Teacher. Derived from ``config.METHODS`` so the two cannot drift apart.
UNLEARN_METHODS = METHODS + ("Bad Teacher",)


def apply_unlearning(
    method: str,
    model: TabularMLP,
    *,
    x_forget,
    y_forget,
    x_retain,
    y_retain,
    **kwargs,
) -> TabularMLP:
    """Apply a named unlearning method, passing through shared keyword options."""
    if method == "Gradient Ascent":
        return gradient_ascent(model, x_forget, y_forget, **kwargs)
    if method == "NegGrad+":
        return neggrad_plus(model, x_forget, y_forget, x_retain, y_retain, **kwargs)
    if method == "Fine-Tuning":
        return fine_tuning(model, x_retain, y_retain, **kwargs)
    if method == "SCRUB":
        return scrub(model, x_forget, x_retain, y_retain, **kwargs)
    if method == "Bad Teacher":
        return bad_teacher(model, x_forget, x_retain, y_retain, **kwargs)
    raise ValueError(f"unknown unlearning method {method!r}; expected one of {UNLEARN_METHODS}")
