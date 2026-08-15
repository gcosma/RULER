"""The tutorial as one runnable script.

Run from the repository root:

    python tutorial/demo.py

This file contains exactly the code blocks of tutorial/README.md, in order.
Its printed output is compared against the tutorial by the test suite.
"""
# ruff: noqa: E402  -- imports appear where the tutorial introduces them

# Make the demo runnable from a bare clone, before `pip install`: put the
# repository root on the path so `import ruler` finds the library.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ======================================================================
# Part 1 — Data and the forget/retain split
# ======================================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
x = StandardScaler().fit_transform(data.data).astype(np.float32)
y = data.target.astype(np.int64)

rng = np.random.RandomState(0)
forget_idx = rng.choice(len(x), 30, replace=False)    # one erasure request: 30 records
retain_idx = np.setdiff1d(np.arange(len(x)), forget_idx)
print(f"{len(x)} records, {x.shape[1]} features: "
      f"{len(forget_idx)} to forget, {len(retain_idx)} to retain")


# ======================================================================
# Part 2 — The models
# ======================================================================

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, 2)
    def forward(self, z):
        return self.head(self.body(z))

def train(model, xs, ys, epochs=200, lr=1e-3):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.as_tensor(xs), torch.as_tensor(ys)
    for _ in range(epochs):
        optimiser.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        optimiser.step()
    return model.eval()

SEED = 0
torch.manual_seed(SEED)
original = train(MLP(), x, y)                          # trained on all records

torch.manual_seed(SEED)                                # the SAME seed -- rule 3
oracle = train(MLP(), x[retain_idx], y[retain_idx])    # never saw the forget records

print("original and oracle trained from the same initialisation seed")

import copy

unlearned = copy.deepcopy(original)
train(unlearned, x[retain_idx], y[retain_idx], epochs=20)
print("unlearned model ready")


# ======================================================================
# Part 3 — Extract the penultimate-layer embeddings
# ======================================================================

def embed(model, xs):
    model.eval()                                       # rule 1: deterministic
    with torch.no_grad():
        return model.body(torch.as_tensor(xs)).numpy()  # h(x), not the head

u_forget, u_retain = embed(unlearned, x[forget_idx]), embed(unlearned, x[retain_idx])
o_forget, o_retain = embed(oracle,    x[forget_idx]), embed(oracle,    x[retain_idx])
orig_forget        = embed(original,  x[forget_idx])

print("embedding shapes:", u_forget.shape, u_retain.shape)


# ======================================================================
# Part 4 — Verify
# ======================================================================

from ruler import m4

pre = m4(embed(original, x[forget_idx]), embed(original, x[retain_idx]))
print(f"pre-unlearning m4 (original model) = {pre:.4f}")

from ruler import all_metrics

scores = all_metrics(u_forget, u_retain, o_forget, o_retain, orig_forget)
for name, value in scores.items():
    print(f"  {name} = {value:+.4f}")


# ======================================================================
# Part 5 — Read against the null interval, not the point estimate
# ======================================================================

n = len(forget_idx)
half = 1.96 / np.sqrt(12 * n)
print(f"n = {n} forget records -> null interval 0.50 ± {half:.3f}")


# ======================================================================
# Part 6 — The two silent failure modes, demonstrated
# ======================================================================

unlearned.train()                                      # dropout active -- wrong
a = unlearned.body(torch.as_tensor(x[forget_idx])).detach().numpy()
b = unlearned.body(torch.as_tensor(x[forget_idx])).detach().numpy()
unlearned.eval()
print("same records, two calls, train mode -> identical:", np.allclose(a, b))
print("in eval mode                       -> identical:",
      np.allclose(embed(unlearned, x[forget_idx]), embed(unlearned, x[forget_idx])))

from ruler import m2

correct = m2(u_forget, o_forget, u_retain, o_retain)
shuffled = m2(u_forget, o_forget, u_retain[::-1], o_retain)
print(f"aligned retain rows:  m2 = {correct:+.5f}")
print(f"one side reversed:    m2 = {shuffled:+.5f}   <- no error, wrong number")
