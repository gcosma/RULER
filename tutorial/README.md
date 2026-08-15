# RULER tutorial

One complete verification, start to finish, on one small dataset. Total
runtime is under a minute on a laptop CPU. Every code block below has been
executed exactly as shown, and every printed output is real.

The starting point: **you have already run your unlearning experiment.** You
trained a model, applied an unlearning method — Bad Teacher, SCRUB, gradient
ascent, fine-tuning, or another — and now hold the resulting models. RULER
reads only the embeddings extracted from your models; the unlearning method
itself stays outside it.

| You provide | Needed for |
|---|---|
| the unlearned model | everything, including `m4` on its own |
| the original model | `m3` and the pre-unlearning check |
| a retrain oracle, trained from scratch on the retain set with the **same initialisation seed** as the original | `m1`, `m2`, `m3` |
| the forget records and retain records | everything |

The dataset here is Breast Cancer, which comes with scikit-learn and loads
directly. The models are small MLPs that train in seconds, standing in for
yours: wherever a model is built below, that is a placeholder for a model you
already have.

```text
pip install git+https://github.com/gcosma/RULER.git torch scikit-learn
```

---

## Part 1 — Data and the forget/retain split

```python
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
```

```text
569 records, 30 features: 30 to forget, 539 to retain
```

The 30 forget records represent one erasure request. In your own data,
everything one request covers — a patient, a document, a person's images —
must sit wholly on one side of this split.

## Part 2 — The models

In real work these already exist. Two rules from the paper are built into the
stand-ins, and both are decisions fixed before training:

- **Train the oracle from scratch on the retain set.** A fine-tuned copy of the
  original would carry the forget records' influence into the model that is
  meant to be free of it.
- **Give the oracle the original model's initialisation seed.** Cosine
  similarity depends on the orientation of the two representations: same-seed
  pairs reach roughly 0.99 cross-model similarity, differently-seeded pairs
  roughly 0.43 (paper Appendix A.16). With an unpaired oracle, `m2` measures
  initialisation geometry rather than unlearning.

```python
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
```

```text
original and oracle trained from the same initialisation seed
```

### The unlearned model — yours goes here

This is the model your unlearning method produced. The stand-in below is
brief fine-tuning on the retain set; **replace it with the output of Bad
Teacher, SCRUB, gradient ascent, or whatever method you are verifying.**
Everything downstream stays the same — the metrics read the embeddings whatever
method produced them.

```python
import copy

unlearned = copy.deepcopy(original)
train(unlearned, x[retain_idx], y[retain_idx], epochs=20)
print("unlearned model ready")
```

```text
unlearned model ready
```

## Part 3 — Extract the penultimate-layer embeddings

The metrics measure `h(x)` where your model is `f(x) = g(h(x))` — the
activation immediately before the output head. For this MLP that is `body`,
the layers ahead of `head`.

```python
def embed(model, xs):
    model.eval()                                       # rule 1: deterministic
    with torch.no_grad():
        return model.body(torch.as_tensor(xs)).numpy()  # h(x), not the head

u_forget, u_retain = embed(unlearned, x[forget_idx]), embed(unlearned, x[retain_idx])
o_forget, o_retain = embed(oracle,    x[forget_idx]), embed(oracle,    x[retain_idx])
orig_forget        = embed(original,  x[forget_idx])

print("embedding shapes:", u_forget.shape, u_retain.shape)
```

```text
embedding shapes: (30, 32) (539, 32)
```

The same rule picks the layer in any architecture:
post-global-average-pooling for ResNet-18, the `[CLS]` vector of the final
transformer layer for BERT. See the table in the
[README](../README.md#obtaining-the-embeddings).

## Part 4 — Verify

### First: was anything memorised to begin with?

`m4` uses a single model, so you can run it on the *original* model to decide
whether unlearning is needed at all — before applying any method:

```python
from ruler import m4

pre = m4(embed(original, x[forget_idx]), embed(original, x[retain_idx]))
print(f"pre-unlearning m4 (original model) = {pre:.4f}")
```

```text
pre-unlearning m4 (original model) = 0.4699
```

Near the null of 0.50: the original model already places these 30 records
among the retain records — they were learnable from the general rule, so
little remains for unlearning to remove. Run this check before unlearning in
real work. In the paper's experiments, one method drove an untouched 0.50 down
to 0.16, creating a representational artefact in records that were already
indistinguishable.

### All four metrics

```python
from ruler import all_metrics

scores = all_metrics(u_forget, u_retain, o_forget, o_retain, orig_forget)
for name, value in scores.items():
    print(f"  {name} = {value:+.4f}")
```

```text
  m1 = +0.9905
  m2 = -0.0068
  m3 = +0.0018
  m4 = +0.4795
```

Reading them:

- **`m2` = −0.0068** — slightly negative: the forget records sit marginally
  further from the oracle than retained records do. A small residual, in the
  direction the paper reports for fine-tuning-style methods.
- **`m3` = +0.0018** — unlearning moved the records towards the oracle;
  right direction, tiny magnitude.
- **`m4` = 0.4795** — near the null, consistent with the pre-unlearning
  check: the forget records blend into the retain distribution, showing no
  residual memorisation (and no over-displacement below the null either).
- **`m1` = 0.9905** — context only. High absolute similarity is expected
  under the paired-seed design; `m2` is `m1` with the baseline subtracted.

## Part 5 — Read against the null interval

`m4` is a mean over forget records; under the null each contributes a
uniform value, so its standard deviation is `1/sqrt(12n)`:

```python
n = len(forget_idx)
half = 1.96 / np.sqrt(12 * n)
print(f"n = {n} forget records -> null interval 0.50 ± {half:.3f}")
```

```text
n = 30 forget records -> null interval 0.50 ± 0.103
```

Both `m4` values above (0.4699 and 0.4795) are inside 0.50 ± 0.103: at this
sample size, neither is evidence of anything. That is the honest reading —
with 30 records the metric resolves only large effects. To resolve small
ones, aggregate several erasure requests (129 records puts the interval at
±0.05, 801 at ±0.02).

## Part 6 — The two silent failure modes, demonstrated

The library validates what the arrays can show and raises named errors.
These two mistakes produce **plausible wrong numbers**, so they are
demonstrated here rather than only described.

### Failure 1: embedding in training mode

```python
unlearned.train()                                      # dropout active -- wrong
a = unlearned.body(torch.as_tensor(x[forget_idx])).detach().numpy()
b = unlearned.body(torch.as_tensor(x[forget_idx])).detach().numpy()
unlearned.eval()
print("same records, two calls, train mode -> identical:", np.allclose(a, b))
print("in eval mode                       -> identical:",
      np.allclose(embed(unlearned, x[forget_idx]), embed(unlearned, x[forget_idx])))
```

```text
same records, two calls, train mode -> identical: False
in eval mode                       -> identical: True
```

In training mode the same record embeds differently on every call, and every
metric silently absorbs that noise. Call `model.eval()` before embedding —
the `embed()` helper above does it for you.

### Failure 2: retain rows out of order

`m2` compares retain rows one-to-one across the two models. Reverse one side
and the call still succeeds — the number is wrong:

```python
from ruler import m2

correct = m2(u_forget, o_forget, u_retain, o_retain)
shuffled = m2(u_forget, o_forget, u_retain[::-1], o_retain)
print(f"aligned retain rows:  m2 = {correct:+.5f}")
print(f"one side reversed:    m2 = {shuffled:+.5f}   <- no error, wrong number")
```

```text
aligned retain rows:  m2 = -0.00680
one side reversed:    m2 = +0.34071   <- no error, wrong number
```

A −0.007 residual became a +0.341 "over-correction" purely from the row
order. Embed the same records, in the same order, under every model.

---

## Reproducing a paper result

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gcosma/RULER/blob/main/tutorial/reproduce_breast_cancer.ipynb)

[`reproduce_breast_cancer.ipynb`](reproduce_breast_cancer.ipynb) reproduces the
paper's result on its smallest tabular dataset, using the pre-trained models in
[`checkpoints/`](../checkpoints) and computing every metric with the library. It
averages the pre-unlearning `m4` over all ten training seeds (reproducing the
paper's ~0.60 for Breast Cancer) and computes `m1`-`m4` for all four unlearning
methods, in the shape of the paper's Table 1 — in about ten seconds on a
laptop CPU. Open it in Jupyter, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    tutorial/reproduce_breast_cancer.ipynb
```

---

That is the whole workflow: split at the erasure request, provide your models,
extract penultimate-layer embeddings, compute the metrics, read them against
their nulls. The full API is documented in each function's docstring
(`help(ruler.m2)`), and the [README](../README.md) covers the embedding layer
for each architecture and the conditions for valid results.
