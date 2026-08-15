# RULER tutorial

A complete first run, start to finish: what the metrics see, a full experiment
on a small model, how to read the numbers, and the two mistakes that silently
corrupt results. Every code block below has been executed exactly as shown, and
every printed output is real.

You need `numpy` for Part 1 and `torch` for Parts 2–5:

```text
pip install ruler-unlearning torch
```

---

## Part 1 — What the metrics see

Before touching a model, build the three situations RULER distinguishes, out
of raw arrays. `m4` needs no oracle: it asks whether the "forgotten" records
are still distinguishable from the retained ones.

```python
import numpy as np
from ruler import m4

rng = np.random.RandomState(0)
retain = rng.randn(1000, 64)                      # 1,000 retained records

# Scenario A: residual memorisation -- the model still embeds the "forgotten"
# records where the retained ones sit
memorised = retain[:50] + rng.randn(50, 64) * 0.01

# Scenario B: records drawn from the same distribution -- genuinely blended in
erased = rng.randn(50, 64)

# Scenario C: records pushed into a subspace the retain set never occupies --
# under cosine similarity, "far away" means orthogonal, not large
displaced = rng.randn(50, 64)
displaced[:, :32] = 0.0                           # lives only in dims 32-63
subspace_retain = retain.copy()
subspace_retain[:, 32:] = 0.0                     # retain lives in dims 0-31

for label, forget, gallery in [("memorised", memorised, retain),
                               ("erased", erased, retain),
                               ("over-displaced", displaced, subspace_retain)]:
    print(f"{label:>15}:  m4 = {m4(forget, gallery):.3f}")
```

```text
      memorised:  m4 = 1.000
         erased:  m4 = 0.520
 over-displaced:  m4 = 0.000
```

Reading against the null of **0.50**:

- **1.000** — every forget record's nearest-retain-neighbour similarity
  exceeds that of the median retained record: residual memorisation.
- **0.520** — indistinguishable from retained records: what genuine erasure
  looks like.
- **0.000** — pushed further out than any retained record: over-displacement,
  the failure that erases *too hard* and that output-level metrics never see.

Note scenario C: under cosine similarity, displacement means pointing in a
*direction* the retain set doesn't use — not being numerically large. Scale is
normalised away.

## Part 2 — A real experiment: original model and paired oracle

Now the full workflow on a small PyTorch MLP. Two rules are baked in here and
explained as they appear.

```python
import torch
from torch import nn

def make_data(seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(400, 20).astype(np.float32)
    y = (x[:, :3].sum(axis=1) > 0).astype(np.int64)
    return x, y

x, y = make_data()
forget_idx = np.arange(0, 40)          # the 40 records we will "delete"
retain_idx = np.arange(40, 400)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, 2)
    def forward(self, z):
        return self.head(self.body(z))

def train(model, xs, ys, epochs=300, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.as_tensor(xs), torch.as_tensor(ys)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        opt.step()
    model.eval()
    return model

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
original = train(MLP(), x, y)                              # sees everything

torch.manual_seed(SEED); np.random.seed(SEED)              # the SAME seed
oracle = train(MLP(), x[retain_idx], y[retain_idx])        # never sees forget

print("trained: original (400 records) and oracle (360 retain records)")
```

```text
trained: original (400 records) and oracle (360 retain records)
```

**Why the same seed twice?** Cosine similarity is not rotation-invariant.
Two networks initialised differently end up in different representational
geometries even when trained on identical data — same-seed pairs sit at ~0.99
cross-model similarity, differently-seeded pairs at ~0.44. If the oracle is
not seed-paired with the original, `m2` measures initialisation, not
unlearning. This decision happens *before training* and cannot be repaired
afterwards.

**Why train the oracle from scratch?** Fine-tuning a copy of the original on
the retain set would carry the forget set's influence straight into the model
that is supposed to be free of it.

## Part 3 — Unlearn, however you like

RULER has no opinion about your unlearning method. Here, the simplest one:
fine-tuning on the retain set only, relying on catastrophic forgetting.

```python
import copy

unlearned = copy.deepcopy(original)
train(unlearned, x[retain_idx], y[retain_idx], epochs=30, lr=1e-3)
print("unlearned: 30 epochs of fine-tuning on the retain set only")
```

```text
unlearned: 30 epochs of fine-tuning on the retain set only
```

## Part 4 — Extract the penultimate embeddings

The metrics measure `h(x)` where your model is `f(x) = g(h(x))` — the
activation immediately before the output head. For this MLP that is `body`,
not `head`.

```python
def embed(model, xs):
    model.eval()                                   # rule 1: deterministic
    with torch.no_grad():
        return model.body(torch.as_tensor(xs)).numpy()   # h(x), not the head

u_forget, u_retain = embed(unlearned, x[forget_idx]), embed(unlearned, x[retain_idx])
o_forget, o_retain = embed(oracle,    x[forget_idx]), embed(oracle,    x[retain_idx])
orig_forget        = embed(original,  x[forget_idx])

print("embedding shapes:", u_forget.shape, u_retain.shape)
```

```text
embedding shapes: (40, 32) (360, 32)
```

For other architectures the same rule picks the layer: post-global-average-
pooling for ResNet-18, the `[CLS]` vector of the final transformer layer for
BERT. See the [manual, §4](manual.md#4-getting-the-embeddings).

## Part 5 — All four metrics

```python
from ruler import all_metrics

scores = all_metrics(u_forget, u_retain, o_forget, o_retain, orig_forget)
for name, value in scores.items():
    print(f"  {name} = {value:+.4f}")

pre = m4(embed(original, x[forget_idx]), embed(original, x[retain_idx]))
print(f"  pre-unlearning m4 (original model) = {pre:.4f}")
```

```text
  m1 = +0.9613
  m2 = -0.0353
  m3 = +0.0019
  m4 = +0.4858
  pre-unlearning m4 (original model) = 0.5044
```

Reading these, in the order that matters:

- **Pre-unlearning `m4` = 0.5044** — at the null. The original model never
  memorised these 40 records at the representation level; they were learnable
  from the general rule. There was nothing to erase. (Run this check *before*
  unlearning in real work: the paper shows methods manufacturing artefacts —
  `m4` driven from 0.50 to 0.16 — where nothing was memorised.)
- **`m2` = −0.0353** — negative: after unlearning, the forget records sit
  further from the oracle than retained records do. Fine-tuning left a
  residual — exactly the paper's finding for this method.
- **`m3` = +0.0019** — unlearning moved the records marginally towards the
  oracle. Direction fine, magnitude tiny.
- **`m4` = 0.4858** — near the null; on its own it says nothing (next part).
- **`m1` = 0.9613** — context only: high absolute similarity, which is why
  the calibrated `m2` is the number to test, not `m1`.

## Part 6 — Read against the null interval, not the point estimate

`m4` is a mean over forget records; under the null each contributes a uniform
value, so its standard deviation is `1/sqrt(12n)`.

```python
n = len(u_forget)
half = 1.96 / np.sqrt(12 * n)
inside = abs(scores["m4"] - 0.50) <= half
verdict = ("INSIDE the band: inconclusive on its own" if inside
           else "OUTSIDE the band: a real deviation")
print(f"n = {n} forget records -> null interval 0.50 ± {half:.3f}")
print(f"m4 = {scores['m4']:.3f} -> {verdict}")
```

```text
n = 40 forget records -> null interval 0.50 ± 0.089
m4 = 0.486 -> INSIDE the band: inconclusive on its own
```

Forty records can only resolve deviations larger than ±0.089. This is why a
single small erasure request rarely supports a per-request verdict — and why
the memorised scenario in Part 1 (m4 = 1.000 on 50 records) *is* conclusive:
it is far outside any band.

## Part 7 — The two silent failure modes, demonstrated

Neither of these raises an error. Both corrupt the numbers.

### Forgetting `model.eval()`

```python
sloppy = copy.deepcopy(original)
sloppy.train()                                     # dropout-style nondeterminism
with torch.no_grad():
    a = sloppy.body(torch.as_tensor(x[:5])).numpy()
    b = sloppy.body(torch.as_tensor(x[:5])).numpy()
print("train mode: identical calls give identical embeddings?", np.allclose(a, b))

sloppy.eval()                                      # the fix is one call
with torch.no_grad():
    a = sloppy.body(torch.as_tensor(x[:5])).numpy()
    b = sloppy.body(torch.as_tensor(x[:5])).numpy()
print("eval mode:  identical calls give identical embeddings?", np.allclose(a, b))
```

```text
train mode: identical calls give identical embeddings? False
eval mode:  identical calls give identical embeddings? True
```

In train mode the same record embeds differently on every call — every metric
downstream becomes noise, and results stop reproducing. The two-line check
above is worth running once in any new pipeline.

### Reordering the retain rows between models

`m2` compares the retain arrays row by row: row *i* under the unlearned model
against row *i* under the oracle — the *same record* in both.

```python
from ruler import m2

honest = m2(u_forget, o_forget, u_retain, o_retain)

shuffle = np.random.RandomState(1).permutation(len(u_retain))
corrupt = m2(u_forget, o_forget, u_retain[shuffle], o_retain)   # rows misaligned

print(f"aligned retain rows:  m2 = {honest:+.5f}")
print(f"shuffled retain rows: m2 = {corrupt:+.5f}   <- silently wrong, no error")
```

```text
aligned retain rows:  m2 = -0.03529
shuffled retain rows: m2 = +0.07650   <- silently wrong, no error
```

The sign flipped: residual memorisation became apparent over-correction, with
no exception raised — the shapes still match, so the library cannot tell. Keep
one index order from the moment you split the data, and never re-shuffle
between embedding passes.

---

## Where next

- The [user manual](manual.md) — full API reference, oracle construction,
  troubleshooting table.
- [Choosing inputs](choosing-inputs.md) — what counts as one record, which
  layer per architecture, the four invariants.
- The paper's own experiments live in `paper/` in the repository, with the
  commands to reproduce every table in the README.
