# RULER user manual

Everything you need to use the library correctly, in one place. For a guided
first run, start with the [tutorial](tutorial.md) instead.

RULER answers one question: **do the forgotten records still leave a trace in
the model's internal representations?** You run your own experiment — your
model, your data, your unlearning method — and hand the library
penultimate-layer embeddings. It returns four numbers and the nulls to read
them against.

---

## Contents

1. [Installation](#1-installation)
2. [Concepts and terminology](#2-concepts-and-terminology)
3. [API reference](#3-api-reference)
4. [Getting the embeddings](#4-getting-the-embeddings)
5. [Building a retrain oracle](#5-building-a-retrain-oracle)
6. [Interpreting results](#6-interpreting-results)
7. [Errors and troubleshooting](#7-errors-and-troubleshooting)
8. [Performance and memory](#8-performance-and-memory)
9. [Limitations](#9-limitations)
10. [FAQ](#10-faq)

---

## 1. Installation

```python
# pip install ruler-unlearning
import ruler
print(ruler.__version__)
```

The library depends on **numpy only**. It sits alongside PyTorch, JAX,
TensorFlow or scikit-learn without pulling in any framework; whatever produced
your embeddings is invisible to it. Python 3.10+.

## 2. Concepts and terminology

| Term | Meaning |
|---|---|
| **forget set** | The training records the model was asked to erase. |
| **retain set** | The training records it was allowed to keep. Always *training* data — never held-out test data. |
| **original model** | The model before unlearning, trained on forget + retain. |
| **unlearned model** | The model after your unlearning method ran. |
| **retrain oracle** | A model trained from scratch on the retain set alone — the gold standard that genuinely never saw the forget set. |
| **penultimate layer** | The activation immediately before the task-specific output head: `h(x)` where your model is `f(x) = g(h(x))`. All metrics operate on it. |
| **erasure request** | What one deletion request covers — a row, a patient, a person's photos. Must never be split across forget and retain. |

The four metrics differ in what evidence they need and what null they are read
against:

| Metric | Needs | Null | Negative / low means | Positive / high means |
|---|---|---|---|---|
| `m4` | unlearned model only | 0.50 | over-displacement | residual memorisation |
| `m2` | + retrain oracle | 0 | residual memorisation | over-correction |
| `m3` | + original model | 0 | moved away from oracle | moved towards oracle |
| `m1` | + retrain oracle | — | (no fixed null; context only) | |

## 3. API reference

Every function takes `(n, p)` arrays — rows are records, columns are embedding
dimensions. Lists of lists are accepted and converted. Embeddings are
L2-normalised internally, so their scale never matters.

The examples below share this setup:

```python
import numpy as np
from ruler import m1, m2, m3, m4, all_metrics, cosine_similarity

rng = np.random.RandomState(0)
u_forget  = rng.randn(40, 64)     # forget records under the unlearned model
u_retain  = rng.randn(500, 64)    # retain records under the unlearned model
o_forget  = rng.randn(40, 64)     # the same forget records under the oracle
o_retain  = rng.randn(500, 64)    # the same retain records under the oracle
orig_forget = rng.randn(40, 64)   # forget records under the original model
```

### `m4(forget, retain) -> float`

The oracle-free percentile rank (paper Eqs. 5–7). **Null 0.50.**

For each forget record: take its cosine similarity to its nearest retain
record, then rank that value within the retain records' own leave-one-out
nearest-neighbour similarities. The result is the mean rank over forget
records. A record that has genuinely blended in sits at the median: 0.50.

```python
score = m4(u_forget, u_retain)
print(round(score, 4))
```

- Both arrays must come from the **same** model.
- `retain` needs at least 2 rows (the leave-one-out distribution needs a
  neighbour to exclude).
- Also works on the **original** model as a pre-unlearning diagnostic: near
  0.50 there means the records were never memorised and there is nothing to
  erase.

### `m2(unlearned_forget, oracle_forget, unlearned_retain, oracle_retain) -> float`

The signed calibration gap (Eq. 3). **Null 0.** The paper's primary metric.

Mean forget-record similarity to the oracle, minus the **median** similarity
that retain records achieve between the same two models. The retain baseline
calibrates away everything except what unlearning did.

```python
gap = m2(u_forget, o_forget, u_retain, o_retain)
print(round(gap, 5))
```

- The retain arrays are compared **row by row**: same records, same order, or
  the number is silently wrong.
- Only meaningful when the original model and the oracle share a training
  seed (see [§5](#5-building-a-retrain-oracle)).
- The median (not mean) baseline is deliberate: the retain similarity
  distribution is right-skewed, and a mean baseline is inflated by roughly the
  size of the gap itself.

### `m3(unlearned_forget, original_forget, oracle_forget) -> float`

Representation shift towards the oracle (Eq. 4). **Null 0.**

```python
shift = m3(u_forget, orig_forget, o_forget)
print(round(shift, 5))
```

Positive: unlearning moved the forget records towards where the oracle puts
them — the intended direction. Negative: it moved them further away. `m2`
tells you where you ended up; `m3` tells you which way you travelled.

### `m1(unlearned_forget, oracle_forget) -> float`

Mean forget-record similarity to the oracle (Eq. 2). **No fixed null** — its
expected value depends on dataset and seed, which is exactly what `m2` fixes
by subtracting the retain baseline. Reported for context.

```python
similarity = m1(u_forget, o_forget)
print(round(similarity, 4))
```

### `all_metrics(u_forget, u_retain, o_forget, o_retain, original_forget=None) -> dict`

All four in one call, as `{"m1": ..., "m2": ..., "m3": ..., "m4": ...}`.

```python
scores = all_metrics(u_forget, u_retain, o_forget, o_retain, orig_forget)
print({k: None if v is None else round(v, 4) for k, v in scores.items()})

without_original = all_metrics(u_forget, u_retain, o_forget, o_retain)
print(without_original["m3"])     # None — not a misleading zero
```

`m3` is `None` when no original model is supplied, never a fabricated value.

### `cosine_similarity(a, b) -> np.ndarray`

The row-wise similarity primitive the metrics are built on (Eq. 1). Row *i* of
`a` against row *i* of `b`; returns a length-`n` array. Exposed for your own
diagnostics.

```python
sims = cosine_similarity(u_forget, o_forget)
print(sims.shape, round(float(sims.mean()), 4))
```

## 4. Getting the embeddings

The one piece of work the library leaves to you, because only you know your
architecture.

**Which layer:** the activation immediately before the task-specific output
head — `h(x)` where `f(x) = g(h(x))`. Not earlier (low-level features are
shared across records and do not separate individuals) and not the output
(logits discard the geometry).

| Architecture | Penultimate layer | Typical dim |
|---|---|---|
| MLP | last hidden activation | — |
| Residual MLP / FT-Transformer | final hidden layer | 128 |
| Small CNN | last fully-connected layer before the classifier | 256 |
| ResNet-18 | post-global-average-pooling — **not** the last conv block | 512 |
| BERT-family | `[CLS]` of the final transformer layer — **not** the LM head | 768 |

```python
# requires: your model
import torch

model.eval()                                       # 1. required — see below
with torch.no_grad():                              # 2. no gradients needed

    # ResNet-style: everything up to the classifier
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    embeddings = backbone(x).flatten(1).cpu().numpy()

    # BERT-style (from a BertForMaskedLM: .bert is the encoder, .cls the head)
    embeddings = model.bert(**batch).last_hidden_state[:, 0, :].cpu().numpy()
```

**Three rules, all silent if broken:**

1. **`model.eval()` first.** Dropout or a live batch-norm update makes the
   same record embed differently on every call, and every metric compares
   embeddings across records or models. Symptom: results change between runs
   with nothing else changed.
2. **One model per array.** `m4`'s two arrays come from the same model; `m2`'s
   four come from exactly two models, paired as named.
3. **Never split an erasure request.** If deletion means a whole patient,
   document or identity, every one of its records goes on one side. Otherwise
   the "forgotten" rows keep near neighbours from the same unit in the retain
   set, and `m4` measures your split rather than the model.

## 5. Building a retrain oracle

Needed only for `m1`, `m2`, `m3`. Two requirements, neither checkable from
the weights afterwards:

1. **Same initialisation seed as the original model.** Cosine similarity is
   not rotation-invariant; two independently initialised networks differ by
   geometry alone (same-seed pairs sit at ~0.99 cross-model similarity,
   differently-seeded pairs at ~0.44). An unpaired oracle measures
   initialisation, not unlearning.
2. **Trained from scratch on the retain set** — never fine-tuned from the
   original model, which would carry the forget set's influence straight into
   the reference.

```python
# requires: your training code
torch.manual_seed(SEED); np.random.seed(SEED)
original = train(fresh_model(), x_all, y_all)

torch.manual_seed(SEED); np.random.seed(SEED)     # the SAME seed
oracle = train(fresh_model(), x_retain, y_retain)
```

**Sanity-check the pairing before trusting `m2`.** Train two or more oracles
under *different* seeds, none of which saw the forget set, and apply `m2` to
each pair. Nothing was unlearned, so the values should centre on zero
(paper Appendix A.5). A centre far from zero means the setup is biased —
usually unpaired initialisation — and every result would read as residual
memorisation:

```python
# requires: several oracles
from itertools import combinations

gaps = [m2(fa, fb, ra, rb)
        for (fa, ra), (fb, rb) in combinations(oracle_embeddings, 2)]
print(np.mean(gaps))     # should be close to zero
```

## 6. Interpreting results

### The nulls

| Metric | Null | Above | Below |
|---|---|---|---|
| `m4` | 0.50 | residual memorisation | over-displacement |
| `m2` | 0 | over-correction | residual memorisation |
| `m3` | 0 | moved towards oracle (good) | moved away (bad) |

Note `m4` and `m2` point in **opposite directions**: high `m4` and *negative*
`m2` both indicate incomplete erasure.

### Sample size sets what you can resolve

`m4` is a mean over forget records; under the null each contributes a uniform
value, so its standard deviation is `1/sqrt(12n)`. On a small forget set a
striking value can be pure noise:

| forget records | values consistent with **no** memorisation (95%) |
|---|---|
| 10 | 0.50 ± 0.18 |
| 33 | 0.50 ± 0.10 |
| 129 | 0.50 ± 0.05 |
| 801 | 0.50 ± 0.02 |

```python
n = len(u_forget)
half_width = 1.96 / np.sqrt(12 * n)
print(f"null interval at n={n}: 0.50 ± {half_width:.3f}")
```

An individual erasure request is usually tens of records — often below the
metric's resolution. Aggregate several requests, or ask "is this model
leaking?" rather than "did this one deletion work?".

### A reading checklist

1. Is the value outside the null interval for your `n`? If not, the honest answer
   is *inconclusive*, whatever the point estimate looks like.
2. If you have an oracle, `m2` leads: it is calibrated and compares against
   the gold standard. `m4` corroborates.
3. `m4` far **below** 0.50 after unlearning is a finding too:
   over-displacement, where the records were pushed further out than
   retraining would leave them. In the paper this reached 0.16 on ResNet-18 —
   an artefact manufactured where nothing was memorised.
4. Run `m4` on the **original** model first. If it is already at 0.50 there
   was nothing to erase, and any post-unlearning deviation you then create is
   an artefact, not progress.

## 7. Errors and troubleshooting

### Errors the library raises

| Message | Cause | Fix |
|---|---|---|
| `... must be a 2-D (n, p) array, got shape ...` | A 1-D vector or 3-D tensor was passed. | One row per record: `x.reshape(1, -1)` for a single record; flatten sequence dims first. |
| `... is empty` | Zero rows in an input. | Check your forget/retain masks actually select records. |
| `shape mismatch: ... same records in the same order` | The two arrays given to `cosine_similarity` (or the forget pair / retain pair in `m1`–`m3`) differ in shape. | Embed the *same* records under each model; don't re-shuffle between models. |
| `embedding dimension mismatch: forget X vs retain Y` | The two `m4` arrays came from different layers or models. | Extract both from the same layer of the same model. |
| `m4 needs at least 2 retain records ...` | Retain gallery has < 2 rows. | Your split is degenerate; enlarge the retain side. |
| `... contains NaN or inf in N row(s)` | A diverged model, or embeddings taken from the wrong tensor. | Inspect the model that produced them; a NaN loss during unlearning usually means the learning rate is too high. |
| `... squared norm overflows float64` | Row magnitudes around 1e154 or larger. | Rescale the embeddings (any constant factor works — scale does not affect the metrics). |
| `... squared norm underflows float64 to zero` | Row magnitudes around 1e-162 or smaller. | Rescale the embeddings upwards; without the check these rows would score as garbage. |

### Silent failure modes (no error, wrong numbers)

| Symptom | Likely cause | Check |
|---|---|---|
| Results differ between identical runs | model left in training mode (dropout / batch-norm updating) | embed the same batch twice; the arrays must be identical |
| Every `m2` strongly negative, even for a freshly retrained model | oracle not seed-paired with the original | run the Appendix A.5 calibration in [§5](#5-building-a-retrain-oracle) |
| `m4` high even before unlearning, on data you don't believe is memorised | an erasure request split across forget/retain | confirm every patient/document/identity is wholly on one side |
| `m4` ≈ 0.50 read as "erased" on 10–30 records | value is inside the null interval | compare against the table in [§6](#6-interpreting-results) |
| `m2` sign flips when you recompute | retain rows reordered between the two models | same records, same order, both arrays |

## 8. Performance and memory

`m4` is the only non-trivial computation: a forget-to-retain and a chunked
retain-to-retain nearest-neighbour search — `O(|F|·|R|·p + |R|²·p)` time,
with memory bounded by a 512-row chunk so an `|R|×|R|` matrix is never
materialised.

For very large retain sets, subsample a gallery first; the paper caps it at
2,000 (Section 4.3):

```python
big_retain = rng.randn(50_000, 64)
gallery_idx = np.random.RandomState(42).choice(len(big_retain), 2_000, replace=False)
score = m4(u_forget, big_retain[gallery_idx])
print(round(score, 4))
```

Use a fixed seed for the subsample so the number is reproducible. `m1`–`m3`
are linear in the number of records and effectively free.

## 9. Limitations

Stated in the paper (Appendix A.2, §3.4) and worth restating here:

- **Not a compliance test.** The metrics characterise representational
  residuals, not regulatory adequacy under GDPR Article 17.
- **A null result is not proof of erasure.** The metrics assume memorisation
  shows up as geometric deviation under cosine similarity; information encoded
  another way is invisible to them.
- **`m2` requires the paired-seed design.** Without it the metric is not
  wrong — it is meaningless.
- **`m4` is dataset-specific** (its variance is dominated by dataset
  identity), so compare values within a dataset, not across datasets.

## 10. FAQ

**Do I need PyTorch?** No. The library is numpy-only; any framework (or none)
can produce the embeddings.

**Do I need a retrain oracle?** Only for `m1`/`m2`/`m3`. `m4` needs just the
unlearned model and the data split — it is the metric most users can actually
run.

**Can I use held-out test data as the retain set?** No. `m4` asks whether a
forget record blends in among records the model *was trained on*.

**My forget set is one person's records — can I get a per-person verdict?**
Usually not: tens of records put the null interval at roughly ±0.1–0.2. Aggregate
requests, or treat the answer as being about the model.

**Do embedding dimensions have to match across experiments?** Only within a
single comparison. The paper spans 128-d to 768-d across settings.

**Does normalisation matter?** No — embeddings are L2-normalised internally.

**Where is the paper's own experiment code?** In `paper/` in the repository,
separate from the library; see the README's "Reproducing the paper" section.
