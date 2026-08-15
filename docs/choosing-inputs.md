# Choosing inputs: records, layers and models

Applying RULER to a new setting involves three choices. Only the second is
covered by the paper's tables; the first is where most mistakes happen.

---

## 1. What counts as one record?

The unit someone asks you to erase and the unit your model embeds are often not
the same thing. RULER measures embeddings, so you have to bridge them.

| Setting | Erasure request | Model embeds | Bridge |
|---|---|---|---|
| Tabular | a row | a row | identical — no decision |
| Image, per-image deletion | an image | an image | identical |
| Image, per-person deletion | an identity | an image | expand: forget set is every image of that person |
| Text | a document, note or patient | a sentence | expand: partition by document, then split into sentences |

**The rule: partition at the level of the erasure request, then expand to whatever the model
embeds.**

### Why this matters

If half a patient's sentences land in the forget set and half in the retain set,
the "forgotten" sentences still have near neighbours from the same patient
sitting in the gallery. M4 then reads high for a reason that has nothing to do
with what the model memorised — you are measuring your own split.

Sample at the level of the erasure request, then expand:

```python
# requires: your own patient ids and sentence arrays
# Erase whole patients, never individual sentences.
forget_patients = rng.choice(patient_ids, n, replace=False)
is_forget = np.isin(patients, forget_patients)
forget, retain = sentences[is_forget], sentences[~is_forget]
```

### Expand or pool?

The alternative to expanding is **pooling**: average a document's sentence
embeddings into one vector, so one record equals one erasure request.

- *Expand* (what the paper does) gives more records and a tighter estimate.
- *Pool* gives a number that maps directly onto "this individual", but with
  `|D_f|` in the tens — where the 95% interval under M4's null spans roughly ±0.18.

Expand unless you specifically need per-individual answers, and check the
sample size either way (see "Reading the result" below).

---

## 2. Which layer?

One rule covers every architecture: **the activation immediately before the
task-specific output head.**

| Architecture | Layer | Dim |
|---|---|---|
| Tabular MLP | second ReLU | 128 |
| Residual MLP / FT-Transformer | final hidden layer | 128 |
| Three-layer CNN | second fully-connected layer | 256 |
| ResNet-18 | post-global-average-pooling | 512 |
| BERT-family | `[CLS]` of the final transformer layer | 768 |

Not earlier layers: they share low-level features across records, so they do not
separate individuals. Not the output layer: it collapses into logits or a token
distribution, discarding the geometry the metrics need.

Embedding dimension does not need to match across settings — the paper spans 128
to 768, with output dimensions from 2 to about 30,000, and the metrics are
label-free and dimension-agnostic. Consistency is required only *within* one
comparison.

```python
# requires: your trained model (PyTorch shown)
model.eval()        # required: embeddings must be deterministic

# ResNet-18 — everything up to the classifier
backbone = torch.nn.Sequential(*list(model.children())[:-1])
embeddings = backbone(images).flatten(1).cpu().numpy()

# BERT — the [CLS] vector of the last layer
embeddings = model.bert(**batch).last_hidden_state[:, 0, :].cpu().numpy()
```

---

## 3. Which models?

Determined by which metrics you want.

| Metric | Needs | Cost |
|---|---|---|
| `m4` | the unlearned model alone | one forward pass |
| `m1`, `m2` | + a retrain oracle | full retraining |
| `m3` | + the original model | — |

```python
# requires: your models and an embed() helper
m4(embed(unlearned, forget), embed(unlearned, retain))          # M4 only
m2(embed(unlearned, forget), embed(oracle, forget),
   embed(unlearned, retain), embed(oracle, retain))            # M2
m3(embed(unlearned, forget), embed(original, forget),
   embed(oracle, forget))                                      # M3
```

**The oracle must share the original model's initialisation seed.** Cosine
similarity is not rotation-invariant, so an independently seeded oracle measures
initialisation geometry rather than unlearning: same-seed pairs sit at 0.99
similarity, differently-seeded pairs at 0.44.

Check this in your own setup before trusting M2: apply `m2` to *pairs of
independently retrained oracles*, none of which saw the forget set. Neither is
unlearned, so the result should centre on zero (Appendix A.5).

```python
# requires: several independently retrained oracles
from itertools import combinations

gaps = [m2(a_forget, b_forget, a_retain, b_retain)
        for (a_forget, a_retain), (b_forget, b_retain) in combinations(oracles, 2)]
np.mean(gaps)       # should be close to zero
```

A centre far from zero means M2 is biased in your setup — usually unpaired
initialisation — and every result would look like residual memorisation.

---

## Four invariants

1. **Retain set = training retain data**, not held-out test data. M4 asks
   whether a forget record blends in among records the model *was trained on*.
2. **Same records, in the same order, across models.** M2 pairs them row-wise; a
   reordered retain set produces silently wrong numbers, not an error.
3. **Evaluation mode.** Dropout or a live batch-norm update makes a record's
   embedding non-deterministic, and every metric compares embeddings across
   models. Call `model.eval()` before embedding.
4. **Never split an erasure request** across forget and retain.

---

## Reading the result

M4 is a mean over forget records, and under the null each contributes a uniform
value, so its standard deviation is `1/sqrt(12n)`. Its precision is set by the
number of forget records:

| \|D_f\| | 95% null interval | resolves |
|---|---|---|
| 10 | ±0.18 | almost nothing |
| 33 | ±0.10 | large effects |
| 129 | ±0.05 | moderate effects |
| 801 | ±0.02 | small effects |

An individual erasure request usually covers tens of records, so a single
request is often below the resolution of the metric. Where the question is "is
this model leaking?" rather than "did this one deletion work?", aggregate
several requests before auditing.
