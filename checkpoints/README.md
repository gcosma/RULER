# Cached model checkpoints

400 pre-trained model weights: 10 tabular datasets × 10 training seeds ×
{original, oracle at forget fraction 1%, 5%, 10%}. The tutorial's
[`reproduce_breast_cancer.ipynb`](../tutorial/reproduce_breast_cancer.ipynb)
loads them and computes the metrics directly, so it skips the training step.
The library works from your own embeddings; these checkpoints serve the
tutorial alone.

## Naming

```
<dataset>_seed<i>_orig.pt          original model, trained on the full training set
<dataset>_seed<i>_oracle_ff01.pt   retrain oracle, retain set only, forget fraction 1%
<dataset>_seed<i>_oracle_ff05.pt   ... 5%
<dataset>_seed<i>_oracle_ff10.pt   ... 10%
```

Datasets: `adult`, `bank_marketing`, `breast_cancer`, `diabetes130`,
`electricity`, `german_credit`, `heart_disease`, `magic`, `phoneme`,
`wine_quality`. Seeds 0–9.

## Loading a checkpoint

Each file is a `state_dict` for a two-hidden-layer MLP (`d → 128 → 128 → 2`,
dropout after each ReLU), with keys `net.0`, `net.3`, `net.6` — the three
`nn.Linear` layers. Load one with a matching module:

```python
import torch
from torch import nn

class TabularMLP(nn.Module):
    def __init__(self, d, h=128, o=2, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, o),
        )

model = TabularMLP(30)                    # breast_cancer has 30 features
state = torch.load("checkpoints/breast_cancer_seed0_orig.pt", weights_only=True)
model.net.load_state_dict({k[len("net."):]: v for k, v in state.items()})
model.eval()
```

The penultimate-layer embedding RULER reads is `model.net[:5](x)` — the
activation after the second ReLU, before the output head.

## Paired-seed design

For each seed, the original model and its oracle were trained from the *same*
random initialisation. Cosine similarity depends on the orientation of the two
representations, so the oracle-comparative metrics (`m1`, `m2`, `m3`) rely on
this pairing: with it, the difference between two models reflects unlearning;
with mismatched seeds, that difference reflects initialisation geometry
instead. Same-seed original–oracle similarity averages ≈ 0.99, against ≈ 0.43
for oracles trained with different seeds (paper Appendix A.16).

## The experiment code

This set stays small: unlearned models are recomputed from the original model
and a fixed unlearning seed each time. The training scripts and the full
experiment pipeline that produced these checkpoints live on the
[`paper-experiments`](https://github.com/gcosma/RULER/tree/paper-experiments)
branch.
