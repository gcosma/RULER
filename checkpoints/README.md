# Cached model checkpoints

400 pre-trained `TabularMLP` state dicts: 10 datasets x 10 training seeds x
{original, oracle at ff = 1%, 5%, 10%}. Their presence lets the experiments skip
training entirely.

## Naming

```
<dataset>_seed<i>_orig.pt            original model, trained on the full training set
<dataset>_seed<i>_oracle_ff01.pt     retrain oracle, trained on the retain set at ff = 1%
<dataset>_seed<i>_oracle_ff05.pt     ... at ff = 5%
<dataset>_seed<i>_oracle_ff10.pt     ... at ff = 10%
```

Datasets: `adult`, `bank_marketing`, `breast_cancer`, `diabetes130`,
`electricity`, `german_credit`, `heart_disease`, `magic`, `phoneme`,
`wine_quality`. Seeds 0-9.

## Contents

Each file is a `state_dict` with keys `net.0`, `net.3`, `net.6` — the three
`nn.Linear` layers of the architecture in Fig. 2 (`d -> 128 -> 128 -> 2`, with
dropout after each ReLU). Load them with:

```python
from ruler.train import load_model, checkpoint_path

model = load_model(checkpoint_path("checkpoints", "adult", seed=0, kind="orig"), input_dim=14)
oracle = load_model(
    checkpoint_path("checkpoints", "adult", seed=0, kind="oracle", forget_fraction=0.05),
    input_dim=14,
)
```

`load_model` checks the checkpoint's input dimension against the dataset's and
raises on a mismatch, which catches the case where a different OpenML version
has been fetched.

## Paired-seed design

For a given seed the original model and the oracle were trained from the *same*
random initialisation. Cosine similarity is not rotation-invariant, so the Lens 1
metrics (`M1`, `M2`, `M3`) are only meaningful under this pairing: without it,
the difference between two models would be dominated by initialisation geometry
rather than by unlearning.

The effect is large and directly measurable in these files — same-seed
original–oracle similarity averages 0.99, against 0.44 for oracles trained with
different seeds (Appendix A.16). `tests/test_models.py` asserts this gap.

## What is not cached

Unlearned models. They are fully determined by the original model plus the fixed
unlearning seed (100) and are cheap to recompute, so the pipeline regenerates
them on every run.

## Regenerating

Delete the files and re-run `experiments/run_primary.py`; anything missing is
retrained and re-cached. Note that the sensitivity analyses which alter training
(`--batch-size`, `--forget-seed`) must use a separate `--checkpoint-dir`, since
their models are not the ones cached here.
