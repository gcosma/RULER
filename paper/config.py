"""Experimental constants for the RULER primary tabular evaluation.

Every value here is fixed by the paper (Section 4) and is required for the
cached checkpoints in ``checkpoints/`` to be reproducible.  Changing a value
invalidates the cache: delete the affected ``.pt`` files before re-running.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Random states (Section 4.1, Appendix A.16)
# ---------------------------------------------------------------------------

#: Train/test split and forget-set sampling share this random state.
SPLIT_SEED = 999
FORGET_SEED = 999

#: Retain-set subsample used for the M2 calibration baseline and the M4
#: nearest-neighbour gallery (Section 4.3).
SUBSAMPLE_SEED = 42

#: Unlearning is run once per training seed with this fixed seed, so that
#: variability across runs reflects training initialisation only.
UNLEARN_SEED = 100

#: Training seeds. The original model and the retrain oracle for seed i are
#: trained with the same seed -- the paired-seed design of Section 3.3.
TRAIN_SEEDS = tuple(range(10))

# ---------------------------------------------------------------------------
# Architecture and training (Section 4.1)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 128
DROPOUT = 0.2
OUTPUT_DIM = 2

TRAIN_EPOCHS = 50
TRAIN_LR = 1e-3

#: The primary experiments use full-batch gradient descent to remove
#: optimisation noise. Appendix A.6 repeats ff=5% with this set to 128.
BATCH_SIZE = None

TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Unlearning (Section 4.2)
# ---------------------------------------------------------------------------

UNLEARN_LR = 5e-4
GA_EPOCHS = 5
UNLEARN_EPOCHS = 10

#: Retain/forget weighting for NegGrad+, SCRUB and Bad Teacher.
ALPHA = 0.6

#: Distillation temperature for SCRUB and Bad Teacher.
TEMPERATURE = 2.0

#: Bad Teacher uses its own teacher initialisation seed (Appendix A.11).
BAD_TEACHER_SEED = 101

# ---------------------------------------------------------------------------
# Forget fractions (Section 4.1)
# ---------------------------------------------------------------------------

FORGET_FRACTIONS = (0.01, 0.05, 0.10)

#: Smallest permitted forget set; prevents degenerate sets on small datasets.
MIN_FORGET_SIZE = 10

# ---------------------------------------------------------------------------
# Metric evaluation (Section 4.3)
# ---------------------------------------------------------------------------

#: Retain records drawn per seed to estimate the M2 median baseline.
M2_RETAIN_SUBSAMPLE = 500

#: Cap on the retain gallery for M4, for memory. Applies to both the forget
#: records' neighbour search and the retain leave-one-out distribution.
M4_RETAIN_CAP = 2000

#: A method passes output-level evaluation when its mean MIA accuracy over the
#: training seeds lies inside this window around chance.
MIA_PASS_WINDOW = 0.05

# ---------------------------------------------------------------------------
# Null values
# ---------------------------------------------------------------------------

M2_NULL = 0.0
M4_NULL = 0.50
MIA_NULL = 0.50

# ---------------------------------------------------------------------------
# Method names, in the order used by the paper's tables and figures
# ---------------------------------------------------------------------------

#: The four approximate methods of the primary evaluation. Bad Teacher is
#: evaluated separately (Section 5.4); for the full set including it, use
#: ``ruler.unlearn.UNLEARN_METHODS``.
METHODS = ("Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB")
